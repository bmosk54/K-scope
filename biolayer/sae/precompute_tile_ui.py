"""Per-tile evidence: what did the model use to read THIS tile, and does its read survive?

The population view ("how does the model compute LYM in general") is a statement about the
model. This is a statement about the USER'S TILE -- the one sitting in the Prompt view -- which
is what the rest of the dashboard is about. Same tile, same substrate (slide_demo.json declares
h_optimus_0), all the way through.

For the one input tile we:
  1. ENCODE it exactly as the corpus was encoded (block 27 CLS, post-LN, registers excluded).
  2. READ  the model's answer: which of the 9 tissue types, and how sure.
  3. FIND  the ~40 visual features the SAE says fired in THIS tile.
  4. SCORE each one CAUSALLY: delete it alone from the live model, measure how far the model's
     read of THIS tile moves. 40 features -> 40 forward passes -> seconds.
  5. GROUND the top features in real tissue: the corpus tiles that fire each one hardest, so the
     user can SEE what the model recognises, plus where in their own tile it fires.
  6. TEST  cumulatively, against a random control and against the linear probe.

    HF_TOKEN=... python scripts/precompute_tile_ui.py --tile <path> --out <dashboard>/public/sae
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import mcp_mechinterp as M  # noqa: E402
from exemplars import load_tiles  # noqa: E402
from heatmap import patch_codes  # noqa: E402

ART = M.ART
KS = [1, 2, 5, 10, 20, 40]
GRID = 16
TISSUE = {"ADI": "adipose", "BACK": "background", "DEB": "debris", "LYM": "lymphocytes",
          "MUC": "mucus", "MUS": "smooth muscle", "NORM": "normal mucosa",
          "STR": "cancer-associated stroma", "TUM": "tumour epithelium"}


def encode(model, im, layer=M.LAYER):
    """The tile -> block-`layer` CLS. Same path as extract_hoptimus: norm=True, registers split."""
    x = M._tf(im).unsqueeze(0).to("cuda")
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
        patch, prefix = model.get_intermediate_layers(
            x, n=[layer], return_prefix_tokens=True, norm=True)[0]
    return prefix[:, 0].float(), patch.float()          # (1,1536), (1,256,1536)


def contact_sheet(idx, out, title):
    ims = load_tiles([int(i) for i in idx])
    fig, axes = plt.subplots(1, len(ims), figsize=(1.55 * len(ims), 1.85))
    for a, im in zip(np.atleast_1d(axes), ims):
        a.imshow(im); a.set_xticks([]); a.set_yticks([])
        for sp in a.spines.values():
            sp.set_edgecolor("#c0392b"); sp.set_linewidth(1.6)
    fig.suptitle(title, fontsize=8.5, y=1.06)
    fig.savefig(out, dpi=112, bbox_inches="tight", transparent=True)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tile", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--top", type=int, default=4, help="features to ground with exemplar tiles")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    s = M._state()
    model, head, sae, W = s["model"], s["head"], s["sae"], s["W"]
    codes, y, cn = s["codes"], s["labels"], s["class_names"]
    ck, dev = s["ck"], s["dev"]

    im = Image.open(a.tile).convert("RGB")
    px = M._tf(im).unsqueeze(0).to(dev)

    # ---- 2. what does the model say about THIS tile? -----------------------------------
    base = M._ablate_and_run(model, head, px, None)[0]
    ci = int(np.argmax(base))
    p0 = float(base[ci])
    print(f"[tile] model reads it as {cn[ci]} ({TISSUE[cn[ci]]}) at {p0:.3f}", flush=True)

    # ---- 3. which features fired IN THIS TILE? -----------------------------------------
    cls27, patch27 = encode(model, im)
    z = sae(((cls27 - ck["mu"].to(dev)) / ck["scale"].to(dev)))[1][0]
    active = torch.nonzero(z > 0).flatten().cpu().numpy()
    print(f"[tile] {len(active)} of {ck['n_features']} features are active (TopK k={ck['k']})", flush=True)

    # ---- 4. score each ACTIVE feature causally: delete it alone, see how far the read moves
    eff = []
    for f in active:
        p = M._ablate_and_run(model, head, px, W[:, [int(f)]])[0, ci]
        eff.append(p0 - float(p))
    eff = np.array(eff)
    order = np.argsort(-eff)
    ranked = active[order]                    # this tile's features, by causal effect
    print(f"[tile] strongest single feature drops the read by {eff[order[0]]:.3f}", flush=True)

    # ---- 6. cumulative deletion, vs a matched-random control ---------------------------
    rng = np.random.default_rng(0)
    ks = [k for k in KS if k <= len(ranked)]
    curve, null = [], []
    for k in ks:
        curve.append(float(M._ablate_and_run(model, head, px, W[:, [int(f) for f in ranked[:k]]])[0, ci]))
        null.append(float(np.mean([
            M._ablate_and_run(model, head, px,
                              W[:, rng.choice(codes.shape[1], k, replace=False)])[0, ci]
            for _ in range(3)])))
    # the probe direction for the SAME class, ablated the SAME way -> the control that matters
    pz = np.load(f"{ART}/probes_L{M.LAYER}.npz", allow_pickle=True)
    pdir = torch.from_numpy(pz["directions"][ci].astype(np.float32)).reshape(-1, 1).to(dev)
    p_probe = float(M._ablate_and_run(model, head, px, pdir)[0, ci])
    print(f"[tile] sae->{curve[-1]:.3f}  random->{null[-1]:.3f}  probe->{p_probe:.3f}", flush=True)

    # what does it read the tile as, once its evidence is gone?
    final = M._ablate_and_run(model, head, px, W[:, [int(f) for f in ranked[: ks[-1]]]])[0]
    shift = final - base
    fallback = {cn[i]: round(float(shift[i]), 3) for i in np.argsort(-shift)[:3] if shift[i] > 0.02}

    # ---- 5. GROUND the top features in real tissue -------------------------------------
    feats = []
    for r, f in enumerate(ranked[: a.top]):
        f = int(f)
        ex = np.argsort(-codes[:, f])[:6]                     # corpus tiles that fire it hardest
        cnt = np.bincount(y[np.argsort(-codes[:, f])[:100]], minlength=len(cn))
        purity = sorted(enumerate(cnt), key=lambda kv: -kv[1])
        top_cls, top_pct = cn[purity[0][0]], int(purity[0][1])
        png = f"sae/tile_feat{f}.png"
        contact_sheet(ex, os.path.join(a.out, os.path.basename(png)),
                      f"feature {f} — fires on {TISSUE[top_cls]} ({top_pct}% of its top 100 tiles)")
        feats.append({
            "feature": f,
            "effect": round(float(eff[order[r]]), 4),
            "activation": round(float(z[f]), 3),
            "looks_like": TISSUE[top_cls],
            "purity_pct": top_pct,
            "exemplars_png": png,
        })
        print(f"  feat {f}: effect -{eff[order[r]]:.3f}  looks like {TISSUE[top_cls]} ({top_pct}%)",
              flush=True)

    # ---- WHERE in THIS tile does the TOP FEATURE fire? ---------------------------------
    # It must be the SAME feature the card shows, or the picture and the caption are about two
    # different things. The earlier version used the PATCH SAE's own basis (a different set of
    # features entirely, chosen for the class rather than for this tile) -- coherent-looking and
    # quietly wrong.
    #
    # The tile SAE lives in the block-27 residual space, and so do the 256 patch tokens: the same
    # 1536 dims. So we project each patch token onto feature f's ENCODER direction. Read this as
    # "how strongly does this patch point along feature f", not as an SAE inference -- the SAE was
    # fit on CLS vectors, so we use its direction, not its (CLS-calibrated) thresholds.
    # TRIED AND REJECTED: projecting the TILE SAE's feature direction onto the patch tokens, so the
    # map would show the same feature as card 1. It does not work -- that SAE was fit on CLS
    # vectors, and its encoder direction is nonzero on ~1 of the 256 patch tokens. The map came out
    # as a single lit square. Off-distribution, exactly as feared. Do not re-try without refitting.
    #
    # So we use the PATCH SAE (fit on 1.6M patch tokens, correct distribution) and the patch
    # features most selective for the model's own answer. This is an HONEST but DIFFERENT question:
    # "where does tumour-typical structure sit in this tile", NOT "where does card 1's feature
    # fire". The caption in the UI says so.
    pc = patch_codes(model, s["patch_sae"], s["patch_ck"], [im], dev)[0].cpu().numpy()  # (256, F)
    pf = M._concept_patch_feats(s, ci, n=8)
    proj = pc[:, pf].sum(-1).reshape(GRID, GRID)
    pos = proj[proj > 0]
    scale = np.percentile(pos, 90) if pos.size else 1.0
    hm = np.clip(proj / (scale + 1e-9), 0, 1)

    fig, ax = plt.subplots(figsize=(3.4, 3.4))
    ax.imshow(im, alpha=0.6)
    ax.imshow(np.kron(hm, np.ones((14, 14))), cmap="inferno",
              alpha=0.5, interpolation="bilinear")
    ax.set_xticks([]); ax.set_yticks([])
    fig.savefig(os.path.join(a.out, "tile_where.png"), dpi=118, bbox_inches="tight", transparent=True)
    plt.close(fig)

    out = {
        "tile_png": "input_tile.png",
        "answer": cn[ci],
        "answer_tissue": TISSUE[cn[ci]],
        "confidence": round(p0, 4),
        "n_active": int(len(active)),
        "n_features": int(ck["n_features"]),
        "k": int(ck["k"]),
        "layer": M.LAYER,
        "n_blocks": M.N_BLOCKS,
        "ks": ks,
        "curves": {"sae": [round(v, 4) for v in curve],
                   "random": [round(v, 4) for v in null],
                   "probe": [round(p_probe, 4)] * len(ks)},
        "probe_accuracy": round(float(pz["accuracy"][ci]), 4),
        "features": feats,
        "where_png": "sae/tile_where.png",
        "fallback": fallback,
    }
    with open(os.path.join(a.out, "tile.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {a.out}/tile.json", flush=True)


if __name__ == "__main__":
    main()
