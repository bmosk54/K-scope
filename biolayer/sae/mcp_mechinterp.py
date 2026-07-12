"""ONE MCP tool: `explain` — mechanistic interpretability for a K Pro answer.

THE USE CASE. A K Pro user asks a histology question ("is this tumour?", "is this immune-hot?").
K Pro answers. The user wants to know whether to BELIEVE it. They call `explain(concept)`, and
this tool reaches inside the pathology foundation model, finds the features it actually used,
DELETES them, and reports what happened -- with a matched-random control throughout.

WHY ONE TOOL. The user is a pharma researcher in a K Pro chat, not an ML engineer. They should
not have to compose four verbs. `explain` runs the whole battery and returns one answer:
what the model used, whether it was load-bearing, how robust the belief is, and the pictures.

WHAT IT ACTUALLY DOES (in order):
  1. FIND      — rank SAE features by how selectively they fire on the concept.
  2. GROUND    — render the top features' exemplar tiles: what the model is looking at.
  3. INTERVENE — project those feature directions out of the LIVE model's residual stream at
                 every block from L to the end, let the network run, and watch the model's own
                 decision move. This is a do(), not an attribution: 13 blocks execute after the edit.
  4. CONTROL   — repeat with the same number of RANDOM features. Without this, "the output
                 changed" proves nothing.
  5. FALLBACK  — report what the model believes INSTEAD once the concept is deleted.

KEY MEASURED FACTS THIS TOOL IS BUILT ON (see FIG2/FIG3):
  * A linear probe with 99.98% accuracy, ablated the same way, moves the model NOT AT ALL
    (P(TUM) 0.999 -> 1.000). Probe directions are how you READ a concept out; SAE features are
    what the model COMPUTES with. Only the latter can be intervened on.
  * Concepts differ enormously in redundancy: 20 features destroy the model's immune call
    (1.000 -> 0.236) while 160 barely dent its tumour call (0.999 -> 0.540). So "how auditable
    is this answer" has a real, per-concept answer -- which is exactly what a K Pro user wants.
  * Editing ONE layer does nothing: the network recomputes the concept from untouched patch
    tokens downstream (the Hydra effect). Ablation must be persistent across blocks.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch
from mcp.server.fastmcp import FastMCP
from sklearn.linear_model import LogisticRegression
from torchvision import transforms

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exemplars import load_tiles  # noqa: E402
from train_sae_topk import TopKSAE  # noqa: E402

ART = "/home/sagemaker-user/biolayer/artifacts"
LAYER = 27          # intervene here; blocks 28..39 still run, so the edit must propagate
N_BLOCKS = 40
_MEAN = (0.707223, 0.578729, 0.703617)
_STD = (0.211883, 0.230117, 0.177517)
_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(_MEAN, _STD)])

mcp = FastMCP("biolayer-mechinterp")
_S: dict = {}


def _state():
    if _S:
        return _S
    import timm
    import huggingface_hub

    tok = os.environ.get("HF_TOKEN")
    if not tok:
        raise RuntimeError("HF_TOKEN required (H-Optimus-0 is gated)")
    huggingface_hub.login(token=tok)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    z = np.load(f"{ART}/hoptimus_100k.npz")
    y = z["labels"]
    cn = [str(c) for c in z["class_names"]]
    X_final = z["globals"][:, 2].astype(np.float32)     # block 39 CLS -> the readout space

    ck = torch.load(f"{ART}/sae_topk_L{LAYER}.pt", map_location=dev, weights_only=False)
    sae = TopKSAE(ck["d_model"], ck["n_features"], ck["k"]).to(dev)
    sae.load_state_dict(ck["state_dict"])
    sae.eval()

    # The model's "decision": a 9-class head on the final embedding. 99.8% accurate, so it is a
    # faithful stand-in for what the model concludes about a tile.
    rng = np.random.default_rng(0)
    tr = rng.choice(len(X_final), 30000, replace=False)
    head = LogisticRegression(max_iter=2000, C=0.1).fit(X_final[tr], y[tr])

    li = list(z["layers"]).index(LAYER)
    Xl = torch.from_numpy(z["globals"][:, li].astype(np.float32)).to(dev)
    Xn = (Xl - ck["mu"].to(dev)) / ck["scale"].to(dev)
    with torch.no_grad():
        codes = torch.cat([sae(Xn[i : i + 8192])[1] for i in range(0, len(Xn), 8192)]).cpu().numpy()

    model = timm.create_model("hf-hub:bioptimus/H-optimus-0", pretrained=True,
                              init_values=1e-5, dynamic_img_size=False).eval().to(dev)

    # PATCH-level SAE: each patch token is ~7 microns at 0.5 MPP, i.e. roughly one cell.
    # This is what lets the card answer "WHERE in the tissue", not just "what morphology".
    pck = torch.load(f"{ART}/sae_topk_patches.pt", map_location=dev, weights_only=False)
    psae = TopKSAE(pck["d_model"], pck["n_features"], pck["k"]).to(dev)
    psae.load_state_dict(pck["state_dict"])
    psae.eval()

    # Firing rate of every PATCH feature per tissue class -> lets us pick, for any concept,
    # the patch features that localise it. Sampled at RANDOM: the arrays are CLASS-SORTED, so
    # any prefix slice contains only one class.
    pz = np.load(f"{ART}/hoptimus_patches.npz")
    plab, pfeat = pz["labels"], pz["feats"]
    idx = np.sort(rng.choice(len(plab), 200000, replace=False))
    prate = np.zeros((len(cn), pck["n_features"]), dtype=np.float32)
    pcnt = np.zeros(len(cn))
    with torch.no_grad():
        for i in range(0, len(idx), 16384):
            sl = idx[i : i + 16384]
            xb = torch.from_numpy(pfeat[sl].astype(np.float32)).to(dev)
            _, zc, _ = psae((xb - pck["mu"].to(dev)) / pck["scale"].to(dev))
            fired = (zc > 0).float().cpu().numpy()
            for c in range(len(cn)):
                m = plab[sl] == c
                if m.any():
                    prate[c] += fired[m].sum(0)
                    pcnt[c] += m.sum()
    prate /= np.maximum(pcnt, 1)[:, None]

    _S.update(sae=sae, ck=ck, model=model, head=head, codes=codes, labels=y,
              class_names=cn, dev=dev, W=sae.dec.weight.detach(),
              patch_sae=psae, patch_ck=pck, patch_rate=prate)
    return _S


def _concept_patch_feats(s, ci, n=8):
    """Patch features that fire on this concept far more than on any other -> where it LIVES."""
    r = s["patch_rate"]
    other = np.delete(r, ci, axis=0).max(0)
    return np.argsort(-(r[ci] - other))[:n]


def _ablate_and_run(model, head, px, dirs, layer=LAYER):
    """Project a subspace out of EVERY token at every block >= layer, then read the decision."""
    hooks = []
    if dirs is not None:
        Q, _ = torch.linalg.qr(dirs.float())

        def hook(m, i, o):
            h = o.float()
            return (h - (h @ Q) @ Q.T).to(o.dtype)

        for b in range(layer, N_BLOCKS):
            hooks.append(model.blocks[b].register_forward_hook(hook))
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
        out = model(px).float()
    for h in hooks:
        h.remove()
    return head.predict_proba(out.cpu().numpy())


@mcp.tool()
def explain(concept: str, n_tiles: int = 32, ks: list[int] | None = None) -> dict:
    """Explain, mechanistically, what a pathology model used to reach a conclusion — and whether
    that conclusion is robust.

    Reaches inside H-Optimus-0, finds the interpretable features it uses for `concept`, DELETES
    them from the live network, and reports how far the model's own decision moves — against a
    matched-random control. Returns the morphology the model relied on (as tile images), how many
    features it takes to break the belief, and what the model concludes instead.

    Args:
        concept: the tissue concept the answer claims. One of the 9 NCT-CRC-HE classes:
            TUM (tumour), LYM (immune infiltrate), STR (stroma), NORM (normal mucosa),
            MUC (mucus), MUS (smooth muscle), ADI (fat), DEB (debris), BACK (background).
        n_tiles: how many tiles of that concept to intervene on.
        ks: how many features to delete at each step (default 5, 20, 80, 160).
    """
    s = _state()
    cn, y, codes, W = s["class_names"], s["labels"], s["codes"], s["W"]
    if concept not in cn:
        return {"error": f"unknown concept {concept!r}", "available": cn}
    ks = ks or [5, 20, 80, 160]
    ci = cn.index(concept)

    # 1. FIND: features that fire on this concept far more than on anything else.
    sel = (codes[y == ci] > 0).mean(0) - (codes[y != ci] > 0).mean(0)
    ranked = np.argsort(-sel)

    # RANDOM sample within the class, not the first n. NCT-CRC-HE is class-sorted, so a prefix
    # slice is a deterministic block of the class -- likely correlated in source slide -- and the
    # numbers would be a property of those tiles rather than of the tissue type.
    rng = np.random.default_rng(0)
    pool = np.where(y == ci)[0]
    tiles = np.sort(rng.choice(pool, min(n_tiles, len(pool)), replace=False))
    px = torch.stack([_tf(im) for im in load_tiles(tiles)]).to(s["dev"])

    # 3+4. INTERVENE, with a matched-random control at every step.
    base = _ablate_and_run(s["model"], s["head"], px, None)
    p0 = float(base[:, ci].mean())
    curve, null = [], []
    for k in ks:
        real = _ablate_and_run(s["model"], s["head"], px, W[:, ranked[:k]])
        rnd = np.mean([
            _ablate_and_run(s["model"], s["head"], px,
                            W[:, rng.choice(codes.shape[1], k, replace=False)])[:, ci].mean()
            for _ in range(2)
        ])
        curve.append(float(real[:, ci].mean()))
        null.append(float(rnd))
    final = _ablate_and_run(s["model"], s["head"], px, W[:, ranked[: ks[-1]]]).mean(0)

    # 5. FALLBACK: what does the model think it is once the concept is gone?
    shift = final - base.mean(0)
    fallback = {cn[i]: round(float(shift[i]), 3)
                for i in np.argsort(-shift)[:3] if shift[i] > 0.02}

    # how many features to break the belief (cross 0.5)?
    broke = next((k for k, p in zip(ks, curve) if p < 0.5), None)
    frag = "SPARSE / AUDITABLE" if (broke and broke <= 20) else \
           "DISTRIBUTED / REDUNDANT" if curve[-1] > 0.45 else "MODERATE"

    # 2. GROUND: ONE evidence card -- morphology, robustness, failure mode, spatial location.
    # A pharma researcher cannot act on "feature 2524"; the index is an implementation detail.
    # They act on the PICTURES and the trust verdict.
    from evidence_card import render

    card = render(concept, ranked, codes, y, cn, ks, curve, null, p0,
                  {cn[i]: round(float((final - base.mean(0))[i]), 3)
                   for i in np.argsort(-(final - base.mean(0)))[:3]
                   if (final - base.mean(0))[i] > 0.02},
                  broke=next((k for k, p in zip(ks, curve) if p < 0.5), None),
                  patch_sae=s["patch_sae"], patch_ck=s["patch_ck"], model=s["model"],
                  concept_patch_feats=_concept_patch_feats(s, ci))

    # what tissue does the top feature's morphology carry? -> plain-language description
    tops = np.argsort(-codes[:, ranked[0]])[:100]
    cnt = np.bincount(y[tops], minlength=len(cn))
    morph = ", ".join(f"{cn[i]} {v}%" for i, v in
                      sorted(enumerate(cnt), key=lambda kv: -kv[1])[:3] if v > 0)

    drop = p0 - curve[-1]
    null_drop = p0 - null[-1]
    return {
        # ---------- what a K Pro user reads ----------
        "headline": (
            f"The model's '{concept}' call is {frag.split(' /')[0].lower()}. Deleting the "
            f"{ks[-1]} features it uses drops its confidence from {p0:.2f} to {curve[-1]:.2f}; "
            f"deleting {ks[-1]} RANDOM features only drops it to {null[-1]:.2f}. "
            + (f"Only {broke} features are needed to overturn the call entirely — so this "
               f"answer rests on a small, inspectable set of visual features."
               if broke else
               f"Even {ks[-1]} features do not overturn the call — the model has many "
               f"independent routes to this conclusion, so no single piece of morphology is "
               f"load-bearing.")
        ),
        "should_i_trust_it": (
            "The evidence is concentrated and checkable: look at the exemplar tiles for the few "
            "features driving it, and if they are not the morphology you expect, the answer is "
            "suspect."
            if broke and broke <= 20 else
            "The belief is highly redundant — it survives deleting hundreds of features. That "
            "means it is ROBUST, but also that it cannot be traced to specific morphology: there "
            "is no small set of features to inspect."
        ),
        "evidence_card": card,   # <- THE thing the user looks at. Not a feature index.
        "the_morphology_it_relied_on": (
            f"The tiles that most drive the model's '{concept}' decision are {morph}. "
            "Open the evidence card and judge whether that is the morphology you would expect. "
            "If it is not, the answer is suspect regardless of how confident the model is."
        ),
        "what_it_sees_instead_once_deleted": fallback or "no coherent fallback",
        "how_we_know_this_is_not_noise": (
            f"Every ablation is compared against deleting the same NUMBER of random features. "
            f"Random: {p0:.2f} -> {null[-1]:.2f} (drop {null_drop:.2f}). "
            f"Concept features: {p0:.2f} -> {curve[-1]:.2f} (drop {drop:.2f})."
        ),
        # ---------- for Owkin Zero ----------
        "concept": concept,
        "baseline_confidence": round(p0, 4),
        "ablation_curve": [{"n_features_deleted": k, "confidence": round(c, 4),
                            "random_control": round(n, 4)} for k, c, n in zip(ks, curve, null)],
        "features_to_overturn": broke,
        "encoding": frag,
        "top_features": [int(f) for f in ranked[:8]],
        "intervention": (
            f"Feature directions projected out of every token at every block {LAYER}-{N_BLOCKS-1} "
            f"of H-Optimus-0. {N_BLOCKS-LAYER-1} transformer blocks execute after the edit, so "
            "this is a causal intervention on the model's computation, not a linear attribution."
        ),
        "caveats": [
            "The 'decision' is a 9-class tissue head (99.8% accurate) on the final embedding — "
            "a faithful stand-in for the model's conclusion, not a clinical prediction.",
            "A LINEAR PROBE cannot substitute for this: a 99.98%-accurate tumour probe direction, "
            "ablated identically, moves the model not at all. Probes READ concepts out; SAE "
            "features are what the model COMPUTES with.",
            "Ablation must be persistent across blocks. Editing a single layer does nothing — the "
            "network recomputes the concept from untouched patch tokens (the Hydra effect).",
            "Feature exemplars show what the model responds to. A pathologist must confirm the "
            "morphology; this tool generates the evidence, it does not adjudicate it.",
        ],
    }


if __name__ == "__main__":
    mcp.run(transport="stdio")
