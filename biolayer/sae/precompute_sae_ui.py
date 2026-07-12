"""Precompute the SAE evidence panel into static JSON + PNG for the dashboard.

The dashboard must render with no GPU, no HF gating, and no 75MB checkpoints in git --
same pattern as dashboard/precompute_heatmaps.py. So we run the live intervention ONCE,
here, and commit the result.

For each concept we emit three ablation curves, all measured the same way (project the
directions out of every token at every block >= LAYER, let the remaining blocks run, read
the model's own 9-class decision):

    sae     -- the SAE features the model computes the concept WITH    -> collapses the call
    random  -- the same NUMBER of random features (the house-rule null) -> barely moves
    probe   -- the L2-normalised linear probe direction for the concept -> moves it not at all

The probe line is the point. A probe that reads the concept out at ~99% accuracy is not what
the model computes with, so deleting it changes nothing. That is the difference between
explainability and mechanistic interpretability, and it is one chart.

    HF_TOKEN=... python scripts/precompute_sae_ui.py --out <dashboard>/public/sae
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import mcp_mechinterp as M  # noqa: E402

ART = M.ART
CONCEPTS = ["TUM", "LYM", "STR", "NORM", "MUC", "MUS", "ADI", "DEB", "BACK"]  # all 9, not a pick
KS = [5, 20, 80, 160]


def probe_curve(s, px, ci):
    """Ablate the PROBE direction, the same way, at the same depth. The control that matters.

    A probe direction is ONE direction -- it does not grow with k, so the line is flat by
    construction. That is not a limitation of the plot, it IS the finding: once you have
    deleted the direction the concept is read out along, there is nothing further to delete,
    and the model still knows.
    """
    z = np.load(f"{ART}/probes_L{M.LAYER}.npz", allow_pickle=True)
    d = torch.from_numpy(z["directions"][ci].astype(np.float32)).reshape(-1, 1).to(s["dev"])
    p = float(M._ablate_and_run(s["model"], s["head"], px, d)[:, ci].mean())
    return [round(p, 4)] * len(KS), float(z["accuracy"][ci])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="dashboard/public/sae")
    ap.add_argument("--concepts", nargs="+", default=CONCEPTS)
    ap.add_argument("--n-tiles", type=int, default=32)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    explain = getattr(M.explain, "fn", M.explain)  # unwrap @mcp.tool() if it wrapped
    s = M._state()
    cn = s["class_names"]

    out = {"layer": M.LAYER, "n_blocks": M.N_BLOCKS, "ks": KS, "concepts": {}}
    for c in a.concepts:
        print(f"[{c}] running live intervention …", flush=True)
        r = explain(c, n_tiles=a.n_tiles, ks=list(KS))
        if "error" in r:
            print(f"[{c}] SKIP: {r['error']}", flush=True)
            continue

        ci = cn.index(c)
        tiles = np.where(s["labels"] == ci)[0][: a.n_tiles]
        px = torch.stack([M._tf(im) for im in M.load_tiles(tiles)]).to(s["dev"])
        pcurve, pacc = probe_curve(s, px, ci)

        card_src = r.get("evidence_card")
        card_dst = os.path.join(a.out, f"CARD_{c}.png")
        if card_src and os.path.exists(card_src):
            shutil.copyfile(card_src, card_dst)

        out["concepts"][c] = {
            "concept": c,
            "headline": r["headline"],
            "should_i_trust_it": r["should_i_trust_it"],
            "morphology": r["the_morphology_it_relied_on"],
            "fallback": r["what_it_sees_instead_once_deleted"],
            "not_noise": r["how_we_know_this_is_not_noise"],
            "baseline": r["baseline_confidence"],
            "features_to_overturn": r["features_to_overturn"],
            "encoding": r["encoding"],
            "top_features": r["top_features"],
            "intervention": r["intervention"],
            "evidence_card": f"sae/CARD_{c}.png" if os.path.exists(card_dst) else None,
            "curves": {
                "sae": [p["confidence"] for p in r["ablation_curve"]],
                "random": [p["random_control"] for p in r["ablation_curve"]],
                "probe": pcurve,
            },
            "probe_accuracy": round(pacc, 4),
        }
        cc = out["concepts"][c]["curves"]
        print(f"[{c}] p0={r['baseline_confidence']:.3f}  sae->{cc['sae'][-1]:.3f}  "
              f"random->{cc['random'][-1]:.3f}  probe->{cc['probe'][-1]:.3f}  "
              f"(probe acc {pacc:.4f})", flush=True)

    with open(os.path.join(a.out, "sae.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {a.out}/sae.json  concepts={list(out['concepts'])}")


if __name__ == "__main__":
    main()
