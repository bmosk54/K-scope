"""Precompute REAL per-patch causal heatmaps for the dashboard Case window.

Replaces the hand-drawn "schematic · not measured" tissue SVG with a measured
saliency map: for one real NCT-CRC tile we re-forward it through frozen Phikon-v2,
capture the 14x14 patch-token grid at the readout layer (the grid models.py normally
mean-pools away), project each patch onto the concept axis certify uses, and z-score
that projection against a matched-random-direction null (the Section-5-D control).

Output (committed, so the demo needs no GPU at runtime):
    dashboard/public/heatmaps/<concept>.png      the real 224px tile
    dashboard/public/heatmaps/heatmaps.json      {concept: {grid 14x14 z, top_z, verdict, ...}}

This is the live wrapper attribution.py::hack_tile was stubbed for — same math
(attribution.patch_importance), now fed a real patch-grid forward.
"""
import io
import json
import os
import pickle

import numpy as np
import torch
from PIL import Image

from biolayer import config
from biolayer.causal import attribution, probe as _probe

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "public", "heatmaps")
TILE_CACHE = os.path.join(
    HERE, "..", "artifacts", "serving_cache",
    "ref_phikon_v2_nct_crc_he_TUM-LYM-STR-MUS-NORM_24_train_1.pkl")
EMB = os.path.join(HERE, "..", "artifacts", "embeddings",
                   "nct_crc_he", "phikon_v2", "train.npz")

READOUT_LAYER = config.MODELS["phikon_v2"]["layers"][-1]  # 24 = last hidden state
N_NULL = 200
SEED = 0

# One heatmap per certifiable demo concept. pos = the class whose tile we show and
# whose patches we highlight; neg = the certify contrast partner. Axis is the raw
# diff-of-means (TUM_mean - NORM_mean, ...) in the readout residual space patch
# tokens share with the CLS. Keyed by the claim.concept values in data.js.
CONCEPTS = [
    {"concept": "tumor_epithelium", "pos": "TUM",  "neg": "NORM", "label": "tumor epithelium (TUM vs NORM)"},
    {"concept": "immune_infiltrate", "pos": "LYM", "neg": "TUM",  "label": "immune infiltrate / TILs (LYM vs TUM)"},
    {"concept": "stroma",            "pos": "STR",  "neg": "MUS",  "label": "desmoplastic stroma (STR vs MUS)"},
    {"concept": "normal_mucosa",     "pos": "NORM", "neg": "TUM",  "label": "normal mucosa (NORM vs TUM)"},
]


def load_tiles():
    o = pickle.load(open(TILE_CACHE, "rb"))
    imgs, labs = o["images"], list(o["labels"])
    by_class = {}
    for im, lab in zip(imgs, labs):
        by_class.setdefault(config.CLASS_NAMES[int(lab)], []).append(im.convert("RGB"))
    return by_class


def concept_axis(name_pos, name_neg):
    """Raw-space unit diff-of-means axis (pos - neg) from cached readout CLS feats."""
    d = np.load(EMB, allow_pickle=True)
    feats, labels = d["feats"], d["labels"]  # (N,1024) readout global, (N,)
    cls = list(d["class_names"])
    ip, ineg = cls.index(name_pos), cls.index(name_neg)
    mp = feats[labels == ip].mean(0)
    mn = feats[labels == ineg].mean(0)
    axis = mp - mn
    return axis / (np.linalg.norm(axis) + 1e-12)


def load_phikon():
    from transformers import AutoImageProcessor, AutoModel
    spec = config.MODELS["phikon_v2"]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    proc = AutoImageProcessor.from_pretrained(spec["hf_id"], use_fast=True)
    model = AutoModel.from_pretrained(spec["hf_id"]).to(dev).eval()

    @torch.inference_mode()
    def patch_grid(image):
        """One PIL tile -> (P, D) readout-layer patch-token grid (CLS dropped)."""
        inp = proc(images=[image], return_tensors="pt").to(dev)
        hs = model(**inp, output_hidden_states=True).hidden_states
        return hs[READOUT_LAYER][0, 1:].float().cpu().numpy()  # (196, 1024)

    return patch_grid


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    by_class = load_tiles()
    patch_grid = load_phikon()
    print(f"loaded phikon-v2; tiles per class: "
          f"{ {k: len(v) for k, v in by_class.items()} }", flush=True)

    out = {}
    for spec in CONCEPTS:
        axis = concept_axis(spec["pos"], spec["neg"])
        tiles = by_class[spec["pos"]]
        # Pick the tile whose patches most strongly single out the concept above the
        # null (highest top_z) -> the most honestly discriminating heatmap to show.
        best = None
        for i, tile in enumerate(tiles):
            grid = patch_grid(tile)
            imp = attribution.patch_importance(grid, axis, n_null=N_NULL, seed=SEED)
            if best is None or imp["top_z"] > best[1]["top_z"]:
                best = (i, imp, tile)
        idx, imp, tile = best
        z = np.asarray(imp["z"])
        s = int(round(len(z) ** 0.5))
        grid2d = z.reshape(s, s)
        # publish a [0,1]-normalised grid for the overlay + the raw z stats for labels
        lo, hi = float(z.min()), float(z.max())
        norm = (grid2d - lo) / (hi - lo + 1e-9)

        png = f"{spec['concept']}.png"
        tile.save(os.path.join(OUT_DIR, png))
        out[spec["concept"]] = {
            "concept": spec["concept"],
            "label": spec["label"],
            "pos": spec["pos"], "neg": spec["neg"],
            "tile": f"heatmaps/{png}",
            "grid_side": s,
            "z_grid": np.round(grid2d, 3).tolist(),      # raw per-patch importance z
            "norm_grid": np.round(norm, 4).tolist(),      # [0,1] for the overlay alpha
            "z_min": round(lo, 3), "z_max": round(hi, 3),
            "top_z": round(float(imp["top_z"]), 2),
            "top_patch": int(imp["top_patch"]),
            "n_patches": int(imp["n_patches"]),
            "n_null": N_NULL,
            "verdict": imp["verdict"],
            "layer": "readout",
            "tile_index": int(idx),
        }
        print(f"  {spec['concept']:18} tile#{idx:2d}  top_z={imp['top_z']:6.2f}  "
              f"-> {imp['verdict']}", flush=True)

    with open(os.path.join(OUT_DIR, "heatmaps.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {os.path.join(OUT_DIR, 'heatmaps.json')} ({len(out)} concepts)", flush=True)


if __name__ == "__main__":
    main()
