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


# Human labels for the NCT-CRC tissue classes we resolve a patch's direction against
# (BACK/background excluded — not a tissue of interest).
CLASS_LABEL = {
    "ADI": "adipose", "DEB": "debris/necrosis", "LYM": "immune (lymphocytes)",
    "MUC": "mucus", "MUS": "muscle", "NORM": "normal mucosa",
    "STR": "stroma", "TUM": "tumor epithelium",
}


def class_directions():
    """Standardized class centroids from readout feats. Standardizing (mu/sd from the CLS
    feature stats) puts patch tokens and the class centroids in one comparable space, so a
    patch's nearest centroid is a meaningful tissue call despite the patch<->CLS shift."""
    d = np.load(EMB, allow_pickle=True)
    feats, labels = d["feats"], d["labels"]
    cls = list(d["class_names"])
    mu, sd = feats.mean(0), feats.std(0) + 1e-6
    names, cents = [], []
    for name in CLASS_LABEL:
        if name in cls:
            c = (feats[labels == cls.index(name)].mean(0) - mu) / sd   # standardized centroid
            names.append(name)
            cents.append(c / (np.linalg.norm(c) + 1e-12))
    return names, np.stack(cents), mu, sd      # (C, D) unit centroid dirs


def patch_directions(grid, class_names, cents, mu, sd):
    """Per-patch dominant tissue: standardize the patch into CLS space, then take the
    nearest class centroid by cosine. Returns (dominant class-index, softmax confidence)."""
    P = (grid - mu) / sd                                   # patch -> CLS-standardized space
    P = P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-9)
    sim = P @ cents.T                                      # (P, C) cosine to each centroid
    dom = sim.argmax(1)
    e = np.exp((sim - sim.max(1, keepdims=True)) * 6.0)    # sharpen for a readable confidence
    conf = (e / e.sum(1, keepdims=True))[np.arange(len(dom)), dom]
    return dom, conf


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


def diverging_map(grid, pos, neg):
    """Signed/diverging per-patch causal map for one (pos vs neg) axis on ONE patch grid.
    + toward pos concept, - toward neg; magnitude = |z| vs the matched-random null."""
    from collections import Counter
    axis = concept_axis(pos, neg)
    imp = attribution.patch_importance(grid, axis, n_null=N_NULL, seed=SEED)
    zsigned = np.asarray(imp["z"]) * np.sign(np.asarray(imp["scores"]))
    s = int(round(len(zsigned) ** 0.5))
    signed = zsigned.reshape(s, s); mag = np.abs(signed)
    lo, hi = float(mag.min()), float(mag.max())
    norm = (mag - lo) / (hi - lo + 1e-9)
    flat = signed.ravel(); top = int(np.argmax(flat))
    posL, negL = CLASS_LABEL.get(pos, pos), CLASS_LABEL.get(neg, neg)

    def _dir(p):
        v = float(flat[p]); toward = pos if v >= 0 else neg
        return {"patch": int(p), "row": int(p // s), "col": int(p % s),
                "toward": toward, "label": CLASS_LABEL.get(toward, toward),
                "pole": "pos" if v >= 0 else "neg", "z": round(v, 2)}
    order = np.argsort(-mag.ravel()); hot = [int(p) for p in order[:8]]
    return {
        "pos": pos, "neg": neg, "pos_label": posL, "neg_label": negL, "grid_side": s,
        "z_grid": np.round(signed, 3).tolist(), "norm_grid": np.round(norm, 4).tolist(),
        "z_min": round(float(signed.min()), 3), "z_max": round(float(signed.max()), 3),
        "top_z": round(float(imp["top_z"]), 2), "top_patch": top,
        "n_patches": int(imp["n_patches"]), "n_null": N_NULL, "verdict": imp["verdict"],
        "top_dir": _dir(top), "hot_dirs": [_dir(p) for p in hot],
        "hot_share": dict(Counter(_dir(p)["toward"] for p in hot)),
    }


def build_input_tile_maps(patch_grid):
    """Diverging maps for MANY concept axes, all on the ONE input tile the dashboard's
    Case + AutoResearch views take as input — so AutoResearch can render, per loop
    iteration, where the concept it just probed fires on THAT tile. Keyed by 'POS_NEG'."""
    from PIL import Image
    tile_path = os.path.join(OUT_DIR, "..", "input_tile.png")
    if not os.path.exists(tile_path):
        print("  (no input_tile.png — skipping input-tile maps)", flush=True); return
    grid = patch_grid(Image.open(tile_path).convert("RGB"))
    classes = ["TUM", "LYM", "STR", "NORM", "MUS", "MUC", "DEB", "ADI"]
    axes = {}
    for pos in classes:
        for neg in classes:
            if pos != neg:
                axes[f"{pos}_{neg}"] = diverging_map(grid, pos, neg)
    with open(os.path.join(OUT_DIR, "input_tile.json"), "w") as f:
        json.dump({"tile": "input_tile.png", "axes": axes}, f)
    print(f"  input-tile maps: {len(axes)} axes on input_tile.png -> input_tile.json", flush=True)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    by_class = load_tiles()
    patch_grid = load_phikon()
    print(f"loaded phikon-v2; tiles per class: "
          f"{ {k: len(v) for k, v in by_class.items()} }", flush=True)

    out = {}
    for spec in CONCEPTS:
        axis = concept_axis(spec["pos"], spec["neg"])
        pos_lbl = CLASS_LABEL.get(spec["pos"], spec["pos"])
        neg_lbl = CLASS_LABEL.get(spec["neg"], spec["neg"])
        tiles = by_class[spec["pos"]]

        # SIGNED importance: patch·axis is + when the patch leans toward the POS concept and
        # - when it leans toward the NEG concept; |·| z-scored vs the matched-random null is
        # the importance. So each patch has a magnitude (how concept-carrying) AND a direction
        # (which pole). Pick the tile with the strongest TOWARD-CONCEPT (positive) peak, so the
        # highlighted patches genuinely lean toward the concept the heatmap is about.
        best = None
        for i, tile in enumerate(tiles):
            grid = patch_grid(tile)
            imp = attribution.patch_importance(grid, axis, n_null=N_NULL, seed=SEED)
            zsigned = np.asarray(imp["z"]) * np.sign(np.asarray(imp["scores"]))
            peak = float(zsigned.max())          # strongest toward-POS-concept patch
            if best is None or peak > best[0]:
                best = (peak, i, imp, tile, zsigned)
        peak, idx, imp, tile, zsigned = best
        s = int(round(len(zsigned) ** 0.5))
        signed = zsigned.reshape(s, s)           # diverging: + toward pos, - toward neg
        mag = np.abs(signed)
        lo, hi = float(mag.min()), float(mag.max())
        norm = (mag - lo) / (hi - lo + 1e-9)     # 0..1 importance (drives the veil)

        flat = signed.ravel()
        top_patch = int(np.argmax(flat))         # most toward-concept patch (the ringed one)

        def _dir(p):
            v = float(flat[p]); toward = spec["pos"] if v >= 0 else spec["neg"]
            return {"patch": int(p), "row": int(p // s), "col": int(p % s),
                    "toward": toward, "label": CLASS_LABEL.get(toward, toward),
                    "pole": "pos" if v >= 0 else "neg", "z": round(v, 2)}
        order = np.argsort(-mag.ravel())          # patches by importance magnitude
        hot = [int(p) for p in order[:8]]
        hot_dirs = [_dir(p) for p in hot]
        from collections import Counter
        share = Counter(h["toward"] for h in hot_dirs)   # e.g. {"TUM":6,"NORM":2}

        png = f"{spec['concept']}.png"
        tile.save(os.path.join(OUT_DIR, png))
        out[spec["concept"]] = {
            "concept": spec["concept"],
            "label": spec["label"],
            "pos": spec["pos"], "neg": spec["neg"],
            "pos_label": pos_lbl, "neg_label": neg_lbl,
            "tile": f"heatmaps/{png}",
            "grid_side": s,
            "z_grid": np.round(signed, 3).tolist(),      # SIGNED per-patch z (+pos / -neg)
            "norm_grid": np.round(norm, 4).tolist(),      # [0,1] importance magnitude -> veil
            "z_min": round(float(signed.min()), 3), "z_max": round(float(signed.max()), 3),
            "top_z": round(float(imp["top_z"]), 2),
            "top_patch": top_patch,
            "n_patches": int(imp["n_patches"]),
            "n_null": N_NULL,
            "verdict": imp["verdict"],
            "layer": "readout",
            "tile_index": int(idx),
            # which pole each highlighted patch leans toward (the two ends of THIS axis)
            "top_dir": _dir(top_patch),                  # the ringed patch's direction
            "hot_dirs": hot_dirs,                         # top-8 by importance + their lean
            "hot_share": dict(share),                    # {class: count} among the hottest
        }
        print(f"  {spec['concept']:18} tile#{idx:2d}  top_z={imp['top_z']:6.2f}  "
              f"top patch -> {out[spec['concept']]['top_dir']['label']}  "
              f"lean {dict(share)}", flush=True)

    with open(os.path.join(OUT_DIR, "heatmaps.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {os.path.join(OUT_DIR, 'heatmaps.json')} ({len(out)} concepts)", flush=True)

    build_input_tile_maps(patch_grid)


if __name__ == "__main__":
    main()
