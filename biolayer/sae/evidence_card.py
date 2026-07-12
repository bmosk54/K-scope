"""Render ONE evidence card a pharma researcher can read without knowing what an SAE is.

THE PROBLEM THIS FIXES. The tool was returning `top_features: [2524, 3307, ...]` -- feature
indices. That is an implementation detail that leaked into the product. A K Pro user is a
pharma researcher, not an ML engineer; "feature 2524" tells them nothing and they cannot act
on it. What they can act on is:

    1. WHAT the morphology is        -> exemplar tiles, not numbers
    2. WHERE in the image it is      -> a spatial map over the tile
    3. HOW ROBUST the answer is      -> how many features overturn it, vs a random control
    4. WHAT the model sees INSTEAD   -> the failure mode

The card carries all four in one image. Feature indices stay in the machine-readable payload
for Owkin Zero; they never appear as the headline for a human.
"""

from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exemplars import load_tiles  # noqa: E402
from heatmap import patch_codes  # noqa: E402

ART = "/home/sagemaker-user/biolayer/artifacts"
GRID = 16


def render(
    concept: str,
    top_feats: np.ndarray,
    codes: np.ndarray,
    labels: np.ndarray,
    class_names: list[str],
    ks: list[int],
    curve: list[float],
    null: list[float],
    p0: float,
    fallback: dict,
    broke: int | None,
    patch_sae=None,
    patch_ck=None,
    model=None,
    concept_patch_feats=None,
    out: str | None = None,
) -> str:
    """One card: the morphology, where it is, how robust the answer is, what replaces it."""
    out = out or f"{ART}/figs/CARD_{concept}.png"
    os.makedirs(os.path.dirname(out), exist_ok=True)

    fig = plt.figure(figsize=(15.5, 8.4))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.05, 1.0], hspace=0.32, wspace=0.22)

    # ---------- ROW 1: WHAT the model looked at (morphology, not numbers) ----------
    ex = np.argsort(-codes[:, top_feats[0]])[:4]
    for j, im in enumerate(load_tiles(ex)):
        a = fig.add_subplot(gs[0, j])
        a.imshow(im)
        a.set_xticks([]); a.set_yticks([])
        for sp in a.spines.values():
            sp.set_edgecolor("#c0392b"); sp.set_linewidth(2.5)
        if j == 0:
            a.set_ylabel("THE MORPHOLOGY\nthe model relied on", fontsize=10.5,
                         fontweight="bold", color="#c0392b")

    # what tissue do those tiles carry? -> a plain-language description of the morphology
    top = np.argsort(-codes[:, top_feats[0]])[:100]
    cnt = np.bincount(labels[top], minlength=len(class_names))
    spread = ", ".join(f"{class_names[i]} {v}%" for i, v in
                       sorted(enumerate(cnt), key=lambda kv: -kv[1])[:3] if v > 0)
    fig.text(0.5, 0.955,
             f"What the model used to decide '{concept}'  —  tiles that most drive its top feature "
             f"({spread})",
             ha="center", fontsize=11.5)

    # ---------- ROW 2a: HOW ROBUST (the intervention + its control) ----------
    ax = fig.add_subplot(gs[1, :2])
    xs = [0] + list(ks)
    ax.plot(xs, [p0] + curve, "o-", color="#c0392b", lw=2.6,
            label=f"delete the features the model uses for '{concept}'")
    ax.plot(xs, [p0] + null, "s--", color="#8e8e8e", lw=2,
            label="delete the same number of RANDOM features (control)")
    ax.axhline(0.5, color="k", ls=":", lw=1)
    ax.text(xs[-1], 0.52, "answer overturned below here", ha="right", fontsize=8, color="#555")
    ax.set_xscale("symlog", linthresh=1); ax.set_xlim(-0.05, xs[-1] * 1.25); ax.set_ylim(0, 1.08)
    ax.set_xlabel("number of visual features deleted from the model")
    ax.set_ylabel(f"model's confidence that this is {concept}")
    verdict = (f"AUDITABLE — only {broke} features carry this answer"
               if broke and broke <= 20 else
               "REDUNDANT — no small set of features carries this answer")
    ax.set_title(f"How robust is the answer?   {verdict}", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8.5, loc="lower left"); ax.grid(alpha=0.25)

    # ---------- ROW 2b: WHAT IT SEES INSTEAD (the failure mode) ----------
    ax2 = fig.add_subplot(gs[1, 2])
    if fallback:
        ks_, vs_ = list(fallback.keys()), list(fallback.values())
        ax2.barh(ks_, vs_, color="#2e86ab")
        ax2.invert_yaxis()
    ax2.set_xlabel("probability gained")
    ax2.set_title("What the model sees\nINSTEAD, once deleted", fontsize=10.5, fontweight="bold")
    ax2.grid(alpha=0.2, axis="x")

    # ---------- ROW 2c: WHERE in the image (spatial, cell-scale) ----------
    ax3 = fig.add_subplot(gs[1, 3])
    if patch_sae is not None and model is not None and concept_patch_feats is not None:
        # "Where is it?" is only informative on a MIXED tile. On a pure-LYM tile the immune
        # features fire on every patch, the map saturates, and the answer degenerates to
        # "everywhere" -- true, useless, and it hides the tissue. So pick, among candidates,
        # the tile with the highest spatial CONTRAST in this concept's patch features.
        cands = np.argsort(-codes[:, top_feats[0]])[:40:4]
        tiles = load_tiles([int(t) for t in cands])
        pcs = patch_codes(model, patch_sae, patch_ck, tiles, "cuda").cpu().numpy()
        maps = pcs[:, :, concept_patch_feats].sum(-1)            # (n, 256)
        contrast = maps.std(1) / (maps.mean(1) + 1e-6)           # spatial variation
        b = int(np.argmax(contrast))
        hm = maps[b].reshape(GRID, GRID)
        ax3.imshow(tiles[b], alpha=0.62)
        ax3.imshow(np.kron(hm / (hm.max() + 1e-9), np.ones((14, 14))), cmap="inferno",
                   alpha=0.5, interpolation="bilinear")
    ax3.set_xticks([]); ax3.set_yticks([])
    ax3.set_title("WHERE in the tissue\n(~7 microns/patch ≈ cell scale)", fontsize=10.5,
                  fontweight="bold")

    fig.savefig(out, dpi=118, bbox_inches="tight")
    plt.close(fig)
    return out
