"""FIGURE 1 — "What is the model seeing in THIS image?"

Decompose one histology tile into the SAE features that build its representation, and show
what each of those features means by its own exemplar tiles.

This is the SAE's entire value proposition on one page. Without the SAE, the model's view of a
tile is a single opaque 1536-d vector. With it, that vector becomes a short list of features,
each of which you can SEE the meaning of. The left column is the input; each row to the right
is one component of the model's internal description of it, illustrated by the other tiles in
the dataset that drive the same feature.

Reading it: if the exemplars in a row look like the input tile, that feature is a coherent,
nameable concept. If a row's exemplars look like nothing in particular, that feature is not
interpretable and you should not build a story on it. The figure is honest either way.

Uses NOTHING but the SAE and the raw tiles -- no probe directions anywhere.
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exemplars import load_tiles  # noqa: E402
from hypothesis import class_purity  # noqa: E402
from train_sae_topk import TopKSAE  # noqa: E402

ART = "/home/sagemaker-user/biolayer/artifacts"
N_FEATS = 5      # components of the representation to show
N_EX = 5         # exemplar tiles per component


def main() -> None:
    z = np.load(f"{ART}/hoptimus_100k.npz")
    X = z["globals"][:, 2].astype(np.float32)   # block 39 CLS
    y = z["labels"]
    cn = [str(c) for c in z["class_names"]]

    ck = torch.load(f"{ART}/sae_topk_hoptimus_L39_global.pt", map_location="cuda", weights_only=False)
    sae = TopKSAE(ck["d_model"], ck["n_features"], ck["k"]).cuda()
    sae.load_state_dict(ck["state_dict"])
    sae.eval()

    xt = torch.from_numpy(X).cuda()
    xt = (xt - ck["mu"].cuda()) / ck["scale"].cuda()
    with torch.no_grad():
        C = torch.cat([sae(xt[i : i + 8192])[1] for i in range(0, len(xt), 8192)]).cpu().numpy()

    # Purity of every live feature: >=0.6 means a tissue label already names it.
    purity_of = {}
    for f in range(C.shape[1]):
        if (C[:, f] > 0).sum() >= 100:
            purity_of[f], _ = class_purity(C[:, f], y, 9)

    # Pick a TUMOUR tile whose top-N features include at least one the label vocabulary CANNOT
    # name -- otherwise the figure has nothing to illustrate. This is an ILLUSTRATIVE example,
    # chosen to show the mechanism; the evidence that unnamed features are real and not noise
    # is the null test in `discover`, not this picture. Most tiles decompose entirely into
    # named features, and that is itself the honest finding.
    tum = np.where(y == cn.index("TUM"))[0]
    target = None
    for t in tum[np.argsort(-C[tum].sum(1))]:
        top = np.argsort(-C[t])[:N_FEATS]
        if any(purity_of.get(int(f), 1.0) < 0.6 for f in top):
            target = t
            break
    if target is None:
        raise SystemExit("no tumour tile has an unnamed feature in its top-5 -- report that instead")

    order = np.argsort(-C[target])[:N_FEATS]
    total = C[target].sum()

    # cols: [input tile | text label | N_EX exemplars]
    fig = plt.figure(figsize=(4.6 + N_EX * 2.0, 1.6 + N_FEATS * 2.0))
    gs = fig.add_gridspec(N_FEATS, N_EX + 2, width_ratios=[1.7, 1.35] + [1.0] * N_EX,
                          wspace=0.05, hspace=0.10)

    # left: the input tile, spanning all rows
    ax = fig.add_subplot(gs[:, 0])
    ax.imshow(load_tiles([target])[0])
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"INPUT TILE\n(labelled {cn[y[target]]})", fontsize=11, fontweight="bold", pad=8)
    for sp in ax.spines.values():
        sp.set_linewidth(2.5)

    for r, f in enumerate(order):
        purity, counts = class_purity(C[:, f], y, 9)
        spread = ", ".join(f"{cn[i]} {v}%" for i, v in sorted(counts.items(), key=lambda kv: -kv[1])[:3])
        named = purity >= 0.6
        share = 100 * C[target, f] / max(total, 1e-9)
        col = "#888888" if named else "#c0392b"

        # dedicated text axis, so nothing overlaps the input tile
        t = fig.add_subplot(gs[r, 1])
        t.axis("off")
        tag = "already named by\na tissue label" if named else "NOT nameable by\nany tissue label"
        t.text(0.97, 0.5,
               f"feature {int(f)}\n{share:.0f}% of this tile's code\n{spread}\n{tag}",
               transform=t.transAxes, ha="right", va="center", fontsize=9.5,
               color="#333333" if named else "#c0392b",
               fontweight="normal" if named else "bold", linespacing=1.5)

        for c, im in enumerate(load_tiles(np.argsort(-C[:, f])[:N_EX])):
            a = fig.add_subplot(gs[r, c + 2])
            a.imshow(im)
            a.set_xticks([]); a.set_yticks([])
            for sp in a.spines.values():
                sp.set_edgecolor(col); sp.set_linewidth(2.2)

    fig.suptitle(
        "How the model represents ONE tile: its 1536-d embedding decomposed into 5 SAE features\n"
        "Each row = one component of the model's internal description. The tiles to the right are what "
        "that component means.\nGrey = the component matches a tissue label you already have. "
        "Red = the model is using something no tissue label can name.",
        fontsize=11.5, y=0.99,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out = f"{ART}/figs/FIG1_tile_decomposition.png"
    fig.savefig(out, dpi=115, bbox_inches="tight")
    print("wrote", out)
    for f in order:
        p, _ = class_purity(C[:, f], y, 9)
        print(f"   feature {int(f):<5d} purity={p:.2f}  {'named' if p>=0.6 else 'UNNAMED'}")


if __name__ == "__main__":
    main()
