"""Decompose a TILE into spatial regions using patch-level SAE features.

Hits three of the target tasks at once:
  * SPATIAL DECOMPOSITION -- every one of the 256 patch tokens is assigned the SAE feature
    that dominates it, giving a 16x16 segmentation of the tile into morphological regions.
    (NOTE: a TILE, not a slide. Neither NCT-CRC-HE nor TCGA-CRC-DX ships whole-slide images --
    both are pre-tiled and non-contiguous. There is no WSI to paint. Do not say "slide".)
  * IMMUNE-HOT -- the patch features that fire on LYM tissue give a per-patch immune score,
    i.e. a spatial map of lymphocytic infiltration inside the tile.
  * CELL MORPHOLOGY -- H-Optimus-0 is patch-14 at ~0.5 microns/pixel, so ONE PATCH TOKEN
    covers ~7 microns: roughly a single cell. These features are therefore operating at
    approximately cell scale, not tissue scale.

Uses only the SAE and the tissue labels. No probe directions.
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import ListedColormap

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exemplars import load_tiles  # noqa: E402
from heatmap import patch_codes  # noqa: E402
from train_sae_topk import TopKSAE  # noqa: E402

ART = "/home/sagemaker-user/biolayer/artifacts"
GRID = 16
N_TILES = 4


def main() -> None:
    import timm
    import huggingface_hub

    huggingface_hub.login(token=os.environ["HF_TOKEN"])

    ck = torch.load(f"{ART}/sae_topk_patches.pt", map_location="cuda", weights_only=False)
    sae = TopKSAE(ck["d_model"], ck["n_features"], ck["k"]).cuda()
    sae.load_state_dict(ck["state_dict"])
    sae.eval()
    model = timm.create_model("hf-hub:bioptimus/H-optimus-0", pretrained=True,
                              init_values=1e-5, dynamic_img_size=False).eval().cuda()

    z = np.load(f"{ART}/hoptimus_patches.npz")
    lab, feats = z["labels"], z["feats"]
    cn = [str(c) for c in z["class_names"]]
    rng = np.random.default_rng(0)

    # Firing rate of every patch feature per tissue class -- computed on a RANDOM sample,
    # because the arrays are CLASS-SORTED and any prefix slice is biased.
    idx = np.sort(rng.choice(len(lab), 300000, replace=False))
    mu, sc = ck["mu"].cuda(), ck["scale"].cuda()
    rate = np.zeros((len(cn), ck["n_features"]), dtype=np.float32)
    cnt = np.zeros(len(cn))
    with torch.no_grad():
        for i in range(0, len(idx), 16384):
            sl = idx[i : i + 16384]
            x = torch.from_numpy(feats[sl].astype(np.float32)).cuda()
            _, zc, _ = sae((x - mu) / sc)
            fired = (zc > 0).float().cpu().numpy()
            for c in range(len(cn)):
                m = lab[sl] == c
                if m.any():
                    rate[c] += fired[m].sum(0)
                    cnt[c] += m.sum()
    rate /= np.maximum(cnt, 1)[:, None]

    LYM, TUM, STR = cn.index("LYM"), cn.index("TUM"), cn.index("STR")
    # selectivity = fires on this class much more than on any other
    def selective(c, n=8):
        other = np.delete(rate, c, axis=0).max(0)
        return np.argsort(-(rate[c] - other))[:n]

    immune_feats = selective(LYM)
    print("immune (LYM-selective) patch features:", immune_feats[:5].tolist(), flush=True)

    # Pick tiles spanning the IMMUNE SPECTRUM. An immune-hot map is only meaningful if you can
    # contrast hot against cold -- showing four immune-cold tumour tiles proves nothing.
    # Score each tile by how much its stored patches drive the LYM-selective features.
    tid = z["tile_ids"]
    samp = np.sort(rng.choice(len(tid), 400000, replace=False))
    imm_score = {}
    with torch.no_grad():
        for i in range(0, len(samp), 16384):
            sl = samp[i : i + 16384]
            x = torch.from_numpy(feats[sl].astype(np.float32)).cuda()
            _, zc, _ = sae((x - mu) / sc)
            s = zc[:, immune_feats].sum(1).cpu().numpy()
            for t, v in zip(tid[sl], s):
                imm_score[int(t)] = imm_score.get(int(t), 0.0) + float(v)
    # Rank WITHIN tumour tiles: the coldest tiles overall are fat/background, not tumour, so
    # filtering the global tail for TUM finds nothing. We want immune-hot vs immune-cold TUMOUR.
    tile_label = {}
    for t in imm_score:
        tile_label[t] = int(lab[tid == t][0])
    tum_tiles = [t for t in imm_score if tile_label[t] == TUM]
    tum_ranked = sorted(tum_tiles, key=lambda t: -imm_score[t])
    hot = tum_ranked[:2]
    cold = tum_ranked[-2:]
    chosen = np.array(hot + cold)
    tags = ["immune-HOT tumour", "immune-HOT tumour", "immune-COLD tumour", "immune-COLD tumour"]
    print("chosen tiles:", chosen.tolist(), "labels:",
          [cn[lab[tid == t][0]] for t in chosen], flush=True)

    imgs = load_tiles(chosen)
    codes = patch_codes(model, sae, ck, imgs, "cuda")   # (B, 256, F) -- ALL 256 patches

    # Region features: pick the ones that VARY WITHIN the tiles. Choosing by global max gives
    # one feature that blankets every patch and a segmentation with no contrast.
    P = codes.cpu().numpy()                                  # (B, 256, F)
    within_var = P.std(axis=1).mean(0)                       # variation across patches
    top_global = np.argsort(-within_var)[:6]
    palette = ListedColormap(["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#46f0f0"])

    vmax_imm = float(np.percentile(P[:, :, immune_feats].sum(-1), 99.5))
    fig, axes = plt.subplots(3, N_TILES, figsize=(3.3 * N_TILES, 10.2))
    for j in range(N_TILES):
        C = codes[j].cpu().numpy()                      # (256, F)

        axes[0, j].imshow(imgs[j]); axes[0, j].set_title(f"{tags[j]}  ({cn[lab[tid==chosen[j]][0]]})", fontsize=10, fontweight="bold")

        # spatial decomposition: dominant feature per patch
        seg = np.array([top_global[np.argmax(C[p, top_global])] for p in range(GRID * GRID)])
        segi = np.array([list(top_global).index(s) for s in seg]).reshape(GRID, GRID)
        axes[1, j].imshow(imgs[j], alpha=0.35)
        axes[1, j].imshow(np.kron(segi, np.ones((14, 14))), cmap=palette, alpha=0.65,
                          vmin=0, vmax=5, interpolation="nearest")
        axes[1, j].set_title("spatial regions\n(dominant SAE feature per patch)", fontsize=9)

        # immune-hot map: total activation of LYM-selective features per patch
        imm = C[:, immune_feats].sum(1).reshape(GRID, GRID)
        axes[2, j].imshow(imgs[j], alpha=0.45)
        # SHARED colour scale across all tiles -- otherwise per-tile normalisation makes an
        # immune-cold tile look just as "hot" as a hot one.
        axes[2, j].imshow(np.kron(imm, np.ones((14, 14))), cmap="inferno", alpha=0.62,
                          vmin=0, vmax=vmax_imm, interpolation="bilinear")
        axes[2, j].set_title(f"immune-hot map (shared scale)\ntotal LYM activation = {imm.sum():.0f}",
                             fontsize=9)

    for a in axes.ravel():
        a.set_xticks([]); a.set_yticks([])
    fig.suptitle(
        "Patch-level SAE decomposition of a 224x224 tile (16x16 = 256 patch tokens)\n"
        "Each patch token covers ~7 microns at 0.5 MPP -- roughly one cell. "
        "This is a TILE, not a whole slide (neither dataset ships WSIs).",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = f"{ART}/figs/FIG4_spatial_decomposition.png"
    fig.savefig(out, dpi=120)
    print("wrote", out)


if __name__ == "__main__":
    main()
