"""Visualize the ranked-patch 'patchwork' + where the patches came from on the WSI.

Rank a slide's 14x14 patch tokens by a concept axis (the pooling/ranking direction), then:
  1. assemble the top-N patches into a square 'patchwork' image — the patched-together square
     the question-conditioned pooler would summarize;
  2. show WHERE those patches sit on the slide via the EXISTING patch-gallery visualizer
     (biolayer/data/wsi_patch_gallery: multiple regions on the whole-slide map, click to
     highlight the selected one) — the regional distribution — plus a density overlay of all
     selected patch centers on the slide thumbnail.

    python deploy/sagemaker/patchwork_view.py \
        --wsi s3://bucketbiolayer/wsi/BRACS/BRACS_1003675.svs --pos TUM

Outputs (default /tmp/patchwork): <stem>_patchwork.png, <stem>_distribution.png,
<stem>_gallery.html (the existing interactive WSI-map visualizer over the top regions).
"""
import argparse
import html
import json
import os
import sys
import tempfile

import boto3
import numpy as np
from PIL import Image, ImageDraw

Image.MAX_IMAGE_PIXELS = None
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

from biolayer.causal import probe as P                     # noqa: E402
from biolayer.data import loader                            # noqa: E402
from biolayer.data import wsi_patch_gallery as gallery      # noqa: E402
from biolayer.data.wsi_reader import open_wsi               # noqa: E402

B, REGION = "bucketbiolayer", "us-west-2"
s3 = boto3.client("s3", region_name=REGION)


def build_gallery_lowmem(reader, slide_uri, stem, regions, out_html, win=1536,
                         quality=86, display_max=1024, overview_max=1400):
    """Same visualizer UI as wsi_patch_gallery (its _TEMPLATE + _jpeg_uri), but crop each
    region via openslide read_region (KB-scale random access) instead of the whole level-0
    plane — so it runs on a small box where the full-plane read (here 8 GB) OOMs."""
    l0w, l0h = reader.dimensions
    thumb, _ = reader.thumbnail(overview_max)
    over_uri, _ = gallery._jpeg_uri(thumb, quality=82)
    out = []
    for cx, cy, title, desc in regions:
        ox = max(0, min(int(cx) - win // 2, l0w - win))
        oy = max(0, min(int(cy) - win // 2, l0h - win))
        crop = np.asarray(reader.read_region((ox, oy), 0, (win, win)).convert("RGB"))
        uri, _ = gallery._jpeg_uri(crop, quality=quality, maxside=display_max)
        out.append({"title": title, "desc": desc, "cx": ox + win // 2, "cy": oy + win // 2,
                    "ox": ox, "oy": oy, "w": win, "h": win, "img": uri,
                    "box": {"l": round(100 * ox / l0w, 3), "t": round(100 * oy / l0h, 3),
                            "w": round(100 * win / l0w, 3), "h": round(100 * win / l0h, 3)}})
    mpp = reader.mpp
    mag = round(10.0 / mpp) if mpp else None
    page = (gallery._TEMPLATE
            .replace("__STEM__", html.escape(stem)).replace("__SRCURI__", html.escape(slide_uri))
            .replace("__OVERVIEW__", over_uri).replace("__PATCHES__", json.dumps(out))
            .replace("__L0W__", str(l0w)).replace("__L0H__", str(l0h))
            .replace("__MPP_NUM__", repr(mpp if mpp else 0.0))
            .replace("__MPP_TXT__", f"{mpp:.4f} µm/px" if mpp else "unknown")
            .replace("__MAG_TXT__", f"≈ {mag}×" if mag else "unknown"))
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(page)
    return {"n_patches": len(out)}


def _dl(key, local):
    if not os.path.exists(local):
        print(f"downloading {key} ...", flush=True)
        s3.download_file(B, key, local)
    return local


def rank_patches(stem, pos, tmp, chunk=20_000, dataset_slug=None):
    """Score every 14x14 patch of one slide by the pos-vs-rest axis; return sorted meta.
    float32 + small chunks so a multi-hundred-k-row patch shard doesn't blow local RAM.

    dataset_slug picks which reference the concept axis is fit from: None = default colon
    (NCT-CRC) axes; "bcss_breast" = breast-native TUM/STR/LYM axes. Use the breast axes on a
    breast WSI so the ranking is not a cross-tissue transfer."""
    V = np.load(_dl(f"embeddings/wsi/{stem}/patch_vectors.npy", os.path.join(tmp, f"{stem}_pv.npy")),
                mmap_mode="r")
    M = np.load(_dl(f"embeddings/wsi/{stem}/patch_meta.npz", os.path.join(tmp, f"{stem}_pm.npz")))
    feats, labels, cn, src = loader.load("h_optimus_0", "train", dataset_slug=dataset_slug)
    print(f"[patchwork] '{pos}-vs-rest' axis fit from {src}", flush=True)
    labels, cn = np.asarray(labels), list(cn)
    fit = P.fit_probe(np.asarray(feats), (labels == cn.index(pos)).astype(int))
    d = fit["direction"].astype("float32")
    mean = fit["scaler"].mean_.astype("float32")
    scale = fit["scaler"].scale_.astype("float32")
    n = len(V)
    scores = np.empty(n, dtype="float32")
    for i in range(0, n, chunk):
        Vo = np.asarray(V[i:i + chunk], dtype="float32")
        scores[i:i + len(Vo)] = ((Vo - mean) / scale) @ d
    order = np.argsort(-scores)
    print(f"[patchwork] scored {n} patches (score {scores.min():+.2f}..{scores.max():+.2f})", flush=True)
    return order, scores, M


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--wsi", default="s3://bucketbiolayer/wsi/BRACS/BRACS_1003675.svs")
    ap.add_argument("--pos", default="TUM", help="axis positive class (one-vs-rest) used to rank")
    ap.add_argument("--axis-dataset", default=None,
                    help="reference the concept axis is fit from: default = colon (NCT-CRC); "
                         "'bcss_breast' = breast-native axes (use for a breast WSI)")
    ap.add_argument("--n-square", type=int, default=256, help="patches in the patchwork (16x16=256)")
    ap.add_argument("--n-regions", type=int, default=24, help="top regions to show in the gallery")
    ap.add_argument("--cell", type=int, default=28, help="px per patch in the patchwork")
    ap.add_argument("--out", default="/tmp/patchwork")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    tmp = tempfile.gettempdir()

    stem = os.path.splitext(os.path.basename(args.wsi))[0]
    local = _dl(args.wsi[5:].split("/", 1)[1], os.path.join(tmp, os.path.basename(args.wsi)))
    order, scores, M = rank_patches(stem, args.pos, tmp, dataset_slug=args.axis_dataset)

    reader = open_wsi(local)
    mpp = reader.mpp or 0.25
    step = int(round(224 * max(0.5 / mpp, 1.0)))            # tile footprint in level-0 px
    psz = step / 16.0                                       # one 14x14 patch's level-0 footprint
    tx, ty = M["tile_x"], M["tile_y"]
    pr, pc = M["patch_row"], M["patch_col"]

    def patch_box(idx):
        x = int(tx[idx] + pc[idx] * psz)
        y = int(ty[idx] + pr[idx] * psz)
        return x, y, int(round(psz))

    # 1) assemble the patchwork square (top-N patches in rank order, raster 16xK)
    top = order[:args.n_square]
    side = int(round(len(top) ** 0.5))
    canvas = Image.new("RGB", (side * args.cell, side * args.cell), (245, 245, 245))
    for k, idx in enumerate(top):
        x, y, sz = patch_box(idx)
        patch = reader.read_region((x, y), 0, (sz, sz)).convert("RGB").resize((args.cell, args.cell))
        r, c = divmod(k, side)
        if r < side:
            canvas.paste(patch, (c * args.cell, r * args.cell))
    pw_path = os.path.join(args.out, f"{stem}_patchwork.png")
    canvas.save(pw_path)
    print(f"[patchwork] {len(top)} patches -> {pw_path}  "
          f"(score {scores[top].min():+.2f}..{scores[top].max():+.2f})", flush=True)

    # 2) density overlay: all selected patch centers on the slide thumbnail
    thumb, ds = reader.thumbnail(1400)
    ov = Image.fromarray(thumb).convert("RGB")
    draw = ImageDraw.Draw(ov, "RGBA")
    for idx in top:
        x, y, sz = patch_box(idx)
        cx, cy = (x + sz / 2) / ds, (y + sz / 2) / ds
        draw.ellipse([cx - 2, cy - 2, cx + 2, cy + 2], fill=(205, 40, 110, 170))
    dist_path = os.path.join(args.out, f"{stem}_distribution.png")
    ov.save(dist_path)
    print(f"[patchwork] distribution overlay -> {dist_path}", flush=True)

    # 3) the EXISTING visualizer over the top regions (WSI map + highlight the selected)
    regions = []
    for k, idx in enumerate(order[:args.n_regions]):
        x, y, sz = patch_box(idx)
        regions.append((x + sz // 2, y + sz // 2, f"#{k + 1}  {args.pos}", f"score {scores[idx]:+.2f}"))
    html_path = os.path.join(args.out, f"{stem}_gallery.html")
    meta = build_gallery_lowmem(reader, args.wsi, stem, regions, html_path)
    print(f"[patchwork] regional gallery ({meta['n_patches']} regions, existing visualizer UI) "
          f"-> {html_path}", flush=True)


if __name__ == "__main__":
    main()
