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

# Known WSIs the gallery's source switcher can jump between. Each becomes an entry in the
# clickable source menu; the href is the sibling gallery file for the SAME concept, so the
# switch preserves which axis you're viewing. Extend as more slides are extracted.
KNOWN_WSIS = [
    {"uri": "s3://bucketbiolayer/wsi/TCGA-BRCA/"
            "TCGA-E2-A14P-01Z-00-DX1.663B02FF-C64B-41A6-8685-FD61CD76F9C6.svs",
     "slug": "tcga-brca-a14p", "title": "TCGA-BRCA · invasive carcinoma"},
    {"uri": "s3://bucketbiolayer/wsi/BRACS/BRACS_1003675.svs",
     "slug": "bracs-1003675", "title": "BRACS · breast subtyping"},
]


# Concepts the axis switcher can flip between (order preserved). Label shown in the menu.
AXIS_LABELS = {"TUM": "Tumor epithelium (TUM)", "STR": "Stroma (STR)", "LYM": "Lymphocytes (LYM)"}


def _resolve_name(wsi_uri, data_stem, slug_override=None, name_override=None):
    """Map a WSI to a readable (slug, title): filename/URL slug + display title. Looks the
    slide up in KNOWN_WSIS; --slug/--name override; unknown slides fall back to the raw stem."""
    entry = next((w for w in KNOWN_WSIS if w["uri"] == wsi_uri
                  or os.path.splitext(os.path.basename(w["uri"]))[0] == data_stem), None)
    slug = slug_override or (entry["slug"] if entry else data_stem)
    title = name_override or (entry["title"] if entry else data_stem)
    return slug, title


def crop_payload(reader, regions, regions_bottom=None, win=1536, quality=86,
                 display_max=1024, overview_max=1400):
    """EXPENSIVE step: whole-slide overview + native-res crops -> a template-ready payload
    (crops as JPEG data-URIs, coords, dims). Cropped via openslide read_region (KB-scale
    random access) so it runs on a small box. This payload is what gets cached; a UI/template
    change never needs to recompute it."""
    l0w, l0h = reader.dimensions
    thumb, _ = reader.thumbnail(overview_max)
    over_uri, _ = gallery._jpeg_uri(thumb, quality=82)

    def crop_set(regs):
        out = []
        for cx, cy, title, desc in regs:
            ox = max(0, min(int(cx) - win // 2, l0w - win))
            oy = max(0, min(int(cy) - win // 2, l0h - win))
            crop = np.asarray(reader.read_region((ox, oy), 0, (win, win)).convert("RGB"))
            uri, _ = gallery._jpeg_uri(crop, quality=quality, maxside=display_max)
            out.append({"title": title, "desc": desc, "cx": ox + win // 2, "cy": oy + win // 2,
                        "ox": ox, "oy": oy, "w": win, "h": win, "img": uri,
                        "box": {"l": round(100 * ox / l0w, 3), "t": round(100 * oy / l0h, 3),
                                "w": round(100 * win / l0w, 3), "h": round(100 * win / l0h, 3)}})
        return out

    mpp = reader.mpp
    return {"out": crop_set(regions),
            "out_bottom": crop_set(regions_bottom) if regions_bottom else None,
            "over_uri": over_uri, "l0w": l0w, "l0h": l0h,
            "mpp": mpp, "mag": round(10.0 / mpp) if mpp else None}


def render_payload(payload, out_html, stem, slide_uri, axis_note="", sources=None, axes=None):
    """CHEAP step: fill the gallery template from a precomputed payload. No ranking, no slide
    reads — this is the ONLY step a UI/template change affects, so it runs in milliseconds."""
    out, out_bottom, mpp, mag = payload["out"], payload.get("out_bottom"), payload["mpp"], payload["mag"]
    page = (gallery._TEMPLATE
            .replace("__STEM__", html.escape(stem)).replace("__SRC_TITLE__", html.escape(stem))
            .replace("__SRCURI__", html.escape(slide_uri))
            .replace("__OVERVIEW__", payload["over_uri"]).replace("__PATCHES__", json.dumps(out))
            .replace("__PATCHES_BOTTOM__", json.dumps(out_bottom) if out_bottom is not None else "null")
            .replace("__L0W__", str(payload["l0w"])).replace("__L0H__", str(payload["l0h"]))
            .replace("__MPP_NUM__", repr(mpp if mpp else 0.0))
            .replace("__MPP_TXT__", f"{mpp:.4f} µm/px" if mpp else "unknown")
            .replace("__MAG_TXT__", f"≈ {mag}×" if mag else "unknown")
            .replace("__AXIS_NOTE__", axis_note)
            .replace("__SOURCES__", json.dumps(sources) if sources else "null")
            .replace("__AXES__", json.dumps(axes) if axes else "null"))
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
    ap.add_argument("--name", default=None, help="readable display title shown in the gallery "
                    "(default: from KNOWN_WSIS, else the raw slide stem)")
    ap.add_argument("--slug", default=None, help="short filesystem/URL slug for output filenames "
                    "(default: from KNOWN_WSIS, else the raw slide stem)")
    ap.add_argument("--axis-dataset", default=None,
                    help="reference the concept axis is fit from: default = colon (NCT-CRC); "
                         "'bcss_breast' = breast-native axes (use for a breast WSI)")
    ap.add_argument("--n-square", type=int, default=256, help="patches in the patchwork (16x16=256)")
    ap.add_argument("--n-regions", type=int, default=24, help="top regions to show in the gallery")
    ap.add_argument("--min-sep", type=int, default=1536,
                    help="min level-0 px between selected regions (spatial NMS; default = crop "
                         "window, so shown crops don't overlap). Set 0 to disable.")
    ap.add_argument("--cell", type=int, default=28, help="px per patch in the patchwork")
    ap.add_argument("--out", default="/tmp/patchwork")
    ap.add_argument("--rerank", action="store_true",
                    help="force a full re-rank + re-crop even if a matching payload cache exists")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    tmp = tempfile.gettempdir()

    data_stem = os.path.splitext(os.path.basename(args.wsi))[0]   # real stem: the S3 embeddings key
    slug, title = _resolve_name(args.wsi, data_stem, args.slug, args.name)
    html_path = os.path.join(args.out, f"{slug}_{args.pos}_gallery.html")

    # UI-cheap bits (recomputed every run so renames / switcher edits take effect immediately)
    ref_label = {"bcss_breast": "breast (BCSS)", None: "colon (NCT-CRC)"}.get(
        args.axis_dataset, args.axis_dataset)
    # the concept pill is a switcher: click to flip the ranking axis for THIS slide
    axis_note = (
        '<p class="axis-note">Patches ranked by the '
        '<span class="ax-switch" id="ax-switch">'
        f'<span class="pill" id="ax-pill">{args.pos}</span>'
        '<span class="ax-caret" id="ax-caret" hidden>▾</span>'
        '<span class="ax-menu" id="ax-menu" hidden></span>'
        '</span>'
        f' concept axis · fit from {ref_label}</p>')
    axes = [{"label": AXIS_LABELS.get(a, a), "href": f"{slug}_{a}_gallery.html",
             "current": a == args.pos} for a in AXIS_LABELS]
    sources = [{"label": w["title"], "href": f"{w['slug']}_{args.pos}_gallery.html",
                "current": w["slug"] == slug} for w in KNOWN_WSIS]

    # Payload cache: the ranking + crops are deterministic in these params and are the whole
    # cost (13GB mmap + 4.3M-patch matmul + slide crops). Cache them, keyed by the ranking
    # params only, so a UI/template change re-renders in milliseconds. --rerank forces recompute.
    cache_path = os.path.join(args.out, f".payload_{slug}_{args.pos}.json")
    key = {"data_stem": data_stem, "pos": args.pos, "axis_dataset": args.axis_dataset,
           "min_sep": args.min_sep, "n_regions": args.n_regions,
           "win": 1536, "quality": 86, "display_max": 1024, "overview_max": 1400}
    payload = None
    if not args.rerank and os.path.exists(cache_path):
        try:
            cached = json.load(open(cache_path))
            if cached.get("key") == key:
                payload = cached["payload"]
                print(f"[patchwork] CACHE HIT — re-render only (no re-rank): {os.path.basename(cache_path)}",
                      flush=True)
        except Exception:
            payload = None

    if payload is None:
        local = _dl(args.wsi[5:].split("/", 1)[1], os.path.join(tmp, os.path.basename(args.wsi)))
        order, scores, M = rank_patches(data_stem, args.pos, tmp, dataset_slug=args.axis_dataset)
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

        # patchwork square (top-N patches in rank order, raster 16xK)
        top = order[:args.n_square]
        side = int(round(len(top) ** 0.5))
        canvas = Image.new("RGB", (side * args.cell, side * args.cell), (245, 245, 245))
        for k, idx in enumerate(top):
            x, y, sz = patch_box(idx)
            patch = reader.read_region((x, y), 0, (sz, sz)).convert("RGB").resize((args.cell, args.cell))
            r, c = divmod(k, side)
            if r < side:
                canvas.paste(patch, (c * args.cell, r * args.cell))
        canvas.save(os.path.join(args.out, f"{slug}_patchwork.png"))

        # density overlay: all selected patch centers on the slide thumbnail
        thumb, ds = reader.thumbnail(1400)
        ov = Image.fromarray(thumb).convert("RGB")
        draw = ImageDraw.Draw(ov, "RGBA")
        for idx in top:
            x, y, sz = patch_box(idx)
            cx, cy = (x + sz / 2) / ds, (y + sz / 2) / ds
            draw.ellipse([cx - 2, cy - 2, cx + 2, cy + 2], fill=(205, 40, 110, 170))
        ov.save(os.path.join(args.out, f"{slug}_distribution.png"))

        # Spatial NMS: the ranking scores patch TOKENS (~28px apart) but each is shown as a
        # `win`-px crop, so neighbouring tokens in one homogeneous blob would render as
        # ~98%-overlapping duplicates. Greedily accept a region only if its center is >= min_sep
        # from every already-accepted one -> n spatially DISTINCT regions, not n top scores.
        def select_spread(ranked, n, min_sep):
            picked, centers, s2 = [], [], min_sep * min_sep
            for idx in ranked:
                x, y, sz = patch_box(idx)
                cx, cy = x + sz / 2, y + sz / 2
                if all((cx - px) ** 2 + (cy - py) ** 2 >= s2 for px, py in centers):
                    picked.append(idx); centers.append((cx, cy))
                    if len(picked) >= n:
                        break
            return picked

        def to_regions(idxs):
            return [(x + sz // 2, y + sz // 2, f"#{k + 1}", f"score {scores[idx]:+.2f}")
                    for k, idx in enumerate(idxs) for x, y, sz in [patch_box(idx)]]

        regions = to_regions(select_spread(order, args.n_regions, args.min_sep))
        regions_bottom = to_regions(select_spread(order[::-1], args.n_regions, args.min_sep))
        payload = crop_payload(reader, regions, regions_bottom)
        json.dump({"key": key, "payload": payload}, open(cache_path, "w"))
        print(f"[patchwork] ranked + cropped; payload cached -> {os.path.basename(cache_path)}", flush=True)

    meta = render_payload(payload, html_path, title, args.wsi, axis_note=axis_note,
                          sources=sources, axes=axes)
    print(f"[patchwork] regional gallery ({meta['n_patches']} regions) -> {html_path}", flush=True)


if __name__ == "__main__":
    main()
