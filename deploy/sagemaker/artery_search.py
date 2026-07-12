"""Artery search — certify a vascularization axis, then LOCALIZE vasculature on a WSI.

The interpretability probe for vascularization: rank a slide's tiles/patches along a
CERTIFIED vessel axis so the top of the ranking IS the slide's vasculature. Two steps, in
order, because a confident vessel map on a confounded axis is confidently wrong:

  1. CERTIFY the VASC-vs-STR axis (vessel vs its stromal background) on the BCSS reference:
     the held-out-AUROC + intensity-collinearity gate AND the matched-random null. If the
     axis can't beat random directions at separating held-out vessel tiles from stroma, or
     it just rides RBC/eosin color, we say so and REFUSE to localize (unless --force).
  2. LOCALIZE on the WSI. PRIMARY = rank the 224 px TILES on that axis (its native
     granularity: fit on 224 px tile CLS) -> top/bottom tile montages + tile-density overlay +
     an on-slide color-confound re-check. SECONDARY = the interactive WSI-map gallery (opens in
     tile view) + the 14x14 patch-token patchwork (finer than a vessel, washes out — kept only
     as illustration). deploy/sagemaker/patchwork_view.py, reused.

Prereqs: run `python deploy/sagemaker/bcss_vessel_fit.py` once to build the vasculature
reference; a warm hoptimus-embed endpoint + fresh AWS creds for that build; the WSI's tile
embeddings already in s3://bucketbiolayer/embeddings/wsi/<slide>/.

    python deploy/sagemaker/artery_search.py \
        --wsi s3://bucketbiolayer/wsi/BRACS/BRACS_1003675.svs
    python deploy/sagemaker/artery_search.py --no-localize     # just the certification card
"""
import argparse
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)                                       # bcss_vessel_fit
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))     # repo root (biolayer)

import bcss_vessel_fit as vf                                   # noqa: E402

B, REGION = "bucketbiolayer", "us-west-2"


def tile_localize(wsi, ax, out, k=15, n_corr=60):
    """PRIMARY localization: rank the slide's 224 px TILES on the certified VASC axis — the
    axis's NATIVE granularity (it was fit on 224 px tile CLS, VASC-region vs stroma-region),
    unlike the 14x14 patch tokens which are smaller than a whole vessel. Saves top/bottom-k
    tile montages + a tile-center density overlay, and re-checks the color confound ON THIS
    SLIDE. Reuses the axis from certification (no refit)."""
    import boto3
    import numpy as np
    from PIL import Image, ImageDraw

    from color_inspect import crop_tile, montage, color_stats, STAT_NAMES
    from biolayer.data.wsi_reader import open_wsi

    Image.MAX_IMAGE_PIXELS = None
    s3 = boto3.client("s3", region_name=REGION)
    tmp = tempfile.gettempdir()
    b, key = wsi[5:].split("/", 1)
    stem = os.path.splitext(os.path.basename(wsi))[0]
    gp = os.path.join(tmp, f"{stem}_global.npz")
    if not os.path.exists(gp):
        s3.download_file(B, f"embeddings/wsi/{stem}/global.npz", gp)
    slide = os.path.join(tmp, os.path.basename(key))
    if not os.path.exists(slide):
        print(f"[artery-search] downloading slide {os.path.basename(key)} ...", flush=True)
        s3.download_file(b, key, slide)

    Z = np.load(gp)
    V = np.asarray(Z["vectors"], "float32")
    coords = np.asarray(Z["coords"])
    sc = ((V - ax.scaler_mean) / ax.scaler_scale) @ ax.direction     # certified axis, no refit
    order = np.argsort(-sc)
    reader = open_wsi(slide)
    os.makedirs(out, exist_ok=True)

    top = [crop_tile(reader, *coords[i]) for i in order[:k]]
    bot = [crop_tile(reader, *coords[i]) for i in order[-k:]]
    tp = os.path.join(out, f"{stem}_VASC_top{k}_tiles.png")
    bp = os.path.join(out, f"{stem}_VASC_bottom{k}_tiles.png")
    montage(top, cols=5).save(tp)
    montage(bot, cols=5).save(bp)

    # tile-center density overlay on the slide thumbnail (where the vessels are)
    mpp = reader.mpp or 0.25
    step = int(round(224 * max(0.5 / mpp, 1.0)))
    thumb, ds = reader.thumbnail(1400)
    ov = Image.fromarray(thumb).convert("RGB")
    draw = ImageDraw.Draw(ov, "RGBA")
    for i in order[:max(64, k)]:
        cx, cy = (coords[i][0] + step / 2) / ds, (coords[i][1] + step / 2) / ds
        draw.ellipse([cx - 3, cy - 3, cx + 3, cy + 3], fill=(205, 40, 110, 180))
    dp = os.path.join(out, f"{stem}_VASC_tile_distribution.png")
    ov.save(dp)

    # color-confound check on THIS slide (top+bottom ranked tiles): |r|>=0.5 => stain-driven
    idx = np.r_[order[:n_corr], order[-n_corr:]]
    stats = np.array([color_stats(crop_tile(reader, *coords[i])) for i in idx])
    print(f"\n[artery-search] PRIMARY 224px-TILE ranking ({len(V)} tiles, "
          f"score {sc[order[-1]]:+.2f}..{sc[order[0]]:+.2f}) — axis-native granularity:", flush=True)
    for j, n in enumerate(STAT_NAMES):
        r = float(np.corrcoef(sc[idx], stats[:, j])[0, 1])
        print(f"    corr(VASC score, {n:11s}) = {r:+.3f}"
              + ("  <- stain-driven" if abs(r) >= 0.5 else ""), flush=True)
    print(f"    top-{k} tiles  -> {tp}", flush=True)
    print(f"    bottom-{k}     -> {bp}", flush=True)
    print(f"    distribution  -> {dp}", flush=True)
    return {"top": tp, "bottom": bp, "distribution": dp}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wsi", default="s3://bucketbiolayer/wsi/BRACS/BRACS_1003675.svs",
                    help="breast WSI to localize vasculature on (tiles must be embedded already)")
    ap.add_argument("--pos", default="VASC", help="vessel class in the vasculature reference")
    ap.add_argument("--neg", default="STR",
                    help="hard negative for the certified axis (STR = the vessel's background)")
    ap.add_argument("--n-null", type=int, default=200, help="matched-random directions in the null")
    ap.add_argument("--out", default="/tmp/artery_search", help="output dir for the localization viz")
    ap.add_argument("--k", type=int, default=15, help="top/bottom tiles in the primary montage")
    ap.add_argument("--no-localize", action="store_true", help="only certify the axis; skip the WSI viz")
    ap.add_argument("--force", action="store_true",
                    help="localize even if the axis fails its gate/null (viz is then UNTRUSTED)")
    args, extra = ap.parse_known_args()                        # extra flags pass through to patchwork_view

    # 1) CERTIFY the vessel axis (gate + held-out matched-random null)
    try:
        ax, null, src = vf.certify_vessel_axis(pos=args.pos, neg=args.neg, n_null=args.n_null)
    except Exception as e:
        print(f"cannot certify the vessel axis: {e}\n"
              f"-> build the reference first: python {os.path.join(HERE, 'bcss_vessel_fit.py')}",
              file=sys.stderr)
        return 2
    print(vf.format_card(ax, null, src))
    trusted = ax.certified and null["survives_null"]
    if not trusted and not args.force:
        print("\n[artery-search] REFUSING to localize on an un-trusted axis (pass --force to override). "
              "A confident vessel map on a confounded axis is confidently wrong.", flush=True)
        return 1
    if args.no_localize:
        return 0

    # 2) LOCALIZE — PRIMARY: 224px tile ranking on the SAME certified axis (axis-native scale)
    if not trusted:
        print("\n[artery-search] NOTE: axis is UNTRUSTED (--force) — localization below is not reliable.",
              flush=True)
    try:
        tile_localize(args.wsi, ax, args.out, k=args.k)
    except Exception as e:
        print(f"[artery-search] tile localization failed: {e}", file=sys.stderr)

    # 3) SECONDARY / illustrative: the interactive WSI-map gallery (opens in 224px-tile view) +
    #    the 14x14 patch-token patchwork square (finer scale, washes out — kept as illustration).
    cmd = [sys.executable, os.path.join(HERE, "patchwork_view.py"),
           "--wsi", args.wsi, "--pos", args.pos, "--neg", args.neg,
           "--axis-dataset", vf.DATASET_SLUG, "--out", args.out] + extra
    print(f"\n[artery-search] secondary gallery + patch-token patchwork -> {' '.join(cmd)}", flush=True)
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
