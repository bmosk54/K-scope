"""Extract a viewable thumbnail (and optional native-res crop) from a WSI .svs in S3.

Ready to run the moment the SageMaker execution role is granted s3:GetObject on
s3://bucketbiolayer/wsi/* (see deploy/grant_wsi_access.sh). Until then it exits with
a clear pointer to the permission fix instead of a raw boto3 traceback.

    # smallest pyramid level of the BRACS slide -> PNG (the fast "what is this slide?" view)
    python -m biolayer.data.wsi_thumbnail

    # a specific slide, at a chosen downsample cap
    python -m biolayer.data.wsi_thumbnail s3://bucketbiolayer/wsi/TCGA-BRCA/TCGA-E2-A14P-01Z-00-DX1.663B02FF-C64B-41A6-8685-FD61CD76F9C6.svs

    # a native-resolution crop (level-0 coords): x y w h
    python -m biolayer.data.wsi_thumbnail --region 20000 15000 2048 2048 --region-level 0

Uses tifffile + Pillow only — no OpenSlide/zarr needed for the thumbnail path, so it
runs with the deps already installed. The embedding/tiling pipeline still goes through
wsi_reader.open_wsi / tile_wsi.py (which prefer OpenSlide once it is in the container).
"""
import argparse
import os
import re
import sys

import numpy as np
from PIL import Image

from .. import config

DEFAULT_S3 = f"s3://{config.BUCKET}/wsi/BRACS/BRACS_1003675.svs"
SCRATCH = os.environ.get("BIOLAYER_SCRATCH", "/tmp/biolayer_wsi")

# Aperio packs "|MPP = 0.2517|" and friends into the level-0 ImageDescription.
_APERIO_KV = re.compile(r"\|?\s*([A-Za-z ]+?)\s*=\s*([^|]+)")


def _split_s3(uri: str):
    if not uri.startswith("s3://"):
        raise ValueError(f"expected an s3:// URI, got {uri!r}")
    bucket, _, key = uri[len("s3://"):].partition("/")
    if not key:
        raise ValueError(f"s3 URI has no key: {uri!r}")
    return bucket, key


def _download(uri: str, dest_dir: str) -> str:
    """Pull the slide to a local path (tifffile needs random access to a real file).

    Auth is the SageMaker execution role — no keys. On AccessDenied we surface the
    exact remediation rather than a bare traceback, because that is the expected
    state until the bucket grant lands.
    """
    import boto3
    from botocore.exceptions import ClientError

    bucket, key = _split_s3(uri)
    os.makedirs(dest_dir, exist_ok=True)
    local = os.path.join(dest_dir, os.path.basename(key))
    if os.path.exists(local) and os.path.getsize(local) > 0:
        print(f"[cache] using already-downloaded {local}", file=sys.stderr)
        return local

    client = boto3.client("s3", region_name=config.REGION)
    print(f"[download] {uri} -> {local}", file=sys.stderr)
    try:
        client.download_file(bucket, key, local)
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code", "")
        if code in ("403", "AccessDenied", "Forbidden"):
            sys.exit(
                f"\nAccessDenied reading {uri}.\n"
                "The SageMaker execution role can list this bucket but not GetObject yet.\n"
                "Grant it (as an IAM admin) with deploy/grant_wsi_access.sh, or:\n\n"
                "  aws iam put-role-policy --role-name <execution-role> \\\n"
                "    --policy-name BiolayerWSIRead --policy-document file://deploy/wsi_read_policy.json\n\n"
                "If the bucket is SSE-KMS encrypted, also grant kms:Decrypt on its key.\n"
            )
        raise
    return local


def _parse_mpp(description: str):
    """Microns/pixel from the Aperio ImageDescription, or None."""
    if not description:
        return None
    for k, v in _APERIO_KV.findall(description):
        if k.strip().upper() == "MPP":
            try:
                return float(v.strip())
            except ValueError:
                return None
    return None


def _to_rgb(arr: np.ndarray) -> Image.Image:
    arr = np.asarray(arr)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    return Image.fromarray(arr[..., :3].astype("uint8"), "RGB")


def extract(local_path: str, level: int, max_size: int, out_path: str,
            region=None, region_level: int = 0) -> dict:
    """Read one pyramid level (or a native-res crop) and write a PNG. Returns metadata.

    level=-1 selects the smallest level — the whole-slide overview. `region` is
    (x, y, w, h) in region_level-pixel coords for a full-detail crop instead.
    """
    import tifffile

    with tifffile.TiffFile(local_path) as tif:
        base = tif.series[0]
        # Pyramidal SVS exposes .levels (largest first); guard for flat TIFFs.
        levels = list(getattr(base, "levels", None) or [base])
        level_dims = [(lv.shape[1], lv.shape[0]) for lv in levels]  # (w, h)
        w0, h0 = level_dims[0]
        desc = tif.pages[0].description or ""
        mpp = _parse_mpp(desc)
        # Associated images (thumbnail/label/macro) are extra series in Aperio SVS.
        associated = [s.name for s in tif.series[1:] if s.name]

        if region is not None:
            x, y, w, h = region
            lv = levels[region_level]
            arr = lv.asarray()[y:y + h, x:x + w]
            picked = f"region {w}x{h}@({x},{y}) level={region_level}"
        else:
            li = level if level >= 0 else len(levels) + level
            li = max(0, min(li, len(levels) - 1))
            arr = levels[li].asarray()
            picked = f"level={li} ({level_dims[li][0]}x{level_dims[li][1]})"

        img = _to_rgb(arr)
        if max(img.size) > max_size:
            img.thumbnail((max_size, max_size), Image.LANCZOS)

        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        img.save(out_path)

    return {
        "source": local_path,
        "level0_dimensions": [w0, h0],
        "level_count": len(levels),
        "level_dimensions": level_dims,
        "mpp_um_per_px": mpp,
        "magnification": (round(10.0 / mpp) if mpp else None),  # ~40x at 0.25, ~20x at 0.5
        "associated_images": associated,
        "extracted": picked,
        "output_png": out_path,
        "output_size": list(img.size),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("slide", nargs="?", default=DEFAULT_S3,
                    help=f"s3:// URI or local path to an .svs (default: {DEFAULT_S3})")
    ap.add_argument("--level", type=int, default=-1,
                    help="pyramid level; -1 = smallest/overview (default), 0 = full-res")
    ap.add_argument("--max-size", type=int, default=2048,
                    help="cap the PNG's longest side (default 2048)")
    ap.add_argument("--region", type=int, nargs=4, metavar=("X", "Y", "W", "H"),
                    help="native-res crop in region-level pixels, instead of a whole level")
    ap.add_argument("--region-level", type=int, default=0,
                    help="pyramid level the --region coords refer to (default 0)")
    ap.add_argument("--out", default=None,
                    help="output PNG path (default: <scratch>/<slide>__<what>.png)")
    ap.add_argument("--scratch", default=SCRATCH, help=f"download/output dir (default {SCRATCH})")
    ap.add_argument("--keep", action="store_true",
                    help="keep the downloaded .svs (default: it stays cached in --scratch)")
    args = ap.parse_args(argv)

    local = args.slide if not args.slide.startswith("s3://") else _download(args.slide, args.scratch)

    stem = os.path.splitext(os.path.basename(local))[0]
    what = ("region" if args.region else f"L{args.level}")
    out = args.out or os.path.join(args.scratch, f"{stem}__{what}.png")

    meta = extract(local, args.level, args.max_size, out,
                   region=tuple(args.region) if args.region else None,
                   region_level=args.region_level)

    import json
    print(json.dumps(meta, indent=2))
    print(f"\nPNG ready: {meta['output_png']}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
