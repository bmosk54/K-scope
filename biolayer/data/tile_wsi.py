"""Tile a WSI into fixed-size tiles at a target magnification, with an OPTIONAL,
decoupled post-tiling filter stage that drops tiles by metrics (e.g. whitespace).

Format-agnostic (svs/tiff) via wsi_reader.open_wsi. The filter stage is deliberately
separable so you can either filter inline OR run it later on an already-tiled slide:

  1. tile_slide(...)       tissue-masked grid -> tile PNGs + manifest.jsonl (+ metrics)
  2. filter_tile_files(...) drop tiles from an existing manifest by metric filters

Adding a new filter = one entry in FILTERS. The manifest records EVERY candidate
tile's metrics (+ a `kept` flag), so filtering is auditable and re-runnable — nothing
is silently discarded.

CLI:
  python -m biolayer.data.tile_wsi slide.svs --out tiles/                 # tile, no filter
  python -m biolayer.data.tile_wsi slide.svs --out tiles/ --filters whitespace,tissue
  python -m biolayer.data.tile_wsi --filter-existing tiles/manifest.jsonl --filters whitespace
  # s3:// input is downloaded to a temp file first (OpenSlide needs a local path)

Container deps: openslide-python + openslide-bin, tifffile, zarr, Pillow, numpy.
"""
import argparse
import json
import os
import tempfile

import numpy as np
from PIL import Image

try:
    from .wsi_reader import open_wsi           # package module
except ImportError:
    from wsi_reader import open_wsi            # standalone (bundled in a container)


# ---------------------------------------------------------------------------
# Post-tiling filters — each maps an RGB tile to a scalar metric; a rule decides
# whether that metric means "drop". Extend by adding one entry here.
# ---------------------------------------------------------------------------
def m_whitespace(rgb: np.ndarray, white_thresh: float = 0.85) -> float:
    """Fraction of near-white pixels (all channels bright) — background/glass."""
    v = rgb.astype("float32") / 255.0
    return float((v > white_thresh).all(axis=-1).mean())


def m_tissue(rgb: np.ndarray) -> float:
    """Mean HSV saturation — low saturation = background, high = stained tissue."""
    mx = rgb.max(-1).astype("float32")
    mn = rgb.min(-1).astype("float32")
    return float(np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0.0).mean())


# name -> (metric_fn, drop_when in {"above","below"}, threshold)
FILTERS = {
    "whitespace": (m_whitespace, "above", 0.70),   # drop if >70% white
    "tissue":     (m_tissue,     "below", 0.05),    # drop if mean saturation <0.05
}


def eval_filters(rgb: np.ndarray, names):
    """Return (keep: bool, metrics: dict). A tile is dropped if ANY filter fires."""
    metrics, keep = {}, True
    for name in names:
        fn, when, thr = FILTERS[name]
        val = fn(rgb)
        metrics[name] = round(val, 4)
        if (when == "above" and val > thr) or (when == "below" and val < thr):
            keep = False
    return keep, metrics


# ---------------------------------------------------------------------------
# Tissue mask (coarse) — prunes the grid before reading full-res tiles
# ---------------------------------------------------------------------------
def tissue_mask(reader, sat_thresh: float = 0.05, max_dim: int = 4096):
    """Binary tissue mask from a BOUNDED thumbnail + its level-0 downsample.

    Uses reader.thumbnail() (never a full-res plane) so slides whose smallest pyramid
    level is still gigapixel don't trip Pillow's decompression-bomb guard.
    """
    thumb, ds = reader.thumbnail(max_dim)
    mx = thumb.max(-1).astype("float32")
    mn = thumb.min(-1).astype("float32")
    sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0.0)
    return sat > sat_thresh, ds


# ---------------------------------------------------------------------------
# Tiling
# ---------------------------------------------------------------------------
def tile_slide(path: str, out_dir: str, tile_px: int = 224, target_mpp: float = 0.5,
               filters=(), assume_mpp: float = None, min_tissue_frac: float = 0.35,
               max_tiles: int = None):
    """Tile a slide into `tile_px` tiles at ~`target_mpp` µm/px.

    Writes kept tiles as PNGs to out_dir and a manifest.jsonl (one row per CANDIDATE
    tile with coords + metrics + `kept`). `filters` = iterable of FILTERS names to
    apply inline (empty = keep all). `max_tiles` stops after that many KEPT tiles
    (for quick trial runs). Returns (n_kept, n_candidates).
    """
    os.makedirs(out_dir, exist_ok=True)
    reader = open_wsi(path)
    try:
        mpp = reader.mpp or assume_mpp
        if mpp is None:
            raise ValueError(f"{path}: no MPP metadata — pass assume_mpp (e.g. 0.5)")
        # Feed tiles at ~target_mpp: read from the finest pyramid level that is still at
        # least the target resolution, then downsample to tile_px so the tile is truly at
        # target_mpp (not the slide's native magnification).
        d = target_mpp / mpp                      # desired downsample from level 0
        if d < 1:                                 # slide coarser than target — can't upsample
            d = 1.0
        level = reader.level_for_downsample(d)
        ds = reader.level_downsamples[level]
        read_px = max(1, int(round(tile_px * d / ds)))   # region size at the read level
        step = int(round(tile_px * d))            # step in level-0 coords (contiguous tiles)
        eff_mpp = round(mpp * d, 4)               # effective tile mpp after resampling
        W, H = reader.dimensions
        mask, mask_ds = tissue_mask(reader)
        mh, mw = mask.shape

        manifest = open(os.path.join(out_dir, "manifest.jsonl"), "w")
        n_kept = n_cand = 0
        done = False
        for y0 in range(0, H - step + 1, step):
            if done:
                break
            for x0 in range(0, W - step + 1, step):
                # coarse tissue gate: does this tile's footprint hit any tissue?
                mx0, my0 = int(x0 / mask_ds), int(y0 / mask_ds)
                mx1, my1 = int((x0 + step) / mask_ds), int((y0 + step) / mask_ds)
                cell = mask[my0:min(my1 + 1, mh), mx0:min(mx1 + 1, mw)]
                if cell.size == 0 or cell.mean() < min_tissue_frac:
                    continue
                n_cand += 1
                tile = reader.read_region((x0, y0), level, (read_px, read_px))
                if tile.size != (tile_px, tile_px):          # resample to target mpp
                    tile = tile.resize((tile_px, tile_px), Image.BILINEAR)
                rgb = np.asarray(tile)
                keep, metrics = eval_filters(rgb, filters)
                fname = f"tile_x{x0}_y{y0}.png"
                row = {"file": fname, "x": x0, "y": y0, "level": level,
                       "mpp": eff_mpp, "metrics": metrics, "kept": keep}
                if keep:
                    tile.save(os.path.join(out_dir, fname))
                    n_kept += 1
                else:
                    row["file"] = None  # dropped: metrics logged, no file written
                manifest.write(json.dumps(row) + "\n")
                if max_tiles and n_kept >= max_tiles:
                    done = True
                    break
        manifest.close()
        print(f"[tile] {path}: {n_kept}/{n_cand} tiles kept "
              f"(read level {level} @ {read_px}px → {tile_px}px, ~{eff_mpp} µm/px)", flush=True)
        return n_kept, n_cand
    finally:
        reader.close()


# ---------------------------------------------------------------------------
# Post-hoc filtering of an already-tiled slide ("remove images from the list")
# ---------------------------------------------------------------------------
def filter_tile_files(manifest_path: str, filters, delete: bool = False):
    """Apply filters to tiles already on disk; rewrite the manifest, optionally
    deleting the dropped tile files. Returns (n_kept, n_dropped)."""
    from PIL import Image

    base = os.path.dirname(manifest_path)
    rows = [json.loads(ln) for ln in open(manifest_path) if ln.strip()]
    kept = dropped = 0
    out = []
    for row in rows:
        if not row.get("file"):
            out.append(row)
            continue
        fpath = os.path.join(base, row["file"])
        if not os.path.exists(fpath):
            out.append(row)
            continue
        rgb = np.asarray(Image.open(fpath).convert("RGB"))
        keep, metrics = eval_filters(rgb, filters)
        row.setdefault("metrics", {}).update(metrics)
        row["kept"] = keep
        if keep:
            kept += 1
        else:
            dropped += 1
            if delete:
                os.remove(fpath)
                row["file"] = None
        out.append(row)
    with open(manifest_path, "w") as f:
        for row in out:
            f.write(json.dumps(row) + "\n")
    print(f"[filter] {manifest_path}: kept {kept}, dropped {dropped}"
          f"{' (files deleted)' if delete else ''}", flush=True)
    return kept, dropped


def _maybe_fetch_s3(path: str) -> str:
    """OpenSlide needs a local file; download an s3:// slide to a temp path."""
    if not path.startswith("s3://"):
        return path
    import boto3
    bucket, key = path[5:].split("/", 1)
    local = os.path.join(tempfile.gettempdir(), os.path.basename(key))
    if not os.path.exists(local):
        boto3.client("s3").download_file(bucket, key, local)
    return local


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("slide", nargs="?", help="local path or s3:// to an .svs/.tiff")
    ap.add_argument("--out", help="output tile dir")
    ap.add_argument("--tile-px", type=int, default=224)
    ap.add_argument("--mpp", type=float, default=0.5, help="target µm/px (~20x)")
    ap.add_argument("--assume-mpp", type=float, help="µm/px if the slide lacks metadata")
    ap.add_argument("--filters", default="", help="comma list, e.g. whitespace,tissue")
    ap.add_argument("--filter-existing", help="manifest.jsonl to re-filter in place")
    ap.add_argument("--delete", action="store_true", help="delete dropped tile files")
    args = ap.parse_args()

    names = [f for f in args.filters.split(",") if f]
    bad = [f for f in names if f not in FILTERS]
    if bad:
        ap.error(f"unknown filters {bad}; available: {list(FILTERS)}")

    if args.filter_existing:
        filter_tile_files(args.filter_existing, names or list(FILTERS), delete=args.delete)
    else:
        if not args.slide or not args.out:
            ap.error("need <slide> and --out (or --filter-existing)")
        tile_slide(_maybe_fetch_s3(args.slide), args.out, tile_px=args.tile_px,
                   target_mpp=args.mpp, filters=names, assume_mpp=args.assume_mpp)


if __name__ == "__main__":
    main()
