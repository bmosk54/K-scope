"""Render the top-activating tiles for an SAE feature as an image grid.

This is how an SAE feature stops being a number and becomes a claim about morphology.
Everything upstream (differential activation, nulls, probe alignment) tells you a feature
is REAL and UNNAMED; only the pixels tell you what it IS. A pathologist looks at the grid
and says "that's mucinous differentiation" -- that is the ground-truth step, and it is the
one the whole `hypothesis` verb exists to set up.

tile_id indexes the parquet shards in the same deterministic order used at extraction, so
row i of the .npz is row i of the concatenated shards.
"""

from __future__ import annotations

import glob
import io
import os

import numpy as np
import pyarrow.parquet as pq
from PIL import Image, ImageDraw

DATA_DIR = "/home/sagemaker-user/biolayer/data/nct_crc_he/data"


def _shard_index(data_dir: str = DATA_DIR):
    shards = sorted(glob.glob(os.path.join(data_dir, "NCT_CRC_HE_100K-*.parquet")))
    counts = [pq.ParquetFile(s).metadata.num_rows for s in shards]
    starts = np.cumsum([0] + counts[:-1])
    return shards, np.asarray(starts), np.asarray(counts)


def load_tiles(tile_ids, data_dir: str = DATA_DIR) -> list[Image.Image]:
    """Fetch raw tiles by global row index, reading only the shards actually needed."""
    shards, starts, counts = _shard_index(data_dir)
    out: dict[int, Image.Image] = {}
    tile_ids = list(map(int, tile_ids))
    for si, (shard, st, ct) in enumerate(zip(shards, starts, counts)):
        want = [t for t in tile_ids if st <= t < st + ct]
        if not want:
            continue
        col = pq.read_table(shard, columns=["image"]).column("image").to_pylist()
        for t in want:
            out[t] = Image.open(io.BytesIO(col[t - st]["bytes"])).convert("RGB")
    return [out[t] for t in tile_ids]


def grid(
    tile_ids,
    labels=None,
    class_names=None,
    ncol: int = 8,
    pad: int = 3,
    label_h: int = 14,
    data_dir: str = DATA_DIR,
) -> Image.Image:
    """Contact sheet of tiles, optionally captioned with their tissue class."""
    tiles = load_tiles(tile_ids, data_dir)
    n = len(tiles)
    ncol = min(ncol, n)
    nrow = (n + ncol - 1) // ncol
    w = h = tiles[0].size[0]
    cap = label_h if labels is not None else 0
    W = ncol * w + (ncol + 1) * pad
    H = nrow * (h + cap) + (nrow + 1) * pad
    canvas = Image.new("RGB", (W, H), (245, 245, 245))
    d = ImageDraw.Draw(canvas)
    for i, im in enumerate(tiles):
        r, c = divmod(i, ncol)
        x = pad + c * (w + pad)
        y = pad + r * (h + cap + pad)
        canvas.paste(im, (x, y))
        if labels is not None:
            name = class_names[labels[int(tile_ids[i])]] if class_names is not None else str(labels[i])
            d.text((x + 2, y + h + 1), name, fill=(20, 20, 20))
    return canvas


def feature_exemplars(Z: np.ndarray, feature: int, k: int = 24) -> np.ndarray:
    """Global tile_ids of the k tiles that most activate `feature`."""
    return np.argsort(-Z[:, feature])[:k]
