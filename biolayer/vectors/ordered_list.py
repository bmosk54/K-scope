"""Ordered, rerankable vector lists over the WSI tile embeddings.

The embed job emits two lists (deploy/sagemaker/tile_embed_entry.py):

  GLOBAL — one CLS "257th" vector per sensible tile.
           s3://<bucket>/embeddings/lists/global.npz              (self-contained)
  PATCH  — the 256 patch vectors per tile (tile-major, patch-row-major).
           s3://<bucket>/embeddings/lists/patch.manifest.json     (sharded per slide)

An `OrderedVectorList` couples the vectors (N, D) with row-aligned metadata and a mutable
`order` (indices into the rows). `rerank(scores)` permutes ONLY `order` — the vectors and
metadata never move — so one list can carry many orderings from different mech-interp
scoring metrics, and `top(k)` / `ordered()` read them back in the current ranking.

The PATCH list is potentially tens of GB, so it is backed by a lazy, sharded array
(`_ShardedArray`) over the per-slide `.npy` memmaps: fancy-indexing gathers only the rows a
rerank actually touches, and nothing forces the whole list into RAM. Heavy scoring should
run in-region (where the shards live); this class is the interface, not a bulk mover.
"""
import io
import json

import numpy as np

# --------------------------------------------------------------------------- #
# Sharded, memmap-backed array — the PATCH list's storage without materializing
# --------------------------------------------------------------------------- #
class _ShardedArray:
    """Read-only (M, D) view over per-shard 2-D arrays. `offsets` is the cumulative
    row count [0, rows0, rows0+rows1, ...]; row r lives in shard s = searchsorted-1."""

    def __init__(self, arrays, offsets):
        self._a = arrays
        self._off = np.asarray(offsets, dtype=np.int64)
        self.dtype = arrays[0].dtype
        self.shape = (int(self._off[-1]), int(arrays[0].shape[1]))

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, idx):
        idx = np.asarray(idx)
        if idx.ndim == 0:                                   # single row
            s = int(np.searchsorted(self._off, idx, side="right") - 1)
            return self._a[s][int(idx) - int(self._off[s])]
        out = np.empty((len(idx), self.shape[1]), dtype=self.dtype)
        shard = np.searchsorted(self._off, idx, side="right") - 1
        for s in np.unique(shard):
            m = shard == s
            out[m] = self._a[s][idx[m] - self._off[s]]
        return out


# --------------------------------------------------------------------------- #
# The list
# --------------------------------------------------------------------------- #
class OrderedVectorList:
    """Vectors (N, D) + row-aligned metadata + a mutable `order` (an index permutation)."""

    def __init__(self, vectors, meta=None, kind="global", order=None):
        self.vectors = vectors                              # ndarray or _ShardedArray
        self.meta = dict(meta or {})                        # str -> array aligned to rows
        self.kind = kind
        n = len(vectors)
        self.order = np.arange(n, dtype=np.int64) if order is None else np.asarray(order, np.int64)

    def __len__(self):
        return len(self.vectors)

    @property
    def dim(self):
        return self.vectors.shape[1]

    def rerank(self, scores, descending=True):
        """Set `order` to sort rows by `scores` (one per row). Stable. Returns the order."""
        s = np.asarray(scores, dtype=float)
        if len(s) != len(self):
            raise ValueError(f"scores length {len(s)} != rows {len(self)}")
        rank = np.argsort(s, kind="stable")
        self.order = rank[::-1].copy() if descending else rank
        return self.order

    def reset_order(self):
        self.order = np.arange(len(self), dtype=np.int64)
        return self.order

    def top(self, k):
        """(row_indices, vectors) for the current top-k of `order` (gathers lazily)."""
        idx = self.order[:k]
        return idx, self.vectors[idx]

    def ordered(self):
        """All vectors in the current `order` (materializes — avoid on the huge PATCH list)."""
        return self.vectors[self.order]

    def meta_of(self, row_indices):
        """Row-aligned metadata for the given absolute row indices."""
        ri = np.asarray(row_indices)
        return {k: v[ri] for k, v in self.meta.items()}

    def save_order(self, path, **extra):
        """Persist just the current ranking (+ optional scores) — vectors stay put."""
        np.savez(path, order=self.order, kind=self.kind, **extra)


# --------------------------------------------------------------------------- #
# Loaders
# --------------------------------------------------------------------------- #
def _load_npz(source):
    if isinstance(source, (bytes, bytearray)):
        return np.load(io.BytesIO(source), allow_pickle=True)
    return np.load(source, allow_pickle=True)


def load_global(source):
    """GLOBAL list from a global.npz (path, file-like, or raw bytes)."""
    z = _load_npz(source)
    meta = {k: z[k] for k in ("slide", "coords", "keys") if k in z}
    order = z["order"] if "order" in z else None
    return OrderedVectorList(z["vectors"], meta=meta, kind="global", order=order)


def load_patch_manifest(manifest, fetch):
    """PATCH list from a patch.manifest.json (dict or JSON str/bytes).

    `fetch(s3_uri) -> local_path` downloads (or maps) each shard's `.npy` + `.npz`; pass
    a resolver that caches in-region. Shards are opened memmap (mmap_mode='r') so gathering
    a rerank's top-k touches only those rows.
    """
    if isinstance(manifest, (str, bytes, bytearray)):
        manifest = json.loads(manifest)
    arrays, offsets, metas = [], [0], []
    for sh in manifest["shards"]:
        arrays.append(np.load(fetch(sh["vectors"]), mmap_mode="r"))
        metas.append(_load_npz(fetch(sh["meta"])))
        offsets.append(offsets[-1] + int(sh["rows"]))
    meta = {}
    for key in ("tile_index", "patch_no", "patch_row", "patch_col", "tile_x", "tile_y"):
        if all(key in m for m in metas):
            meta[key] = np.concatenate([m[key] for m in metas])
    meta["slide"] = np.concatenate([np.full(int(sh["rows"]), sh["slide"])
                                    for sh in manifest["shards"]])
    return OrderedVectorList(_ShardedArray(arrays, offsets), meta=meta, kind="patch")
