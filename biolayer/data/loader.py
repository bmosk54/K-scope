"""Shared embeddings loader for the causal battery and the MCP verbs.

Prefers the local artifacts/ mirror; falls back to the shared bucket. Understands
the multi-layer, local+global .npz written by biolayer.data.extract:

    load(model, split)                        -> readout global feats (back-compat)
    load_layer(model, split, layer, space)    -> (N, dim) at one layer/space
    available_layers(model, split)            -> which layer names are present
"""
import os

import numpy as np

from .. import config

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ARTIFACTS_DIR = os.path.join(_REPO_ROOT, "artifacts")


def local_npz_path(model_key, split, artifacts_dir=ARTIFACTS_DIR, dataset_slug=None):
    slug = dataset_slug or config.DATASET_SLUG
    return os.path.join(artifacts_dir, config.embeddings_key(model_key, split, slug))


# In-memory cache of MATERIALIZED npz arrays, so a warm inference backend reads each
# embeddings file (from disk or S3) exactly ONCE and every later call reuses the arrays
# in RAM — no repeated disk reads, no repeated S3 downloads. Keyed by (model, split,
# slug, artifacts_dir). Call clear_cache() to drop it.
_NPZ_CACHE = {}


def _open(model_key, split, artifacts_dir=ARTIFACTS_DIR, dataset_slug=None):
    """Return (npz_dict, source). In-memory cache first, then local mirror, then S3."""
    slug = dataset_slug or config.DATASET_SLUG
    ck = (model_key, split, slug, artifacts_dir)
    if ck in _NPZ_CACHE:
        return _NPZ_CACHE[ck]
    path = local_npz_path(model_key, split, artifacts_dir, slug)
    if os.path.exists(path):
        d = np.load(path, allow_pickle=True)
        materialized = {k: d[k] for k in d.files}      # pull into RAM, drop the file handle
        result = (materialized, f"local:{path}")
    else:
        # Fall back to the shared bucket (needs the S3 role fix — see SETUP.md).
        import io

        from . import s3_utils
        key = config.embeddings_key(model_key, split, slug)
        buf = io.BytesIO()
        s3_utils.s3().download_fileobj(config.BUCKET, key, buf)
        buf.seek(0)
        d = np.load(buf, allow_pickle=True)
        result = ({k: d[k] for k in d.files}, f"s3://{config.BUCKET}/{key}")
    _NPZ_CACHE[ck] = result
    return result


def clear_cache():
    """Drop the in-memory embeddings cache (e.g. after re-extracting a split)."""
    _NPZ_CACHE.clear()


def load(model_key="phikon_v2", split="train", artifacts_dir=ARTIFACTS_DIR,
         dataset_slug=None):
    """Return (feats, labels, class_names, source) — readout global (back-compat)."""
    d, source = _open(model_key, split, artifacts_dir, dataset_slug)
    return d["feats"], d["labels"], list(d["class_names"]), source


def available_layers(model_key="phikon_v2", split="train", artifacts_dir=ARTIFACTS_DIR,
                     dataset_slug=None):
    d, _ = _open(model_key, split, artifacts_dir, dataset_slug)
    if "layer_names" in d:
        return list(d["layer_names"])
    return ["readout"]  # old single-layer npz


def load_layer(model_key="phikon_v2", split="train", layer="readout",
               space="global", artifacts_dir=ARTIFACTS_DIR, dataset_slug=None):
    """Return (X (N,dim), labels, class_names, source) at one layer + space.

    space: "global" (CLS) | "local" (mean patch). Falls back to the back-compat
    `feats` array for old single-layer npz files (readout/global only).
    """
    d, source = _open(model_key, split, artifacts_dir, dataset_slug)
    labels, class_names = d["labels"], list(d["class_names"])

    key = {"global": "globals", "local": "locals"}[space]
    if key not in d:  # old-format npz: only readout global exists
        if layer == "readout" and space == "global":
            return d["feats"], labels, class_names, source
        raise KeyError(
            f"{source} is an old single-layer npz — only (readout, global) available; "
            f"re-run `python -m biolayer.data.extract` for multi-layer local+global.")

    names = list(d["layer_names"])
    if layer not in names:
        raise KeyError(f"layer {layer!r} not in {names}")
    li = names.index(layer)
    return d[key][:, li, :], labels, class_names, source
