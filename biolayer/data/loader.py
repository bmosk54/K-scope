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


def local_npz_path(model_key, split, artifacts_dir=ARTIFACTS_DIR):
    return os.path.join(artifacts_dir, config.embeddings_key(model_key, split))


def _open(model_key, split, artifacts_dir=ARTIFACTS_DIR):
    """Return (npz_dict, source). Local mirror first, then S3."""
    path = local_npz_path(model_key, split, artifacts_dir)
    if os.path.exists(path):
        return np.load(path, allow_pickle=True), f"local:{path}"
    # Fall back to the shared bucket (needs the S3 role fix — see SETUP.md).
    import io

    from . import s3_utils
    key = config.embeddings_key(model_key, split)
    buf = io.BytesIO()
    s3_utils.s3().download_fileobj(config.BUCKET, key, buf)
    buf.seek(0)
    return np.load(buf, allow_pickle=True), f"s3://{config.BUCKET}/{key}"


def load(model_key="phikon_v2", split="train", artifacts_dir=ARTIFACTS_DIR):
    """Return (feats, labels, class_names, source) — readout global (back-compat)."""
    d, source = _open(model_key, split, artifacts_dir)
    return d["feats"], d["labels"], list(d["class_names"]), source


def available_layers(model_key="phikon_v2", split="train", artifacts_dir=ARTIFACTS_DIR):
    d, _ = _open(model_key, split, artifacts_dir)
    if "layer_names" in d:
        return list(d["layer_names"])
    return ["readout"]  # old single-layer npz


def load_layer(model_key="phikon_v2", split="train", layer="readout",
               space="global", artifacts_dir=ARTIFACTS_DIR):
    """Return (X (N,dim), labels, class_names, source) at one layer + space.

    space: "global" (CLS) | "local" (mean patch). Falls back to the back-compat
    `feats` array for old single-layer npz files (readout/global only).
    """
    d, source = _open(model_key, split, artifacts_dir)
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
