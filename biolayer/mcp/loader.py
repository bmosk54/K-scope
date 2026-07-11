"""Shared embeddings loader for the MCP verbs.

Prefers the local artifacts/ mirror (the transfer channel while the SageMaker role
lacks S3 Get/Put); falls back to the shared bucket via biolayer.data.s3_utils.
Returns the (feats, labels, class_names) triple every verb operates on.
"""
import os

import numpy as np

from .. import config

# Repo-root artifacts dir (…/owkin-hack/artifacts), independent of CWD.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ARTIFACTS_DIR = os.path.join(_REPO_ROOT, "artifacts")


def local_npz_path(model_key, split, artifacts_dir=ARTIFACTS_DIR):
    return os.path.join(artifacts_dir, config.embeddings_key(model_key, split))


def load(model_key="phikon_v2", split="train", artifacts_dir=ARTIFACTS_DIR):
    """Return (feats, labels, class_names). Local mirror first, then S3."""
    path = local_npz_path(model_key, split, artifacts_dir)
    if os.path.exists(path):
        d = np.load(path, allow_pickle=True)
        return d["feats"], d["labels"], list(d["class_names"]), f"local:{path}"

    # Fall back to the shared bucket (needs the S3 role fix — see SETUP.md).
    from ..data import s3_utils
    feats, labels, class_names = s3_utils.load_embeddings(model_key, split)
    key = config.embeddings_key(model_key, split)
    return feats, labels, class_names, f"s3://{config.BUCKET}/{key}"
