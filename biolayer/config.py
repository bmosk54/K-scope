"""Central config: bucket, region, S3 prefix layout, model registry, dataset id.

Single source of truth so extract.py and downstream causal-battery code agree
on where artifacts live and how each encoder is loaded.
"""

# ---------------------------------------------------------------------------
# S3 — team's shared source of truth
# ---------------------------------------------------------------------------
BUCKET = "bucketbiolayer"
REGION = "us-west-2"

# Prefix layout under s3://bucketbiolayer/
#   embeddings/<dataset>/<model>/<split>.npz   frozen CLS features + labels
#   directions/                                 concept directions (TCAV / diff-of-means)
#   sae/                                         sparse autoencoders on the residual stream
#   certificates/                               per-prediction causal evidence cards
PREFIX = {
    "embeddings": "embeddings",
    "directions": "directions",
    "sae": "sae",
    "certificates": "certificates",
}

# ---------------------------------------------------------------------------
# Dataset — pre-tiled 224x224 H&E, native tissue-class labels (no WSI/tiling)
# ---------------------------------------------------------------------------
DATASET_ID = "1aurent/NCT-CRC-HE"
DATASET_SLUG = "nct_crc_he"  # used in S3 keys

# The HF repo exposes one config with three splits. We map friendly names to
# the real HF split names so the CLI reads `--split train`.
SPLITS = {
    "train": "NCT_CRC_HE_100K",            # 100k, Macenko-normalized
    "train_nonorm": "NCT_CRC_HE_100K_NONORM",  # 100k, un-normalized
    "val": "CRC_VAL_HE_7K",                # 7180, held-out validation
    "test": "CRC_VAL_HE_7K",               # alias
}

# 9 native tissue classes (order == ClassLabel index in the HF dataset)
CLASS_NAMES = ["ADI", "BACK", "DEB", "LYM", "MUC", "MUS", "NORM", "STR", "TUM"]

IMAGE_COLUMN = "image"
LABEL_COLUMN = "label"

# ---------------------------------------------------------------------------
# Model registry — frozen encoders, CLS-pooled
# ---------------------------------------------------------------------------
# backend selects the load path in models.py:
#   "transformers" -> AutoModel, CLS = last_hidden_state[:, 0, :]
#   "timm"         -> timm.create_model, CLS (+ mean patch) pooling
MODELS = {
    "phikon_v2": {
        "hf_id": "owkin/phikon-v2",
        "backend": "transformers",
        "dim": 1024,
        "gated": False,
        "pool": "cls",
    },
    "h_optimus_0": {
        "hf_id": "bioptimus/H-optimus-0",
        "backend": "timm",
        # Flagship H-optimus-0: ViT-giant/14, CLS embedding = 1536-d.
        # model(x) returns the pooled (B, 1536) CLS directly.
        "dim": 1536,
        "gated": True,        # gated=AUTO -> instant approval on accepting terms
        "pool": "cls",
        # H-optimus-0 requires these non-default timm construction args:
        "timm_kwargs": {"init_values": 1e-5, "dynamic_img_size": False},
    },
    "h0_mini": {
        "hf_id": "bioptimus/H0-mini",
        "backend": "timm",
        # CLS token is 768-d (the recommended probing feature). 1536 is ONLY if
        # you concatenate CLS + mean-patch; we default to CLS to match CytoSyn's
        # H0-mini conditioning space. Keep 768 consistent across probe/SAE/index.
        "dim": 768,
        "gated": True,        # gated + approval-queued; needs `hf auth login`
        "pool": "cls",        # "cls" (768) | "cls_meanpatch" (1536)
        # H0-mini requires these non-default timm construction args:
        "timm_kwargs": {"mlp_layer": "SwiGLUPacked", "act_layer": "SiLU"},
    },
}


# ---------------------------------------------------------------------------
# Key builders
# ---------------------------------------------------------------------------
def embeddings_key(model_key: str, split: str, dataset_slug: str = DATASET_SLUG) -> str:
    """S3 key for an embeddings .npz, e.g. embeddings/nct_crc_he/phikon_v2/train.npz"""
    return f"{PREFIX['embeddings']}/{dataset_slug}/{model_key}/{split}.npz"


def resolve_split(split: str) -> str:
    """Map a friendly split name to the real HF split; pass through if unknown."""
    return SPLITS.get(split, split)
