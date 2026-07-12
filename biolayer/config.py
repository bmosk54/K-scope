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
# HistoPLUS cell-type substrate (expands the certifiable vocabulary past tissue)
# ---------------------------------------------------------------------------
# Owkin HistoPLUS (github.com/owkin/histoplus, CellViT on H0-mini) detects +
# classifies nuclei into 13 pan-cancer cell types. We treat each cell type as a
# label source on the H0-mini substrate: embed the nucleus crop with H0-mini, take
# HistoPLUS's class as the label, then the same causal battery runs on it. Labels
# live natively in H0-mini's space — no VLM/text tower needed.
#   embeddings/histoplus_celltype/h0_mini/<split>.npz  (feats (N,768), labels, class_names)
HISTOPLUS_SLUG = "histoplus_celltype"
# order = HistoPLUS class index; short codes used as label names in the npz.
HISTOPLUS_CLASSES = [
    "CANCER",   # cancer / malignant epithelium
    "EPI",      # non-cancerous epithelium
    "LYM",      # lymphocyte
    "PLASMA",   # plasmocyte
    "NEU",      # neutrophil
    "EOS",      # eosinophil
    "MAC",      # macrophage
    "FIB",      # fibroblast
    "SMC",      # smooth muscle cell
    "ENDO",     # endothelial cell
    "RBC",      # red blood cell
    "MITOSIS",  # mitotic figure
    "APOP",     # apoptotic body
]

# ---------------------------------------------------------------------------
# Multi-layer, local + global extraction
# ---------------------------------------------------------------------------
# Every encoder is probed at THREE depths, and at each depth we keep BOTH the
# global (CLS) token and a local (mean patch-token) vector:
#   global : the tile-level representation the model reads out from (CLS)
#   local  : the mean of the spatial patch tokens (local morphology / texture)
# The three depths are named positions so the causal card is model-agnostic:
LAYER_NAMES = ("mid_early", "mid", "readout")   # ~L/3, ~2L/3, final
SPACES = ("global", "local")                     # CLS vs mean-patch

# Layer indices are BACKEND-SPECIFIC and stored per model:
#   transformers -> indices into output_hidden_states (0=embeddings .. n_blocks=final)
#   timm         -> block indices for get_intermediate_layers (0 .. n_blocks-1)

# ---------------------------------------------------------------------------
# Model registry — frozen encoders; multi-layer global (CLS) + local (mean patch)
# ---------------------------------------------------------------------------
# backend selects the load path in models.py:
#   "transformers" -> AutoModel(output_hidden_states=True)
#   "timm"         -> timm ViT get_intermediate_layers(return_prefix_tokens=True)
MODELS = {
    "phikon_v2": {
        "hf_id": "owkin/phikon-v2",
        "backend": "transformers",
        "dim": 1024,
        "gated": False,
        "blocks": 24,                 # ViT-L: 24 transformer blocks
        # hidden_states indices: 0=emb .. 24=final. ~L/3, ~2L/3, readout:
        "layers": (8, 16, 24),
    },
    "h_optimus_0": {
        "hf_id": "bioptimus/H-optimus-0",
        "backend": "timm",
        # Flagship H-optimus-0: ViT-giant/14, embedding = 1536-d.
        "dim": 1536,
        "gated": True,                # gated=AUTO -> instant approval on accepting terms
        "blocks": 40,                 # ViT-g: 40 blocks
        # timm block indices (0..39); readout = last block:
        "layers": (13, 27, 39),
        # H-optimus-0 requires these non-default timm construction args:
        "timm_kwargs": {"init_values": 1e-5, "dynamic_img_size": False},
    },
    "h0_mini": {
        "hf_id": "bioptimus/H0-mini",
        "backend": "timm",
        # CLS token is 768-d (the recommended probing feature, matches CytoSyn's
        # H0-mini conditioning space). We store CLS(768) as global and mean-patch
        # (768) as local separately — richer than the old cls_meanpatch concat.
        "dim": 768,
        "gated": True,                # gated + approval-queued; needs `hf auth login`
        "blocks": 12,                 # ViT-b: 12 blocks
        # timm block indices (0..11); readout = last block:
        "layers": (3, 7, 11),
        # H0-mini requires these non-default timm construction args:
        "timm_kwargs": {"mlp_layer": "SwiGLUPacked", "act_layer": "SiLU"},
    },
}


# ---------------------------------------------------------------------------
# Key builders
# ---------------------------------------------------------------------------
# All artifact kinds are namespaced per-dataset/per-model so phikon_v2 and
# h_optimus_0 (and h0_mini) never collide — each gets its own folder:
#   embeddings/<dataset>/<model>/<split>.npz
#   directions/<dataset>/<model>/<name>.npz
#   sae/<dataset>/<model>/<name>.pt
#   certificates/<dataset>/<model>/<split>_<pos>_vs_<neg>.json
def model_prefix(kind: str, model_key: str, dataset_slug: str = DATASET_SLUG) -> str:
    """Per-model folder for an artifact kind, e.g. embeddings/nct_crc_he/phikon_v2."""
    return f"{PREFIX[kind]}/{dataset_slug}/{model_key}"


def embeddings_key(model_key: str, split: str, dataset_slug: str = DATASET_SLUG) -> str:
    """embeddings/<dataset>/<model>/<split>.npz"""
    return f"{model_prefix('embeddings', model_key, dataset_slug)}/{split}.npz"


def certificate_key(model_key: str, split: str, pos: str, neg: str,
                    dataset_slug: str = DATASET_SLUG) -> str:
    """certificates/<dataset>/<model>/<split>_<pos>_vs_<neg>.json"""
    return f"{model_prefix('certificates', model_key, dataset_slug)}/{split}_{pos}_vs_{neg}.json"


def directions_key(model_key: str, name: str, dataset_slug: str = DATASET_SLUG) -> str:
    """directions/<dataset>/<model>/<name>.npz"""
    return f"{model_prefix('directions', model_key, dataset_slug)}/{name}.npz"


def sae_key(model_key: str, name: str, dataset_slug: str = DATASET_SLUG) -> str:
    """sae/<dataset>/<model>/<name>.pt"""
    return f"{model_prefix('sae', model_key, dataset_slug)}/{name}.pt"


def resolve_split(split: str) -> str:
    """Map a friendly split name to the real HF split; pass through if unknown."""
    return SPLITS.get(split, split)
