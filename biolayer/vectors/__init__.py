"""biolayer.vectors — the two ordered, rerankable vector lists over WSI tile embeddings.

    GLOBAL list — one CLS ("257th") vector per sensible tile.
    PATCH  list — the 256 patch vectors per tile (tile-major, patch-row-major).

Produced by the SageMaker embed job (deploy/sagemaker/tile_embed_entry.py) and loaded
here as `OrderedVectorList`s that a future mech-interp scoring pass can rerank.
"""
from .ordered_list import OrderedVectorList, load_global, load_patch_manifest
from .rerank import axis_scores, rerank_by_axis, rerank_by_concept
from .pool import (directional_pool, mean_pool, pool_slide, concept_score, concept_profile)

__all__ = ["OrderedVectorList", "load_global", "load_patch_manifest",
           "axis_scores", "rerank_by_axis", "rerank_by_concept",
           "directional_pool", "mean_pool", "pool_slide", "concept_score", "concept_profile"]
