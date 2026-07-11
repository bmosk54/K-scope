"""Slide-level MIL aggregation by REUSING a pathology-ViT's own final block.

See DESIGN_MIL_AGGREGATOR.md for the rationale, the permutation-invariance
argument, and the caveats (distribution mismatch, must-train, mean-pool control).

    from biolayer.mil import SlideMILAggregator, MeanPoolBaseline, extract_final_block
"""
from .aggregate import (
    MeanPoolBaseline,
    SlideMILAggregator,
    extract_final_block,
    make_synthetic_bags,
)

__all__ = [
    "SlideMILAggregator",
    "MeanPoolBaseline",
    "extract_final_block",
    "make_synthetic_bags",
]
