"""biolayer — Bio-Interp causal battery ported to pathology FMs.

Three layers, one per subpackage:
    biolayer.data     frozen encoder extraction + S3 artifact storage
    biolayer.causal   the causal battery (probe, necessity, sufficiency,
                      specificity, + the intervene/confound pillars)
    biolayer.mcp      the MCP surface: certify(prediction) -> evidence card

config.py stays at the top level as the single source of truth (bucket, model
registry, dataset, S3 key layout) that all three layers import.
"""

from . import config  # noqa: F401

__all__ = ["config"]
