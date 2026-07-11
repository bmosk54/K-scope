"""biolayer — extraction + storage infra for the Owkin Bio-Interp pathology port.

Frozen pathology-FM feature extraction (Phikon-v2, H0-mini) over NCT-CRC-HE,
with embeddings stored in the team's shared S3 bucket as the source of truth
for the downstream causal battery (directions / SAE / certificates).
"""

from . import config  # noqa: F401

__all__ = ["config"]
