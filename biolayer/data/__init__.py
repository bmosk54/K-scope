"""biolayer.data — frozen pathology-FM feature extraction + artifact storage.

The data/infra layer: load a frozen encoder (Phikon-v2, H-optimus-0, H0-mini),
embed an NCT-CRC-HE subset to CLS features, and push/pull the resulting .npz
through the team's shared S3 bucket. Everything downstream (biolayer.causal,
biolayer.mcp) consumes the embeddings this layer produces.
"""

from . import s3_utils  # noqa: F401
from .models import DEVICE, load_encoder  # noqa: F401

__all__ = ["s3_utils", "load_encoder", "DEVICE"]
