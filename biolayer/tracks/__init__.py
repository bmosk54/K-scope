"""biolayer.tracks — the separate model pipelines.

Two tracks, different objectives/datasets/depths, sharing only the S3 layout:
    phikon  Phikon-v2  | NCT-CRC-HE tissue classes | TUM vs LYM   (grounded, demo lead)
    h0      H0-mini    | cell-morphology substrate  | TUM vs NORM  (separate objective)

Everything downstream is track-parameterized: `python -m biolayer.data.extract
--track phikon` vs `--track h0` run independent pipelines into per-model folders.
"""
from .base import Objective, Track
from .h0 import H0
from .phikon import PHIKON

TRACKS = {"phikon": PHIKON, "h0": H0}


def get(name: str) -> Track:
    if name not in TRACKS:
        raise KeyError(f"unknown track {name!r}; known: {list(TRACKS)}")
    return TRACKS[name]


__all__ = ["Track", "Objective", "TRACKS", "get", "PHIKON", "H0"]
