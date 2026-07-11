"""Track = one self-contained model pipeline: model + dataset + objective + layers.

Phikon-v2 and H0 are *separate tracks* with different objectives, datasets, depths
and dims. A Track bundles everything extract.py / the causal battery / the MCP verbs
need, so the two never share assumptions beyond the shared S3 layout in config.py.
"""
from dataclasses import dataclass

from .. import config


@dataclass(frozen=True)
class Objective:
    """What a track is trying to certify: a concept pair + an orthogonal distractor."""
    concept: tuple      # (pos, neg) e.g. ("TUM", "LYM")
    distractor: tuple   # (pos, neg) e.g. ("STR", "MUS") — the specificity control
    description: str

    @property
    def pos(self): return self.concept[0]

    @property
    def neg(self): return self.concept[1]


@dataclass(frozen=True)
class Track:
    name: str            # "phikon" | "h0"
    model_key: str       # key into config.MODELS
    dataset_id: str      # HF dataset id
    dataset_slug: str    # slug used in S3/artifact keys
    splits: dict         # friendly -> real HF split name
    class_names: tuple   # label order (ClassLabel index)
    objective: Objective
    image_column: str = config.IMAGE_COLUMN
    label_column: str = config.LABEL_COLUMN

    # ---- model-derived properties (single source of truth = config.MODELS) ----
    @property
    def spec(self):
        return config.MODELS[self.model_key]

    @property
    def layers(self):
        """The 3 backend-specific layer indices probed for this model."""
        return self.spec["layers"]

    @property
    def dim(self):
        return self.spec["dim"]

    @property
    def backend(self):
        return self.spec["backend"]

    def resolve_split(self, split: str) -> str:
        return self.splits.get(split, split)

    def embeddings_key(self, split: str) -> str:
        return config.embeddings_key(self.model_key, split, self.dataset_slug)
