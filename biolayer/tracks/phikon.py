"""Phikon-v2 track — the grounded, load-bearing pipeline (STRATEGY.md scope).

Objective: the tumor-immune interface (TUM vs LYM) on NCT-CRC-HE tissue classes,
with STR/MUS as the orthogonal specificity distractor. This is the proven substrate
(RESULTS.md: probe acc 1.000, steering flip 1.000 vs 0.000 random) and the demo lead.
"""
from .. import config
from .base import Objective, Track

PHIKON = Track(
    name="phikon",
    model_key="phikon_v2",
    dataset_id=config.DATASET_ID,        # 1aurent/NCT-CRC-HE
    dataset_slug=config.DATASET_SLUG,    # nct_crc_he
    splits=config.SPLITS,
    class_names=tuple(config.CLASS_NAMES),
    objective=Objective(
        concept=("TUM", "LYM"),
        distractor=("STR", "MUS"),
        description="tumor epithelium vs immune infiltrate — the tumor-immune interface",
    ),
)
