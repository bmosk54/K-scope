"""H0 track — the second, DELIBERATELY SEPARATE pipeline.

Different model, different objective, and (intended) different dataset from the
Phikon track. H0-mini's 768-d CLS is the CytoSyn conditioning space, so this track
is aimed at a cell-morphology / malignancy objective rather than Phikon's
tissue-class tumor-immune interface.

Two things the H0-track owner sets (kept runnable-today by default, marked TODO):
  - MODEL: uses `h_optimus_0` (gated=AUTO, cached, extracted). `h0_mini` — the
    CytoSyn-aligned target — is approval-gated and NOT accessible here, so the tissue
    track runs on h_optimus_0; both are timm ViTs and share the live-intervention path
    (`TimmLiveEncoder`), so h0_mini drops in unchanged once its weights + npz land.
  - DATASET: defaults to NCT-CRC-HE so the track runs today, but the intended
    divergence is a cell-type substrate (HistoPLUS 13 cell types / CytoSyn
    counterfactuals). Swap dataset_id/slug/class_names/objective when that lands.
"""
from .. import config
from .base import Objective, Track

# h_optimus_0 (gated=AUTO, cached) — h0_mini is approval-gated / not accessible here.
# Restore "h0_mini" once its weights + a multi-layer npz are available.
H0_MODEL_KEY = "h_optimus_0"

H0 = Track(
    name="h0",
    model_key=H0_MODEL_KEY,
    # TODO(h0 owner): swap to the cell-type substrate (HistoPLUS / CytoSyn) — the
    # real objective for this track. NCT-CRC keeps it runnable in the meantime.
    dataset_id=config.DATASET_ID,
    dataset_slug=config.DATASET_SLUG,
    splits=config.SPLITS,
    class_names=tuple(config.CLASS_NAMES),
    objective=Objective(
        concept=("TUM", "NORM"),
        distractor=("MUS", "ADI"),
        description="malignant epithelium vs normal mucosa — malignancy detection "
                    "(placeholder for the H0 cell-morphology objective)",
    ),
)
