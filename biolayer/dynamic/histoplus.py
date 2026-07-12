"""HistoPLUS cell-type label source — extraction contract + status.

HistoPLUS (github.com/owkin/histoplus, HF Owkin-Bioptimus/histoplus) is a CellViT
nuclei segmentation+classification model on H0-mini that assigns each detected
nucleus one of 13 pan-cancer cell types. Because it is *built on H0-mini*, its
labels live natively in a space we already probe — so cell-type concepts plug into
the exact same causal battery as tissue concepts, no VLM/text tower required.

This module does NOT run the model here (HistoPLUS weights + H0-mini + a nucleus
crop pipeline are not in this environment, and H0-mini is approval-gated). It
defines the ONE artifact the certifier needs and how to produce it, so cell-type
concepts light up the moment that artifact exists — same posture as the confound
gate waiting on multi-site data.

Target artifact (drop-in — matches the loader's schema):
    artifacts/embeddings/histoplus_celltype/h0_mini/<split>.npz
      feats        (N, 768)   H0-mini CLS embedding of each nucleus crop
      labels       (N,)       HistoPLUS class index (0..12)
      class_names  (13,)      == config.HISTOPLUS_CLASSES

Extraction recipe (to build that npz):
  1. Run HistoPLUS on tiles -> per-nucleus (bbox, class_id) for the 13 types.
  2. Crop each nucleus (its context patch) from the tile.
  3. Embed each crop with frozen H0-mini (timm, CLS token = 768-d) — reuse
     biolayer.data.models.load_encoder("h0_mini"); take the readout global.
  4. Balance per class (per-class cap, like data.extract), stack, save the npz above.

Once present, biolayer.dynamic.concepts.resolve() flips the 13 cell-type concepts
from NOT_CERTIFIABLE(needs_data) to certifiable automatically — no code change.
"""
import os

from .. import config
from ..data import loader

REPO = "https://github.com/owkin/histoplus"
CLASSES = config.HISTOPLUS_CLASSES


def status(split="train"):
    """Is the HistoPLUS cell-type substrate available to certify against yet?"""
    path = os.path.join(loader.ARTIFACTS_DIR,
                        config.embeddings_key("h0_mini", split, config.HISTOPLUS_SLUG))
    ready = os.path.exists(path)
    return {
        "source": "histoplus_celltype",
        "substrate": "h0_mini",
        "n_classes": len(CLASSES),
        "classes": list(CLASSES),
        "ready": ready,
        "expected_npz": path,
        "model": REPO,
        "note": ("ready — cell-type concepts certifiable" if ready else
                 "needs the H0-mini HistoPLUS embedding npz (see module docstring for "
                 "the extraction recipe); H0-mini is approval-gated"),
    }
