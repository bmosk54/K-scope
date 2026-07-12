"""Concept registry — the extensible label sources certification resolves against.

A *label source* is a labeled corpus living on a substrate: it pins a dataset slug,
a class list, and which encoder substrates have embeddings for it. A *concept* names
a (pos, neg, distractor) contrast within one source's classes plus the free-text
keywords K-Pro might use for it.

Two sources today:
  - nct_crc_tissue : NCT-CRC-HE 9 tissue classes on Phikon-v2 / H-optimus-0  [READY]
  - histoplus_celltype : HistoPLUS 13 TME cell types on H0-mini             [needs data]

This is the ONLY place a taxonomy leaks in, and it is explicit + auditable. Abstract
clinical claims (grade, MSI, "biomarker for X") deliberately have no entry -> they
resolve to NOT_CERTIFIABLE. Adding a new labeled corpus = one LabelSource + its
ConceptSpecs; nothing downstream changes.
"""
import os
from dataclasses import dataclass

from .. import config
from ..data import loader


@dataclass(frozen=True)
class LabelSource:
    name: str
    dataset_slug: str
    classes: tuple          # class-name order in the npz
    substrates: tuple       # model_keys that have embeddings for this dataset
    description: str
    ready: bool = True       # False = corpus not extractable/available yet


@dataclass(frozen=True)
class ConceptSpec:
    concept: str
    source: str             # LabelSource.name
    pos: str
    neg: str
    distractor: tuple
    keywords: tuple
    description: str


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------
TISSUE = LabelSource(
    "nct_crc_tissue", config.DATASET_SLUG, tuple(config.CLASS_NAMES),
    ("phikon_v2", "h_optimus_0"),
    "NCT-CRC-HE 9 tissue classes (tile-level)")

CELL = LabelSource(
    "histoplus_celltype", config.HISTOPLUS_SLUG, tuple(config.HISTOPLUS_CLASSES),
    ("h0_mini",),
    "HistoPLUS 13 TME cell types (nucleus-level, CellViT on H0-mini)")

SOURCES = {s.name: s for s in (TISSUE, CELL)}


# ---------------------------------------------------------------------------
# Concepts
# ---------------------------------------------------------------------------
_TISSUE_CONCEPTS = (
    ConceptSpec("tumor_epithelium", TISSUE.name, "TUM", "NORM", ("STR", "MUS"),
                ("tumor", "tumour", "carcinoma", "malignan", "adenocarcinoma", "neoplas"),
                "malignant epithelium vs normal mucosa"),
    ConceptSpec("immune_infiltrate", TISSUE.name, "LYM", "TUM", ("STR", "MUS"),
                ("lymphocytic", "immune", "infiltrat", "til", "inflammat"),
                "lymphocytic/immune infiltrate vs tumor epithelium (tile-level)"),
    ConceptSpec("stroma", TISSUE.name, "STR", "MUS", ("TUM", "LYM"),
                ("stroma", "desmoplas", "fibro"),
                "cancer-associated stroma vs muscle"),
    ConceptSpec("mucus", TISSUE.name, "MUC", "NORM", ("STR", "MUS"),
                ("mucin", "mucus", "mucous"), "mucus vs normal mucosa"),
    ConceptSpec("muscle", TISSUE.name, "MUS", "STR", ("TUM", "LYM"),
                ("muscularis", "smooth muscle"), "smooth muscle vs stroma"),
    ConceptSpec("adipose", TISSUE.name, "ADI", "NORM", ("STR", "MUS"),
                ("adipos", "fat tissue", "fatty"), "adipose vs normal mucosa"),
    ConceptSpec("debris_necrosis", TISSUE.name, "DEB", "NORM", ("STR", "MUS"),
                ("necros", "debris"), "debris/necrosis vs normal mucosa"),
    ConceptSpec("normal_mucosa", TISSUE.name, "NORM", "TUM", ("STR", "MUS"),
                ("normal mucosa", "benign", "non-neoplastic", "healthy"),
                "normal mucosa vs tumor"),
)

# Cell-level concepts on the HistoPLUS substrate. These are the expressiveness win:
# the "understudied" TME cell types K-Pro talks about (plasmocyte, eosinophil,
# neutrophil, macrophage, endothelial, mitotic/apoptotic) that tissue labels can't
# reach. Resolve only when H0-mini HistoPLUS embeddings exist (else needs_data).
_CELL_CONCEPTS = (
    ConceptSpec("cancer_cell", CELL.name, "CANCER", "EPI", ("FIB", "SMC"),
                ("cancer cell", "malignant cell", "tumor cell", "tumour cell"),
                "cancer nucleus vs benign epithelial nucleus"),
    ConceptSpec("lymphocyte_cell", CELL.name, "LYM", "CANCER", ("FIB", "ENDO"),
                ("lymphocyte", "til cell"), "lymphocyte nucleus vs cancer nucleus"),
    ConceptSpec("plasma_cell", CELL.name, "PLASMA", "LYM", ("FIB", "SMC"),
                ("plasma cell", "plasmocyte"), "plasmocyte vs lymphocyte"),
    ConceptSpec("neutrophil", CELL.name, "NEU", "LYM", ("FIB", "ENDO"),
                ("neutrophil",), "neutrophil vs lymphocyte"),
    ConceptSpec("eosinophil", CELL.name, "EOS", "NEU", ("FIB", "SMC"),
                ("eosinophil",), "eosinophil vs neutrophil"),
    ConceptSpec("macrophage", CELL.name, "MAC", "LYM", ("FIB", "ENDO"),
                ("macrophage", "histiocyte"), "macrophage vs lymphocyte"),
    ConceptSpec("fibroblast_cell", CELL.name, "FIB", "SMC", ("CANCER", "LYM"),
                ("fibroblast",), "fibroblast vs smooth muscle cell"),
    ConceptSpec("smooth_muscle_cell", CELL.name, "SMC", "FIB", ("CANCER", "LYM"),
                ("smooth muscle cell",), "smooth muscle cell vs fibroblast"),
    ConceptSpec("endothelial", CELL.name, "ENDO", "FIB", ("CANCER", "LYM"),
                ("endothelial", "vascular lining"), "endothelial cell vs fibroblast"),
    ConceptSpec("red_blood_cell", CELL.name, "RBC", "ENDO", ("FIB", "SMC"),
                ("red blood cell", "erythrocyte", "rbc"), "RBC vs endothelial"),
    ConceptSpec("mitotic_figure", CELL.name, "MITOSIS", "CANCER", ("FIB", "LYM"),
                ("mitotic", "mitosis"), "mitotic figure vs cancer nucleus"),
    ConceptSpec("apoptotic_body", CELL.name, "APOP", "LYM", ("FIB", "SMC"),
                ("apoptotic", "apoptosis"), "apoptotic body vs lymphocyte"),
)

CONCEPTS = _TISSUE_CONCEPTS + _CELL_CONCEPTS
_BY_NAME = {c.concept: c for c in CONCEPTS}


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------
def _substrate_for(source, preferred_model_key):
    """Pick the substrate to certify this source on: prefer the caller's, else the
    source's first (only) substrate."""
    if preferred_model_key in source.substrates:
        return preferred_model_key
    return source.substrates[0]


def _embeddings_exist(model_key, dataset_slug, split):
    path = os.path.join(loader.ARTIFACTS_DIR,
                        config.embeddings_key(model_key, split, dataset_slug))
    return os.path.exists(path)


@dataclass
class Resolution:
    spec: ConceptSpec = None
    model_key: str = None
    dataset_slug: str = None
    status: str = "certifiable"   # certifiable | not_certifiable
    reason: str = ""


def match(text):
    """Concepts whose keywords hit `text`, best-first (most hits, then longest kw)."""
    low = text.lower()
    scored = []
    for c in CONCEPTS:
        hits = sum(k in low for k in c.keywords)
        if hits:
            scored.append((hits, max(len(k) for k in c.keywords if k in low), c))
    scored.sort(key=lambda s: (s[0], s[1]), reverse=True)
    return [c for _, _, c in scored]


def resolve(concept_name, preferred_model_key, split="train"):
    """Resolve one concept name to (spec, substrate, dataset) + a certifiability read.

    A concept is certifiable only if its label source is ready AND embeddings exist
    for the chosen substrate — otherwise NOT_CERTIFIABLE with an honest reason
    (needs_data), never a force-fit.
    """
    spec = _BY_NAME.get(concept_name)
    if spec is None:
        return Resolution(status="not_certifiable", reason="no substrate-labeled concept")
    src = SOURCES[spec.source]
    model_key = _substrate_for(src, preferred_model_key)
    if not src.ready:
        return Resolution(spec=spec, model_key=model_key, dataset_slug=src.dataset_slug,
                          status="not_certifiable",
                          reason=f"label source {src.name!r} not available yet")
    if not _embeddings_exist(model_key, src.dataset_slug, split):
        return Resolution(spec=spec, model_key=model_key, dataset_slug=src.dataset_slug,
                          status="not_certifiable",
                          reason=(f"needs {src.name} embeddings on {model_key} "
                                  f"({src.dataset_slug}/{model_key}/{split}.npz absent)"))
    return Resolution(spec=spec, model_key=model_key, dataset_slug=src.dataset_slug)


def resolvable_concepts(preferred_model_key, split="train"):
    """Concept names that certify TODAY on the given substrate (for LLM vocab)."""
    return [c.concept for c in CONCEPTS
            if resolve(c.concept, preferred_model_key, split).status == "certifiable"]


def coverage_summary(preferred_model_key, split="train"):
    ready = resolvable_concepts(preferred_model_key, split)
    return {"total_concepts": len(CONCEPTS),
            "certifiable_now": len(ready),
            "by_source": {s.name: {"classes": len(s.classes), "ready": s.ready,
                                   "substrates": list(s.substrates)}
                          for s in SOURCES.values()}}
