"""Warm inference backend — load weights + tensors ONCE, serve from RAM/GPU.

Inference (live source-intervention) is served from this process. To keep it fast and
avoid re-downloading or re-loading anything on every call:

  - the frozen encoder is a **process-resident singleton** (weights come from the local
    HF cache; when cached we force offline mode so no network call is made at all);
  - the cached population embeddings are **memoized in RAM** by `data.loader`;
  - the live **reference set** (fixed tiles used only to fit the readout probe) is
    fetched once and cached to disk + RAM — the per-call input (the slide being
    certified) is the only thing that varies.

Typical use in the MCP backend:
    serving.warmup()                       # once, at startup — loads the model + reference
    ctx = serving.live_ctx(watch_images=..., watch_labels=...)   # per K-Pro slide
    card = certify_answer(..., live_ctx=ctx)
"""
import os
import pickle

from . import config
from .data import loader

# process-resident caches
_ENCODERS = {}     # model_key -> LiveEncoder (weights resident on GPU)
_REF = {}          # cache-key -> {"images", "labels"}  (reference tiles resident in RAM)

_CACHE_DIR = os.path.join(loader.ARTIFACTS_DIR, "serving_cache")
# default reference classes = the headline TME tissue concepts certify grounds today
# (tumor / immune / stroma). Kept small so the one-time fetch is quick; extend as needed.
DEFAULT_REF_CLASSES = ("TUM", "LYM", "STR", "MUS", "NORM")
_REF_SHUFFLE_BUFFER = 400   # small buffer -> fast one-time streaming fetch


# ---------------------------------------------------------------------------
# Encoder — resident singleton, no re-download
# ---------------------------------------------------------------------------
def _hf_cache_has(hf_id):
    """Is this HF repo already in the local hub cache? (then we can load offline)."""
    hub = os.path.expanduser(os.environ.get("HF_HOME",
                             os.path.join("~", ".cache", "huggingface")))
    hub = os.path.join(os.path.expanduser(hub), "hub") if not hub.endswith("hub") else hub
    folder = "models--" + hf_id.replace("/", "--")
    return os.path.isdir(os.path.join(hub, folder))


def warm_encoder(model_key="phikon_v2"):
    """Return the resident live encoder for `model_key`, loading it AT MOST ONCE.

    Weights load from the local HF cache; if the repo is already cached we load with
    `local_files_only=True` so the construction makes NO network request (no re-download,
    not even a metadata HEAD). Unlike the HF_HUB_OFFLINE env var, this is a per-call flag
    and does not freeze the whole process offline (which would break tile fetching).
    Every subsequent call returns the same GPU-resident model — no reload, no download.
    """
    if model_key in _ENCODERS:
        return _ENCODERS[model_key]
    from .causal import live as _live
    cached = _hf_cache_has(config.MODELS[model_key]["hf_id"])
    # timm's create_model doesn't plumb local_files_only, so only the transformers path
    # gets the offline flag; the singleton still guarantees a single load either way.
    lfo = cached and config.MODELS[model_key]["backend"] == "transformers"
    _ENCODERS[model_key] = _live.make_live_encoder(model_key, local_files_only=lfo)
    return _ENCODERS[model_key]


# ---------------------------------------------------------------------------
# Reference set — fetched once, cached to disk + RAM
# ---------------------------------------------------------------------------
def reference(model_key="phikon_v2", classes=DEFAULT_REF_CLASSES, per_class=24,
             split="train", seed=1):
    """Fixed reference tiles (for fitting the live readout probe), loaded AT MOST ONCE.

    RAM cache -> disk cache -> HF fetch. The reference set is fixed and reusable across
    every inference call; only the per-call input slide varies.
    """
    ds_slug = _dataset_slug_for(model_key)
    ck = (model_key, ds_slug, tuple(classes), per_class, split, seed)
    if ck in _REF:
        return _REF[ck]
    os.makedirs(_CACHE_DIR, exist_ok=True)
    fname = f"ref_{model_key}_{ds_slug}_{'-'.join(classes)}_{per_class}_{split}_{seed}.pkl"
    path = os.path.join(_CACHE_DIR, fname)
    if os.path.exists(path):
        with open(path, "rb") as f:
            data = pickle.load(f)
    else:
        from .data import tiles
        imgs, labs = tiles.fetch(list(classes), per_class=per_class, split=split,
                                 seed=seed, shuffle_buffer=_REF_SHUFFLE_BUFFER)
        data = {"images": imgs, "labels": labs}
        with open(path, "wb") as f:
            pickle.dump(data, f)
    _REF[ck] = data
    return data


def _dataset_slug_for(model_key):
    # tissue substrates use NCT-CRC; cell-type substrate uses HistoPLUS.
    return (config.HISTOPLUS_SLUG if model_key == "h0_mini" else config.DATASET_SLUG)


# ---------------------------------------------------------------------------
# live_ctx — the object certify_answer consumes
# ---------------------------------------------------------------------------
def live_ctx(model_key="phikon_v2", watch_images=None, watch_labels=None,
             ref_classes=DEFAULT_REF_CLASSES, ref_per_class=24, watch_per_class=6,
             n_null=12, split="train"):
    """Assemble a live_ctx backed by the RESIDENT encoder + reference set.

    Production: pass the K-Pro slide's tiles as (watch_images, watch_labels) — those are
    the thing being certified. Demo: omit them and a small fixed watch set is fetched
    (disjoint seed from the reference). Everything else is warm/resident.
    """
    enc = warm_encoder(model_key)
    ref = reference(model_key, ref_classes, ref_per_class, split, seed=1)
    if watch_images is None:
        w = reference(model_key, ref_classes, watch_per_class, split, seed=7)  # disjoint
        watch_images, watch_labels = w["images"], w["labels"]
    return {"images": watch_images, "image_labels": watch_labels,
            "ref_images": ref["images"], "ref_labels": ref["labels"],
            "encoder": enc, "n_null": n_null}


def warmup(model_key="phikon_v2", ref_classes=DEFAULT_REF_CLASSES, split="train"):
    """Prime the backend: load the encoder + reference set + population embeddings so the
    first real inference call is already warm. Call once at server startup."""
    import time
    t0 = time.time()
    warm_encoder(model_key)
    reference(model_key, ref_classes, split=split)
    loader.load(model_key, split)  # memoize the population npz
    return {"model": model_key, "encoder_resident": True,
            "reference_classes": list(ref_classes),
            "offline_load": _hf_cache_has(config.MODELS[model_key]["hf_id"]),
            "warmup_seconds": round(time.time() - t0, 2)}


def status():
    return {"encoders_resident": list(_ENCODERS),
            "reference_sets_cached": len(_REF),
            "embeddings_cached_in_ram": len(loader._NPZ_CACHE),
            "cache_dir": _CACHE_DIR}
