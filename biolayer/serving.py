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
            "slide_models_precomputed": len(_SLIDE),
            "embeddings_cached_in_ram": len(loader._NPZ_CACHE),
            "cache_dir": _CACHE_DIR}


# ===========================================================================
# Per-slide inference — ONE slide in (its tiles), readout distribution + per-slide
# necessity out. The readout probe and the per-class concept axes are PRECOMPUTED
# ONCE from a labeled reference (so the input slide needs no labels), then only the
# slide's tiles are forward-passed at inference.
# ===========================================================================
_SLIDE = {}     # (model_key, classes, per_class, split) -> precomputed readout+axes


def precompute_slide(model_key="phikon_v2", classes=DEFAULT_REF_CLASSES, per_class=40,
                     split="train", ref=None):
    """Fit — ONCE — the multiclass readout probe + per-class per-layer concept axes from
    a labeled reference (live-embedded so the representation matches the forward pass).
    Cached; the slide being certified never touches this."""
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from .causal import probe as _probe

    key = (model_key, tuple(classes), per_class, split)
    if key in _SLIDE:
        return _SLIDE[key]
    enc = warm_encoder(model_key)
    ref = ref or reference(model_key, classes, per_class, split, seed=1)
    imgs, labs = ref["images"], np.asarray(ref["labels"])
    readout, hidden = enc.hidden_cls(imgs)                 # (N,dim), (N,L+1,dim)

    scaler = StandardScaler().fit(readout)
    clf = LogisticRegression(max_iter=3000).fit(scaler.transform(readout), labs)  # multinomial
    cn = config.CLASS_NAMES
    layer_idx = list(config.MODELS[model_key]["layers"])
    axes = {}                                               # class-name -> {layer: unit dir}
    for c in classes:
        ci = cn.index(c)
        y = (labs == ci).astype(int)                       # class-vs-rest
        if y.sum() < 2 or (1 - y).sum() < 2:
            continue
        axes[c] = {L: _probe.diff_of_means(hidden[:, L, :], y) for L in layer_idx}
    _SLIDE[key] = {"scaler": scaler, "clf": clf, "class_names": cn,
                   "cols": {cn[c]: j for j, c in enumerate(clf.classes_)},
                   "axes": axes, "layers": layer_idx}
    return _SLIDE[key]


def slide_readout_distribution(pre, readout):
    """Mean softmax over the slide's tiles -> {class: probability}."""
    P = pre["clf"].predict_proba(pre["scaler"].transform(readout)).mean(0)
    return {cn: float(P[j]) for cn, j in pre["cols"].items()}


def certify_slide(model_key="phikon_v2", slide_images=None, concepts=None, n_null=10,
                  seed=0, pre=None):
    """ONE slide in (its H&E tiles) -> readout distribution + per-slide necessity.

    Embeds the slide's tiles once, reports the readout class distribution, then for each
    concept edits THIS slide's real forward pass (project the precomputed class axis out
    of the CLS in the residual stream, propagate) and measures the drop in the slide's
    readout probability for that class vs a matched-random null. intervened_on_input=True.
    The slide carries NO labels — axis + probe come from the precomputed reference.
    """
    import numpy as np
    from .causal import live as _live
    from .causal import probe as _probe

    pre = pre or precompute_slide(model_key)
    enc = warm_encoder(model_key)
    clean_readout, _ = enc.hidden_cls(slide_images)

    def meanP(readout):
        return pre["clf"].predict_proba(pre["scaler"].transform(readout)).mean(0)
    P0 = meanP(clean_readout)
    dist = {cn: float(P0[j]) for cn, j in pre["cols"].items()}

    concepts = concepts or list(pre["axes"])
    per_concept = {}
    for c in concepts:
        if c not in pre["axes"] or c not in pre["cols"]:
            continue
        j = pre["cols"][c]
        base = float(P0[j])
        curve = []
        for L in pre["layers"]:
            axis = pre["axes"][c][L]
            block = L - 1
            abl = float(meanP(enc.embed(slide_images, edit=_live.project_out(axis),
                                        block_idx=block))[j])
            R = _probe.matched_random_dirs(axis.shape[0], n_null, seed=seed)
            null = np.array([float(meanP(enc.embed(slide_images,
                             edit=_live.project_out(r), block_idx=block))[j]) for r in R])
            concept_drop = base - abl
            null_drops = base - null
            nm, ns = float(null_drops.mean()), float(null_drops.std())
            gap = concept_drop - nm
            curve.append({"block": block, "base_P": round(base, 3),
                          "concept_ablated_P": round(abl, 3),
                          "random_ablated_P_mean": round(float(null.mean()), 3),
                          "necessity_gap": round(gap, 4),
                          "z": round(gap / (ns + 1e-9), 2),
                          "bites": bool(gap > 0 and gap / (ns + 1e-9) >= 1.645)})
        per_concept[c] = {"readout_prob": base, "curve": curve}

    return {"model": model_key, "n_tiles": len(slide_images),
            "intervened_on_input": True,
            "readout_distribution": dist,
            "per_concept_necessity": per_concept,
            "note": "one slide in; readout distribution + per-slide necessity by editing "
                    "THIS slide's real forward pass (axis + probe precomputed from a "
                    "labeled reference; the slide carries no labels)"}
