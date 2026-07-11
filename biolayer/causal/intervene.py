"""Layer-resolved necessity — the *rigor* pillar, over the 3 extracted layers.

Two variants, honest about what each certifies:

1. layered_curve()  [IMPLEMENTED]  — cheap, cached, no forward pass.
   Uses the multi-layer local+global embeddings from biolayer.data.extract. At
   each of the 3 depths it fits the concept probe and runs the readout-space
   necessity check (project the concept axis out) vs a matched-random null, on
   either the global (CLS) or local (mean-patch) space. This shows HOW the
   concept's linear presence + readout-necessity evolve with depth, and lets us
   contrast global vs local morphology signal — a real layer-resolved rigor curve.

2. necessity_curve()  [TODO — track #3, live hooks] — the true source-intervention.
   Hook encoder.layer[L], EDIT the activations in place, and let the edit PROPAGATE
   to the readout. Only this tests the Hydra/redundancy claim ("ablate mid-layer ->
   model recomputes downstream"), which cached embeddings cannot: propagation needs
   the forward pass with the edit applied. layered_curve is the informative
   stand-in until this lands.
"""
import numpy as np

from ..data import loader
from .. import config
from . import probe as _probe
from .battery import _proj_out

STATUS = "implemented_cached_layered"


def layered_curve(model_key="phikon_v2", split="train", pos="TUM", neg="LYM",
                  space="global", n_null=200, seed=0, artifacts_dir=None):
    """Layer-resolved probe separability + readout necessity across the 3 layers.

    space: "global" (CLS) | "local" (mean patch token). Returns a per-layer curve,
    each entry vs a matched-random null.
    """
    kw = {} if artifacts_dir is None else {"artifacts_dir": artifacts_dir}
    layer_names = loader.available_layers(model_key, split, **kw)
    curve = []
    for layer in layer_names:
        X, labels, class_names, source = loader.load_layer(
            model_key, split, layer=layer, space=space, **kw)
        Xp, y = _probe.select_pair(X, labels, class_names, pos, neg)
        fit = _probe.fit_probe(Xp, y, seed=seed)
        Z = fit["scaler"].transform(Xp)
        u = fit["direction"]
        base = float(fit["clf"].score(Z, y))
        concept_abl = float(fit["clf"].score(_proj_out(Z, u), y))
        R = _probe.matched_random_dirs(Z.shape[1], n_null, seed=seed)
        null = np.array([fit["clf"].score(_proj_out(Z, r), y) for r in R])
        curve.append({
            "layer": layer,
            "probe_acc": base,
            "concept_ablated_acc": concept_abl,
            "random_ablated_acc_mean": float(null.mean()),
            "random_ablated_acc_std": float(null.std()),
        })
    return {
        "status": STATUS,
        "space": space,
        "layers": layer_names,
        "n_null": n_null,
        "curve": curve,
        "note": ("cached readout-space necessity per layer (probe separability + "
                 "concept-axis projection vs matched-random null); the live "
                 "source-intervention propagation test is necessity_curve() [track #3]"),
    }


def necessity_curve(model_key, pos="TUM", neg="LYM", layers=None, n_null=200, seed=0):
    """TRUE source-intervention necessity (edit @ layer L, propagate to readout).

    NOT YET IMPLEMENTED (track #3): needs live forward hooks, not cached features.
    Use layered_curve() for the cached layer-resolved stand-in.
    """
    raise NotImplementedError(
        "necessity_curve: live source-intervention (hook encoder.layer[L], edit "
        "activations, propagate to readout) not built yet — track #3. "
        "layered_curve() is the cached stand-in.")


def pending_report(model_key="phikon_v2", split="train", pos="TUM", neg="LYM",
                   space="global", artifacts_dir=None):
    """Best-available layered report for the evidence card.

    Runs the cached layered_curve if multi-layer embeddings exist; otherwise
    returns a structured 'pending' marker (never a false certification).
    """
    try:
        kw = {} if artifacts_dir is None else {"artifacts_dir": artifacts_dir}
        layers = loader.available_layers(model_key, split, **kw)
        if len(layers) <= 1:
            return {"status": "pending",
                    "note": "single-layer npz; re-extract for the multi-layer curve"}
        return layered_curve(model_key, split, pos, neg, space=space, **kw)
    except Exception as e:  # loader/probe failure -> honest pending, not a crash
        return {"status": "pending", "note": f"layered curve unavailable: {e}"}
