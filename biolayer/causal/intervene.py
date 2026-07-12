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


def live_necessity(model_key, images, image_labels, class_names, pos="TUM", neg="LYM",
                   ref_images=None, ref_labels=None, split="train", n_null=30, seed=0,
                   readout_pos=None, readout_neg=None, artifacts_dir=None, encoder=None):
    """TRUE source-intervention necessity: edit @ layer L on THIS tile's forward pass,
    propagate to the readout, measure the effect vs a matched-random null.

    For each configured layer L we derive the concept axis at L (diff-of-means on this
    run's hidden states), hook the block that produces hidden_states[L], project that
    axis out of the CLS token, and let L+1..final RECOMPUTE. We then read the readout
    probe's decision MARGIN (not saturating probability) on the readout-positive tiles
    and measure how much the concept ablation drops it vs matched-random ablations.

    The readout probe is fit LIVE on `ref_images` (same representation as the
    intervened forward pass — avoids the cached-npz representation mismatch). Passing a
    reference set disjoint from `images` keeps it non-circular; without one we fall back
    to the cached probe and flag the mismatch risk.

    readout_pos/neg override the SCORED concept (default = the ablated concept) — set a
    different pair for the ablate-A-score-B cross-interference variant. Sets
    intervened_on_input=True: the certificate's per-slide causal claim.
    """
    from . import live as _live
    from . import probe as _probe

    kw = {} if artifacts_dir is None else {"artifacts_dir": artifacts_dir}
    cls_list = list(class_names)
    rpos, rneg = (readout_pos or pos), (readout_neg or neg)
    image_labels = np.asarray(image_labels)
    enc = encoder or _live.LiveEncoder(model_key)

    # Readout probe — live-fit on reference tiles (matched representation) preferred.
    if ref_images is not None:
        ref_labels = np.asarray(ref_labels)
        rp, rn = cls_list.index(rpos), cls_list.index(rneg)
        rmask = (ref_labels == rp) | (ref_labels == rn)
        Xref = enc.embed([ref_images[i] for i in np.where(rmask)[0]])
        rfit = _probe.fit_probe(Xref, (ref_labels[rmask] == rp).astype(int), seed=seed)
        probe_source = "live_reference_fit"
    else:
        feats_r, lab_r, cn_r, _ = loader.load(model_key, split, **kw)
        Xr, yr = _probe.select_pair(feats_r, lab_r, cn_r, rpos, rneg)
        rfit = _probe.fit_probe(Xr, yr, seed=seed)
        probe_source = "cached_probe(representation_mismatch_risk)"

    def margin(cls):  # signed distance toward readout-pos (graded, non-saturating)
        return rfit["clf"].decision_function(rfit["scaler"].transform(cls))

    rpos_idx = cls_list.index(rpos)
    watch = image_labels == rpos_idx
    pos_idx, neg_idx = cls_list.index(pos), cls_list.index(neg)
    pn = (image_labels == pos_idx) | (image_labels == neg_idx)
    if watch.sum() == 0 or (image_labels == pos_idx).sum() < 2 or (image_labels == neg_idx).sum() < 2:
        return {"status": "insufficient_tiles", "intervened_on_input": True,
                "note": f"need >=2 {pos} and >=2 {neg} tiles for the axis, and >=1 {rpos} to watch"}

    clean_readout, clean_hidden = enc.hidden_cls(images)   # (N,dim), (N,n_blocks+1,dim)
    m0 = margin(clean_readout)                             # post-LN readout, matches probe
    base = float(m0[watch].mean())

    layer_names = list(config.LAYER_NAMES)
    layer_idx = list(config.MODELS[model_key]["layers"])   # hidden_states indices
    curve = []
    for name, L in zip(layer_names, layer_idx):
        Xp = clean_hidden[pn, L, :]
        cdir = _probe.diff_of_means(Xp, (image_labels[pn] == pos_idx).astype(int))
        block_idx = L - 1                                   # hidden_states[L] = block[L-1] output

        mc = margin(enc.embed(images, edit=_live.project_out(cdir), block_idx=block_idx))
        R = _probe.matched_random_dirs(cdir.shape[0], n_null, seed=seed)
        mr = np.array([margin(enc.embed(images, edit=_live.project_out(r),
                                        block_idx=block_idx)) for r in R])   # (n_null, N)

        concept_drop = float((m0[watch] - mc[watch]).mean())
        null_drops = (m0[watch][None, :] - mr[:, watch]).mean(axis=1)        # (n_null,)
        n_mean, n_std = float(null_drops.mean()), float(null_drops.std())
        gap = concept_drop - n_mean            # >0 => concept ablation bites harder than random
        z = gap / (n_std + 1e-9)
        curve.append({
            "layer": name, "block_idx": block_idx,
            "base_margin": round(base, 3),
            "concept_ablation_drop": round(concept_drop, 3),
            "random_ablation_drop_mean": round(n_mean, 3),
            "random_ablation_drop_std": round(n_std, 3),
            "necessity_gap": round(gap, 3),
            "gap_vs_null_z": round(z, 2),
            "bites": bool(gap > 0 and z >= 1.645),
        })
    return {
        "status": "live_source_intervention",
        "intervened_on_input": True,
        "model": model_key, "readout_probe": probe_source,
        "ablated_concept": f"{pos}_vs_{neg}",
        "scored_concept": f"{rpos}_vs_{rneg}",
        "cross_interference": (rpos, rneg) != (pos, neg),
        "n_tiles": int(len(images)), "n_watched": int(watch.sum()), "n_null": n_null,
        "curve": curve,
        "note": ("edit @ layer L on the real forward pass, propagated to readout "
                 "(margin drop vs matched-random null). Small mid-layer gap => concept "
                 "recomputed downstream (redundancy/Hydra); a real per-slide causal read."),
    }


def necessity_curve(*a, **k):
    """Back-compat alias — live source intervention is `live_necessity`."""
    return live_necessity(*a, **k)


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
