"""certify — the modular causal certificate.

Assembles necessity / sufficiency / specificity into one auditable card with:
  - normalized per-pillar blocks (same shape, composable)
  - a UNIVERSAL confidence score in [0,1] (comparable across concepts/models/tracks)
  - a deterministic reasoning trace (why the score is what it is)
  - reuse handles (concept direction + scaler + alpha) persisted so a FUTURE steer/
    ablate MCP call runs with no recompute: Z=(X-mean)/scale; Z+=a*u (steer) or
    Z-=(Z.u)u (ablate)

Confidence method (documented on the card so it is interpretable):
  suff = clip(concept_flip - random_flip, 0, 1)
  nec  = clip((rand_abl - concept_abl) / (rand_abl - chance), 0, 1)
  spec = clip((target_acc/base) * (1 - cos), 0, 1)
  overall = geomean(suff, nec, spec)  gated by the matched-random null (0 if random
            reproduces any effect -- the Section-5-D falsifier) and scaled by the
            confound factor.
"""
import os

import numpy as np

from .. import config
from . import confound as _confound
from . import intervene as _intervene

SCHEMA_VERSION = "0.2"

CITATIONS = {
    "confound": {"ref": "Kömen et al. 2024, arXiv:2411.05489",
                 "claim": "pathology FMs retain site/scanner signatures; scanner-ID ~1.000 for Phikon-v2"},
    "necessity_redundancy": {"ref": "Bio-Interp D02/D04; McGrath et al. 2023 (Hydra)",
                             "claim": "concepts redundantly encoded; mid-layer ablation recomputed downstream"},
    "sufficiency_asymmetry": {"ref": "SwordBench, arXiv:2605.16372",
                              "claim": "on pathology FMs steering is the clean, concept-specific axis"},
    "positioning": {"ref": "SpatialProp (Zou et al. 2025, PMC12822716)",
                    "claim": "we bring do()-style necessity/sufficiency/specificity + a confound gate into a pathology-image FM latent"},
}

HONESTY_CAVEAT = (
    "A latent do() is an intervention on the model's REPRESENTATION, not on tissue "
    "biology. This certifies model-internal causal use; biological validity rests on "
    "encoder faithfulness — which is why the confound gate and literature grounding exist.")


def _clip01(x):
    return float(max(0.0, min(1.0, x)))


def _geomean(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return 0.0
    prod = 1.0
    for v in vals:
        prod *= max(v, 1e-9)
    return float(prod ** (1.0 / len(vals)))


# --------------------------------------------------------------------------
# Modular pillars: each returns the same normalized shape.
# --------------------------------------------------------------------------
def _pillar(name, statistic, null, effect, confidence, passed, verdict, raw):
    return {"pillar": name, "statistic": statistic, "null": null, "effect": effect,
            "confidence": _clip01(confidence), "passed": bool(passed),
            "verdict": verdict, "raw": raw}


# On cleanly separable concepts the readout probe is ~1-D, so projecting out its OWN axis
# collapses accuracy to chance TAUTOLOGICALLY -> readout necessity saturates at 1.0 and
# carries no information (CLAUDE.md: necessity is redundancy-limited, only "bites" near the
# readout). So (a) prefer a live/layer-resolved source-intervention curve when one is
# present (graded, per-slide, discriminating), else (b) cap the readout-only necessity so a
# tautological collapse cannot prop the score up to ~1.
READOUT_ONLY_NECESSITY_CAP = 0.5

# An UNCHECKED confound gate (single-source data) means biological validity is unverified;
# a model-internal-only certificate must not read near-certain. Scale the headline down
# until multi-site site-probe data exists. See STRATEGY.md — the confound gate is the wedge.
UNCHECKED_CONFOUND_FACTOR = 0.7


def _necessity_confidence(n, chance, live=None, layered=None):
    """(confidence, basis). Graded per-slide necessity when a REAL source-intervention curve
    is present (one that bites before the readout); else the readout ablation, capped and
    flagged redundancy-limited because a 1-D-probe project-out is near-tautological."""
    for src in (live, layered):
        curve = src.get("curve") if isinstance(src, dict) else None
        if curve and all("necessity_gap" in c for c in curve):
            readout = next((c["necessity_gap"] for c in curve if c.get("layer") == "readout"), None)
            pre = [c["necessity_gap"] for c in curve if c.get("layer") != "readout"]
            if readout and readout > 1e-6 and pre:
                # graded: fraction of the readout necessity already irreversible BEFORE the
                # readout. Low => the concept is recomputed downstream until the last layer
                # (redundancy-limited / Hydra effect); high => it bites early = truly necessary.
                return _clip01(max(pre) / readout), "live source-intervention (pre-readout / readout)"
    eff = (n["random_ablated_acc_mean"] - n["concept_ablated_acc"]) / \
          max(n["random_ablated_acc_mean"] - chance, 1e-6)
    return _clip01(min(eff, READOUT_ONLY_NECESSITY_CAP)), "readout-only (redundancy-limited, capped)"


def build_pillars(bc, chance=0.5, live=None, layered=None):
    """Normalize the raw battery card into the three composable pillars."""
    n, s = bc["necessity_readout"], bc["sufficiency_steering"]
    sp = bc.get("specificity", {})

    nec_conf, nec_basis = _necessity_confidence(n, chance, live=live, layered=layered)
    necessity = _pillar(
        "necessity",
        statistic=n["concept_ablated_acc"], null=n["random_ablated_acc_mean"],
        effect=n["random_ablated_acc_mean"] - n["concept_ablated_acc"],
        confidence=nec_conf,
        passed=n["concept_ablated_acc"] <= chance + 0.05,
        verdict=n["verdict"], raw={**n, "necessity_basis": nec_basis})

    suff_eff = s["concept_flip_rate"] - s["random_flip_rate_mean"]
    sufficiency = _pillar(
        "sufficiency",
        statistic=s["concept_flip_rate"], null=s["random_flip_rate_mean"],
        effect=suff_eff, confidence=suff_eff,
        passed=s["concept_flip_rate"] > 0.5 and s["random_flip_rate_mean"] < 0.1,
        verdict=s["verdict"], raw=s)

    if "target_acc_after_distractor_ablation" in sp:
        intact = sp["target_acc_after_distractor_ablation"] / max(sp["base_acc"], 1e-6)
        orth = 1.0 - sp["cos_with_concept_axis"]
        specificity = _pillar(
            "specificity",
            statistic=sp["target_acc_after_distractor_ablation"], null=sp["base_acc"],
            effect=orth, confidence=_clip01(intact) * _clip01(orth),
            passed=sp["target_acc_after_distractor_ablation"] > sp["base_acc"] - 0.05,
            verdict=sp["verdict"], raw=sp)
    else:
        specificity = None
    return {"necessity": necessity, "sufficiency": sufficiency, "specificity": specificity}


def compute_confidence(pillars, bc, confound_result, chance=0.5):
    """Universal [0,1] confidence: geomean of pillars, gated by null, scaled by confound."""
    n, s = bc["necessity_readout"], bc["sufficiency_steering"]

    # Falsifier gate: the matched-random null must be inert.
    integrity, reasons = True, []
    if s["random_flip_rate_mean"] > 0.1:
        integrity = False; reasons.append("random steering flips > 10%")
    if n["random_ablated_acc_mean"] < chance + 0.05:
        integrity = False; reasons.append("random ablation already at chance")

    per = {k: v["confidence"] for k, v in pillars.items() if v is not None}
    base = _geomean(list(per.values()))

    # Confound scaling.
    cf = confound_result.get("confound_gate", confound_result)
    if cf.get("status") == "ok":
        checked = True
        factor = (1.0 - cf.get("cos_concept_with_site", 0.0)) if cf.get("confounded") else 1.0
    else:
        checked, factor = False, 1.0

    overall = 0.0 if not integrity else _clip01(base * factor)
    # Honest cap: while the confound gate is UNCHECKED (single-source data), biology is
    # unverified, so the headline cannot read near-certain — scale it down.
    unchecked_cap = integrity and not checked
    if unchecked_cap:
        overall = _clip01(overall * UNCHECKED_CONFOUND_FACTOR)
    return {
        "overall": overall,
        "pillars": per,
        "null_integrity": integrity,
        "null_integrity_reasons": reasons,
        "confound_checked": checked,
        "confound_factor": factor,
        "confound_uncertainty_capped": bool(unchecked_cap),
        "unchecked_confound_factor": UNCHECKED_CONFOUND_FACTOR if unchecked_cap else 1.0,
        "method": "geomean(necessity, sufficiency, specificity) x confound_factor, zeroed "
                  "if the matched-random null is not inert; necessity is redundancy-limited "
                  f"(readout-only capped at {READOUT_ONLY_NECESSITY_CAP}); overall x"
                  f"{UNCHECKED_CONFOUND_FACTOR} while the confound gate is UNCHECKED "
                  "(single-source: biological validity unverified)",
        "interpretation": ("high = concept axis is necessary AND sufficient AND specific, "
                           "with an inert random null and no site confound"),
    }


def reasoning_trace(bc, pillars, confidence, confound_result):
    """Deterministic step-by-step narrative built from the numbers."""
    n, s = bc["necessity_readout"], bc["sufficiency_steering"]
    p = bc["probe"]
    t = [
        {"n": 1, "pillar": "probe",
         "observation": f"probe test acc = {p['test_acc']:.3f} (n_test={p['n_test']})",
         "interpretation": "concept is linearly encoded on the frozen CLS"},
        {"n": 2, "pillar": "necessity",
         "observation": f"project concept axis out: {n['base_acc']:.3f} -> "
                        f"{n['concept_ablated_acc']:.3f}; random axes -> "
                        f"{n['random_ablated_acc_mean']:.3f}+/-{n['random_ablated_acc_std']:.3f}",
         "interpretation": "the concept axis carries the readout; random axes do not "
                           "(readout-space only; redundancy story needs the live curve)"},
        {"n": 3, "pillar": "sufficiency",
         "observation": f"inject concept dir: flip {s['concept_flip_rate']:.3f} vs "
                        f"random {s['random_flip_rate_mean']:.3f}",
         "interpretation": "steering is concept-specific — the headline signal"},
    ]
    sp = pillars.get("specificity")
    if sp:
        t.append({"n": 4, "pillar": "specificity",
                  "observation": f"distractor ablation: target {sp['statistic']:.3f} "
                                 f"(base {sp['null']:.3f})",
                  "interpretation": "target survives an orthogonal distractor ablation"})
    cf = confound_result.get("confound_gate", confound_result)
    t.append({"n": len(t) + 1, "pillar": "confound",
              "observation": f"confound gate: {cf.get('status')}",
              "interpretation": ("no site variation to test (single-source)" if
                                 cf.get("status") == "no_multisite_data" else
                                 cf.get("verdict", ""))})
    t.append({"n": len(t) + 1, "pillar": "confidence",
              "observation": f"overall confidence = {confidence['overall']:.3f}",
              "interpretation": confidence["method"]})
    # Standardized schema: every trace step carries BOTH `step` and `pillar` (same
    # value) so a UI can bind to either key regardless of which certify verb produced it.
    for s in t:
        s["step"] = s["pillar"]
    return t


def persist_handles(handles, model_key, split, pos, neg, artifacts_dir):
    """Persist reuse handles to directions/<dataset>/<model>/<pos>_vs_<neg>.npz.

    Returns the reuse block for the card (key + scalar params). On write failure
    (e.g. read-only artifacts) returns a block with direction_key=None + note.
    """
    key = config.directions_key(model_key, f"{pos}_vs_{neg}")
    reuse = {
        "direction_key": key, "alpha_classwidth": handles["alpha_classwidth"],
        "chance": handles["chance"], "pos": pos, "neg": neg,
        "model": model_key, "split": split, "layer": "readout", "space": "global",
        "ready_for": ["steer", "ablate"],
        "recipe": {"steer": "Z=(X-mean)/scale; Z += alpha*direction_std; predict(coef,intercept)",
                   "ablate": "Z=(X-mean)/scale; Z -= (Z.direction_std)*direction_std"},
    }
    try:
        path = os.path.join(artifacts_dir, key)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez_compressed(
            path, direction_std=handles["direction_std"],
            scaler_mean=handles["scaler_mean"], scaler_scale=handles["scaler_scale"],
            coef=handles["coef"], intercept=handles["intercept"],
            alpha_classwidth=handles["alpha_classwidth"], chance=handles["chance"],
            pos=pos, neg=neg)
        reuse["persisted"] = True
    except OSError as e:
        reuse["direction_key"] = None
        reuse["persisted"] = False
        reuse["note"] = f"could not persist handles: {e}"
    return reuse


def load_direction(model_key, pos, neg, artifacts_dir):
    """Load persisted reuse handles for a future steer/ablate (no recompute)."""
    path = os.path.join(artifacts_dir, config.directions_key(model_key, f"{pos}_vs_{neg}"))
    d = np.load(path, allow_pickle=True)
    return {k: d[k] for k in d.files}


def apply_steer(X_raw, handles, alpha=None):
    """Steer raw CLS features toward pos using ONLY the persisted handles (no probe).

    Returns (predictions, logits). This is the future-MCP steer path from a card.
    """
    a = handles["alpha_classwidth"] if alpha is None else alpha
    Z = (np.asarray(X_raw) - handles["scaler_mean"]) / handles["scaler_scale"]
    Zs = Z + float(a) * handles["direction_std"]
    logits = Zs @ handles["coef"] + float(handles["intercept"])
    return (logits > 0).astype(int), logits


def apply_ablate(X_raw, handles):
    """Ablate the concept axis from raw CLS features using ONLY persisted handles.

    Returns (predictions, logits). This is the future-MCP ablate path from a card.
    """
    Z = (np.asarray(X_raw) - handles["scaler_mean"]) / handles["scaler_scale"]
    u = handles["direction_std"]
    Za = Z - np.outer(Z @ u, u)
    logits = Za @ handles["coef"] + float(handles["intercept"])
    return (logits > 0).astype(int), logits


def certify(feats, labels, class_names, pos, neg, distractor, model_key, split,
            source, n_null=200, artifacts_dir=".", seed=0):
    """Full modular certificate: pillars + universal confidence + trace + reuse handles."""
    from .battery import run_battery  # local import avoids battery<->certify cycle

    bc, handles = run_battery(feats, labels, class_names, pos=pos, neg=neg,
                              distractor=distractor, n_null=n_null, seed=seed,
                              return_handles=True)
    confound_result = _confound.confound_gate(
        feats, labels, class_names, site_labels=None, pos=pos, neg=neg)
    layered = _intervene.pending_report(model_key, split, pos, neg,
                                        artifacts_dir=artifacts_dir)

    chance = 0.5
    pillars = build_pillars(bc, chance=chance, layered=layered)
    confidence = compute_confidence(pillars, bc, {"confound_gate": confound_result}, chance)
    trace = reasoning_trace(bc, pillars, confidence, {"confound_gate": confound_result})
    reuse = persist_handles(handles, model_key, split, pos, neg, artifacts_dir)

    return {
        "schema_version": SCHEMA_VERSION,
        "prediction": {"model": model_key, "split": split,
                       "concept": f"{pos}_vs_{neg}", "distractor": list(distractor),
                       "embeddings_source": source},
        "confidence": confidence,
        "pillars": pillars,
        "probe": bc["probe"],
        "necessity_layered": layered,
        "confound_gate": confound_result,
        "reasoning_trace": trace,
        "reuse": reuse,
        "citations": CITATIONS,
        "caveat": HONESTY_CAVEAT,
        "certified_verb": ("concept-specific steering + confound triage; necessity "
                           "reported honestly as redundancy-limited"),
    }
