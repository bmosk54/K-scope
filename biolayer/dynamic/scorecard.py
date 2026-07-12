"""Per-claim scoring: numeric per-pillar confidence + honest 3-bin verdict.

Two outputs on purpose:
  - NUMERIC scores in [0,1] for necessity / sufficiency / specificity (the
    continuous read the operator asked for), each with its raw effect + a z vs the
    matched-random null where a null exists.
  - A discrete VERDICT (GROUNDED / WEAK / NULL) keyed on sign + significance, which
    is the load-bearing claim — magnitude is run-variable (Bio-Interp D05b->D07), so
    the bin, not the raw number, is what a certificate should stand behind.

Necessity is reported redundancy-limited: on a redundant pathology FM it collapses
only near the readout (Hydra effect), so a "probe survives ablation" is NOT read as
"concept absent" — it's marked redundancy-limited and never downgrades sufficiency.
"""
import math
from dataclasses import dataclass, field

Z_CRIT = 1.645          # one-sided p<0.05
GROUNDED, WEAK, NULL = "GROUNDED", "WEAK", "NULL"


def _clip01(x):
    return float(max(0.0, min(1.0, x)))


def _p_from_z(z):
    """One-sided normal survival function p-value from a z-score."""
    if z is None or math.isnan(z):
        return 1.0
    return 0.5 * math.erfc(z / math.sqrt(2))


@dataclass
class PillarScore:
    name: str
    score: float            # [0,1] numeric confidence
    effect: float           # raw effect size
    z: float                # sigmas above the matched-random null (nan if no null)
    p: float                # one-sided p vs null
    has_null: bool
    verdict: str


@dataclass
class ClaimScore:
    concept: str
    pillars: dict           # name -> PillarScore
    verdict: str            # rolled-up claim verdict
    confounded: bool
    intervened_on_input: bool
    survives_correction: bool = False   # set by Holm pass over the answer
    contrast_capped: bool = False       # verdict capped because the contrast failed the gate
    necessity_capped: bool = False      # GROUNDED denied: no genuine (non-readout/live) necessity
    notes: list = field(default_factory=list)

    @property
    def min_p(self):
        ps = [p.p for p in self.pillars.values() if p.has_null]
        return min(ps) if ps else 1.0


def _necessity_live(live):
    """Necessity from a LIVE source-intervention curve (edit @ layer L on the real
    forward pass, margin-drop vs matched-random null). This is the graded, per-slide
    read — GROUNDED requires biting at a NON-readout layer (genuine distributed
    necessity), since readout-only ablation ~= projecting out the probe's own axis.
    Score = fraction of the readout-necessity that is already irreversible before the
    readout (how little the model recomputes it)."""
    curve = live["curve"]
    readout_gap = max((c["necessity_gap"] for c in curve if c["layer"] == "readout"),
                      default=0.0)
    non_ro = [c for c in curve if c["layer"] != "readout"]
    best = max(non_ro, key=lambda c: c["gap_vs_null_z"]) if non_ro else curve[-1]
    gap, z = best["necessity_gap"], best["gap_vs_null_z"]
    score = _clip01(gap / readout_gap) if readout_gap > 1e-6 else _clip01(gap)
    v = GROUNDED if (z >= Z_CRIT and gap > 0) else (WEAK if gap > 0 else NULL)
    ps = PillarScore("necessity", score, float(gap), float(z), _p_from_z(z), True, v)
    return ps, best["layer"], True


def _necessity(bc, layered, live=None):
    """Numeric + z for necessity. If a live source-intervention curve is present, use
    it (graded, per-slide); else fall back to the cached readout necessity (capped WEAK
    at the readout — near-tautological)."""
    if isinstance(live, dict) and live.get("curve"):
        ps, bites_layer, _ = _necessity_live(live)
        return ps, bites_layer
    n = bc["necessity_readout"]
    chance = 0.5
    eff = (n["random_ablated_acc_mean"] - n["concept_ablated_acc"]) / \
          max(n["random_ablated_acc_mean"] - chance, 1e-6)
    z = n["concept_vs_null_z"]
    bites_layer = "readout"
    if isinstance(layered, dict) and layered.get("curve"):
        gaps = [(c["random_ablated_acc_mean"] - c["concept_ablated_acc"], c["layer"])
                for c in layered["curve"]]
        if gaps:
            bites_layer = max(gaps)[1]
    # readout-only necessity is near-tautological (projecting out a ~1-D probe's own axis
    # always collapses to chance), so cap the NUMERIC score too — not just the verdict —
    # or a redundancy-limited concept reads as necessity=1.0. Graded per-slide necessity
    # (the live source-intervention path above) is unaffected.
    score = _clip01(min(eff, 0.5) if bites_layer == "readout" else eff)
    p = _p_from_z(z)
    if bites_layer == "readout":
        v = WEAK if eff > 0.0 else NULL
    else:
        v = GROUNDED if (z >= Z_CRIT and eff > 0.05) else (WEAK if eff > 0.0 else NULL)
    ps = PillarScore("necessity", score, float(n["random_ablated_acc_mean"]
                     - n["concept_ablated_acc"]), float(z), p, True, v)
    return ps, bites_layer


def _sufficiency(bc):
    s = bc["sufficiency_steering"]
    eff = s["concept_flip_rate"] - s["random_flip_rate_mean"]
    z = (s["concept_flip_rate"] - s["random_flip_rate_mean"]) / \
        (s["random_flip_rate_std"] + 1e-9)
    # Graded steering-AUC (mean flip over push strengths) instead of the full-class-width
    # flip rate, which saturates at 1.0 on any separable concept. Raw flip stays in .effect.
    if "steering_auc" in s:
        score = _clip01(s["steering_auc"] - s.get("random_steering_auc", 0.0))
    else:
        score = _clip01(eff)
    p = _p_from_z(z)
    v = GROUNDED if (s["concept_flip_rate"] > 0.5 and s["random_flip_rate_mean"] < 0.1
                     and z >= Z_CRIT) else (WEAK if eff > 0.0 else NULL)
    return PillarScore("sufficiency", score, float(eff), float(z), p, True, v)


def _specificity(bc):
    sp = bc.get("specificity", {})
    if "target_acc_after_distractor_ablation" not in sp:
        return PillarScore("specificity", 0.0, 0.0, float("nan"), 1.0, False, NULL)
    intact = sp["target_acc_after_distractor_ablation"] / max(sp["base_acc"], 1e-6)
    orth = 1.0 - sp["cos_with_concept_axis"]
    # In high dim two unrelated axes are near-orthogonal by default (1-cos ~ 0.95 ~ what a
    # random distractor scores), so credit only orthogonality ABOVE that baseline -> a
    # realistic band instead of a flat ~0.95. Raw 1-cos stays in .effect.
    score = _clip01(0.6 + 0.35 * ((orth - 0.85) / 0.15)) * _clip01(intact)
    # no matched-random null in the current specificity path (n_null=1) -> no z.
    passed = sp["target_acc_after_distractor_ablation"] > sp["base_acc"] - 0.05
    v = GROUNDED if (passed and sp["cos_with_concept_axis"] < 0.3) else (
        WEAK if passed else NULL)
    return PillarScore("specificity", score, float(orth), float("nan"), 1.0, False, v)


def score_claim(concept, bc, layered, confound_result, intervened_on_input=False,
                live_necessity=None, contrast_valid=True, contrast_warnings=()):
    """Assemble one claim's numeric scores + per-pillar and rolled-up verdicts.

    If `live_necessity` (a live source-intervention curve) is present, the necessity
    pillar is the graded per-slide read and `intervened_on_input` should be True.
    """
    nec, bites_layer = _necessity(bc, layered, live=live_necessity)
    suf = _sufficiency(bc)
    spec = _specificity(bc)
    pillars = {p.name: p for p in (nec, suf, spec)}

    cf = confound_result.get("confound_gate", confound_result)
    confounded = bool(cf.get("confounded", False))

    if live_necessity and live_necessity.get("curve"):
        notes = [f"necessity via LIVE source-intervention (edit @ layer on this slide's "
                 f"forward pass): bites from layer {bites_layer}; readout drop dominates, "
                 f"early layers partly recomputed (redundancy resolved, not degenerate)"]
    else:
        notes = [f"necessity is redundancy-limited: bites at layer {bites_layer} "
                 f"(cached readout-space; live source-intervention is the real read)"]
    if cf.get("status") != "ok":
        notes.append("confound gate UNAVAILABLE (single-source data) — biological "
                     "validity not established")
    if not intervened_on_input:
        notes.append("certified on reference-set separability, NOT on this input's "
                     "forward pass — pass live_ctx for the per-slide intervention")

    # Roll-up. Sufficiency is the clean, de-circularized signal (concept flip vs a
    # matched-random null), but ON ITS OWN it is near-circular: you inject the
    # class-mean-diff axis and score a probe built on it. So a GROUNDED verdict must ALSO
    # rest on a GENUINE, non-tautological necessity — either the live per-slide
    # source-intervention, or a cached necessity that bites at a NON-readout layer (real
    # distributed necessity). Readout-only necessity is near-tautological (projecting out a
    # ~1-D probe's own axis always collapses it), so it can NOT promote GROUNDED. This is
    # what stops a WEAK necessity + near-circular sufficiency from certifying GROUNDED.
    genuine_necessity = intervened_on_input or (bites_layer != "readout"
                                                and nec.verdict == GROUNDED)
    necessity_capped = False
    if suf.verdict == NULL and nec.verdict == NULL:
        verdict = NULL
    elif suf.verdict == GROUNDED and spec.verdict != NULL:
        if genuine_necessity:
            verdict = GROUNDED
        else:
            verdict = WEAK
            necessity_capped = True
            notes.append(
                "capped at WEAK: sufficiency + specificity pass, but necessity is the "
                "near-tautological readout-space projection (no live / non-readout bite) — "
                "a GROUNDED verdict must not rest on near-circular sufficiency alone. Pass "
                "live_ctx for the per-slide source-intervention, or run the layered sweep "
                "(fast=False) to measure genuine distributed necessity.")
    else:
        verdict = WEAK
    if confounded and verdict == GROUNDED:
        verdict = WEAK
        notes.append("capped at WEAK: causal axis overlaps a site/scanner signature")

    # SAFETY CAP: a contrast that fails the validation gate (low held-out AUROC, or an
    # axis riding the staining/intensity proxy) may separate perfectly yet certify the
    # wrong thing. It must NEVER read GROUNDED — cap at WEAK regardless of the pillars.
    contrast_capped = False
    if not contrast_valid and verdict == GROUNDED:
        verdict = WEAK
        contrast_capped = True
        notes.append("capped at WEAK: contrast failed validation ("
                     + "; ".join(contrast_warnings)
                     + ") — the probe may ride staining/intensity, not clean biology")

    return ClaimScore(concept=concept, pillars=pillars, verdict=verdict,
                      confounded=confounded, intervened_on_input=intervened_on_input,
                      contrast_capped=contrast_capped, necessity_capped=necessity_capped,
                      notes=notes)


def holm_correction(claim_scores, alpha=0.05):
    """Holm-Bonferroni over per-claim min-p; gate GROUNDED on survival.

    Agents probe many concepts per answer; without correction the confident
    certificates are cherry-picked. A claim can only stay GROUNDED if its smallest
    pillar p survives the step-down correction.
    """
    ranked = sorted(claim_scores, key=lambda c: c.min_p)
    m = len(ranked)
    for i, cs in enumerate(ranked):
        thresh = alpha / (m - i) if (m - i) > 0 else alpha
        cs.survives_correction = cs.min_p <= thresh
        if not cs.survives_correction and cs.verdict == GROUNDED:
            cs.verdict = WEAK
            cs.notes.append(f"downgraded to WEAK: min-p {cs.min_p:.4f} fails "
                            f"Holm-Bonferroni (thresh {thresh:.4f}, {m} claims)")
    return claim_scores
