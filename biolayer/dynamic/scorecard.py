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
    notes: list = field(default_factory=list)

    @property
    def min_p(self):
        ps = [p.p for p in self.pillars.values() if p.has_null]
        return min(ps) if ps else 1.0


def _necessity(bc, layered):
    """Numeric + z from readout necessity; effect taken as the max over layers."""
    n = bc["necessity_readout"]
    chance = 0.5
    eff = (n["random_ablated_acc_mean"] - n["concept_ablated_acc"]) / \
          max(n["random_ablated_acc_mean"] - chance, 1e-6)
    z = n["concept_vs_null_z"]
    # layer-resolved: keep the best-biting layer for honesty (usually the readout).
    bites_layer = "readout"
    if isinstance(layered, dict) and layered.get("curve"):
        gaps = [(c["random_ablated_acc_mean"] - c["concept_ablated_acc"], c["layer"])
                for c in layered["curve"]]
        if gaps:
            bites_layer = max(gaps)[1]
    score = _clip01(eff)
    p = _p_from_z(z)
    # Readout-space necessity is near-tautological: projecting the class-mean-diff
    # out one layer below the readout is projecting out the probe's own axis
    # (Bio-Interp: readout collapse leaking upward). Genuine distributed necessity
    # requires biting at a NON-readout layer. So readout-only necessity is capped at
    # WEAK (redundancy-limited), no matter how large z is.
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
    score = _clip01(intact) * _clip01(orth)
    # no matched-random null in the current specificity path (n_null=1) -> no z.
    passed = sp["target_acc_after_distractor_ablation"] > sp["base_acc"] - 0.05
    v = GROUNDED if (passed and sp["cos_with_concept_axis"] < 0.3) else (
        WEAK if passed else NULL)
    return PillarScore("specificity", score, float(orth), float("nan"), 1.0, False, v)


def score_claim(concept, bc, layered, confound_result, intervened_on_input=False):
    """Assemble one claim's numeric scores + per-pillar and rolled-up verdicts."""
    nec, bites_layer = _necessity(bc, layered)
    suf = _sufficiency(bc)
    spec = _specificity(bc)
    pillars = {p.name: p for p in (nec, suf, spec)}

    cf = confound_result.get("confound_gate", confound_result)
    confounded = bool(cf.get("confounded", False))

    notes = [f"necessity is redundancy-limited: bites at layer {bites_layer} "
             f"(mid-network ablation is recomputed downstream — Hydra effect)"]
    if cf.get("status") != "ok":
        notes.append("confound gate UNAVAILABLE (single-source data) — biological "
                     "validity not established")
    if not intervened_on_input:
        notes.append("certified on reference-set separability, NOT on this input's "
                     "forward pass — live-hook intervention is the pending upgrade")

    # Roll-up: sufficiency is the clean, load-bearing signal on this substrate;
    # necessity is reported honestly as redundancy-limited and does not veto.
    if suf.verdict == NULL and nec.verdict == NULL:
        verdict = NULL
    elif suf.verdict == GROUNDED and spec.verdict != NULL:
        verdict = GROUNDED
    else:
        verdict = WEAK
    if confounded and verdict == GROUNDED:
        verdict = WEAK
        notes.append("capped at WEAK: causal axis overlaps a site/scanner signature")

    return ClaimScore(concept=concept, pillars=pillars, verdict=verdict,
                      confounded=confounded, intervened_on_input=intervened_on_input,
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
