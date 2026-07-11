"""Evidence-card schema + builder.

The `certify` verb's output is a structured, auditable causal evidence card — not
a trust score, not a dashboard (STRATEGY.md §3). This module assembles the card
from the causal-battery result plus the confound and layer-resolved-necessity
pillars, and staples the literature grounding onto every claim.
"""

SCHEMA_VERSION = "0.1"

# Literature grounding — cited on the card so the certification is auditable.
CITATIONS = {
    "confound": {
        "ref": "Kömen et al. 2024, arXiv:2411.05489",
        "claim": ("pathology FMs retain linearly-recoverable site/scanner "
                  "signatures; scanner-ID ~1.000 for Phikon-v2"),
    },
    "necessity_redundancy": {
        "ref": "Bio-Interp D02/D04; McGrath et al. 2023 (Hydra effect)",
        "claim": ("concepts are redundantly encoded; mid-layer ablation is "
                  "recomputed downstream, so necessity is redundancy-limited"),
    },
    "sufficiency_asymmetry": {
        "ref": "SwordBench, arXiv:2605.16372",
        "claim": ("ablation(necessity) robust / injection(sufficiency) fragile in "
                  "vision; on pathology FMs the asymmetry inverts — steering is clean"),
    },
    "positioning": {
        "ref": "SpatialProp (Sun, Buendia, Brunet & Zou 2025, PMC12822716)",
        "claim": ("certifies causality in transcriptomic space; we bring do()-style "
                  "necessity/sufficiency/specificity + a confound gate into a "
                  "pathology-image FM latent"),
    },
}

# The honesty caveat, verbatim from CLAUDE.md — stated on every card.
HONESTY_CAVEAT = (
    "A latent do() is an intervention on the model's REPRESENTATION, not on tissue "
    "biology. This certifies model-internal causal use; biological validity rests on "
    "encoder faithfulness — which is why the confound gate and literature grounding exist."
)


def build_evidence_card(battery_card, confound_result, intervene_report,
                        model_key, split, source, pos, neg):
    """Assemble the full certify() evidence card from the pillar results."""
    return {
        "schema_version": SCHEMA_VERSION,
        "prediction": {"model": model_key, "split": split,
                       "concept": f"{pos}_vs_{neg}", "embeddings_source": source},
        # ---- causal pillars (each vs a matched-random null) -----------------
        "probe": battery_card.get("probe"),
        "necessity_readout": battery_card.get("necessity_readout"),
        "sufficiency_steering": battery_card.get("sufficiency_steering"),
        "specificity": battery_card.get("specificity"),
        "necessity_layered": intervene_report,   # rigor curve (track #3, pending)
        "confound_gate": confound_result,        # differentiator (track #2)
        # ---- audit trail -----------------------------------------------------
        "citations": CITATIONS,
        "caveat": HONESTY_CAVEAT,
        "certified_verb": ("concept-specific steering + confound triage; necessity "
                           "reported honestly as redundancy-limited"),
    }
