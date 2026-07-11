"""Verb implementations — the plumbing behind the MCP tools.

Each verb loads the frozen embeddings and returns a JSON-able slice of the
evidence card. `certify` orchestrates all pillars into the full card. Kept free
of any MCP/FastMCP import so the verbs are unit-testable in isolation; server.py
is the thin adapter that registers them as tools.
"""
from ..causal import battery, confound, intervene
from . import card as card_mod
from . import loader


def _battery(model, split, pos, neg, n_null):
    feats, labels, class_names, source = loader.load(model, split)
    result = battery.run_battery(feats, labels, class_names, pos=pos, neg=neg,
                                 n_null=n_null)
    return result, (feats, labels, class_names, source)


def probe(model="phikon_v2", split="train", pos="TUM", neg="LYM"):
    """Derive the concept direction and report linear-probe separability."""
    result, _ = _battery(model, split, pos, neg, n_null=1)
    return {"concept": f"{pos}_vs_{neg}", "model": model, "probe": result["probe"]}


def ablate(model="phikon_v2", split="train", pos="TUM", neg="LYM", n_null=200):
    """Necessity (readout space) + matched-random null."""
    result, _ = _battery(model, split, pos, neg, n_null)
    return {"concept": f"{pos}_vs_{neg}", "model": model,
            "necessity_readout": result["necessity_readout"],
            "caveat": "readout-space projection only; layer-resolved curve is track #3"}


def specificity(model="phikon_v2", split="train", pos="TUM", neg="LYM",
                distractor_pos="STR", distractor_neg="MUS", n_null=1):
    """Ablate an orthogonal distractor axis; target probe should stay intact."""
    feats, labels, class_names, _ = loader.load(model, split)
    result = battery.run_battery(feats, labels, class_names, pos=pos, neg=neg,
                                 distractor=(distractor_pos, distractor_neg),
                                 n_null=n_null)
    return {"concept": f"{pos}_vs_{neg}", "model": model,
            "specificity": result["specificity"]}


def steer(model="phikon_v2", split="train", pos="TUM", neg="LYM", n_null=200):
    """Sufficiency: inject the concept direction to flip neg->pos vs random null."""
    result, _ = _battery(model, split, pos, neg, n_null)
    return {"concept": f"{pos}_vs_{neg}", "model": model,
            "sufficiency_steering": result["sufficiency_steering"]}


def confound_verb(model="phikon_v2", split="train", pos="TUM", neg="LYM"):
    """Confound gate — site/scanner-probe alignment on the causal axis.

    Returns 'no_multisite_data' until track #2 lands >=2-site data (NCT-CRC is
    single-source, so there is no honest site signal to test yet).
    """
    feats, labels, class_names, _ = loader.load(model, split)
    return {"concept": f"{pos}_vs_{neg}", "model": model,
            "confound_gate": confound.confound_gate(
                feats, labels, class_names, site_labels=None, pos=pos, neg=neg)}


def certify(model="phikon_v2", split="train", pos="TUM", neg="LYM", n_null=200):
    """Full evidence card: all pillars + confound gate + literature + caveat."""
    result, (feats, labels, class_names, source) = _battery(model, split, pos, neg, n_null)
    confound_result = confound.confound_gate(
        feats, labels, class_names, site_labels=None, pos=pos, neg=neg)
    intervene_report = intervene.pending_report()
    return card_mod.build_evidence_card(
        battery_card=result, confound_result=confound_result,
        intervene_report=intervene_report, model_key=model, split=split,
        source=source, pos=pos, neg=neg)
