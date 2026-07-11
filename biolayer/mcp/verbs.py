"""Verb implementations — the plumbing behind the MCP tools.

Each verb loads the frozen embeddings and returns a JSON-able slice of the
evidence card. `certify` orchestrates all pillars into the full card, and can be
driven either by (model, pos, neg) directly or by a track name (which supplies
the model + objective + distractor for you). Kept free of any FastMCP import so
the verbs are unit-testable; server.py is the thin adapter.
"""
from .. import tracks
from ..causal import battery, confound, intervene
from ..data import loader
from . import card as card_mod

DEFAULT_DISTRACTOR = ("STR", "MUS")


def _resolve(track, model, pos, neg):
    """A track name fills in model + concept + distractor; else use the args."""
    if track is not None:
        t = tracks.get(track)
        return t.model_key, t.objective.pos, t.objective.neg, t.objective.distractor
    return model, pos, neg, DEFAULT_DISTRACTOR


def probe(model="phikon_v2", split="train", pos="TUM", neg="LYM", track=None):
    """Derive the concept direction and report linear-probe separability."""
    model, pos, neg, _ = _resolve(track, model, pos, neg)
    feats, labels, class_names, _ = loader.load(model, split)
    result = battery.run_battery(feats, labels, class_names, pos=pos, neg=neg, n_null=1)
    return {"concept": f"{pos}_vs_{neg}", "model": model, "probe": result["probe"]}


def ablate(model="phikon_v2", split="train", pos="TUM", neg="LYM", n_null=200, track=None):
    """Necessity (readout space) + matched-random null."""
    model, pos, neg, _ = _resolve(track, model, pos, neg)
    feats, labels, class_names, _ = loader.load(model, split)
    result = battery.run_battery(feats, labels, class_names, pos=pos, neg=neg, n_null=n_null)
    return {"concept": f"{pos}_vs_{neg}", "model": model,
            "necessity_readout": result["necessity_readout"],
            "caveat": "readout-space projection only; layer-resolved curve is `layered`"}


def specificity(model="phikon_v2", split="train", pos="TUM", neg="LYM",
                distractor_pos=None, distractor_neg=None, track=None):
    """Ablate an orthogonal distractor axis; the target probe should stay intact."""
    model, pos, neg, dist = _resolve(track, model, pos, neg)
    if distractor_pos and distractor_neg:
        dist = (distractor_pos, distractor_neg)
    feats, labels, class_names, _ = loader.load(model, split)
    result = battery.run_battery(feats, labels, class_names, pos=pos, neg=neg,
                                 distractor=dist, n_null=1)
    return {"concept": f"{pos}_vs_{neg}", "model": model,
            "specificity": result["specificity"]}


def steer(model="phikon_v2", split="train", pos="TUM", neg="LYM", n_null=200, track=None):
    """Sufficiency: inject the concept direction to flip neg->pos vs random null."""
    model, pos, neg, _ = _resolve(track, model, pos, neg)
    feats, labels, class_names, _ = loader.load(model, split)
    result = battery.run_battery(feats, labels, class_names, pos=pos, neg=neg, n_null=n_null)
    return {"concept": f"{pos}_vs_{neg}", "model": model,
            "sufficiency_steering": result["sufficiency_steering"]}


def layered(model="phikon_v2", split="train", pos="TUM", neg="LYM",
            space="global", n_null=200, track=None):
    """Layer-resolved necessity curve across the 3 extracted layers (global|local)."""
    model, pos, neg, _ = _resolve(track, model, pos, neg)
    return {"concept": f"{pos}_vs_{neg}", "model": model, "space": space,
            "necessity_layered": intervene.layered_curve(
                model, split, pos, neg, space=space, n_null=n_null)}


def confound_verb(model="phikon_v2", split="train", pos="TUM", neg="LYM", track=None):
    """Confound gate — site/scanner-probe alignment on the causal axis.

    Returns 'no_multisite_data' until multi-site data lands (track #2).
    """
    model, pos, neg, _ = _resolve(track, model, pos, neg)
    feats, labels, class_names, _ = loader.load(model, split)
    return {"concept": f"{pos}_vs_{neg}", "model": model,
            "confound_gate": confound.confound_gate(
                feats, labels, class_names, site_labels=None, pos=pos, neg=neg)}


def certify(model="phikon_v2", split="train", pos="TUM", neg="LYM", n_null=200, track=None):
    """Full evidence card: all pillars + layered curve + confound + literature."""
    model, pos, neg, dist = _resolve(track, model, pos, neg)
    feats, labels, class_names, source = loader.load(model, split)
    result = battery.run_battery(feats, labels, class_names, pos=pos, neg=neg,
                                 distractor=dist, n_null=n_null)
    confound_result = confound.confound_gate(
        feats, labels, class_names, site_labels=None, pos=pos, neg=neg)
    intervene_report = intervene.pending_report(model, split, pos, neg)
    return card_mod.build_evidence_card(
        battery_card=result, confound_result=confound_result,
        intervene_report=intervene_report, model_key=model, split=split,
        source=source, pos=pos, neg=neg)
