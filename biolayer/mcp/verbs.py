"""Verb implementations — the plumbing behind the MCP tools.

Each verb loads the frozen embeddings and returns a JSON-able slice of the
evidence card. `certify` orchestrates all pillars into the full card, and can be
driven either by (model, pos, neg) directly or by a track name (which supplies
the model + objective + distractor for you). Kept free of any FastMCP import so
the verbs are unit-testable; server.py is the thin adapter.
"""
import numpy as np

from .. import tracks
from ..causal import attribution, battery, confound, intervene
from ..causal import probe as _probe
from ..data import loader
from . import card as card_mod

DEFAULT_DISTRACTOR = ("STR", "MUS")


def _resolve(track, model, pos, neg):
    """A track name fills in model + concept + distractor; else use the args."""
    if track is not None:
        t = tracks.get(track)
        return t.model_key, t.objective.pos, t.objective.neg, t.objective.distractor
    return model, pos, neg, DEFAULT_DISTRACTOR


def hypothesis(track="phikon", split="train"):
    """Workflow entry point: state the causal hypothesis a track will certify.

    Returns the concept + distractor + rationale and the ordered pipeline of verbs
    (probe -> ablate -> steer -> specificity -> layered -> attribution -> confound
    -> certify) so each role knows what to run. No model call — pure planning.
    """
    t = tracks.get(track)
    o = t.objective
    return {
        "track": t.name, "model": t.model_key, "dataset": t.dataset_slug,
        "hypothesis": f"{o.pos} vs {o.neg} is a concept-specific, certifiable causal "
                      f"axis in {t.model_key}'s latent — {o.description}",
        "concept": [o.pos, o.neg], "distractor": list(o.distractor),
        "layers": list(t.layers),
        "pipeline": ["probe", "ablate", "steer", "specificity", "layered",
                     "attribution", "confound", "certify"],
        "falsifier": "a matched-random direction must NOT reproduce any effect; "
                     "if it does, the certificate is void (Section-5-D control)",
        "citations": card_mod.CITATIONS,
    }


def attribution_verb(model="phikon_v2", split="train", pos="TUM", neg="LYM",
                     mode="soft", patch_npz=None, n_null=200, track=None):
    """Patch-level 'hack': which patches build the concept-carrying global.

    Derives the concept axis from cached embeddings. If a per-patch grid is
    available (`patch_npz` with a `patch_tokens` (N,P,D) array) it runs the full
    attribution card on the first tile; otherwise it returns the concept axis and a
    'needs_patch_grid' status (per-patch grids aren't in the cached npz yet).
    """
    model, pos, neg, _ = _resolve(track, model, pos, neg)
    feats, labels, class_names, source = loader.load(model, split)
    Xp, y = _probe.select_pair(feats, labels, class_names, pos, neg)
    concept_dir = _probe.diff_of_means(Xp, y)  # raw-space unit concept axis

    if patch_npz is None:
        return {"concept": f"{pos}_vs_{neg}", "model": model,
                "status": "needs_patch_grid",
                "note": ("per-patch grids are not cached (npz stores mean-patch 'local'). "
                         "Provide patch_npz with patch_tokens (N,P,D), or use the live "
                         "hack_tile forward. Core attribution is ready."),
                "concept_dir_norm": float(np.linalg.norm(concept_dir))}

    grid = np.load(patch_npz, allow_pickle=True)["patch_tokens"][0]  # (P, D)
    report = attribution.attribution_report(grid, concept_dir, mode=mode, n_null=n_null)
    return {"concept": f"{pos}_vs_{neg}", "model": model, "status": "ok",
            "attribution": report}


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
