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
from ..causal import certify as certify_mod
from ..causal import probe as _probe
from ..data import loader
from ..dynamic import certify_answer as _certify_answer
from ..dynamic import probe_design as _probe_design

DEFAULT_DISTRACTOR = ("STR", "MUS")


def certify_answer(prompt, answer, track="phikon", split="train", n_null=200,
                   fast=False, use_bedrock=True, live_ctx=None, explain=False):
    """Dynamic, answer-bound certification of a free-form K-Pro answer.

    Decomposes the answer into atomic concept-claims (tissue labels + HistoPLUS cell
    types), certifies each against the substrate it resolves to with the full
    do()-battery + matched-random null + specificity + confound gate + held-out
    validation + Holm-Bonferroni correction, and returns numeric per-pillar scores
    plus a GROUNDED/WEAK/NULL verdict per claim + coverage. Claims with no substrate
    label (or whose label source has no embeddings yet) are NOT_CERTIFIABLE, not
    force-fit.
    """
    return _certify_answer(prompt, answer, track=track, split=split,
                           n_null=n_null, fast=fast, use_bedrock=use_bedrock,
                           live_ctx=live_ctx, explain=explain)


def warmup(model="phikon_v2"):
    """Prime the warm inference backend: load the frozen encoder + reference set +
    population embeddings ONCE so live certification is served hot (no per-call weight
    reload or re-download). Call at server startup."""
    from .. import serving
    return serving.warmup(model)


def serving_status():
    """What the warm backend currently holds resident (encoders, reference sets, RAM
    embeddings)."""
    from .. import serving
    return serving.status()


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
        "citations": certify_mod.CITATIONS,
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
    """Full modular certificate: pillars + universal confidence + reasoning trace +
    confound + reusable steer/ablate handles + literature."""
    model, pos, neg, dist = _resolve(track, model, pos, neg)
    feats, labels, class_names, source = loader.load(model, split)
    return certify_mod.certify(
        feats, labels, class_names, pos, neg, dist, model, split, source,
        n_null=n_null, artifacts_dir=loader.ARTIFACTS_DIR)


# --------------------------------------------------------------------------
# Sonnet-driven hypothesis generation + the closed certify loop
# --------------------------------------------------------------------------
def design(question, model="phikon_v2", split="train", track=None, max_probes=8,
           certify_each=False, n_null=200):
    """Let Sonnet DESIGN a causal probe battery for a free-form pathology question, then
    optionally certify each proposed probe (generate-then-certify).

    The LLM proposes (pos, neg, distractor) contrasts over the substrate's real tissue
    classes; the deterministic battery decides certifiability, so an ill-posed probe is
    never certified. This is the generative counterpart to the static `hypothesis` verb.
    """
    if track is not None:
        model = tracks.get(track).model_key
    feats, labels, class_names, source = loader.load(model, split)
    try:
        probes = _probe_design.design_probes(question, class_names, max_probes=max_probes)
    except Exception as e:  # bedrock unavailable / bad model response — degrade, don't crash
        return {"verb": "design", "status": "unavailable", "question": question,
                "model": model, "available_classes": list(class_names),
                "note": f"probe design needs Claude-on-Bedrock and it is unavailable: {e}"}
    out = {"verb": "design", "question": question, "model": model, "split": split,
           "designed_by": "claude-sonnet (bedrock)", "n_probes": len(probes),
           "probes": probes,
           "note": "LLM proposes contrasts; the battery decides certifiability"}
    if certify_each:
        certs = []
        for p in probes:
            dist = tuple(p.get("distractor") or DEFAULT_DISTRACTOR)
            card = certify_mod.certify(feats, labels, class_names, p["pos"], p["neg"],
                                       dist, model, split, source, n_null=n_null,
                                       artifacts_dir=loader.ARTIFACTS_DIR)
            certs.append({
                "concept": card["prediction"]["concept"],
                "score": card["confidence"]["overall"],
                "pillars": {k: v["verdict"] for k, v in card["pillars"].items() if v},
                "reasoning_trace": card["reasoning_trace"]})
        out["certified"] = certs
    return out


def rehypothesize(model="phikon_v2", split="train", pos="TUM", neg="LYM", n_null=200,
                  track=None, certificate=None, use_bedrock=True):
    """Close the loop: certify a concept, read its SCORE + REASONING TRACE, and have Sonnet
    propose the NEXT hypothesis + a follow-up probe + a message to feed to K-Pro or another
    Claude.

    Pass an existing `certificate` dict to reflect without recomputing; otherwise a fresh
    certificate is produced. The proposed follow-up probe is validated against real classes;
    the battery still decides certifiability, so the reflected hypothesis carries no verdict.
    """
    model, pos, neg, _ = _resolve(track, model, pos, neg)
    if certificate is None:
        certificate = certify(model, split, pos, neg, n_null, track=track)
    _, _, class_names, _ = loader.load(model, split)
    nxt = _probe_design.next_hypothesis(certificate, class_names, use_bedrock=use_bedrock)
    return {
        "verb": "rehypothesize",
        "from_concept": certificate.get("prediction", {}).get("concept"),
        "certificate_score": certificate.get("confidence", {}).get("overall"),
        "reasoning_trace": certificate.get("reasoning_trace"),
        "next_hypothesis": nxt,
        "loop": "certify(score+trace) -> claude -> next_hypothesis "
                "-> feed to K-Pro/Claude -> certify",
    }


# --------------------------------------------------------------------------
# Gap-2 reuse path: steer / ablate from a card's persisted handles (no recompute)
# --------------------------------------------------------------------------
def _load_or_use(x, feats, labels, class_names, split, pos, neg, want):
    """Return (X, source_label): caller-supplied features, else the split's pos/neg rows."""
    if x is not None:
        return np.atleast_2d(np.asarray(x, dtype=float)), "caller-supplied"
    Xp, y = _probe.select_pair(feats, labels, class_names, pos, neg)  # y=1 for pos
    sel = Xp[y == (1 if want == "pos" else 0)]
    return sel, f"{pos if want == 'pos' else neg} rows from {split} (n={len(sel)})"


def _base_preds(handles, X):
    """Unsteered/unablated probe predictions on raw features via the persisted handles."""
    Z = (np.asarray(X, dtype=float) - handles["scaler_mean"]) / handles["scaler_scale"]
    return ((Z @ handles["coef"] + float(handles["intercept"])) > 0).astype(int)


def _probe_acc(handles, X, y):
    """Accuracy of the PERSISTED probe (scaler+coef) on current features — the staleness test."""
    return float((_base_preds(handles, X) == y).mean())


def _valid_handles(model, split, pos, neg, min_acc=0.7):
    """Load persisted direction handles and VERIFY they still classify the CURRENT
    embeddings above chance; if missing or stale (e.g. embeddings re-extracted since the
    card was written), recompute + re-persist so the reuse path never silently returns
    wrong numbers. Returns (handles, feats, labels, class_names, meta).
    """
    feats, labels, class_names, _ = loader.load(model, split)
    Xp, y = _probe.select_pair(feats, labels, class_names, pos, neg)
    meta = {"source": "persisted", "recompute": False}
    try:
        handles = certify_mod.load_direction(model, pos, neg, loader.ARTIFACTS_DIR)
        acc = _probe_acc(handles, Xp, y)
        if acc >= min_acc:
            meta["probe_acc"] = round(acc, 4)
            return handles, feats, labels, class_names, meta
        meta["stale_probe_acc"] = round(acc, 4)  # handles present but don't fit this data
    except (FileNotFoundError, OSError):
        meta["missing"] = True

    # Self-heal: refit the probe on the current embeddings and re-persist the handles.
    from ..causal.battery import run_battery
    _, handles = run_battery(feats, labels, class_names, pos=pos, neg=neg,
                             n_null=1, return_handles=True)
    certify_mod.persist_handles(handles, model, split, pos, neg, loader.ARTIFACTS_DIR)
    meta.update(source="refreshed", recompute=True,
                probe_acc=round(_probe_acc(handles, Xp, y), 4),
                note="persisted handles were missing/stale vs current embeddings; "
                     "recomputed + re-persisted")
    return handles, feats, labels, class_names, meta


def steer_from_card(x=None, model="phikon_v2", split="train", pos="TUM", neg="LYM",
                    alpha=None, track=None, max_report=64):
    """Sufficiency from a card's persisted direction handles (no probe refit on the fast
    path). Handles are validated against the current embeddings and self-healed if stale, so
    the result is always correct. Omit `x` to steer the split's `neg`-class rows as a demo.
    """
    model, pos, neg, _ = _resolve(track, model, pos, neg)
    try:
        handles, feats, labels, class_names, meta = _valid_handles(model, split, pos, neg)
    except (FileNotFoundError, OSError, ValueError) as e:
        return {"verb": "steer_from_card", "status": "unavailable",
                "concept": f"{pos}_vs_{neg}", "model": model,
                "note": f"no embeddings/handles obtainable for {model}/{split}: {e}"}
    X, src = _load_or_use(x, feats, labels, class_names, split, pos, neg, want="neg")
    preds, _ = certify_mod.apply_steer(X, handles, alpha=alpha)
    a = float(handles["alpha_classwidth"]) if alpha is None else float(alpha)
    return {"verb": "steer_from_card", "status": "ok", "recompute": meta["recompute"],
            "handles": meta, "concept": f"{pos}_vs_{neg}", "model": model, "input": src,
            "n": int(len(X)), "alpha": a, "flipped_to_pos_rate": float((preds == 1).mean()),
            "predictions": preds.tolist() if len(X) <= max_report else f"omitted (n>{max_report})",
            "note": "steered via persisted direction handles (Gap-2 reuse path, staleness-checked)"}


def ablate_from_card(x=None, model="phikon_v2", split="train", pos="TUM", neg="LYM",
                     track=None, max_report=64):
    """Necessity from a card's persisted direction handles (no probe refit on the fast path).
    Handles are validated against the current embeddings and self-healed if stale. Omit `x`
    to ablate the split's `pos`-class rows and report how many still read as `pos` (drops).
    """
    model, pos, neg, _ = _resolve(track, model, pos, neg)
    try:
        handles, feats, labels, class_names, meta = _valid_handles(model, split, pos, neg)
    except (FileNotFoundError, OSError, ValueError) as e:
        return {"verb": "ablate_from_card", "status": "unavailable",
                "concept": f"{pos}_vs_{neg}", "model": model,
                "note": f"no embeddings/handles obtainable for {model}/{split}: {e}"}
    # Necessity is an ACCURACY-drop over BOTH classes (matches battery.necessity_readout):
    # ablating the exact probe axis collapses predictions to the intercept, so a pos-rate
    # on one class is uninformative. Use the labeled pos+neg set unless the caller gives x.
    if x is None:
        X, y = _probe.select_pair(feats, labels, class_names, pos, neg)
        src = f"{pos}+{neg} rows from {split} (n={len(X)})"
    else:
        X, y, src = np.atleast_2d(np.asarray(x, dtype=float)), None, "caller-supplied"
    abl_preds, _ = certify_mod.apply_ablate(X, handles)
    out = {"verb": "ablate_from_card", "status": "ok", "recompute": meta["recompute"],
           "handles": meta, "concept": f"{pos}_vs_{neg}", "model": model, "input": src,
           "n": int(len(X)),
           "note": "readout-space necessity via persisted handles; single-axis ablation is "
                   "redundancy-limited on pathology FMs — reported honestly (per STRATEGY)"}
    if y is not None:
        base_acc, abl_acc = _probe_acc(handles, X, y), float((abl_preds == y).mean())
        out.update(base_acc=round(base_acc, 4), ablated_acc=round(abl_acc, 4),
                   necessity_drop=round(base_acc - abl_acc, 4), chance=0.5)
    else:
        out.update(prediction_changed_rate=float((abl_preds != _base_preds(handles, X)).mean()),
                   predictions=abl_preds.tolist() if len(X) <= max_report else f"omitted (n>{max_report})")
    return out
