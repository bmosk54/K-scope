"""AutoResearch — an autonomous causal-circuit discovery loop over the MCP battery.

Given a free-form pathology PROBLEM, this runs a closed generate -> certify -> locate
-> ablate -> reflect loop, entirely on the existing verbs/battery:

  1. DESIGN   an initial probe for the problem (Sonnet proposes a pos/neg/distractor
              contrast over the substrate's real classes; heuristic fallback if no Bedrock).
  2. CERTIFY  it — the deterministic causal battery (necessity / sufficiency / specificity
              vs a matched-random null) returns a universal score + per-pillar verdicts.
  3. LOCATE   *where in the circuit* the concept becomes causal: the layer-resolved
              necessity curve (project the concept axis out at each depth) tells us which
              layer's activations the readout actually depends on.
  4. ABLATE   turn those activations off at the load-bearing layer and report the
              readout collapse vs the matched-random null (the intervention result).
  5. REFLECT  read the score + reasoning trace, diagnose the weakest pillar, and propose
              the NEXT hypothesis / probe — which becomes the next iteration's input.

The loop never lets the LLM decide certifiability; the battery does. It streams one
record per iteration so a UI can watch the investigation unfold and stop it any time.
"""
from ..causal import certify as _certify
from ..data import loader
from . import probe_design as _pd


def _circuit_from_card(card):
    """Distill the certificate's layer-resolved necessity into a per-layer causal circuit.

    Each node: probe separability at that depth, accuracy after projecting the concept
    axis out, the matched-random baseline, the necessity gap (random - concept), and
    whether ablating the axis there *bites* (drops the readout below the random null).
    Falls back to a single readout node from the necessity pillar if no layer curve.
    """
    layered = card.get("necessity_layered") or {}
    curve = layered.get("curve") or []
    nodes = []
    for c in curve:
        base = c.get("probe_acc", 0.0)
        concept_abl = c.get("concept_ablated_acc", base)
        rand = c.get("random_ablated_acc_mean", base)
        gap = rand - concept_abl
        nodes.append({
            "layer": str(c.get("layer")),
            "probe_acc": round(float(base), 4),
            "concept_ablated_acc": round(float(concept_abl), 4),
            "random_ablated_acc": round(float(rand), 4),
            "necessity_gap": round(float(gap), 4),
            "bites": bool(gap > 0.02),
        })
    if not nodes:  # single-layer npz -> synthesize a readout node from the necessity pillar
        nec = (card.get("pillars", {}) or {}).get("necessity") or {}
        base = (card.get("probe", {}) or {}).get("test_acc", 1.0)
        concept_abl = nec.get("statistic", base)
        rand = nec.get("null", base)
        gap = rand - concept_abl
        nodes.append({
            "layer": "readout", "probe_acc": round(float(base), 4),
            "concept_ablated_acc": round(float(concept_abl), 4),
            "random_ablated_acc": round(float(rand), 4),
            "necessity_gap": round(float(gap), 4), "bites": bool(gap > 0.02)})
    return nodes


def _load_axis(nodes):
    """The load-bearing layer = the one where turning the concept axis off bites hardest."""
    biting = [n for n in nodes if n["bites"]] or nodes
    return max(biting, key=lambda n: n["necessity_gap"])


def _live_circuit(model_key, pos_class, n_null=8, n_tiles=10):
    """LIVE dynamic-intervention circuit: edit THIS concept's axis out at each layer on a
    real forward pass and measure the graded readout collapse vs a matched-random null.

    Unlike the cached readout-space projection (flat, near-tautological), this shows the
    *graded* circuit — early layers get recomputed downstream (redundancy), necessity
    concentrates toward the readout. Returns cached-shaped nodes, or None on any failure
    (no GPU / no HF token / substrate not hook-capable) so the loop falls back to cached.
    """
    import numpy as np
    from .. import config, serving
    try:
        cls_names = config.CLASS_NAMES
        if pos_class not in cls_names:
            return None
        pre = serving.precompute_slide(model_key)
        ref = serving.reference(model_key)
        labs = np.asarray(ref["labels"])
        ci = cls_names.index(pos_class)
        idx = np.where(labs == ci)[0][:n_tiles]
        if len(idx) < 3:
            return None
        slide = [ref["images"][i] for i in idx]
        res = serving.certify_slide(model_key, slide_images=slide, concepts=[pos_class],
                                    n_null=n_null, pre=pre)
        curve = (res.get("per_concept_necessity", {}).get(pos_class, {}) or {}).get("curve", [])
        if not curve:
            return None
        nodes = []
        for c in curve:
            gap = float(c.get("necessity_gap", 0.0))
            nodes.append({
                "layer": f"L{c.get('block', 0) + 1}",
                "probe_acc": float(c.get("base_P", 0.0)),
                "concept_ablated_acc": float(c.get("concept_ablated_P", 0.0)),
                "random_ablated_acc": float(c.get("random_ablated_P_mean", 0.0)),
                "necessity_gap": round(gap, 4),
                "z": c.get("z"),
                "bites": bool(c.get("bites", False)),
            })
        return nodes
    except Exception:
        return None


def _first_probe(problem, class_names, use_bedrock):
    """Design the opening probe for the problem (Sonnet), else a registry tissue contrast."""
    if use_bedrock:
        try:
            probes = _pd.design_probes(problem, class_names, max_probes=6)
            if probes:
                p = probes[0]
                return {"concept": p["concept"], "pos": p["pos"], "neg": p["neg"],
                        "distractor": list(p["distractor"]), "rationale": p.get("rationale", ""),
                        "proposed_by": "claude-sonnet (design)"}
        except Exception:
            pass
    # Deterministic opener: tumor-vs-immune, the substrate's cleanest causal axis.
    cset = set(class_names)
    pos = "TUM" if "TUM" in cset else list(class_names)[0]
    neg = "LYM" if "LYM" in cset else next(c for c in class_names if c != pos)
    return {"concept": f"{pos}_vs_{neg}", "pos": pos, "neg": neg,
            "distractor": ["STR", "MUS"], "rationale": "cleanest tumor/immune causal axis",
            "proposed_by": "heuristic (no bedrock)"}


def _valid_probe(probe, class_names):
    cset = set(class_names)
    return (isinstance(probe, dict) and probe.get("pos") in cset
            and probe.get("neg") in cset and probe.get("pos") != probe.get("neg"))


def _fresh_probe(class_names, seen):
    """Pick a contrast whose concept has NOT been certified yet, so the loop keeps
    exploring instead of re-running the same axis. Anchors on TUM (the cleanest positive)
    and walks unseen foils; if TUM is exhausted, rotates the positive; only if the whole
    contrast space is seen does it fall back to the default opener (bounded by max_iters).
    """
    cset = [str(c) for c in class_names]
    anchor = "TUM" if "TUM" in cset else cset[0]
    order = [anchor] + [c for c in cset if c != anchor]
    for p in order:
        for neg in [c for c in cset if c not in (p, "BACK")]:
            if f"{p}_vs_{neg}" not in seen:
                return {"concept": f"{p}_vs_{neg}", "pos": p, "neg": neg,
                        "distractor": ["STR", "MUS"],
                        "rationale": f"unexplored foil for {p}",
                        "proposed_by": "auto (fresh foil)"}
    return _first_probe("", class_names, use_bedrock=False)


def iterate(problem, track="phikon", max_iters=5, n_null=60, use_bedrock=True,
            split="train", converge_score=0.95, live=False):
    """Yield one research record per iteration of the autonomous causal loop.

    Stops after `max_iters`, or early once two consecutive iterations both land a
    GROUNDED verdict above `converge_score` (the circuit is nailed down). With `live=True`
    the circuit is measured by a real per-slide source-intervention (graded, honest);
    it falls back to the cached readout-space curve if the live path is unavailable.
    """
    from .. import tracks as _tracks
    model = _tracks.get(track).model_key
    feats, labels, class_names, source = loader.load(model, split)

    probe = _first_probe(problem, class_names, use_bedrock)
    seen, strong_streak = set(), 0

    for i in range(max_iters):
        if not _valid_probe(probe, class_names):
            probe = _first_probe(problem, class_names, use_bedrock=False)
        pos, neg = str(probe["pos"]), str(probe["neg"])
        dist = tuple(str(d) for d in (probe.get("distractor") or ["STR", "MUS"])[:2])
        seen.add(f"{pos}_vs_{neg}")

        card = _certify.certify(feats, labels, class_names, pos, neg, dist, model,
                                split, source, n_null=n_null,
                                artifacts_dir=loader.ARTIFACTS_DIR)

        conf = card.get("confidence", {}) or {}
        score = float(conf.get("overall") or 0.0)
        # static certify pillars carry a `passed` bool + a numeric confidence; the
        # descriptive `verdict` is prose, so roll up GROUNDED/WEAK/NULL from passed+score.
        pillars = {k: {"confidence": round(float(v.get("confidence") or 0.0), 3),
                       "passed": bool(v.get("passed")),
                       "verdict": "PASS" if v.get("passed") else "—"}
                   for k, v in (card.get("pillars", {}) or {}).items() if v}
        circuit = None
        circuit_mode = "cached"
        if live:
            circuit = _live_circuit(model, pos)
            if circuit:
                circuit_mode = "live"
        if not circuit:
            circuit = _circuit_from_card(card)
        axis = _load_axis(circuit)
        suf_ok = pillars.get("sufficiency", {}).get("passed", False)
        spec_ok = pillars.get("specificity", {}).get("passed", False)
        integrity = bool(conf.get("null_integrity", True))
        verdict = ("GROUNDED" if (suf_ok and spec_ok and integrity and score >= 0.5)
                   else "WEAK" if score > 0.0 else "NULL")

        # Reflect: read this card's score + trace, propose the next hypothesis/probe.
        reflection = _pd.next_hypothesis(card, class_names, use_bedrock=use_bedrock)
        nxt = reflection.get("proposed_probe") or {}
        # don't loop on the same axis twice in a row — nudge to an unseen foil if repeated
        if _valid_probe(nxt, class_names) and f"{nxt['pos']}_vs_{nxt['neg']}" not in seen:
            next_probe = {"concept": nxt.get("concept", f"{nxt['pos']}_vs_{nxt['neg']}"),
                          "pos": nxt["pos"], "neg": nxt["neg"],
                          "distractor": nxt.get("distractor", ["STR", "MUS"]),
                          "rationale": nxt.get("rationale", ""),
                          "proposed_by": reflection.get("proposed_by", "reflection")}
        else:
            # proposed probe was invalid or already certified — pick an unexplored
            # contrast so we never re-run the same axis two iterations running.
            next_probe = _fresh_probe(class_names, seen)

        yield {
            "iter": i + 1, "max_iters": max_iters, "problem": problem, "track": track,
            "concept": card.get("prediction", {}).get("concept"),
            "contrast": {"pos": pos, "neg": neg, "distractor": list(dist)},
            "hypothesis": probe.get("rationale") or probe.get("concept"),
            "proposed_by": probe.get("proposed_by"),
            "verdict": verdict, "score": round(score, 4),
            "pillars": pillars,
            "circuit": circuit, "circuit_mode": circuit_mode,
            "ablation": {
                "layer": axis["layer"],
                "probe_acc": axis["probe_acc"],
                "ablated_acc": axis["concept_ablated_acc"],
                "random_acc": axis["random_ablated_acc"],
                "necessity_gap": axis["necessity_gap"], "bites": axis["bites"],
                "note": (f"turned off the {card.get('prediction', {}).get('concept')} axis @ "
                         f"{axis['layer']}: probe {axis['probe_acc']:.3f} -> "
                         f"{axis['concept_ablated_acc']:.3f} (random null {axis['random_ablated_acc']:.3f})"),
            },
            "diagnosis": reflection.get("diagnosis"),
            "weakest_pillar": reflection.get("weakest_pillar"),
            "next_hypothesis": reflection.get("next_hypothesis"),
            "next_probe": {"pos": str(next_probe["pos"]), "neg": str(next_probe["neg"]),
                           "distractor": [str(d) for d in next_probe.get("distractor", [])]},
            "message_to_downstream": reflection.get("message_to_downstream"),
        }

        strong_streak = strong_streak + 1 if (verdict == "GROUNDED" and score >= converge_score) else 0
        if strong_streak >= 2:
            yield {"iter": i + 1, "done": True, "reason": "converged",
                   "note": f"two consecutive GROUNDED verdicts >= {converge_score}; circuit nailed"}
            return
        probe = next_probe

    yield {"done": True, "reason": "max_iters", "note": f"reached {max_iters} iterations"}
