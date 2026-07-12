"""Per-claim reasoning trace — the auditable "why this score, why this verdict".

DETERMINISTIC and instant: every step is assembled from numbers the battery already
computed (no LLM, no latency). Each step carries the observed value(s) AND the
interpretation, so a governance reviewer can step through probe -> contrast validation
-> necessity -> sufficiency -> specificity -> confound -> multiple-comparisons ->
verdict and see exactly what each number means and how it drove the call.

An OPTIONAL plain-English narration (explain=True in certify_answer) turns the whole
card's traces into short prose in ONE batched LLM call — it never gates the fast path.
"""


def _s(n, step, observation, interpretation):
    # Standardized schema: carry BOTH `step` and `pillar` (same value) so a UI can bind
    # to either key regardless of which certify verb produced the trace.
    return {"n": n, "step": step, "pillar": step, "observation": observation,
            "interpretation": interpretation}


def build_claim_trace(sc, cs, live, conf, bc):
    """sc = ClaimScore (pillars, verdict, ...), cs = ContrastSet, live = live_necessity
    dict|None, conf = confound_gate dict, bc = raw battery card (for raw stats)."""
    nec = sc.pillars["necessity"]
    suf = sc.pillars["sufficiency"]
    spec = sc.pillars["specificity"]
    s_raw = bc.get("sufficiency_steering", {})
    sp_raw = bc.get("specificity", {})
    sub = bc.get("substrate", {})
    p_raw = bc.get("probe", {})
    nr = bc.get("necessity_readout", {})
    dim = sub.get("dim")                                   # CLS width (1024 phikon / 1536 h-opt)
    nnull = sub.get("n_null")                              # matched-random directions per test
    dpos, dneg = (cs.distractor if getattr(cs, "distractor", None) else (None, None))
    alpha = s_raw.get("alpha_classwidth")
    t = []

    # 1. probe / contrast validation ---------------------------------------
    # cheap screen, then (if it fires) the controlled re-test — report both numbers.
    _adj = getattr(cs, "confound_adjudication", "screen-clean")
    intensity_txt = f"intensity screen |r|={cs.intensity_collinearity:.3f}"
    if getattr(cs, "intensity_suspect", False):
        intensity_txt += (f" -> SUSPECT; controlled re-test (intensity-matched): "
                          f"AUROC={cs.matched_auroc:.3f}, |r|={cs.matched_intensity_collinearity:.3f}"
                          f", n={cs.n_matched} -> {_adj}")
    verdict = ("PASS" if cs.valid else "WARN: " + "; ".join(cs.warnings))
    if cs.valid and getattr(cs, "flags", ()):
        verdict += " (FLAGGED: " + "; ".join(cs.flags) + ")"
    dimtxt = f"{dim}-d " if dim else ""
    ntr, nte = p_raw.get("n_train"), p_raw.get("n_test")
    n_txt = (f"n={cs.n_pos}(+)/{cs.n_neg}(−) tiles"
             + (f", {ntr} fit / {nte} held-out" if ntr and nte else ""))
    t.append(_s(1, "linear_probe + contrast_validation",
                f"fit a linear probe — L2 logistic regression on the standardized {dimtxt}"
                f"frozen CLS — for {cs.pos}(+) vs {cs.neg}(−), {n_txt}. held-out AUROC="
                f"{cs.heldout_auroc:.3f}. nuisance {intensity_txt} -> {verdict}",
                ("the concept axis is the probe's unit decision direction in standardized "
                 "CLS space; the pool separates the concept and, where intensity was "
                 "suspected, the signal survives a controlled intensity-matched re-test"
                 if cs.valid else
                 "the separation does not survive controlling for intensity, or could not "
                 "be adjudicated — treat with caution")))

    # 2. necessity (live source-intervention, or cached readout-space) -----
    if isinstance(live, dict) and live.get("curve"):
        curve = live["curve"]
        blocks = ", ".join(str(c.get("block_idx", c.get("block"))) for c in curve)
        series = " -> ".join(f"{c['layer']}:{c['necessity_gap']:+.2f}(z{c['gap_vs_null_z']:+.0f})"
                             for c in curve)
        nn = live.get("n_null", nnull)
        watched = live.get("n_watched")
        t.append(_s(2, "necessity · live source-intervention",
                    f"for {len(curve)} depths, hooked encoder.layer[{blocks}] and projected "
                    f"the diff-of-means concept axis OUT of the CLS token in the residual "
                    f"stream, then let the forward pass recompute to the readout. margin-drop "
                    f"vs {nn} matched-random unit directions ablated identically"
                    + (f", measured on {watched} readout-positive tiles of this slide" if watched else "")
                    + f": {series} (null ~0). necessity score={nec.score:.3f} = fraction of "
                    f"readout-necessity already irreversible before the readout.",
                    ("the model's decision on THIS tile causally depends on the concept from "
                     "mid-network on; early-layer ablation is recomputed downstream "
                     "(redundancy/Hydra). intervened_on_input=true — a per-slide causal read")))
    else:
        base_a, abl_a = nr.get("base_acc"), nr.get("concept_ablated_acc")
        rm, rs = nr.get("random_ablated_acc_mean"), nr.get("random_ablated_acc_std")
        detail = (f"probe accuracy {base_a:.3f} -> {abl_a:.3f} (chance 0.50); "
                  f"{nnull} matched-random unit axes projected out identically left it at "
                  f"{rm:.3f}±{rs:.3f}." if base_a is not None else "")
        t.append(_s(2, "necessity · readout-space projection",
                    f"projected the unit concept axis OUT of the final {dimtxt}CLS (post-hoc, "
                    f"no forward re-pass). {detail} necessity score={nec.score:.3f}, "
                    f"verdict={nec.verdict}.",
                    ("redundancy-limited: mid-layer ablation is recomputed downstream, so only "
                     "the readout bites — near-tautological. pass a slide (live_ctx) for the "
                     "real per-slide source-intervention through the network")))

    # 3. sufficiency (caveated secondary) ----------------------------------
    at = f"α={alpha:.2f} (one inter-class 'class-width' along the axis)" if alpha is not None else "α=class-width"
    t.append(_s(3, "sufficiency · steering / injection",
                f"added the concept direction to held-out {cs.neg} CLS vectors, {at}: "
                f"{s_raw.get('concept_flip_rate', 0):.2f} crossed the probe boundary to "
                f"{cs.pos}, vs {s_raw.get('random_flip_rate_mean', 0):.2f}±"
                f"{s_raw.get('random_flip_rate_std', 0):.2f} for {nnull} matched-random unit "
                f"directions injected at the same α. score={suf.score:.3f}.",
                ("a concept-specific steering axis, de-circularized by the matched-random "
                 "null (random directions injected at the same α do not flip) — but on its "
                 "own still near-circular (inject the class-mean-diff axis, score a probe "
                 "built on it), so it can NOT ground a GROUNDED verdict alone: a genuine "
                 "necessity (live or non-readout) plus specificity must carry it")))

    # 4. specificity -------------------------------------------------------
    cos = sp_raw.get("cos_with_concept_axis")
    ta, ba = sp_raw.get("target_acc_after_distractor_ablation"), sp_raw.get("base_acc")
    dtxt = f"{dpos} vs {dneg}" if dpos else "an orthogonal pair"
    t.append(_s(4, "specificity · distractor control",
                f"fit a SEPARATE linear probe for the distractor pair {dtxt}, expressed its "
                f"axis in the concept probe's space, and projected THAT out"
                + (f". cos(concept axis, distractor axis)={cos:.3f} (≈orthogonal)" if cos is not None else "")
                + (f"; concept probe accuracy after the distractor ablation {ta:.3f} (base {ba:.3f})"
                   if ta is not None else "")
                + f" -> target {'intact' if spec.verdict != 'NULL' else 'leaked'}. score={spec.score:.3f}.",
                ("the effect is targeted to the concept axis, not general damage — ablating a "
                 "near-orthogonal distractor leaves the concept probe intact")))

    # 5. confound gate -----------------------------------------------------
    cf = conf.get("confound_gate", conf) if isinstance(conf, dict) else {}
    if cf.get("status") == "ok":
        t.append(_s(5, "confound · site-probe alignment",
                    f"fit a linear site/scanner probe (Kömen-style) and measured its axis "
                    f"against the concept axis: cos(concept, site)="
                    f"{cf.get('cos_concept_with_site', 0):.3f} (flag threshold "
                    f"{cf.get('threshold_cos', 0.30)}) -> {'FLAG' if cf.get('confounded') else 'PASS'}.",
                    ("the causal axis overlaps a scanner/site signature — possible batch "
                     "artifact" if cf.get('confounded') else
                     "the causal axis is not aligned with the site signature")))
    else:
        t.append(_s(5, "confound · site-probe alignment",
                    "UNCHECKED — single-source data (no site/scanner variation to fit a site "
                    "probe against). With >=2 sites: fit a linear scanner/site probe and flag "
                    "if cos(concept axis, site axis) > 0.30.",
                    ("cannot rule out that this is a batch/scanner artifact rather than "
                     "biology; biological validity is NOT established here")))

    # 6. multiple comparisons ----------------------------------------------
    t.append(_s(6, "multiple_comparisons · Holm-Bonferroni",
                f"the claim's smallest matched-random-null p-value across pillars = "
                f"{sc.min_p:.4f}; Holm-Bonferroni step-down over the certifiable claims in "
                f"this answer -> {'survives' if sc.survives_correction else 'does NOT survive'}.",
                ("the effect is not a cherry-pick from probing many concepts per answer"
                 if sc.survives_correction else
                 "downgraded: does not survive correction for the many claims tested")))

    # 7. verdict -----------------------------------------------------------
    caps = []
    if getattr(sc, "contrast_capped", False):
        caps.append("contrast failed the validation gate")
    if getattr(sc, "necessity_capped", False):
        caps.append("no genuine necessity (readout-only, near-tautological) — near-circular "
                    "sufficiency alone can't ground")
    if sc.confounded and sc.verdict != "GROUNDED":
        caps.append("site-confound flag")
    cap_str = f" (CAPPED at WEAK: {', '.join(caps)})" if caps else ""
    t.append(_s(7, "verdict", f"-> {sc.verdict}{cap_str}", _verdict_reason(sc)))
    return t


def _verdict_reason(sc):
    nv = sc.pillars["necessity"].verdict
    sv = sc.pillars["sufficiency"].verdict
    pv = sc.pillars["specificity"].verdict
    if sc.verdict == "GROUNDED":
        return (f"necessity={nv}, sufficiency={sv}, specificity={pv}, survives correction, "
                f"contrast passed the gate, not confounded -> the claim is causally "
                f"load-bearing on this substrate")
    if sc.verdict == "NULL":
        return "no effect above the matched-random null on the load-bearing pillars"
    if getattr(sc, "contrast_capped", False):
        return ("the pillars separate, but the contrast RIDES the staining/intensity proxy "
                "(failed the validation gate) — so the axis is not certified as clean "
                "biology; capped at WEAK. A probe that rides intensity never reads GROUNDED")
    tail = " and capped at WEAK by a site-confound flag" if sc.confounded else ""
    return (f"necessity={nv}, sufficiency={sv}, specificity={pv}: a positive but not fully "
            f"significant / redundancy-limited signal{tail}")


def summary_trace(coverage, conf_badge, declined):
    """Answer-level trace: coverage reasoning + what was declined and why."""
    steps = [_s(1, "coverage",
                coverage["summary"],
                ("claims of different epistemic status were asserted in identical prose; "
                 "the tool separates the testable from the untestable"))]
    if declined:
        why = "; ".join(f"{d.get('concept') or d['text'][:24]}: {d['reason']}" for d in declined[:6])
        steps.append(_s(2, "declined", why,
                        "declined rather than force-fit — an honest NOT_CERTIFIABLE, "
                        "including any claim that would change treatment"))
    steps.append(_s(len(steps) + 1, "confound_badge", conf_badge,
                    "the one honest hole named up front; biological validity beyond "
                    "model-internal use rests on this + encoder faithfulness"))
    return steps


# --------------------------------------------------------------------------
# Optional batched narration (one LLM call for the whole card; off by default)
# --------------------------------------------------------------------------
def narrate(card):
    """Turn the deterministic traces into short prose in ONE Bedrock call. Returns
    {per_claim: {concept: sentence}, overall: str} or a status dict on failure. Only
    called when certify_answer(explain=True); never on the fast path."""
    from . import bedrock as _bedrock
    client = _bedrock.ClaudeBedrock()
    if not client.available():
        return {"status": "unavailable", "note": client._err}
    compact = {
        "question": card.get("prompt"),
        "coverage": card["coverage"]["summary"],
        "confound": card["guardrails"]["confound_gate"],
        "claims": [{"concept": c["concept"], "verdict": c["verdict"],
                    "scores": c["scores"],
                    "trace": [f"{s['step']}: {s['observation']}" for s in c["reasoning_trace"]]}
                   for c in card["claims"]],
    }
    try:
        return client.narrate(compact)
    except Exception as e:
        return {"status": "error", "note": repr(e)}
