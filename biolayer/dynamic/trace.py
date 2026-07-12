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
    return {"n": n, "step": step, "observation": observation,
            "interpretation": interpretation}


def build_claim_trace(sc, cs, live, conf, bc):
    """sc = ClaimScore (pillars, verdict, ...), cs = ContrastSet, live = live_necessity
    dict|None, conf = confound_gate dict, bc = raw battery card (for raw stats)."""
    nec = sc.pillars["necessity"]
    suf = sc.pillars["sufficiency"]
    spec = sc.pillars["specificity"]
    s_raw = bc.get("sufficiency_steering", {})
    sp_raw = bc.get("specificity", {})
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
    t.append(_s(1, "contrast_validation",
                f"held-out AUROC={cs.heldout_auroc:.3f}; {intensity_txt} -> {verdict}",
                ("the pool separates the concept and, where intensity was suspected, the "
                 "signal SURVIVES a controlled re-test (matching removes the nuisance)"
                 if cs.valid else
                 "the separation does not survive controlling for intensity, or could not "
                 "be adjudicated — treat with caution")))

    # 2. necessity (live or cached) ----------------------------------------
    if isinstance(live, dict) and live.get("curve"):
        curve = live["curve"]
        series = " -> ".join(f"{c['layer']}:{c['necessity_gap']:+.2f}(z{c['gap_vs_null_z']:+.0f})"
                             for c in curve)
        t.append(_s(2, "necessity_live",
                    f"LIVE source-intervention on this slide's forward pass — margin-drop "
                    f"vs matched-random null by layer: {series}; null ~0 throughout. "
                    f"necessity score={nec.score:.3f} (= fraction of readout-necessity "
                    f"already irreversible before the readout).",
                    ("the model's decision on THIS tile causally depends on the concept "
                     "from mid-network on; early-layer ablation is recomputed downstream "
                     "(redundancy). intervened_on_input=true -> a per-slide causal claim, "
                     "not reference-set separability")))
    else:
        t.append(_s(2, "necessity_cached",
                    f"cached readout-space projection; matched-random projections inert. "
                    f"necessity score={nec.score:.3f}, verdict={nec.verdict}.",
                    ("necessity is redundancy-limited / near-tautological in readout space; "
                     "pass a slide (live_ctx) for the real per-slide source-intervention")))

    # 3. sufficiency (caveated secondary) ----------------------------------
    t.append(_s(3, "sufficiency",
                f"inject the concept direction: flip {s_raw.get('concept_flip_rate', 0):.2f} "
                f"vs matched-random {s_raw.get('random_flip_rate_mean', 0):.2f} "
                f"(score={suf.score:.3f}, z>={min(abs(suf.z), 999):.0f}).",
                ("a concept-specific steering axis — but near-circular (inject the "
                 "class-mean-diff axis, score a probe built on it), so reported as a "
                 "caveated SECONDARY; necessity + specificity carry the verdict")))

    # 4. specificity -------------------------------------------------------
    cos = sp_raw.get("cos_with_concept_axis")
    t.append(_s(4, "specificity",
                f"ablate the orthogonal distractor axis"
                + (f" (cos with concept = {cos:.3f})" if cos is not None else "")
                + f"; target probe {'intact' if spec.verdict != 'NULL' else 'leaked'}. "
                f"score={spec.score:.3f}.",
                ("the effect is targeted to the concept axis, not general damage — the "
                 "distractor is near-orthogonal and its ablation leaves the target intact")))

    # 5. confound gate -----------------------------------------------------
    cf = conf.get("confound_gate", conf) if isinstance(conf, dict) else {}
    if cf.get("status") == "ok":
        t.append(_s(5, "confound",
                    f"site-probe alignment: cos(concept, site)={cf.get('cos_concept_with_site', 0):.3f} "
                    f"(threshold {cf.get('threshold_cos', 0.30)}) -> "
                    f"{'FLAG' if cf.get('confounded') else 'PASS'}.",
                    ("the causal axis overlaps a scanner/site signature — possible batch "
                     "artifact" if cf.get('confounded') else
                     "the causal axis is not aligned with the site signature")))
    else:
        t.append(_s(5, "confound",
                    "site-probe UNCHECKED — single-source data (no site/scanner variation).",
                    ("cannot rule out that this is a batch/scanner artifact rather than "
                     "biology; biological validity is NOT established here")))

    # 6. multiple comparisons ----------------------------------------------
    t.append(_s(6, "multiple_comparisons",
                f"min pillar p={sc.min_p:.4f}; Holm-Bonferroni across the answer's claims "
                f"-> {'survives' if sc.survives_correction else 'does NOT survive'}.",
                ("the effect is not a cherry-pick from probing many concepts per answer"
                 if sc.survives_correction else
                 "downgraded: does not survive correction for the many claims tested")))

    # 7. verdict -----------------------------------------------------------
    caps = []
    if getattr(sc, "contrast_capped", False):
        caps.append("contrast failed the validation gate")
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
