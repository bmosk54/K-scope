"""Orchestrator: (prompt, K-Pro answer) -> per-claim causal certificate.

    answer
      -> decompose into atomic claims, resolved against the concept registry
         (tissue labels + HistoPLUS cell types)                     (claims.py; agent)
      -> per certifiable claim, on the substrate it resolved to:
           assemble + VALIDATE a contrast set                       (contrast.py)
           do()-battery: necessity/sufficiency/specificity vs null  (causal/battery.py)
           layer-resolved necessity (redundancy honesty)            (causal/intervene.py)
           confound gate: concept axis vs site/scanner axis         (causal/confound.py)
           numeric scores + GROUNDED/WEAK/NULL verdict              (scorecard.py)
      -> Holm-Bonferroni across claims (multiple comparisons)       (scorecard.py)
      -> answer-level certificate + coverage + guardrail audit + assumptions

Runs on cached embeddings (~1-3 s/claim). `fast=True` skips the layer sweep. Live
per-input hooks (intervene on THIS slide's forward pass) are the pending upgrade;
the certificate records `intervened_on_input: false` so it is never overstated.
"""
from .. import config, tracks
from ..causal import battery, confound, intervene
from ..causal import live as _live
from . import claims as _claims
from . import concepts as _concepts
from . import contrast as _contrast
from . import scorecard as _score
from . import trace as _trace

SCHEMA_VERSION = "dyn-0.2"
CAVEAT = ("Certifies model-internal causal use of an answer's concept claims in the "
          "encoder's representation — NOT biological/clinical validity.")

# The single most important honesty statement on the card: WHICH question it answers.
CONCEPT_LEVEL_CLAIM = (
    "CONCEPT-LEVEL: the concepts the answer invokes are ones this encoder genuinely "
    "represents, uses causally, and that are not staining/batch artifacts. This is NOT a "
    "per-slide / per-prediction claim — it does NOT verify the concept is present in any "
    "specific input tile, and in the default (no live_ctx) mode it CANNOT catch a "
    "per-slide hallucination such as 'tumor' asserted about a stroma tile.")
SLIDE_LEVEL_CLAIM = (
    "SLIDE-LEVEL (live source-intervention): necessity was measured by editing THIS "
    "slide's real forward pass (intervened_on_input=true) — a per-slide causal read of "
    "whether the readout depends on the concept axis.")


def _scope(any_live):
    """The card's explicit self-description: which of the three distinct questions it
    answers, so the framing never outruns the evidence."""
    return {
        "level": "slide-level (live) + concept-level" if any_live else "concept-level",
        "concept_level_claim": CONCEPT_LEVEL_CLAIM,
        "slide_level_claim": (SLIDE_LEVEL_CLAIM if any_live else
            "NOT RUN — no live_ctx provided. Pass live_ctx (this slide's tiles + a "
            "reference set) for the per-slide, input-dependent necessity test."),
        "questions": {
            "is the concept real/causal/unconfounded in the model?":
                "YES — this is what the concept-level card certifies",
            "does K-Pro's answer for THIS slide use the concept?":
                "not determinable here — no K-Pro internals",
            "is the concept present in THIS tile?":
                ("tested live per-slide (see slide_level_claim)" if any_live else
                 "NOT tested in concept-level mode"),
        },
    }

# Both assumptions printed on the card, not buried.
ASSUMPTIONS = [
    "encoder-faithfulness: a latent do() moves the model's REPRESENTATION, not tissue "
    "biology",
    "K-Pro-faithfulness: K-Pro's pathology inference reads from Phikon/H0-family "
    "features the way this probe reads them (unverified from here)",
]


def certify_answer(prompt, answer, track="phikon", split="train", n_null=200,
                   fast=False, use_bedrock=True, site_labels=None, artifacts_dir=None,
                   live_ctx=None, explain=False):
    """Answer-bound certification. If `live_ctx` (this slide's tiles + a reference set)
    is given, the necessity pillar is measured by a LIVE source-intervention on the real
    forward pass (intervened_on_input=True) for concepts on a hook-capable substrate;
    else necessity falls back to the cached readout-space read.

    live_ctx = {images, image_labels, ref_images, ref_labels, encoder?, n_null?}
    """
    t = tracks.get(track)
    claim_list = _claims.decompose(answer, preferred_model_key=t.model_key,
                                   split=split, use_bedrock=use_bedrock)

    certified, skipped, scores = [], [], []
    conf_seen = None
    any_live = False
    for cl in claim_list:
        if cl.status != "certifiable":
            skipped.append({"text": cl.text, "concept": cl.concept,
                            "verdict": "NOT_CERTIFIABLE", "reason": cl.reason})
            continue

        cs, feats, labels, class_names, source = _contrast.assemble(
            cl, split=split, artifacts_dir=artifacts_dir)
        spec = cl.spec

        bc = battery.run_battery(feats, labels, class_names, pos=spec.pos, neg=spec.neg,
                                 distractor=spec.distractor, n_null=n_null)

        # Necessity: LIVE source-intervention if a slide is supplied and the substrate
        # is hook-capable (transformers/phikon); else cached readout-space.
        live_nec, intervened, layered = None, False, None
        if live_ctx and _live.supports_live(cl.model_key):
            try:
                # Reuse the resident/warm encoder — never cold-load weights per call.
                enc = live_ctx.get("encoder")
                if enc is None:
                    from .. import serving as _serving
                    enc = _serving.warm_encoder(cl.model_key)
                ln = intervene.live_necessity(
                    cl.model_key, live_ctx["images"], live_ctx["image_labels"], class_names,
                    pos=spec.pos, neg=spec.neg, ref_images=live_ctx["ref_images"],
                    ref_labels=live_ctx["ref_labels"], n_null=live_ctx.get("n_null", 12),
                    encoder=enc, artifacts_dir=artifacts_dir)
                if ln.get("status") == "live_source_intervention":
                    live_nec, intervened, any_live = ln, True, True
                else:
                    live_nec = ln  # e.g. insufficient_tiles — recorded, falls back
            except Exception as e:
                live_nec = {"status": "error", "note": repr(e)}
        if not intervened and not fast:
            kw = {} if artifacts_dir is None else {"artifacts_dir": artifacts_dir}
            try:
                layered = intervene.pending_report(cl.model_key, split, spec.pos,
                                                   spec.neg, **kw)
            except Exception as e:
                layered = {"status": "pending", "note": f"layered unavailable: {e}"}

        conf = confound.confound_gate(feats, labels, class_names, site_labels=site_labels,
                                      pos=spec.pos, neg=spec.neg)
        conf_seen = conf

        sc = _score.score_claim(cl.concept, bc, layered, {"confound_gate": conf},
                                intervened_on_input=intervened,
                                live_necessity=live_nec if intervened else None,
                                contrast_valid=cs.valid, contrast_warnings=cs.warnings)
        scores.append(sc)
        certified.append({"claim": cl.text, "concept": cl.concept, "polarity": cl.polarity,
                          "substrate": cl.model_key, "source": cl.dataset_slug,
                          "spec": sc, "contrast": cs, "battery": bc, "confound": conf,
                          "live": live_nec})

    _score.holm_correction(scores)

    n_total = len(certified) + len(skipped)
    coverage = {"claims_total": n_total, "certifiable": len(certified),
                "not_certifiable": len(skipped),
                "summary": f"{len(certified)} of {n_total} claims certifiable"}
    conf_badge = _conf_badge(conf_seen)
    card = {
        "schema_version": SCHEMA_VERSION,
        "prompt": prompt, "answer": answer,
        "track": t.name, "preferred_substrate": t.model_key, "split": split,
        "coverage": coverage,
        # WHICH question this card answers — concept-level unless a slide was intervened on.
        "certification_scope": _scope(any_live),
        "concept_vocabulary": _concepts.coverage_summary(t.model_key, split),
        "claims": [_render(c) for c in certified],
        "not_certifiable": skipped,
        # answer-level deterministic trace (coverage reasoning + honest declines + confound)
        "summary_trace": _trace.summary_trace(coverage, conf_badge, skipped),
        "guardrails": {
            "matched_random_null": True,
            "specificity_control": True,
            "confound_gate": conf_badge,
            "multiple_comparisons": f"holm-bonferroni over {len(scores)} claims",
            "held_out_contrast_validation": True,
            "intervened_on_input": bool(any_live),
            "necessity_mode": ("live source-intervention (per-slide forward pass)"
                               if any_live else "cached readout-space"),
            "redundancy_honesty": "necessity reported layer-resolved / redundancy-limited",
        },
        "assumptions": ASSUMPTIONS,
        "caveat": CAVEAT,
    }
    # Optional plain-English narration — ONE batched LLM call, off by default so the
    # fast path stays instant. The numbers/verdicts are the deterministic trace's.
    if explain:
        card["narration"] = _trace.narrate(card)
    return card


def _conf_badge(conf):
    if conf is None:
        return "n/a (no certifiable claims)"
    return ("CHECKED (multi-site)" if conf.get("status") == "ok"
            else "UNCHECKED (single-source data)")


def _render(c):
    sc = c["spec"]
    cs = c["contrast"]
    return {
        "claim": c["claim"], "concept": c["concept"], "polarity": c["polarity"],
        "substrate": c["substrate"], "label_source": c["source"],
        "verdict": sc.verdict,
        "scores": {n: round(p.score, 3) for n, p in sc.pillars.items()},
        "pillars": {n: {"score": round(p.score, 3), "effect": round(p.effect, 3),
                        "z": (None if p.z != p.z else round(max(min(p.z, 999.0), -999.0), 2)),
                        "p": round(p.p, 4), "verdict": p.verdict}
                    for n, p in sc.pillars.items()},
        "confounded": sc.confounded,
        "contrast_capped": sc.contrast_capped,
        "survives_multiple_comparisons": sc.survives_correction,
        "live_necessity": ({"intervened_on_input": True,
                            "curve": [{"layer": cc["layer"], "gap": cc["necessity_gap"],
                                       "z": cc["gap_vs_null_z"], "bites": cc["bites"]}
                                      for cc in c["live"]["curve"]]}
                           if c.get("live") and c["live"].get("curve") else None),
        "contrast_validation": {
            "pos": cs.pos, "neg": cs.neg, "n_pos": cs.n_pos, "n_neg": cs.n_neg,
            "heldout_auroc": round(cs.heldout_auroc, 3),
            # intensity: report BOTH the cheap screen AND the controlled re-test
            "intensity_screen_r": round(cs.intensity_collinearity, 3),
            "intensity_suspect": cs.intensity_suspect,
            "intensity_adjudication": cs.confound_adjudication,
            "intensity_matched_auroc": (None if cs.matched_auroc != cs.matched_auroc
                                        else round(cs.matched_auroc, 3)),
            "intensity_matched_r": (None if cs.matched_intensity_collinearity
                                    != cs.matched_intensity_collinearity
                                    else round(cs.matched_intensity_collinearity, 3)),
            "n_matched": cs.n_matched,
            "valid": cs.valid, "warnings": list(cs.warnings), "flags": list(cs.flags)},
        "notes": sc.notes,
        # deterministic, instant — the auditable "why this score, why this verdict"
        "reasoning_trace": _trace.build_claim_trace(sc, cs, c.get("live"), c["confound"],
                                                    c["battery"]),
    }


def _demo():
    import json
    # Core question the demo answers: "Characterize the tumor microenvironment."
    ans = ("Tumor epithelium with a brisk peritumoral lymphocytic infiltrate, "
           "desmoplastic stroma, scattered eosinophils and plasma cells, high-grade.")
    card = certify_answer(
        prompt="Characterize the tumor microenvironment.",
        answer=ans, track="phikon", split="train", n_null=100, fast=True,
        use_bedrock=False)
    print(json.dumps({k: v for k, v in card.items() if k != "answer"},
                     indent=2, default=str))


if __name__ == "__main__":
    _demo()
