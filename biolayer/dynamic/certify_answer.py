"""Orchestrator: (prompt, K-Pro answer) -> per-claim causal certificate.

The pipeline the analysis specified, with every guardrail baked into the tool
contract (not left to agent discretion):

    answer
      -> decompose into atomic, substrate-labeled claims          (claims.py; agent)
      -> per claim:
           assemble + VALIDATE a contrast set                      (contrast.py)
           run the do()-battery: necessity/sufficiency/specificity
             vs matched-random null                                (causal/battery.py)
           layer-resolved necessity (redundancy honesty)           (causal/intervene.py)
           confound gate: concept axis vs site/scanner axis        (causal/confound.py)
           numeric scores + GROUNDED/WEAK/NULL verdict             (scorecard.py)
      -> Holm-Bonferroni across claims (multiple comparisons)      (scorecard.py)
      -> answer-level certificate + guardrail audit

Runs on cached embeddings (~1-3 s/claim). `fast=True` skips the layer sweep and
uses readout-space necessity only. Live per-input hooks (intervene on THIS slide's
forward pass) are the pending upgrade; the certificate records
`intervened_on_input: false` so it is never overstated.
"""
from .. import tracks
from ..causal import battery, confound, intervene
from . import claims as _claims
from . import contrast as _contrast
from . import scorecard as _score

SCHEMA_VERSION = "dyn-0.1"
CAVEAT = ("Certifies model-internal causal use of an answer's concept claims in the "
          "encoder's representation — NOT biological/clinical validity, which rests "
          "on encoder faithfulness + the confound gate + external evidence.")


def certify_answer(prompt, answer, track="phikon", split="train", n_null=200,
                   fast=False, use_bedrock=True, site_labels=None, artifacts_dir=None):
    t = tracks.get(track)
    claim_list = _claims.decompose(answer, t, use_bedrock=use_bedrock)

    certified, skipped, scores = [], [], []
    for cl in claim_list:
        if cl.status != "certifiable":
            skipped.append({"text": cl.text, "verdict": "NOT_CERTIFIABLE",
                            "reason": cl.reason})
            continue

        cs, feats, labels, class_names, source = _contrast.assemble(
            t, cl, split=split, artifacts_dir=artifacts_dir)
        spec = cl.spec

        bc = battery.run_battery(feats, labels, class_names, pos=spec.pos, neg=spec.neg,
                                 distractor=spec.distractor, n_null=n_null)
        layered = None
        if not fast:
            kw = {} if artifacts_dir is None else {"artifacts_dir": artifacts_dir}
            layered = intervene.pending_report(t.model_key, split, spec.pos, spec.neg, **kw)
        conf = confound.confound_gate(feats, labels, class_names, site_labels=site_labels,
                                      pos=spec.pos, neg=spec.neg)

        sc = _score.score_claim(cl.concept, bc, layered, {"confound_gate": conf},
                                intervened_on_input=False)
        scores.append(sc)
        certified.append({"claim": cl.text, "concept": cl.concept,
                          "polarity": cl.polarity, "spec": sc, "contrast": cs,
                          "battery": bc, "confound": conf, "source": source})

    _score.holm_correction(scores)  # mutates verdicts in place

    return {
        "schema_version": SCHEMA_VERSION,
        "prompt": prompt,
        "answer": answer,
        "track": t.name, "model": t.model_key, "split": split,
        "claims": [_render(c) for c in certified],
        "not_certifiable": skipped,
        "guardrails": {
            "matched_random_null": True,
            "specificity_control": True,
            "confound_gate": conf_status(certified),
            "multiple_comparisons": f"holm-bonferroni over {len(scores)} claims",
            "held_out_contrast_validation": True,
            "intervened_on_input": False,
            "redundancy_honesty": "necessity reported layer-resolved / redundancy-limited",
        },
        "caveat": CAVEAT,
    }


def conf_status(certified):
    if not certified:
        return "n/a"
    st = certified[0]["confound"].get("status")
    return "ran (multi-site)" if st == "ok" else "UNAVAILABLE (single-source data)"


def _render(c):
    sc = c["spec"]
    cs = c["contrast"]
    return {
        "claim": c["claim"], "concept": c["concept"], "polarity": c["polarity"],
        "verdict": sc.verdict,
        "scores": {n: round(p.score, 3) for n, p in sc.pillars.items()},
        "pillars": {n: {"score": round(p.score, 3), "effect": round(p.effect, 3),
                        # cap display: a saturated readout yields z~1e8 (AUROC-1.0
                        # everywhere) — report ">=999" rather than a meaningless number
                        "z": (None if p.z != p.z else round(max(min(p.z, 999.0), -999.0), 2)),
                        "p": round(p.p, 4), "verdict": p.verdict}
                    for n, p in sc.pillars.items()},
        "confounded": sc.confounded,
        "survives_multiple_comparisons": sc.survives_correction,
        "contrast_validation": {
            "pos": cs.pos, "neg": cs.neg, "n_pos": cs.n_pos, "n_neg": cs.n_neg,
            "heldout_auroc": round(cs.heldout_auroc, 3),
            "intensity_collinearity": round(cs.intensity_collinearity, 3),
            "valid": cs.valid, "warnings": list(cs.warnings)},
        "notes": sc.notes,
    }


def _demo():
    """Runnable smoke test on cached embeddings (no Bedrock needed)."""
    import json
    ans = ("Tumor epithelium with a brisk peritumoral lymphocytic infiltrate, "
           "surrounding desmoplastic stroma, high-grade.")
    card = certify_answer(
        prompt="Describe this colorectal H&E tile and any biomarker signal.",
        answer=ans, track="phikon", split="train", n_null=100, fast=True,
        use_bedrock=False)
    # Trim the heavy raw battery blocks for a readable dump.
    print(json.dumps({k: v for k, v in card.items()
                      if k not in ("answer",)}, indent=2, default=str))


if __name__ == "__main__":
    _demo()
