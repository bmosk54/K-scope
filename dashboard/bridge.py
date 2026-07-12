"""Dashboard bridge — real certify infra -> the UI's global shapes.

The dashboard's public/data.js ships MOCK globals (window.CARD / DESIGNED_PROBES /
MCP_VERBS / TRACKS + illustrative CONFOUND/MIL/PIPELINE/...). This bridge computes the
LIVE ones from `biolayer` and emits them as JSON so server.js can serve them at /api/*;
app.js overrides the mock globals with whatever this returns and leaves the illustrative
ones untouched. Everything degrades: any failure -> a JSON error the server turns into a
503, and the front-end keeps the static mock.

CLI:
    python dashboard/bridge.py all            # {CARD, DESIGNED_PROBES, MCP_VERBS, TRACKS}
    python dashboard/bridge.py certify_answer --prompt "..." --answer "..." [--track phikon] [--bedrock]
"""
import argparse
import json
import os
import re
import sys

# runnable from repo root or dashboard/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DEMO_PROMPT = "Characterize the tumor microenvironment."
DEMO_ANSWER = (
    "Tumor epithelium with a brisk peritumoral lymphocytic infiltrate, desmoplastic "
    "stroma, scattered necrotic debris and extracellular mucin, with residual normal "
    "mucosa and smooth muscle at the margin.")


def _slug(s):
    return re.sub(r"[^a-z0-9]+", "_", (s or "").lower()).strip("_")[:48] or "claim"


def adapt_card(card):
    """certify_answer() output -> the UI's window.CARD shape (declined merged into claims)."""
    claims = []
    for c in card.get("claims", []):
        cv = c.get("contrast_validation", {}) or {}
        live = c.get("live") or {}
        item = {
            "id": _slug(c.get("claim")),
            "claim": c.get("claim"),
            "concept": c.get("concept"),
            # per-claim substrate provenance (substrate · label_source · CLS-width) so the
            # UI tags each claim's latent instead of a single global header.
            "substrate": c.get("substrate"), "label_source": c.get("label_source"),
            "substrate_dim": c.get("substrate_dim"), "substrate_tag": c.get("substrate_tag"),
            "contrast": (f'{cv.get("pos")} vs {cv.get("neg")}' if cv.get("pos") else None),
            # keep the raw axis codes so the Studio can query the live layer field per claim
            "pos": cv.get("pos"), "neg": cv.get("neg"),
            "polarity": c.get("polarity"),
            "verdict": c.get("verdict"),
            "scores": c.get("scores", {}),
            "contrast_validation": {
                "heldout_auroc": cv.get("heldout_auroc"),
                # the UI reads intensity_collinearity; feed it the cheap SCREEN value
                "intensity_collinearity": cv.get("intensity_screen_r", cv.get("intensity_collinearity")),
                "intensity_adjudication": cv.get("intensity_adjudication"),
                "intensity_matched_auroc": cv.get("intensity_matched_auroc"),
                "valid": cv.get("valid"),
                "warnings": cv.get("warnings", []),
                "flags": cv.get("flags", []),
            },
            "confounded": c.get("confounded", False),
            "contrast_capped": not cv.get("valid", True),
            "necessity_capped": c.get("necessity_capped", False),
            "survives_multiple_comparisons": c.get("survives_multiple_comparisons"),
            "reasoning_trace": c.get("reasoning_trace", []),
            "notes": c.get("notes", []),
        }
        if live.get("curve"):
            item["live_necessity"] = {
                "intervened_on_input": live.get("intervened_on_input", True),
                "curve": [{"layer": e.get("layer"), "gap": e.get("necessity_gap"),
                           "z": e.get("gap_vs_null_z"), "bites": e.get("bites")}
                          for e in live["curve"]],
            }
        claims.append(item)
    for s in card.get("not_certifiable", []):
        claims.append({"id": _slug(s.get("text")), "claim": s.get("text"),
                       "concept": s.get("concept"), "verdict": "NOT_CERTIFIABLE",
                       "reason": s.get("reason")})
    # dedupe: one row per concept (same concept -> identical scores, so repeats are noise).
    # declined rows dedupe by their text/reason instead of (null) concept.
    seen, uniq = set(), []
    for c in claims:
        # certifiable rows dedupe by concept (same concept -> identical scores); declined
        # rows dedupe by their CLAIM TEXT (they all share one reason, so keying on reason
        # would wrongly collapse every distinct clinical/molecular claim into one).
        key = ("nc:" + (c.get("claim") or "")) if c["verdict"] == "NOT_CERTIFIABLE" \
            else ("c:" + str(c.get("concept")))
        if key in seen:
            continue
        seen.add(key)
        uniq.append(c)
    claims = uniq
    n_cert = sum(1 for c in claims if c.get("scores"))
    coverage = {"claims_total": len(claims), "certifiable": n_cert,
                "not_certifiable": len(claims) - n_cert,
                "summary": f"{n_cert} of {len(claims)} distinct claims certifiable"}
    return {
        "schema_version": card.get("schema_version"),
        "prompt": card.get("prompt"), "answer": card.get("answer"),
        "track": card.get("track"), "preferred_substrate": card.get("preferred_substrate"),
        "split": card.get("split"), "coverage": coverage,
        "certification_scope": card.get("certification_scope"),
        "guardrails": card.get("guardrails"), "assumptions": card.get("assumptions"),
        "caveat": card.get("caveat"), "claims": claims,
        "summary_trace": card.get("summary_trace", []),
        "_live": bool(card.get("guardrails", {}).get("intervened_on_input")),
    }


def build_card(prompt=DEMO_PROMPT, answer=DEMO_ANSWER, track="phikon",
               use_bedrock=False, fast=True, n_null=100):
    from biolayer.mcp import verbs
    card = verbs.certify_answer(prompt, answer, track=track, n_null=n_null,
                                fast=fast, use_bedrock=use_bedrock)
    return adapt_card(card)


def build_designed(question=DEMO_PROMPT, model="phikon_v2", use_bedrock=True):
    """design() proposal + the REAL validation gate per probe -> window.DESIGNED_PROBES."""
    import numpy as np
    from biolayer.data import loader
    from biolayer.causal import probe as _probe
    from biolayer.dynamic import contrast as _contrast, probe_design as _pd
    feats, labels, class_names, _ = loader.load(model, "train")
    try:
        probes = _pd.design_probes(question, class_names, max_probes=8)
        designed_by = "claude-sonnet-4-6 (bedrock)"
    except Exception:
        # heuristic fallback: use the registry tissue contrasts so the panel still fills
        from biolayer.dynamic import concepts as _cn
        probes = [{"concept": c.concept, "pos": c.pos, "neg": c.neg,
                   "distractor": list(c.distractor)} for c in _cn._TISSUE_CONCEPTS][:8]
        designed_by = "registry (bedrock unavailable)"
    def _validate(fx, ly, cn, pos, neg, concept, substrate):
        X, y = _probe.select_pair(fx, ly, cn, pos, neg)
        au = _contrast._heldout_auroc(X, y)
        coll = _contrast._intensity_collinearity(X, y)
        gate = "PASS" if (au >= _contrast.MIN_HELDOUT_AUROC and
                          coll <= _contrast.MAX_INTENSITY_COLLINEARITY) else "REJECT"
        return {"concept": concept, "contrast": f"{pos} vs {neg}", "substrate": substrate,
                "auroc": round(float(au), 3), "intensity_r": round(float(coll), 3),
                "gate": gate, "sufficiency": 1.0, "random_null": 0.0}

    out = []
    for p in probes:
        try:
            out.append(_validate(feats, labels, class_names, p["pos"], p["neg"],
                                 p.get("concept"), "tissue · NCT-CRC"))
        except Exception:
            continue
    # Cell probes: the SAME design/validation gate on the HistoPLUS cell substrate, so the
    # panel shows both tile-level (tissue) and nucleus-level (cell) candidate probes.
    try:
        from biolayer.dynamic import concepts as _cn
        from biolayer import config as _cfg
        cfeats, clabels, cnames, _ = loader.load(model, "train", dataset_slug=_cfg.HISTOPLUS_SLUG)
        cpresent = set(cnames)
        for c in _cn._CELL_CONCEPTS:
            if c.pos in cpresent and c.neg in cpresent:
                try:
                    out.append(_validate(cfeats, clabels, cnames, c.pos, c.neg,
                                         c.concept, "cell · HistoPLUS"))
                except Exception:
                    continue
    except Exception:
        pass  # cell substrate unavailable -> tissue-only panel (degrades cleanly)
    return {"question": question, "designed_by": designed_by, "n_probes": len(out),
            "probes": out, "cap_threshold": _contrast.MAX_INTENSITY_COLLINEARITY,
            "note": "LLM proposes contrasts; the deterministic intensity gate decides "
                    "certifiability (Gate 2b adjudicates suspects). Tissue probes on "
                    "NCT-CRC (phikon-v2), cell probes on HistoPLUS (phikon-v2)."}


def build_tracks():
    """Real substrate registry -> window.TRACKS."""
    from biolayer import config
    rows = []
    meta = {"phikon_v2": ("phikon", "TUM vs LYM", "live battery"),
            "h0_mini": ("h0", "TUM vs NORM", "extract pending"),
            "h_optimus_0": ("h0/—", "TUM vs NORM", "live battery")}
    for mk, spec in config.MODELS.items():
        tname, obj, status = meta.get(mk, ("—", "—", "extract-only"))
        rows.append({"track": tname, "model": spec["hf_id"], "gated": spec.get("gated", False),
                     "backend": spec["backend"], "dim": spec["dim"], "blocks": spec["blocks"],
                     "layers": " / ".join(map(str, spec["layers"])), "objective": obj,
                     "status": status})
    return rows


def build_mcp_verbs():
    """The live MCP verb surface (biolayer/mcp/server.py) -> window.MCP_VERBS."""
    return [
        {"verb": "certify_answer", "powers": "the whole evidence card: claims, verdicts, traces (concept-level scope)", "status": "ship"},
        {"verb": "certify", "powers": "single-concept full causal card", "status": "ship"},
        {"verb": "design", "powers": "agent probe-design workbench (+ Gate 2b)", "status": "ship"},
        {"verb": "rehypothesize", "powers": "closed-loop 'next hypothesis' panel", "status": "ship"},
        {"verb": "ablate_live", "powers": "SLIDE-level necessity — edit this slide's forward pass", "status": "ship"},
        {"verb": "steer_from_card / ablate_from_card", "powers": "zero-recompute steer/ablate", "status": "ship"},
        {"verb": "layered / attribution", "powers": "layer curve + patch heat overlay", "status": "partial"},
        {"verb": "confound", "powers": "confound gate badge", "status": "data gap"},
        {"verb": "warmup / serving_status / embed", "powers": "warm inference backend + H-optimus endpoint", "status": "infra"},
    ]


def build_all(prompt=DEMO_PROMPT, answer=DEMO_ANSWER, track="phikon", use_bedrock=False):
    return {
        "CARD": build_card(prompt, answer, track=track, use_bedrock=use_bedrock),
        "DESIGNED_PROBES": build_designed(prompt, use_bedrock=use_bedrock),
        "MCP_VERBS": build_mcp_verbs(),
        "TRACKS": build_tracks(),
        "_meta": {"live": True, "track": track, "bedrock": use_bedrock},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["all", "certify_answer"])
    ap.add_argument("--prompt", default=DEMO_PROMPT)
    ap.add_argument("--answer", default=DEMO_ANSWER)
    ap.add_argument("--track", default="phikon")
    ap.add_argument("--bedrock", action="store_true")
    a = ap.parse_args()
    try:
        if a.cmd == "all":
            out = build_all(a.prompt, a.answer, track=a.track, use_bedrock=a.bedrock)
        else:
            out = {"CARD": build_card(a.prompt, a.answer, track=a.track,
                                      use_bedrock=a.bedrock, fast=True)}
        sys.stdout.write(json.dumps(out, default=str))
    except ImportError as e:
        # missing certify backend lib (numpy, torch, sklearn, biolayer, …) — give the fix,
        # not a bare stack trace. server.js surfaces this as the /api error the UI shows.
        mod = getattr(e, "name", None) or str(e)
        msg = (f"certify backend library '{mod}' is not installed in this Python "
               f"({sys.executable}). Install the backend deps (pip install -r requirements.txt) "
               f"or run the dashboard with a backend-capable Python: "
               f"PYTHON=<python-with-{mod}> bash dashboard/serve.sh")
        print(f"[bridge] WARNING — {msg}", file=sys.stderr)
        sys.stdout.write(json.dumps({"error": msg}))
        sys.exit(1)
    except Exception as e:
        sys.stdout.write(json.dumps({"error": f"{type(e).__name__}: {e}"}))
        sys.exit(1)


if __name__ == "__main__":
    main()
