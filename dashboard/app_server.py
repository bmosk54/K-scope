"""Warm dashboard backend — one resident Flask process serving the UI + the MCP verbs.

Replaces the spawn-per-request Node bridge: biolayer (torch/transformers/embeddings) is
imported ONCE at startup, so every /api call is just the battery (~2-6s) instead of a
30s cold import. Serves the static dashboard AND the live certify infra from one port.

    HF_TOKEN=... python dashboard/app_server.py            # :4173
    PORT=8080 python dashboard/app_server.py

Endpoints:
    GET  /                      -> dashboard
    GET  /api/all               -> {CARD, DESIGNED_PROBES, MCP_VERBS, TRACKS}
    POST /api/certify_answer     -> {CARD}         body: {prompt, answer, track, bedrock}
    POST /api/verb/<name>        -> raw MCP verb output (certify/hypothesis/steer/ablate/
                                    design/specificity/rehypothesize/confound/layered/probe)
"""
import json
import os
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from flask import Flask, request, jsonify, send_from_directory, Response

# warm imports — pay the torch/biolayer cost ONCE, here, not per request
import bridge  # dashboard/bridge.py (adapters + build_*)
from biolayer.mcp import verbs
from biolayer import tracks as _tracks
from biolayer.dynamic import bedrock as _bedrock
from biolayer.dynamic import autoresearch as _autoresearch

PUBLIC = os.path.join(HERE, "public")
app = Flask(__name__, static_folder=None)

DEFAULT_Q = "Assess the tumor-infiltrating lymphocyte response and stromal desmoplasia."

_KPRO_SYS = (
    "You are K-Pro, a pathology foundation model reading a SINGLE 224px colorectal H&E tile "
    "(one NCT-CRC-HE tile — tile-level, not a whole slide, with no spatial-adjacency "
    "information). You are given the encoder's class for this tile. Answer the pathologist's "
    "question in 2-3 plain sentences — NO headings, NO markdown, NO report format. Anchor on "
    "the encoder's tile class, but characterize like a pathologist: name the tissue once, and "
    "where relevant describe the immune infiltrate at the CELLULAR level (lymphocytes, plasma "
    "cells, eosinophils), mitotic activity / nuclear grade, and any necrosis. Mention each "
    "finding only once. Be specific and clinical.")
_OPT_SYS = (
    "You refine a pathology question so it is SPECIFIC and answerable against tile-level "
    "tissue concepts the certifier can ground: tumor epithelium, lymphocytic/immune "
    "infiltrate, cancer-associated stroma, mucus, necrosis, smooth muscle, normal mucosa. "
    "You are given the current question and the last certificate's TRACE — the per-claim "
    "verdicts (which concepts certified GROUNDED/WEAK vs were declined NOT_CERTIFIABLE) and "
    "the reason each decline happened. USE that trace: keep and sharpen the concepts that "
    "grounded, and drop or reframe the parts that were declined as un-testable (cell/"
    "subcellular morphology, spatial/positional, molecular/clinical). Return ONE tighter, "
    "more specific question that targets the certifiable concepts. Output ONLY the question "
    "— no preamble, no quotes.")


def _slide():
    try:
        return json.load(open(os.path.join(PUBLIC, "slide_demo.json")))
    except Exception:
        return {"ho_composition": "", "prompt": DEFAULT_Q}


# ---- live source-intervention context (lazy) ------------------------------
# GROUNDED now requires GENUINE necessity (a live edit on the input's forward pass),
# not the cached readout-space read — so the dashboard supplies a live_ctx: the warm
# frozen phikon-v2 encoder + a fixed NCT-CRC tissue reference set (serving.py). Warmed
# once on first use; degrades to None (all-WEAK cached path) if anything is unavailable.
_LIVE = {"ctx": None, "tried": False}


def _live_ctx():
    if _LIVE["tried"]:
        return _LIVE["ctx"]
    _LIVE["tried"] = True
    try:
        from biolayer import serving
        serving.warmup("phikon_v2")
        _LIVE["ctx"] = serving.live_ctx("phikon_v2")
        print("[app_server] live source-intervention ready (phikon-v2 + tissue reference)", flush=True)
    except Exception:
        traceback.print_exc()
        _LIVE["ctx"] = None
    return _LIVE["ctx"]


# ---- MCP verb dispatch (all warm) -----------------------------------------
def _certify_answer(a):
    # live_ctx only matches the phikon-v2 substrate (its encoder + tissue reference); use it
    # by default so tissue concepts can earn GROUNDED. Pass {"live": false} to force cached.
    track = a.get("track", "phikon")
    ctx = _live_ctx() if (track == "phikon" and a.get("live", True)) else None
    return {"CARD": bridge.build_card(
        a.get("prompt", DEFAULT_Q), a.get("answer", bridge.DEMO_ANSWER),
        track=track, use_bedrock=bool(a.get("bedrock", False)), live_ctx=ctx)}


def _model_key(a):
    """Resolve the substrate model key from the track name (default phikon)."""
    return _tracks.get(a.get("track", "phikon")).model_key


# layered / axis_field take an EXPLICIT concept axis (pos/neg) per claim. Pass track=None +
# model so verbs._resolve keeps the requested axis instead of the track's default objective.
def _layered(a):
    return verbs.layered(model=_model_key(a), pos=a.get("pos", "TUM"),
                         neg=a.get("neg", "LYM"), space=a.get("space", "global"), track=None)


def _axis_field(a):
    return verbs.axis_field(model=_model_key(a), pos=a.get("pos", "TUM"),
                            neg=a.get("neg", "LYM"), space=a.get("space", "global"), track=None)


VERBS = {
    "certify_answer": _certify_answer,
    "certify":       lambda a: verbs.certify(track=a.get("track", "phikon"), n_null=a.get("n_null", 100)),
    "hypothesis":    lambda a: verbs.hypothesis(track=a.get("track", "phikon")),
    "design":        lambda a: bridge.build_designed(a.get("question", DEFAULT_Q),
                                                     use_bedrock=bool(a.get("bedrock", True))),
    "steer":         lambda a: verbs.steer(track=a.get("track", "phikon"), n_null=a.get("n_null", 100)),
    "ablate":        lambda a: verbs.ablate(track=a.get("track", "phikon"), n_null=a.get("n_null", 100)),
    "specificity":   lambda a: verbs.specificity(track=a.get("track", "phikon")),
    "rehypothesize": lambda a: verbs.rehypothesize(track=a.get("track", "phikon")),
    "confound":      lambda a: verbs.confound_verb(track=a.get("track", "phikon")),
    "layered":       _layered,
    "axis_field":    _axis_field,
    "probe":         lambda a: verbs.probe(track=a.get("track", "phikon")),
}


@app.after_request
def cors(resp):
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    resp.headers["Cache-Control"] = "no-store"
    return resp


@app.route("/api/all")
def api_all():
    try:
        return jsonify(bridge.build_all())
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"{type(e).__name__}: {e}"}), 503


@app.route("/api/certify_answer", methods=["POST", "OPTIONS"])
def api_certify():
    if request.method == "OPTIONS":
        return ("", 204)
    try:
        return jsonify(_certify_answer(request.get_json(silent=True) or {}))
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"{type(e).__name__}: {e}"}), 503


@app.route("/api/verb/<name>", methods=["POST", "OPTIONS"])
def api_verb(name):
    if request.method == "OPTIONS":
        return ("", 204)
    fn = VERBS.get(name)
    if fn is None:
        return jsonify({"error": f"unknown verb {name!r}", "verbs": sorted(VERBS)}), 404
    try:
        return jsonify({"verb": name, "result": fn(request.get_json(silent=True) or {})})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"{type(e).__name__}: {e}", "verb": name}), 503


@app.route("/api/answer", methods=["POST", "OPTIONS"])
def api_answer():
    """Submit the input slide + a prompt -> K-Pro (Claude) infers an answer from the
    slide's encoder composition. The 'inference' step the certificate then audits."""
    if request.method == "OPTIONS":
        return ("", 204)
    a = request.get_json(silent=True) or {}
    prompt = (a.get("prompt") or DEFAULT_Q).strip()
    slide = _slide()
    comp = slide.get("ho_composition", "")
    client = _bedrock.ClaudeBedrock()
    if not client.available():
        return jsonify({"error": "bedrock unavailable"}), 503
    try:
        answer = client._invoke(
            _KPRO_SYS, f"Tile encoder class (readout): {comp}\n\n"
                       f"Question: {prompt}\n\nAnswer:", max_tokens=350).strip()
        return jsonify({"answer": answer, "composition": comp,
                        "substrate": slide.get("substrate"), "prompt": prompt})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"{type(e).__name__}: {e}"}), 503


@app.route("/api/optimize_prompt", methods=["POST", "OPTIONS"])
def api_optimize():
    """Hypothesis step: refine the current prompt into a tighter, more certifiable one."""
    if request.method == "OPTIONS":
        return ("", 204)
    a = request.get_json(silent=True) or {}
    prompt = (a.get("prompt") or DEFAULT_Q).strip()
    client = _bedrock.ClaudeBedrock()
    if not client.available():
        return jsonify({"error": "bedrock unavailable"}), 503
    u = f"Current question: {prompt}\n"
    if a.get("coverage"):
        u += f"Last certificate coverage: {a['coverage']}\n"
    # The certify TRACE: per-claim verdicts + why each was declined, so the rewrite is
    # grounded in what actually certified rather than just the coverage headline.
    claims = a.get("claims") or []
    if claims:
        u += "Per-claim certify outcomes:\n"
        for c in claims:
            line = f"  - {c.get('claim', '?')!r} -> {c.get('verdict', '?')}"
            if c.get("concept"):
                line += f" [{c['concept']}"
                line += f": {c['contrast']}]" if c.get("contrast") else "]"
            if c.get("reason"):
                line += f" — declined: {c['reason']}"
            u += line + "\n"
    for s in (a.get("summary_trace") or []):
        if s.get("observation"):
            u += f"  · {s.get('step', '')}: {s['observation']}\n"
    u += ("Using these outcomes, return ONE tighter question that KEEPS the concepts that "
          "certified and DROPS or reframes the parts that were declined:")
    try:
        opt = client._invoke(_OPT_SYS, u, max_tokens=120).strip().strip('"')
        return jsonify({"prompt": opt, "from": prompt})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"{type(e).__name__}: {e}"}), 503


@app.route("/api/autoresearch")
def api_autoresearch():
    """Stream the autonomous causal-circuit discovery loop as Server-Sent Events.

    One SSE `data:` frame per iteration (design -> certify -> locate causal layer ->
    ablate -> reflect -> next), then a final `{done:true}` frame. GET so the browser's
    EventSource can drive it; the loop runs entirely on the existing certify battery.
    """
    def _int(name, default, lo, hi):
        try:
            return max(lo, min(int(request.args.get(name, default)), hi))
        except (TypeError, ValueError):
            return default

    problem = request.args.get("problem", DEFAULT_Q)
    track = request.args.get("track", "phikon")
    iters = _int("iters", 5, 1, 12)
    n_null = _int("n_null", 60, 20, 300)
    bedrock = request.args.get("bedrock", "0") in ("1", "true", "True")
    live = request.args.get("live", "0") in ("1", "true", "True")
    screen = request.args.get("screen", "0") in ("1", "true", "True")

    def gen():
        try:
            for rec in _autoresearch.iterate(problem, track=track, max_iters=iters,
                                             n_null=n_null, use_bedrock=bedrock, live=live,
                                             screen=screen):
                yield f"data: {json.dumps(rec, default=str)}\n\n"
        except Exception as e:
            traceback.print_exc()
            yield f"event: error\ndata: {json.dumps({'error': f'{type(e).__name__}: {e}'})}\n\n"

    return Response(gen(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no",
                             "Connection": "keep-alive"})


def _no_store(resp):
    # Demo iterates on JS/CSS live — never let the browser hand back a stale copy.
    resp.headers["Cache-Control"] = "no-store, must-revalidate"
    return resp


@app.route("/")
def index():
    return _no_store(send_from_directory(PUBLIC, "index.html"))


@app.route("/<path:fn>")
def static_file(fn):
    return _no_store(send_from_directory(PUBLIC, fn))


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "4173"))
    print(f"[app_server] warming biolayer … serving {PUBLIC} on :{port}", flush=True)
    app.run(host="0.0.0.0", port=port, threaded=True)
