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

from flask import Flask, request, jsonify, send_from_directory

# warm imports — pay the torch/biolayer cost ONCE, here, not per request
import bridge  # dashboard/bridge.py (adapters + build_*)
from biolayer.mcp import verbs
from biolayer import tracks as _tracks
from biolayer.dynamic import bedrock as _bedrock

PUBLIC = os.path.join(HERE, "public")
app = Flask(__name__, static_folder=None)

DEFAULT_Q = "Assess the tumor-infiltrating lymphocyte response and stromal desmoplasia."

_KPRO_SYS = (
    "You are K-Pro, a pathology foundation model reading a colorectal H&E slide. You are "
    "given the slide's tissue composition as classified by the encoder. Answer the "
    "pathologist's question in 2-3 plain sentences — NO headings, NO markdown, NO report "
    "format. Anchor on the tissue composition, but characterize like a pathologist: name "
    "the compartments once each, and where relevant describe the immune infiltrate at the "
    "CELLULAR level (lymphocytes, plasma cells, eosinophils), mitotic activity / nuclear "
    "grade, and any necrosis. Mention each finding only once. Be specific and clinical.")
_OPT_SYS = (
    "You refine a pathology question so it is SPECIFIC and answerable against tile-level "
    "tissue concepts the certifier can ground: tumor epithelium, lymphocytic/immune "
    "infiltrate, cancer-associated stroma, mucus, necrosis, smooth muscle, normal mucosa. "
    "Given the current question (and, if provided, which claims certified vs were declined), "
    "return ONE tighter, more specific question that targets the certifiable concepts and "
    "drops un-testable parts. Output ONLY the question — no preamble, no quotes.")


def _slide():
    try:
        return json.load(open(os.path.join(PUBLIC, "slide_demo.json")))
    except Exception:
        return {"ho_composition": "", "prompt": DEFAULT_Q}


# ---- MCP verb dispatch (all warm) -----------------------------------------
def _certify_answer(a):
    return {"CARD": bridge.build_card(
        a.get("prompt", DEFAULT_Q), a.get("answer", bridge.DEMO_ANSWER),
        track=a.get("track", "phikon"), use_bedrock=bool(a.get("bedrock", False)))}


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
            _KPRO_SYS, f"Slide tissue composition (encoder readout): {comp}\n\n"
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
    u += "Return ONE tighter, more specific question:"
    try:
        opt = client._invoke(_OPT_SYS, u, max_tokens=120).strip().strip('"')
        return jsonify({"prompt": opt, "from": prompt})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"{type(e).__name__}: {e}"}), 503


@app.route("/")
def index():
    return send_from_directory(PUBLIC, "index.html")


@app.route("/<path:fn>")
def static_file(fn):
    return send_from_directory(PUBLIC, fn)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "4173"))
    print(f"[app_server] warming biolayer … serving {PUBLIC} on :{port}", flush=True)
    app.run(host="0.0.0.0", port=port, threaded=True)
