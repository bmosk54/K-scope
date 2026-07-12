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

PUBLIC = os.path.join(HERE, "public")
app = Flask(__name__, static_folder=None)

DEFAULT_Q = "Characterize the tumor microenvironment."


# ---- MCP verb dispatch (all warm) -----------------------------------------
def _certify_answer(a):
    return {"CARD": bridge.build_card(
        a.get("prompt", DEFAULT_Q), a.get("answer", bridge.DEMO_ANSWER),
        track=a.get("track", "phikon"), use_bedrock=bool(a.get("bedrock", False)))}


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
    "layered":       lambda a: verbs.layered(track=a.get("track", "phikon")),
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
