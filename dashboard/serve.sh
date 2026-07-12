#!/usr/bin/env bash
# Deploy the BioLayer dashboard on SageMaker Studio, backed by THIS box's certify infra
# (Node static server + Python bridge -> biolayer.mcp.verbs). No extra infra: the Studio
# jupyter-server-proxy exposes the local port.
#
#   bash dashboard/serve.sh            # start on :4173 (default)
#   PORT=8080 bash dashboard/serve.sh  # custom port
#
# Then open in the Studio browser (auth via your Studio session cookie):
#   https://<DOMAIN>.studio.<REGION>.sagemaker.aws/jupyterlab/default/proxy/<PORT>/
#
# HF_TOKEN (optional) turns on the live Bedrock `design()` panel; without it that panel
# falls back to the registry contrasts. Everything else is live regardless.
set -euo pipefail
cd "$(dirname "$0")"
export PORT="${PORT:-4173}"
export PYTHON="${PYTHON:-python3}"

# Pick a Python that can run the warm Flask backend (app_server.py): it serves the UI AND
# every /api route the front-end submit calls (answer / optimize_prompt / verb / certify).
# Needs flask + numpy (certify) + boto3 (gallery fetch). If none is found, warn upfront.
for cand in "$PYTHON" "$HOME/miniconda3/envs/owkin-env/bin/python" python3; do
  if command -v "$cand" >/dev/null 2>&1 && "$cand" -c 'import flask, numpy, boto3' 2>/dev/null; then
    PYTHON="$cand"; export PYTHON; break
  fi
done
if ! "$PYTHON" -c 'import flask, numpy' 2>/dev/null; then
  echo "WARNING: '$PYTHON' is missing backend libs (flask/numpy) — the dashboard API (Submit," >&2
  echo "         certify, autoresearch) will be unavailable. Fix: pip install flask numpy boto3," >&2
  echo "         or point PYTHON at a capable interpreter: PYTHON=<python> bash dashboard/serve.sh" >&2
fi

# Pull the prebuilt patch galleries (cached in S3, not in git — 15-22 MB each) into public/
# so the Slide Gallery resolves without a rebuild. Best-effort: won't block startup.
"$PYTHON" fetch_galleries.py || true

# Launch the warm Flask backend (biolayer imported once at startup). Replaces the old
# zero-dep node server.js, which only exposed /api/all + /api/certify_answer and 404'd the
# Submit-tile routes (/api/answer, /api/optimize_prompt, /api/verb/*) as plain text.
pkill -f "node server.js" 2>/dev/null || true
pkill -f "app_server.py" 2>/dev/null || true
sleep 1
nohup "$PYTHON" app_server.py > /tmp/biolayer-dashboard.log 2>&1 &
sleep 2

# best-effort: surface the Studio proxy URL from the app metadata
META=/opt/ml/metadata/resource-metadata.json
if [ -f "$META" ]; then
  DOMAIN=$(grep -oE '"DomainId":"[^"]+"' "$META" | cut -d'"' -f4)
  REGION="${AWS_REGION:-us-west-2}"
  echo "dashboard up on :$PORT (log: /tmp/biolayer-dashboard.log)"
  echo "open: https://${DOMAIN}.studio.${REGION}.sagemaker.aws/jupyterlab/default/proxy/${PORT}/"
else
  echo "dashboard up on :$PORT — proxy path: /jupyterlab/default/proxy/${PORT}/"
fi
