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

# Pull the prebuilt patch galleries (cached in S3, not in git — they embed JPEG crops and
# run 15-22 MB each) into public/ so the Slide Gallery resolves without a rebuild.
# Best-effort: won't block startup if boto3/creds are absent.
"$PYTHON" fetch_galleries.py || true

pkill -f "node server.js" 2>/dev/null || true
sleep 1
nohup node server.js > /tmp/biolayer-dashboard.log 2>&1 &
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
