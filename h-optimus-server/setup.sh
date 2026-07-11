#!/usr/bin/env bash
# EC2 bootstrap for the H-optimus-0 inference server.
#
# Assumes an AWS Deep Learning AMI (Ubuntu, with NVIDIA drivers + CUDA
# preinstalled) so this script does not install GPU drivers itself. Run as
# EC2 user-data, or SSH in and run manually.
#
# Required env vars (export before running, or edit the values below):
#   HF_TOKEN   - Hugging Face token with access to the gated bioptimus/H-optimus-0 repo
# Optional:
#   PORT           (default 8000)
#   MAX_BATCH_SIZE (default 64)

set -euo pipefail

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "ERROR: HF_TOKEN is not set. Export it before running setup.sh." >&2
  exit 1
fi

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PORT="${PORT:-8000}"
MAX_BATCH_SIZE="${MAX_BATCH_SIZE:-64}"

cd "$APP_DIR"

python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

export HF_TOKEN
export PORT
export MAX_BATCH_SIZE

echo "Starting H-optimus-0 server on 0.0.0.0:${PORT} ..."
exec uvicorn server:app --host 0.0.0.0 --port "${PORT}"
