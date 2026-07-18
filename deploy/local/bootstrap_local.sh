#!/usr/bin/env bash
# One-time local setup — run this ONCE (re-run is safe/idempotent) to make this laptop
# fully self-sufficient: no SageMaker, no S3, no AWS account required for the certify
# path. Only the optional K-Pro-answer / prompt-optimizer buttons need an LLM key (see
# deploy/local/README.md) — everything else (certify/steer/ablate/probe/specificity) is
# pure local torch + numpy once this script has run.
#
#   bash deploy/local/bootstrap_local.sh                  # phikon-v2 track, 300 tiles/class
#   PER_CLASS=800 bash deploy/local/bootstrap_local.sh    # more tiles = tighter probes, slower
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PER_CLASS="${PER_CLASS:-300}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

echo "=== 1/4  virtualenv (.venv) ==="
if [ ! -d .venv ]; then
  "$PYTHON_BIN" -m venv .venv
fi
source .venv/bin/activate
python -m pip install --upgrade pip -q

echo "=== 2/4  Python deps (this pulls torch/transformers — the big one) ==="
pip install -r requirements.txt

echo "=== 3/4  local embeddings — phikon-v2 (ungated, no HF login needed) ==="
if [ -f "artifacts/embeddings/nct_crc_he/phikon_v2/train.npz" ]; then
  echo "artifacts/embeddings/nct_crc_he/phikon_v2/train.npz already exists — skipping"
  echo "(delete it, or set PER_CLASS + re-run, to regenerate with more tiles)"
else
  echo "downloading NCT-CRC-HE tiles + embedding with phikon-v2 (CPU-friendly ViT-L)."
  echo "per_class=${PER_CLASS} -> ~$((PER_CLASS * 9)) tiles. This can take a while on a"
  echo "laptop CPU (minutes, not hours) — it only needs to happen once."
  python -m biolayer.data.extract --track phikon --split train \
    --per-class "$PER_CLASS" --no-upload
fi

echo "=== 4/4  sanity check — load the local artifacts back ==="
python - <<'PY'
from biolayer.data import loader
feats, labels, class_names, source = loader.load("phikon_v2", "train")
print(f"OK: {feats.shape[0]} tiles, {feats.shape[1]}-d, classes={class_names}")
print(f"source: {source}")
assert source.startswith("local:"), "expected a local artifact, not S3 — extraction didn't stick"
PY

echo
echo "=== Ready. No AWS/SageMaker/S3 needed from here. ==="
echo "Next:"
echo "  cp deploy/local/env.example .env.local   # then paste your OPENAI_API_KEY"
echo "  bash deploy/local/setup_cloudflare_tunnel.sh   # optional: permanent public URL"
echo "  bin/kscope start                                # launch the dashboard"
