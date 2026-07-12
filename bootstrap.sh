#!/usr/bin/env bash
# One-shot environment setup for a fresh SageMaker box (or any GPU instance).
# Run from the repo root after cloning:  bash bootstrap.sh
#
# Idempotent: safe to re-run. Installs deps, then preflights HF auth, AWS, and GPU
# so you know the box is ready before kicking off an extraction/training job.
# Does NOT store any secrets — HF auth and AWS creds live outside git.

set -uo pipefail

# --- sanity: are we at the repo root? -------------------------------------
if [[ ! -f requirements.txt || ! -d biolayer ]]; then
  echo "ERROR: run this from the repo root (where requirements.txt and biolayer/ live)." >&2
  exit 1
fi

echo "=== 1/4  Python deps ==="
python -m pip install --upgrade pip -q
python -m pip install -r requirements.txt
echo "deps installed into: $(python -c 'import sys; print(sys.prefix)')"
echo

echo "=== 2/4  Hugging Face auth (needed only for gated models: H-optimus-0, H0-mini) ==="
if python -c "from huggingface_hub import HfApi; HfApi().whoami()" >/dev/null 2>&1; then
  HF_USER=$(python -c "from huggingface_hub import HfApi; print(HfApi().whoami()['name'])" 2>/dev/null)
  echo "HF authenticated as: ${HF_USER}"
else
  echo "NOT authenticated to Hugging Face."
  echo "  Run:  hf auth login        # browser OAuth, or --token <hf_...> for a manual PAT"
  echo "  Then accept model terms once at:"
  echo "    https://huggingface.co/bioptimus/H-optimus-0   (gated=auto, instant)"
  echo "  (Phikon-v2 is ungated — no login needed.)"
fi
echo

echo "=== 3/4  AWS / S3 preflight ==="
if aws sts get-caller-identity >/dev/null 2>&1; then
  aws sts get-caller-identity --query 'Arn' --output text
  if aws s3 ls s3://bucketbiolayer >/dev/null 2>&1; then
    echo "s3://bucketbiolayer reachable (shared artifact store)"
  else
    echo "WARN: cannot list s3://bucketbiolayer — check the execution role's bucket policy."
  fi
else
  echo "WARN: no AWS credentials resolved. On a SageMaker box this should come from the"
  echo "      execution role automatically; otherwise configure creds before S3/GPU jobs."
fi
echo

echo "=== 4/4  GPU check ==="
python - <<'PY'
try:
    import torch
    print("cuda available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("device:", torch.cuda.get_device_name(0))
except Exception as e:
    print("torch not importable yet:", e)
PY
echo

echo "=== Ready. Next steps (see docs/SETUP.md) ==="
echo "  # Phikon track — multi-layer embeddings (TUM vs LYM):"
echo "  python -m biolayer.data.extract --track phikon --split train --per-class 600 --no-upload"
echo "  # Readout-space causal battery -> evidence card:"
echo "  python -m biolayer.causal.battery --model phikon_v2 --split train --pos TUM --neg LYM"
echo "  # GPU H-optimus-0 job (SageMaker training job, CLI):"
echo "  python deploy/sagemaker/launch.py"
