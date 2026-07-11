# Setup / instance transfer

Everything needed to reproduce is on the **`eddiebae`** branch. No secrets are in
git — the HF token and AWS auth live outside the repo. To bring a fresh SageMaker
JupyterLab instance up to speed:

## 1. Clone + branch
```bash
git clone git@github.com:bmosk54/owkin-hack.git    # or HTTPS + a GitHub PAT
cd owkin-hack
git checkout eddiebae
```

## 2. Python deps
```bash
pip install -r requirements.txt
```

## 3. Hugging Face auth (only needed for gated models: H-optimus-0, H0-mini)
Not stored in git. On the new box:
```bash
hf auth login          # paste a token from hf.co/settings/tokens (read scope)
```
Accept model terms once in the browser:
- H-optimus-0: https://huggingface.co/bioptimus/H-optimus-0  (gated=auto, instant)
- H0-mini:     https://huggingface.co/bioptimus/H0-mini      (gated=manual, review queue)

Phikon-v2 is ungated — no login required.

## 4. AWS / S3
Auth is the SageMaker execution role (no keys). Verify:
```bash
aws sts get-caller-identity
aws s3 ls s3://bucketbiolayer
```
> NOTE: the current execution role has **ListBucket only** — no Get/PutObject on
> `bucketbiolayer`. Until that role policy is fixed, embeddings can't go through
> S3, so they are committed to this branch under `artifacts/` as the transfer
> channel. See `--no-upload` on the extractor.

## 5. Reproduce
```bash
# Phikon-v2 embeddings (ungated, ~1.5 tiles/s on CPU)
python -m biolayer.extract --model phikon_v2 --split train --per-class 200 --no-upload

# H-optimus-0 embeddings (gated=auto; ViT-giant, slower on CPU — keep per-class small)
python -m biolayer.extract --model h_optimus_0 --split train --per-class 150 --no-upload

# Base causal battery (readout space, matched-random nulls) -> evidence-card JSON
python -m biolayer.battery --model phikon_v2 --split train --pos TUM --neg LYM
```

## Pull the committed embeddings back into Python
```python
import numpy as np
d = np.load("artifacts/embeddings/nct_crc_he/phikon_v2/train.npz", allow_pickle=True)
feats, labels, class_names = d["feats"], d["labels"], list(d["class_names"])
```
(When the S3 role is fixed, use `biolayer.s3_utils.load_embeddings("phikon_v2", "train")` instead.)
