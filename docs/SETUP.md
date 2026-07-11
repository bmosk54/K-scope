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
> S3. They are **not** committed to git (`artifacts/`, `*.npz` are gitignored —
> too large/binary for the shared baseline); regenerate them locally with the
> `--no-upload` extractor below. Once the role is fixed, `--upload` + `s3_utils`
> become the shared channel.

## 5. Reproduce

Work is split into two independent **tracks** (`biolayer.tracks`) — each is its own
model + dataset + objective + layer set, so Phikon-v2 and H0 never share assumptions:

| Track | Model | Dataset | Objective | Layers |
|---|---|---|---|---|
| `phikon` | Phikon-v2 (ungated) | NCT-CRC-HE | TUM vs LYM (tumor-immune) | 8/16/24 |
| `h0` | H0-mini (gated) | NCT-CRC-HE → cell-type (TODO) | TUM vs NORM (malignancy) | 3/7/11 |

Every tile is embedded at **3 layers**, each with **global (CLS)** and **local
(mean patch)** features.

```bash
# Phikon track — multi-layer local+global embeddings (A10G GPU)
python -m biolayer.data.extract --track phikon --split train --per-class 600 --no-upload

# H0 track — separate pipeline/objective (gated; h0_mini approval-queued, or flip
# H0_MODEL_KEY to h_optimus_0 in tracks/h0.py to run today)
python -m biolayer.data.extract --track h0 --split train --per-class 600 --no-upload

# Readout-space causal battery -> evidence-card JSON
python -m biolayer.causal.battery --model phikon_v2 --split train --pos TUM --neg LYM

# MCP server: certify(prediction) -> evidence card (stdio transport)
python -m biolayer.mcp.server
```

## Load embeddings back into Python
Embeddings are gitignored, so regenerate them first (step 5), then:
```python
from biolayer.data import loader
# readout global (back-compat single feature)
feats, labels, class_names, src = loader.load("phikon_v2", "train")
# any of the 3 layers x {global (CLS), local (mean patch)}
Xg, *_ = loader.load_layer("phikon_v2", "train", layer="mid", space="global")
Xl, *_ = loader.load_layer("phikon_v2", "train", layer="readout", space="local")
```
(When the S3 role is fixed, use `biolayer.data.s3_utils.load_embeddings("phikon_v2", "train")` instead.)
