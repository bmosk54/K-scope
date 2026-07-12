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
Not stored in git. `hf auth login` now supports a **browser/OAuth flow** (huggingface_hub
1.x) — run it and pick "Login from your browser"; it caches the token to
`~/.cache/huggingface/`:
```bash
hf auth login          # browser OAuth (or --token for a manual PAT)
```
Accept model terms once in the browser:
- H-optimus-0: https://huggingface.co/bioptimus/H-optimus-0  (gated=auto, instant)
- H0-mini:     https://huggingface.co/bioptimus/H0-mini      (gated=manual, review queue)

Phikon-v2 is ungated — no login required.

> Local dev shortcut: the workspace VSCode terminal profile auto-exports `HF_TOKEN`
> (read live from the HF cache), `SAGEMAKER_ROLE_ARN`, and the AWS session creds, and
> activates `owkin-env` — a fresh terminal is ready with no manual steps.

## 4. AWS / S3
Local auth: workspace session creds (`.owkin_hack_aws.sh`, gitignored). On a SageMaker
box: the execution role. Verify:
```bash
aws sts get-caller-identity
aws s3 ls s3://bucketbiolayer            # read/write for the team (bucket policy grants it)
```
Two shared stores:
- **`s3://bucketbiolayer`** (object storage) — **read/write** for the team accounts via
  its bucket policy (`GetObject`/`PutObject`/`DeleteObject`/`ListBucket`). Holds
  `embeddings/`, `directions/`, `sae/`, `certificates/`, and SageMaker `sagemaker/code`
  + `sagemaker/output`. `--upload` + `biolayer.data.s3_utils` are the shared channel;
  `*.npz`/`artifacts/` stay gitignored (too large/binary for git).
- **`h0-vector`** — an **S3 Vectors** store
  (`arn:aws:s3vectors:us-west-2:528759081002:bucket/h0-vector`, team-granted
  `PutVectors`/`QueryVectors`/…). Destination for tile/slide embeddings → the
  biodiscovery retrieval layer.

## 4b. GPU on SageMaker — CLI only (no Studio/UI)
H-optimus-0 (ViT-g/14) needs a GPU. Run it as a **SageMaker Training Job** via boto3 —
role + token are already wired (see the shortcut above):
```bash
python deploy/sagemaker/launch.py                 # pretrained H-optimus-0 on ml.g5.2xlarge
python deploy/sagemaker/launch.py --pretrained 0 --instance-type ml.m5.large   # CPU smoke test
aws sagemaker describe-training-job --training-job-name <name> --query TrainingJobStatus --output text
```
Details + status in [../deploy/sagemaker/README.md](../deploy/sagemaker/README.md).

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
(Or pull the shared copy from S3: `biolayer.data.s3_utils.load_embeddings("phikon_v2", "train")` — `bucketbiolayer` is read/write for the team, so `--upload` publishes and this reads back.)
