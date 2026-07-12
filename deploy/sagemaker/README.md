# H-optimus-0 on SageMaker — CLI only (no Studio/UI)

Load H-optimus-0 via `timm`, modify its weights arbitrarily, run it on a
`ml.g5.2xlarge` — all from the terminal via the SageMaker **Training Job** API.

| File | Role |
|---|---|
| [`hoptimus.py`](hoptimus.py) | Load via timm (`hf-hub:` + required kwargs) + `edit_weights()` for arbitrary weight surgery |
| [`entry.py`](entry.py) | Training-job entry point: load → edit → embed → save to S3 |
| [`launch.py`](launch.py) | CLI launcher (raw boto3) — packages to `bucketbiolayer`, submits the job, no console |
| [`requirements.txt`](requirements.txt) | `timm` (auto-installed in the container) |

## Quick local check (no AWS, no HF)

```bash
pip install timm torch
python -c "import deploy.sagemaker.hoptimus as h; m=h.load_hoptimus(pretrained=False); \
print('blocks', len(m.blocks)); print(h.example_edits(m))"
```

Loads a random-init ViT-g/14 and runs the arbitrary weight edits — proves the
manipulation API without the gated download.

## Run on SageMaker (CLI)

`SAGEMAKER_ROLE_ARN` + `HF_TOKEN` are auto-exported by the workspace terminal (see
[../../docs/SETUP.md](../../docs/SETUP.md) §3) — no manual export needed.

```bash
python deploy/sagemaker/launch.py                                    # pretrained g5 run
python deploy/sagemaker/launch.py --pretrained 0 --instance-type ml.m5.large   # CPU smoke test
```

`launch.py` (raw boto3) tars this dir → `s3://bucketbiolayer/sagemaker/code`, then submits
a training job in the prebuilt PyTorch DLC (which pip-installs `requirements.txt` and runs
`entry.py`). Outputs (`hoptimus_edited.pt`, `sample_cls.pt`, `run.json`) land in
`s3://bucketbiolayer/sagemaker/output`. Track:
```bash
aws sagemaker describe-training-job --training-job-name <name> --query TrainingJobStatus --output text
```

## Status (2026-07-11)

1. **Execution role — CREATED.** `owkin-sm-exec`
   (`arn:aws:iam::735570134926:role/owkin-sm-exec`), trusts `sagemaker.amazonaws.com`,
   has `AmazonSageMakerFullAccess` + `AmazonS3FullAccess`. The workspace env
   (`.owkin_hack_aws.sh`) exports `SAGEMAKER_ROLE_ARN`, so `launch.py` finds it with no flags.

2. **GPU quota — OK.** `ml.g5.2xlarge for training job usage` = 1 (one job at a time).
   The EC2 G quota being 0 is irrelevant — SageMaker training uses its own pool.

3. **Last unconfirmed step: `iam:PassRole` + `sagemaker:CreateTrainingJob`.** The
   participant role carries an explicit-deny guardrail policy (`iam_policy-0`) that blocks
   `iam:SimulatePrincipalPolicy`, so we can't pre-check these. The only way to confirm is
   to launch — `create-training-job` checks PassRole at submit time. If it returns
   `AccessDenied`, the guardrail blocks job submission and the organizers must run it; if
   it returns a job name (or a capacity/quota error), permissions are fine.
