# H-optimus-0 on SageMaker — CLI only (no Studio/UI)

Load H-optimus-0 via `timm`, modify its weights arbitrarily, run it on a
`ml.g5.2xlarge` — all from the terminal via the SageMaker **Training Job** API.

| File | Role |
|---|---|
| [`hoptimus.py`](hoptimus.py) | Load via timm (`hf-hub:` + required kwargs) + `edit_weights()` for arbitrary weight surgery |
| [`entry.py`](entry.py) | Training-job entry point: load → edit → embed → save to S3 |
| [`launch.py`](launch.py) | CLI launcher (SageMaker SDK) — submits the job, no console |
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

```bash
export SAGEMAKER_ROLE_ARN=arn:aws:iam::735570134926:role/<execution-role>
export HF_TOKEN=hf_xxx                 # gated H-optimus-0 download; omit + use --pretrained 0 to skip
python deploy/sagemaker/launch.py --wait
```

Outputs (`hoptimus_edited.pt`, `sample_cls.pt`, `run.json`) land in the job's S3
output path; track with `aws sagemaker describe-training-job --training-job-name <name>`.

## ⚠️ Blockers on this workshop account (verified 2026-07-11)

1. **No passable execution role.** The only SageMaker-trusting role is the
   service-linked `AWSServiceRoleForAmazonSageMakerNotebooks`, which **cannot** be a
   training-job `RoleArn`. You need a role that trusts `sagemaker.amazonaws.com` with
   `AmazonSageMakerFullAccess` + S3 access. **Ask the organizers for its ARN.** If your
   role has `iam:CreateRole`+`PassRole`, create one:
   ```bash
   aws iam create-role --role-name owkin-sm-exec \
     --assume-role-policy-document '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"sagemaker.amazonaws.com"},"Action":"sts:AssumeRole"}]}'
   aws iam attach-role-policy --role-name owkin-sm-exec --policy-arn arn:aws:iam::aws:policy/AmazonSageMakerFullAccess
   aws iam attach-role-policy --role-name owkin-sm-exec --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess
   ```
   (Participant roles often can't do this — expect to need the organizers.)

2. **GPU quota.** `ml.g5.2xlarge for training job usage` = 1 ✅ (so one job at a time is
   fine). EC2 G quota is 0 but that's irrelevant here — SageMaker training uses its own
   pool.

Once the role ARN exists, everything above runs unchanged.
