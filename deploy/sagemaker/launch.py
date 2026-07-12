"""CLI launcher — submit entry.py as a SageMaker Training Job on ml.g5.2xlarge.

Raw boto3 (no SDK estimator — SageMaker SDK v3 dropped it), fully CLI/API-native.
Packages this dir, uploads to s3://bucketbiolayer/sagemaker/code, and submits a
training job that runs entry.py in a prebuilt PyTorch DLC (which pip-installs
requirements.txt and runs the script via the SageMaker training toolkit).

    export SAGEMAKER_ROLE_ARN=arn:aws:iam::735570134926:role/owkin-sm-exec   # auto-set
    export HF_TOKEN=hf_...                                                    # auto-set
    python deploy/sagemaker/launch.py                 # full pretrained g5 run
    python deploy/sagemaker/launch.py --pretrained 0 --instance-type ml.m5.large  # cpu smoke

Track:  aws sagemaker describe-training-job --training-job-name <printed name>
"""
import argparse
import io
import json
import os
import tarfile
import time

import boto3

HERE = os.path.dirname(os.path.abspath(__file__))
BUCKET = os.environ.get("SM_BUCKET", "bucketbiolayer")
PREFIX = "sagemaker"
# Public AWS Deep Learning Container: PyTorch training toolkit (us-west-2).
DLC = ("763104351884.dkr.ecr.us-west-2.amazonaws.com/"
       "pytorch-training:2.3.0-gpu-py311-cu121-ubuntu20.04-sagemaker")


def _make_source_tarball():
    """Tar entry.py + hoptimus.py + requirements.txt into an in-memory .tar.gz."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for fname in ("entry.py", "hoptimus.py", "requirements.txt"):
            tar.add(os.path.join(HERE, fname), arcname=fname)
    buf.seek(0)
    return buf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--role", default=os.environ.get("SAGEMAKER_ROLE_ARN"))
    ap.add_argument("--instance-type", default="ml.g5.2xlarge")
    ap.add_argument("--image", default=DLC)
    ap.add_argument("--pretrained", type=int, default=1)
    ap.add_argument("--region", default=os.environ.get("AWS_DEFAULT_REGION", "us-west-2"))
    ap.add_argument("--max-runtime", type=int, default=3600)
    args = ap.parse_args()

    if not args.role:
        raise SystemExit("Set SAGEMAKER_ROLE_ARN (or --role).")

    job = f"hoptimus-edit-{int(time.time())}"
    code_key = f"{PREFIX}/code/{job}/sourcedir.tar.gz"

    s3 = boto3.client("s3", region_name=args.region)
    s3.upload_fileobj(_make_source_tarball(), BUCKET, code_key)
    submit_dir = f"s3://{BUCKET}/{code_key}"
    print(f"[launch] uploaded source -> {submit_dir}", flush=True)

    # Reserved hyperparameters the SageMaker training toolkit reads to fetch + run
    # our code. All values are JSON-encoded (matches how the SDK serializes them).
    hp = {
        "sagemaker_program": "entry.py",
        "sagemaker_submit_directory": submit_dir,
        "pretrained": args.pretrained,
    }
    env = {"HF_TOKEN": os.environ["HF_TOKEN"]} if os.environ.get("HF_TOKEN") else {}

    sm = boto3.client("sagemaker", region_name=args.region)
    sm.create_training_job(
        TrainingJobName=job,
        AlgorithmSpecification={"TrainingImage": args.image, "TrainingInputMode": "File"},
        RoleArn=args.role,
        HyperParameters={k: json.dumps(v) for k, v in hp.items()},
        OutputDataConfig={"S3OutputPath": f"s3://{BUCKET}/{PREFIX}/output"},
        ResourceConfig={"InstanceType": args.instance_type, "InstanceCount": 1,
                        "VolumeSizeInGB": 100},
        StoppingCondition={"MaxRuntimeInSeconds": args.max_runtime},
        Environment=env,
    )
    print(f"[launch] SUBMITTED {job} on {args.instance_type}")
    print(f"[launch] track: aws sagemaker describe-training-job --training-job-name {job} "
          "--query TrainingJobStatus --output text")


if __name__ == "__main__":
    main()
