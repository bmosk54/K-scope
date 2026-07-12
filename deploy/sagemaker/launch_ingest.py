"""CLI launcher — ingest GDC WSIs into S3 via an IN-REGION SageMaker CPU job.

The speed fix: instead of streaming 1.6 GB/slide through a laptop, a cheap in-region
ml.m5.large downloads GDC -> S3 where both hops are fast. Raw boto3, no console.

    # a few IDs inline:
    python deploy/sagemaker/launch_ingest.py 4730b23e-... <uuid2> <portal-url> ...
    # or a manifest file (one UUID/URL per line):
    python deploy/sagemaker/launch_ingest.py --manifest wsi_ids.txt

Uses SAGEMAKER_ROLE_ARN (auto-set by the workspace env). Track:
    aws sagemaker describe-training-job --training-job-name <name> --query TrainingJobStatus
"""
import argparse
import io
import json
import os
import tarfile
import time

import boto3

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
BUCKET = os.environ.get("SM_BUCKET", "bucketbiolayer")
PREFIX = "sagemaker"
# CPU PyTorch DLC (has boto3); no GPU needed for a network->S3 job.
DLC = ("763104351884.dkr.ecr.us-west-2.amazonaws.com/"
       "pytorch-training:2.3.0-cpu-py311-ubuntu20.04-sagemaker")


def _tarball():
    """Bundle ingest_entry.py + the wsi_ingest module (as a top-level import)."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        tar.add(os.path.join(HERE, "ingest_entry.py"), arcname="ingest_entry.py")
        tar.add(os.path.join(REPO, "biolayer", "data", "wsi_ingest.py"),
                arcname="wsi_ingest.py")
    buf.seek(0)
    return buf


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("ids", nargs="*", help="GDC UUIDs / data links / portal URLs")
    ap.add_argument("--manifest", help="local file, one UUID/URL per line")
    ap.add_argument("--role", default=os.environ.get("SAGEMAKER_ROLE_ARN"))
    ap.add_argument("--instance-type", default="ml.m5.large")
    ap.add_argument("--region", default=os.environ.get("AWS_DEFAULT_REGION", "us-west-2"))
    args = ap.parse_args()
    if not args.role:
        raise SystemExit("Set SAGEMAKER_ROLE_ARN (or --role).")

    items = list(args.ids)
    if args.manifest:
        with open(args.manifest) as f:
            items += [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]
    if not items:
        ap.error("provide UUIDs/URLs or --manifest")

    s3 = boto3.client("s3", region_name=args.region)
    sm = boto3.client("sagemaker", region_name=args.region)
    job = f"wsi-ingest-{int(time.time())}"

    # Ship the whole list as a manifest object (avoids env-var length limits).
    manifest_key = f"{PREFIX}/code/{job}/manifest.txt"
    s3.put_object(Bucket=BUCKET, Key=manifest_key, Body="\n".join(items).encode())
    code_key = f"{PREFIX}/code/{job}/sourcedir.tar.gz"
    s3.upload_fileobj(_tarball(), BUCKET, code_key)

    sm.create_training_job(
        TrainingJobName=job,
        AlgorithmSpecification={"TrainingImage": DLC, "TrainingInputMode": "File"},
        RoleArn=args.role,
        HyperParameters={
            "sagemaker_program": json.dumps("ingest_entry.py"),
            "sagemaker_submit_directory": json.dumps(f"s3://{BUCKET}/{code_key}"),
        },
        Environment={
            "MANIFEST_S3": f"s3://{BUCKET}/{manifest_key}",
            "SM_BUCKET": BUCKET,
        },
        OutputDataConfig={"S3OutputPath": f"s3://{BUCKET}/{PREFIX}/output"},
        ResourceConfig={"InstanceType": args.instance_type, "InstanceCount": 1,
                        "VolumeSizeInGB": 30},
        StoppingCondition={"MaxRuntimeInSeconds": 21600},
    )
    print(f"[launch] SUBMITTED {job} ({len(items)} slides) on {args.instance_type}")
    print(f"[launch] track: aws sagemaker describe-training-job --training-job-name {job} "
          "--query TrainingJobStatus --output text")


if __name__ == "__main__":
    main()
