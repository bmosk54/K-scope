"""CLI launcher — download the BCSS breast ROI images into S3 via an IN-REGION CPU job.

The dev box has poor bandwidth to non-AWS hosts; a cheap ml.m5.large pulls BCSS from the
Kitware girder server and uploads to s3://bucketbiolayer/datasets/bcss/images/ where both hops
are fast. Raw boto3, no console.

    python deploy/sagemaker/launch_bcss_ingest.py

Track: aws sagemaker describe-training-job --training-job-name <name> --query TrainingJobStatus
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
DLC = ("763104351884.dkr.ecr.us-west-2.amazonaws.com/"
       "pytorch-training:2.3.0-cpu-py311-ubuntu20.04-sagemaker")


def _tarball():
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        tar.add(os.path.join(HERE, "bcss_ingest_entry.py"), arcname="bcss_ingest_entry.py")
    buf.seek(0)
    return buf


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--role", default=os.environ.get("SAGEMAKER_ROLE_ARN"))
    ap.add_argument("--instance-type", default="ml.m5.large")
    ap.add_argument("--prefix", default="datasets/bcss/images", help="S3 key prefix for the ROIs")
    ap.add_argument("--region", default=os.environ.get("AWS_DEFAULT_REGION", "us-west-2"))
    args = ap.parse_args()
    if not args.role:
        raise SystemExit("Set SAGEMAKER_ROLE_ARN (or --role).")

    job = f"bcss-ingest-{int(time.time())}"
    code_key = f"{PREFIX}/code/{job}/sourcedir.tar.gz"
    s3 = boto3.client("s3", region_name=args.region)
    s3.upload_fileobj(_tarball(), BUCKET, code_key)

    sm = boto3.client("sagemaker", region_name=args.region)
    sm.create_training_job(
        TrainingJobName=job,
        AlgorithmSpecification={"TrainingImage": DLC, "TrainingInputMode": "File"},
        RoleArn=args.role,
        HyperParameters={
            "sagemaker_program": json.dumps("bcss_ingest_entry.py"),
            "sagemaker_submit_directory": json.dumps(f"s3://{BUCKET}/{code_key}"),
        },
        Environment={"SM_BUCKET": BUCKET, "BCSS_PREFIX": args.prefix},
        OutputDataConfig={"S3OutputPath": f"s3://{BUCKET}/{PREFIX}/output"},
        ResourceConfig={"InstanceType": args.instance_type, "InstanceCount": 1,
                        "VolumeSizeInGB": 60},
        StoppingCondition={"MaxRuntimeInSeconds": 21600},
    )
    print(f"[launch] SUBMITTED {job} on {args.instance_type} -> s3://{BUCKET}/{args.prefix}/")
    print(f"[launch] track: aws sagemaker describe-training-job --training-job-name {job} "
          "--query TrainingJobStatus --output text")


if __name__ == "__main__":
    main()
