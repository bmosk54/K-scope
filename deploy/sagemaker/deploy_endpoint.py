"""Deploy (or tear down) the warm H-optimus-0 real-time endpoint — raw boto3, no console.

The endpoint hosts H-optimus-0 persistently so external triggers (MCP `embed`, a K-Pro
answer needing a fresh tile) get a 1536-d vector without the 4 GB model re-download that
makes the training-job path wasteful for one-off queries. Weights load from the S3 cache
the tiling/embed job already seeded (offline, no HF hit), so no HF token is needed here.

    python deploy/sagemaker/deploy_endpoint.py                 # create/update + wait InService
    python deploy/sagemaker/deploy_endpoint.py --delete        # spin down (endpoint is billable)
    python deploy/sagemaker/deploy_endpoint.py --instance-type ml.g5.xlarge

Uses the endpoint g5 quota (=2), separate from the training g5 quota (=1), so it does not
contend with tiling/embed jobs. Track:
    aws sagemaker describe-endpoint --endpoint-name hoptimus-embed --query EndpointStatus
"""
import argparse
import io
import os
import tarfile
import time

import boto3

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
BUCKET = os.environ.get("SM_BUCKET", "bucketbiolayer")
PREFIX = "sagemaker/endpoint"
NAME = "hoptimus-embed"
VECTOR_BUCKET_ARN = "arn:aws:s3vectors:us-west-2:528759081002:bucket/h0-vector"
DLC = ("763104351884.dkr.ecr.us-west-2.amazonaws.com/"
       "pytorch-inference:2.3.0-gpu-py311-cu121-ubuntu20.04-sagemaker")


def _model_tar():
    """model.tar.gz: code/ = inference handler + bundled modules + reqs. No weights
    (restored from the S3 cache at container start)."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        tar.add(os.path.join(HERE, "endpoint", "inference.py"), arcname="code/inference.py")
        tar.add(os.path.join(HERE, "hoptimus.py"), arcname="code/hoptimus.py")
        tar.add(os.path.join(REPO, "biolayer", "data", "wsi_reader.py"), arcname="code/wsi_reader.py")
        tar.add(os.path.join(REPO, "biolayer", "data", "tile_wsi.py"), arcname="code/tile_wsi.py")
        tar.add(os.path.join(HERE, "requirements-tile.txt"), arcname="code/requirements.txt")
    buf.seek(0)
    return buf


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--role", default=os.environ.get("SAGEMAKER_ROLE_ARN"))
    ap.add_argument("--instance-type", default="ml.g5.2xlarge")
    ap.add_argument("--region", default=os.environ.get("AWS_DEFAULT_REGION", "us-west-2"))
    ap.add_argument("--name", default=NAME)
    ap.add_argument("--delete", action="store_true", help="delete endpoint + config + model (stop billing)")
    args = ap.parse_args()
    sm = boto3.client("sagemaker", region_name=args.region)

    if args.delete:
        for fn, key in ((sm.delete_endpoint, "EndpointName"),
                        (sm.delete_endpoint_config, "EndpointConfigName"),
                        (sm.delete_model, "ModelName")):
            try:
                fn(**{key: args.name})
                print(f"[endpoint] deleted {key}={args.name}")
            except Exception as e:
                print(f"[endpoint] skip {key}: {type(e).__name__}")
        return

    if not args.role:
        raise SystemExit("Set SAGEMAKER_ROLE_ARN (or --role).")

    tag = int(time.time())                                   # unique model/config per deploy
    model_name = f"{args.name}-m-{tag}"
    cfg_name = f"{args.name}-c-{tag}"
    s3 = boto3.client("s3", region_name=args.region)
    key = f"{PREFIX}/{model_name}/model.tar.gz"
    s3.upload_fileobj(_model_tar(), BUCKET, key)
    print(f"[endpoint] uploaded model artifact -> s3://{BUCKET}/{key}")

    sm.create_model(
        ModelName=model_name,
        PrimaryContainer={
            "Image": DLC,
            "ModelDataUrl": f"s3://{BUCKET}/{key}",
            "Environment": {
                "SAGEMAKER_PROGRAM": "inference.py",
                "SAGEMAKER_SUBMIT_DIRECTORY": "/opt/ml/model/code",
                "SM_BUCKET": BUCKET,
                "AWS_DEFAULT_REGION": args.region,
                "MODEL_CACHE_KEY": "models/hf-cache-h-optimus-0.tar",
                "VECTOR_BUCKET_ARN": VECTOR_BUCKET_ARN,
                "TS_DEFAULT_RESPONSE_TIMEOUT": "600",       # allow slow cold model_fn
            },
        },
        ExecutionRoleArn=args.role,
    )
    sm.create_endpoint_config(
        EndpointConfigName=cfg_name,
        ProductionVariants=[{
            "VariantName": "main", "ModelName": model_name,
            "InstanceType": args.instance_type, "InitialInstanceCount": 1,
            "ModelDataDownloadTimeoutInSeconds": 1200,
            "ContainerStartupHealthCheckTimeoutInSeconds": 900,
        }],
    )

    try:
        sm.describe_endpoint(EndpointName=args.name)
        sm.update_endpoint(EndpointName=args.name, EndpointConfigName=cfg_name)
        print(f"[endpoint] UPDATING {args.name} -> {cfg_name}")
    except sm.exceptions.ClientError:
        sm.create_endpoint(EndpointName=args.name, EndpointConfigName=cfg_name)
        print(f"[endpoint] CREATING {args.name} -> {cfg_name}")

    print("[endpoint] waiting for InService (cold start pulls the DLC + restores the cache; "
          "~8-12 min)...")
    while True:
        d = sm.describe_endpoint(EndpointName=args.name)
        st = d["EndpointStatus"]
        print(f"  {time.strftime('%H:%M:%S')} {st}")
        if st in ("InService", "Failed"):
            if st == "Failed":
                raise SystemExit(f"[endpoint] FAILED: {d.get('FailureReason')}")
            break
        time.sleep(30)
    print(f"[endpoint] InService: {args.name}  (spin down with --delete when idle)")


if __name__ == "__main__":
    main()
