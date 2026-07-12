"""CLI launcher — WSI -> tiles -> H-optimus-0 -> S3 features + h0-vector, on a g5.

Raw boto3, no console. Bundles the entry + the biolayer modules it needs, submits a
GPU training job. Role + HF token come from the workspace env.

    python deploy/sagemaker/launch_tile_embed.py \
        s3://bucketbiolayer/wsi/TCGA-BRCA/TCGA-E2-A14P-...svs \
        [--filters whitespace,tissue] [--mpp 0.5] \
        [--vector-index <name>]        # push vectors too (needs an existing h0-vector index)

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
REPO = os.path.dirname(os.path.dirname(HERE))
BUCKET = os.environ.get("SM_BUCKET", "bucketbiolayer")
PREFIX = "sagemaker"
VECTOR_BUCKET_ARN = "arn:aws:s3vectors:us-west-2:528759081002:bucket/h0-vector"
DLC = ("763104351884.dkr.ecr.us-west-2.amazonaws.com/"
       "pytorch-training:2.3.0-gpu-py311-cu121-ubuntu20.04-sagemaker")


def _tarball():
    """entry + bundled biolayer modules (imported top-level in the container) + reqs."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        tar.add(os.path.join(HERE, "tile_embed_entry.py"), arcname="tile_embed_entry.py")
        tar.add(os.path.join(HERE, "hoptimus.py"), arcname="hoptimus.py")
        tar.add(os.path.join(REPO, "biolayer", "data", "wsi_reader.py"), arcname="wsi_reader.py")
        tar.add(os.path.join(REPO, "biolayer", "data", "tile_wsi.py"), arcname="tile_wsi.py")
        tar.add(os.path.join(HERE, "requirements-tile.txt"), arcname="requirements.txt")
    buf.seek(0)
    return buf


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("slides", nargs="+", help="one or more s3:// .svs/.tiff (batched: model loads once)")
    ap.add_argument("--role", default=os.environ.get("SAGEMAKER_ROLE_ARN"))
    ap.add_argument("--instance-type", default="ml.g5.2xlarge")
    ap.add_argument("--filters", default="whitespace,tissue")
    ap.add_argument("--mpp", default="0.5")
    ap.add_argument("--tile-px", default="224")
    ap.add_argument("--max-tiles", type=int, help="cap kept tiles (quick trial run); "
                    "omit to embed ALL sensible tiles")
    ap.add_argument("--patch-dtype", default="float16",
                    help="dtype for the 256-vector PATCH list (float16 halves size/RAM vs float32)")
    ap.add_argument("--vector-index", default="layerbioindex",
                    help="h0-vector index for the GLOBAL/CLS list (dim 1536, cosine); '' to skip")
    ap.add_argument("--reseed-cache", action="store_true",
                    help="re-download H-optimus-0 from HuggingFace and refresh the S3 model "
                         "cache (needs HF auth in env). Default reuses the cached weights "
                         "offline, so the HF token is NOT sent (it would leak into CloudWatch).")
    ap.add_argument("--region", default=os.environ.get("AWS_DEFAULT_REGION", "us-west-2"))
    args = ap.parse_args()
    if not args.role:
        raise SystemExit("Set SAGEMAKER_ROLE_ARN (or --role).")

    job = f"tile-embed-{int(time.time())}"
    code_key = f"{PREFIX}/code/{job}/sourcedir.tar.gz"
    s3 = boto3.client("s3", region_name=args.region)
    s3.upload_fileobj(_tarball(), BUCKET, code_key)

    env = {
        "SLIDES_S3": ",".join(args.slides), "SM_BUCKET": BUCKET,
        "FILTERS": args.filters, "MPP": args.mpp, "TILE_PX": args.tile_px,
        "PATCH_DTYPE": args.patch_dtype,
    }
    if args.max_tiles:
        env["MAX_TILES"] = str(args.max_tiles)
    if args.vector_index:
        env["VECTOR_BUCKET_ARN"] = VECTOR_BUCKET_ARN
        env["VECTOR_INDEX"] = args.vector_index

    hp = {
        "sagemaker_program": json.dumps("tile_embed_entry.py"),
        "sagemaker_submit_directory": json.dumps(f"s3://{BUCKET}/{code_key}"),
    }
    # The HF token is ONLY needed to (re)download the gated weights. By default we reuse the
    # S3 model cache offline and never send it — it would otherwise echo into CloudWatch. Pass
    # it (and force a fresh download + reseed) only when --reseed-cache is set.
    if args.reseed_cache:
        env["FORCE_RESEED"] = "1"
        if not os.environ.get("HF_TOKEN"):
            raise SystemExit("--reseed-cache needs HF_TOKEN in env (run `hf auth login`).")
        hp["HF_TOKEN"] = json.dumps(os.environ["HF_TOKEN"])   # hyperparam dodges 512-char env cap

    sm = boto3.client("sagemaker", region_name=args.region)
    sm.create_training_job(
        TrainingJobName=job,
        AlgorithmSpecification={"TrainingImage": DLC, "TrainingInputMode": "File"},
        RoleArn=args.role,
        HyperParameters=hp,
        Environment=env,
        OutputDataConfig={"S3OutputPath": f"s3://{BUCKET}/{PREFIX}/output"},
        ResourceConfig={"InstanceType": args.instance_type, "InstanceCount": 1,
                        "VolumeSizeInGB": 200},
        StoppingCondition={"MaxRuntimeInSeconds": 21600},
    )
    print(f"[launch] SUBMITTED {job} on {args.instance_type} for {len(args.slides)} slide(s)")
    print(f"[launch] track: aws sagemaker describe-training-job --training-job-name {job} "
          "--query TrainingJobStatus --output text")


if __name__ == "__main__":
    main()
