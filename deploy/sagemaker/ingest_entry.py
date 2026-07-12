"""SageMaker entry point — ingest GDC WSIs into S3 from IN-REGION compute.

Runs on a cheap CPU instance (ml.m5.large) in us-west-2, where the GDC->instance
and instance->S3 hops are both fast (vs. routing 1.6 GB/slide through a laptop).
Reuses the bundled wsi_ingest module. IDs come from env:

    WSI_IDS      comma-separated UUIDs / GDC URLs / portal URLs
    MANIFEST_S3  s3://... text file, one UUID/URL per line (# comments ok)
    SM_BUCKET    destination bucket (default bucketbiolayer)
"""
import os
import urllib.parse

import boto3

import wsi_ingest  # bundled alongside this file by launch_ingest.py


def _read_manifest_s3(uri: str) -> list:
    p = urllib.parse.urlparse(uri)
    body = boto3.client("s3").get_object(Bucket=p.netloc, Key=p.path.lstrip("/"))["Body"]
    text = body.read().decode()
    return [ln.strip() for ln in text.splitlines() if ln.strip() and not ln.startswith("#")]


def main():
    bucket = os.environ.get("SM_BUCKET", "bucketbiolayer")
    items = [x for x in os.environ.get("WSI_IDS", "").split(",") if x.strip()]
    if os.environ.get("MANIFEST_S3"):
        items += _read_manifest_s3(os.environ["MANIFEST_S3"])
    if not items:
        raise SystemExit("no WSI_IDS or MANIFEST_S3 provided")

    s3 = boto3.client("s3")
    counts = {}
    for raw in items:
        try:
            status = wsi_ingest.ingest_one(wsi_ingest.parse_uuid(raw), s3, bucket)
        except Exception as e:
            print(f"ERROR on {raw!r}: {type(e).__name__}: {e}", flush=True)
            status = "error"
        counts[status] = counts.get(status, 0) + 1
    print("summary:", ", ".join(f"{k}={v}" for k, v in sorted(counts.items())), flush=True)


if __name__ == "__main__":
    main()
