"""Ingest TCGA whole-slide images from the GDC into S3 — idempotently.

Streams open-access WSIs from the GDC data API straight into
`s3://bucketbiolayer/wsi/<project_id>/<file_name>` WITHOUT a full local copy, and
skips any file already present (same key + same size). Feed it GDC file UUIDs, bare
`api.gdc.cancer.gov/data/<uuid>` links, or full portal URLs (the UUID is parsed out).

    python -m biolayer.data.wsi_ingest 4730b23e-aea1-49a2-ba63-2231fd88b592
    python -m biolayer.data.wsi_ingest "https://portal.gdc.cancer.gov/....&ids=<uuid>..."
    python -m biolayer.data.wsi_ingest --manifest wsi_ids.txt      # one UUID/URL per line

Controlled-access files are skipped with a warning (they'd need a GDC token).
"""
import argparse
import json
import os
import re
import sys
import urllib.request

import boto3

# Works both as a package module (python -m biolayer.data.wsi_ingest) and standalone
# (copied into a SageMaker container, imported as top-level `wsi_ingest`).
try:
    from .. import config
    _BUCKET, _REGION = config.BUCKET, config.REGION
except ImportError:  # standalone: no parent package
    _BUCKET = os.environ.get("SM_BUCKET", "bucketbiolayer")
    _REGION = os.environ.get("AWS_DEFAULT_REGION", "us-west-2")

GDC_FILES_API = "https://api.gdc.cancer.gov/files/"
GDC_DATA_API = "https://api.gdc.cancer.gov/data/"
# GDC file UUIDs are lowercase 8-4-4-4-12 (the uppercase one in a filename is NOT it).
_UUID_RE = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")


def parse_uuid(s: str) -> str:
    m = _UUID_RE.search(s)
    if not m:
        raise ValueError(f"no GDC file UUID found in {s!r}")
    return m.group(0)


def gdc_metadata(uuid: str) -> dict:
    """file_name, size, access, state, project_id for a GDC file UUID."""
    url = (f"{GDC_FILES_API}{uuid}"
           "?fields=file_name,file_size,access,state,cases.project.project_id")
    with urllib.request.urlopen(url, timeout=30) as r:
        d = json.load(r)["data"]
    cases = d.get("cases") or [{}]
    project = (cases[0].get("project") or {}).get("project_id", "UNKNOWN")
    return {"file_name": d["file_name"], "size": int(d["file_size"]),
            "access": d["access"], "state": d.get("state"), "project": project}


def _head(s3, bucket: str, key: str):
    try:
        return s3.head_object(Bucket=bucket, Key=key)
    except s3.exceptions.ClientError as e:
        if e.response["Error"]["Code"] in ("404", "NoSuchKey", "NotFound"):
            return None
        raise


def ingest_one(uuid: str, s3, bucket: str) -> str:
    meta = gdc_metadata(uuid)
    key = f"wsi/{meta['project']}/{meta['file_name']}"
    dest = f"s3://{bucket}/{key}"

    if meta["access"] != "open":
        print(f"SKIP (access={meta['access']}, needs GDC token): {meta['file_name']}")
        return "skipped-access"
    if meta["state"] != "released":
        print(f"SKIP (state={meta['state']}): {meta['file_name']}")
        return "skipped-state"

    head = _head(s3, bucket, key)
    if head and head["ContentLength"] == meta["size"]:
        print(f"present, skip: {dest} ({meta['size'] / 1e9:.2f} GB)")
        return "present"
    if head:
        print(f"re-uploading (size {head['ContentLength']} != {meta['size']}): {dest}")

    print(f"downloading {meta['file_name']} ({meta['size'] / 1e9:.2f} GB, {meta['project']}) "
          f"-> {dest}", flush=True)
    with urllib.request.urlopen(GDC_DATA_API + uuid, timeout=60) as resp:
        # boto3 reads the HTTP response in chunks and does a multipart upload —
        # no full-file buffering on local disk.
        s3.upload_fileobj(resp, bucket, key)
    print(f"done: {dest}", flush=True)
    return "uploaded"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("ids", nargs="*", help="GDC UUIDs / data links / portal URLs")
    ap.add_argument("--manifest", help="file with one UUID/URL per line (# comments ok)")
    ap.add_argument("--bucket", default=_BUCKET)
    ap.add_argument("--region", default=_REGION)
    args = ap.parse_args()

    items = list(args.ids)
    if args.manifest:
        with open(args.manifest) as f:
            items += [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]
    if not items:
        ap.error("provide UUIDs/URLs or --manifest")

    s3 = boto3.client("s3", region_name=args.region)
    counts = {}
    for raw in items:
        try:
            uuid = parse_uuid(raw)
            status = ingest_one(uuid, s3, args.bucket)
        except Exception as e:  # keep going on the rest of the batch
            print(f"ERROR on {raw!r}: {type(e).__name__}: {e}", file=sys.stderr)
            status = "error"
        counts[status] = counts.get(status, 0) + 1
    print("summary:", ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))


if __name__ == "__main__":
    main()
