#!/usr/bin/env python3
"""Fetch (or publish) the prebuilt patch galleries for the dashboard's Slide Gallery.

The gallery HTML files are large — self-contained, with the whole-slide overview and ~96
native-res patch crops embedded as base64 JPEGs (15-22 MB each) — so they are deliberately
NOT committed to git. They are assembled once by deploy/sagemaker/patchwork_view.py and
cached in S3; this script pulls them into dashboard/public/ on load, so the cockpit works
without ever rebuilding the 13 GB-ranking pipeline.

    python dashboard/fetch_galleries.py            # download any missing galleries from S3
    python dashboard/fetch_galleries.py --force    # re-download all (overwrite local)
    python dashboard/fetch_galleries.py --upload    # publish local galleries -> S3 cache

Best-effort by design: if boto3 or AWS creds are unavailable it warns and exits 0, so
serve.sh still starts the dashboard (only the Slide Gallery iframe stays empty until a
later fetch succeeds).
"""
import glob
import os
import sys

BUCKET, REGION, PREFIX = "bucketbiolayer", "us-west-2", "galleries/"
PUBLIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "public")


def _client():
    import boto3
    return boto3.client("s3", region_name=REGION)


def upload():
    s3 = _client()
    files = sorted(glob.glob(os.path.join(PUBLIC, "*_gallery.html")))
    for f in files:
        s3.upload_file(f, BUCKET, PREFIX + os.path.basename(f),
                       ExtraArgs={"ContentType": "text/html; charset=utf-8"})
        print(f"  uploaded {os.path.basename(f)} ({os.path.getsize(f) // 1048576} MB)")
    print(f"published {len(files)} galleries -> s3://{BUCKET}/{PREFIX}")


def download(force=False):
    s3 = _client()
    os.makedirs(PUBLIC, exist_ok=True)
    objs = [o for o in s3.list_objects_v2(Bucket=BUCKET, Prefix=PREFIX).get("Contents", [])
            if o["Key"].endswith("_gallery.html")]
    got = 0
    for o in objs:
        dst = os.path.join(PUBLIC, os.path.basename(o["Key"]))
        if os.path.exists(dst) and not force:
            continue
        s3.download_file(BUCKET, o["Key"], dst)
        got += 1
        print(f"  fetched {os.path.basename(o['Key'])}")
    print(f"galleries ready in dashboard/public ({got} fetched, {len(objs)} available in S3)")


def main():
    # 1) missing library — the most common cause; give the exact fix, don't just fail silently.
    try:
        import boto3  # noqa: F401
    except ImportError:
        print(f"[fetch_galleries] WARNING — boto3 is not installed in this Python "
              f"({sys.executable}); the Slide Gallery will be empty.\n"
              f"  fix: pip install boto3   — or start the dashboard with a boto3-capable Python:\n"
              f"       PYTHON=<python-with-boto3> bash dashboard/serve.sh", file=sys.stderr)
        return
    # 2) run; distinguish a missing-credentials warning from any other failure.
    try:
        if "--upload" in sys.argv[1:]:
            upload()
        else:
            download(force="--force" in sys.argv[1:])
    except Exception as e:  # noqa: BLE001 — best-effort: never block the dashboard from starting
        name = type(e).__name__
        if name in ("NoCredentialsError", "PartialCredentialsError", "CredentialRetrievalError"):
            print("[fetch_galleries] WARNING — no AWS credentials found; can't reach the S3 "
                  "gallery cache. Source your creds and re-run (e.g. `source .owkin_hack_aws.sh`).",
                  file=sys.stderr)
        else:
            print(f"[fetch_galleries] WARNING — {name}: {e}; the Slide Gallery will be empty "
                  f"until galleries are fetched from s3://{BUCKET}/{PREFIX}", file=sys.stderr)


if __name__ == "__main__":
    main()
