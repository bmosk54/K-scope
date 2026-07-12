"""SageMaker in-region CPU job: download BCSS breast ROIs -> S3 (images only).

BCSS (Amgad et al. 2019, CC0) = breast-cancer ROIs cropped from TCGA-BRCA with dense tissue
masks. Here we only pull the RGB ROI images (the ask) via the authors' official downloader,
which crops each ROI server-side from the public Kitware girder host — no credentials, no
gigapixel WSI transfer. Uploads to s3://<bucket>/datasets/bcss/images/ (idempotent).

Masks are one flag away (the downloader's PIPELINE) for when we build the labeled breast
reference; not fetched here.
"""
import glob
import os
import subprocess

import boto3

REPO = "https://github.com/CancerDataScience/CrowdsourcingDataset-Amgadetal2019"
BUCKET = os.environ.get("SM_BUCKET", "bucketbiolayer")
DEST_ROOT = os.environ.get("BCSS_ROOT", "datasets/bcss")            # images/ + masks/ under here
PIPELINE = os.environ.get("BCSS_PIPELINE", "images,masks")          # which local dirs to fetch+upload
REGION = os.environ.setdefault("AWS_DEFAULT_REGION", "us-west-2")


def sh(cmd, cwd=None):
    print("+ " + cmd, flush=True)
    subprocess.check_call(cmd, shell=True, cwd=cwd)


def _upload_dir(s3, local_dir, prefix):
    files = sorted(glob.glob(os.path.join(local_dir, "*")))
    up = skip = 0
    for p in files:
        key = f"{prefix}/{os.path.basename(p)}"
        try:
            s3.head_object(Bucket=BUCKET, Key=key)                # skip-if-present
            skip += 1
            continue
        except Exception:
            pass
        s3.upload_file(p, BUCKET, key)
        up += 1
    print(f"[bcss] {os.path.basename(local_dir)}: uploaded {up} new (+{skip} present) -> "
          f"s3://{BUCKET}/{prefix}/", flush=True)
    return len(files)


def main():
    stages = tuple(x.strip() for x in PIPELINE.split(",") if x.strip())   # e.g. ('images','masks')
    work = "/opt/ml/bcss"
    dl = os.path.join(work, "dl")
    os.makedirs(work, exist_ok=True)
    sh("pip install -q girder-client requests 'numpy<2' scikit-image imageio pillow")
    sh(f"git clone --depth 1 {REPO} {dl}")

    # Appending wins because configs.py is imported as a module (last assignment sticks) —
    # robust to however the original PIPELINE line is formatted.
    with open(os.path.join(dl, "configs.py"), "a") as f:
        f.write(f"\nPIPELINE = {stages!r}\n")

    sh("python download_crowdsource_dataset.py", cwd=dl)          # writes ./images and/or ./masks

    s3 = boto3.client("s3", region_name=REGION)
    codes = os.path.join(dl, "meta", "gtruth_codes.tsv")         # class code -> tissue name
    if os.path.exists(codes):
        s3.upload_file(codes, BUCKET, f"{DEST_ROOT}/gtruth_codes.tsv")
        print(f"[bcss] uploaded gtruth_codes.tsv -> s3://{BUCKET}/{DEST_ROOT}/", flush=True)

    total = 0
    for stage in stages:
        total += _upload_dir(s3, os.path.join(dl, stage), f"{DEST_ROOT}/{stage}")
    if not total:
        raise SystemExit("[bcss] nothing downloaded — check the girder host / PIPELINE")


if __name__ == "__main__":
    main()
