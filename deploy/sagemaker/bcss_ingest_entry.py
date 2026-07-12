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
DEST_PREFIX = os.environ.get("BCSS_PREFIX", "datasets/bcss/images")
REGION = os.environ.setdefault("AWS_DEFAULT_REGION", "us-west-2")


def sh(cmd, cwd=None):
    print("+ " + cmd, flush=True)
    subprocess.check_call(cmd, shell=True, cwd=cwd)


def main():
    work = "/opt/ml/bcss"
    dl = os.path.join(work, "dl")
    os.makedirs(work, exist_ok=True)
    sh("pip install -q girder-client requests 'numpy<2' scikit-image imageio pillow")
    sh(f"git clone --depth 1 {REPO} {dl}")

    # Force images-only: appending wins because configs.py is imported as a module (last
    # assignment sticks) — robust to however the original PIPELINE line is formatted.
    with open(os.path.join(dl, "configs.py"), "a") as f:
        f.write("\nPIPELINE = ('images',)\n")

    sh("python download_crowdsource_dataset.py", cwd=dl)          # writes ./images/*.png

    imgs = sorted(glob.glob(os.path.join(dl, "images", "*")))
    print(f"[bcss] downloaded {len(imgs)} ROI images", flush=True)
    if not imgs:
        raise SystemExit("[bcss] no images downloaded — check the girder host / PIPELINE")

    s3 = boto3.client("s3", region_name=REGION)
    up = skip = 0
    for p in imgs:
        key = f"{DEST_PREFIX}/{os.path.basename(p)}"
        try:
            s3.head_object(Bucket=BUCKET, Key=key)               # skip-if-present
            skip += 1
            continue
        except Exception:
            pass
        s3.upload_file(p, BUCKET, key)
        up += 1
    print(f"[bcss] uploaded {up} new (+{skip} already present) -> "
          f"s3://{BUCKET}/{DEST_PREFIX}/", flush=True)


if __name__ == "__main__":
    main()
