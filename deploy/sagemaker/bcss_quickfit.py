"""Low-effort BREAST-native direction from BCSS via the warm H-optimus-0 endpoint.

The cheap transfer test: sample pure-class 224px@0.5µm/px tiles (tumor / stroma /
lymphocytic) from a SUBSET of BCSS ROIs + masks in S3, embed them through the hoptimus-embed
endpoint (no new GPU job), fit concept directions, and run the confound/transfer check on the
WSI GLOBAL list — to see whether a BREAST direction clears the TCGA-vs-BRACS batch null that
the COLON (NCT-CRC) direction failed.

Not the full curated reference — a subset sanity test. If a breast axis clears the null,
industrialize with a proper bcss_extract job. Requires: BCSS images+masks in
s3://bucketbiolayer/datasets/bcss/{images,masks}/ and the endpoint InService.

    python deploy/sagemaker/bcss_quickfit.py
"""
import base64
import io
import os
import sys
import tempfile

import boto3
import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)                                   # endpoint_client
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))  # repo root (biolayer)

from endpoint_client import embed as ep_embed              # noqa: E402
from biolayer.causal import probe as P                     # noqa: E402
from biolayer.causal.rank import fit_certified_axis        # noqa: E402
from biolayer.data import loader                           # noqa: E402
from biolayer.vectors import load_global                   # noqa: E402
from biolayer.vectors.transfer_check import (               # noqa: E402
    confound_check, format_report, axis_from_probe, axis_from_certified, group_mask)

B = "bucketbiolayer"
REGION = "us-west-2"
CLASSES = {1: "TUM", 2: "STR", 3: "LYM"}                   # BCSS gt code -> our concept name
NATIVE_MPP, TARGET_MPP, TILE = 0.25, 0.5, 224
CROP = int(round(TILE * TARGET_MPP / NATIVE_MPP))          # 448 native px -> 224 @ 0.5 µm/px
PER_CLASS, PURITY = 240, 0.9
MAX_ROIS, PER_ROI_PER_CLASS = 60, 25

s3 = boto3.client("s3", region_name=REGION)


def _ls(prefix):
    keys, tok = [], None
    while True:
        kw = dict(Bucket=B, Prefix=prefix)
        if tok:
            kw["ContinuationToken"] = tok
        r = s3.list_objects_v2(**kw)
        keys += [o["Key"] for o in r.get("Contents", [])]
        tok = r.get("NextContinuationToken")
        if not tok:
            break
    return keys


def _dl(key):
    p = os.path.join(tempfile.gettempdir(), "bcss_" + os.path.basename(key))
    if not os.path.exists(p):
        s3.download_file(B, key, p)
    return p


def sample_tiles():
    masks = {os.path.basename(k): k for k in _ls("datasets/bcss/masks/") if k.endswith(".png")}
    images = {os.path.basename(k): k for k in _ls("datasets/bcss/images/") if k.endswith(".png")}
    common = sorted(set(masks) & set(images))
    print(f"BCSS ROIs with image+mask: {len(common)} (images={len(images)} masks={len(masks)})",
          flush=True)
    rng = np.random.default_rng(0)
    tiles = {c: [] for c in CLASSES}                       # code -> list of 224x224x3 arrays
    for name in common[:MAX_ROIS]:
        if all(len(tiles[c]) >= PER_CLASS for c in CLASSES):
            break
        m = np.array(Image.open(_dl(masks[name])))
        if m.ndim == 3:
            m = m[..., 0]
        rgb = Image.open(_dl(images[name])).convert("RGB")
        H, Wd = m.shape
        for code in CLASSES:
            if len(tiles[code]) >= PER_CLASS:
                continue
            ys, xs = np.where(m == code)
            if len(xs) == 0:
                continue
            got = 0
            for j in rng.permutation(len(xs)):
                if got >= PER_ROI_PER_CLASS or len(tiles[code]) >= PER_CLASS:
                    break
                x0 = int(np.clip(xs[j] - CROP // 2, 0, max(0, Wd - CROP)))
                y0 = int(np.clip(ys[j] - CROP // 2, 0, max(0, H - CROP)))
                win = m[y0:y0 + CROP, x0:x0 + CROP]
                if win.shape != (CROP, CROP) or (win == code).mean() < PURITY:
                    continue
                crop = rgb.crop((x0, y0, x0 + CROP, y0 + CROP)).resize((TILE, TILE), Image.BILINEAR)
                tiles[code].append(np.asarray(crop))
                got += 1
        print(f"  {name}: " + " ".join(f"{CLASSES[c]}={len(tiles[c])}" for c in CLASSES), flush=True)
    return tiles


def embed_tiles(tiles):
    codes = sorted(tiles)
    class_names = [CLASSES[c] for c in codes]
    feats, labs = [], []
    for ci, c in enumerate(codes):
        arrs = tiles[c]
        for i in range(0, len(arrs), 32):
            b64 = []
            for a in arrs[i:i + 32]:
                buf = io.BytesIO()
                Image.fromarray(a, "RGB").save(buf, format="PNG")
                b64.append(base64.b64encode(buf.getvalue()).decode())
            r = ep_embed(images=b64)
            if not r.get("embeddings"):
                raise SystemExit(f"endpoint embed failed: {r}")
            feats += r["embeddings"]
            labs += [ci] * len(b64)
    return np.asarray(feats, dtype="float32"), np.asarray(labs), class_names


def main():
    tiles = sample_tiles()
    counts = {CLASSES[c]: len(tiles[c]) for c in CLASSES}
    print("sampled:", counts, flush=True)
    if min(counts.values()) < 30:
        raise SystemExit(f"too few tiles for some class {counts}; is masks/ populated in S3?")

    feats, labels, cn = embed_tiles(tiles)
    print(f"BCSS embedded via endpoint: {feats.shape} classes={cn} "
          f"counts={[int((labels == i).sum()) for i in range(len(cn))]}", flush=True)
    buf = io.BytesIO()
    np.savez_compressed(buf, feats=feats, labels=labels, class_names=np.array(cn))
    buf.seek(0)
    s3.upload_fileobj(buf, B, "embeddings/bcss_breast/h_optimus_0/train.npz")
    print("saved BCSS reference -> s3://%s/embeddings/bcss_breast/h_optimus_0/train.npz" % B, flush=True)

    # breast (BCSS) directions
    ax_b = fit_certified_axis(feats, labels, cn, "TUM", "LYM")
    print("\nBCSS TUM-vs-LYM: certified=%s heldout-AUROC=%.3f intensity|r|=%.2f (%s)"
          % (ax_b.certified, ax_b.heldout_auroc, ax_b.intensity_collinearity, ax_b.reason))
    cancer_b = P.fit_probe(feats, (labels == cn.index("TUM")).astype(int))

    # colon (NCT-CRC) directions for the head-to-head
    cf, cl, ccn, _ = loader.load("h_optimus_0", "train")
    cf, cl, ccn = np.asarray(cf), np.asarray(cl), list(ccn)
    ax_c = fit_certified_axis(cf, cl, ccn, "TUM", "LYM")
    cancer_c = P.fit_probe(cf, (cl == ccn.index("TUM")).astype(int))

    gl = load_global(s3.get_object(Bucket=B, Key="embeddings/lists/global.npz")["Body"].read())
    y = group_mask(gl, "TCGA")
    axes = {"BCSS_cancer_ovr": axis_from_probe(cancer_b),
            "BCSS_TUM_vs_LYM": axis_from_certified(ax_b),
            "CRC_cancer_ovr":  axis_from_probe(cancer_c),
            "CRC_TUM_vs_LYM":  axis_from_certified(ax_c)}
    rep = confound_check(gl.vectors, y, axes, n_null=200)
    print("\n" + format_report(rep, title="(BREAST-BCSS vs COLON-CRC directions -> TCGA vs BRACS)"))


if __name__ == "__main__":
    main()
