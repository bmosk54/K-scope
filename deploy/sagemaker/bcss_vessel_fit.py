"""Artery search, step 1 — a CERTIFIED vascularization axis from BCSS vessel masks.

"Artery search" = localize vasculature on a WSI by ranking its tiles/patches along a
concept axis. This script builds the axis's LABELED reference: it samples vasculature
tiles (BCSS blood_vessel + angioinvasion + lymphatics, unioned) plus stroma/tumor/lymph
negatives from BCSS masks in S3, embeds them through the warm H-optimus-0 endpoint (no new
GPU job), saves a self-contained `bcss_vasculature` reference, and fits + CERTIFIES the
VASC-vs-STR axis with the non-negotiable matched-random null.

Two honesty points baked in, not glossed:
  * PURITY. A vessel is smaller than a 112 µm (224 px @ 0.5 µm/px) tile and sits INSIDE
    stroma, so a "pure vessel" tile does not exist at this scale. VASC here means a
    *vessel-containing region* (>=30 % vasculature); STR/TUM/LYM keep strict 0.9 purity.
    The VASC-vs-STR contrast is therefore "stroma-WITH-vessel vs stroma-WITHOUT-vessel" —
    which is exactly the right question (it controls for the stromal background), and we
    say so.
  * COLOR CONFOUND. Vessel lumens are full of RBCs (bright eosin/orange). A probe could
    detect "blood is here" rather than vessel morphology. The intensity-collinearity screen
    (in fit_certified_axis) + the held-out matched-random null below are what test that; a
    high |r| or a null-band AUROC is reported, never hidden.

    python deploy/sagemaker/bcss_vessel_fit.py            # build ref + certify (cache-aware)
    python deploy/sagemaker/bcss_vessel_fit.py --force    # rebuild the reference

Requires: BCSS images+masks in s3://bucketbiolayer/datasets/bcss/{images,masks}/ and the
hoptimus-embed endpoint InService (fresh AWS creds).
"""
import base64
import io
import os
import re
import sys
import tempfile

import boto3
import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

Image.MAX_IMAGE_PIXELS = None
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)                                     # endpoint_client
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))   # repo root (biolayer)

from endpoint_client import embed as ep_embed               # noqa: E402
from biolayer.causal import probe as P                      # noqa: E402
from biolayer.causal.rank import fit_certified_axis         # noqa: E402
from biolayer.data import loader                            # noqa: E402

B = "bucketbiolayer"
REGION = "us-west-2"
DATASET_SLUG = "bcss_vasculature"
OUT_KEY = "embeddings/bcss_vasculature/h_optimus_0/train.npz"

# name -> (set of BCSS gt codes, min union fraction of the tile).
# VASC unions the three BCSS vascular labels; see the module docstring on why 0.30.
CLASS_SPEC = {
    "VASC": ({14, 18, 19}, 0.30),   # lymphatics(14) + blood_vessel(18) + angioinvasion(19)
    "STR":  ({2}, 0.90),            # cancer-associated stroma — the vessel's background
    "TUM":  ({1}, 0.90),
    "LYM":  ({3}, 0.90),
}
NATIVE_MPP, TARGET_MPP, TILE = 0.25, 0.5, 224
CROP = int(round(TILE * TARGET_MPP / NATIVE_MPP))            # 448 native px -> 224 @ 0.5 µm/px
PER_CLASS, MAX_ROIS, PER_ROI_PER_CLASS = 220, 151, 30        # scan all ROIs; vasculature is rare
MAX_TRIES = 400                                              # candidate windows examined per (ROI,class)
# ^ bounds the sampling loop: an abundant class covers millions of pixels, and a rare VASC
# window rarely clears its purity, so an unbounded scan is billions of ops per ROI. Cap it.

s3 = boto3.client("s3", region_name=REGION)


def _s3_exists(key):
    try:
        s3.head_object(Bucket=B, Key=key)
        return True
    except s3.exceptions.ClientError:
        return False


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


def _stem(k):
    return re.sub(r"_(MPP|MAG)-[^_/]*\.png$", "", os.path.basename(k))


def sample_tiles():
    """Sample class tiles from BCSS ROIs; VASC by vessel-containing region, rest by purity.
    Reports how many VASC tiles came from how many ROIs (the data-sufficiency read)."""
    masks = {_stem(k): k for k in _ls("datasets/bcss/masks/") if k.endswith(".png")}
    images = {_stem(k): k for k in _ls("datasets/bcss/images/") if k.endswith(".png")}
    common = sorted(set(masks) & set(images))
    print(f"BCSS ROIs with image+mask: {len(common)} (images={len(images)} masks={len(masks)})",
          flush=True)
    rng = np.random.default_rng(0)
    tiles = {c: [] for c in CLASS_SPEC}
    vasc_rois = 0                                            # ROIs that contributed a VASC tile
    for ri, name in enumerate(common[:MAX_ROIS]):
        if all(len(tiles[c]) >= PER_CLASS for c in CLASS_SPEC):
            break
        m = np.array(Image.open(_dl(masks[name])))
        if m.ndim == 3:
            m = m[..., 0]
        # Which still-needed classes actually have pixels in THIS ROI's mask. If none, skip the
        # ROI WITHOUT decoding its (multi-MB) RGB image — the win for the rare-VASC tail.
        todo = {c: (list(codes), purity) for c, (codes, purity) in CLASS_SPEC.items()
                if len(tiles[c]) < PER_CLASS and np.isin(m, list(codes)).any()}
        if not todo:
            continue
        rgb = Image.open(_dl(images[name])).convert("RGB")
        H, Wd = m.shape
        before_vasc = len(tiles["VASC"])
        for cname, (codes, purity) in todo.items():
            ys, xs = np.where(np.isin(m, codes))
            got = 0
            for j in rng.integers(0, len(xs), size=MAX_TRIES):   # bounded candidate windows
                if got >= PER_ROI_PER_CLASS or len(tiles[cname]) >= PER_CLASS:
                    break
                x0 = int(np.clip(xs[j] - CROP // 2, 0, max(0, Wd - CROP)))
                y0 = int(np.clip(ys[j] - CROP // 2, 0, max(0, H - CROP)))
                win = m[y0:y0 + CROP, x0:x0 + CROP]
                if win.shape != (CROP, CROP) or np.isin(win, codes).mean() < purity:
                    continue
                crop = rgb.crop((x0, y0, x0 + CROP, y0 + CROP)).resize((TILE, TILE), Image.BILINEAR)
                tiles[cname].append(np.asarray(crop))
                got += 1
        if len(tiles["VASC"]) > before_vasc:
            vasc_rois += 1
        print(f"  [{ri + 1}/{len(common)}] {name}: "
              + " ".join(f"{c}={len(tiles[c])}" for c in CLASS_SPEC), flush=True)
    print(f"[data] VASC tiles drawn from {vasc_rois} distinct ROIs "
          f"(few ROIs => the axis may just memorize a handful of vessels)", flush=True)
    return tiles


def embed_tiles(tiles):
    names = [c for c in CLASS_SPEC if tiles[c]]
    feats, labs = [], []
    for ci, c in enumerate(names):
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
    return np.asarray(feats, dtype="float32"), np.asarray(labs), names


def heldout_null(feats, labels, class_names, pos, neg, n_null=200, seed=0):
    """The non-negotiable control: does the pos-vs-neg axis beat RANDOM directions at
    separating HELD-OUT tiles? Split 70/30, fit the real axis on train and score test;
    score the same test set with n_null random unit directions (same train scaler). The
    axis is only trustworthy if its folded held-out AUROC clears the null's 95th pct."""
    X, y = P.select_pair(np.asarray(feats), np.asarray(labels), list(class_names), pos, neg)
    rng = np.random.default_rng(seed)
    # STRATIFIED 70/30 split: VASC is rare, so a plain split could hand the test set a single
    # class and crash roc_auc_score. Split each class separately so both appear in train+test.
    tr, te = [], []
    for c in (0, 1):
        ci = np.where(y == c)[0]
        rng.shuffle(ci)
        cut = max(1, int(round(0.7 * len(ci))))
        tr.append(ci[:cut]); te.append(ci[cut:])
    tr, te = np.concatenate(tr), np.concatenate(te)
    if len(np.unique(y[te])) < 2 or len(np.unique(y[tr])) < 2:
        raise SystemExit(f"too few tiles to split {pos}-vs-{neg} (n={len(y)}); build more with --force")
    sc = StandardScaler().fit(X[tr])
    Ztr, Zte = sc.transform(X[tr]), sc.transform(X[te])

    def folded(a):
        return max(a, 1.0 - a)

    clf = LogisticRegression(max_iter=2000, random_state=seed).fit(Ztr, y[tr])
    real = folded(roc_auc_score(y[te], clf.decision_function(Zte)))
    null = np.empty(n_null)
    for i in range(n_null):
        r = rng.standard_normal(Zte.shape[1])
        r /= np.linalg.norm(r)
        null[i] = folded(roc_auc_score(y[te], Zte @ r))
    p95 = float(np.quantile(null, 0.95))
    return {"pos": pos, "neg": neg, "n_test": int(len(te)),
            "real_heldout_auroc": float(real), "null_mean": float(null.mean()),
            "null_p95": p95, "null_max": float(null.max()),
            "beats_random_frac": float((null < real).mean()),
            "survives_null": bool(real > p95)}


def certify_vessel_axis(pos="VASC", neg="STR", n_null=200):
    """Load the bcss_vasculature reference, fit + certify the pos-vs-neg axis, run the
    held-out matched-random null. Returns (CertifiedAxis, null_dict). Raises a clear error
    if the reference does not exist yet (run this script's main() first)."""
    feats, labels, cn, src = loader.load("h_optimus_0", "train", dataset_slug=DATASET_SLUG)
    feats, labels, cn = np.asarray(feats), np.asarray(labels), list(cn)
    ax = fit_certified_axis(feats, labels, cn, pos, neg)
    null = heldout_null(feats, labels, cn, pos, neg, n_null=n_null)
    return ax, null, src


def format_card(ax, null, src):
    trusted = ax.certified and null["survives_null"]
    L = [f"=== VASC axis certification ({ax.pos}-vs-{ax.neg}, ref={src}) ===",
         f"gate:  certified={ax.certified}  ({ax.reason})",
         f"       held-out AUROC {ax.heldout_auroc:.3f}  |  intensity |r| {ax.intensity_collinearity:.2f}"
         f"  (color-confound screen)",
         f"null:  held-out real {null['real_heldout_auroc']:.3f} vs matched-random "
         f"mean {null['null_mean']:.3f} / 95th {null['null_p95']:.3f} "
         f"(beats {null['beats_random_frac']:.0%} of random, n_test={null['n_test']})",
         f"VERDICT: {'TRUSTED — axis clears gate AND null' if trusted else 'DO NOT TRUST — ' + ('null band' if not null['survives_null'] else 'gate fail')}"]
    if ax.flags:
        L.append(f"flags: {', '.join(ax.flags)}")
    if ax.warnings:
        L.append(f"warnings: {', '.join(ax.warnings)}")
    return "\n".join(L)


def main():
    force = "--force" in sys.argv[1:]
    if not (_s3_exists(OUT_KEY) and not force):
        tiles = sample_tiles()
        counts = {c: len(tiles[c]) for c in CLASS_SPEC}
        print("sampled:", counts, flush=True)
        if counts["VASC"] < 30 or counts.get("STR", 0) < 30:
            raise SystemExit(f"too few tiles {counts}; VASC needs >=30 (is masks/ populated, "
                             f"and does BCSS have vascular labels in these ROIs?)")
        feats, labels, cn = embed_tiles(tiles)
        print(f"embedded via endpoint: {feats.shape} classes={cn} "
              f"counts={[int((labels == i).sum()) for i in range(len(cn))]}", flush=True)
        buf = io.BytesIO()
        np.savez_compressed(buf, feats=feats, labels=labels, class_names=np.array(cn))
        buf.seek(0)
        s3.upload_fileobj(buf, B, OUT_KEY)
        loader.clear_cache()                                # so certify sees the fresh ref
        print(f"saved vasculature reference -> s3://{B}/{OUT_KEY}", flush=True)
    else:
        print(f"CACHE HIT: s3://{B}/{OUT_KEY} exists — skipping build (pass --force to rebuild).",
              flush=True)

    ax, null, src = certify_vessel_axis()
    print("\n" + format_card(ax, null, src))


if __name__ == "__main__":
    main()
