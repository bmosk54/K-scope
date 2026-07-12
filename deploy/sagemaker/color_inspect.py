"""Is a concept axis MORPHOLOGY or just STAIN/COLOR? A within-slide diagnostic.

Rank one slide's tiles by a concept axis, then two honest checks:
  1. correlate the axis score with per-tile color/stain stats (brightness, saturation, and
     Macenko hematoxylin/eosin via skimage.rgb2hed) — a high |r| means the axis rides stain,
     not structure. Done WITHIN one slide, so the cross-slide batch effect can't inflate it.
  2. save top-K / bottom-K tile montages, so you can eyeball whether the model picked tumor
     morphology (crowded atypical nuclei, architecture) or merely the darkest/most-purple tiles.

    python deploy/sagemaker/color_inspect.py --slide-prefix TCGA \
        --wsi s3://bucketbiolayer/wsi/TCGA-BRCA/TCGA-E2-A14P-...svs
"""
import argparse
import os
import sys
import tempfile

import boto3
import numpy as np
from PIL import Image
from skimage.color import rgb2hed, rgb2hsv

Image.MAX_IMAGE_PIXELS = None
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

from biolayer.causal import probe as P                 # noqa: E402
from biolayer.data import loader                        # noqa: E402
from biolayer.data.wsi_reader import open_wsi           # noqa: E402
from biolayer.vectors import load_global                # noqa: E402

B, REGION = "bucketbiolayer", "us-west-2"
s3 = boto3.client("s3", region_name=REGION)


def crop_tile(reader, x, y, tile_px=224, target_mpp=0.5):
    mpp = reader.mpp or 0.25
    d = max(target_mpp / mpp, 1.0)
    level = reader.level_for_downsample(d)
    ds = reader.level_downsamples[level]
    read_px = max(1, int(round(tile_px * d / ds)))
    t = reader.read_region((int(x), int(y)), level, (read_px, read_px))
    if t.size != (tile_px, tile_px):
        t = t.resize((tile_px, tile_px), Image.BILINEAR)
    return np.asarray(t)


def color_stats(rgb):
    f = rgb.astype("float64") / 255.0
    hed = rgb2hed(f)
    return (f.mean(),                       # brightness
            rgb2hsv(f)[..., 1].mean(),      # saturation
            hed[..., 0].mean(),             # hematoxylin (nuclei / blue-purple)
            hed[..., 1].mean())             # eosin (cytoplasm / pink)


STAT_NAMES = ["brightness", "saturation", "hematoxylin", "eosin"]


def montage(tiles, cols=10, cell=112):
    rows = (len(tiles) + cols - 1) // cols
    canvas = Image.new("RGB", (cols * cell, rows * cell), (245, 245, 245))
    for i, t in enumerate(tiles):
        canvas.paste(Image.fromarray(t).resize((cell, cell)), ((i % cols) * cell, (i // cols) * cell))
    return canvas


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--slide-prefix", default="TCGA")
    ap.add_argument("--wsi", required=True, help="s3:// to the slide's .svs")
    ap.add_argument("--list-key", default="embeddings/lists/global.npz")
    ap.add_argument("--pos", default="TUM", help="cancer axis positive class (one-vs-rest)")
    ap.add_argument("--n-rank", type=int, default=60)
    ap.add_argument("--n-corr", type=int, default=400)
    ap.add_argument("--out", default="/tmp/color_inspect")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    gl = load_global(s3.get_object(Bucket=B, Key=args.list_key)["Body"].read())
    slides = np.asarray(gl.meta["slide"]).astype(str)
    m = np.char.startswith(slides, args.slide_prefix)
    V = np.asarray(gl.vectors)[m]
    coords = np.asarray(gl.meta["coords"])[m]
    print(f"slide {args.slide_prefix}: {len(V)} tiles", flush=True)

    feats, labels, cn, _ = loader.load("h_optimus_0", "train")
    labels, cn = np.asarray(labels), list(cn)
    fit = P.fit_probe(np.asarray(feats), (labels == cn.index(args.pos)).astype(int))
    scores = ((V - fit["scaler"].mean_) / fit["scaler"].scale_) @ fit["direction"]
    order = np.argsort(-scores)

    b, k = args.wsi[5:].split("/", 1)
    local = os.path.join(tempfile.gettempdir(), os.path.basename(k))
    if not os.path.exists(local):
        print(f"downloading {args.wsi} ...", flush=True)
        s3.download_file(b, k, local)
    reader = open_wsi(local)

    # 1) correlation on a random sample spanning the score range
    rng = np.random.default_rng(0)
    ridx = rng.choice(len(V), min(args.n_corr, len(V)), replace=False)
    stats = np.array([color_stats(crop_tile(reader, *coords[i])) for i in ridx])
    print(f"\n[{args.slide_prefix}] axis={args.pos}-one-vs-rest — within-slide corr(score, stain) "
          f"on {len(ridx)} random tiles:", flush=True)
    for j, name in enumerate(STAT_NAMES):
        r = float(np.corrcoef(scores[ridx], stats[:, j])[0, 1])
        flag = "  <-- stain-driven" if abs(r) >= 0.5 else ""
        print(f"    {name:12s} r = {r:+.3f}{flag}", flush=True)

    # 2) montages to eyeball morphology vs color
    top = [crop_tile(reader, *coords[i]) for i in order[:args.n_rank]]
    bot = [crop_tile(reader, *coords[i]) for i in order[-args.n_rank:]]
    montage(top).save(os.path.join(args.out, f"{args.slide_prefix}_top.png"))
    montage(bot).save(os.path.join(args.out, f"{args.slide_prefix}_bottom.png"))
    print(f"\nsaved montages: {args.out}/{args.slide_prefix}_top.png (most cancer-like) + _bottom.png",
          flush=True)
    print(f"score range: top mean {scores[order[:args.n_rank]].mean():+.3f} | "
          f"bottom mean {scores[order[-args.n_rank:]].mean():+.3f}", flush=True)


if __name__ == "__main__":
    main()
