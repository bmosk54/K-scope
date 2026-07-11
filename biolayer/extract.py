"""Extract frozen CLS embeddings for a stratified NCT-CRC-HE subset and push to S3.

    python -m biolayer.extract --model phikon_v2 --split train
    python -m biolayer.extract --model h0_mini   --split val --per-class 800

Streams the HF dataset with image decode DISABLED so tiles we skip (once a class
is full) are never decoded — a ~10k stratified subset is pulled from 100k cheaply
on CPU. Saves one .npz {feats, labels, class_names} locally, then uploads to
s3://bucketbiolayer/embeddings/<dataset>/<model>/<split>.npz.
"""
import argparse
import io
import os
import time

import numpy as np
from PIL import Image

from . import config, s3_utils
from .models import DEVICE, load_encoder


def _stratified_stream(hf_split, per_class, seed, shuffle_buffer):
    """Yield (PIL.Image, label_int) building a per-class-balanced subset.

    Reads labels first and only decodes the image bytes for tiles we keep, so
    filling 9 classes x per_class from a 100k stream stays fast on CPU.
    """
    from datasets import Image as HFImage, load_dataset

    ds = load_dataset(config.DATASET_ID, split=hf_split, streaming=True)
    ds = ds.cast_column(config.IMAGE_COLUMN, HFImage(decode=False))  # keep raw bytes
    if shuffle_buffer > 0:
        ds = ds.shuffle(seed=seed, buffer_size=shuffle_buffer)

    n_classes = len(config.CLASS_NAMES)
    counts = [0] * n_classes
    kept = 0
    for ex in ds:
        lbl = int(ex[config.LABEL_COLUMN])
        if counts[lbl] >= per_class:
            if all(c >= per_class for c in counts):
                break
            continue
        raw = ex[config.IMAGE_COLUMN]
        img = Image.open(io.BytesIO(raw["bytes"])).convert("RGB")
        counts[lbl] += 1
        kept += 1
        if kept % 500 == 0:
            print(f"  collected {kept} tiles  per-class={counts}", flush=True)
        yield img, lbl
    print(f"  final per-class counts = {counts} (total {sum(counts)})", flush=True)


def extract(model_key, split, per_class, batch_size, seed, shuffle_buffer,
            out_dir, upload):
    hf_split = config.resolve_split(split)
    spec = config.MODELS[model_key]
    print(f"[extract] model={model_key} ({spec['hf_id']}) split={split}->{hf_split} "
          f"per_class={per_class} batch={batch_size} device={DEVICE}")
    if spec["gated"]:
        print(f"[extract] NOTE: {model_key} is GATED — needs `huggingface-cli login` "
              f"+ accepted terms at hf.co/{spec['hf_id']}")

    print("[extract] loading encoder ...", flush=True)
    embed, spec = load_encoder(model_key)

    # Gather the stratified subset (PIL images + labels) up front.
    t0 = time.time()
    images, labels = [], []
    for img, lbl in _stratified_stream(hf_split, per_class, seed, shuffle_buffer):
        images.append(img)
        labels.append(lbl)
    print(f"[extract] gathered {len(images)} tiles in {time.time()-t0:.1f}s", flush=True)

    # Batched embedding.
    t0 = time.time()
    feats = []
    for i in range(0, len(images), batch_size):
        batch = images[i:i + batch_size]
        feats.append(embed(batch))
        done = min(i + batch_size, len(images))
        if done % (batch_size * 10) == 0 or done == len(images):
            print(f"  embedded {done}/{len(images)}  "
                  f"({done/(time.time()-t0):.1f} tiles/s)", flush=True)
    feats = np.concatenate(feats, axis=0).astype(np.float32)
    labels = np.asarray(labels, dtype=np.int64)
    assert feats.shape[1] == spec["dim"], \
        f"dim mismatch: got {feats.shape[1]}, expected {spec['dim']}"
    print(f"[extract] embedded {feats.shape} in {time.time()-t0:.1f}s", flush=True)

    # Save .npz locally (S3-mirroring path).
    local_path = os.path.join(
        out_dir, config.embeddings_key(model_key, split))
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    np.savez_compressed(
        local_path, feats=feats, labels=labels,
        class_names=np.array(config.CLASS_NAMES))
    print(f"[extract] saved {local_path} "
          f"({os.path.getsize(local_path)/1e6:.1f} MB)")

    # Upload to the shared bucket.
    if upload:
        key = config.embeddings_key(model_key, split)
        uri = s3_utils.upload_file(local_path, key)
        print(f"[extract] uploaded -> {uri}")
    else:
        print("[extract] --no-upload set; skipping S3 upload")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, choices=list(config.MODELS))
    p.add_argument("--split", default="train",
                   help="friendly split: train | val | test | train_nonorm")
    p.add_argument("--per-class", type=int, default=1100,
                   help="tiles per class (9 classes -> ~10k total)")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--shuffle-buffer", type=int, default=3000,
                   help="streaming shuffle buffer for tile diversity; 0 disables")
    p.add_argument("--out-dir", default="artifacts",
                   help="local dir mirroring the S3 layout")
    p.add_argument("--no-upload", action="store_true")
    args = p.parse_args()

    extract(
        model_key=args.model, split=args.split, per_class=args.per_class,
        batch_size=args.batch_size, seed=args.seed,
        shuffle_buffer=args.shuffle_buffer, out_dir=args.out_dir,
        upload=not args.no_upload,
    )


if __name__ == "__main__":
    main()
