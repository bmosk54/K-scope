"""Extract multi-layer, local+global embeddings for a TRACK and push to S3.

    python -m biolayer.data.extract --track phikon --split train
    python -m biolayer.data.extract --track h0     --split train --per-class 800

A track (biolayer.tracks) bundles the model + dataset + objective + the 3 layers.
For every tile we store BOTH global (CLS) and local (mean patch) features at all 3
layers, so the causal battery can run layer-resolved and on either space.

Saved .npz keys:
    globals     (N, L, dim)   CLS token at each of the L=3 layers
    locals      (N, L, dim)   mean patch token at each layer
    feats       (N, dim)      = globals[:, readout, :]  (back-compat: old CLS feature)
    labels      (N,)          class index
    class_names (C,)  layers (L,)  layer_names (L,)
"""
import argparse
import io
import os
import time

import numpy as np
from PIL import Image

from .. import config
from . import s3_utils
from .models import DEVICE, load_encoder


def _stratified_stream(track, hf_split, per_class, seed, shuffle_buffer):
    """Yield (PIL.Image, label_int) building a per-class-balanced subset.

    Reads labels first and only decodes image bytes for tiles we keep.
    """
    from datasets import Image as HFImage, load_dataset

    ds = load_dataset(track.dataset_id, split=hf_split, streaming=True)
    ds = ds.cast_column(track.image_column, HFImage(decode=False))  # keep raw bytes
    if shuffle_buffer > 0:
        ds = ds.shuffle(seed=seed, buffer_size=shuffle_buffer)

    n_classes = len(track.class_names)
    counts = [0] * n_classes
    kept = 0
    for ex in ds:
        lbl = int(ex[track.label_column])
        if counts[lbl] >= per_class:
            if all(c >= per_class for c in counts):
                break
            continue
        raw = ex[track.image_column]
        img = Image.open(io.BytesIO(raw["bytes"])).convert("RGB")
        counts[lbl] += 1
        kept += 1
        if kept % 500 == 0:
            print(f"  collected {kept} tiles  per-class={counts}", flush=True)
        yield img, lbl
    print(f"  final per-class counts = {counts} (total {sum(counts)})", flush=True)


def extract(track, split, per_class, batch_size, seed, shuffle_buffer,
            out_dir, upload):
    spec = track.spec
    hf_split = track.resolve_split(split)
    layers = list(track.layers)
    print(f"[extract] track={track.name} model={track.model_key} ({spec['hf_id']}) "
          f"split={split}->{hf_split} per_class={per_class} layers={layers} "
          f"device={DEVICE}")
    print(f"[extract] objective: {track.objective.description}")
    if spec["gated"]:
        print(f"[extract] NOTE: {track.model_key} is GATED — needs `hf auth login` "
              f"+ accepted terms at hf.co/{spec['hf_id']}")

    print("[extract] loading encoder ...", flush=True)
    embed, spec = load_encoder(track.model_key)

    # Gather the stratified subset up front.
    t0 = time.time()
    images, labels = [], []
    for img, lbl in _stratified_stream(track, hf_split, per_class, seed, shuffle_buffer):
        images.append(img)
        labels.append(lbl)
    print(f"[extract] gathered {len(images)} tiles in {time.time()-t0:.1f}s", flush=True)

    # Batched multi-layer embedding: two (B, L, dim) arrays per batch.
    t0 = time.time()
    g_all, l_all = [], []
    for i in range(0, len(images), batch_size):
        g, l = embed(images[i:i + batch_size])
        g_all.append(g)
        l_all.append(l)
        done = min(i + batch_size, len(images))
        if done % (batch_size * 10) == 0 or done == len(images):
            print(f"  embedded {done}/{len(images)}  "
                  f"({done/(time.time()-t0):.1f} tiles/s)", flush=True)
    globals_ = np.concatenate(g_all, axis=0).astype(np.float32)   # (N, L, dim)
    locals_ = np.concatenate(l_all, axis=0).astype(np.float32)    # (N, L, dim)
    labels = np.asarray(labels, dtype=np.int64)
    assert globals_.shape[1:] == (len(layers), spec["dim"]), \
        f"shape mismatch: got {globals_.shape}, expected (N,{len(layers)},{spec['dim']})"
    print(f"[extract] embedded globals={globals_.shape} locals={locals_.shape} "
          f"in {time.time()-t0:.1f}s", flush=True)

    # readout layer = last of the configured triple -> back-compat single CLS feature.
    feats = globals_[:, -1, :]

    local_path = os.path.join(out_dir, track.embeddings_key(split))
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    np.savez_compressed(
        local_path,
        globals=globals_, locals=locals_, feats=feats, labels=labels,
        class_names=np.array(track.class_names),
        layers=np.array(layers),
        layer_names=np.array(config.LAYER_NAMES),
    )
    print(f"[extract] saved {local_path} ({os.path.getsize(local_path)/1e6:.1f} MB)")

    if upload:
        uri = s3_utils.upload_file(local_path, track.embeddings_key(split))
        print(f"[extract] uploaded -> {uri}")
    else:
        print("[extract] --no-upload set; skipping S3 upload")


def main():
    from .. import tracks

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--track", default="phikon", choices=list(tracks.TRACKS),
                   help="which model pipeline to run (phikon | h0)")
    p.add_argument("--split", default="train",
                   help="friendly split: train | val | test | train_nonorm")
    p.add_argument("--per-class", type=int, default=1100,
                   help="tiles per class")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--shuffle-buffer", type=int, default=3000)
    p.add_argument("--out-dir", default="artifacts")
    p.add_argument("--no-upload", action="store_true")
    args = p.parse_args()

    extract(
        track=tracks.get(args.track), split=args.split, per_class=args.per_class,
        batch_size=args.batch_size, seed=args.seed,
        shuffle_buffer=args.shuffle_buffer, out_dir=args.out_dir,
        upload=not args.no_upload,
    )


if __name__ == "__main__":
    main()
