"""Extract Phikon-v2 embeddings for the NCT-CRC-HE 100K tiles, at every depth
and representation the causal battery probes.

ARCHITECTURE.md sec.3: the `phikon` track is owkin/phikon-v2, 1024-d, 24 blocks,
probed at layers 8/16/24, each as {global CLS, local mean-patch}. We emit all six
so the SAE can be trained in whichever space Eddie's probe directions turn out to
live in, without a re-extract.

Writes artifacts/phikon_100k.npz:
    globals     (N, 3, 1024) float16   CLS token at layers 8/16/24
    locals      (N, 3, 1024) float16   mean over patch tokens at layers 8/16/24
    labels      (N,)         int64     0..8
    tile_ids    (N,)         int64     row index into the parquet shards, in order
    layers      (3,)         int64     [8, 16, 24]
    class_names (9,)         <U4

tile_ids is the join key back to the raw image, so `hypothesis` can hand a
pathologist the actual top-activating tiles for a feature.
"""

import argparse
import glob
import io
import json
import os
import time

import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from transformers import AutoImageProcessor, AutoModel

MODEL_ID = "owkin/phikon-v2"
LAYERS = [8, 16, 24]  # hidden_states index; 0 is the embedding output, so these are blocks 8/16/24
DIM = 1024


class ShardDS(Dataset):
    """One parquet shard, held in memory. Streaming shard-by-shard keeps peak RSS at
    ~one shard (~300MB) instead of the full 8.6GB of tile bytes."""

    def __init__(self, path, tf, limit=0):
        self.tf = tf
        t = pq.read_table(path, columns=["image", "label"])
        imgs = t.column("image").to_pylist()
        lbls = t.column("label").to_pylist()
        if limit:
            imgs, lbls = imgs[:limit], lbls[:limit]
        self.rows = [(im["bytes"], lb) for im, lb in zip(imgs, lbls)]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        raw, lbl = self.rows[i]
        return self.tf(Image.open(io.BytesIO(raw)).convert("RGB")), lbl


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="/home/sagemaker-user/biolayer/data/nct_crc_he/data")
    ap.add_argument("--out", default="/home/sagemaker-user/biolayer/artifacts/phikon_100k.npz")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    shards = sorted(glob.glob(os.path.join(args.data_dir, "NCT_CRC_HE_100K-*.parquet")))
    if not shards:
        raise SystemExit(f"no parquet shards under {args.data_dir}")

    meta = pq.ParquetFile(shards[0]).schema_arrow.metadata
    class_names = json.loads(meta[b"huggingface"].decode())["info"]["features"]["label"]["names"]

    proc = AutoImageProcessor.from_pretrained(MODEL_ID)
    tf = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=proc.image_mean, std=proc.image_std),
        ]
    )

    n = sum(pq.ParquetFile(s).metadata.num_rows for s in shards)
    if args.limit:
        n = min(n, args.limit)
    print(f"device={device}  tiles={n}  layers={LAYERS}  classes={class_names}", flush=True)

    model = AutoModel.from_pretrained(MODEL_ID).eval().to(device)

    globals_ = np.zeros((n, len(LAYERS), DIM), dtype=np.float16)
    locals_ = np.zeros((n, len(LAYERS), DIM), dtype=np.float16)
    labels = np.zeros(n, dtype=np.int64)

    i = 0
    t0 = time.time()
    for shard in shards:
        if i >= n:
            break
        ds = ShardDS(shard, tf, limit=(n - i) if args.limit else 0)
        dl = DataLoader(ds, batch_size=args.batch_size, num_workers=args.workers, pin_memory=True)
        for x, y in dl:
            if i >= n:
                break
            x = x.to(device, non_blocking=True)
            k = min(x.shape[0], n - i)
            with torch.inference_mode():
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device == "cuda"):
                    out = model(pixel_values=x, output_hidden_states=True)
                for j, L in enumerate(LAYERS):
                    # hidden_states[L] is the PRE-final-layernorm residual stream. Eddie's
                    # cached features are post-LN CLS (verified: norm 8.80 vs his 8.73), and
                    # the battery's extractor uses get_intermediate_layers(norm=True) -- the
                    # final norm is applied at EVERY probed depth. Reproduce that, or the SAE
                    # lands in a space 15x larger than the probe directions and every cosine
                    # alignment is silently meaningless.
                    h = model.layernorm(out.hidden_states[L])  # (B, 197, 1024)
                    globals_[i : i + k, j] = h[:k, 0].float().cpu().numpy().astype(np.float16)
                    locals_[i : i + k, j] = h[:k, 1:].mean(1).float().cpu().numpy().astype(np.float16)
            labels[i : i + k] = y.numpy()[:k]
            i += k
            if i % (args.batch_size * 20) < args.batch_size:
                rate = i / (time.time() - t0)
                print(f"  {i}/{n}  {rate:.0f} tiles/s  eta {(n-i)/max(rate,1e-6)/60:.1f} min", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez(
        args.out,
        globals=globals_,
        locals=locals_,
        labels=labels,
        tile_ids=np.arange(n, dtype=np.int64),
        layers=np.asarray(LAYERS, dtype=np.int64),
        class_names=np.asarray(class_names),
    )
    print(f"wrote {args.out}  globals={globals_.shape} in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
