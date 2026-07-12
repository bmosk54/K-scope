"""Extract H-Optimus-0 embeddings for the NCT-CRC-HE 100K tiles at every probed depth.

Track config (ARCHITECTURE.md sec.3): bioptimus/H-optimus-0, timm, 1536-d, 40 blocks,
probed at blocks 13/27/39 (0-indexed; 39 is the final block). Each depth is emitted as
{global CLS, local mean-patch}, so the SAE can be trained in whichever representation
Eddie's probe directions live in without a re-extract.

TWO THINGS THAT WILL SILENTLY CORRUPT THIS IF YOU GET THEM WRONG:

  1. num_prefix_tokens == 5, NOT 1. H-Optimus-0 has 1 CLS + 4 REGISTER tokens.
     (DESIGN_MIL_AGGREGATOR.md sec.5.1 claims registers are an H0-mini-only concern and
     that H-optimus-0 has num_prefix_tokens == 1 -- verified false at runtime.) Slicing
     patches as tokens[:, 1:] would pull 4 high-norm register tokens in with the 256
     patch tokens and quietly poison every local mean-patch vector. We use
     get_intermediate_layers(return_prefix_tokens=True), which separates them properly,
     and we assert the count.

  2. norm=True. The final layernorm must be applied at EVERY probed depth. Verified on
     Phikon: the raw residual stream has ~15x the norm of the post-LN CLS that the probe
     directions are fit in. Get this wrong and the SAE trains in a different space than
     the probes, and every feature/probe cosine is meaningless -- with no error raised.

Writes artifacts/hoptimus_100k.npz:
    globals     (N, 3, 1536) float16   CLS (prefix[:, 0]) at blocks 13/27/39
    locals      (N, 3, 1536) float16   mean over the 256 PATCH tokens at those blocks
    labels      (N,)         int64
    tile_ids    (N,)         int64     row index into the parquet shards, in order
    layers      (3,)         int64     [13, 27, 39]
    class_names (9,)         <U4
"""

import argparse
import glob
import io
import json
import os
import time

import numpy as np
import pyarrow.parquet as pq
import timm
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

MODEL_ID = "hf-hub:bioptimus/H-optimus-0"
LAYERS = [13, 27, 39]  # 0-indexed timm block indices; 39 == final block of 40
DIM = 1536
EXPECTED_PREFIX_TOKENS = 5  # 1 CLS + 4 registers

# H-Optimus-0's official normalization (matches the team's model.py).
_MEAN = (0.707223, 0.578729, 0.703617)
_STD = (0.211883, 0.230117, 0.177517)


class ShardDS(Dataset):
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
    ap.add_argument("--out", default="/home/sagemaker-user/biolayer/artifacts/hoptimus_100k.npz")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    if not os.environ.get("HF_TOKEN"):
        raise SystemExit("HF_TOKEN required (bioptimus/H-optimus-0 is gated)")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    shards = sorted(glob.glob(os.path.join(args.data_dir, "NCT_CRC_HE_100K-*.parquet")))
    if not shards:
        raise SystemExit(f"no parquet shards under {args.data_dir}")

    meta = pq.ParquetFile(shards[0]).schema_arrow.metadata
    class_names = json.loads(meta[b"huggingface"].decode())["info"]["features"]["label"]["names"]
    n = sum(pq.ParquetFile(s).metadata.num_rows for s in shards)
    if args.limit:
        n = min(n, args.limit)

    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=_MEAN, std=_STD)])

    model = timm.create_model(MODEL_ID, pretrained=True, init_values=1e-5, dynamic_img_size=False)
    model.eval().to(device)

    n_prefix = getattr(model, "num_prefix_tokens", 1)
    assert model.embed_dim == DIM, f"expected embed_dim {DIM}, got {model.embed_dim}"
    assert n_prefix == EXPECTED_PREFIX_TOKENS, (
        f"num_prefix_tokens changed: expected {EXPECTED_PREFIX_TOKENS} (1 CLS + 4 registers), "
        f"got {n_prefix}. Re-check which tokens are patches before trusting `locals`."
    )
    assert max(LAYERS) < len(model.blocks), f"layer {max(LAYERS)} >= {len(model.blocks)} blocks"
    print(
        f"device={device} tiles={n} dim={model.embed_dim} blocks={len(model.blocks)} "
        f"prefix_tokens={n_prefix} layers={LAYERS}",
        flush=True,
    )

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
                    outs = model.get_intermediate_layers(
                        x, n=LAYERS, return_prefix_tokens=True, norm=True
                    )
                for j, (patch, prefix) in enumerate(outs):
                    # patch: (B, 256, 1536) -- true patch tokens, registers already excluded
                    # prefix: (B, 5, 1536)  -- [CLS, reg, reg, reg, reg]
                    globals_[i : i + k, j] = prefix[:k, 0].float().cpu().numpy().astype(np.float16)
                    locals_[i : i + k, j] = patch[:k].mean(1).float().cpu().numpy().astype(np.float16)
            labels[i : i + k] = y.numpy()[:k]
            i += k
            if i % (args.batch_size * 40) < args.batch_size:
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
