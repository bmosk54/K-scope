"""Extract H-Optimus-0 PATCH-token activations -- the cheap way to get ~20x more SAE data.

A 224x224 tile produces 256 patch tokens (16x16 grid at patch14) plus 5 prefix tokens.
Training the SAE on CLS only throws away 256 vectors per tile and leaves us with ~90k
training samples for 6144 features (~15 samples/feature; the SAE literature uses millions).
The model already computes the patch tokens on every forward pass -- keeping them costs
nothing extra at inference time.

We sample PATCHES_PER_TILE patches per tile rather than all 256, purely to keep the artifact
around 5GB instead of 78GB. The forward pass is the expensive part, not the token count,
so this is nearly free.

Bonus: patch features are SPATIALLY LOCALISED. With `patch_pos` recorded we can render a
heatmap of exactly WHERE inside a tile a feature fires, instead of only which tiles it likes.

Writes artifacts/hoptimus_patches.npz:
    feats      (N*P, 1536) float16   post-LN patch tokens at block 39
    tile_ids   (N*P,)      int64     which tile each patch came from
    patch_pos  (N*P,)      int16     0..255, position in the 16x16 grid
    labels     (N*P,)      int64     tile-level tissue label (inherited)
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
LAYER = 39
DIM = 1536
GRID = 16  # 224 / 14
_MEAN = (0.707223, 0.578729, 0.703617)
_STD = (0.211883, 0.230117, 0.177517)


class ShardDS(Dataset):
    def __init__(self, path, tf, offset):
        self.tf = tf
        t = pq.read_table(path, columns=["image", "label"])
        self.rows = [
            (im["bytes"], lb) for im, lb in zip(t.column("image").to_pylist(), t.column("label").to_pylist())
        ]
        self.offset = offset

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        raw, lbl = self.rows[i]
        return self.tf(Image.open(io.BytesIO(raw)).convert("RGB")), lbl, self.offset + i


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="/home/sagemaker-user/biolayer/data/nct_crc_he/data")
    ap.add_argument("--out", default="/home/sagemaker-user/biolayer/artifacts/hoptimus_patches.npz")
    ap.add_argument("--patches-per-tile", type=int, default=16)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if not os.environ.get("HF_TOKEN"):
        raise SystemExit("HF_TOKEN required")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    shards = sorted(glob.glob(os.path.join(args.data_dir, "NCT_CRC_HE_100K-*.parquet")))
    counts = [pq.ParquetFile(s).metadata.num_rows for s in shards]
    n_tiles = sum(counts)
    if args.limit:
        n_tiles = min(n_tiles, args.limit)
    P = args.patches_per_tile
    total = n_tiles * P

    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=_MEAN, std=_STD)])
    model = timm.create_model(MODEL_ID, pretrained=True, init_values=1e-5, dynamic_img_size=False)
    model.eval().to(dev)
    assert model.embed_dim == DIM
    print(f"tiles={n_tiles} patches/tile={P} total_vectors={total} (~{total*DIM*2/1e9:.1f}GB)", flush=True)

    feats = np.zeros((total, DIM), dtype=np.float16)
    tile_ids = np.zeros(total, dtype=np.int64)
    patch_pos = np.zeros(total, dtype=np.int16)
    labels = np.zeros(total, dtype=np.int64)

    gen = torch.Generator().manual_seed(args.seed)
    w = 0
    seen = 0
    t0 = time.time()
    offset = 0
    for shard, cnt in zip(shards, counts):
        if seen >= n_tiles:
            break
        ds = ShardDS(shard, tf, offset)
        offset += cnt
        dl = DataLoader(ds, batch_size=args.batch_size, num_workers=args.workers, pin_memory=True)
        for x, y, tid in dl:
            if seen >= n_tiles:
                break
            x = x.to(dev, non_blocking=True)
            b = x.shape[0]
            with torch.inference_mode():
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=dev == "cuda"):
                    outs = model.get_intermediate_layers(
                        x, n=[LAYER], return_prefix_tokens=True, norm=True
                    )
                patch, _ = outs[0]  # (B, 256, 1536) -- registers already excluded
            # sample P patch positions per tile (same positions across the batch is fine and
            # cheaper; re-drawn every batch so coverage is uniform over the corpus)
            sel = torch.randperm(patch.shape[1], generator=gen)[:P]
            sub = patch[:, sel].float().cpu().numpy().astype(np.float16)  # (B, P, D)
            k = b * P
            feats[w : w + k] = sub.reshape(-1, DIM)
            tile_ids[w : w + k] = np.repeat(tid.numpy(), P)
            patch_pos[w : w + k] = np.tile(sel.numpy().astype(np.int16), b)
            labels[w : w + k] = np.repeat(y.numpy(), P)
            w += k
            seen += b
            if seen % (args.batch_size * 40) < args.batch_size:
                r = seen / (time.time() - t0)
                print(f"  {seen}/{n_tiles} tiles  {r:.0f} tiles/s  eta {(n_tiles-seen)/max(r,1e-6)/60:.1f} min", flush=True)

    feats, tile_ids, patch_pos, labels = feats[:w], tile_ids[:w], patch_pos[:w], labels[:w]
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez(
        args.out,
        feats=feats,
        tile_ids=tile_ids,
        patch_pos=patch_pos,
        labels=labels,
        layers=np.asarray([LAYER]),
        grid=np.asarray([GRID]),
        class_names=np.asarray(
            json.loads(pq.ParquetFile(shards[0]).schema_arrow.metadata[b"huggingface"].decode())["info"][
                "features"
            ]["label"]["names"]
        ),
    )
    print(f"wrote {args.out}  feats={feats.shape} in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
