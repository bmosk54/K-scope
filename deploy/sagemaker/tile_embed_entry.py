"""SageMaker GPU entry: WSI (S3) -> tiles -> H-optimus-0 -> TWO ordered vector lists.

Runs on ml.g5.2xlarge. Embeds EVERY sensible tile (tissue-masked + filtered; no cap
unless MAX_TILES is set) and, per 224px tile, keeps BOTH:
  * the CLS "257th" vector  -> the GLOBAL list  (one 1536-d vector per tile)
  * the 256 patch vectors   -> the PATCH  list  (256 x 1536-d per tile, row-major)

Both come out as ORDERED vector lists (tile-major; patches then patch-row-major) with
aligned metadata, so a future mech-interp scoring pass can rerank them without moving the
vectors (see biolayer/vectors/ordered_list.py). Full-slide patch tensors never sit in RAM:
they stream to an on-disk memmap sized from the tile manifest, then upload as a per-slide
shard referenced by a manifest.

Artifacts (per slide, then combined):
  s3://<bucket>/embeddings/wsi/<slide>/global.npz          CLS + coords + keys (fp32)
  s3://<bucket>/embeddings/wsi/<slide>/patch_vectors.npy   (N*P, 1536) memmap shard (fp16)
  s3://<bucket>/embeddings/wsi/<slide>/patch_meta.npz      row-aligned tile/patch indices
  s3://<bucket>/embeddings/lists/global.npz                LIST 1 (all tiles, all slides)
  s3://<bucket>/embeddings/lists/patch.manifest.json       LIST 2 (sharded, in order)

Env:
  SLIDE_S3 / SLIDES_S3 / MANIFEST_S3   slides to process (required)
  SM_BUCKET        feature-output bucket (default bucketbiolayer)
  FILTERS          post-tiling filters   (default whitespace,tissue)  -> "sensible" tiles
  MPP / TILE_PX    target magnification / tile size (default 0.5 / 224)
  MAX_TILES        optional cap per slide (trial runs); unset = ALL sensible tiles
  PATCH_DTYPE      patch-vector dtype (default float16 — half the size/RAM of fp32)
  VECTOR_BUCKET_ARN / VECTOR_INDEX     optional: also push the GLOBAL list to S3 Vectors
  HF_TOKEN         only used on a cold model cache
"""
import io
import json
import os
import tempfile

import boto3
import numpy as np
import torch
from numpy.lib.format import open_memmap

import hoptimus
import tile_wsi


def _resolve_hf_token():
    """HF token from env, else the hyperparameters file (where launch puts it to
    dodge the 512-char Environment cap). Values there are JSON-encoded."""
    if os.environ.get("HF_TOKEN"):
        return os.environ["HF_TOKEN"]
    p = "/opt/ml/input/config/hyperparameters.json"
    if os.path.exists(p):
        v = json.load(open(p)).get("HF_TOKEN")
        if v:
            try:
                return json.loads(v)
            except Exception:
                return v
    return None


# H-optimus-0 (~4 GB) is downloaded from HuggingFace ONCE and cached in S3; every later
# job restores it from S3 in-region (fast + no HF rate limits) and loads offline.
MODEL_CACHE_KEY = "models/hf-cache-h-optimus-0.tar"


def _hf_home():
    p = "/opt/ml/hf-cache"                       # on the 200 GB job volume
    os.environ["HF_HOME"] = p
    os.makedirs(p, exist_ok=True)
    return p


def _restore_model_cache(s3, bucket):
    """Restore the HF cache from S3 if present → offline load, no HF download."""
    import tarfile
    home = _hf_home()
    tarp = "/opt/ml/hf-cache.tar"
    try:
        s3.download_file(bucket, MODEL_CACHE_KEY, tarp)
    except Exception:
        return False
    with tarfile.open(tarp) as t:
        t.extractall(home)
    os.environ["HF_HUB_OFFLINE"] = "1"
    print(f"[embed] restored model cache from s3://{bucket}/{MODEL_CACHE_KEY}", flush=True)
    return True


def _seed_model_cache(s3, bucket):
    """Upload the freshly-downloaded HF cache to S3 for next time."""
    import tarfile
    tarp = "/opt/ml/hf-cache.tar"
    with tarfile.open(tarp, "w") as t:                 # uncompressed: weights already binary
        t.add(os.environ["HF_HOME"], arcname=".")
    s3.upload_file(tarp, bucket, MODEL_CACHE_KEY)
    print(f"[embed] seeded model cache -> s3://{bucket}/{MODEL_CACHE_KEY}", flush=True)


def _build_transform(model):
    from timm.data import create_transform, resolve_data_config
    return create_transform(**resolve_data_config(model.pretrained_cfg, model=model))


@torch.inference_mode()
def _embed_slide(model, tf, tile_dir, device, patch_path, batch=32, patch_dtype="float16"):
    """Embed every kept tile. Returns (G, coords, kept, N, P):
      G       (N, D) fp32   — CLS "257th" vector per tile (the GLOBAL list rows)
      coords  (N, 2)        — tile (x, y) at level 0
      P                     — patch-token count per tile (256 for H-optimus-0 @ 224/14)
    The N*P patch tokens STREAM to an on-disk memmap at `patch_path` (shape (N*P, D),
    `patch_dtype`) so a full slide's patch tensor never sits in RAM.
    """
    from PIL import Image, ImageFile, PngImagePlugin
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    PngImagePlugin.MAX_TEXT_CHUNK = 100 * 1024 * 1024

    rows = [json.loads(ln) for ln in open(os.path.join(tile_dir, "manifest.jsonl")) if ln.strip()]
    kept = [r for r in rows if r.get("file")]          # tiling order == list order
    N, D = len(kept), model.embed_dim
    G = np.zeros((N, D), dtype="float32")
    patch_mm, P = None, None
    for i in range(0, N, batch):
        chunk = kept[i:i + batch]
        x = torch.stack([tf(Image.open(os.path.join(tile_dir, r["file"])).convert("RGB"))
                         for r in chunk]).to(device)
        outs = model.get_intermediate_layers(x, n=1, return_prefix_tokens=True, norm=True)
        patches = outs[-1][0].float().cpu().numpy()    # (b, P, D) — spatial tokens
        cls = outs[-1][1][:, 0].float().cpu().numpy()  # (b, D)   — CLS (the 257th vector)
        b = len(chunk)
        G[i:i + b] = cls
        if patch_mm is None:                           # allocate once P is known
            P = patches.shape[1]
            patch_mm = open_memmap(patch_path, mode="w+", dtype=np.dtype(patch_dtype),
                                   shape=(N * P, D))
        patch_mm[i * P:(i + b) * P] = patches.reshape(b * P, D).astype(patch_dtype)
    if patch_mm is not None:
        patch_mm.flush()
    coords = np.array([(r["x"], r["y"]) for r in kept], dtype=np.int64)
    return G, coords, kept, N, (P or 0)


def _push_vectors(F, coords, stem, bucket_arn, index, region):
    c = boto3.client("s3vectors", region_name=region)          # explicit region (no default)
    index_arn = f"{bucket_arn}/index/{index}"                   # put_vectors wants indexArn
    vecs = [{"key": f"{stem}/{int(x)}_{int(y)}",
             "data": {"float32": F[i].astype("float32").tolist()},
             "metadata": {"slide": stem, "x": int(x), "y": int(y)}}
            for i, (x, y) in enumerate(coords)]
    for i in range(0, len(vecs), 500):                          # API batches ~500
        c.put_vectors(indexArn=index_arn, vectors=vecs[i:i + 500])
    return len(vecs)


def _slides_from_env(s3):
    """Slides to process: SLIDE_S3 (single) + SLIDES_S3 (comma) + MANIFEST_S3 (S3 list)."""
    out = []
    if os.environ.get("SLIDE_S3"):
        out.append(os.environ["SLIDE_S3"])
    out += [x.strip() for x in os.environ.get("SLIDES_S3", "").split(",") if x.strip()]
    if os.environ.get("MANIFEST_S3"):
        p = os.environ["MANIFEST_S3"][5:].split("/", 1)
        body = s3.get_object(Bucket=p[0], Key=p[1])["Body"].read().decode()
        out += [ln.strip() for ln in body.splitlines() if ln.strip() and not ln.startswith("#")]
    return out


def process_slide(slide_s3, s3, model, tf, device, cfg):
    """One slide: download → tile (all sensible) → embed → per-slide global + patch shards.
    Returns a record the caller folds into the two combined ordered lists."""
    b, k = slide_s3[5:].split("/", 1)
    local = os.path.join(tempfile.gettempdir(), os.path.basename(k))
    print(f"[embed] downloading {slide_s3}", flush=True)
    s3.download_file(b, k, local)
    stem = os.path.splitext(os.path.basename(k))[0]
    prefix = f"embeddings/wsi/{stem}"

    tile_dir = os.path.join(tempfile.gettempdir(), "tiles", stem)
    tile_wsi.tile_slide(local, tile_dir, tile_px=cfg["tile_px"], target_mpp=cfg["mpp"],
                        filters=cfg["filters"], max_tiles=cfg["max_tiles"])

    patch_path = os.path.join(tempfile.gettempdir(), f"{stem}_patch.npy")
    G, coords, kept, N, P = _embed_slide(model, tf, tile_dir, device, patch_path,
                                         patch_dtype=cfg["patch_dtype"])
    print(f"[embed] {stem}: {N} tiles -> global {G.shape}, "
          f"patch list ({N * P}, {G.shape[1]}) P={P} dtype={cfg['patch_dtype']}", flush=True)
    keys = [f"{stem}/{int(x)}_{int(y)}" for x, y in coords]

    # --- GLOBAL (CLS) per-slide artifact (fp32) — also under the legacy name ---
    gbuf = io.BytesIO()
    np.savez_compressed(gbuf, vectors=G, coords=coords, keys=np.array(keys),
                        slide=stem, files=np.array([r["file"] for r in kept]))
    gdata = gbuf.getvalue()                            # bytes now; upload_fileobj closes its arg
    for key in (f"{prefix}/global.npz", f"{prefix}/hoptimus.npz"):
        s3.upload_fileobj(io.BytesIO(gdata), cfg["bucket"], key)

    # --- PATCH per-slide shard: raw memmap .npy + row-aligned numeric meta ---
    gw = int(round(P ** 0.5)) if P else 0              # patch grid width (16 for H-optimus-0)
    tile_index = np.repeat(np.arange(N, dtype=np.int64), P)
    patch_no = np.tile(np.arange(P, dtype=np.int64), N)
    mbuf = io.BytesIO()
    np.savez_compressed(mbuf, tile_index=tile_index, patch_no=patch_no,
                        patch_row=(patch_no // gw if gw else patch_no),
                        patch_col=(patch_no % gw if gw else patch_no),
                        tile_x=np.repeat(coords[:, 0], P), tile_y=np.repeat(coords[:, 1], P),
                        slide=stem, grid_w=gw)
    s3.upload_file(patch_path, cfg["bucket"], f"{prefix}/patch_vectors.npy")
    mbuf.seek(0)
    s3.upload_fileobj(mbuf, cfg["bucket"], f"{prefix}/patch_meta.npz")
    os.remove(patch_path)                              # free EBS after upload

    # --- optional: push the GLOBAL list to the ANN index (existing behaviour) ---
    if cfg["arn"] and cfg["index"] and N:
        try:
            n = _push_vectors(G, coords, stem, cfg["arn"], cfg["index"], cfg["region"])
            print(f"[embed] pushed {n} global vectors -> index={cfg['index']}", flush=True)
        except Exception as e:
            print(f"[embed] WARN vector push failed ({type(e).__name__}: {e})", flush=True)

    return {"slide": stem, "n_tiles": int(N), "P": int(P), "dim": int(G.shape[1]),
            "patch_dtype": cfg["patch_dtype"], "global_vectors": G, "coords": coords,
            "keys": keys, "patch_rows": int(N * P),
            "patch_shard": f"s3://{cfg['bucket']}/{prefix}/patch_vectors.npy",
            "patch_meta": f"s3://{cfg['bucket']}/{prefix}/patch_meta.npz"}


def _write_combined_lists(s3, bucket, results):
    """Fold per-slide records into the two ordered lists, in slide order.
      LIST 1 (global): one self-contained npz (CLS is small — fits in memory).
      LIST 2 (patch):  a manifest of per-slide shards + offsets (too big for one object).
    """
    if not results:
        return {}
    G = np.concatenate([r["global_vectors"] for r in results], 0)
    slide = np.concatenate([np.full(r["n_tiles"], r["slide"]) for r in results])
    coords = np.concatenate([r["coords"] for r in results], 0)
    keys = np.concatenate([np.array(r["keys"]) for r in results])
    gbuf = io.BytesIO()
    np.savez_compressed(gbuf, vectors=G, slide=slide, coords=coords, keys=keys,
                        order=np.arange(len(G), dtype=np.int64), kind="global",
                        dim=int(G.shape[1]))
    gbuf.seek(0)
    s3.upload_fileobj(gbuf, bucket, "embeddings/lists/global.npz")

    shards, off = [], 0
    for r in results:
        shards.append({"slide": r["slide"], "vectors": r["patch_shard"], "meta": r["patch_meta"],
                       "rows": r["patch_rows"], "P": r["P"], "dtype": r["patch_dtype"],
                       "offset": off})
        off += r["patch_rows"]
    manifest = {"kind": "patch", "dim": int(results[0]["dim"]), "total_rows": int(off),
                "order": "identity: tile-major then patch-row-major; rerank via "
                         "biolayer.vectors.ordered_list.OrderedVectorList",
                "shards": shards}
    s3.put_object(Bucket=bucket, Key="embeddings/lists/patch.manifest.json",
                  Body=json.dumps(manifest, indent=2).encode())
    print(f"[embed] LIST 1 global: {len(G)} vectors -> s3://{bucket}/embeddings/lists/global.npz",
          flush=True)
    print(f"[embed] LIST 2 patch:  {off} vectors ({len(shards)} shards) -> "
          f"s3://{bucket}/embeddings/lists/patch.manifest.json", flush=True)
    return {"global_rows": int(len(G)), "patch_rows": int(off)}


def main():
    os.environ.setdefault("AWS_DEFAULT_REGION", "us-west-2")    # container has no default
    region = os.environ["AWS_DEFAULT_REGION"]
    cfg = {
        "bucket": os.environ.get("SM_BUCKET", "bucketbiolayer"),
        "filters": [f for f in os.environ.get("FILTERS", "whitespace,tissue").split(",") if f],
        "mpp": float(os.environ.get("MPP", "0.5")),
        "tile_px": int(os.environ.get("TILE_PX", "224")),
        "max_tiles": int(os.environ["MAX_TILES"]) if os.environ.get("MAX_TILES") else None,
        "patch_dtype": os.environ.get("PATCH_DTYPE", "float16"),
        "arn": os.environ.get("VECTOR_BUCKET_ARN"),
        "index": os.environ.get("VECTOR_INDEX"),
        "region": region,
    }
    s3 = boto3.client("s3", region_name=region)
    slides = _slides_from_env(s3)
    if not slides:
        raise SystemExit("no slides: set SLIDE_S3 / SLIDES_S3 / MANIFEST_S3")

    # Load H-optimus-0 ONCE for the whole batch (cached in S3 across jobs). Default: restore
    # the cache and load OFFLINE — no HF token needed (the launcher only ships it on reseed).
    _hf_home()                                                  # always set HF_HOME (DL + seed target)
    force = os.environ.get("FORCE_RESEED") == "1"
    restored = False if force else _restore_model_cache(s3, cfg["bucket"])
    tok = _resolve_hf_token()
    if not restored:                                            # need a fresh gated download
        if not tok:
            raise SystemExit("no S3 model cache to restore and no HF_TOKEN — relaunch the "
                             "launcher with --reseed-cache (HF auth) to (re)seed the cache.")
        os.environ["HF_TOKEN"] = tok
        os.environ["HUGGING_FACE_HUB_TOKEN"] = tok
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = hoptimus.load_hoptimus(pretrained=True, device=device)
    if not restored:
        try:
            _seed_model_cache(s3, cfg["bucket"])
        except Exception as e:
            print(f"[embed] WARN seed cache failed: {e}", flush=True)
    tf = _build_transform(model)

    print(f"[embed] processing {len(slides)} slide(s); patches={cfg['patch_dtype']}, "
          f"cap={cfg['max_tiles'] or 'ALL sensible'}", flush=True)
    results = [process_slide(sl, s3, model, tf, device, cfg) for sl in slides]
    lists = _write_combined_lists(s3, cfg["bucket"], results)

    mdir = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
    os.makedirs(mdir, exist_ok=True)
    summary = [{k: r[k] for k in ("slide", "n_tiles", "P", "dim", "patch_rows",
                                  "patch_shard", "patch_meta")} for r in results]
    with open(os.path.join(mdir, "summary.json"), "w") as f:
        json.dump({"slides": summary, "lists": lists, "mpp": cfg["mpp"],
                   "filters": cfg["filters"], "patch_dtype": cfg["patch_dtype"]}, f, indent=2)


if __name__ == "__main__":
    main()
