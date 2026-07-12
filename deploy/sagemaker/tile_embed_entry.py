"""SageMaker GPU entry: WSI (S3) -> tiles -> H-optimus-0 CLS -> S3 features + h0-vector.

Runs on ml.g5.2xlarge. Reuses the bundled wsi_reader/tile_wsi/hoptimus modules:
  1. download the slide (svs or tiff) from S3
  2. tile it at ~MPP µm/px with tissue mask + filters (tile_wsi)
  3. embed kept tiles with H-optimus-0 -> CLS [N, 1536]
  4. write features .npz to s3://<bucket>/embeddings/wsi/<slide>/hoptimus.npz
  5. (if a vector index is configured) push vectors to the h0-vector S3 Vectors store

Env:
  SLIDE_S3           s3://bucketbiolayer/wsi/<project>/<file>.svs|.tiff   (required)
  SM_BUCKET          feature-output bucket (default bucketbiolayer)
  FILTERS            comma list applied post-tiling (default whitespace,tissue)
  MPP / TILE_PX      target magnification / tile size (default 0.5 / 224)
  VECTOR_BUCKET_ARN  arn:aws:s3vectors:...:bucket/h0-vector   (optional)
  VECTOR_INDEX       index name inside that bucket             (optional)
  HF_TOKEN           gated H-optimus-0 download
"""
import io
import json
import os
import tempfile

import boto3
import numpy as np
import torch

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
def _embed(model, tf, tile_dir, device, batch=32):
    from PIL import Image

    rows = [json.loads(ln) for ln in open(os.path.join(tile_dir, "manifest.jsonl")) if ln.strip()]
    kept = [r for r in rows if r.get("file")]
    feats = []
    for i in range(0, len(kept), batch):
        chunk = kept[i:i + batch]
        x = torch.stack([tf(Image.open(os.path.join(tile_dir, r["file"])).convert("RGB"))
                         for r in chunk]).to(device)
        outs = model.get_intermediate_layers(x, n=1, return_prefix_tokens=True, norm=True)
        feats.append(outs[-1][1][:, 0].float().cpu().numpy())   # CLS -> (b, 1536)
    F = np.concatenate(feats, 0) if feats else np.zeros((0, model.embed_dim), "float32")
    coords = np.array([(r["x"], r["y"]) for r in kept], dtype=np.int64)
    return F, coords, kept


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
    """One slide: download → tile → embed → features to S3 → vectors. Model is shared."""
    b, k = slide_s3[5:].split("/", 1)
    local = os.path.join(tempfile.gettempdir(), os.path.basename(k))
    print(f"[embed] downloading {slide_s3}", flush=True)
    s3.download_file(b, k, local)
    stem = os.path.splitext(os.path.basename(k))[0]

    tile_dir = os.path.join(tempfile.gettempdir(), "tiles", stem)
    tile_wsi.tile_slide(local, tile_dir, tile_px=cfg["tile_px"], target_mpp=cfg["mpp"],
                        filters=cfg["filters"], max_tiles=cfg["max_tiles"])
    F, coords, kept = _embed(model, tf, tile_dir, device)
    print(f"[embed] {stem}: embedded {F.shape[0]} tiles -> {F.shape}", flush=True)

    fkey = f"embeddings/wsi/{stem}/hoptimus.npz"
    buf = io.BytesIO()
    np.savez_compressed(buf, feats=F, coords=coords, files=np.array([r["file"] for r in kept]))
    buf.seek(0)
    s3.upload_fileobj(buf, cfg["bucket"], fkey)
    print(f"[embed] features -> s3://{cfg['bucket']}/{fkey}", flush=True)

    if cfg["arn"] and cfg["index"] and F.shape[0]:
        try:
            n = _push_vectors(F, coords, stem, cfg["arn"], cfg["index"], cfg["region"])
            print(f"[embed] pushed {n} vectors -> index={cfg['index']}", flush=True)
        except Exception as e:
            print(f"[embed] WARN vector push failed ({type(e).__name__}: {e}); "
                  "features are safe in S3.", flush=True)
    return stem, int(F.shape[0])


def main():
    os.environ.setdefault("AWS_DEFAULT_REGION", "us-west-2")    # container has no default
    region = os.environ["AWS_DEFAULT_REGION"]
    cfg = {
        "bucket": os.environ.get("SM_BUCKET", "bucketbiolayer"),
        "filters": [f for f in os.environ.get("FILTERS", "whitespace,tissue").split(",") if f],
        "mpp": float(os.environ.get("MPP", "0.5")),
        "tile_px": int(os.environ.get("TILE_PX", "224")),
        "max_tiles": int(os.environ["MAX_TILES"]) if os.environ.get("MAX_TILES") else None,
        "arn": os.environ.get("VECTOR_BUCKET_ARN"),
        "index": os.environ.get("VECTOR_INDEX"),
        "region": region,
    }
    s3 = boto3.client("s3", region_name=region)
    slides = _slides_from_env(s3)
    if not slides:
        raise SystemExit("no slides: set SLIDE_S3 / SLIDES_S3 / MANIFEST_S3")

    # Load H-optimus-0 ONCE for the whole batch (cached in S3 across jobs).
    restored = _restore_model_cache(s3, cfg["bucket"])
    tok = _resolve_hf_token()
    if tok and not restored:                                    # auth only needed for fresh DL
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

    print(f"[embed] processing {len(slides)} slide(s)", flush=True)
    summary = [dict(zip(("slide", "n_tiles"), process_slide(sl, s3, model, tf, device, cfg)))
               for sl in slides]

    mdir = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
    os.makedirs(mdir, exist_ok=True)
    with open(os.path.join(mdir, "summary.json"), "w") as f:
        json.dump({"slides": summary, "dim": 1536, "mpp": cfg["mpp"],
                   "filters": cfg["filters"]}, f, indent=2)


if __name__ == "__main__":
    main()
