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


def _push_vectors(F, coords, stem, bucket_arn, index):
    c = boto3.client("s3vectors")
    index_arn = f"{bucket_arn}/index/{index}"                   # put_vectors wants indexArn
    vecs = [{"key": f"{stem}/{int(x)}_{int(y)}",
             "data": {"float32": F[i].astype("float32").tolist()},
             "metadata": {"slide": stem, "x": int(x), "y": int(y)}}
            for i, (x, y) in enumerate(coords)]
    for i in range(0, len(vecs), 500):                          # API batches ~500
        c.put_vectors(indexArn=index_arn, vectors=vecs[i:i + 500])
    return len(vecs)


def main():
    slide_s3 = os.environ["SLIDE_S3"]
    bucket = os.environ.get("SM_BUCKET", "bucketbiolayer")
    filters = [f for f in os.environ.get("FILTERS", "whitespace,tissue").split(",") if f]
    mpp = float(os.environ.get("MPP", "0.5"))
    tile_px = int(os.environ.get("TILE_PX", "224"))
    s3 = boto3.client("s3")

    # 1. download slide
    b, k = slide_s3[5:].split("/", 1)
    local = os.path.join(tempfile.gettempdir(), os.path.basename(k))
    print(f"[embed] downloading {slide_s3}", flush=True)
    s3.download_file(b, k, local)
    stem = os.path.splitext(os.path.basename(k))[0]

    # 2. tile
    tile_dir = os.path.join(tempfile.gettempdir(), "tiles", stem)
    tile_wsi.tile_slide(local, tile_dir, tile_px=tile_px, target_mpp=mpp, filters=filters)

    # 3. embed
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = hoptimus.load_hoptimus(pretrained=True, device=device)
    F, coords, kept = _embed(model, _build_transform(model), tile_dir, device)
    print(f"[embed] {stem}: embedded {F.shape[0]} tiles -> {F.shape}", flush=True)

    # 4. features -> S3
    fkey = f"embeddings/wsi/{stem}/hoptimus.npz"
    buf = io.BytesIO()
    np.savez_compressed(buf, feats=F, coords=coords,
                        files=np.array([r["file"] for r in kept]))
    buf.seek(0)
    s3.upload_fileobj(buf, bucket, fkey)
    print(f"[embed] features -> s3://{bucket}/{fkey}", flush=True)

    # 5. vectors -> h0-vector (optional; needs an existing index)
    arn, index = os.environ.get("VECTOR_BUCKET_ARN"), os.environ.get("VECTOR_INDEX")
    if arn and index and F.shape[0]:
        try:
            n = _push_vectors(F, coords, stem, arn, index)
            print(f"[embed] pushed {n} vectors -> {arn} index={index}", flush=True)
        except Exception as e:
            print(f"[embed] WARN vector push failed ({type(e).__name__}: {e}); "
                  "features are safe in S3. Is the index created with dim 1536?", flush=True)
    else:
        print("[embed] no VECTOR_INDEX/ARN set — skipped vector push (features in S3).",
              flush=True)

    # summary artifact
    os.makedirs(os.environ.get("SM_MODEL_DIR", "/opt/ml/model"), exist_ok=True)
    with open(os.path.join(os.environ.get("SM_MODEL_DIR", "/opt/ml/model"), "summary.json"), "w") as f:
        json.dump({"slide": stem, "n_tiles": int(F.shape[0]), "dim": int(F.shape[1] if F.ndim == 2 else 0),
                   "features_key": fkey, "filters": filters, "mpp": mpp}, f, indent=2)


if __name__ == "__main__":
    main()
