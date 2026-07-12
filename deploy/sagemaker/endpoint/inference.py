"""SageMaker real-time inference handler — H-optimus-0 CLS embeddings on demand.

Hosts H-optimus-0 WARM on a g5 endpoint so external triggers (an MCP `embed` call,
a K-Pro answer that needs a fresh tile embedded) get a 1536-d vector in ~one forward
pass — no 4 GB model re-download per call. The weights are restored ONCE at container
start from the S3 cache the tiling/embedding job seeded (offline load, no HF hit).

SageMaker PyTorch inference contract (model artifact carries this under code/):
  model_fn(model_dir)                  -> warm model + transform + device  (called once)
  input_fn(body, content_type)         -> parsed request dict
  predict_fn(data, ctx)                -> {"dim", "n", "embeddings", "keys", "pushed"}
  output_fn(pred, accept)              -> JSON bytes

Request JSON (application/json) — any ONE tile source, smallest first:
  {"images":   ["<base64 PNG/JPEG>", ...]}          # tile bytes inline (MCP fast path)
  {"s3_tiles": ["s3://bucket/key.png", ...]}        # tiles already in S3
  {"slide_s3": "s3://.../slide.svs", "max_tiles": 64,   # tile+embed a slide/region
   "filters": ["whitespace","tissue"], "mpp": 0.5}      # (bounded — real-time 60s cap)
Optional push to the vector store:
  {"push": {"index": "layerbioindex", "slide": "<stem>",
            "bucket_arn": "arn:aws:s3vectors:...:bucket/h0-vector"}}

Whole-slide embedding stays on the training-job path (launch_tile_embed.py); this
endpoint is for on-demand, few-tile queries that must be fast.
"""
import base64
import io
import json
import os
import tarfile
import tempfile

import boto3
import numpy as np
import torch
from PIL import Image, ImageFile, PngImagePlugin

import hoptimus

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True
PngImagePlugin.MAX_TEXT_CHUNK = 100 * 1024 * 1024

REGION = os.environ.get("AWS_DEFAULT_REGION", "us-west-2")
BUCKET = os.environ.get("SM_BUCKET", "bucketbiolayer")
MODEL_CACHE_KEY = os.environ.get("MODEL_CACHE_KEY", "models/hf-cache-h-optimus-0.tar")


# ---------------------------------------------------------------------------
# Warm load (once per container)
# ---------------------------------------------------------------------------
def _restore_model_cache(s3):
    """Restore the HF cache from S3 → offline load, no HF download (same tar the
    tiling/embed job seeds). Returns True if restored."""
    home = "/opt/ml/hf-cache"
    os.environ["HF_HOME"] = home
    os.makedirs(home, exist_ok=True)
    tarp = "/opt/ml/hf-cache.tar"
    try:
        s3.download_file(BUCKET, MODEL_CACHE_KEY, tarp)
    except Exception as e:
        print(f"[infer] no S3 model cache ({e}); will need HF auth for a cold load", flush=True)
        return False
    with tarfile.open(tarp) as t:
        t.extractall(home)
    os.environ["HF_HUB_OFFLINE"] = "1"
    print(f"[infer] restored model cache from s3://{BUCKET}/{MODEL_CACHE_KEY}", flush=True)
    return True


def model_fn(model_dir):
    from timm.data import create_transform, resolve_data_config

    s3 = boto3.client("s3", region_name=REGION)
    restored = _restore_model_cache(s3)
    if not restored:                                   # cold cache: fall back to HF auth
        tok = os.environ.get("HF_TOKEN")
        if tok:
            os.environ["HUGGING_FACE_HUB_TOKEN"] = tok
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = hoptimus.load_hoptimus(pretrained=True, device=device)
    tf = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
    print(f"[infer] H-optimus-0 warm on {device} (dim={model.embed_dim})", flush=True)
    return {"model": model, "tf": tf, "device": device, "s3": s3}


# ---------------------------------------------------------------------------
# Request parsing
# ---------------------------------------------------------------------------
def input_fn(body, content_type="application/json"):
    if content_type and "json" not in content_type:
        raise ValueError(f"unsupported content-type {content_type!r}; send application/json")
    if isinstance(body, (bytes, bytearray)):
        body = body.decode("utf-8")
    return json.loads(body)


def _tiles_from_request(req, ctx):
    """Resolve the request to a list of (key, RGB PIL.Image) tiles."""
    tiles = []
    if req.get("images"):
        for i, b64 in enumerate(req["images"]):
            img = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
            tiles.append((req.get("keys", [None] * len(req["images"]))[i] or f"img_{i}", img))
    elif req.get("s3_tiles"):
        s3 = ctx["s3"]
        for uri in req["s3_tiles"]:
            b, k = uri[5:].split("/", 1)
            data = s3.get_object(Bucket=b, Key=k)["Body"].read()
            tiles.append((os.path.basename(k), Image.open(io.BytesIO(data)).convert("RGB")))
    elif req.get("slide_s3"):
        tiles = _tile_slide(req, ctx)
    else:
        raise ValueError("request needs one of: images | s3_tiles | slide_s3")
    return tiles


def _tile_slide(req, ctx):
    """Bounded tile+read of a slide/region (real-time has a 60s cap — keep max_tiles small)."""
    import tile_wsi

    s3 = ctx["s3"]
    uri = req["slide_s3"]
    b, k = uri[5:].split("/", 1)
    local = os.path.join(tempfile.gettempdir(), os.path.basename(k))
    if not os.path.exists(local):
        s3.download_file(b, k, local)
    out_dir = os.path.join(tempfile.gettempdir(), "ep_tiles", os.path.splitext(os.path.basename(k))[0])
    tile_wsi.tile_slide(local, out_dir, tile_px=int(req.get("tile_px", 224)),
                        target_mpp=float(req.get("mpp", 0.5)),
                        filters=req.get("filters", ["whitespace", "tissue"]),
                        max_tiles=int(req.get("max_tiles", 64)))
    rows = [json.loads(ln) for ln in open(os.path.join(out_dir, "manifest.jsonl")) if ln.strip()]
    return [(r["file"], Image.open(os.path.join(out_dir, r["file"])).convert("RGB"))
            for r in rows if r.get("file")]


# ---------------------------------------------------------------------------
# Embed + optional vector push
# ---------------------------------------------------------------------------
@torch.inference_mode()
def predict_fn(req, ctx, batch=32):
    tiles = _tiles_from_request(req, ctx)
    if not tiles:
        return {"dim": ctx["model"].embed_dim, "n": 0, "embeddings": [], "keys": [], "pushed": None}
    model, tf, device = ctx["model"], ctx["tf"], ctx["device"]
    keys = [k for k, _ in tiles]
    feats = []
    for i in range(0, len(tiles), batch):
        chunk = tiles[i:i + batch]
        x = torch.stack([tf(img) for _, img in chunk]).to(device)
        outs = model.get_intermediate_layers(x, n=1, return_prefix_tokens=True, norm=True)
        feats.append(outs[-1][1][:, 0].float().cpu().numpy())        # CLS -> (b, 1536)
    F = np.concatenate(feats, 0)

    pushed = None
    if req.get("push"):
        pushed = _push_vectors(F, keys, req["push"], ctx)
    return {"dim": int(F.shape[1]), "n": int(F.shape[0]),
            "embeddings": F.astype("float32").tolist(), "keys": keys, "pushed": pushed}


def _push_vectors(F, keys, push, ctx):
    arn = push.get("bucket_arn") or os.environ.get("VECTOR_BUCKET_ARN")
    index = push.get("index")
    if not (arn and index):
        return None
    stem = push.get("slide", "endpoint")
    c = boto3.client("s3vectors", region_name=REGION)
    index_arn = f"{arn}/index/{index}"
    vecs = [{"key": f"{stem}/{keys[i]}",
             "data": {"float32": F[i].astype("float32").tolist()},
             "metadata": {"slide": stem, "tile": keys[i]}}
            for i in range(len(keys))]
    for i in range(0, len(vecs), 500):
        c.put_vectors(indexArn=index_arn, vectors=vecs[i:i + 500])
    return len(vecs)


def output_fn(pred, accept="application/json"):
    return json.dumps(pred), "application/json"
