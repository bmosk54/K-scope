"""Thin client for the warm H-optimus-0 endpoint — used by the MCP `embed` verb and CLI.

    from deploy.sagemaker.endpoint_client import embed
    r = embed(images=["<b64 png>"])                       # -> {"dim":1536,"n":1,"embeddings":[[...]]}
    r = embed(s3_tiles=["s3://bucketbiolayer/.../tile.png"])
    r = embed(slide_s3="s3://.../slide.svs", max_tiles=16)
    r = embed(images=[...], push={"index":"layerbioindex","slide":"query"})

CLI:
    python -m deploy.sagemaker.endpoint_client --image tile.png
    python -m deploy.sagemaker.endpoint_client --s3-tile s3://.../tile.png --push-index layerbioindex
"""
import base64
import json
import os

ENDPOINT = os.environ.get("HOPTIMUS_ENDPOINT", "hoptimus-embed")
REGION = os.environ.get("AWS_DEFAULT_REGION", "us-west-2")


def embed(images=None, s3_tiles=None, slide_s3=None, keys=None, push=None,
          endpoint=None, region=None, **slide_kw):
    """Invoke the endpoint. Provide exactly one tile source. Returns the parsed dict,
    or {"status":"unavailable", ...} if the endpoint isn't deployed / not InService."""
    import boto3

    payload = {}
    if images is not None:
        payload["images"] = [b if isinstance(b, str) else base64.b64encode(b).decode() for b in images]
        if keys:
            payload["keys"] = keys
    elif s3_tiles is not None:
        payload["s3_tiles"] = s3_tiles
    elif slide_s3 is not None:
        payload["slide_s3"] = slide_s3
        payload.update({k: v for k, v in slide_kw.items() if v is not None})
    else:
        raise ValueError("give one of: images | s3_tiles | slide_s3")
    if push:
        payload["push"] = push

    rt = boto3.client("sagemaker-runtime", region_name=region or REGION)
    try:
        resp = rt.invoke_endpoint(EndpointName=endpoint or ENDPOINT,
                                  ContentType="application/json",
                                  Body=json.dumps(payload).encode())
        return json.loads(resp["Body"].read())
    except rt.exceptions.ClientError as e:
        code = e.response.get("Error", {}).get("Code", "")
        return {"status": "unavailable", "endpoint": endpoint or ENDPOINT, "error": code,
                "note": f"endpoint not reachable ({code}); deploy it with "
                        "`python deploy/sagemaker/deploy_endpoint.py`"}


def _load_image_b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def main():
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--image", action="append", help="local tile image (repeatable)")
    ap.add_argument("--s3-tile", action="append", help="s3:// tile key (repeatable)")
    ap.add_argument("--slide-s3", help="s3:// slide to tile+embed (bounded)")
    ap.add_argument("--max-tiles", type=int, default=16)
    ap.add_argument("--push-index", help="push results to this h0-vector index")
    ap.add_argument("--slide-name", default="query", help="slide/stem tag for pushed vectors")
    ap.add_argument("--endpoint", default=ENDPOINT)
    args = ap.parse_args()

    push = {"index": args.push_index, "slide": args.slide_name} if args.push_index else None
    if args.image:
        r = embed(images=[_load_image_b64(p) for p in args.image],
                  keys=[os.path.basename(p) for p in args.image], push=push, endpoint=args.endpoint)
    elif args.s3_tile:
        r = embed(s3_tiles=args.s3_tile, push=push, endpoint=args.endpoint)
    elif args.slide_s3:
        r = embed(slide_s3=args.slide_s3, max_tiles=args.max_tiles, push=push, endpoint=args.endpoint)
    else:
        ap.error("give --image / --s3-tile / --slide-s3")
    # Don't dump the full vectors to the terminal.
    summary = {k: v for k, v in r.items() if k != "embeddings"}
    summary["embeddings"] = f"[{r.get('n', 0)} x {r.get('dim', '?')}]" if "embeddings" in r else None
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
