"""Smoke test against a running H-optimus-0 server.

Usage:
    python client_test.py [base_url]

base_url defaults to http://localhost:8000
"""

import base64
import io
import sys

import requests
from PIL import Image

EXPECTED_DIM = 1536


def make_random_tile() -> bytes:
    import numpy as np

    arr = (np.random.rand(224, 224, 3) * 255).astype("uint8")
    image = Image.fromarray(arr, mode="RGB")
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def main() -> None:
    base_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"

    health = requests.get(f"{base_url}/health", timeout=30)
    health.raise_for_status()
    print("GET /health ->", health.json())
    assert health.json()["status"] == "ok"

    tile_bytes = make_random_tile()
    tile_b64 = base64.b64encode(tile_bytes).decode("ascii")

    embed_resp = requests.post(
        f"{base_url}/embed",
        json={"image_base64": tile_b64},
        timeout=60,
    )
    embed_resp.raise_for_status()
    features = embed_resp.json()["features"]
    print(f"POST /embed -> {len(features)}-dim embedding")
    assert len(features) == EXPECTED_DIM, f"expected {EXPECTED_DIM} floats, got {len(features)}"

    batch_resp = requests.post(
        f"{base_url}/embed_batch",
        json={"images_base64": [tile_b64, tile_b64]},
        timeout=60,
    )
    batch_resp.raise_for_status()
    batch_features = batch_resp.json()["features"]
    print(f"POST /embed_batch -> {len(batch_features)} embeddings of dim {len(batch_features[0])}")
    assert len(batch_features) == 2
    assert all(len(f) == EXPECTED_DIM for f in batch_features)

    print("All checks passed.")


if __name__ == "__main__":
    main()
