# H-optimus-0 inference server

A networked GPU inference server for [Bioptimus's H-optimus-0](https://huggingface.co/bioptimus/H-optimus-0)
histology foundation model, for tile-level feature extraction. Ships both an
HTTP API (`server.py`) and an MCP tool server (`mcp_server.py`) so an agent
can call the same extraction over stdio.

## Model facts

- **1536-dim** CLS embedding per tile.
- Input: **224x224 RGB** tiles at ~0.5 microns/pixel. The server does **not**
  resample MPP — resize/crop to 224x224 at the right zoom level before
  sending; the server only validates the dimensions.
- Loaded via `timm`, not `transformers`:
  `timm.create_model("hf-hub:bioptimus/H-optimus-0", pretrained=True, init_values=1e-5, dynamic_img_size=False)`
- The HF repo is **gated** — you need a Hugging Face token with access
  (request access on the [model page](https://huggingface.co/bioptimus/H-optimus-0)),
  passed via the `HF_TOKEN` env var. Never hardcode it.
- Inference runs under `torch.autocast(device_type="cuda", dtype=torch.float16)`
  and `torch.inference_mode()`.
- Needs a CUDA GPU with ~24GB VRAM.

## Repo layout

- `model.py` — model loading, preprocessing, and inference. Shared by both servers.
- `server.py` — FastAPI HTTP server.
- `mcp_server.py` — MCP tool server (stdio transport).
- `requirements.txt` — pinned server dependencies.
- `requirements-test.txt` — extra deps for `client_test.py` only.
- `setup.sh` — EC2 bootstrap script.
- `client_test.py` — smoke test against a running server.

## Launching on EC2

**Recommended AMI:** an AWS **Deep Learning AMI** (Ubuntu, GPU-enabled) so
NVIDIA drivers and CUDA are already installed — `setup.sh` does not install
GPU drivers itself.

**Recommended instance type:** `g6.xlarge` (default assumed below). `g5.xlarge`
also works.

**Security group:** lock inbound TCP `8000` to your own IP only
(`<your-ip>/32`), not `0.0.0.0/0`. This server has no auth in front of it.

You provision the EC2 instance yourself (this repo does not call AWS APIs or
touch credentials). Steps:

1. Launch a `g6.xlarge` instance from a Deep Learning AMI, in a security group
   that only allows inbound `8000` from your IP.
2. Copy this directory to the instance (`scp`/`git clone`).
3. SSH in, then:
   ```bash
   export HF_TOKEN=hf_...           # your Hugging Face token with H-optimus-0 access
   cd h-optimus-server
   chmod +x setup.sh
   ./setup.sh
   ```
   `setup.sh` creates a venv, installs `requirements.txt`, and launches
   `uvicorn server:app` on `0.0.0.0:8000` (override with `PORT`).
4. From your machine, test it:
   ```bash
   pip install -r requirements-test.txt
   python client_test.py http://<instance-public-ip>:8000
   ```

To run it as a persistent service (survives SSH disconnect), run `setup.sh`
under `tmux`/`screen`, or wrap it in a systemd unit that runs the same
`uvicorn` command with the same env vars.

## Env vars

| Var | Required | Default | Purpose |
|---|---|---|---|
| `HF_TOKEN` | yes | — | Hugging Face token with access to the gated H-optimus-0 repo |
| `PORT` | no | `8000` | HTTP server port |
| `MAX_BATCH_SIZE` | no | `64` | Max tiles per `/embed_batch` request |

## HTTP API

- `GET /health` -> `{"status": "ok", "device": "cuda", "model_loaded": true}`
- `POST /embed` — multipart file upload **or** JSON `{"image_base64": "..."}` ->
  `{"features": [...1536 floats...]}`
- `POST /embed_batch` — JSON `{"images_base64": ["...", "..."]}` ->
  `{"features": [[...], [...], ...]}`

## Running locally (no GPU)

The server falls back to CPU if CUDA isn't available (autocast is skipped).
This works for smoke-testing the API shape but will be slow — real usage
needs a GPU.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export HF_TOKEN=hf_...
uvicorn server:app --reload
```

## MCP server

Reuses the same `model.py` (load-once, identical preprocessing) as `server.py`.

```bash
export HF_TOKEN=hf_...
python mcp_server.py
```

Exposes two tools over stdio:

- `extract_features(tile_base64: str) -> list[float]`
- `extract_features_batch(tiles_base64: list[str]) -> list[list[float]]`

Point your MCP client config at `python mcp_server.py` (with `HF_TOKEN` set
in its environment) to use it as a tool.
