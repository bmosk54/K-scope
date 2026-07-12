"""SageMaker training-job entry point — loads H-optimus-0, modifies weights, embeds.

SageMaker runs this on the g5.2xlarge. It reads hyperparameters (CLI args SageMaker
injects) + env (HF_TOKEN), loads the model via hoptimus.py, applies the example weight
edits, runs a forward pass, and writes the modified state_dict + a sample embedding to
/opt/ml/model (which SageMaker uploads to your S3 output path).

Not tied to the UI in any way — it's just a script the training job executes.
"""
import argparse
import json
import os

import torch

import hoptimus


def main():
    ap = argparse.ArgumentParser()
    # SageMaker passes hyperparameters as --key value.
    ap.add_argument("--pretrained", type=int, default=1)   # 0 = random init smoke test
    ap.add_argument("--n-tiles", type=int, default=8)      # dummy tiles to embed
    ap.add_argument("--model-dir", default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    args = ap.parse_args()

    if os.environ.get("HF_TOKEN"):
        os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", os.environ["HF_TOKEN"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[entry] device={device} torch={torch.__version__}", flush=True)

    model = hoptimus.load_hoptimus(pretrained=bool(args.pretrained), device=device)
    print(f"[entry] loaded H-optimus-0: dim={model.embed_dim} blocks={len(model.blocks)}",
          flush=True)

    # --- arbitrary weight modification ---
    edits_log = hoptimus.example_edits(model)
    for line in edits_log:
        print(f"[entry] edit: {line}", flush=True)

    # --- forward pass: embed dummy 224x224 tiles -> CLS [n, 1536] ---
    x = torch.randn(args.n_tiles, 3, 224, 224, device=device)
    with torch.inference_mode():
        outs = model.get_intermediate_layers(x, n=1, return_prefix_tokens=True, norm=True)
        cls = outs[-1][1][:, 0]          # (n, 1536) CLS per tile
    print(f"[entry] embedded {tuple(cls.shape)} tiles", flush=True)

    # --- persist modified weights + a sample embedding for S3 upload ---
    os.makedirs(args.model_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(args.model_dir, "hoptimus_edited.pt"))
    torch.save(cls.cpu(), os.path.join(args.model_dir, "sample_cls.pt"))
    with open(os.path.join(args.model_dir, "run.json"), "w") as f:
        json.dump({"edits": edits_log, "cls_shape": list(cls.shape),
                   "pretrained": bool(args.pretrained)}, f, indent=2)
    print("[entry] wrote hoptimus_edited.pt + sample_cls.pt + run.json", flush=True)


if __name__ == "__main__":
    main()
