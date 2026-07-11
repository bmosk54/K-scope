"""Frozen encoder loading. load_encoder(model_key) -> embed(images) -> (N, dim).

Both backends share the same contract:
    embed(images: list[PIL.Image]) -> np.ndarray of shape (len(images), dim)
so extract.py batches identically regardless of which FM is loaded. CPU-safe:
everything runs under torch.inference_mode() on the default device (cpu here).
"""
import numpy as np
import torch

from . import config

# c5.4xlarge is CPU-only; make CPU inference deterministic + threaded.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_transformers(spec):
    """Phikon-v2 (ungated): AutoModel, CLS = last_hidden_state[:, 0, :]."""
    from transformers import AutoImageProcessor, AutoModel

    processor = AutoImageProcessor.from_pretrained(spec["hf_id"], use_fast=True)
    model = AutoModel.from_pretrained(spec["hf_id"]).to(DEVICE).eval()

    @torch.inference_mode()
    def embed(images):
        inputs = processor(images=[im.convert("RGB") for im in images],
                           return_tensors="pt").to(DEVICE)
        out = model(**inputs)
        cls = out.last_hidden_state[:, 0, :]  # CLS token -> (B, 1024)
        return cls.float().cpu().numpy()

    return embed


def _load_timm(spec):
    """H0-mini (gated): timm ViT. Embedding = concat(CLS, mean patch) -> 1536.

    Requires `huggingface-cli login` + accepted terms at hf.co/bioptimus/H0-mini
    before the weights will download. DINOv2-style: prefix tokens (CLS + registers)
    are skipped when meaning the patch tokens.
    """
    import timm

    model = timm.create_model(
        f"hf-hub:{spec['hf_id']}", pretrained=True, num_classes=0
    ).to(DEVICE).eval()

    data_cfg = timm.data.resolve_model_data_config(model)
    transform = timm.data.create_transform(**data_cfg, is_training=False)
    num_prefix = getattr(model, "num_prefix_tokens", 1)  # CLS (+ registers)
    pool = spec.get("pool", "cls")

    @torch.inference_mode()
    def embed(images):
        x = torch.stack([transform(im.convert("RGB")) for im in images]).to(DEVICE)
        feats = model.forward_features(x)  # (B, T, D) for ViT
        if feats.ndim == 3:
            cls = feats[:, 0]
            if pool == "cls_meanpatch":
                patch_mean = feats[:, num_prefix:].mean(dim=1)
                emb = torch.cat([cls, patch_mean], dim=-1)  # (B, 2D) = 1536
            else:
                emb = cls
        else:  # already pooled -> (B, D)
            emb = feats
        return emb.float().cpu().numpy()

    return embed


_LOADERS = {"transformers": _load_transformers, "timm": _load_timm}


def load_encoder(model_key: str):
    """Return (embed_fn, spec) for a registered model.

    embed_fn(images: list[PIL.Image]) -> np.ndarray (len(images), spec['dim']).
    """
    if model_key not in config.MODELS:
        raise KeyError(f"unknown model {model_key!r}; known: {list(config.MODELS)}")
    spec = config.MODELS[model_key]
    embed = _LOADERS[spec["backend"]](spec)
    return embed, spec
