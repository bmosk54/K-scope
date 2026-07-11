"""Frozen encoder loading. load_encoder(model_key) -> embed(images) -> (N, dim).

Both backends share the same contract:
    embed(images: list[PIL.Image]) -> np.ndarray of shape (len(images), dim)
so extract.py batches identically regardless of which FM is loaded. Everything
runs under torch.inference_mode() on the auto-detected device.
"""
import os

import numpy as np
import torch

from . import config

# This instance has an NVIDIA A10G GPU -> DEVICE=cuda (H-optimus-0 ViT-g/14 needs
# it in practice). Falls back to CPU (all vCPUs) if no GPU is present.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type == "cpu":
    torch.set_num_threads(os.cpu_count() or 1)


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
    """H0-mini (gated): timm ViT. Recommended feature = CLS token (768-d).

    Requires `hf auth login` + APPROVED terms at hf.co/bioptimus/H0-mini before
    weights download (approval is queued — HF signup email must match the
    institutional/Dartmouth email). H0-mini needs non-default construction args
    (SwiGLU MLP + SiLU act) and uses its OWN pretrained_cfg normalization.

    pool: "cls" -> 768 (default, matches CytoSyn conditioning space)
          "cls_meanpatch" -> concat(CLS, mean patch) = 1536
    """
    import timm

    # Resolve string kwargs from config into real timm/torch objects.
    kw = dict(spec.get("timm_kwargs", {}))
    if kw.get("mlp_layer") == "SwiGLUPacked":
        kw["mlp_layer"] = timm.layers.SwiGLUPacked
    if kw.get("act_layer") == "SiLU":
        kw["act_layer"] = torch.nn.SiLU

    model = timm.create_model(
        f"hf-hub:{spec['hf_id']}", pretrained=True, **kw
    ).to(DEVICE).eval()

    # Use the model's own normalization, per the model card.
    from timm.data import create_transform, resolve_data_config
    transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
    num_prefix = getattr(model, "num_prefix_tokens", 5)  # CLS + 4 registers
    pool = spec.get("pool", "cls")

    @torch.inference_mode()
    def embed(images):
        x = torch.stack([transform(im.convert("RGB")) for im in images]).to(DEVICE)
        out = model(x)  # H0-mini returns tokens (B, 1+reg+patches, 768)
        if out.ndim == 3:
            cls = out[:, 0]  # (B, 768) — recommended probing feature
            if pool == "cls_meanpatch":
                patch_mean = out[:, num_prefix:].mean(dim=1)
                emb = torch.cat([cls, patch_mean], dim=-1)  # (B, 1536)
            else:
                emb = cls
        else:  # already pooled -> (B, D)
            emb = out
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
