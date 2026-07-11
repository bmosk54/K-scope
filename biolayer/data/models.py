"""Frozen encoder loading with MULTI-LAYER, LOCAL+GLOBAL embedding.

load_encoder(model_key) -> embed(images) -> (globals, locals), spec

At each of the model's 3 configured layers we return BOTH:
    global : the CLS token           -> (B, L, dim)   tile-level readout feature
    local  : the mean patch token    -> (B, L, dim)   local morphology / texture

Both backends share this contract so extract.py is model-agnostic:
    transformers -> AutoModel(output_hidden_states=True); global=hs[l][:,0],
                    local=hs[l][:,1:].mean(1)
    timm         -> get_intermediate_layers(return_prefix_tokens=True); global=
                    prefix[:,0] (CLS), local=patch_tokens.mean(1)
Everything runs under torch.inference_mode() on the auto-detected device.
"""
import os

import numpy as np
import torch

from .. import config

# A10G GPU on this instance -> cuda (H-optimus-0 ViT-g/14 needs it). CPU fallback.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type == "cpu":
    torch.set_num_threads(os.cpu_count() or 1)


def _stack(globals_list, locals_list):
    """(list of (B,D)) x L -> ((B,L,D) global, (B,L,D) local) as float32 numpy."""
    g = torch.stack(globals_list, dim=1).float().cpu().numpy()
    l = torch.stack(locals_list, dim=1).float().cpu().numpy()
    return g, l


def _load_transformers(spec):
    """Phikon-v2 (ungated): AutoModel, hidden states at the configured layers.

    hidden_states is a tuple of length n_blocks+1 (index 0 = embeddings,
    index i = after block i, index n_blocks = last_hidden_state). Phikon-v2 has a
    single CLS prefix token (no registers), so patches = tokens[:, 1:].
    """
    from transformers import AutoImageProcessor, AutoModel

    processor = AutoImageProcessor.from_pretrained(spec["hf_id"], use_fast=True)
    model = AutoModel.from_pretrained(spec["hf_id"]).to(DEVICE).eval()
    layers = spec["layers"]

    @torch.inference_mode()
    def embed(images):
        inputs = processor(images=[im.convert("RGB") for im in images],
                           return_tensors="pt").to(DEVICE)
        out = model(**inputs, output_hidden_states=True)
        hs = out.hidden_states  # tuple, each (B, 1+P, dim)
        globs = [hs[l][:, 0] for l in layers]          # CLS  -> global
        locs = [hs[l][:, 1:].mean(dim=1) for l in layers]  # mean patch -> local
        return _stack(globs, locs)

    return embed


def _load_timm(spec):
    """H0-mini / H-optimus-0 (gated timm ViTs): get_intermediate_layers.

    Requires `hf auth login` + accepted terms. Uses the model's own normalization.
    prefix tokens = CLS (+ registers for H0-mini); CLS = prefix[:, 0].
    """
    import timm

    kw = dict(spec.get("timm_kwargs", {}))
    if kw.get("mlp_layer") == "SwiGLUPacked":
        kw["mlp_layer"] = timm.layers.SwiGLUPacked
    if kw.get("act_layer") == "SiLU":
        kw["act_layer"] = torch.nn.SiLU

    model = timm.create_model(
        f"hf-hub:{spec['hf_id']}", pretrained=True, **kw
    ).to(DEVICE).eval()

    from timm.data import create_transform, resolve_data_config
    transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
    layers = spec["layers"]

    @torch.inference_mode()
    def embed(images):
        x = torch.stack([transform(im.convert("RGB")) for im in images]).to(DEVICE)
        # one (patch_tokens (B,P,D), prefix (B,num_prefix,D)) pair per requested layer
        outs = model.get_intermediate_layers(
            x, n=layers, return_prefix_tokens=True, norm=True)
        globs = [prefix[:, 0] for _, prefix in outs]        # CLS -> global
        locs = [patch.mean(dim=1) for patch, _ in outs]     # mean patch -> local
        return _stack(globs, locs)

    return embed


_LOADERS = {"transformers": _load_transformers, "timm": _load_timm}


def load_encoder(model_key: str):
    """Return (embed_fn, spec) for a registered model.

    embed_fn(images: list[PIL.Image]) -> (globals, locals), each
    np.ndarray of shape (len(images), len(spec['layers']), spec['dim']).
    """
    if model_key not in config.MODELS:
        raise KeyError(f"unknown model {model_key!r}; known: {list(config.MODELS)}")
    spec = config.MODELS[model_key]
    embed = _LOADERS[spec["backend"]](spec)
    return embed, spec
