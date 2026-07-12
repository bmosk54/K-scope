"""Spatial heatmaps: WHERE inside a tile does an SAE feature fire?

This is what patch tokens buy that CLS tokens cannot. A CLS-token SAE can only say "these 24
tiles activate feature f". A patch-token SAE says "feature f fires on THESE 16x16 regions
INSIDE this tile" -- which is what turns a feature into something a pathologist can point at.

We do NOT need the stored patch subsample for this. For any tile, we run that one tile
through H-Optimus-0 at inference and encode all 256 of its patch tokens. Full spatial
resolution, no storage cost. The stored 16-patches-per-tile sample was only ever needed to
TRAIN the SAE.
"""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

_MEAN = (0.707223, 0.578729, 0.703617)
_STD = (0.211883, 0.230117, 0.177517)
_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=_MEAN, std=_STD)])
GRID = 16
LAYER = 39


@torch.no_grad()
def patch_codes(model, sae, ck, images: list[Image.Image], device: str = "cuda") -> torch.Tensor:
    """All 256 patch tokens per tile -> SAE codes. Returns (B, 256, n_features)."""
    x = torch.stack([_tf(im) for im in images]).to(device)
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device == "cuda"):
        patch, _ = model.get_intermediate_layers(
            x, n=[LAYER], return_prefix_tokens=True, norm=True
        )[0]
    p = patch.float()  # (B, 256, 1536)
    B, N, D = p.shape
    m = sae.m if hasattr(sae, "m") else sae
    flat = (p.reshape(-1, D) - ck["mu"].to(device)) / ck["scale"].to(device)
    out = m(flat)
    z = out[1] if isinstance(out, tuple) else out
    return z.reshape(B, N, -1)


def heatmap(codes_bn: torch.Tensor, feature: int) -> np.ndarray:
    """(256,) activations -> 16x16 spatial map, normalised to [0,1]."""
    a = codes_bn[:, feature].detach().cpu().numpy().reshape(GRID, GRID)
    if a.max() > a.min():
        a = (a - a.min()) / (a.max() - a.min())
    return a


def overlay(tile: Image.Image, hm: np.ndarray, alpha: float = 0.55) -> Image.Image:
    """Red-hot overlay of a feature's spatial activation on the original tile."""
    h = Image.fromarray((hm * 255).astype(np.uint8), mode="L").resize(tile.size, Image.BICUBIC)
    h = np.asarray(h, dtype=np.float32) / 255.0
    base = np.asarray(tile.convert("RGB"), dtype=np.float32)
    hot = np.stack([np.full_like(h, 255.0), 40.0 * (1 - h), 40.0 * (1 - h)], -1)
    blend = base * (1 - alpha * h[..., None]) + hot * (alpha * h[..., None])
    return Image.fromarray(blend.clip(0, 255).astype(np.uint8))
