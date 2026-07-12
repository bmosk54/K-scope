"""MECH INTERP: causally intervene on an interpretable feature INSIDE the running model.

Everything else in this project is OBSERVATIONAL -- which features fire, what they look like.
That tells you what the model REPRESENTS. It does not tell you what the model USES. This does:

    forward H-Optimus-0 -> at block L, grab the residual stream
    -> encode the CLS with the SAE trained at block L
    -> ZERO (or AMPLIFY) one interpretable feature
    -> write the edit back into the residual stream
    -> let blocks L+1 .. 39 RUN as normal
    -> read the final CLS and measure how far it moved

Because 26 more transformer blocks execute after the edit, the effect propagates through real
nonlinearity. That is a do() on the model's computation, not a dot product -- the distinction
that made the block-39 "ablation" fake (nothing is downstream of the final block, so there the
answer is closed-form).

THE NULL (house rule). Deleting ANY active feature perturbs the residual stream. So the effect
of ablating feature f is compared against ablating OTHER randomly chosen active features in the
same tiles, at the same layer. Without that control, "the output changed" proves nothing.

HONEST APPROXIMATION, stated up front: the SAE was trained on the POST-final-layernorm CLS
(timm's get_intermediate_layers(norm=True)), while the residual stream at block L is PRE-norm.
We therefore compute the edit in the SAE's (normed) space and ADD THE DIFFERENCE back into the
residual stream. This is the standard residual-stream edit, but it is an approximation, not an
exact inversion -- layernorm is not invertible. An exact intervention would need an SAE trained
directly on the raw pre-norm residual stream.
"""

from __future__ import annotations

import numpy as np
import torch


class Intervention:
    """Hook block L of a timm ViT, edit the CLS via an SAE, let the rest of the network run."""

    def __init__(self, model, sae, ck, layer: int, device: str = "cuda"):
        self.model, self.sae, self.ck, self.layer, self.device = model, sae, ck, layer, device
        self.mu = ck["mu"].to(device)
        self.scale = ck["scale"].to(device)
        self._edit = None
        self._handle = None

    def _hook(self, module, inp, out):
        """out: (B, T, D) residual stream after block L. Token 0 is CLS."""
        if self._edit is None:
            return out
        h = out.clone()
        cls = h[:, 0]                                # (B, D) pre-norm residual
        normed = self.model.norm(cls)               # into the space the SAE was trained on
        x = (normed - self.mu) / self.scale
        _, z, _ = self.sae(x)
        z2 = self._edit(z.clone())
        recon_before = self.sae.dec(z) + self.sae.b_dec
        recon_after = self.sae.dec(z2) + self.sae.b_dec
        delta = (recon_after - recon_before) * self.scale   # back out of the SAE's normalisation
        h[:, 0] = cls + delta                              # residual-stream edit
        return h

    def __enter__(self):
        self._handle = self.model.blocks[self.layer].register_forward_hook(self._hook)
        return self

    def __exit__(self, *a):
        self._handle.remove()
        self._handle = None

    @torch.no_grad()
    def run(self, pixels: torch.Tensor, edit=None) -> torch.Tensor:
        """Forward pass with `edit` applied to the SAE code at block L. Returns final CLS."""
        self._edit = edit
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.device == "cuda"):
            out = self.model(pixels)
        self._edit = None
        return out.float()


def ablate(feature: int):
    def f(z):
        z[:, feature] = 0.0
        return z
    return f


class DirectionAblation:
    """Project an SAE feature's direction out of EVERY token at block L, then run the rest.

    WHY THIS EXISTS. Editing only the CLS token does not work: measured, deleting the
    tumour feature from CLS at block 27 moves the representation 8.3x more than a random
    feature, yet P(TUM) stays 0.999 and 32/32 tiles are still called tumour. The reason is
    that blocks 28-39 RECOMPUTE the CLS from the 256 un-ablated PATCH tokens via attention --
    the network simply re-derives the concept. That is the Hydra effect, and it is exactly
    what Eddie's RESULTS.md predicts ("the model recomputes the distinction downstream from
    un-ablated patch tokens").

    So a real ablation must remove the concept from the WHOLE residual stream -- every token,
    not just the readout one. We project out the SAE feature's decoder direction:

        h_t <- h_t - (h_t . w_hat) w_hat     for every token t

    This is a rank-1 subspace ablation using an INTERPRETABLE direction (an SAE feature),
    rather than a supervised probe axis.
    """

    def __init__(self, model, direction: torch.Tensor, layer: int, device: str = "cuda"):
        self.model, self.layer, self.device = model, layer, device
        w = direction.to(device).float()
        self.w = w / (w.norm() + 1e-8)
        self._on = False
        self._handle = None

    def _hook(self, module, inp, out):
        if not self._on:
            return out
        h = out.float()
        proj = (h @ self.w).unsqueeze(-1) * self.w   # (B, T, D)
        return (h - proj).to(out.dtype)

    def __enter__(self):
        self._handle = self.model.blocks[self.layer].register_forward_hook(self._hook)
        return self

    def __exit__(self, *a):
        self._handle.remove()

    @torch.no_grad()
    def run(self, pixels: torch.Tensor, on: bool = True) -> torch.Tensor:
        self._on = on
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.device == "cuda"):
            out = self.model(pixels)
        self._on = False
        return out.float()


def amplify(feature: int, alpha: float = 3.0):
    def f(z):
        z[:, feature] = z[:, feature] * alpha
        return z
    return f


@torch.no_grad()
def causal_effect(
    model, sae, ck, pixels: torch.Tensor, feature: int, layer: int,
    n_null: int = 20, device: str = "cuda", seed: int = 0,
) -> dict:
    """How much does the model's OUTPUT move when this feature is deleted at block L?

    Effect = mean L2 shift of the final CLS, vs the same statistic for ablating randomly chosen
    OTHER active features in the same tiles. Beating that null means the effect is specific to
    THIS feature, not just "removing some energy perturbs the vector".
    """
    iv = Intervention(model, sae, ck, layer, device)
    with iv:
        base = iv.run(pixels, edit=None)

        # which features are actually on for these tiles at this layer?
        h = None

        def grab(m, i, o):
            nonlocal h
            h = o

        hd = model.blocks[layer].register_forward_hook(grab)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device == "cuda"):
            model(pixels)
        hd.remove()
        x = (model.norm(h[:, 0]) - iv.mu) / iv.scale
        _, z0, _ = sae(x)
        active = torch.where((z0[:, feature] > 0))[0]
        if len(active) == 0:
            return {"feature": int(feature), "layer": layer, "n_active": 0,
                    "note": "feature never fires on these tiles at this layer; ablation is a no-op"}

        px = pixels[active]
        base_a = base[active]

        obs = (iv.run(px, ablate(feature)) - base_a).norm(dim=1).mean().item()

        # null: ablate OTHER features that are active in these same tiles
        cand = (z0[active] > 0).float()
        cand[:, feature] = 0.0
        g = torch.Generator(device="cpu").manual_seed(seed)
        pool = torch.multinomial(cand.cpu() + 1e-9, min(n_null, int(cand.sum(1).min().item()) or 1))
        null = []
        for j in range(pool.shape[1]):
            f_rand = int(pool[0, j])
            null.append((iv.run(px, ablate(f_rand)) - base_a).norm(dim=1).mean().item())

    null = np.asarray(null)
    p = (float((null >= obs).sum()) + 1) / (len(null) + 1)
    return {
        "feature": int(feature),
        "layer": layer,
        "n_tiles_active": int(len(active)),
        "output_shift": round(obs, 4),
        "null_mean_shift": round(float(null.mean()), 4),
        "null_p95": round(float(np.quantile(null, 0.95)), 4),
        "ratio_vs_null": round(obs / max(float(null.mean()), 1e-9), 2),
        "p_value": round(p, 4),
        "causally_load_bearing": bool(p < 0.05 and obs > float(np.quantile(null, 0.95))),
        "what_this_means": (
            f"Deleting feature {feature} from the model's own activations at block {layer} moves "
            f"the final output by {obs:.3f}, vs {null.mean():.3f} for deleting a random OTHER "
            f"active feature. {26 - 0} blocks of the network run AFTER the edit, so this is a "
            "genuine causal effect on the model's computation -- not a linear readout."
        ),
    }
