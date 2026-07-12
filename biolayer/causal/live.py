"""Live source-intervention — hook encoder.layer[L], edit activations, propagate.

This is the real do(): unlike the cached battery (which ablates in readout space
post-hoc), we register a forward hook on a transformer block, project a direction
out of the CLS token IN the residual stream at layer L, and let blocks L+1..final
RECOMPUTE. That is what tests the redundancy/Hydra claim ("does the model recover
the concept downstream?") — and it is what makes `intervened_on_input=true`: the
intervention is on THIS tile's forward pass, not a reference-set projection.

Phikon-v2 is a Dinov2Model (24 blocks, CLS=token0, no registers). We hook
`model.encoder.layer[block_idx]` whose output tensor becomes hidden_states[block_idx+1].
"""
import numpy as np
import torch

from .. import config
from ..data.models import DEVICE


def project_out(direction):
    """Edit fn: remove `direction` from the CLS token (index 0) of a hidden state."""
    d = torch.as_tensor(np.asarray(direction), dtype=torch.float32, device=DEVICE)
    d = d / (d.norm() + 1e-12)

    def edit(h):                      # h: (B, T, dim)
        h = h.clone()
        cls = h[:, 0, :]
        h[:, 0, :] = cls - (cls @ d).unsqueeze(1) * d
        return h
    return edit


class LiveEncoder:
    """Frozen transformers ViT with an ablation hook. phikon-family (Dinov2) only."""

    def __init__(self, model_key="phikon_v2"):
        spec = config.MODELS[model_key]
        if spec["backend"] != "transformers":
            raise NotImplementedError(
                f"live hook implemented for transformers ViT (phikon); {model_key} is "
                f"{spec['backend']} — timm live hooks are a follow-on.")
        from transformers import AutoImageProcessor, AutoModel
        self.processor = AutoImageProcessor.from_pretrained(spec["hf_id"], use_fast=True)
        self.model = AutoModel.from_pretrained(spec["hf_id"]).to(DEVICE).eval()
        self.blocks = self.model.encoder.layer
        self.dim = spec["dim"]
        self.n_blocks = len(self.blocks)

    @torch.inference_mode()
    def _forward(self, images, edit=None, block_idx=None, batch_size=32, want_all=False):
        """Readout CLS = last_hidden_state[:,0] (post-final-LN — the representation the
        readout probe is fit on). If edit+block_idx given, apply the edit at that block's
        output and let it propagate. want_all also returns per-layer CLS
        (N, n_blocks+1, dim) from hidden_states (pre-LN) for deriving per-layer axes —
        only the clean pass needs it, so ablated/null passes stay cheap."""
        use_amp = DEVICE.type == "cuda"
        finals, alls = [], []
        for i in range(0, len(images), batch_size):
            batch = [im.convert("RGB") for im in images[i:i + batch_size]]
            inputs = self.processor(images=batch, return_tensors="pt").to(DEVICE)
            handle = None
            if edit is not None and block_idx is not None:
                def hook(module, inp, out, _edit=edit):
                    h = out[0] if isinstance(out, tuple) else out
                    h = _edit(h)
                    return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
                handle = self.blocks[block_idx].register_forward_hook(hook)
            try:
                with torch.autocast(device_type=DEVICE.type, dtype=torch.float16,
                                    enabled=use_amp):
                    out = self.model(**inputs, output_hidden_states=want_all)
            finally:
                if handle:
                    handle.remove()
            finals.append(out.last_hidden_state[:, 0].float().cpu().numpy())
            if want_all:
                alls.append(torch.stack([h[:, 0] for h in out.hidden_states], dim=1)
                            .float().cpu().numpy())
        final = np.concatenate(finals, axis=0)
        return (final, np.concatenate(alls, axis=0)) if want_all else final

    def embed(self, images, edit=None, block_idx=None, batch_size=32):
        """Readout CLS (N, dim), optionally under an ablation hook at block_idx. Cheap
        (no per-layer stacking)."""
        return self._forward(images, edit, block_idx, batch_size, want_all=False)

    def hidden_cls(self, images, batch_size=32):
        """One clean pass -> (readout CLS (N, dim), per-layer CLS (N, n_blocks+1, dim))."""
        return self._forward(images, None, None, batch_size, want_all=True)


class TimmLiveEncoder:
    """Frozen timm ViT (H-optimus-0 / H0-mini) with a residual-stream ablation hook.

    Same do() contract as LiveEncoder, for the timm backend. We manually replay
    `forward_features` (patch_embed -> _pos_embed -> patch_drop -> norm_pre -> blocks
    -> norm) so we can (a) capture the CLS token after every block and (b) project a
    direction out of the CLS token in the residual stream at block L and let
    L+1..final RECOMPUTE. CLS is prefix token 0 (H-optimus carries 4 register tokens
    after it); the readout is the global_pool='token' CLS = norm(tokens)[:, 0], which
    matches `data.models` extraction (verified: manual == forward_features, diff 0.0).

    Index convention mirrors LiveEncoder so `intervene.live_necessity` is backend-
    agnostic: hidden_cls returns (N, n_blocks+1, dim) with index i = CLS after i
    blocks (index 0 = post pos-embed), and embed(edit, block_idx) edits the output of
    block `block_idx` — so clean_hidden[:, L] pairs with block_idx = L-1 as before.
    """

    def __init__(self, model_key="h_optimus_0"):
        spec = config.MODELS[model_key]
        if spec["backend"] != "timm":
            raise NotImplementedError(
                f"TimmLiveEncoder is for timm ViTs; {model_key} is {spec['backend']} "
                f"(use LiveEncoder for transformers).")
        import timm

        kw = dict(spec.get("timm_kwargs", {}))
        if kw.get("mlp_layer") == "SwiGLUPacked":
            kw["mlp_layer"] = timm.layers.SwiGLUPacked
        if kw.get("act_layer") == "SiLU":
            kw["act_layer"] = torch.nn.SiLU
        self.model = timm.create_model(
            f"hf-hub:{spec['hf_id']}", pretrained=True, **kw).to(DEVICE).eval()
        from timm.data import create_transform, resolve_data_config
        self.transform = create_transform(
            **resolve_data_config(self.model.pretrained_cfg, model=self.model))
        self.blocks = self.model.blocks
        self.n_blocks = len(self.blocks)
        self.dim = spec["dim"]

    def _prep(self, images):
        return torch.stack([self.transform(im.convert("RGB")) for im in images]).to(DEVICE)

    def _tokens_in(self, x):
        m = self.model
        t = m.patch_embed(x)
        t = m._pos_embed(t)           # prepends CLS(+registers) + pos embed
        t = m.patch_drop(t)           # Identity on H-optimus
        t = m.norm_pre(t)             # Identity on H-optimus
        return t

    @torch.inference_mode()
    def _run(self, x, edit=None, block_idx=None, capture=False):
        use_amp = DEVICE.type == "cuda"
        with torch.autocast(device_type=DEVICE.type, dtype=torch.float16, enabled=use_amp):
            t = self._tokens_in(x)
            cls = [t[:, 0]] if capture else None       # index 0 = post pos-embed
            for i, blk in enumerate(self.blocks):
                t = blk(t)
                if edit is not None and i == block_idx:  # ablate CLS in residual stream
                    t = edit(t)
                if capture:
                    cls.append(t[:, 0])                # index i+1 = after block i
            readout = self.model.norm(t)[:, 0]         # global_pool='token' readout CLS
        if capture:
            return readout, torch.stack(cls, dim=1)    # (B,dim), (B,n_blocks+1,dim)
        return readout

    def embed(self, images, edit=None, block_idx=None, batch_size=32):
        """Readout CLS (N, dim), optionally under an ablation hook at block_idx."""
        outs = []
        for i in range(0, len(images), batch_size):
            x = self._prep(images[i:i + batch_size])
            outs.append(self._run(x, edit, block_idx, capture=False).float().cpu().numpy())
        return np.concatenate(outs, axis=0)

    def hidden_cls(self, images, batch_size=32):
        """One clean pass -> (readout CLS (N, dim), per-layer CLS (N, n_blocks+1, dim))."""
        fin, alls = [], []
        for i in range(0, len(images), batch_size):
            x = self._prep(images[i:i + batch_size])
            r, a = self._run(x, capture=True)
            fin.append(r.float().cpu().numpy())
            alls.append(a.float().cpu().numpy())
        return np.concatenate(fin, axis=0), np.concatenate(alls, axis=0)


def supports_live(model_key):
    """True if a live source-intervention encoder exists for this model's backend."""
    return config.MODELS[model_key]["backend"] in ("transformers", "timm")


def make_live_encoder(model_key="phikon_v2"):
    """Backend-dispatched live encoder: Dinov2/transformers -> LiveEncoder, timm ViT
    (H-optimus / H0-mini) -> TimmLiveEncoder."""
    backend = config.MODELS[model_key]["backend"]
    if backend == "transformers":
        return LiveEncoder(model_key)
    if backend == "timm":
        return TimmLiveEncoder(model_key)
    raise NotImplementedError(f"no live encoder for backend {backend!r} ({model_key})")
