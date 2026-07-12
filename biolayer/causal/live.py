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
