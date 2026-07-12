"""Load H-optimus-0 via timm and modify its weights arbitrarily.

Runs anywhere with a GPU + HF auth (locally, or inside the SageMaker training job
via entry.py). Two things it gives you:

  load_hoptimus()  -> the timm ViT-g/14, ready to manipulate
  edit_weights()   -> apply an arbitrary in-place transform to any named parameter

Everything is a plain nn.Module, so `.blocks`, `.named_parameters()`, hooks, freezing,
and block surgery all work. See DESIGN_MIL_AGGREGATOR.md for the block-reuse case.
"""
import torch
import torch.nn as nn

HF_ID = "hf-hub:bioptimus/H-optimus-0"
# H-optimus-0 needs these non-default timm args (mirrors biolayer.config).
TIMM_KWARGS = dict(init_values=1e-5, dynamic_img_size=False)
DIM = 1536


def load_hoptimus(pretrained: bool = True, device: str = None):
    """timm.create_model with the correct 'hf-hub:' prefix + required kwargs.

    pretrained=False skips the gated download (random init) — handy to smoke-test
    the plumbing without HF auth. Pretrained needs `hf auth login` + accepted terms.
    """
    import timm

    model = timm.create_model(HF_ID, pretrained=pretrained, **TIMM_KWARGS).eval()
    assert model.embed_dim == DIM, model.embed_dim
    if device:
        model = model.to(device)
    return model


# ---------------------------------------------------------------------------
# Arbitrary weight modification
# ---------------------------------------------------------------------------
def edit_weights(model: nn.Module, edits: dict) -> nn.Module:
    """Apply arbitrary in-place edits to named parameters.

    `edits` maps a parameter name (as in model.named_parameters()) to a function
    tensor -> tensor. Runs under no_grad and copies the result back in place.

        edit_weights(model, {
            "blocks.39.mlp.fc2.weight": lambda w: w * 0.5,          # scale
            "blocks.39.attn.proj.bias": lambda b: torch.zeros_like(b),  # zero out
        })
    """
    named = dict(model.named_parameters())
    with torch.no_grad():
        for name, fn in edits.items():
            if name not in named:
                raise KeyError(f"no parameter {name!r}; "
                               f"e.g. {list(named)[:3]} ... ({len(named)} total)")
            p = named[name]
            new = fn(p)
            if new.shape != p.shape:
                raise ValueError(f"{name}: edit changed shape {tuple(p.shape)} -> "
                                 f"{tuple(new.shape)}")
            p.copy_(new)
    return model


def ablate_direction(weight: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
    """Project a unit `direction` out of every row of `weight` (concept ablation)."""
    d = direction / direction.norm()
    return weight - (weight @ d).unsqueeze(-1) * d


def freeze(model: nn.Module, trainable_prefixes=()):
    """Freeze all params except those whose name starts with a given prefix."""
    for name, p in model.named_parameters():
        p.requires_grad_(any(name.startswith(pre) for pre in trainable_prefixes))
    return model


# ---------------------------------------------------------------------------
# Example edits (illustrative — compose your own)
# ---------------------------------------------------------------------------
def example_edits(model: nn.Module):
    """Demonstrate a few arbitrary manipulations; returns a description list."""
    log = []

    # 1. Scale the last block's MLP output projection.
    edit_weights(model, {"blocks.39.mlp.fc2.weight": lambda w: w * 0.9})
    log.append("scaled blocks.39.mlp.fc2.weight by 0.9")

    # 2. Ablate a (here random) concept direction from the final norm's affine.
    dirn = torch.randn(model.embed_dim)
    if "norm.weight" in dict(model.named_parameters()):
        edit_weights(model, {"norm.weight": lambda w: ablate_direction(w.unsqueeze(0), dirn).squeeze(0)})
        log.append("ablated a direction from norm.weight")

    # 3. Freeze everything except the last block (typical for the MIL-reuse fine-tune).
    freeze(model, trainable_prefixes=("blocks.39",))
    n_train = sum(p.requires_grad for p in model.parameters())
    log.append(f"froze all but blocks.39 ({n_train} trainable tensors)")

    # 4. Register an activation hook on a mid block (causal-intervention style).
    def _hook(_m, _inp, out):
        return out  # no-op; edit `out` here to intervene on activations
    model.blocks[27].register_forward_hook(_hook)
    log.append("registered forward hook on blocks.27 (activation intervention point)")

    return log
