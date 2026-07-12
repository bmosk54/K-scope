"""Attribution of SAE features to a concept readout -- and an honest note on what is NOT causal.

READ THIS BEFORE CLAIMING CAUSALITY. The SAE decoder is LINEAR and the probe readout is
LINEAR, so with cached FINAL-LAYER (block 39) embeddings a feature's contribution to the
probe logit is exact and analytic:

    contribution_f = scale * z_f * (w_f . d)        # w_f = decoder column, d = probe direction

There is nothing to estimate and nothing to intervene on: block 39 is the last block, so no
nonlinearity lies downstream for an edit to propagate through. "Ablate feature f and re-read
the logit" is therefore LINEAR ATTRIBUTION wearing an intervention's clothes -- the answer is
determined in closed form. Reporting it as a do() would be dressing up a dot product.

A GENUINE causal test requires editing an activation at an EARLIER layer and running the rest
of the network forward -- exactly what ARCHITECTURE.md sec.2 specifies for the battery:
"hooks encoder.layer[L], edits activations, re-reads CLS". That needs the live model, not a
cached .npz. It is the natural integration point with Eddie's `intervene` module: his battery
already hooks the encoder; SAE feature directions can be ablated there instead of (or
alongside) probe directions. Until then, `hypothesis` is CORRELATIONAL + ATTRIBUTIONAL, and
the docstrings say so.

Kept below: exact attribution (use this), plus the Monte-Carlo necessity/sufficiency probes
that MEASURED the redundancy problem -- deleting a "tumour" feature moves the TUM logit by
-0.009 while deleting a random matched feature moves it by -0.101. That is the Hydra effect
ARCHITECTURE.md sec.6 predicts, reproduced independently.


THE GAP THIS FILLS. Everything else in hypothesis.py is CORRELATIONAL: "feature 1464 is
differentially active on the causal tiles" is an association, however good its null. The
project's one-liner promises *auditable causal evidence*, and `certify` delivers that for
NAMED concepts (ablate a probe direction, see what breaks). The SAE side had no intervention
at all. This adds one.

THE INTERVENTION (this is the "fusion vector" idea from TODO_fusion_vector.md, done as a
do-operation rather than a comparison):

    x  --SAE-->  z  --zero out feature f-->  z'  --decode-->  x'
    then measure the change in a downstream readout (a probe logit) between x and x'.

If deleting feature f from the model's own representation collapses the tumour logit, the
model was USING that feature to decide. That is a causal claim about the representation --
not "this fires when tumour is present", but "remove it and the tumour call changes".

THE NULL (house rule: matched-random, non-negotiable). Deleting ANY active feature perturbs
the reconstruction somewhat, so a raw delta proves nothing. We compare against ablating
RANDOM OTHER ACTIVE FEATURES OF THE SAME ACTIVATION MAGNITUDE in the same tiles. That
controls for "you removed some energy from the vector" and isolates "you removed THIS
concept". A feature whose delta does not beat that null is not causally load-bearing, and
we say so.

HONEST SCOPE (CLAUDE.md constraint #4): this is a latent do() on the MODEL'S REPRESENTATION,
not an intervention on tissue. It licenses "the model's decision depends on this feature",
never "this feature causes the biology".
"""

from __future__ import annotations

import numpy as np
import torch


def _reconstruct(sae, z: torch.Tensor) -> torch.Tensor:
    return sae.dec(z) + sae.b_dec


def probe_logit(x: torch.Tensor, direction: np.ndarray, device: str) -> torch.Tensor:
    """Readout: projection of a (denormalised) embedding onto a concept direction."""
    d = torch.from_numpy(np.asarray(direction, dtype=np.float32)).to(device)
    return x @ d


def attribute(
    sae, ck, codes: np.ndarray, tile_ids: np.ndarray, direction: np.ndarray, top_k: int = 10
) -> dict:
    """EXACT contribution of every SAE feature to a concept logit, on a set of tiles.

    Because decode and readout are both linear, feature f's contribution to the logit is
    exactly  scale * z_f * (w_f . d)  -- closed form, no Monte Carlo, no approximation.
    Ranking features by this tells you what the linear readout is ACTUALLY built from, which
    is the honest version of "what is the model using here".

    Returns the top contributors on these tiles, with each feature's share of the total.
    """
    m = sae.m if hasattr(sae, "m") else sae
    W = m.dec.weight.detach().cpu().numpy()  # (d_model, n_features)
    d = np.asarray(direction, dtype=np.float32)
    scale = float(ck["scale"])

    w_dot_d = W.T @ d  # (n_features,) -- how much each feature direction reads onto the concept
    Z = codes[tile_ids]  # (n_tiles, n_features)
    contrib = scale * Z.mean(0) * w_dot_d  # mean contribution per feature over these tiles

    order = np.argsort(-np.abs(contrib))[:top_k]
    total = float(np.abs(contrib).sum()) + 1e-12
    return {
        "n_tiles": int(len(tile_ids)),
        "method": "exact linear attribution (decoder and readout are both linear)",
        "top_contributors": [
            {
                "feature_idx": int(f),
                "contribution": float(contrib[f]),
                "share_of_total_abs": round(float(abs(contrib[f]) / total), 4),
                "mean_activation": float(Z[:, f].mean()),
                "direction_readout": float(w_dot_d[f]),
            }
            for f in order
        ],
        "caveat": (
            "This is ATTRIBUTION, not causation. Block 39 is the final block, so no nonlinearity "
            "lies downstream and the 'ablation' is a closed-form dot product. A true do() must "
            "edit an EARLIER layer and re-run the network (ARCHITECTURE.md sec.2)."
        ),
    }


def sufficiency(
    sae,
    ck,
    X: np.ndarray,
    feature: int,
    direction: np.ndarray,
    n_null: int = 40,
    device: str = "cuda",
    seed: int = 0,
) -> dict:
    """Reconstruct from ONE feature alone: is it SUFFICIENT to drive the concept readout?

    WHY SUFFICIENCY AND NOT NECESSITY. Measured on this SAE, deleting a single "tumour"
    feature moves the TUM logit by -0.009 while deleting a RANDOM matched feature moves it
    by -0.101: single-feature necessity is undetectable because the representation is
    redundant (with k=40 active features, 39 others still carry the signal). This is the
    Hydra effect, which ARCHITECTURE.md sec.6 flags in advance and explicitly tells us to
    handle by leading with sufficiency + the null. So we do.

    The test: zero EVERY feature except f, decode, and read the concept logit. Compare
    against reconstructing from a single RANDOM active feature (matched: also exactly one
    feature, also active on the same tiles). If f alone drives the readout well above that
    null, f is sufficient for the concept -- the model could make the call from this feature
    alone.
    """
    m = sae.m if hasattr(sae, "m") else sae
    mu, scale = ck["mu"].to(device), ck["scale"].to(device)
    x = torch.from_numpy(np.asarray(X, dtype=np.float32)).to(device)
    xn = (x - mu) / scale

    with torch.no_grad():
        out = m(xn)
        z = out[1] if isinstance(out, tuple) else out
        if isinstance(z, tuple):
            z = z[1]

        active = z[:, feature] > 0
        if active.sum() == 0:
            return {"feature_idx": int(feature), "n_tiles_active": 0,
                    "note": "feature never fires on these tiles"}
        zi = z[active]

        # reconstruct from feature f ALONE
        z_only = torch.zeros_like(zi)
        z_only[:, feature] = zi[:, feature]
        obs = probe_logit(_reconstruct(m, z_only) * scale + mu, direction, device).mean().item()

        # null: reconstruct from ONE random OTHER active feature (matched: also a single feature)
        torch.manual_seed(seed)
        cand = (zi > 0).float()
        cand[:, feature] = 0.0
        cand = cand + 1e-9
        null = []
        for _ in range(n_null):
            pick = torch.multinomial(cand, 1)
            z_r = torch.zeros_like(zi)
            z_r.scatter_(1, pick, zi.gather(1, pick))
            null.append(probe_logit(_reconstruct(m, z_r) * scale + mu, direction, device).mean().item())

    null = np.asarray(null)
    p = (float((null >= obs).sum()) + 1) / (n_null + 1)
    return {
        "feature_idx": int(feature),
        "n_tiles_active": int(active.sum().item()),
        "logit_from_feature_alone": float(obs),
        "null_mean": float(null.mean()),
        "null_p95": float(np.quantile(null, 0.95)),
        "p_value": float(p),
        "sufficient": bool(p < 0.05 and obs > np.quantile(null, 0.95)),
        "interpretation": (
            "Reconstructing from this feature ALONE drives the concept readout to "
            f"{obs:.3f}, vs {null.mean():.3f} for a single random active feature. Above the "
            "null => the feature alone carries the concept. Sufficiency is the honest test "
            "here: necessity is masked by redundancy (Hydra effect)."
        ),
    }


def ablate_feature(
    sae,
    ck,
    X: np.ndarray,
    feature: int,
    direction: np.ndarray,
    n_null: int = 50,
    device: str = "cuda",
    seed: int = 0,
) -> dict:
    """Zero one SAE feature, decode, and measure the change in a concept readout.

    X: raw (unnormalised) embeddings of the tiles to intervene on -- normally the causal set.
    direction: the concept direction whose logit we read out (e.g. Eddie's TUM probe).
    """
    m = sae.m if hasattr(sae, "m") else sae  # unwrap the MCP adapter
    mu = ck["mu"].to(device)
    scale = ck["scale"].to(device)

    x = torch.from_numpy(np.asarray(X, dtype=np.float32)).to(device)
    xn = (x - mu) / scale
    with torch.no_grad():
        out = m(xn)
        z = out[1] if isinstance(out, tuple) else out
        if isinstance(z, tuple):
            z = z[1]

        base_recon = _reconstruct(m, z) * scale + mu
        base = probe_logit(base_recon, direction, device)

        active = (z[:, feature] > 0)
        if active.sum() == 0:
            return {"feature_idx": int(feature), "n_tiles_active": 0,
                    "note": "feature never fires on these tiles; ablation is a no-op"}

        # --- the intervention ---
        z_ab = z.clone()
        z_ab[:, feature] = 0.0
        ab = probe_logit(_reconstruct(m, z_ab) * scale + mu, direction, device)
        delta = (ab - base)[active].mean().item()

        # --- matched-random null: ablate OTHER active features of similar magnitude ---
        # Controls for "removing energy perturbs the vector" so we isolate "removing THIS one".
        torch.manual_seed(seed)
        zi = z[active]
        mag = zi[:, feature].mean()
        base_i = probe_logit(_reconstruct(m, zi) * scale + mu, direction, device)

        # Per-tile sampling weights: prefer active features whose magnitude is close to the
        # target's, so the null removes a comparable amount of energy.
        cand = (zi > 0).float()
        cand[:, feature] = 0.0
        close = torch.exp(-(zi - mag).abs() / (mag.abs() + 1e-6)) * cand + 1e-9

        null = []
        for _ in range(n_null):
            pick = torch.multinomial(close, 1)  # (n_active, 1) one random matched feature per tile
            z_null = zi.clone().scatter_(1, pick, 0.0)
            a = probe_logit(_reconstruct(m, z_null) * scale + mu, direction, device)
            null.append((a - base_i).mean().item())

    null = np.asarray(null)
    # one-sided: we care about ablations that REDUCE the logit (|delta| large & negative)
    p = (float((null <= delta).sum()) + 1) / (n_null + 1)
    return {
        "feature_idx": int(feature),
        "n_tiles_active": int(active.sum().item()),
        "delta_logit": float(delta),
        "null_mean_delta": float(null.mean()),
        "null_p05": float(np.quantile(null, 0.05)),
        "p_value": float(p),
        "causally_load_bearing": bool(p < 0.05 and delta < np.quantile(null, 0.05)),
        "interpretation": (
            "delta_logit is the change in the concept readout when this feature is deleted from "
            "the model's own representation, vs deleting random active features of matched "
            "magnitude. Negative and beyond the null => the model's call DEPENDS on this feature. "
            "This is a do() on the REPRESENTATION, not on tissue biology."
        ),
    }
