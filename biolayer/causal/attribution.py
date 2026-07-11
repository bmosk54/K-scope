"""Patch-level attribution — the "hack": which patches build the concept global.

The global (CLS) embedding aggregates patch tokens. This module finds which patches
carry the concept and rebuilds a concept-focused global from them, each certified
against a matched-random null (random directions must not single out patches).

Model-free core (operates on a patch-token grid + a concept direction), so it is
unit-testable on synthetic/cached grids exactly like battery.py. `hack_tile` is the
live wrapper that re-forwards one real tile to obtain the patch grid.

Two constructions of the new global (state which you mean — they differ in honesty):
  soft  : softmax(proj)-weighted pool of patch tokens  -> a CONSTRUCTED feature,
          not the model's CLS. Cheap, no re-forward.
  topk  : mean of the top-k concept patches            -> also constructed.
The faithful "mask patches and recompute CLS" variant is a live source-intervention
(shares hooks with intervene.necessity_curve, track #3) — not done here.
"""
import numpy as np

from . import probe as _probe


def _softmax(x):
    x = x - x.max()
    e = np.exp(x)
    return e / (e.sum() + 1e-12)


def patch_importance(patch_tokens, concept_dir, n_null=200, seed=0):
    """Rank patches by concept-projection, certified vs a matched-random-direction null.

    patch_tokens : (P, D) patch token grid for one tile at one layer
    concept_dir  : (D,) unit concept direction (raw feature space)
    Returns per-patch signed score, |score|, a null-z (how far the patch's concept
    projection exceeds random-direction projections), and the importance ranking.
    """
    patch_tokens = np.asarray(patch_tokens, dtype=np.float64)
    P, D = patch_tokens.shape
    concept_dir = concept_dir / (np.linalg.norm(concept_dir) + 1e-12)
    scores = patch_tokens @ concept_dir               # (P,) signed projection
    absr = np.abs(scores)
    R = _probe.matched_random_dirs(D, n_null, seed=seed)
    null = np.abs(patch_tokens @ R.T)                 # (P, n_null)
    null_mean = null.mean(axis=1)
    null_std = null.std(axis=1) + 1e-9
    z = (absr - null_mean) / null_std                 # per-patch importance z
    order = np.argsort(-absr)
    # headline = how far the TOP patches exceed the null (the mean is drowned by
    # background patches, which are ~0 by construction).
    topk = max(1, min(10, P // 10))
    top_z = float(np.sort(z)[::-1][:topk].mean())
    return {
        "n_patches": int(P),
        "scores": scores,
        "abs": absr,
        "z": z,
        "ranking": order,
        "top_z": top_z,
        "topk": topk,
        "top_patch": int(order[0]),
        "verdict": ("concept singles out specific patches above the random-direction null"
                    if top_z > 3 else "no patch stands out above the null"),
    }


def concept_focused_global(patch_tokens, concept_dir, mode="soft", k=None, temp=1.0):
    """Build a new, concept-focused global embedding from the patch grid.

    mode="soft": softmax(proj/temp)-weighted pool.  mode="topk": mean of top-k patches.
    Returns (new_global (D,), weights (P,)).
    """
    patch_tokens = np.asarray(patch_tokens, dtype=np.float64)
    concept_dir = concept_dir / (np.linalg.norm(concept_dir) + 1e-12)
    scores = patch_tokens @ concept_dir
    if mode == "soft":
        w = _softmax(scores / max(temp, 1e-6))
    elif mode == "topk":
        k = k or max(1, len(scores) // 4)
        idx = np.argsort(-scores)[:k]
        w = np.zeros(len(scores))
        w[idx] = 1.0 / len(idx)
    else:
        raise ValueError(f"unknown mode {mode!r}")
    return w @ patch_tokens, w


def heatmap(scores):
    """Reshape per-patch scores to a square grid for a tile saliency map (or None)."""
    P = len(scores)
    s = int(round(P ** 0.5))
    return np.asarray(scores).reshape(s, s).tolist() if s * s == P else None


def attribution_report(patch_tokens, concept_dir, mode="soft", n_null=200, seed=0):
    """Full patch-attribution card: importance ranking + null + concept-focused global."""
    imp = patch_importance(patch_tokens, concept_dir, n_null=n_null, seed=seed)
    new_global, w = concept_focused_global(patch_tokens, concept_dir, mode=mode)
    return {
        "n_patches": imp["n_patches"],
        "top_patches": imp["ranking"][:8].tolist(),
        "top_z": imp["top_z"],
        "verdict": imp["verdict"],
        "new_global_mode": mode,
        "new_global_norm": float(np.linalg.norm(new_global)),
        "weight_concentration_top8": float(np.sort(w)[::-1][:8].sum()),
        "heatmap": heatmap(imp["abs"]),
        "caveat": ("the concept-focused global is a CONSTRUCTED feature (weighted patch "
                   "pool), not the model's CLS; the faithful mask-and-recompute variant "
                   "is a live source-intervention (track #3)"),
    }


def hack_tile(model_key, image, layer="readout", pos="TUM", neg="LYM",
              split="train", mode="soft", n_null=200, seed=0):
    """LIVE wrapper: re-forward one PIL tile -> patch grid -> attribution report.

    Needs the model (a forward pass) + cached embeddings to derive the concept axis.
    NOT YET IMPLEMENTED: requires a patch-grid forward (models.py currently mean-pools
    the local tokens). Track that owns intervene's live hooks builds this — the same
    forward that yields patch tokens here yields the mask-and-recompute global there.
    """
    raise NotImplementedError(
        "hack_tile: live patch-grid forward not built yet — add a patch-token return "
        "to models.embed (or a hook) and derive concept_dir from loader; then call "
        "attribution_report. Core (patch_importance / concept_focused_global) is ready.")
