"""screen — the spatial CRISPR knockout screen (leave-one-region-out ablation).

The per-patch heatmap in `attribution.py` is ATTRIBUTION: it projects each patch
token onto the concept axis and z-scores it vs a matched-random-direction null. That
says which patches *correlate* with the concept inside the model — it never removes
anything. This module runs the *interventional* analog: like a CRISPR knockout screen,
it knocks out one tissue region at a time, re-forwards the whole tile, and measures the
collapse in the readout margin. Regions are ranked by their causal effect on the SAME
readout probe `certify` grounds, and the top-k are certified against a matched-random-
REGION null (the spatial version of the non-negotiable Section-5-D control).

Fidelity: this is an INPUT occlusion do() — mask the pixels of region i and re-forward
the intact-elsewhere tile. It is a faithful intervention on the model's input (no hooks,
fully on the real forward pass); it is the honest, cheapest grade of the "mask patches
and recompute CLS" variant `attribution.hack_tile` was stubbed for. The latent patch-
token knockout (hook the residual stream, let downstream blocks recompute) is the
follow-on grade that also exposes spatial redundancy — not built here.

Split like `battery.py` / `attribution.py`:
  - MODEL-FREE core `assemble_screen(...)`  — pure numpy over CLS vectors + certify
    handles; unit-testable on synthetic grids with no model.
  - LIVE wrappers `screen_tile` / `screen_concept` — mask real tiles and re-forward
    through a resident encoder (`serving.warm_encoder`) whose `.embed(images)` returns
    the readout CLS the certify probe is fit on.

Honesty caveat (same spirit as certify): occluding pixels tests the model's dependence
on that image region, not a biological knockout of that tissue. It certifies WHERE in
the input the readout causally reads from — biological validity still rests on encoder
faithfulness (why the confound gate and literature grounding exist).
"""
import numpy as np

from PIL import Image


# --------------------------------------------------------------------------
# Readout margin — the SAME probe certify persists (coef/intercept in the
# standardized CLS space Z=(X-mean)/scale). margin > 0 => the pos class.
# --------------------------------------------------------------------------
def _margin(cls, handles):
    """Probe logit(s) for CLS vector(s) under certify's persisted readout probe."""
    Z = (np.asarray(cls, dtype=np.float64) - handles["scaler_mean"]) / handles["scaler_scale"]
    return Z @ np.asarray(handles["coef"], dtype=np.float64).ravel() + float(handles["intercept"])


# --------------------------------------------------------------------------
# Model-free core: given the clean CLS + one CLS per single-region knockout
# (+ optional concept/random SET knockouts), assemble the causal screen card.
# --------------------------------------------------------------------------
def assemble_screen(clean_cls, region_ko_cls, handles, grid_side,
                    concept_set_cls=None, random_set_cls=None, topk=None):
    """Assemble the knockout map + matched-random-region null from precomputed CLS.

    clean_cls       : (D,)    CLS of the intact tile
    region_ko_cls   : (R, D)  CLS after masking each region once; row r <-> region r,
                              R == grid_side**2 (exhaustive leave-one-region-out)
    concept_set_cls : (D,)    CLS after masking the top-k concept regions TOGETHER
    random_set_cls  : (M, D)  CLS after masking M matched-size RANDOM region sets
    Returns the screen card (all json-safe).
    """
    clean_cls = np.asarray(clean_cls, dtype=np.float64)
    region_ko_cls = np.asarray(region_ko_cls, dtype=np.float64)
    R = region_ko_cls.shape[0]
    if R != grid_side * grid_side:
        raise ValueError(f"expected {grid_side**2} region knockouts, got {R}")

    m0 = float(_margin(clean_cls[None], handles)[0])
    m = _margin(region_ko_cls, handles)          # (R,)
    delta = m0 - m                               # +ve => region SUPPORTED the concept
    mu, sd = float(delta.mean()), float(delta.std() + 1e-9)
    z = (delta - mu) / sd                        # per-region causal z vs the occlusion field
    order = np.argsort(-delta)                   # most concept-supporting region first
    topk = int(topk or max(1, R // 8))

    pos = np.clip(delta, 0.0, None)              # concept-supporting deltas only
    norm = (pos / (pos.max() + 1e-9)).reshape(grid_side, grid_side)  # [0,1] overlay alpha
    card = {
        "kind": "input_occlusion",
        "grid_side": grid_side,
        "n_regions": int(R),
        "clean_margin": round(m0, 4),
        "predicted_pos": bool(m0 > 0),
        "delta_grid": np.round(delta.reshape(grid_side, grid_side), 4).tolist(),
        "z_grid": np.round(z.reshape(grid_side, grid_side), 3).tolist(),
        "norm_grid": np.round(norm, 4).tolist(),
        "delta_min": round(float(delta.min()), 4),
        "delta_max": round(float(delta.max()), 4),
        "top_regions": [int(i) for i in order[:topk]],
        "top_region": int(order[0]),
        "top_region_rc": [int(order[0] // grid_side), int(order[0] % grid_side)],
        "top_delta": round(float(delta[order[0]]), 4),
        "top_z": round(float(z[order[0]]), 2),
        "topk": topk,
    }

    # Matched-random-REGION null (the Section-5-D falsifier, spatial form): masking the
    # k concept regions together must collapse the margin more than masking k RANDOM
    # regions of the same area. If it does not, the concept is spatially diffuse /
    # redundant and no localization claim is safe.
    if concept_set_cls is not None and random_set_cls is not None:
        m_concept = float(_margin(np.asarray(concept_set_cls)[None], handles)[0])
        m_random = _margin(np.asarray(random_set_cls), handles)     # (M,)
        d_concept = m0 - m_concept
        d_random = m0 - m_random
        r_mu, r_sd = float(d_random.mean()), float(d_random.std() + 1e-9)
        gap = d_concept - r_mu
        null_z = gap / r_sd
        concept_flips = bool(m0 > 0 and m_concept < 0)
        rand_flip = float(np.mean((np.asarray(m_random) < 0))) if m0 > 0 else 0.0
        passed = bool(null_z > 3 and gap > 0)
        card["null"] = {
            "k": topk,
            "concept_region_drop": round(d_concept, 4),
            "random_region_drop_mean": round(r_mu, 4),
            "random_region_drop_std": round(r_sd, 4),
            "necessity_gap": round(gap, 4),
            "null_z": round(null_z, 2),
            "concept_ko_margin": round(m_concept, 4),
            "concept_ko_flips_prediction": concept_flips,
            "random_ko_flip_rate": round(rand_flip, 3),
            "m_random_sets": int(len(m_random)),
            "passed": passed,
        }
        card["verdict"] = (
            f"concept LOCALIZES: knocking out the top-{topk} regions collapses the "
            f"readout margin beyond the matched-random-region null (gap {gap:+.3f}, "
            f"null-z {null_z:.1f}" + (", flips the prediction" if concept_flips else "") + ")"
            if passed else
            "diffuse / spatially redundant: no region set clears the matched-random-"
            "region null — localization is not certifiable on this tile")
    else:
        card["verdict"] = ("concept concentrates in specific regions above the occlusion "
                           f"field (top-z {card['top_z']:.1f})" if card["top_z"] > 3 else
                           "no single region stands out above the occlusion field")
    return card


# --------------------------------------------------------------------------
# Masking helpers (input-space occlusion on the real tile).
# --------------------------------------------------------------------------
def _region_box(idx, grid_side, w, h):
    """Pixel box (x0, y0, x1, y1) for region `idx` of a grid_side x grid_side tiling."""
    r, c = divmod(idx, grid_side)
    x0, x1 = int(round(c * w / grid_side)), int(round((c + 1) * w / grid_side))
    y0, y1 = int(round(r * h / grid_side)), int(round((r + 1) * h / grid_side))
    return x0, y0, x1, y1


def _fill_color(arr, mode):
    """Neutral occlusion fill. 'image_mean' stays closest to the tile's manifold."""
    if mode == "image_mean":
        return tuple(int(v) for v in arr.reshape(-1, arr.shape[-1]).mean(0))
    if mode == "gray":
        return (128, 128, 128)
    if mode == "zero":
        return (0, 0, 0)
    raise ValueError(f"unknown fill {mode!r}")


def _masked(arr, boxes, color):
    """Copy `arr` (HxWx3 uint8) with every box painted `color`. One or many boxes."""
    out = arr.copy()
    for x0, y0, x1, y1 in boxes:
        out[y0:y1, x0:x1, :] = color
    return Image.fromarray(out)


# --------------------------------------------------------------------------
# Live wrappers: mask real tiles, re-forward, assemble.
# --------------------------------------------------------------------------
def screen_tile(enc, image, handles, grid_side=7, m_random=16, topk=None,
                fill="image_mean", seed=0, batch_size=64):
    """Occlusion screen ONE tile through a resident encoder (`enc.embed(images)` -> CLS).

    Forwards: 1 clean + R single-region KO + 1 concept-set KO + m_random random-set KO.
    Returns the screen card (see assemble_screen). `enc` is any object with an
    `.embed(list_of_PIL) -> (N, D)` readout-CLS method (serving.warm_encoder gives one).
    """
    img = image.convert("RGB")
    arr = np.asarray(img)
    h, w = arr.shape[:2]
    color = _fill_color(arr, fill)
    R = grid_side * grid_side
    topk = int(topk or max(1, R // 8))
    boxes = [_region_box(i, grid_side, w, h) for i in range(R)]

    clean_cls = enc.embed([img], batch_size=batch_size)[0]                     # (D,)
    region_cls = enc.embed([_masked(arr, [b], color) for b in boxes],
                           batch_size=batch_size)                              # (R, D)

    # Pick the top-k concept regions from the single-KO deltas (same delta assemble uses),
    # then build the concept-set KO and a matched-random-region null.
    delta = float(_margin(clean_cls[None], handles)[0]) - _margin(region_cls, handles)
    top_ids = np.argsort(-delta)[:topk]
    concept_cls = enc.embed([_masked(arr, [boxes[i] for i in top_ids], color)],
                            batch_size=batch_size)[0]
    rng = np.random.default_rng(seed)
    rand_imgs = [_masked(arr, [boxes[int(i)] for i in rng.choice(R, size=topk, replace=False)],
                         color) for _ in range(m_random)]
    random_cls = enc.embed(rand_imgs, batch_size=batch_size)

    card = assemble_screen(clean_cls, region_cls, handles, grid_side,
                           concept_set_cls=concept_cls, random_set_cls=random_cls, topk=topk)
    card["fill"] = fill
    card["region_px"] = int(round(w / grid_side))
    return card


def screen_concept(enc, pos_images, handles, max_tiles=6, **kw):
    """Screen the most confidently-positive pos-class tile (highest clean readout margin),
    so the knockout map is measured on a tile the model actually calls the concept.
    Returns the screen card (+ which candidate tile), or None if no tiles given."""
    imgs = [im.convert("RGB") for im in (pos_images or [])[:max_tiles]]
    if not imgs:
        return None
    cls = enc.embed(imgs)
    best = int(np.argmax(_margin(cls, handles)))
    card = screen_tile(enc, imgs[best], handles, **kw)
    card["tile_index"] = best
    card["n_candidate_tiles"] = len(imgs)
    return card
