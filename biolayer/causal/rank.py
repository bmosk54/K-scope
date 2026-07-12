"""Certified tile ranking — project tiles onto a GATED concept axis, sorted.

Ranking a WSI's tiles by "how <concept>-like" is a dot product against a concept
direction. The value here is NOT the sort — a raw probe does that. It is that the
direction is CERTIFIED first: it must clear the same held-out-AUROC + intensity gate
(`dynamic/contrast.py`) the certificate uses, or the ranking is refused / loudly
flagged. A confident ranking on an axis that rides staining intensity is confidently
wrong (the immune-exclusion case).

    axis = fit_certified_axis(ref_feats, ref_labels, class_names, "TUM", "LYM")
    ranking = rank_tiles(wsi_tile_feats, axis)   # fit on ref, score on WSI — never the same tiles
    ranking.order          # tile indices, most-concept-like first
    ranking.scores         # signed distance along the concept axis (original tile order)

The axis is the normalized probe weight vector (`probe.fit_probe`); scoring standardizes
each tile with the probe's own scaler, then projects onto that unit direction. The tile
embeddings are exactly the H-optimus-0 CLS produced by
`deploy/sagemaker/tile_embed_entry.py::_embed` (or `data.models` extraction).
"""
from dataclasses import dataclass

import numpy as np

from ..dynamic import contrast as _contrast
from . import probe as _probe


@dataclass
class CertifiedAxis:
    """A concept direction + the gate verdict that says whether ranking on it is trusted."""
    pos: str
    neg: str
    direction: np.ndarray          # unit concept axis in STANDARDIZED feature space
    scaler_mean: np.ndarray
    scaler_scale: np.ndarray
    coef: np.ndarray               # probe weight in standardized space (for live grounding)
    intercept: float
    heldout_auroc: float
    intensity_collinearity: float  # |corr(concept proj, CLS-norm)| — the nuisance screen
    adjudication: str              # screen-clean | survives-control | confound-real | undecidable
    certified: bool                # gate cleared -> ranking is trustworthy
    reason: str
    flags: tuple = ()              # non-invalidating annotations (admitted-with-flag)
    warnings: tuple = ()
    n_ref: int = 0


def fit_certified_axis(ref_feats, ref_labels, class_names, pos, neg, seed=0):
    """Fit the concept axis on a LABELED reference set and run the certification gate.

    Reuses `probe.fit_probe` for the direction and `contrast.py`'s gate (held-out AUROC +
    intensity screen + Gate-2b controlled adjudication) — no re-invented direction fitting.
    `certified` is True iff the gate clears: AUROC >= threshold AND the intensity screen is
    clean or the signal SURVIVES the intensity-matched control.
    """
    X, y = _probe.select_pair(np.asarray(ref_feats), np.asarray(ref_labels),
                              class_names, pos, neg)
    fit = _probe.fit_probe(X, y, seed=seed)

    auroc = _contrast._heldout_auroc(X, y, seed=seed)
    coll = _contrast._intensity_collinearity(X, y)
    adjudication, flags, warnings = "screen-clean", [], []
    if not (auroc >= _contrast.MIN_HELDOUT_AUROC):
        warnings.append(f"held-out AUROC {auroc:.3f} < {_contrast.MIN_HELDOUT_AUROC}")
    if coll > _contrast.MAX_INTENSITY_COLLINEARITY:  # Gate 2b — control, don't veto on corr
        adjudication, _ma, _mc, _nm, adj_w, adj_f = \
            _contrast._adjudicate_intensity(X, y, coll, seed=seed)
        warnings += list(adj_w)
        flags += list(adj_f)

    certified = (auroc >= _contrast.MIN_HELDOUT_AUROC and
                 adjudication in ("screen-clean", "survives-control"))
    reason = ("gate PASS" if certified else
              f"gate FAIL (AUROC={auroc:.3f}, intensity |r|={coll:.2f}, "
              f"adjudication={adjudication})")
    return CertifiedAxis(
        pos=pos, neg=neg, direction=fit["direction"],
        scaler_mean=fit["scaler"].mean_, scaler_scale=fit["scaler"].scale_,
        coef=fit["w"], intercept=fit["b"],
        heldout_auroc=auroc, intensity_collinearity=coll,
        adjudication=adjudication, certified=certified, reason=reason,
        flags=tuple(flags), warnings=tuple(warnings), n_ref=int(len(y)))


@dataclass
class TileRanking:
    order: np.ndarray          # tile indices sorted by score, most-concept-like first
    scores: np.ndarray         # score per tile in ORIGINAL order (signed dist along axis)
    ranked_scores: np.ndarray  # scores[order] — the sorted scores
    axis: CertifiedAxis
    certified: bool
    note: str


def rank_tiles(tile_feats, axis, require_certified=True):
    """Rank unlabeled tile embeddings by alignment to a certified concept axis.

        score_i = standardize(tile_i) . axis.direction   (signed distance along the axis)

    Returns tiles sorted most-concept-like first. Fit the axis on a labeled REFERENCE set
    (fit_certified_axis) and pass DIFFERENT, unlabeled tiles here — never fit and score on
    the same tiles. If the axis is not certified, refuses (require_certified=True, default)
    or returns with a loud flag (require_certified=False).
    """
    if not axis.certified:
        msg = (f"axis {axis.pos}_vs_{axis.neg} is NOT certified — {axis.reason}; "
               f"a ranking on it is untrustworthy")
        if require_certified:
            raise ValueError(msg + " (pass require_certified=False to override)")
    T = np.asarray(tile_feats, dtype=np.float64)
    if T.ndim != 2 or T.shape[1] != axis.direction.shape[0]:
        raise ValueError(f"tile_feats must be (N,{axis.direction.shape[0]}); got {T.shape}")
    Z = (T - axis.scaler_mean) / axis.scaler_scale
    scores = Z @ axis.direction
    order = np.argsort(-scores)
    note = ("certified axis (gate PASS)" if axis.certified else
            f"UNCERTIFIED axis — {axis.reason}; scores may be confounded")
    return TileRanking(order=order, scores=scores, ranked_scores=scores[order],
                       axis=axis, certified=axis.certified, note=note)
