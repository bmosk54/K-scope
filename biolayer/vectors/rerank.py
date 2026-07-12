"""Rerank the two ordered vector lists by the CERTIFIED concept-axis dot-product ranker.

Wires `biolayer.causal.rank` (a teammate's certified tile ranking) into `OrderedVectorList`:
`rank_tiles` scores a tile by `standardize(tile) · direction` on a concept axis that first
had to clear the held-out-AUROC + intensity gate. Here we apply that SAME scoring to a
list's rows and rerank in place — directly for the GLOBAL/CLS list, and in memmap-friendly
chunks for the huge PATCH list (its shards are never fully materialized).

    from biolayer.vectors import load_global, rerank_by_concept
    gl = load_global(open("global.npz","rb").read())
    verdict = rerank_by_concept(gl, "TUM", "LYM")     # fit axis on H-optimus-0 NCT-CRC ref
    gl.order                                           # tiles, most-TUM-like first
    verdict["certified"], verdict["heldout_auroc"]     # why the ranking is (un)trusted

The axis is fit on a LABELED reference set (NCT-CRC), the WSI tiles are scored — never the
same tiles, per the ranker's contract.
"""
import numpy as np

from ..causal.rank import fit_certified_axis


def axis_scores(vectors, axis, chunk=100_000):
    """Signed distance of each row along the certified axis: `standardize(row) · direction`
    (identical to causal.rank.rank_tiles). Streams over `vectors` (ndarray or the PATCH
    list's `_ShardedArray`) in chunks so memmap shards are never fully materialized."""
    mean = np.asarray(axis.scaler_mean, dtype=np.float64)
    scale = np.asarray(axis.scaler_scale, dtype=np.float64)
    d = np.asarray(axis.direction, dtype=np.float64)
    n = len(vectors)
    out = np.empty(n, dtype=np.float64)
    for i in range(0, n, chunk):
        j = min(i + chunk, n)
        T = np.asarray(vectors[np.arange(i, j)], dtype=np.float64)
        out[i:j] = ((T - mean) / scale) @ d
    return out


def _verdict(axis, vlist, scores):
    order = vlist.order
    return {"concept": f"{axis.pos}_vs_{axis.neg}", "certified": bool(axis.certified),
            "reason": axis.reason, "heldout_auroc": float(axis.heldout_auroc),
            "intensity_collinearity": float(axis.intensity_collinearity),
            "adjudication": axis.adjudication, "n": int(len(vlist)),
            "top_score": float(scores[order[0]]) if len(vlist) else None,
            "flags": list(axis.flags), "warnings": list(axis.warnings)}


def rerank_by_axis(vlist, axis, require_certified=True, chunk=100_000):
    """Rerank an `OrderedVectorList` by alignment to a pre-fit `CertifiedAxis`
    (most-concept-like first). Refuses on an uncertified axis unless `require_certified=False`
    (then the verdict carries the gate failure). Mutates only `vlist.order`; returns a verdict."""
    if not axis.certified and require_certified:
        raise ValueError(f"axis {axis.pos}_vs_{axis.neg} is NOT certified — {axis.reason}; "
                         "a ranking on it is untrustworthy (pass require_certified=False to override)")
    if vlist.dim != int(np.asarray(axis.direction).shape[0]):
        raise ValueError(f"list dim {vlist.dim} != axis dim {np.asarray(axis.direction).shape[0]}")
    scores = axis_scores(vlist.vectors, axis, chunk=chunk)
    vlist.rerank(scores)                                    # descending: highest score first
    return _verdict(axis, vlist, scores)


def rerank_by_concept(vlist, pos, neg, ref_model="h_optimus_0", split="train",
                      require_certified=True, chunk=100_000):
    """Fit a certified concept axis on the labeled NCT-CRC reference set (H-optimus-0 by
    default — the lists' encoder) and rerank `vlist` by it. One call from a list + a concept
    contrast to a trusted, gated ranking. Returns the gate verdict."""
    from ..data import loader

    feats, labels, class_names, _ = loader.load(ref_model, split)
    axis = fit_certified_axis(feats, labels, class_names, pos, neg)
    return rerank_by_axis(vlist, axis, require_certified=require_certified, chunk=chunk)
