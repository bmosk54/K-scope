"""Direction-as-query attention pooling — a question-conditioned slide embedding.

    global(question) = Σ_i softmax(vᵢ · direction / τ) · vᵢ

Single-query attention (query = a concept direction, keys = values = the tokens): a
concept-weighted centroid of a slide's tokens. τ→∞ recovers plain mean-pool (the
unconditional centroid); τ→0 collapses to the single top-ranked token; a moderate τ is a soft
top-k. Because the embedding space is ~linear in the concepts we fit axes for, the pooled
vector is a PROPER embedding: standardization is affine and the weights sum to 1, so for any
concept axis u,  standardize(pooled)·u = Σ_i wᵢ (standardize(vᵢ)·u) — the pooled vector's
read-out under every linear probe equals the weighted-average read-out of its tokens. It is a
faithful centroid (first moment), not a re-encoding: it discards non-linear / relational /
spatial structure (that is what the block-reuse aggregator would try to buy back).

House rule ([CLAUDE.md](../../CLAUDE.md)): always report the mean-pool control. If directional
pooling doesn't beat the unweighted centroid on the thing you care about, the conditioning is
adding nothing — and we say so.

Streams over the token list (ndarray or the PATCH list's sharded memmap) with a numerically
stable ONLINE softmax, so the multi-GB patch list is never materialized.
"""
import numpy as np


def _std(V, mean, scale):
    return (np.asarray(V, dtype=np.float64) - mean) / scale


class _Subset:
    """A fixed-row view over a base array (ndarray or _ShardedArray), so we can pool one
    slide's rows without copying. Indexing with an arange maps through `idx`."""

    def __init__(self, base, idx):
        self.base = base
        self.idx = np.asarray(idx)
        self.shape = (len(self.idx), base.shape[1])

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, k):
        return self.base[self.idx[np.asarray(k)]]


def directional_pool(vectors, direction, mean, scale, tau=1.0, chunk=100_000):
    """Σ_i softmax(stdᵢ·direction / τ) · vᵢ, streamed with an online softmax.

    `direction/mean/scale` define the standardized concept axis (as in a probe fit). Returns
    the pooled vector in the ORIGINAL token space (a convex combination of the tokens).
    """
    d = np.asarray(direction, dtype=np.float64)
    mean, scale = np.asarray(mean), np.asarray(scale)
    n = len(vectors)
    m, Z, acc = -np.inf, 0.0, None                    # running max, denom, numerator
    for i in range(0, n, chunk):
        j = min(i + chunk, n)
        Vo = np.asarray(vectors[np.arange(i, j)], dtype=np.float64)
        s = (_std(Vo, mean, scale) @ d) / tau
        nm = max(m, float(s.max()))
        rescale = np.exp(m - nm) if np.isfinite(m) else 0.0
        w = np.exp(s - nm)
        if acc is None:
            acc = np.zeros(Vo.shape[1])
        acc = acc * rescale + (w[:, None] * Vo).sum(0)
        Z = Z * rescale + float(w.sum())
        m = nm
    return acc / Z


def mean_pool(vectors, chunk=100_000):
    """Unweighted centroid — the control (τ→∞ limit of directional_pool)."""
    n = len(vectors)
    acc = None
    for i in range(0, n, chunk):
        Vo = np.asarray(vectors[np.arange(i, min(i + chunk, n))], dtype=np.float64)
        acc = Vo.sum(0) if acc is None else acc + Vo.sum(0)
    return acc / n


# --------------------------------------------------------------------------- #
# Axis adapters + per-slide pooling + concept read-out profile
# --------------------------------------------------------------------------- #
def _axis_parts(axis):
    """Accept a CertifiedAxis, a probe.fit_probe dict, or a plain (direction, mean, scale)."""
    if hasattr(axis, "direction") and hasattr(axis, "scaler_mean"):        # CertifiedAxis
        return np.asarray(axis.direction), np.asarray(axis.scaler_mean), np.asarray(axis.scaler_scale)
    if isinstance(axis, dict) and "scaler" in axis:                        # fit_probe dict
        return np.asarray(axis["direction"]), np.asarray(axis["scaler"].mean_), np.asarray(axis["scaler"].scale_)
    d, m, s = axis                                                         # tuple
    return np.asarray(d), np.asarray(m), np.asarray(s)


def concept_score(vec, axis):
    """Read-out of a (pooled) vector along a concept axis: standardize(vec)·direction."""
    d, m, s = _axis_parts(axis)
    return float(_std(np.atleast_2d(vec), m, s) @ d)


def pool_slide(vlist, axis, tau=1.0, group_key="slide", chunk=100_000):
    """One question-conditioned embedding per slide (+ the mean-pool control), by the axis
    used as the attention query. Returns {slide: {pooled, mean, n}}."""
    d, m, s = _axis_parts(axis)
    groups = np.asarray(vlist.meta[group_key]).astype(str)
    out = {}
    for g in sorted(set(groups)):
        sub = _Subset(vlist.vectors, np.where(groups == g)[0])
        out[g] = {"pooled": directional_pool(sub, d, m, s, tau=tau, chunk=chunk),
                  "mean": mean_pool(sub, chunk=chunk), "n": len(sub)}
    return out


def concept_profile(vec, axes):
    """The pooled vector's read-out across a set of named concept axes (its linear profile)."""
    return {name: concept_score(vec, ax) for name, ax in axes.items()}


# --------------------------------------------------------------------------- #
# Demo / MVP — pool the WSI GLOBAL list per slide by the CRC tumor axis, vs mean-pool
# --------------------------------------------------------------------------- #
def _main():
    import argparse
    import boto3

    from ..causal import probe as _probe
    from ..causal.rank import fit_certified_axis
    from ..data import loader
    from . import load_global

    ap = argparse.ArgumentParser(description="direction-as-query pooling demo on the GLOBAL list")
    ap.add_argument("--bucket", default="bucketbiolayer")
    ap.add_argument("--list-key", default="embeddings/lists/global.npz")
    ap.add_argument("--ref-model", default="h_optimus_0")
    ap.add_argument("--split", default="train")
    ap.add_argument("--pos", default="TUM")
    ap.add_argument("--neg", default="LYM")
    ap.add_argument("--taus", default="0.5,1.0,4.0,1e9")   # 1e9 ~ mean-pool sanity
    ap.add_argument("--region", default="us-west-2")
    args = ap.parse_args()

    s3 = boto3.client("s3", region_name=args.region)
    gl = load_global(s3.get_object(Bucket=args.bucket, Key=args.list_key)["Body"].read())
    feats, labels, cn, _ = loader.load(args.ref_model, args.split)
    feats, labels, cn = np.asarray(feats), np.asarray(labels), list(cn)

    # query axis + a small profile basis (each its own probe -> own scaler)
    axis = fit_certified_axis(feats, labels, cn, args.pos, args.neg)
    profile_axes = {c: _probe.fit_probe(feats, (labels == cn.index(c)).astype(int))
                    for c in (args.pos, args.neg, "STR") if c in cn}

    print(f"query axis = {args.pos}-vs-{args.neg} (certified={axis.certified}, "
          f"intensity|r|={axis.intensity_collinearity:.2f})")
    for tau in [float(t) for t in args.taus.split(",")]:
        res = pool_slide(gl, axis, tau=tau)
        print(f"\n--- tau={tau:g} ---")
        for slide, r in res.items():
            ps, ms = concept_score(r["pooled"], axis), concept_score(r["mean"], axis)
            prof = concept_profile(r["pooled"], profile_axes)
            print(f"  {slide[:32]:32s} n={r['n']:5d} | {args.pos}-score pooled {ps:+.3f} "
                  f"vs mean {ms:+.3f} (Δ {ps - ms:+.3f}) | profile "
                  + " ".join(f"{k}={v:+.2f}" for k, v in prof.items()))


if __name__ == "__main__":
    _main()
