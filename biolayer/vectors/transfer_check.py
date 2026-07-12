"""Confound sanity check for a direction applied ACROSS groups (slide / site / batch).

The question a ranking must survive before you trust it across sources: does the concept
direction separate two groups (e.g. cancer slide vs normal slide) *more than a random
direction would*? If a matched-random null separates the groups just as well, the "signal"
is a group/batch confound — different scanner, stain, institution — not the concept. This is
the STRATEGY.md failure mode (naive single-axis transfer on a pathology FM is unsafe), made
falsifiable: an axis is only credible if its group-separation |AUROC| clears the random null.

    from biolayer.vectors import load_global
    from biolayer.vectors.transfer_check import confound_check, axis_from_certified
    from biolayer.causal.rank import fit_certified_axis
    gl = load_global(open("global.npz","rb").read())
    y  = group_mask(gl, "TCGA")                       # 1 = cancer slide, 0 = the other
    ax = fit_certified_axis(ref_feats, ref_labels, cn, "TUM", "LYM")
    rep = confound_check(gl.vectors, y, {"TUM_vs_LYM": axis_from_certified(ax)})
    rep["axes"]["TUM_vs_LYM"]["survives_null"]         # False -> separation is a batch confound

Run as a CLI to reproduce the TCGA-BRCA vs BRACS check on the shipped GLOBAL list.
"""
import numpy as np
from sklearn.metrics import roc_auc_score


# --------------------------------------------------------------------------- #
# Building blocks
# --------------------------------------------------------------------------- #
def _std(V, mean, scale):
    return (np.asarray(V, dtype=np.float64) - np.asarray(mean)) / np.asarray(scale)


def direction_scores(V, direction, mean, scale):
    """Signed distance of each row along a standardized concept direction."""
    return _std(V, mean, scale) @ np.asarray(direction, dtype=np.float64)


def separation(score, y):
    """How well `score` separates group y=1 from y=0 (AUROC + Cohen's d + group stats)."""
    y = np.asarray(y).astype(int)
    a, b = score[y == 1], score[y == 0]
    d = (a.mean() - b.mean()) / np.sqrt((a.var() + b.var()) / 2 + 1e-12)
    return {"auroc": float(roc_auc_score(y, score)), "cohen_d": float(d),
            "pos_mean": float(a.mean()), "neg_mean": float(b.mean()),
            "pos_median": float(np.median(a)), "neg_median": float(np.median(b))}


def matched_random_null(V, y, mean, scale, n=200, seed=0):
    """Group-separation |AUROC| of `n` random unit directions (sign-agnostic). The band a
    real concept axis must beat — its width IS the size of the group/batch confound."""
    rng = np.random.default_rng(seed)
    Z = _std(V, mean, scale)
    y = np.asarray(y).astype(int)
    D = Z.shape[1]
    out = np.empty(n)
    for i in range(n):
        r = rng.standard_normal(D)
        r /= np.linalg.norm(r)
        a = roc_auc_score(y, Z @ r)
        out[i] = max(a, 1.0 - a)                       # direction sign is arbitrary
    return out


# --------------------------------------------------------------------------- #
# Axis adapters — build the {direction, mean, scale} an axis needs from either source
# --------------------------------------------------------------------------- #
def axis_from_certified(ax):
    """From a causal.rank.CertifiedAxis."""
    return {"direction": np.asarray(ax.direction), "mean": np.asarray(ax.scaler_mean),
            "scale": np.asarray(ax.scaler_scale), "certified": bool(ax.certified)}


def axis_from_probe(fit):
    """From a causal.probe.fit_probe dict (e.g. a one-vs-rest cancer axis)."""
    return {"direction": np.asarray(fit["direction"]), "mean": np.asarray(fit["scaler"].mean_),
            "scale": np.asarray(fit["scaler"].scale_), "certified": None}


def group_mask(vlist, prefix, meta_key="slide"):
    """Boolean group label from a list's row metadata (y=1 where meta startswith prefix)."""
    return np.char.startswith(np.asarray(vlist.meta[meta_key]).astype(str), prefix)


# --------------------------------------------------------------------------- #
# The check
# --------------------------------------------------------------------------- #
def confound_check(V, y, axes, n_null=200, seed=0, topk=(100, 1000)):
    """For each named axis: its group-separation vs a matched-random null.

    `axes`: {name -> {"direction","mean","scale"}}. Returns a report dict; `survives_null`
    is True iff the axis's folded |AUROC| exceeds the null's 95th percentile — i.e. it
    separates the groups by MORE than a batch confound would.
    """
    V = np.asarray(V, dtype=np.float64)
    y = np.asarray(y).astype(int)
    ref = next(iter(axes.values()))                    # standardize the null with the 1st axis
    null = matched_random_null(V, y, ref["mean"], ref["scale"], n=n_null, seed=seed)
    p95 = float(np.quantile(null, 0.95))
    report = {"n": int(len(y)), "n_pos": int(y.sum()), "n_neg": int((1 - y).sum()),
              "baseline_pos_frac": float(y.mean()),
              "null": {"mean": float(null.mean()), "p95": p95, "max": float(null.max())},
              "axes": {}}
    for name, ax in axes.items():
        s = direction_scores(V, ax["direction"], ax["mean"], ax["scale"])
        sep = separation(s, y)
        folded = max(sep["auroc"], 1.0 - sep["auroc"])
        order = np.argsort(-s)
        prov = {f"top{K}_pos_frac": float(y[order[:K]].mean()) for K in topk}
        report["axes"][name] = {**sep, "folded_auroc": folded,
                                "beats_random_frac": float((null < folded).mean()),
                                "survives_null": bool(folded > p95), **prov}
    return report


def format_report(rep, title=""):
    """Human-readable one-block summary."""
    L = [f"=== confound / transfer check {title} ===",
         f"groups: pos={rep['n_pos']} neg={rep['n_neg']} (baseline pos={rep['baseline_pos_frac']:.1%})",
         f"matched-random null |AUROC|: mean {rep['null']['mean']:.3f} "
         f"95th {rep['null']['p95']:.3f} max {rep['null']['max']:.3f}"]
    for name, a in rep["axes"].items():
        verdict = "SURVIVES null" if a["survives_null"] else "CONFOUNDED (within null band)"
        L.append(f"[{name}] AUROC {a['auroc']:.3f} (|{a['folded_auroc']:.3f}|, d={a['cohen_d']:.2f}) "
                 f"beats {a['beats_random_frac']:.0%} of random -> {verdict}; "
                 f"top100 pos={a['top100_pos_frac']:.0%}")
    return "\n".join(L)


# --------------------------------------------------------------------------- #
# CLI — reproduce the TCGA-BRCA(cancer) vs BRACS(normal) check on the GLOBAL list
# --------------------------------------------------------------------------- #
def _main():
    import argparse
    import boto3

    from ..causal import probe as _probe
    from ..causal.rank import fit_certified_axis
    from ..data import loader
    from . import load_global

    ap = argparse.ArgumentParser(description="confound/transfer check on a GLOBAL list")
    ap.add_argument("--bucket", default="bucketbiolayer")
    ap.add_argument("--list-key", default="embeddings/lists/global.npz")
    ap.add_argument("--pos-prefix", default="TCGA", help="slide prefix that marks group y=1")
    ap.add_argument("--ref-model", default="h_optimus_0")
    ap.add_argument("--split", default="train")
    ap.add_argument("--n-null", type=int, default=200)
    ap.add_argument("--region", default="us-west-2")
    args = ap.parse_args()

    s3 = boto3.client("s3", region_name=args.region)
    gl = load_global(s3.get_object(Bucket=args.bucket, Key=args.list_key)["Body"].read())
    y = group_mask(gl, args.pos_prefix)
    feats, labels, cn, _ = loader.load(args.ref_model, args.split)
    feats, labels, cn = np.asarray(feats), np.asarray(labels), list(cn)

    axes = {
        "TUM_vs_LYM_certified": axis_from_certified(
            fit_certified_axis(feats, labels, cn, "TUM", "LYM")),
        "cancer_vs_noncancer_ovr": axis_from_probe(
            _probe.fit_probe(feats, (labels == cn.index("TUM")).astype(int))),
    }
    rep = confound_check(gl.vectors, y, axes, n_null=args.n_null)
    print(format_report(rep, title=f"({args.pos_prefix} vs rest, ref={args.ref_model})"))


if __name__ == "__main__":
    _main()
