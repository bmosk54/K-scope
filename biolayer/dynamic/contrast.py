"""Contrast-set assembly + validation — the single biggest risk in the design.

A dynamic probe is only ever as good as its positive/negative pools: a lazy
contrast produces a high-AUROC probe that separates staining intensity or patch
brightness, not the claimed biology. So every contrast is VALIDATED before the
battery runs on it, and the validation numbers ride on the certificate.

Today the pools come from the substrate's labeled corpus (NCT-CRC classes) via
the cached embeddings. The stretch path — agent-curated pools or
H0-mini/CytoSyn-conditioned counterfactuals — plugs in behind the same
`ContrastSet` interface without touching downstream code.
"""
from dataclasses import dataclass

import numpy as np

from ..causal import probe as _probe
from ..data import loader


@dataclass
class ContrastSet:
    pos: str
    neg: str
    distractor: tuple
    n_pos: int
    n_neg: int
    # validation (all ride on the certificate)
    heldout_auroc: float          # probe generalizes off the fit split
    intensity_collinearity: float # |corr(concept proj, CLS L2 norm)| — nuisance SCREEN
    valid: bool
    warnings: tuple
    source: str
    # controlled adjudication of the intensity screen (Gate 2b) — all ride on the card
    intensity_suspect: bool = False        # did the cheap screen fire?
    matched_auroc: float = float("nan")    # held-out AUROC on intensity-matched pools
    matched_intensity_collinearity: float = float("nan")
    n_matched: int = 0
    confound_adjudication: str = "screen-clean"  # screen-clean|survives-control|confound-real|undecidable
    flags: tuple = ()             # non-invalidating annotations (admitted-with-flag)


# A probe that separates the pools worse than this on held-out data isn't a real
# concept axis. A concept projection that tracks the intensity proxy above the
# second threshold is a *suspect* — not a verdict: we then CONTROL for intensity and
# re-measure (Gate 2b) rather than veto on the correlation alone.
MIN_HELDOUT_AUROC = 0.75
MAX_INTENSITY_COLLINEARITY = 0.60
# Below this many intensity-matched samples the controlled re-test is unreliable, so an
# unresolved suspicion is declined (not a clean bill of health).
MIN_MATCHED = 40
_MATCH_BINS = 12


def _intensity_collinearity(X, y):
    """|corr| between the concept projection and a per-tile intensity proxy.

    Proxy = CLS L2 norm (a coarse stand-in for staining/brightness; the real guard
    needs pixel-space patch-mean, a TODO for the live-input path). High collinearity
    flags that the 'concept' axis may just be an intensity axis.
    """
    d = _probe.diff_of_means(X, y)          # raw-space unit concept axis
    proj = X @ d
    intensity = np.linalg.norm(X, axis=1)
    if proj.std() < 1e-9 or intensity.std() < 1e-9:
        return 0.0
    return float(abs(np.corrcoef(proj, intensity)[0, 1]))


def _heldout_auroc(X, y, seed=0):
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.4, stratify=y,
                                           random_state=seed)
    fit = _probe.fit_probe(Xtr, ytr, seed=seed)
    scores = fit["clf"].decision_function(fit["scaler"].transform(Xte))
    try:
        return float(roc_auc_score(yte, scores))
    except ValueError:
        return float("nan")


def _intensity_match(X, y, seed=0, n_bins=_MATCH_BINS):
    """Subsample pos/neg to a SHARED intensity (CLS-norm) distribution so the nuisance
    carries no information about the label — the `do()` on the confound. Bin by norm
    quantiles; in each bin keep min(n_pos, n_neg) from each side. Returns matched (X, y)."""
    norms = np.linalg.norm(X, axis=1)
    edges = np.quantile(norms, np.linspace(0, 1, n_bins + 1))
    edges[-1] += 1e-6
    rng = np.random.default_rng(seed)
    keep = []
    for i in range(n_bins):
        b = np.where((norms >= edges[i]) & (norms < edges[i + 1]))[0]
        pos, neg = b[y[b] == 1], b[y[b] == 0]
        k = min(len(pos), len(neg))
        if k:
            keep += list(rng.choice(pos, k, replace=False))
            keep += list(rng.choice(neg, k, replace=False))
    keep = np.array(keep, dtype=int)
    return X[keep], y[keep]


def _adjudicate_intensity(X, y, coll, seed=0):
    """Gate 2b: the intensity SCREEN fired (coll high). Don't veto on correlation —
    control for intensity and re-measure. Returns
    (adjudication, matched_auroc, matched_coll, n_matched, warnings, flags).

    survives-control : matched separation holds -> admit WITH A FLAG (over-cautious veto)
    confound-real    : separation collapses when intensity is balanced -> invalidate
    undecidable      : too few matched samples to adjudicate -> decline (not a clean bill)
    """
    Xm, ym = _intensity_match(X, y, seed=seed)
    n_matched = int(len(ym))
    if n_matched < MIN_MATCHED or len(np.unique(ym)) < 2:
        return ("undecidable", float("nan"), float("nan"), n_matched,
                (f"intensity-suspect (|r|={coll:.2f}); only {n_matched} intensity-matched "
                 f"samples (<{MIN_MATCHED}) — cannot adjudicate, declined",), ())
    m_auroc = _heldout_auroc(Xm, ym, seed=seed)
    m_coll = _intensity_collinearity(Xm, ym)
    if m_auroc >= MIN_HELDOUT_AUROC:
        return ("survives-control", m_auroc, m_coll, n_matched, (),
                (f"intensity-suspect (screen |r|={coll:.2f}) but signal SURVIVES control: "
                 f"intensity-matched AUROC={m_auroc:.3f} (n={n_matched}, matched |r|={m_coll:.2f}) "
                 f"— admitted with flag",))
    return ("confound-real", m_auroc, m_coll, n_matched,
            (f"intensity confound REAL: matched AUROC {m_auroc:.3f} < {MIN_HELDOUT_AUROC} — "
             f"separation collapses when intensity is balanced (screen |r|={coll:.2f})",), ())


def assemble(claim, split="train", artifacts_dir=None, seed=0):
    """Build + validate the contrast pools for one claim from cached embeddings.

    Loads from the substrate + label source the claim RESOLVED to (tissue on
    Phikon/H-optimus, cell types on H0-mini), not a fixed track. Returns
    (ContrastSet, feats, labels, class_names, source) so the caller can run the
    existing battery on the full arrays (which re-select the pos/neg pool).
    """
    spec = claim.spec
    kw = {"dataset_slug": claim.dataset_slug}
    if artifacts_dir is not None:
        kw["artifacts_dir"] = artifacts_dir
    feats, labels, class_names, source = loader.load(claim.model_key, split, **kw)

    X, y = _probe.select_pair(feats, labels, class_names, spec.pos, spec.neg)
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())

    warnings, flags = [], []
    # Gate 1 — is this a real axis at all? (thin pool / not separable)
    if min(n_pos, n_neg) < 20:
        warnings.append(f"thin pool (pos={n_pos}, neg={n_neg})")
    auroc = _heldout_auroc(X, y, seed=seed)
    if not (auroc >= MIN_HELDOUT_AUROC):
        warnings.append(f"held-out AUROC {auroc:.3f} < {MIN_HELDOUT_AUROC}")

    # Gate 2 — cheap correlational SCREEN against the intensity nuisance.
    coll = _intensity_collinearity(X, y)
    suspect = coll > MAX_INTENSITY_COLLINEARITY
    adjudication, m_auroc, m_coll, n_matched = "screen-clean", float("nan"), float("nan"), 0
    if suspect:
        # Gate 2b — CONTROL for intensity and re-measure instead of vetoing on the
        # correlation. "Does the separation survive when the nuisance is removed?"
        (adjudication, m_auroc, m_coll, n_matched,
         adj_warnings, adj_flags) = _adjudicate_intensity(X, y, coll, seed=seed)
        warnings.extend(adj_warnings)
        flags.extend(adj_flags)

    cs = ContrastSet(
        pos=spec.pos, neg=spec.neg, distractor=spec.distractor,
        n_pos=n_pos, n_neg=n_neg, heldout_auroc=auroc,
        intensity_collinearity=coll,
        valid=(len(warnings) == 0), warnings=tuple(warnings), source=source,
        intensity_suspect=suspect, matched_auroc=m_auroc,
        matched_intensity_collinearity=m_coll, n_matched=n_matched,
        confound_adjudication=adjudication, flags=tuple(flags))
    return cs, feats, labels, class_names, source
