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
    intensity_collinearity: float # |corr(concept proj, CLS L2 norm)| — nuisance guard
    valid: bool
    warnings: tuple
    source: str


# A probe that separates the pools worse than this on held-out data isn't a real
# concept axis. A concept projection that tracks the intensity proxy above the
# second threshold is likely riding staining/brightness, not biology.
MIN_HELDOUT_AUROC = 0.75
MAX_INTENSITY_COLLINEARITY = 0.60


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


def assemble(track, claim, split="train", artifacts_dir=None, seed=0):
    """Build + validate the contrast pools for one claim from cached embeddings.

    Returns (ContrastSet, feats, labels, class_names, source) so the caller can run
    the existing battery on the full arrays (which re-select the pos/neg pool).
    """
    spec = claim.spec
    kw = {} if artifacts_dir is None else {"artifacts_dir": artifacts_dir}
    feats, labels, class_names, source = loader.load(track.model_key, split, **kw)

    X, y = _probe.select_pair(feats, labels, class_names, spec.pos, spec.neg)
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())

    warnings = []
    if min(n_pos, n_neg) < 20:
        warnings.append(f"thin pool (pos={n_pos}, neg={n_neg})")
    auroc = _heldout_auroc(X, y, seed=seed)
    coll = _intensity_collinearity(X, y)
    if not (auroc >= MIN_HELDOUT_AUROC):
        warnings.append(f"held-out AUROC {auroc:.3f} < {MIN_HELDOUT_AUROC}")
    if coll > MAX_INTENSITY_COLLINEARITY:
        warnings.append(f"concept axis collinear with intensity proxy (|r|={coll:.2f})")

    cs = ContrastSet(
        pos=spec.pos, neg=spec.neg, distractor=spec.distractor,
        n_pos=n_pos, n_neg=n_neg, heldout_auroc=auroc,
        intensity_collinearity=coll,
        valid=(len(warnings) == 0), warnings=tuple(warnings), source=source)
    return cs, feats, labels, class_names, source
