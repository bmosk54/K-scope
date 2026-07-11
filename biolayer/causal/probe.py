"""Linear probes and concept directions on frozen CLS features.

Two ways to name a concept axis in CLS space:
  - probe:   logistic-regression decision direction (what the model can read out)
  - diff-of-means: class_a mean - class_b mean (robust, interpretable steering axis)
Both are used by the causal battery; matched-random directions are the null.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .. import config


def _idx(class_names, name):
    return list(class_names).index(name)


def select_pair(feats, labels, class_names, pos, neg):
    """Return (X, y) for a binary pos-vs-neg concept, y=1 for pos."""
    pi, ni = _idx(class_names, pos), _idx(class_names, neg)
    mask = np.isin(labels, [pi, ni])
    X = feats[mask]
    y = (labels[mask] == pi).astype(np.int64)
    return X, y


def fit_probe(X, y, seed=0):
    """Standardize + logistic probe. Returns dict with scaler, clf, direction, acc.

    `direction` is the unit decision axis in the *standardized* feature space.
    """
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    clf = LogisticRegression(max_iter=2000, C=1.0, random_state=seed).fit(Xs, y)
    acc = clf.score(Xs, y)
    w = clf.coef_[0]
    direction = w / (np.linalg.norm(w) + 1e-12)
    return {"scaler": scaler, "clf": clf, "direction": direction,
            "w": w, "b": float(clf.intercept_[0]), "acc": float(acc)}


def diff_of_means(X, y):
    """Unit diff-of-means direction (pos - neg) in raw CLS space."""
    d = X[y == 1].mean(0) - X[y == 0].mean(0)
    return d / (np.linalg.norm(d) + 1e-12)


def matched_random_dirs(dim, n, seed=0):
    """n unit random directions — the Section-5-D matched-random null."""
    rng = np.random.default_rng(seed)
    R = rng.standard_normal((n, dim))
    return R / (np.linalg.norm(R, axis=1, keepdims=True) + 1e-12)
