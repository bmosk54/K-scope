"""Confound gate — the differentiator. [NEEDS MULTI-SITE DATA]

The one question K-Pro provably cannot answer: is a prediction real biology or a
batch/scanner/site artifact? Kömen et al. 2024 showed 9 pathology FMs retain
linearly-recoverable tissue-source-site signatures (>90% site accuracy; scanner-ID
~1.000 for Phikon-v2). So we run the SAME subspace machinery used for the concept
against a *site* probe: if the causal axis the model uses for TUM-vs-LYM is aligned
with a site/scanner axis, the prediction is confounded.

The math here is real and ready; what's missing is multi-site data. NCT-CRC-HE is
single-source and Macenko-normalized, so there is no site variation to probe. Track
#2 sources >=2-site TCGA H&E (or the Kömen setup) and passes `site_labels` in.

Interface:
    result = confound_gate(feats, labels, class_names, site_labels,
                           pos="TUM", neg="LYM")
    # site_labels=None -> {"status": "no_multisite_data", ...}  (current state)
    # site_labels given -> real alignment score + FLAG/PASS verdict
"""
import numpy as np

STATUS_NO_DATA = "no_multisite_data"

# |cos(concept axis, site axis)| above this -> the causal driver overlaps a site
# signature -> flag the prediction as potentially confounded. Tune on TCGA.
CONFOUND_COS_THRESHOLD = 0.30


def site_alignment(concept_dir, feats, site_labels, seed=0):
    """Fit a linear site probe and measure its alignment with the concept axis.

    concept_dir : unit concept direction (raw CLS space)
    feats       : (N, dim) frozen CLS features
    site_labels : (N,) integer site/scanner id per tile
    Returns {"site_acc", "cos_with_concept", "n_sites"}.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    site_labels = np.asarray(site_labels)
    sites = np.unique(site_labels)
    scaler = StandardScaler().fit(feats)
    Xs = scaler.transform(feats)

    # One-vs-rest site probe; take the dominant site axis for the alignment read.
    clf = LogisticRegression(max_iter=2000, multi_class="ovr",
                             random_state=seed).fit(Xs, site_labels)
    site_acc = float(clf.score(Xs, site_labels))
    # Strongest site direction (raw space), unit-normalized.
    w = clf.coef_[np.argmax(np.linalg.norm(clf.coef_, axis=1))]
    site_dir_raw = w / scaler.scale_
    site_dir_raw = site_dir_raw / (np.linalg.norm(site_dir_raw) + 1e-12)
    cos = float(abs(concept_dir @ site_dir_raw))
    return {"site_acc": site_acc, "cos_with_concept": cos, "n_sites": int(len(sites))}


def confound_gate(feats, labels, class_names, site_labels=None,
                  pos="TUM", neg="LYM", seed=0):
    """Run the confound gate on the concept axis.

    With no site_labels (current NCT-CRC single-source reality) this returns a
    structured 'unavailable' verdict rather than a false PASS — the honest state
    until track #2 lands multi-site data.
    """
    if site_labels is None:
        return {
            "status": STATUS_NO_DATA,
            "verdict": "confound gate UNAVAILABLE — single-source data",
            "note": ("NCT-CRC-HE is one Macenko-normalized cohort with no site "
                     "variation. Needs >=2-site TCGA H&E or the Komen 2024 setup "
                     "(track #2) to run the site-probe alignment."),
            "threshold_cos": CONFOUND_COS_THRESHOLD,
        }

    from . import probe as _probe
    X, y = _probe.select_pair(feats, labels, class_names, pos, neg)
    concept_dir = _probe.diff_of_means(X, y)  # raw-space unit concept axis
    align = site_alignment(concept_dir, feats, site_labels, seed=seed)
    flagged = align["cos_with_concept"] >= CONFOUND_COS_THRESHOLD
    return {
        "status": "ok",
        "site_probe_acc": align["site_acc"],
        "n_sites": align["n_sites"],
        "cos_concept_with_site": align["cos_with_concept"],
        "threshold_cos": CONFOUND_COS_THRESHOLD,
        "confounded": bool(flagged),
        "verdict": ("FLAG: causal axis overlaps a site/scanner signature "
                    "(possible batch artifact)" if flagged else
                    "PASS: causal axis is not aligned with the site signature"),
    }
