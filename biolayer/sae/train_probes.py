"""Linear concept probes: one direction per tissue class, in the SAME space as the SAE.

These are the 'known biology' vocabulary. An SAE feature is only NOVEL if it fails to align
with any of them -- so without these directions the novelty test is vacuous (an empty probe
set makes max_alignment 0, and every significant feature gets marked novel, including
blatant lymphocyte detectors).

Eddie owns the canonical probes for `certify`. These are trained the same way, in the same
space (H-Optimus-0, block 39, global CLS, 1536-d), so `hypothesis` is not blocked on his
artifact landing -- and when it does, swap the .npz and the alignment numbers change but
nothing else does.

One-vs-rest logistic regression; the weight vector for class c is the concept direction for
c. Directions are L2-normalised so cosine against SAE decoder columns is well defined.
"""

import argparse

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats", default="/home/sagemaker-user/biolayer/artifacts/hoptimus_100k.npz")
    ap.add_argument("--layer", type=int, default=39)
    ap.add_argument("--rep", choices=["global", "local"], default="global")
    ap.add_argument("--out", default="/home/sagemaker-user/biolayer/artifacts/probes_hoptimus_L39_global.npz")
    ap.add_argument("--n", type=int, default=30000)
    args = ap.parse_args()

    z = np.load(args.feats)
    li = list(z["layers"]).index(args.layer)
    X = z["globals" if args.rep == "global" else "locals"][:, li].astype(np.float32)
    y = z["labels"]
    cn = [str(c) for c in z["class_names"]]

    idx = np.random.default_rng(0).choice(len(X), min(args.n, len(X)), replace=False)
    Xtr, Xte, ytr, yte = train_test_split(X[idx], y[idx], test_size=0.25, stratify=y[idx], random_state=0)

    dirs, accs = [], {}
    for c, name in enumerate(cn):
        clf = LogisticRegression(max_iter=2000, C=0.1).fit(Xtr, (ytr == c).astype(int))
        w = clf.coef_[0]
        dirs.append(w / (np.linalg.norm(w) + 1e-8))
        accs[name] = round(float(clf.score(Xte, (yte == c).astype(int))), 4)

    D = np.stack(dirs)
    # Concept directions should be reasonably distinct; near-duplicate directions would mean
    # the "known vocabulary" is smaller than 9 and novelty is easier to claim than it looks.
    G = D @ D.T
    off = G[~np.eye(9, dtype=bool)]
    print("per-concept one-vs-rest accuracy:", accs)
    print(f"inter-concept |cosine|: mean={np.abs(off).mean():.3f} max={np.abs(off).max():.3f}")

    np.savez(args.out, directions=D, class_names=np.asarray(cn),
             layer=args.layer, rep=args.rep, accuracy=np.asarray([accs[c] for c in cn]))
    print(f"wrote {args.out}  directions={D.shape}")


if __name__ == "__main__":
    main()
