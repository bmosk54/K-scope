"""Base causal battery in readout (CLS) space, with matched-random nulls.

This runs the fast, hook-free subset of the Bio-Interp battery — the checks that
only need frozen CLS features — to validate the RESULTS.md priors on Phikon-v2:

  probe        TUM vs LYM linearly separable on frozen CLS               (baseline)
  necessity    project the concept axis out of CLS -> probe collapses    (readout)
               matched-random axis removed -> probe intact               (null)
  sufficiency  add concept axis to LYM CLS -> flips to TUM               (steering)
               add matched-random axis -> ~0 flips                       (null)
  specificity  ablate STR/MUS distractor axis -> TUM/LYM probe intact

Layer-resolved source-intervention (hooking encoder.layer[L]) — the mid-layer
redundancy / "rigor" curve — is the next module (intervene.py); it is NOT run
here because these readout-space checks are what the CLS features alone certify.
Emits a structured evidence-card JSON to artifacts/certificates/.
"""
import argparse
import json
import os

import numpy as np
from sklearn.model_selection import train_test_split

from .. import config
from . import probe


def _proj_out(X, u):
    """Remove the component of each row along unit vector u."""
    return X - np.outer(X @ u, u)


def _decision(fit, Xs_raw):
    """Probe decision_function on raw (unstandardized) features."""
    return fit["clf"].decision_function(fit["scaler"].transform(Xs_raw))


def run_battery(feats, labels, class_names, pos="TUM", neg="LYM",
                distractor=("STR", "MUS"), n_null=200, seed=0):
    rng = np.random.default_rng(seed)
    card = {"substrate": {"pos": pos, "neg": neg, "distractor": list(distractor),
                          "dim": int(feats.shape[1]), "n_null": n_null}}

    # ---- concept pair: fit probe on train half, test on held-out half -------
    X, y = probe.select_pair(feats, labels, class_names, pos, neg)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.4,
                                          stratify=y, random_state=seed)
    fit = probe.fit_probe(Xtr, ytr, seed=seed)
    # Work in the probe's standardized space so directions are comparable.
    Ztr = fit["scaler"].transform(Xtr)
    Zte = fit["scaler"].transform(Xte)
    u = fit["direction"]  # unit concept axis in standardized space
    base_acc = fit["clf"].score(Zte, yte)
    card["probe"] = {"train_acc": fit["acc"], "test_acc": float(base_acc),
                     "n_train": int(len(ytr)), "n_test": int(len(yte))}

    # ---- necessity (readout): project concept axis out of CLS ---------------
    Zte_abl = _proj_out(Zte, u)
    acc_concept = fit["clf"].score(Zte_abl, yte)
    R = probe.matched_random_dirs(Zte.shape[1], n_null, seed=seed)
    null_accs = np.array([fit["clf"].score(_proj_out(Zte, r), yte) for r in R])
    card["necessity_readout"] = {
        "base_acc": float(base_acc),
        "concept_ablated_acc": float(acc_concept),
        "random_ablated_acc_mean": float(null_accs.mean()),
        "random_ablated_acc_std": float(null_accs.std()),
        # z-score: how many nulls-sigmas below the random baseline the concept sits
        "concept_vs_null_z": float((null_accs.mean() - acc_concept)
                                   / (null_accs.std() + 1e-9)),
        "verdict": ("concept axis carries the readout; random axes do not"
                    if acc_concept < null_accs.mean() - 3 * null_accs.std()
                    else "inconclusive at readout"),
    }

    # ---- sufficiency (steering): push LYM samples along concept axis ---------
    # step size = inter-class projection gap along u (a "one class-width" nudge)
    proj = Ztr @ u
    gap = float(proj[ytr == 1].mean() - proj[ytr == 0].mean())
    neg_te = Zte[yte == 0]
    def flip_rate(direction, alpha):
        moved = neg_te + alpha * direction
        # flipped = probe now calls them pos
        return float((fit["clf"].predict(moved) == 1).mean())
    concept_flip = flip_rate(u, gap)
    null_flips = np.array([flip_rate(r, gap) for r in R])
    card["sufficiency_steering"] = {
        "alpha_classwidth": gap,
        "concept_flip_rate": concept_flip,
        "random_flip_rate_mean": float(null_flips.mean()),
        "random_flip_rate_std": float(null_flips.std()),
        "verdict": ("concept axis is a sufficient, specific steering direction"
                    if concept_flip > 0.5 and null_flips.mean() < 0.1
                    else "steering not clean at this step size"),
    }

    # ---- specificity: ablate an orthogonal distractor axis -------------------
    dpos, dneg = distractor
    try:
        Xd, yd = probe.select_pair(feats, labels, class_names, dpos, dneg)
        dfit = probe.fit_probe(Xd, yd, seed=seed)
        # express distractor axis in the concept probe's standardized space
        dir_raw = dfit["scaler"].scale_ * dfit["direction"]  # -> raw space
        dir_std = dir_raw / fit["scaler"].scale_             # -> concept std space
        dir_std = dir_std / (np.linalg.norm(dir_std) + 1e-12)
        acc_distractor_abl = fit["clf"].score(_proj_out(Zte, dir_std), yte)
        cos = float(abs(u @ dir_std))
        card["specificity"] = {
            "distractor": f"{dpos}_vs_{dneg}",
            "cos_with_concept_axis": cos,
            "target_acc_after_distractor_ablation": float(acc_distractor_abl),
            "base_acc": float(base_acc),
            "verdict": ("target probe intact after distractor ablation"
                        if acc_distractor_abl > base_acc - 0.05
                        else "distractor ablation leaked into target"),
        }
    except ValueError as e:
        card["specificity"] = {"skipped": str(e)}

    return card


def _pretty(card):
    p = card["probe"]; n = card["necessity_readout"]
    s = card["sufficiency_steering"]; sp = card.get("specificity", {})
    print("\n================ CAUSAL BATTERY (readout space) ================")
    print(f"concept: {card['substrate']['pos']} vs {card['substrate']['neg']}"
          f"   dim={card['substrate']['dim']}   nulls={card['substrate']['n_null']}")
    print(f"[probe]        test_acc={p['test_acc']:.3f}  (n_test={p['n_test']})")
    print(f"[necessity]    base={n['base_acc']:.3f}  concept-ablated={n['concept_ablated_acc']:.3f}"
          f"  random-ablated={n['random_ablated_acc_mean']:.3f}±{n['random_ablated_acc_std']:.3f}"
          f"  (z={n['concept_vs_null_z']:.1f})")
    print(f"               -> {n['verdict']}")
    print(f"[sufficiency]  concept-flip={s['concept_flip_rate']:.3f}"
          f"  random-flip={s['random_flip_rate_mean']:.3f}±{s['random_flip_rate_std']:.3f}"
          f"  (alpha={s['alpha_classwidth']:.2f})")
    print(f"               -> {s['verdict']}")
    if "target_acc_after_distractor_ablation" in sp:
        print(f"[specificity]  {sp['distractor']}  cos={sp['cos_with_concept_axis']:.3f}"
              f"  target_acc={sp['target_acc_after_distractor_ablation']:.3f} (base {sp['base_acc']:.3f})")
        print(f"               -> {sp['verdict']}")
    print("================================================================\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="phikon_v2", choices=list(config.MODELS))
    p.add_argument("--split", default="train")
    p.add_argument("--npz", default=None,
                   help="path to embeddings .npz (defaults to local artifacts path)")
    p.add_argument("--pos", default="TUM")
    p.add_argument("--neg", default="LYM")
    p.add_argument("--n-null", type=int, default=200)
    p.add_argument("--out-dir", default="artifacts")
    args = p.parse_args()

    npz_path = args.npz or os.path.join(
        args.out_dir, config.embeddings_key(args.model, args.split))
    d = np.load(npz_path, allow_pickle=True)
    feats, labels = d["feats"], d["labels"]
    class_names = list(d["class_names"])
    print(f"[battery] loaded {feats.shape} from {npz_path}")

    card = run_battery(feats, labels, class_names, pos=args.pos, neg=args.neg,
                       n_null=args.n_null)
    card["_meta"] = {"model": args.model, "split": args.split,
                     "npz": npz_path}
    _pretty(card)

    out = os.path.join(args.out_dir,
                       config.certificate_key(args.model, args.split, args.pos, args.neg))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(card, f, indent=2)
    print(f"[battery] wrote evidence card -> {out}")


if __name__ == "__main__":
    main()
