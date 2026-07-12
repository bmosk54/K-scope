"""Train a sparse autoencoder on pathology-FM tile embeddings.

The SAE must live in the SAME space as the causal battery's probe directions --
same model, same layer, same representation (global CLS vs local mean-patch) --
or the feature/probe cosine alignment in the `hypothesis` endpoint is meaningless.
Hence --layer and --rep are explicit and recorded in the checkpoint.

Design notes (these matter; the failure mode is silent, not a crash):
  * Decoder columns are held to unit norm. Without this the model games the L1
    penalty by shrinking feature directions and inflating codes -- you get
    "sparsity" that means nothing.
  * Encoder is tied-initialised to the decoder transpose, which is what stops
    most features from dying in the first few hundred steps.
  * A pre-encoder bias is subtracted from the input (Anthropic's b_dec trick):
    the SAE reconstructs x - b_dec, so features encode deviation from the mean
    activation rather than wasting capacity on the mean itself.
  * We log L0 (mean active features/tile) and the dead-feature count every epoch.
    On a small corpus an SAE does not fail loudly -- it quietly collapses into
    mostly-dead features, and every downstream "novel feature" is then noise.
"""

import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class TileSAE(nn.Module):
    def __init__(self, d_model: int, n_features: int):
        super().__init__()
        self.d_model = d_model
        self.n_features = n_features
        self.b_dec = nn.Parameter(torch.zeros(d_model))
        self.enc = nn.Linear(d_model, n_features, bias=True)
        self.dec = nn.Linear(n_features, d_model, bias=False)

        # unit-norm decoder columns, encoder tied to its transpose
        with torch.no_grad():
            w = torch.randn(d_model, n_features)
            w /= w.norm(dim=0, keepdim=True)
            self.dec.weight.copy_(w)
            self.enc.weight.copy_(w.T)
            self.enc.bias.zero_()

    def forward(self, x):
        z = F.relu(self.enc(x - self.b_dec))
        x_hat = self.dec(z) + self.b_dec
        return x_hat, z

    @torch.no_grad()
    def renorm_decoder(self):
        self.dec.weight.div_(self.dec.weight.norm(dim=0, keepdim=True) + 1e-8)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats", default="/home/sagemaker-user/biolayer/artifacts/phikon_100k.npz")
    ap.add_argument("--out", default="/home/sagemaker-user/biolayer/artifacts/sae_phikon_L24_global.pt")
    ap.add_argument("--layer", type=int, default=24, help="probed depth; must match Eddie's probes")
    ap.add_argument("--rep", choices=["global", "local"], default="global")
    ap.add_argument("--expansion", type=int, default=4)
    ap.add_argument("--l1", type=float, default=5e-3)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--subsample", type=int, default=0, help="train on only N tiles (corpus-size study)")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    z = np.load(args.feats)
    layers = z["layers"].tolist()
    if args.layer not in layers:
        raise SystemExit(f"layer {args.layer} not extracted; have {layers}")
    li = layers.index(args.layer)

    key = "globals" if args.rep == "global" else "locals"
    X = torch.from_numpy(z[key][:, li].astype(np.float32))
    labels = torch.from_numpy(z["labels"])
    class_names = z["class_names"].tolist()
    d_model = X.shape[1]
    n_features = d_model * args.expansion

    # Center and scale to unit mean-norm. Scaling makes the L1 coefficient mean
    # roughly the same thing across layers (raw norms differ ~2.5x between L8 and L24),
    # so a lambda tuned on one layer transfers to another.
    mu = X.mean(0, keepdim=True)
    scale = (X - mu).norm(dim=1).mean()
    Xn = (X - mu) / scale

    g = torch.Generator().manual_seed(0)
    perm = torch.randperm(len(Xn), generator=g)
    n_val = int(len(Xn) * args.val_frac)
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    if args.subsample:
        # Validation set stays fixed so the comparison across corpus sizes is honest.
        tr_idx = tr_idx[: args.subsample]
    Xtr, Xval = Xn[tr_idx].to(dev), Xn[val_idx].to(dev)

    print(
        f"layer={args.layer} rep={args.rep} d_model={d_model} n_features={n_features} "
        f"train={len(Xtr)} val={len(Xval)} l1={args.l1}",
        flush=True,
    )

    sae = TileSAE(d_model, n_features).to(dev)
    opt = torch.optim.Adam(sae.parameters(), lr=args.lr)

    for ep in range(1, args.epochs + 1):
        sae.train()
        idx = torch.randperm(len(Xtr), device=dev)
        for b in range(0, len(Xtr), args.batch_size):
            xb = Xtr[idx[b : b + args.batch_size]]
            x_hat, zc = sae(xb)
            # Both terms must be per-EXAMPLE sums. Using an element-mean MSE against a
            # feature-sum L1 divides reconstruction by d_model but not sparsity, so the
            # L1 outweighs it ~1000x and the optimiser just zeroes every feature: FVU->1,
            # L0->0, everything dead. Silent, and fatal.
            recon = (x_hat - xb).pow(2).sum(-1).mean()
            sparse = zc.abs().sum(-1).mean()
            (recon + args.l1 * sparse).backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            sae.renorm_decoder()

        if ep % 10 == 0 or ep == args.epochs:
            sae.eval()
            with torch.no_grad():
                xh, zc = sae(Xval)
                # fraction of variance unexplained: 0 = perfect, 1 = no better than the mean
                fvu = ((Xval - xh).pow(2).sum() / (Xval - Xval.mean(0)).pow(2).sum()).item()
                l0 = (zc > 0).float().sum(1).mean().item()
                _, ztr = sae(Xtr)
                dead = int((ztr > 0).sum(0).eq(0).sum().item())
            print(
                f"  ep{ep:3d}  fvu={fvu:.4f}  L0={l0:6.1f}/{n_features}  "
                f"dead={dead} ({100*dead/n_features:.1f}%)",
                flush=True,
            )

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save(
        {
            "state_dict": sae.state_dict(),
            "d_model": d_model,
            "n_features": n_features,
            "layer": args.layer,
            "rep": args.rep,
            "mu": mu,
            "scale": scale,
            "l1": args.l1,
            "class_names": class_names,
            "n_train": len(Xtr),
            "feats_path": args.feats,
        },
        args.out,
    )
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
