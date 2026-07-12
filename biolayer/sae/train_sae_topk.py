"""TopK sparse autoencoder (Gao et al. 2024) -- the architecture upgrade over ReLU+L1.

Why TopK beats the ReLU+L1 baseline in scripts/train_sae.py:

  * NO SHRINKAGE. L1 penalises activation MAGNITUDE as well as count, so every surviving
    feature is biased toward zero and reconstruction is systematically degraded. TopK has
    no magnitude penalty at all -- sparsity comes from the hard top-k mask, so the k
    surviving activations are unbiased.
  * EXACT SPARSITY CONTROL. L0 is a hyperparameter you SET, not an emergent side effect of
    lambda. In the L1 run, L0 drifted 121 -> 40 over 80 epochs; nobody chose 40.
  * NO LAMBDA SWEEP. The L1 coefficient had to be tuned over 7 runs to find a usable
    reconstruction/sparsity trade-off. TopK needs none of that.

Dead features are the one thing TopK does not fix for free (a feature never in the top-k
gets no gradient, ever), so we add the standard AuxK auxiliary loss: the top-k_aux DEAD
features are asked to reconstruct the residual, which revives them.
"""

import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn


class TopKSAE(nn.Module):
    def __init__(self, d_model: int, n_features: int, k: int):
        super().__init__()
        self.d_model, self.n_features, self.k = d_model, n_features, k
        self.b_dec = nn.Parameter(torch.zeros(d_model))
        self.enc = nn.Linear(d_model, n_features, bias=True)
        self.dec = nn.Linear(n_features, d_model, bias=False)
        with torch.no_grad():
            w = torch.randn(d_model, n_features)
            w /= w.norm(dim=0, keepdim=True)
            self.dec.weight.copy_(w)
            self.enc.weight.copy_(w.T)
            self.enc.bias.zero_()

    def encode_pre(self, x):
        return self.enc(x - self.b_dec)

    def forward(self, x):
        pre = self.encode_pre(x)
        # hard top-k mask: exactly k active features per example, no magnitude penalty
        val, idx = pre.topk(self.k, dim=-1)
        val = val.relu()
        z = torch.zeros_like(pre).scatter_(-1, idx, val)
        return self.dec(z) + self.b_dec, z, pre

    @torch.no_grad()
    def renorm_decoder(self):
        self.dec.weight.div_(self.dec.weight.norm(dim=0, keepdim=True) + 1e-8)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats", default="/home/sagemaker-user/biolayer/artifacts/hoptimus_100k.npz")
    ap.add_argument("--out", default="/home/sagemaker-user/biolayer/artifacts/sae_topk_hoptimus_L39_global.pt")
    ap.add_argument("--layer", type=int, default=39)
    ap.add_argument("--rep", choices=["global", "local"], default="global")
    ap.add_argument("--expansion", type=int, default=4)
    ap.add_argument("--k", type=int, default=40, help="exact L0: active features per tile")
    ap.add_argument("--k-aux", type=int, default=256, help="dead features used by the AuxK revival loss")
    ap.add_argument("--aux-coef", type=float, default=1 / 32)
    ap.add_argument("--dead-after", type=int, default=2, help="epochs unfired before a feature counts as dead")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--subsample", type=int, default=0)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    z = np.load(args.feats)
    layers = z["layers"].tolist()
    li = layers.index(args.layer)
    key = "globals" if args.rep == "global" else "locals"
    X = torch.from_numpy(z[key][:, li].astype(np.float32))
    d_model = X.shape[1]
    n_features = d_model * args.expansion

    mu = X.mean(0, keepdim=True)
    scale = (X - mu).norm(dim=1).mean()
    Xn = (X - mu) / scale

    g = torch.Generator().manual_seed(0)
    perm = torch.randperm(len(Xn), generator=g)
    n_val = int(len(Xn) * args.val_frac)
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    if args.subsample:
        tr_idx = tr_idx[: args.subsample]
    Xtr, Xval = Xn[tr_idx].to(dev), Xn[val_idx].to(dev)

    print(
        f"TopK  layer={args.layer} rep={args.rep} d_model={d_model} n_features={n_features} "
        f"k={args.k} train={len(Xtr)} val={len(Xval)}",
        flush=True,
    )

    sae = TopKSAE(d_model, n_features, args.k).to(dev)
    opt = torch.optim.Adam(sae.parameters(), lr=args.lr)
    # steps since each feature last fired -> identifies dead features for AuxK
    last_fired = torch.zeros(n_features, device=dev)
    t0 = time.time()

    for ep in range(1, args.epochs + 1):
        sae.train()
        idx = torch.randperm(len(Xtr), device=dev)
        for b in range(0, len(Xtr), args.batch_size):
            xb = Xtr[idx[b : b + args.batch_size]]
            x_hat, zc, pre = sae(xb)
            recon = (x_hat - xb).pow(2).sum(-1).mean()

            fired = (zc > 0).any(0)
            last_fired = torch.where(fired, torch.zeros_like(last_fired), last_fired + 1)

            # AuxK: let the DEAD features try to explain the residual, so they get gradient
            # and come back to life. Without this, a feature that falls out of the top-k
            # never receives another update.
            aux = torch.zeros((), device=dev)
            dead_mask = last_fired > args.dead_after * (len(Xtr) / args.batch_size)
            if dead_mask.any():
                n_aux = min(args.k_aux, int(dead_mask.sum()))
                pre_dead = pre.masked_fill(~dead_mask, -torch.inf)
                v, i = pre_dead.topk(n_aux, dim=-1)
                z_aux = torch.zeros_like(pre).scatter_(-1, i, v.relu())
                resid = xb - x_hat.detach()
                aux = (sae.dec(z_aux) - resid).pow(2).sum(-1).mean()

            (recon + args.aux_coef * aux).backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            sae.renorm_decoder()

        if ep % 20 == 0 or ep == args.epochs:
            sae.eval()
            with torch.no_grad():
                # Everything here is CHUNKED. A single sae(X) over a large split allocates an
                # (n, n_features) activation tensor: at 1.44M x 6144 that is 35GB, and even the
                # 160k val split is 3.9GB on top of the resident training data. Both OOM a 23GB card.
                val_mu = Xval.mean(0)
                sse = torch.zeros((), device=dev)
                sst = torch.zeros((), device=dev)
                l0_sum = torch.zeros((), device=dev)
                for i in range(0, len(Xval), 16384):
                    xb = Xval[i : i + 16384]
                    xh, zc, _ = sae(xb)
                    sse += (xb - xh).pow(2).sum()
                    sst += (xb - val_mu).pow(2).sum()
                    l0_sum += (zc > 0).float().sum()
                fvu = (sse / sst).item()
                l0 = (l0_sum / len(Xval)).item()

                fired = torch.zeros(n_features, dtype=torch.bool, device=dev)
                for i in range(0, len(Xtr), 16384):
                    _, zt, _ = sae(Xtr[i : i + 16384])
                    fired |= (zt > 0).any(0)
                dead = int((~fired).sum().item())
            print(
                f"  ep{ep:3d}  fvu={fvu:.4f}  L0={l0:5.1f}  dead={dead} ({100*dead/n_features:.1f}%)",
                flush=True,
            )

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save(
        {
            "state_dict": sae.state_dict(),
            "arch": "topk",
            "d_model": d_model,
            "n_features": n_features,
            "k": args.k,
            "layer": args.layer,
            "rep": args.rep,
            "mu": mu,
            "scale": scale,
            "class_names": z["class_names"].tolist(),
            "n_train": len(Xtr),
        },
        args.out,
    )
    print(f"wrote {args.out}  ({(time.time()-t0)/60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
