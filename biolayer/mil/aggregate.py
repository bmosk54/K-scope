"""Reuse a pathology-ViT's final transformer block as a slide-level MIL aggregator.

The whole idea (DESIGN_MIL_AGGREGATOR.md): a ViT block has no positional term, so
running one block over a bag of pre-extracted tile CLS vectors treats them as an
UNORDERED SET — exactly the inductive bias a MIL slide-aggregator needs. We prepend
a learnable slide-CLS token, run the reused block, and read row 0 as the slide vector.

    block, dim = extract_final_block("h_optimus_0")     # blocks[-1] weights, frozen
    model = SlideMILAggregator(block, dim, num_classes=2)
    logits, emb = model(bag_feats)                      # bag_feats: (B, N, dim)

Contract with the extractor: bag_feats rows are the SAME per-tile global (CLS)
features `biolayer.data.extract` writes as `globals[:, readout, :]` (== `feats`).

Honest defaults, per DESIGN §5–§6:
  * The reused block is FROZEN by default — treat its weights as an initialization,
    not a zero-shot pooler (post-norm CLS in vs pre-norm residual-stream expected).
  * Only the slide-CLS token + a fresh norm + linear head train by default.
  * ALWAYS compare against MeanPoolBaseline (the matched control). If attention does
    not beat mean-pool, the reused block earns nothing — and we say so.

Bags are FIXED-SIZE: timm's Attention takes no key-padding mask, and with a
permutation-invariant block any padding token would corrupt the slide-CLS. So we
sample exactly `bag_size` tiles per slide (with replacement if a slide has fewer).
"""
import numpy as np
import torch
import torch.nn as nn

from .. import config
from ..data.models import DEVICE


# ---------------------------------------------------------------------------
# Reuse: pull the last transformer block out of a frozen backbone
# ---------------------------------------------------------------------------
def extract_final_block(model_key: str, freeze: bool = True):
    """Load a registered backbone and return (final_block, embed_dim).

    Only the last block's weights are kept; the rest of the backbone is dropped.
    timm backend only (H-optimus-0 / H0-mini) — the transformers path (Phikon-v2
    `encoder.layer[-1]`) has a different forward signature and is a TODO.
    """
    spec = config.MODELS[model_key]
    if spec["backend"] != "timm":
        raise NotImplementedError(
            f"{model_key}: only timm backbones are wired here; the transformers "
            "path (encoder.layer[-1], tuple I/O) is a TODO — see DESIGN §7.")

    import timm

    kw = dict(spec.get("timm_kwargs", {}))
    if kw.get("mlp_layer") == "SwiGLUPacked":
        kw["mlp_layer"] = timm.layers.SwiGLUPacked
    if kw.get("act_layer") == "SiLU":
        kw["act_layer"] = torch.nn.SiLU

    base = timm.create_model(f"hf-hub:{spec['hf_id']}", pretrained=True, **kw)
    base.eval()

    assert base.embed_dim == spec["dim"], (
        f"embed_dim {base.embed_dim} != config dim {spec['dim']}")

    block = base.blocks[-1]
    if freeze:
        for p in block.parameters():
            p.requires_grad_(False)
    return block, base.embed_dim


# ---------------------------------------------------------------------------
# The aggregator: slide-CLS token -> reused block -> norm -> head
# ---------------------------------------------------------------------------
class SlideMILAggregator(nn.Module):
    """Permutation-invariant slide aggregator built on a reused ViT block.

    forward(bag_feats) -> (logits, slide_embedding)
        bag_feats : (B, N, dim) pre-extracted tile CLS features
        logits    : (B, num_classes)
        embedding : (B, dim) the slide-level vector (reused-block CLS read-out)
    """

    def __init__(self, block: nn.Module, dim: int, num_classes: int = 2,
                 train_block: bool = False):
        super().__init__()
        self.dim = dim
        self.block = block
        if train_block:                       # opt-in: unfreeze the reused block
            for p in self.block.parameters():
                p.requires_grad_(True)
        # The only always-trained parts: the slide query token, a fresh output
        # norm (the backbone's final norm is not reused), and the linear head.
        self.slide_cls = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.trunc_normal_(self.slide_cls, std=0.02)   # ViT class-token init
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

    def forward(self, bag_feats: torch.Tensor):
        b = bag_feats.size(0)
        cls = self.slide_cls.expand(b, -1, -1)            # (B, 1, dim)
        seq = torch.cat([cls, bag_feats], dim=1)          # (B, N+1, dim)
        seq = self.block(seq)                             # set-attention, no order
        emb = self.norm(seq[:, 0, :])                     # (B, dim) slide vector
        return self.head(emb), emb


class MeanPoolBaseline(nn.Module):
    """Matched control (DESIGN §6): mean-pool the SAME tile features -> head.

    If SlideMILAggregator can't beat this, the reused block adds nothing.
    """

    def __init__(self, dim: int, num_classes: int = 2):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

    def forward(self, bag_feats: torch.Tensor):
        emb = self.norm(bag_feats.mean(dim=1))            # (B, dim)
        return self.head(emb), emb


# ---------------------------------------------------------------------------
# Synthetic slide bags from tile-labeled features (NCT-CRC has no slide labels)
# ---------------------------------------------------------------------------
def make_synthetic_bags(npz_path: str, pos_class: str, neg_class: str,
                        bag_size: int = 256, n_bags_per_class: int = 200,
                        contamination: float = 0.3, seed: int = 0):
    """Build fixed-size pseudo-slide bags from an extractor .npz for a binary task.

    A POSITIVE slide = `bag_size` tiles that are mostly `pos_class` with a fraction
    `contamination` drawn from other classes (a realistic "tumor present" bag);
    a NEGATIVE slide = tiles from `neg_class` + other non-pos classes, NO pos tiles.
    This is a demo stand-in for real slide labels (see DESIGN §8 open-Q 4).

    Returns (bags, labels): bags (M, bag_size, dim) float32, labels (M,) int {0,1}.
    """
    z = np.load(npz_path, allow_pickle=True)
    feats = z["feats"] if "feats" in z else z["globals"][:, -1, :]  # (N_tiles, dim)
    labels = z["labels"]
    names = list(z["class_names"])
    pos_i, neg_i = names.index(pos_class), names.index(neg_class)

    idx_pos = np.where(labels == pos_i)[0]
    idx_neg = np.where(labels == neg_i)[0]
    idx_other = np.where((labels != pos_i))[0]     # anything not-positive = background
    rng = np.random.default_rng(seed)

    def _sample(pool, k):
        return rng.choice(pool, size=k, replace=len(pool) < k)

    n_contam = int(round(bag_size * contamination))
    bags, ys = [], []
    for _ in range(n_bags_per_class):              # positive slides
        rows = np.concatenate([_sample(idx_pos, bag_size - n_contam),
                               _sample(idx_other, n_contam)])
        bags.append(feats[rows]); ys.append(1)
    for _ in range(n_bags_per_class):              # negative slides (no pos tiles)
        base = _sample(idx_neg, bag_size - n_contam)
        extra = _sample(idx_other[labels[idx_other] != neg_i], n_contam)
        bags.append(feats[np.concatenate([base, extra])]); ys.append(0)

    bags = np.stack(bags).astype(np.float32)
    ys = np.asarray(ys, dtype=np.int64)
    perm = rng.permutation(len(ys))
    return bags[perm], ys[perm]


# ---------------------------------------------------------------------------
# Minimal train/eval loop + the mean-pool control, as a smoke test / CLI
# ---------------------------------------------------------------------------
def _run_epoch(model, X, y, opt=None, bs=16):
    train = opt is not None
    model.train(train)
    idx = torch.randperm(len(y)) if train else torch.arange(len(y))
    tot, correct, loss_sum = 0, 0, 0.0
    lossf = nn.CrossEntropyLoss()
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for i in range(0, len(y), bs):
            sel = idx[i:i + bs]
            xb = torch.as_tensor(X[sel]).to(DEVICE)
            yb = torch.as_tensor(y[sel]).to(DEVICE)
            logits, _ = model(xb)
            loss = lossf(logits, yb)
            if train:
                opt.zero_grad(); loss.backward(); opt.step()
            loss_sum += loss.item() * len(sel)
            correct += (logits.argmax(1) == yb).sum().item()
            tot += len(sel)
    return loss_sum / tot, correct / tot


def train_and_compare(npz_path: str, model_key: str = "h_optimus_0",
                      pos_class: str = "TUM", neg_class: str = "NORM",
                      bag_size: int = 256, epochs: int = 20, lr: float = 1e-3,
                      seed: int = 0):
    """Train SlideMILAggregator and MeanPoolBaseline on the same bags; report both."""
    torch.manual_seed(seed)
    X, y = make_synthetic_bags(npz_path, pos_class, neg_class,
                               bag_size=bag_size, seed=seed)
    n_val = len(y) // 5
    (Xtr, ytr), (Xva, yva) = (X[n_val:], y[n_val:]), (X[:n_val], y[:n_val])
    dim = X.shape[-1]

    block, dim_b = extract_final_block(model_key, freeze=True)
    assert dim_b == dim, f"feature dim {dim} != backbone dim {dim_b}"

    models = {
        "reused-block-MIL": SlideMILAggregator(block, dim, num_classes=2).to(DEVICE),
        "mean-pool-control": MeanPoolBaseline(dim, num_classes=2).to(DEVICE),
    }
    out = {}
    for name, m in models.items():
        opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=lr)
        best = 0.0
        for ep in range(epochs):
            _run_epoch(m, Xtr, ytr, opt)
            _, va = _run_epoch(m, Xva, yva)
            best = max(best, va)
        out[name] = best
        print(f"{name:>18s}  best val acc = {best:.3f}", flush=True)
    verdict = ("reused block BEATS mean-pool"
               if out["reused-block-MIL"] > out["mean-pool-control"] + 0.01
               else "no gain over mean-pool — reused block earns nothing (report honestly)")
    print(f"VERDICT: {verdict}", flush=True)
    return out


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("npz", help="extractor .npz (e.g. embeddings/.../train.npz)")
    ap.add_argument("--model", default="h_optimus_0", choices=list(config.MODELS))
    ap.add_argument("--pos", default="TUM")
    ap.add_argument("--neg", default="NORM")
    ap.add_argument("--bag-size", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=20)
    args = ap.parse_args()
    train_and_compare(args.npz, args.model, args.pos, args.neg,
                      bag_size=args.bag_size, epochs=args.epochs)
