# Design Note — Reusing H-optimus-0's Final Block as a Slide-Level MIL Aggregator

**Status:** exploratory / stretch. **Author track:** cross-cutting (applies to any
pathology-ViT substrate). **Date:** 2026-07-11.

> **Scope caveat first.** The certified verb this weekend is tile-level `certify(prediction)
> → evidence card` on Phikon-v2 / H0-mini (see [CLAUDE.md](../CLAUDE.md), [STRATEGY.md](STRATEGY.md)).
> This note describes an *optional* way to lift tile embeddings to a **slide-level** vector
> **without importing an external MIL architecture** — reusing weights we already load. It is
> a stretch path, not on the critical demo path. Read it as "how we'd aggregate to whole-slide
> if a slide-level label or slide-level concept axis becomes the target."

---

## 1. Core problem

H-optimus-0 (Bioptimus, ViT-**g**/14, DINOv2-pretrained, **CLS = 1536-d**) is hard-wired to
a single tile: it patch-embeds a 224×224 image into a fixed 16×16 = **256-patch** grid and
adds **2-D positional encodings** to those patches before the transformer stack.

You therefore **cannot** feed it a whole slide, or a bag of patches drawn from across a
10,000×10,000 image, at the image level:

- More than 256 patches overflows the positional-embedding table.
- Patches scrambled from different spatial locations get the **wrong** positional weights,
  so self-attention mixes tiles under a false 2-D geometry and the output embedding is
  corrupted.

The model gives you a clean, order-agnostic representation **per tile** (the CLS vector),
but nothing native above the tile.

## 2. The idea

Reuse H-optimus-0's **own final transformer block** as a set aggregator over
pre-extracted tile embeddings — no ABMIL, no TransMIL, no new backbone.

**Why it is even allowed to work — the one load-bearing fact:** in a ViT, positional
information is injected **once**, at the patch-embedding layer. The transformer *blocks*
themselves contain no position term — they are pure multi-head self-attention + MLP, and
self-attention over a token set is **permutation-invariant**. So if we skip the
patch-embed / positional-encoding front end and hand a block a sequence of tokens directly,
it treats them as an **unordered set** of exactly the right dimensionality (1536). That is
precisely the inductive bias a Multiple-Instance-Learning (MIL) aggregator needs: a bag of
tile instances, order irrelevant, pooled into one slide vector.

This turns a frozen pathology FM into its own slide-level aggregator, keeping everything in
the substrate we already trust and already load.

## 3. Pipeline

```
10k×10k WSI
  └─ tissue mask + tile into 224×224 → N valid tiles   (N ≈ 500–1900 after masking)
       └─ H-optimus-0 forward per tile → take CLS (index 0) → [N, 1536]   ← we already do this
            └─ prepend learnable slide-CLS token → [1, N+1, 1536]
                 └─ blocks[-1]  (permutation-invariant set attention)
                      └─ take output row 0 → slide embedding [1, 1536]
                           └─ linear head → slide-level logit(s)
```

Stage 1 (tile → `[N, 1536]`) is exactly what `biolayer.data.extract` already produces per
tile (global CLS feature). This design only adds stages 2–4 on top of that matrix.

## 4. Reference implementation

```python
import timm
import torch
import torch.nn as nn

# --- 0. Load backbone once; we only keep its last block's weights ---
# NOTE the hyphen: the timm prefix is "hf-hub:", not "hf_hub:".
# H-optimus-0 needs the non-default construction args used in biolayer.config:
base = timm.create_model(
    "hf-hub:bioptimus/H-optimus-0", pretrained=True,
    init_values=1e-5, dynamic_img_size=False,
)
base.eval()

# VERIFY BEFORE TRUSTING (see §5):
#   base.embed_dim                        -> expect 1536
#   getattr(base, "num_prefix_tokens", 1) -> H-optimus-0 = 1 (CLS only); H0-mini adds registers
#   len(base.blocks)                      -> 40 (ViT-g); the reused block is blocks[-1]
mil_block = base.blocks[-1]        # the isolated set-aggregator

DIM = base.embed_dim               # 1536, do not hard-code

# --- 1. Pre-extracted tile CLS features for one slide ---
# Shape [B, N, DIM]. Produced by running each 224x224 tile through the full backbone
# and keeping only CLS (index 0). Here mocked:
tile_feats = torch.randn(1, 1500, DIM)

# --- 2. Learnable slide-level CLS token (this is what actually gets trained) ---
slide_cls = nn.Parameter(torch.zeros(1, 1, DIM))   # zero-init, like ViT's class token

# --- 3. Assemble the bag and run ONE block ---
seq = torch.cat([slide_cls.expand(tile_feats.size(0), -1, -1), tile_feats], dim=1)  # [B, N+1, DIM]
out = mil_block(seq)                                # permutation-invariant over the N tiles
slide_embedding = out[:, 0, :]                      # [B, DIM] — read the slide-CLS row

# --- 4. Trainable head for slide-level labels (tumor vs normal, etc.) ---
head = nn.Linear(DIM, num_classes)
logits = head(slide_embedding)
```

The `torch.no_grad()` "zero-shot" version in the original sketch runs, but see §5.2 — its
output is **not** a meaningful slide embedding until the slide-CLS token (and head) are trained.

## 5. What the quick sketch gets wrong / glosses over

These are the reasons to treat this as a *trainable initialization*, not a plug-in oracle.

**5.1 CLS index & register tokens.** In `biolayer` the per-tile global feature is
`get_intermediate_layers(..., return_prefix_tokens=True, norm=True)` → `prefix[:, 0]`, i.e.
the **post-norm CLS** (feeds directly into §5.2). Register tokens are an **H0-mini** concern
(its prefix = CLS + registers); H-optimus-0 uses CLS only (`num_prefix_tokens == 1`) — still
worth asserting at runtime. When *aggregating* we build our own sequence, so registers are
irrelevant there — just don't reuse the tile-extraction index blindly across models.

**5.2 Distribution mismatch — the big one.** `blocks[-1]` was trained to consume the
**residual-stream hidden state at depth L−1** (pre-final-norm), whose statistics differ from
the **post-final-norm CLS output** we extract per tile. The block is pre-norm (it applies
`norm1` to its input first), which absorbs scale but not the full distribution shift. Net:
the pretrained aggregation weights are being used **off-distribution**. Expect this to be a
*good initialization*, not a correct zero-shot pooler. **You must train** (at minimum the
slide-CLS token + linear head; optionally unfreeze `blocks[-1]`).

**5.3 Zero-init token + `no_grad` ⇒ nothing learned.** An all-zero slide-CLS token has no
information to contribute and no gradient path under `no_grad`; the "slide embedding" it
returns is an untrained attention read-out. It only becomes meaningful after training gives
the token a query worth attending with.

**5.4 Attention entropy at 1500 tokens.** The block's softmax temperature was tuned for a
~257-token context (256 patches + CLS). Over ~1500 tiles, attention mass spreads thinner
(higher entropy) — another reason a short fine-tune (or a learned temperature/scale on the
slide-CLS query) matters. Not fatal; just don't expect trained-context behavior for free.

**5.5 Permutation-invariance is exact — and that's the point.** Unlike the *image-level*
model, the isolated block has **no** positional term, so tile order genuinely doesn't matter.
Corollary: this pooler is **blind to spatial arrangement** between tiles. If slide-level
signal is spatial (architecture, gland topology), pure set-attention discards it — a known
MIL limitation, not a bug of this hack.

## 6. Training recipe (minimal, data-efficient)

Slide-level labels are scarce, so keep the trainable surface small:

| Config | Trainable params | When |
|---|---|---|
| **Frozen-block** (default) | slide-CLS token (1×1536) + `Linear(1536→C)` | few slides; safest |
| **LoRA / unfreeze last block** | + `blocks[-1]` weights | more slides; if frozen underfits |

- Loss: standard CE (or BCE for multi-label) on slide labels (e.g. tumor vs normal).
- **Non-negotiable control (house rule, [CLAUDE.md](../CLAUDE.md)):** report a **matched-random
  baseline** — mean-pool of the same tile features → same head. If block-attention doesn't
  beat mean-pool, the reused block is adding nothing and we say so.
- Sanity ceiling: attention-MIL (ABMIL) with an equivalent param budget, as an external
  reference point (comparison only — not shipped, per the "no outside architectures" rule).

## 7. How this connects to the project

- The slide vector becomes a **new substrate** the frozen causal battery can run on: probe /
  ablate / confound-gate a *slide-level* concept axis instead of a tile axis. The confound
  gate (Kömen-style site-signature probe) is arguably **more** important at slide level,
  where scanner/site batch effects aggregate.
- It reuses the exact tile-CLS matrix `biolayer.data.extract` emits, so no new extraction
  path — only an aggregation head.
- Substrate-agnostic: same trick applies to Phikon-v2 (`.encoder.layer[-1]`, 1024-d) if we
  want the aggregator on the ungated model first.

## 8. Open questions

1. Does `blocks[-1]` frozen actually beat mean-pool on NCT-CRC slide-style bags? (§6 control.)
2. Register-token count for H-optimus-0 — confirm `num_prefix_tokens` empirically.
3. Better to feed **pre-final-norm** tile hidden states (§5.2) than post-norm CLS? Would
   need to hook `blocks[-2]` output during tile extraction — closes the distribution gap at
   the cost of a heavier extract.
4. Slide labels: NCT-CRC is tile-labeled, not slide-labeled — where does slide-level ground
   truth come from for a demo (synthetic bags from tile labels vs. TCGA slide labels)?

---

*Next step if we pursue this: wire the `Linear` head + CE loss and run the §6 mean-pool
control on a synthetic bag built from NCT-CRC tile labels. Ask and I'll draft
`biolayer/mil/aggregate.py` against the real extractor output.*
