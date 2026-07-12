# SAE + `explain` — mechanistic interpretability on H-Optimus-0

**Branch:** `sophia/sae-mechinterp`. Sophia's track. Nothing on `main` is modified.

`certify` (Eddie) and `explain` (this) answer **different questions** and are complementary:

| verb | question | mechanism |
|---|---|---|
| `certify` | *Is the claim supported?* | linear probes — **readout** directions, excellent at detection |
| `explain` | *What did the model use, and how robust is it?* | SAE features — **computational** directions, the only thing you can intervene on |

---

## The single tool

```bash
export HF_TOKEN=...          # H-Optimus-0 is gated (gated=auto, instant approval)
python biolayer/sae/mcp_mechinterp.py     # stdio MCP server
```

**`explain(concept)`** — one verb, five steps:

1. **FIND** — rank SAE features by how selectively they fire on the concept.
2. **GROUND** — render the tiles that drive them: *the morphology the model relied on*.
3. **INTERVENE** — project those feature directions out of the **live model's** residual stream at
   every block 27–39. Twelve blocks run *after* the edit, so this is a `do()`, not an attribution.
4. **CONTROL** — repeat with the same number of **random** features. Without this, "the output
   changed" proves nothing.
5. **FALLBACK** — report what the model believes *instead* once the concept is deleted.

Returns one **evidence card** (`docs/sae_figs/CARD_*.png`): the morphology, the robustness verdict,
the failure mode, and a cell-scale spatial map. **Feature indices never appear in the human-facing
output** — a pharma researcher cannot act on "feature 2524".

---

## Findings

### 1. A probe can *detect* a concept but cannot *ablate* it. SAE features can.

A linear probe identifying tumour at **99.98% accuracy**, projected out of the residual stream at
every block 27–39, moves the model **not at all**: P(TUM) 0.999 → **1.000**. Ablating SAE feature
directions instead: 0.999 → **0.540** (160 random directions: 0.880).

Probe directions are how you *read a concept out*. SAE features are what the model *computes with*.
This is why the two verbs cannot be merged — and it is the technical justification for the SAE.

> **⚠️ For Eddie:** the necessity pillar ablates the probe direction. If that direction is not
> causally load-bearing, a null necessity result may not be the Hydra effect. Worth a look.

### 2. Concepts differ enormously in redundancy — and this is what a K Pro user actually wants to know

| concept | delete 20 features | delete 160 | random-160 control | verdict |
|---|---|---|---|---|
| **LYM** (immune) | **0.236** ← overturned | 0.188 | 0.948 | **SPARSE / AUDITABLE** |
| **TUM** (cancer) | 0.952 | 0.540 | 0.892 | **DISTRIBUTED / REDUNDANT** |

**The model's immune reasoning is auditable; its tumour reasoning is not.** 20 features carry the
immune call — you can look at them. No small set carries the tumour call, so no short explanation
of it exists. `explain` reports this honestly rather than manufacturing a story.

### 3. What the model sees once a concept is deleted

- Delete **tumour** features → it sees **mucus (+0.141), stroma (+0.098), muscle (+0.092)** — *not*
  normal mucosa (+0.054). Tumour is represented as a modification of a mucinous/stromal substrate,
  not as a departure from healthy tissue.
- Delete **immune** features → it sees **smooth muscle (+0.340)**.

### 4. The network self-repairs single-layer edits (Hydra effect — mechanism found)

Editing block 27 alone barely moves the decision: blocks 28–39 **recompute the concept from the 256
untouched patch tokens** via attention. Ablation must be **persistent across blocks**. This confirms
the prediction in `RESULTS.md` — and identifies the route.

### 5. Patch tokens are cell-scale

H-Optimus-0 is patch-14 at ~0.5 µm/px → one patch ≈ **7 µm ≈ one cell**. Cell-level morphology and
immune maps come free, with no cell annotations (`FIG4_spatial_decomposition.png`).

### 6. Layer choice: block 39 (settled empirically)

Early layers *look* richer (10% of block-13 features are cross-class vs 2% at block 39) but those
features fire on **73% of all tiles** and are visually incoherent — **stain/texture artifacts**.
Block 39 has the most usable features (5,538 alive) and they are real morphology.

---

## Things that were tried and REJECTED (do not resurrect)

- **Cosine(SAE decoder, probe direction) as a novelty test.** Geometrically unsound: probe directions
  are discriminative, decoder columns are generative. 100%-pure-LYM features score only 0.16–0.29
  cosine with the LYM probe, so `max_cos < 0.5 → novel` labels obvious lymphocyte detectors as novel.
- **Uncorrected per-feature nulls.** They report **10/10 features significant on 38 RANDOM tiles**.
  Use the family-wise max-statistic null (0/10 on random, 10/10 on real).
- **mean(causal) − mean(background) as the effect statistic.** Assumes a homogeneous causal set; on a
  realistic mixed set (19 TUM + 19 STR) it finds **zero** features. Use activation **rate**.
- **"Probes are blind within a tissue class."** Measured (probe subspace spans 0.5% of within-tumour
  variance vs 0.9% random) but **misinterpreted** — variance-spanned ≠ informativeness. A t-SNE
  falsified it: the probe projection is clearly structured within tumour. **Claim retracted.**
- **ReLU+L1 SAE.** TopK reconstructs 12.5% better at matched sparsity and finds 10/10 real features
  vs L1's 3/10 (identical noise rejection). L1's shrinkage squashes true differential signal.

## Gotchas that fail SILENTLY

1. **H-Optimus-0 has `num_prefix_tokens == 5`** (1 CLS + **4 registers**), not 1. Slicing patches as
   `tokens[:, 1:]` pulls register tokens in with the patches. `DESIGN_MIL_AGGREGATOR.md` §5.1 says
   otherwise — verified false at runtime. **Affects the MIL track.**
2. **Apply the final layernorm at every probed depth** (`norm=True`). The raw residual stream has
   ~15× the norm of the post-LN CLS the probes are fit in.
3. **`probe.py::fit_probe` returns a direction in STANDARDISED space.** Cosine against raw-space
   vectors is meaningless: measured `cos(standardised, raw) = 0.030`. Convert with `w / scaler.scale_`.
   Also: `cos(logistic_raw, diff_of_means) = 0.004` — the two "concept axes" are nearly orthogonal.
4. **NCT-CRC-HE parquet shards are CLASS-SORTED.** Any prefix slice (`feats[:300000]`) is class-biased.
   Always sample at random.
5. **SAEs expand**, so `sae(X)` on a full split allocates (n, n_features): 1.44M × 6144 = **35 GB**.
   Chunk every full-split forward pass.

## Not claimed

- **No molecular or clinical labels** are used (no MSI / BRAF / survival), so **nothing here speaks to
  patient outcome**. TCGA-CRC-DX (Zenodo 2530835) has MSI labels and is downloaded but unused.
- The "decision" is a 9-class tissue head (99.8% accurate) on the final embedding — a faithful stand-in
  for the model's conclusion, **not a clinical prediction**.
- `explain` generates evidence; **a pathologist adjudicates it.**

## Reproduce

```bash
export HF_TOKEN=...
python biolayer/sae/extract_hoptimus.py            # 100k tiles -> blocks 13/27/39, ~24 min
python biolayer/sae/extract_patches.py             # patch tokens (cell scale), ~25 min
python biolayer/sae/train_sae_topk.py --layer 27 --k 40   # ~80 s
python biolayer/sae/mcp_mechinterp.py              # serve `explain`
```

Artifacts (~11 GB of embeddings + SAE checkpoints) are **not** in git. Regenerate with the above.
