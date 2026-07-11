# Owkin-Hack — System Architecture

**One line:** turn a K-Pro pathology-FM prediction into a per-prediction, auditable
**causal evidence card**, served as an MCP verb — by porting the Bio-Interp frozen causal
battery onto pathology foundation models.

This is the single map of the system. Deep-dives live in the sibling docs (§7 index).

---

## 1. What the system does

`certify(prediction) → evidence card`. Given a tile (and a concept, e.g. TUM-vs-LYM), the
system runs a frozen pathology encoder, then a battery of latent interventions on that
encoder's activations, and emits a structured card stating — with matched-random-null
controls — whether the concept is **necessary**, **sufficient**, **specific**, and whether
the signal survives a **site/scanner confound gate**. See [CLAUDE.md](../CLAUDE.md) for the
scope decision and [STRATEGY.md](STRATEGY.md) for why this is the defensible wedge.

## 2. End-to-end pipeline

```
                        ┌──────────────────────────────────────────────────┐
   H&E tiles            │  COMPUTE SUBSTRATE  (AWS GPU — SageMaker / EKS G5) │
 (NCT-CRC-HE,           └──────────────────────────────────────────────────┘
  224px, 9 classes)
        │
        ▼
 ┌─────────────────┐   frozen encoder, 3 layers × {global CLS, local mean-patch}
 │ data.extract    │──────────────────────────────────────────────┐
 │  (per track)    │   globals (N,3,D)  locals (N,3,D)  labels (N)  │
 └─────────────────┘                                               ▼
        │                                              ┌────────────────────────┐
        │  .npz artifacts                              │  data.s3_utils         │
        └─────────────────────────────────────────────▶│  s3://bucketbiolayer/  │
                                                       │  embeddings/…          │
                                                       └────────────────────────┘
        │
        ▼
 ┌─────────────────────────────────────────────┐
 │ causal battery  (biolayer.causal)            │   hooks encoder.layer[L],
 │  probe · intervene · battery · confound      │   edits activations, re-reads CLS
 │  attribution                                 │   → necessity / sufficiency /
 └─────────────────────────────────────────────┘     specificity vs matched-random null
        │
        ▼
 ┌─────────────────────────────────────────────┐
 │ MCP server  (biolayer.mcp)                   │   certify(prediction) → JSON card
 │  server · verbs · card                       │   sub-verbs: probe, ablate,
 └─────────────────────────────────────────────┘   specificity, confound, (steer)
        │
        ▼
   evidence card  (certificates/…json)

        ╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌
   OPTIONAL / STRETCH — slide-level branch (biolayer.mil):
   tile CLS matrix [N,1536] ─▶ reuse H-optimus-0 blocks[-1] as a
   permutation-invariant MIL aggregator ─▶ slide embedding [1,1536]
   ─▶ head. Same causal battery can then run at SLIDE level.
   Details: docs/DESIGN_MIL_AGGREGATOR.md
```

## 3. Substrate (frozen encoders + data)

Model registry and dataset layout are the single source of truth in
[`biolayer/config.py`](../biolayer/config.py). Two independent **tracks**
([`biolayer/tracks`](../biolayer/tracks)) never share assumptions:

| Track | Model | Backend | Dim | Blocks | Layers (probed) | Objective |
|---|---|---|---|---|---|---|
| `phikon` | `owkin/phikon-v2` (ungated) | transformers | 1024 | 24 | 8/16/24 | TUM vs LYM |
| `h0` | `bioptimus/H0-mini` (gated) | timm | 768 | 12 | 3/7/11 | TUM vs NORM |
| (extract-only) | `bioptimus/H-optimus-0` (gated) | timm | 1536 | 40 | 13/27/39 | — |

Every tile is embedded at **3 depths** × **{global CLS, local mean-patch}**. Dataset:
`1aurent/NCT-CRC-HE` (224px, native 9 tissue classes; single-source, Macenko-normalized).

## 4. Module map

| Path | Role |
|---|---|
| [`biolayer/config.py`](../biolayer/config.py) | Model registry, S3 key layout, dataset/split/class constants |
| [`biolayer/tracks/`](../biolayer/tracks) | Per-track bundles (model + dataset + objective + layers) |
| [`biolayer/data/models.py`](../biolayer/data/models.py) | Frozen encoder loading; multi-layer local+global `embed()` |
| [`biolayer/data/extract.py`](../biolayer/data/extract.py) | CLI: tile → `.npz` embeddings, optional S3 upload |
| [`biolayer/data/s3_utils.py`](../biolayer/data/s3_utils.py) | Shared S3 artifact channel |
| [`biolayer/causal/`](../biolayer/causal) | The battery: `probe`, `intervene`, `battery`, `confound`, `attribution` |
| [`biolayer/mcp/`](../biolayer/mcp) | MCP `server` + `verbs` + `card` — the `certify` interface |
| [`biolayer/mil/`](../biolayer/mil) | **New / stretch:** slide-level aggregation by reusing a ViT's final block |

## 5. Compute & infrastructure (AWS)

- **Auth.** Locally, workspace-scoped: a VSCode terminal profile sources
  `.owkin_hack_aws.sh` (gitignored) + activates `owkin-env`. On the box, the SageMaker
  **execution role** provides S3/GPU auth (no keys). See [SETUP.md](SETUP.md).
- **Storage.** `s3://bucketbiolayer/` with per-dataset/per-model prefixes
  (`embeddings/`, `directions/`, `sae/`, `certificates/`). Note the current role has
  **ListBucket only** — until the policy is fixed, embeddings regenerate locally
  (`--no-upload`); they are gitignored (`*.npz`, `artifacts/`).
- **GPU.** H-optimus-0 (ViT-g/14) needs a GPU. EKS cluster `fabulous-pop-sculpture`
  exists (us-west-2); the intended worker is a **managed `g5.2xlarge` nodegroup**. **But
  the account's On-Demand G/VT quota is 0**, so the nodegroup can't launch until an admin
  raises it to ≥ 8 — see [`infra/`](../infra/README.md). Fallback with no quota need:
  **SageMaker `ml.g5.2xlarge`** (quota = 1, same A10G), per [SETUP.md](SETUP.md). (An EKS
  Hybrid-Nodes path — attach a non-EC2 external GPU — is kept in `infra/` as an alternative.)

## 6. Constraints & honesty caveats (non-negotiable)

- **Matched-random null in every claim** (Bio-Interp Section-5-D control). A result that
  doesn't beat a matched-random subspace/direction is not a certificate.
- **Necessity is redundancy-limited** on pathology FMs (Hydra effect) — report it
  layer-resolved, honestly; lead the demo with **sufficiency (steering)** + the null.
- **Confound gate needs multi-site data** (NCT-CRC is single-source) — TCGA/Kömen setup.
- **A latent do() intervenes on the model's representation, not on tissue biology.** We
  certify model-internal causal use; biological validity rests on encoder faithfulness.
- **MOSAIC is EGA/DAC-gated, K-Pro query-only** this weekend — do not architect on raw
  MOSAIC. HistoPLUS / CytoSyn are stretch only, never load-bearing.

## 7. Document index

| Doc | Read it for |
|---|---|
| [CLAUDE.md](../CLAUDE.md) | Scope decision, hard constraints, working style (loaded into context) |
| [STRATEGY.md](STRATEGY.md) | Hypothesis, prior-art scan, feasibility red-team, the wedge |
| [RESULTS.md](RESULTS.md) | Substrate-transfer insights + measured readout-space battery results |
| [SETUP.md](SETUP.md) | Instance transfer, HF/AWS auth, reproduce steps |
| [DESIGN_MIL_AGGREGATOR.md](DESIGN_MIL_AGGREGATOR.md) | Slide-level aggregation by reusing a ViT's final block |
