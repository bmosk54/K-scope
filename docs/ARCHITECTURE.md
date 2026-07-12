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

Alongside the tile-level certify path, a **whole-slide ingestion → embedding → vector-store**
pipeline (§2b) pulls raw WSIs (TCGA/GDC or any URL) into S3, tiles + embeds them with
H-optimus-0, and routes the vectors into the `h0-vector` S3 Vectors store for biodiscovery
retrieval.

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

## 2b. Whole-slide ingestion → embeddings → vector store

The slide-level path that turns raw WSIs into queryable embeddings for biodiscovery
retrieval. Every stage runs as an **in-region SageMaker job** — the dev box has poor
bandwidth to non-AWS hosts (GDC/GCS stall at ~0 MB/s), while AWS↔AWS and AWS↔web are fast.

```
 WSI source: TCGA/GDC file UUID  |  any direct URL (Kaggle/BRACS, …)
        │  in-region ingest job — biolayer.data.wsi_ingest (idempotent, skip-if-present)
        │  launch: deploy/sagemaker/launch_ingest.py  (ml.m5.large)
        ▼
 s3://bucketbiolayer/wsi/<project>/<slide>.svs|.tiff
        │  GPU tile+embed job — deploy/sagemaker/launch_tile_embed.py  (ml.g5.2xlarge)
        ▼
   wsi_reader.open_wsi  (svs + tiff, one interface, MPP-normalized)
        │
   tile_wsi  — coarse tissue mask → 224px grid @ ~0.5 µm/px → post-tiling FILTERS
        │      (whitespace / tissue; decoupled + extensible; --max-tiles for trials)
        ▼
   H-optimus-0 CLS  [N, 1536]
        ├─▶ features  → s3://bucketbiolayer/embeddings/wsi/<slide>/hoptimus.npz
        └─▶ vectors   → h0-vector / index `layerbioindex` (dim 1536, cosine, float32)
                          → biodiscovery retrieval (QueryVectors)
```

The reader is **format-agnostic** (OpenSlide primary, tifffile fallback) so `.svs` and
`.tiff` never branch; the filter stage is a first-class registry applied inline or
post-hoc (`--filter-existing`). See [DESIGN_MIL_AGGREGATOR.md](DESIGN_MIL_AGGREGATOR.md)
for how these tile embeddings roll up to slide level.

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
| [`biolayer/data/wsi_ingest.py`](../biolayer/data/wsi_ingest.py) | Idempotent slide ingest → S3 (GDC UUID or any URL) |
| [`biolayer/data/wsi_reader.py`](../biolayer/data/wsi_reader.py) | Format-agnostic WSI reader (`.svs`+`.tiff`), MPP-normalized |
| [`biolayer/data/tile_wsi.py`](../biolayer/data/tile_wsi.py) | Tissue-masked tiling + decoupled post-tiling filter stage |
| [`biolayer/causal/`](../biolayer/causal) | The battery: `probe`, `intervene`, `battery`, `confound`, `attribution` |
| [`biolayer/mcp/`](../biolayer/mcp) | MCP `server` + `verbs` + `card` — the `certify` interface |
| [`biolayer/mil/`](../biolayer/mil) | **Stretch:** slide-level aggregation by reusing a ViT's final block |
| [`deploy/sagemaker/`](../deploy/sagemaker) | CLI SageMaker jobs: `launch_ingest` (WSI→S3), `launch_tile_embed` (WSI→features+vectors), `launch` (H-optimus-0 weight edits) |

## 5. Compute & infrastructure (AWS)

- **Auth.** Workspace-scoped: a VSCode terminal profile sources `.owkin_hack_aws.sh`
  (gitignored: AWS session creds + `SAGEMAKER_ROLE_ARN`), exports `HF_TOKEN` live from the
  HF cache, and activates `owkin-env`. On a SageMaker box the execution role provides auth.
  See [SETUP.md](SETUP.md).
- **Storage — two shared stores:**
  - **`s3://bucketbiolayer`** (object storage) — **read/write for the team** (bucket
    policy). Prefixes `embeddings/`, `directions/`, `sae/`, `certificates/` +
    `sagemaker/code` and `sagemaker/output`. `--upload`/`s3_utils` = shared channel;
    `*.npz`/`artifacts/` gitignored.
  - **`h0-vector`** (S3 Vectors, acct `528759081002`, us-west-2) — team-granted
    `PutVectors`/`QueryVectors`/…. **The embedding destination**: tile/slide vectors land
    here and are queried by the biodiscovery retrieval layer.
- **GPU — SageMaker Training Job, CLI only** (no Studio/UI). H-optimus-0 (ViT-g/14) runs
  on **`ml.g5.2xlarge`** (A10G, quota = 1) via the raw-boto3 launchers in
  [`deploy/sagemaker/`](../deploy/sagemaker) (`launch_tile_embed` for WSI→embeddings,
  `launch` for weight edits), using execution role `owkin-sm-exec`; ingest runs on a cheap
  `ml.m5.large`. EKS was evaluated and **dropped**:
  the account has **0 EC2 G/VT and 0 HyperPod g5 quota**, so no cluster can attach a GPU
  node — and a single extraction job plus a stdio MCP server need no orchestration. A
  SageMaker **endpoint** (g5 quota = 2) is the path if hosted GPU inference is ever needed.

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
| [../deploy/sagemaker/README.md](../deploy/sagemaker/README.md) | Run H-optimus-0 on SageMaker (CLI GPU) + arbitrary weight edits |
