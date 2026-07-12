# owkin hack

Turn a K-Pro pathology-FM prediction into a per-prediction, auditable **causal evidence
card**, served as an MCP verb — porting the Bio-Interp frozen causal battery onto
pathology foundation models.

## Start here

- **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** — single system map (pipeline, modules, infra).

## Docs

| Doc | For |
|---|---|
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | End-to-end architecture + document index |
| [docs/STRATEGY.md](docs/STRATEGY.md) | Hypothesis, prior-art, feasibility, the wedge |
| [docs/RESULTS.md](docs/RESULTS.md) | Substrate-transfer insights + measured results |
| [docs/SETUP.md](docs/SETUP.md) | Instance transfer, HF/AWS auth, reproduce |
| [docs/DESIGN_MIL_AGGREGATOR.md](docs/DESIGN_MIL_AGGREGATOR.md) | Slide-level aggregation (stretch) |
| [deploy/sagemaker/README.md](deploy/sagemaker/README.md) | Run H-optimus-0 on SageMaker (CLI-only GPU) |
| [CLAUDE.md](CLAUDE.md) | Scope, constraints, working style (agent context) |

Code lives in [`biolayer/`](biolayer/) — see the module map in ARCHITECTURE.md §4.
GPU jobs + weight surgery live in [`deploy/sagemaker/`](deploy/sagemaker/).

**Compute/data:** GPU via SageMaker Training Jobs (CLI only); artifacts in
`s3://bucketbiolayer` (read/write); embeddings routed to the **`h0-vector`** S3 Vectors
store for biodiscovery retrieval. EKS was evaluated and dropped (0 GPU quota).
