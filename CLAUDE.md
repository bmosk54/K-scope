# Owkin Hackathon — Project Context

**Event:** 2026-07-11/12, SF. **Kickoff:** today. **Target:** an MCP tool that turns a
K-Pro pathology-FM prediction into a per-prediction, auditable **causal evidence card**.

This repo is the substrate port of the **Bio-Interp** frozen causal battery from
genomics/protein FMs to **pathology FMs** (Phikon-v2, H0-mini). We already own the
methodology: `ObservableEvo2` source-intervention recipe, error-preserving ablation,
sufficiency-by-mean-ablation, matched-random nulls, on-manifold ablation, conservation /
confound controls.

## Load order for context

1. **[STRATEGY.md](STRATEGY.md)** — hypothesis, prior-art scan, feasibility red-team,
   hackathon scope decision. Read this to understand *why* the project exists and where
   the defensible wedge is.
2. **[RESULTS.md](RESULTS.md)** — the substrate-transfer insights we're building on:
   what to expect from necessity vs sufficiency on a pathology FM, and why the confound
   gate is the differentiator.

## Current scope decision

**Core question the demo answers: _"Characterize the tumor microenvironment."_** This is
the TME-characterization sweet spot: it decomposes (via the Sonnet-4.6 agent step) into
exactly the tissue concepts certify grounds today — tumor epithelium, stroma, immune
infiltrate (LYM-vs-TUM), necrosis, mucus — so every sub-claim earns a real
probe + matched-null + specificity verdict instead of a NOT_CERTIFIABLE. Scope it
**per-prediction / per-slide**, not literally "in this cohort": certify is tile-level and
single-source, so it emits per-claim causal verdicts, *not* cohort survival/response
statistics. Reserve the "cohort" framing for the **confound gate** (is this TME signal
real biology or a site/batch artifact?) — and only once multi-site data is wired.

**Out of scope for `certify`** (route to another verb or decline — verified against the
Sonnet-4.6 test, [Eddie.md](Eddie.md)):
- *Clinical-outcome / biomarker questions* (immunotherapy response, biomarker-for-drug,
  survival correlation, stratify-by-phenotype) — certify never touches outcome labels;
  they resolve NOT_CERTIFIABLE, exactly like MSI-H did in the test.
- *Spatial questions* (immune-inflamed vs -excluded vs -desert, tertiary lymphoid
  structures, tumor↔immune spatial relationship) — no spatial capability on tile-level
  CLS; certify can only assert immune *presence*, not the spatial split.
- *Cell-population dominance* (LYM / NEU / MAC) — the registry has these concepts but the
  HistoPLUS `h0_mini` embeddings don't exist yet; aspirational, unlocks with extraction.
- *Mechanistic "why"* (therapy resistance, CAF activation, hypothesis generation) — that
  is the `hypothesis` verb, not `certify`.

**Ship one MCP verb well: `certify(prediction) → evidence card`.** Sub-verbs: `probe`,
`ablate` (necessity + matched-random null), `specificity` (distractor ablation),
`confound` (Kömen-style site-probe on the causal axis), optional caveated `steer`
(sufficiency).

Insights driving the scope:

- **Sufficiency (steering) is the clean, concept-specific signal on pathology FMs** —
  concept-direction injection flips class assignment while matched-random directions do
  not. Lead the demo with this.
- **Necessity is redundancy-blocked at mid layers** (Hydra effect, Bio-Interp D02/D04)
  and only bites near the readout. Present the layer-resolved necessity curve as the
  *rigor story* — it's exactly why naive single-axis TCAV faithfulness claims on
  pathology FMs are unsafe.
- **Confound gate is the differentiator**, not the concept discovery. It answers the one
  question K-Pro provably cannot: is this prediction real biology or a batch artifact?
  Demoing it needs multi-site data (TCGA or Kömen 2024 setup) — NCT-CRC is single-source.

**Certified verb (do not oversell):** concept-specific **steering + confound triage**,
with necessity reported honestly as redundancy-limited.

## Hard constraints

- **Substrate:** frozen `owkin/phikon-v2` (ViT-L, 24 blocks, CLS=1024-d) + HF
  `1aurent/NCT-CRC-HE` (224px, native TUM/LYM classes). Hook `encoder.layer[i]`.
- **MOSAIC is EGA/DAC-gated, K-Pro query-only this weekend.** Do NOT architect on raw
  MOSAIC data.
- **HistoPLUS / CytoSyn = stretch goals only.** Never load-bearing.
- **Honesty caveat (state proactively in the demo):** a latent do() is an intervention on
  the model's *representation*, not on tissue biology. We certify model-internal causal
  use; biological validity rests on encoder faithfulness — which is why the confound gate
  and literature grounding exist.

## Working style for this repo

- Don't build the demo around necessity-collapse — on redundant pathology concepts it
  shows nothing except at the final layer.
- Matched-random nulls are non-negotiable in every claim (Section-5-D control).
- When suggesting scope creep, check it against STRATEGY.md §3 — depth on one thing.
