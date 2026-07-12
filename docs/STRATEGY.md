# Owkin Hackathon — Idea Pressure-Test & Strategy (Bio-Interp-grounded)

**Date:** 2026-07-11 (event July 11–12, SF). **Goal:** decide whether the "causal
mech-interp layer for K Pro" idea is buildable + defensible, and scope it correctly.

The idea is **not speculative for us** — it is a substrate port of the Bio-Interp frozen
suite. Bio-Interp's whole thesis ("feature *discovery* is crowded/commercialized; *causal
certification* — necessity ∧ sufficiency ∧ specificity against interventional oracles,
beating matched-random-circuit nulls — is the open lane") is exactly the hackathon idea,
built for genomics/protein/single-cell FMs. The hackathon extends it to **pathology FMs**
(Phikon-v2 / H0-mini). We already own the methodology (`ObservableEvo2` source-intervention
recipe, error-preserving ablation, sufficiency-by-mean-ablation, matched-random nulls,
on-manifold ablation, conservation/confound controls).

---

## 1. Verdict from the two research agents (cited)

**Prior-art / novelty (scrutiny agent):**
- **SpatialProp** (Sun, Buendia, Brunet & **Zou**, Stanford — the same Zou whose Evo2
  "no guarantee the model used those concepts" critique is Bio-Interp's north star;
  bioRxiv Nov 2025, PMC12822716) is the *real* competitor: a GNN that predicts multi-gene
  perturbation effects on spatial transcriptomics, ships **CausalInteractionBench**, and
  already does "steering." **But it lives entirely in transcriptomic/microenvironment
  space — no histology image, no pathology-ViT latent, no per-prediction necessity/
  sufficiency/specificity certificate.** That gap is our only defensible wedge.
- **STmiR** = near-namesake distractor (XGBoost spatial miRNA activity). Not a competitor.
- Concept work on pathology FMs already exists: **SAEs on pathology FMs** (PathAI,
  arXiv:2407.10785; NeurIPS 2025 follow-up), **concept-Shapley on UNI/Virchow/CONCH**
  (MICCAI 2025, "Explain Any Pathological Concept"). ⇒ "we find interpretable directions"
  is **done**; do NOT lead with discovery. Lead with the **do()-certification + confound
  gate on a live prediction, as an MCP evidence card** — that packaging is unoccupied.
- **Confound pillar is bulletproof (our strongest asset):** Kömen et al. 2024
  (arXiv:2411.05489) — 9 pathology FMs retain linearly-recoverable tissue-source-site
  signatures; >90% site accuracy; **the best models carry the strongest site signatures**;
  scanner-ID ≈ **1.000 for Phikon-v2**. Corroborated by de Jong et al., *Nat Commun* 2025.
- **CytoSyn** (arXiv:2603.18089, H0-mini-conditioned latent diffusion) + **HistoPLUS**
  (arXiv:2508.09926, 13 cell types) confirmed on HF. CytoSyn counterfactuals are
  architecturally plausible (steer the H0-mini conditioning vector) but 24h-risky ⇒ stretch.
- **Sharpest framing:** *"SpatialProp certifies causality in transcriptomic microenvironment
  space; we bring the same do()-style necessity/sufficiency/specificity certification — plus
  a scanner/site confound gate — into the **latent space of a pathology image FM**, turning
  a K-Pro histology prediction into a per-prediction, auditable evidence card."*
- **Strongest judge objection (pre-empt it):** "ablating an embedding axis proves the model
  uses that feature, not that the biology is causal — this is model-internal, dressed-up
  TCAV." → Scope the claim to **model-internal causal certification + confound triage**, not
  biological causation. (Bio-Interp already carries this exact honesty caveat.)

**Feasibility (build red-team agent):**
- **Phikon-v2** confirmed: `AutoModel`/`AutoImageProcessor`, 224px, CLS = **1024-d**, **24
  blocks** (verified at runtime). Hook `model.encoder.layer[i]` for source interventions.
- **Substrate that is guaranteed to work:** Phikon-v2 (open) + `1aurent/NCT-CRC-HE`
  (224px, native, has **TUM** tumor-epithelium + **LYM** lymphocyte/immune-infiltrate
  classes → the on-message tumor-vs-immune pair). MOSAIC is **EGA/DAC-gated, query-only via
  K Pro** this weekend — do NOT architect on raw MOSAIC.
- **HistoPLUS is too heavy for the critical path** (WSI/openslide/784px, per-cell JSON) —
  stretch goal only. NCT-CRC tissue-class labels already give clean concept-probe targets.
- **Necessity/sufficiency asymmetry independently confirmed for vision:** SwordBench
  (arXiv:2605.16372) — concept *ablation* (necessity) is robust/measurable; concept
  *injection* (sufficiency) is fragile and leaks off-target in genomics-style regimes.
  On pathology FMs the asymmetry inverts (see RESULTS.md) — sufficiency is the clean axis.
  ⇒ lead the demo with concept-specific sufficiency + matched-random null; present the
  layer-resolved necessity curve as the rigor story.

---

## 2. Working hypothesis

Port the Bio-Interp Benchmark-D source-intervention recipe to a frozen Phikon-v2 on
NCT-CRC-HE **TUM vs LYM**. Working priors:

- **NECESSITY** — ablating the TUM–LYM concept subspace at intermediate encoder layers
  is expected to be **redundancy-blocked** on a pathology FM (Hydra effect, Bio-Interp
  D02/D04): the model recomputes the distinction downstream. Necessity should bite only
  near the readout, and must be reported as **layer-resolved** with a matched-random-null
  baseline (Section-5-D mandatory control).
- **SUFFICIENCY** — injecting the direction into LYM patches to flip them to TUM is
  expected to be **clean and concept-specific** on this substrate: high flip rate for the
  concept axis, ~0 for a matched-random direction. This inverts the genomics prior and
  is the demo's headline signal.
- **SPECIFICITY** — ablating an orthogonal distractor direction (STR vs MUS stroma/muscle)
  should leave the TUM/LYM probe intact.

**Falsifier:** if a random-direction ablation drops the probe as much as the concept
direction, necessity is an artifact ("ablating any large direction degrades output") and
the certificate is void — precisely the matched-random-circuit control Bio-Interp
Section 5-D mandates.

**Certifiable verb:** **concept-specific steering + confound triage**, with necessity
reported honestly as redundancy-limited. Do not promise biological causation.

Honest caveat (verbatim from Bio-Interp): a latent do() is an intervention on the **model's
representation**, not on tissue biology. We certify model-internal causal use; biological
validity rests on encoder faithfulness (→ that is why the confound gate + literature
grounding exist).

---

## 3. Recommended hackathon scope (depth-on-one-thing)

**MCP tool = one thing done well: `certify(prediction)` → evidence card.** Verbs:
`probe` (derive concept direction), `ablate` (necessity + matched-random null),
`specificity` (distractor ablation), `confound` (is the driver a scanner/site direction —
Kömen-style linear site-probe on the same axis), and an optional caveated `steer`
(sufficiency). Output = structured evidence card (prediction, causal driver, necessity/
specificity scores, **confound flag**, literature citations), not a trust score, not a
dashboard.

- **Track fit:** Frontier (do()-certification + confound gate on a bio-FM is novel packaging);
  Context (grounds reasoning in the model's *own internal evidence* + You.com literature);
  Best-MCP (clean, integratable, serves pharma's faithfulness-audit procurement blocker).
- **The confound gate is the differentiator**, not the concept-finding. It answers the one
  question K Pro provably cannot: *is this prediction real biology or a batch artifact?*
- **CytoSyn counterfactual = precomputed showpiece, droppable.** Never load-bearing.
- Demo lead: **concept-specific SUFFICIENCY (steering) + matched-random null** (clean,
  visual, substrate-appropriate); present the **layer-resolved necessity curve** as the
  rigor story (why naive single-axis TCAV faithfulness on pathology FMs is unsafe); keep
  the **confound gate** as the differentiator (needs multi-site TCGA to demo, since
  NCT-CRC is single-source).
