# Substrate-transfer insights: Bio-Interp causal battery → Phikon-v2

Working priors for porting the Bio-Interp frozen causal suite from genomics/protein FMs
to a pathology FM (`owkin/phikon-v2`, ViT-L, 24 blocks, CLS=1024-d) on NCT-CRC-HE,
target = **TUM (tumor epithelium) vs LYM (lymphocyte / immune infiltrate)**, distractor
= STR/MUS. Linear probe on the frozen CLS, then Bio-Interp Benchmark-D
**source-intervention** battery (hook `encoder.layer[L]`, edit activations, propagate to
CLS → probe): rank-k concept-subspace ablation vs matched-random-subspace null across
layers; single-direction sweep; sufficiency-by-injection vs random.

## What to expect

| Test | Expected behavior on Phikon-v2 |
|---|---|
| Linear probe (baseline) | Trivially separable at every layer — TUM/LYM is a strongly encoded native class boundary. |
| Readout-space necessity (project probe axis out of final CLS) | Collapses toward chance — mechanism sanity-check. |
| Necessity, mid layers (4/10/16) | Ablation of even a large concept subspace **does not move the probe** — the model recomputes the distinction downstream from un-ablated patch tokens. |
| Necessity, near-readout (L22) | Ablation bites; concept subspace drops the probe below the random-null baseline by a meaningful gap. |
| Sufficiency (inject concept dir @ mid layer) | High LYM→TUM flip rate; **matched-random direction flips ~0%**. |
| Matched-random null (everywhere) | Never moves the probe — the Section-5-D control that separates a real certificate from an artifact. |

## What this means (the real insights)

1. **Naive latent necessity ablation certifies almost nothing on a pathology FM.**
   TUM/LYM is *massively redundantly* encoded — ablating even a large concept subspace at
   early/mid layers leaves the prediction intact because the model recomputes it from the
   un-ablated patch tokens. This is the **Hydra effect / distributed-signal** result
   (Bio-Interp D02/D04; McGrath 2023). Necessity is only *partially* certifiable at the
   final layer(s), and even there the concept subspace only partially drops the probe,
   not to chance.
   ⇒ **A tool that certifies by ablation-collapse would falsely report "not necessary"
   everywhere except the readout.** Necessity must be **layer-resolved** and reported
   honestly as redundancy-limited.

2. **Sufficiency/steering is the CLEAN, concept-specific signal here — the opposite of
   the genomics prior.** On sequence FMs Bio-Interp found necessity works / sufficiency
   fails; on a pathology FM the concept direction is a **sufficient and specific**
   steering axis. This is a genuine cross-substrate contrast and it is *good news* for
   the "steer tissue toward a desired state" application (the perturbation-design loop).

3. **The matched-random null is essential.** Random subspaces/directions must not move
   the probe — this is what certifies concept-specific effects against "any large edit
   degrades the output" (Section-5-D control), and it's what turns a demo into a
   certificate.

## Implications for the hackathon build

- **Do NOT build the demo around necessity-collapse** — on redundant pathology concepts
  it shows nothing except at the final layer. Lead with **concept-specific SUFFICIENCY
  (steering) + the matched-random null**, which is clean, visual (flip rate), and
  substrate-appropriate.
- The **layer-resolved necessity curve is itself a rigor story**: it's exactly the check
  that separates this from naive TCAV/probe attribution, and it demonstrates *why*
  single-axis faithfulness claims on pathology FMs are unsafe.
- **Confound gate needs multi-site data.** NCT-CRC is single-source and Macenko-normalized
  — no scanner/site variation to test against. To demo the confound check, use TCGA H&E
  across ≥2 sites or the Kömen 2024 setup and run the same subspace machinery against a
  *site* probe.
- **Honest caveat (state proactively):** a latent do() is an intervention on the model's
  representation, not on tissue biology; this certifies model-internal causal use, whose
  biological validity rests on encoder faithfulness.

---

# Measured results — readout-space battery (2026-07-11)

First run of the **hook-free, CLS-space** subset of the battery (`biolayer.battery`):
linear probe + readout-space necessity + sufficiency-by-injection + distractor
specificity, each against a **matched-random null** (n=200 unit directions). This is
**not** the layer-resolved source-intervention suite (that needs `intervene.py` hooking
`encoder.layer[L]`, not built yet) — so the mid-layer redundancy / Hydra-effect priors
below remain **untested** here.

**Setup.** NCT-CRC-HE `NCT_CRC_HE_100K` (Macenko-normalized), balanced stratified subset
**600 tiles/class × 9 = 5400**, 224px. Concept **TUM vs LYM**, distractor **STR vs MUS**.
Probe = standardized logistic regression, 60/40 train/test on the pair (n_train=720,
n_test=480). Extraction on **A10G GPU** (not the CPU box originally assumed). Two frozen
encoders:

| Metric | **Phikon-v2** (1024-d) | **H-optimus-0** (1536-d) | Prior (RESULTS above) |
|---|---|---|---|
| Probe test acc | **1.000** | **0.998** | trivially separable ✅ |
| Necessity — concept axis projected out | 1.000 → **0.500** (chance) | 0.998 → **0.500** | readout necessity collapses ✅ |
| Necessity — matched-random axis out (mean±sd) | **1.000 ± 0.000** | **0.998 ± 0.000** | random never moves it ✅ |
| Sufficiency — concept-dir flip rate (LYM→TUM) | **1.000** | **1.000** | high flip rate ✅ |
| Sufficiency — matched-random flip rate | **0.000 ± 0.000** | **0.004 ± 0.002** | random ≈0% ✅ |
| Specificity — cos(concept, distractor axis) | 0.020 | 0.008 | near-orthogonal ✅ |
| Specificity — target acc after distractor ablation | 1.000 (base 1.000) | 0.998 (base 0.998) | target intact ✅ |

Evidence cards: `artifacts/certificates/{phikon_v2,h_optimus_0}_train_TUM_vs_LYM.json`.

## Read of the numbers

1. **Every readout-space prior held, on both encoders.** The concept axis carries the
   TUM/LYM readout (projecting it out drops the probe to exactly chance), matched-random
   projections leave it untouched, and injecting the concept direction flips 100% of LYM
   tiles to TUM while random directions flip ~0%. The two independently-trained FMs agree,
   which is a genuine cross-model replication rather than a single-model artifact.

2. **Sufficiency/steering is the clean, concept-specific signal — confirmed.** 100% vs 0%
   flip with a near-zero random null is exactly the substrate-appropriate result the
   strategy leads with. This is the demo headline.

3. **Honest caveats on the necessity/z number.**
   - The matched-random null has **std = 0** — every random direction leaves accuracy at
     baseline — so the reported `concept_vs_null_z` (~5e8) is a **divide-by-~0 artifact,
     not a real effect size.** Report it categorically ("random projections never move a
     1-D readout in 1024/1536-D space; the concept axis takes it to chance"), not as a
     z-score.
   - This necessity is **readout-space projection only** — it is the *mechanism
     sanity-check* row of the priors, and it collapses to chance by construction. It does
     **not** test the mid-layer redundancy claim. The load-bearing "necessity is
     redundancy-limited except near the readout" result still requires the layered
     source-intervention module and is **not yet demonstrated.**

## Provenance / caveats

- **H-optimus-0, not H0-mini.** H0-mini (768-d, CytoSyn-aligned) is still gated/awaiting
  approval; H-optimus-0 (1536-d, gated-AUTO, CC-BY-NC-ND research-only) is used as the
  second column. Swap in H0-mini once approved.
- **Single-source data** — NCT-CRC is one cohort, Macenko-normalized. The **confound gate
  is still untestable here** (no site/scanner variation). Needs multi-site (TCGA/Kömen).
- **Embeddings not yet in S3.** `train.npz` for both models is local under
  `artifacts/embeddings/...`; the shared-bucket push is blocked on an S3 write permission
  for the SageMaker role (bucket-policy / SCP), not on the pipeline.
