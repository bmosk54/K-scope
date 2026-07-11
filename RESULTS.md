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
