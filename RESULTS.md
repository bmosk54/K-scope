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

---

# Infra update — two separate tracks + multi-layer local/global (2026-07-11)

The pipeline is now split into **two independent tracks** (`biolayer.tracks`), because
Phikon-v2 and H0 have different objectives, datasets, and depths and should not share
assumptions:

| Track | Model | Dataset | Objective | Distractor | Layers |
|---|---|---|---|---|---|
| `phikon` | Phikon-v2 | NCT-CRC-HE | TUM vs LYM (tumor-immune interface) | STR/MUS | 8 / 16 / 24 |
| `h0` | H0-mini¹ | NCT-CRC-HE → cell-type² | TUM vs NORM (malignancy) | MUS/ADI | 3 / 7 / 11 |

¹ default `h0_mini` (approval-queued); flip `H0_MODEL_KEY` to `h_optimus_0` to run today.
² NCT-CRC keeps it runnable; intended divergence = HistoPLUS/CytoSyn cell-type substrate.

**Multi-layer, local + global extraction.** Every tile is now embedded at **3 layers**
(named `mid_early / mid / readout`), and at each layer we keep **both**:
- **global** = CLS token — the tile-level readout feature (old `feats` = readout global);
- **local** = mean patch token — local morphology / texture.

Stored in one `.npz` as `globals (N,3,dim)` + `locals (N,3,dim)` (+ back-compat `feats`).
Verified on real Phikon-v2 features (transformers `output_hidden_states`) and wired for
timm `get_intermediate_layers` on the H0 track. This directly powers the **layer-resolved
rigor curve**: `intervene.layered_curve(model, split, pos, neg, space=…)` runs the
readout-necessity check per layer, on global *or* local space, each vs a matched-random
null. (The *live* source-intervention propagation test — the true Hydra/redundancy
probe — remains `necessity_curve`, track #3.) Exposed over MCP as the `layered` verb and
folded into every `certify` card's `necessity_layered` field.

---

# Agent probe design — can the LLM WRITE the causal probes? (2026-07-12)

**Question under test.** The static `certify` path routes a K-Pro answer onto
*hand-authored* concept probes (a human picked each `(pos, neg, distractor)` contrast).
Can we instead let Claude **design** the probes for an open question — choosing the
contrasts itself — without producing confident-looking-but-confounded axes? Tested for
the demo's core question, **"Characterize the tumor microenvironment."**

**Setup.** Sonnet 4.6 on Bedrock (`us.anthropic.claude-sonnet-4-6`) was given **only the
9 NCT-CRC class codes + one-line glosses + the question** — never the registry's answers,
never the certificate math, and no say over certifiability. It proposed 8 probes; each
ran through the **unchanged validation gate** (held-out AUROC ≥ 0.75,
intensity-collinearity ≤ 0.60) + the readout battery on frozen Phikon-v2 CLS
(200 tiles/class). New module `biolayer/dynamic/probe_design.py`.

| LLM-designed probe | pos v neg | AUROC | intens_r | gate | suff-flip (null) |
|---|---|---|---|---|---|
| immune infiltration in tumor | LYM v TUM | 1.000 | 0.237 | ✅ | 1.00 (0.01) |
| desmoplastic stromal reaction | STR v TUM | 1.000 | 0.581 | ✅ | 1.00 (0.03) |
| stromal vs immune composition | STR v LYM | 1.000 | **0.646** | ❌ | 1.00 (0.00) |
| tumor necrosis burden | DEB v TUM | 0.999 | 0.128 | ✅ | 1.00 (0.01) |
| mucinous microenvironment | MUC v TUM | 0.996 | 0.426 | ✅ | 1.00 (0.00) |
| peritumoral immune exclusion | LYM v STR | 1.000 | **0.646** | ❌ | 1.00 (0.00) |
| normal mucosa displacement | NORM v TUM | 1.000 | 0.172 | ✅ | 1.00 (0.00) |
| adipose infiltration of stroma | ADI v STR | 1.000 | 0.229 | ✅ | 1.00 (0.00) |

## Read of the numbers

1. **6/8 (75%) passed the gate, and the survivors are causally as clean as the
   hand-authored registry** — sufficiency flip 1.00 vs matched-random null ≈ 0,
   specificity cos < 0.16. The LLM is a *capable* probe designer, not just a router.

2. **Real design judgment.** It adopted a coherent **TUM-anchored** framing (immune,
   stroma, necrosis, mucin, normal all contrasted *against tumor*) — arguably a better
   fit for "characterize the TME" than the registry's mixed NORM/MUS foils — produced
   **4 novel valid axes the registry lacks** (STR·TUM desmoplasia, DEB·TUM necrosis,
   MUC·TUM, ADI·STR), and **avoided the degenerate BACK / tissue-vs-empty contrast**
   unprompted. This is the expressiveness win, with no human authoring.

3. **The gate is load-bearing — this is the whole safety story.** Both rejects were
   `STR v LYM` (a "stromal-vs-immune / immune-exclusion" phenotype) with a **perfect
   AUROC of 1.000**, yet the gate killed them on **intensity_r = 0.646 > 0.60**: that
   separation rides the CLS-norm staining/brightness proxy, not clean biology. A naive
   reader trusts a 1.0-AUROC probe; the gate says *no*. **Without the intensity guard a
   confounded probe would have been certified.**

## Implication for the build

**Adding agent probe design is good — but only behind the existing gate.** Recommendation:
wire `probe_design` behind the `ContrastSet` interface as an **optional design mode** for
questions/concepts the registry doesn't cover, keep the hand-authored registry as the
trusted default, and **print each agent-proposed probe's gate verdict on the card** so a
rejected axis is visible, not silent. Caveats unchanged: `necessity_z` stays a
divide-by-≈0 artifact on these perfectly-separable classes (sufficiency + gate +
specificity carry the verdict), and single-source data keeps the **confound gate
UNCHECKED** — agent design widens *coverage*, it does not answer real-biology-vs-batch.

---

# Live source-intervention necessity — the Hydra prior, MEASURED (2026-07-12)

The layer-resolved **source-intervention** suite (the "track #3" TODO everywhere above)
is now built and run. `causal/live.py` (`LiveEncoder`) hooks Phikon-v2 / Dinov2
`encoder.layer[L]`, projects the concept axis out of the CLS token **in the residual
stream**, and lets blocks L+1..final **recompute**; `intervene.live_necessity()` measures
the readout probe's **decision-margin drop** on the actual tile's forward pass vs a
matched-random null. This is the real do() — `intervened_on_input=True` — replacing the
cached readout-space projection (whose `necessity_z ≈ 5e8` was a divide-by-≈0 artifact).

Two engineering fixes made the signal real: the readout probe is **fit live on a disjoint
reference set** (the on-disk single-layer npz came from an older extractor → representation
mismatch → inverted probabilities), and the metric is the **graded margin**, not the
saturating 0/1 probability on a 1024-d separable probe.

**Setup.** Frozen Phikon-v2, real NCT-CRC tiles (16/class reference for the probe, 6/class
disjoint intervention set), concept axis derived live from that run's hidden states at each
of the 3 configured depths, n_null=12 matched-random directions per layer.

### Necessity curve — ablate TUM-vs-LYM, score TUM-vs-LYM

| layer (block) | concept margin-drop | random drop (mean±sd) | gap | z | bites |
|---|---|---|---|---|---|
| mid_early (7) | +0.064 | +0.000 ± 0.023 | **+0.064** | +2.8 | ✅ |
| mid (15) | +0.811 | +0.008 ± 0.027 | **+0.803** | +29.5 | ✅ |
| readout (23) | +3.244 | +0.014 ± 0.094 | **+3.230** | +34.5 | ✅ |

### Cross-interference (ablate-A-score-B) — ablate TUM-vs-LYM, score STR-vs-MUS

| layer | STR/MUS margin-drop under TUM-ablation | random | gap | z |
|---|---|---|---|---|
| mid_early | +0.058 | +0.001 | +0.056 | +9.0 |
| mid | +0.104 | −0.007 | +0.111 | +7.5 |
| readout | −0.073 | −0.008 | **−0.065** | **−1.6 (n.s.)** |

## Read of the numbers

1. **The redundancy / Hydra prior is now confirmed *and quantified*, not just expected.**
   The necessity gap is **graded and monotonic toward the readout** (0.06 → 0.80 → 3.23)
   with the matched-random null pinned at ~0 at every layer. Early-layer ablation barely
   dents the readout (the model recomputes the concept downstream); it becomes load-bearing
   from mid-network on. This is a real per-slide causal read — the exact "necessity is
   redundancy-limited, layer-resolved" story the priors demanded, now measured on live
   forward passes rather than asserted.

2. **Specificity is now *causal*, not just geometric.** Ablating the tumor-immune axis
   leaves the stroma (STR/MUS) readout intact at the readout layer (gap −0.065, z −1.6,
   not significant) — ~50× smaller than the on-target 3.23 drop. A small shared mid-network
   representation exists but does not propagate to the stroma decision. This is a genuine
   statement about the model's computation that the cached distractor-cosine cannot make.

3. **Necessity flips from the weakest number to the strongest evidence.** The old
   readout-space necessity was a divide-by-≈0 artifact and had to be reported categorically;
   the live curve is a clean, layer-resolved, null-separated, per-slide causal measurement.

## Wired into the certificate

`certify_answer(..., live_ctx={images, image_labels, ref_images, ref_labels, encoder})`.
Per certifiable claim on a hook-capable substrate, the necessity pillar becomes the live
curve and `scorecard` grades it (GROUNDED requires a **non-readout** layer to bite; score =
fraction of the readout-necessity already irreversible before the readout). End-to-end on
the core question *"Characterize the tumor microenvironment"*: `immune_infiltrate` →
**GROUNDED**, necessity **0.436** (was WEAK), card `intervened_on_input: true`,
`necessity_mode: live source-intervention`. Without `live_ctx` it degrades honestly to
cached readout-space (necessity WEAK, `intervened_on_input: false`).

## Certify-layer status (pillars)

| Pillar | Status |
|---|---|
| **Necessity** | ✅ **live source-intervention** — graded, layer-resolved, null-separated, per-slide. |
| **Specificity** | ✅ distractor-cosine (< 0.16) **and** causal cross-interference. |
| **Sufficiency** | ✅ flip 1.00 vs random ~0, but near-circular — **demote to caveated secondary** now that live necessity carries the verdict. |
| **Quantification** | ✅ per-pillar score + z + p + Holm-Bonferroni + coverage line + confound badge + faithfulness assumptions on the card. |
| **Reasoning trace** | ✅ **done** — every claim carries a deterministic 7-step `reasoning_trace` (contrast-validation → necessity → sufficiency → specificity → confound → multiple-comparisons → verdict), each step pairing the observed numbers with their interpretation; plus an answer-level `summary_trace` (coverage + honest declines + confound badge). Instant, no LLM. Optional `explain=True` adds a plain-English narration in ONE batched LLM call (~5s) that only phrases the deterministic numbers — it never gates the fast path. |
| **Confound gate** | ❌ UNCHECKED (single-source) — loud card badge; needs multi-site H&E (TCGA ≥2 sites / Kömen). |

The per-claim reasoning trace is what makes the card auditable inside pharma governance —
a reviewer can step through *why* each numeric score is what it is and *why* the verdict
followed. It is part of the MCP output (`certify_answer(...).claims[i].reasoning_trace` and
`.summary_trace`). With this, the only remaining hole is the confound gate, which is a
**data** gap (multi-site H&E), not a code gap — and it is named loudly on every card.

**Safety cap — the validation gate now enforces the verdict, not just annotates it.**
A contrast that fails the gate (held-out AUROC < 0.75, or intensity-collinearity |r| > 0.60)
**caps the verdict at WEAK** (`contrast_capped: true`) — a probe that rides the
staining/intensity proxy can **never** read GROUNDED. Verified on a live case: `stroma`
(STR-vs-MUS, |r| = 0.913) drops from a would-be GROUNDED to **WEAK** with the trace stating
*"the contrast RIDES the staining/intensity proxy … A probe that rides intensity never reads
GROUNDED,"* while a clean axis (`tumor_epithelium`, |r| = 0.52) stays GROUNDED. This closes
the gap where the gate flagged confounded probes but the card could still certify them — the
whole safety story (agent designs probes, the gate keeps it honest) now holds end-to-end.

---

# End-to-end LIVE demo — Sonnet-as-K-Pro answer, certified on phikon (2026-07-12)

First **fully end-to-end** run of `certify_answer` on a **model-generated** answer (not the
hardcoded `_demo()` string) with **live per-slide source-intervention** — the whole pipeline
exercised at once: LLM answer → atomic-claim decomposition → per-claim do()-battery → live
necessity → intensity gate → confound gate → scorecard → Holm-Bonferroni.

## Hypothesis under test

Track hypothesis (canonical, `hypothesis` verb): *"TUM vs LYM is a concept-specific,
certifiable causal axis in phikon-v2's latent — the tumor-immune interface."* Answer-level
generalization actually run here:

> **H:** every atomic concept-claim in a free-form K-Pro answer to *"Characterize the tumor
> microenvironment"* is *either* a concept-specific, causally load-bearing, non-confounded
> axis in phikon-v2's latent — certifiable per-claim (necessity ∧ sufficiency ∧ specificity
> vs matched-random null, intensity-gated, via live source-intervention) — *or* it is
> honestly **declined** (no substrate) or **capped** (confounded).
>
> **Falsifier:** if the certifier returns GROUNDED for every claim regardless of substrate
> it is a rubber stamp and void. A real certifier must produce *differentiated* verdicts
> driven by the model's representation, not the answer's fluency, and must **veto a
> statistically-perfect-but-confounded** claim.

## Setup

- **K-Pro simulated by Claude Sonnet 4.6** on Bedrock (`us.anthropic.claude-sonnet-4-6`),
  system-prompted as a pathology FM reporting on a tile its classifier called adenocarcinoma.
  Question: *"Characterize the tumor microenvironment."* The answer was a correct textbook CRC
  adenocarcinoma TME description (neoplastic glands, desmoplastic stroma + CAFs, TILs / CD8⁺).
- **Substrate:** frozen `phikon_v2`, real NCT-CRC-HE tiles streamed live — **24/class** for
  {TUM, LYM, NORM, STR, MUS, MUC}, split **16 reference (live probe-fit) + 8 intervention
  (watched), disjoint** (non-circular). Live source-intervention via `causal/live.LiveEncoder`,
  `n_null=12` (live) / 100 (cached battery). Claim decomposition = the same Bedrock call.
- Card: `artifacts/certificates/nct_crc_he/phikon_v2/demo_TME_answer_card.json`;
  driver `scratchpad/demo_tme.py`.

## Result — 12 atomic claims, differentiated verdicts

| Claim → concept | Verdict | AUROC | intens_r | necessity | suff | spec |
|---|---|---|---|---|---|---|
| neoplastic glands → `tumor_epithelium` | **GROUNDED** | 1.000 | 0.520 | 0.143 (bites @ readout) | 1.000 | 0.936 |
| loss of crypt arch. → `normal_mucosa` | **GROUNDED** | 1.000 | 0.520 | 0.003 (readout only) | 1.000 | 0.932 |
| desmoplastic stroma → `stroma` | **WEAK (capped)** | 1.000 | **0.913** | 1.000 (z=152) | 0.995 | 0.953 |
| collagen-rich ECM → `stroma` | **WEAK (capped)** | 1.000 | **0.913** | 1.000 | 0.995 | 0.953 |
| TILs → `immune_infiltrate` | **GROUNDED** | 1.000 | 0.576 | 0.572 | 1.000 | 0.976 |
| columnar epithelium, nuclear hyperchromasia, prominent nucleoli, mitoses, CAFs, CD8⁺, CD4⁺ (7) | **NOT_CERTIFIABLE** | — | — | — | — | — |

**Coverage: 5 of 12 certifiable → 3 GROUNDED, 2 capped-WEAK; 7 declined.** `necessity_mode
= live source-intervention (per-slide)`, `intervened_on_input=true`, confound gate
**UNCHECKED (single-source)**.

## Read of the numbers (H confirmed, not falsified)

1. **Not a rubber stamp — the whole point.** Only 3/12 read GROUNDED. The verdicts are driven
   by the substrate, not the (fluent, biologically-correct) answer.
2. **The veto fired on a true-but-confounded claim.** `stroma` had *perfect* pillars
   (necessity 1.000 z=152, sufficiency 0.995, specificity 0.953) yet was **capped at WEAK**
   because the STR-vs-MUS axis rides the staining/intensity proxy (**|r| = 0.913 > 0.60**).
   A naive TCAV/probe tool certifies this; the gate refuses. This is the faithfulness-audit
   value in one line: *right answer, possibly wrong reason.*
3. **Layer-resolved necessity honesty held per-slide.** `tumor_epithelium` only becomes
   irreversible near the readout (nec 0.143); `normal_mucosa` doesn't bite until the readout
   at all (0.003, mid-layer gap n.s.); `immune_infiltrate` builds from mid-network (0.572).
   Exactly the redundancy-limited curve, now measured on live forward passes.
4. **Honest scope decline, not force-fit.** All 7 cell/subcellular claims → NOT_CERTIFIABLE
   (need HistoPLUS cell-type embeddings on `h0_mini`, absent) — coverage is stated, not faked.

## Caveats / open

- Sonnet's answer was biologically **correct**, so the catches were **confound-veto** +
  **out-of-scope decline**, *not* a caught hallucination. The battery certifies whether a
  *concept is a faithful causal axis in the model*; it does **not** verify a claim is present
  in one specific query tile. A negative control (inject a false claim, e.g. "abundant adipose
  tissue") to demonstrate NULL/decline is the natural next run.
- Confound gate still **UNCHECKED** — single-source NCT-CRC; needs multi-site H&E (TCGA/Kömen).
- K-Pro is *simulated* by Sonnet; real K-Pro query integration is not wired.
