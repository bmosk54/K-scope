/**
 * BioLayer demo data — MOCK, structured to match the real `certify_answer()`
 * evidence-card schema emitted by `biolayer/dynamic/certify_answer.py` and the
 * `design()` / cross-model battery outputs in `docs/RESULTS.md`.
 *
 * TO GO LIVE: replace `CARD` below with the real JSON returned by the MCP
 * `certify_answer` tool (same shape: schema_version, prompt, answer, coverage,
 * claims[], not_certifiable[], guardrails, summary_trace, ...). Replace
 * `DESIGNED_PROBES` with a `design()` response, and `MODEL_COMPARISON` /
 * `MCP_VERBS` with whatever the live server reports. Nothing else in the app
 * needs to change — app.js reads only these three globals plus PIPELINE.
 *
 * Numbers here are taken from the project's own measured runs
 * (docs/RESULTS.md, "End-to-end LIVE demo" + "Live source-intervention
 * necessity" sections, 2026-07-12) — not invented from nothing, but this
 * specific object is illustrative wiring, not a live MCP response.
 */

// ---------------------------------------------------------------------------
// The core evidence card: certify_answer("Characterize the tumor
// microenvironment.") on frozen Phikon-v2 / NCT-CRC-HE.
// ---------------------------------------------------------------------------
window.CARD = {
  schema_version: "dyn-0.2",
  prompt: "Characterize the tumor microenvironment.",
  answer:
    "Tumor epithelium with a brisk peritumoral lymphocytic infiltrate, desmoplastic " +
    "stroma with collagen-rich ECM and cancer-associated fibroblasts, high-grade " +
    "columnar epithelium showing nuclear hyperchromasia, prominent nucleoli and " +
    "frequent mitoses, with scattered CD8+ and CD4+ T-cells.",
  track: "phikon",
  preferred_substrate: "phikon_v2",
  split: "train",
  coverage: {
    claims_total: 12,
    certifiable: 5,
    not_certifiable: 7,
    summary: "5 of 12 claims certifiable (3 GROUNDED, 2 WEAK-capped); 7 declined",
  },
  guardrails: {
    matched_random_null: true,
    specificity_control: true,
    confound_gate: "UNCHECKED (single-source data)",
    multiple_comparisons: "holm-bonferroni over 5 claims",
    intervened_on_input: true,
    necessity_mode: "live source-intervention (per-slide forward pass)",
    redundancy_honesty: "necessity reported layer-resolved / redundancy-limited",
  },
  assumptions: [
    "encoder-faithfulness: a latent do() moves the model's REPRESENTATION, not tissue biology",
    "K-Pro-faithfulness: K-Pro's pathology inference reads from Phikon/H0-family features the way this probe reads them (unverified from here)",
  ],
  caveat:
    "Certifies model-internal causal use of an answer's concept claims in the encoder's " +
    "representation — NOT biological/clinical validity.",

  claims: [
    {
      id: "neoplastic_glands",
      claim: "neoplastic glands",
      concept: "tumor_epithelium",
      contrast: "TUM vs NORM",
      polarity: "present",
      verdict: "GROUNDED",
      scores: { necessity: 0.143, sufficiency: 1.0, specificity: 0.936 },
      contrast_validation: {
        heldout_auroc: 1.0,
        intensity_collinearity: 0.52,
        valid: true,
        warnings: [],
      },
      confounded: false,
      contrast_capped: false,
      survives_multiple_comparisons: true,
      live_necessity: {
        intervened_on_input: true,
        curve: [
          { layer: "mid_early", gap: 0.03, z: 1.1, bites: false },
          { layer: "mid", gap: 0.11, z: 2.4, bites: true },
          { layer: "readout", gap: 0.86, z: 18.2, bites: true },
        ],
      },
      reasoning_trace: [
        { n: 1, step: "contrast_validation", observation: "held-out AUROC=1.000; intensity |r|=0.520 (gate: AUROC>=0.75, |r|<=0.60) -> PASS", interpretation: "the pool separates the concept and the axis is not riding a staining/brightness proxy" },
        { n: 2, step: "necessity_live", observation: "LIVE source-intervention on this slide's forward pass — margin-drop vs matched-random null by layer: mid_early:+0.03(z+1) -> mid:+0.11(z+2) -> readout:+0.86(z+18); null ~0 throughout.", interpretation: "the model's decision on this tile mostly recomputes tumor epithelium from un-ablated patch tokens until near the readout (redundancy / Hydra effect)" },
        { n: 3, step: "sufficiency", observation: "inject the concept direction: flip 1.00 vs matched-random 0.00 (score=1.000, z>=999).", interpretation: "a concept-specific steering axis — reported as a caveated secondary signal" },
        { n: 4, step: "specificity", observation: "ablate the orthogonal distractor axis (cos with concept = 0.094); target probe intact. score=0.936.", interpretation: "the effect is targeted to the concept axis, not general damage" },
        { n: 5, step: "confound", observation: "site-probe UNCHECKED — single-source data (no site/scanner variation).", interpretation: "cannot rule out that this is a batch/scanner artifact rather than biology; biological validity is NOT established here" },
        { n: 6, step: "multiple_comparisons", observation: "min pillar p=0.0002; Holm-Bonferroni across the answer's 5 claims -> survives.", interpretation: "the effect is not a cherry-pick from probing many concepts per answer" },
        { n: 7, step: "verdict", observation: "-> GROUNDED", interpretation: "necessity=WEAK(readout-only bite), sufficiency=GROUNDED, specificity=GROUNDED, survives correction, contrast passed the gate, not confounded -> the claim is causally load-bearing on this substrate" },
      ],
      notes: [
        "necessity via LIVE source-intervention: bites from mid-network, dominated by the readout layer",
        "confound gate UNAVAILABLE (single-source data) — biological validity not established",
      ],
    },
    {
      id: "loss_of_crypt_architecture",
      claim: "loss of crypt architecture",
      concept: "normal_mucosa",
      contrast: "NORM vs TUM",
      polarity: "absent",
      verdict: "GROUNDED",
      scores: { necessity: 0.003, sufficiency: 1.0, specificity: 0.932 },
      contrast_validation: {
        heldout_auroc: 1.0,
        intensity_collinearity: 0.52,
        valid: true,
        warnings: [],
      },
      confounded: false,
      contrast_capped: false,
      survives_multiple_comparisons: true,
      live_necessity: {
        intervened_on_input: true,
        curve: [
          { layer: "mid_early", gap: 0.0, z: 0.1, bites: false },
          { layer: "mid", gap: 0.01, z: 0.4, bites: false },
          { layer: "readout", gap: 0.79, z: 12.6, bites: true },
        ],
      },
      reasoning_trace: [
        { n: 1, step: "contrast_validation", observation: "held-out AUROC=1.000; intensity |r|=0.520 -> PASS", interpretation: "clean separation, not an intensity proxy" },
        { n: 2, step: "necessity_live", observation: "margin-drop by layer: mid_early:+0.00(z+0.1) -> mid:+0.01(z+0.4) -> readout:+0.79(z+12.6); null ~0 throughout.", interpretation: "does not bite until the readout at all — near-tautological necessity, honestly reported" },
        { n: 3, step: "sufficiency", observation: "inject the concept direction: flip 1.00 vs matched-random 0.00.", interpretation: "concept-specific steering axis" },
        { n: 4, step: "specificity", observation: "ablate the orthogonal distractor axis (cos with concept = 0.11); target probe intact. score=0.932.", interpretation: "effect is targeted, not general damage" },
        { n: 5, step: "confound", observation: "site-probe UNCHECKED — single-source data.", interpretation: "biological validity not established" },
        { n: 6, step: "multiple_comparisons", observation: "min pillar p=0.0006 -> survives Holm-Bonferroni.", interpretation: "not a cherry-pick" },
        { n: 7, step: "verdict", observation: "-> GROUNDED", interpretation: "certifies normal mucosa displacement, with an honest readout-only necessity caveat" },
      ],
      notes: ["necessity is readout-only — reported honestly, does not downgrade sufficiency"],
    },
    {
      id: "tils",
      claim: "brisk peritumoral lymphocytic infiltrate (TILs)",
      concept: "immune_infiltrate",
      contrast: "LYM vs TUM",
      polarity: "present",
      verdict: "GROUNDED",
      scores: { necessity: 0.572, sufficiency: 1.0, specificity: 0.976 },
      contrast_validation: {
        heldout_auroc: 1.0,
        intensity_collinearity: 0.576,
        valid: true,
        warnings: [],
      },
      confounded: false,
      contrast_capped: false,
      survives_multiple_comparisons: true,
      live_necessity: {
        intervened_on_input: true,
        curve: [
          { layer: "mid_early", gap: 0.064, z: 2.8, bites: true },
          { layer: "mid", gap: 0.811, z: 29.5, bites: true },
          { layer: "readout", gap: 3.244, z: 34.5, bites: true },
        ],
        cross_interference: [
          { layer: "mid_early", gap: 0.056, z: 9.0 },
          { layer: "mid", gap: 0.111, z: 7.5 },
          { layer: "readout", gap: -0.065, z: -1.6 },
        ],
      },
      reasoning_trace: [
        { n: 1, step: "contrast_validation", observation: "held-out AUROC=1.000; intensity |r|=0.576 -> PASS", interpretation: "clean separation" },
        { n: 2, step: "necessity_live", observation: "margin-drop by layer: mid_early:+0.06(z+2.8) -> mid:+0.81(z+29.5) -> readout:+3.24(z+34.5); null ~0 throughout.", interpretation: "the model's decision causally depends on the concept from mid-network on — the strongest per-slide causal read in this answer" },
        { n: 3, step: "sufficiency", observation: "inject the concept direction: flip 1.00 vs matched-random 0.00.", interpretation: "concept-specific steering axis" },
        { n: 4, step: "specificity", observation: "ablate the orthogonal distractor axis (cos with concept = 0.05); target probe intact. score=0.976. Cross-interference check: ablating THIS axis leaves the STR/MUS readout intact at the readout layer (gap -0.065, z -1.6, not significant) — ~50x smaller than the on-target 3.24 drop.", interpretation: "specificity is now causal, not just geometric — a genuine statement about the model's computation" },
        { n: 5, step: "confound", observation: "site-probe UNCHECKED — single-source data.", interpretation: "biological validity not established" },
        { n: 6, step: "multiple_comparisons", observation: "min pillar p<0.0001 -> survives Holm-Bonferroni.", interpretation: "not a cherry-pick" },
        { n: 7, step: "verdict", observation: "-> GROUNDED", interpretation: "best demo claim: live source-intervention bites before the readout and the random null stays flat throughout" },
      ],
      notes: ["flagship claim for the demo — graded, layer-resolved, null-separated, per-slide"],
    },
    {
      id: "desmoplastic_stroma",
      claim: "desmoplastic stroma",
      concept: "stroma",
      contrast: "STR vs MUS",
      polarity: "present",
      verdict: "WEAK",
      scores: { necessity: 1.0, sufficiency: 0.995, specificity: 0.953 },
      contrast_validation: {
        heldout_auroc: 1.0,
        intensity_collinearity: 0.913,
        valid: false,
        warnings: ["concept axis collinear with intensity proxy (|r|=0.91)"],
      },
      confounded: false,
      contrast_capped: true,
      survives_multiple_comparisons: true,
      live_necessity: {
        intervened_on_input: true,
        curve: [
          { layer: "mid_early", gap: 0.41, z: 6.2, bites: true },
          { layer: "mid", gap: 0.9, z: 40.1, bites: true },
          { layer: "readout", gap: 1.35, z: 152.0, bites: true },
        ],
      },
      reasoning_trace: [
        { n: 1, step: "contrast_validation", observation: "held-out AUROC=1.000; intensity |r|=0.913 (gate: |r|<=0.60) -> WARN: concept axis collinear with intensity proxy", interpretation: "the probe may be reading intensity/staining, not biology — treat with caution" },
        { n: 2, step: "necessity_live", observation: "margin-drop by layer: mid_early:+0.41(z+6.2) -> mid:+0.90(z+40.1) -> readout:+1.35(z+152); null ~0 throughout.", interpretation: "on paper this is the strongest necessity signal in the whole answer" },
        { n: 3, step: "sufficiency", observation: "inject the concept direction: flip 0.995 vs matched-random 0.00.", interpretation: "looks like a clean, concept-specific steering axis" },
        { n: 4, step: "specificity", observation: "ablate the orthogonal distractor axis; target probe intact. score=0.953.", interpretation: "looks specific too" },
        { n: 5, step: "confound", observation: "site-probe UNCHECKED — single-source data.", interpretation: "biological validity not established" },
        { n: 6, step: "multiple_comparisons", observation: "min pillar p<0.0001 -> survives Holm-Bonferroni.", interpretation: "not a cherry-pick" },
        { n: 7, step: "verdict", observation: "-> WEAK (CAPPED: contrast failed the validation gate)", interpretation: "the pillars separate perfectly, but the contrast RIDES the staining/intensity proxy (|r|=0.913>0.60) — so the axis is not certified as clean biology. A probe that rides intensity never reads GROUNDED, no matter how good the numbers look." },
      ],
      notes: [
        "THE veto moment: perfect-looking pillars, capped anyway because of the intensity gate",
      ],
    },
    {
      id: "collagen_rich_ecm",
      claim: "collagen-rich ECM",
      concept: "stroma",
      contrast: "STR vs MUS",
      polarity: "present",
      verdict: "WEAK",
      scores: { necessity: 1.0, sufficiency: 0.995, specificity: 0.953 },
      contrast_validation: {
        heldout_auroc: 1.0,
        intensity_collinearity: 0.913,
        valid: false,
        warnings: ["concept axis collinear with intensity proxy (|r|=0.91)"],
      },
      confounded: false,
      contrast_capped: true,
      survives_multiple_comparisons: true,
      live_necessity: {
        intervened_on_input: true,
        curve: [
          { layer: "mid_early", gap: 0.41, z: 6.2, bites: true },
          { layer: "mid", gap: 0.9, z: 40.1, bites: true },
          { layer: "readout", gap: 1.35, z: 152.0, bites: true },
        ],
      },
      reasoning_trace: [
        { n: 1, step: "contrast_validation", observation: "same STR-vs-MUS axis as 'desmoplastic stroma' -> WARN: rides intensity (|r|=0.913)", interpretation: "two different textual claims sharing one capped axis — the UI should make that sharing visible" },
        { n: 7, step: "verdict", observation: "-> WEAK (CAPPED: contrast failed the validation gate)", interpretation: "same cap as the stroma claim above; repeated claims on one axis inherit its verdict" },
      ],
      notes: ["shares the capped stroma axis with 'desmoplastic stroma'"],
    },

    // --- declined: no substrate / out of scope, listed individually -------
    { id: "columnar_epithelium", claim: "columnar epithelium", concept: null, verdict: "NOT_CERTIFIABLE", reason: "cell/subcellular morphology — needs HistoPLUS/H0-mini nucleus-level embeddings, not present" },
    { id: "nuclear_hyperchromasia", claim: "nuclear hyperchromasia", concept: null, verdict: "NOT_CERTIFIABLE", reason: "cell/subcellular morphology — needs HistoPLUS/H0-mini nucleus-level embeddings, not present" },
    { id: "prominent_nucleoli", claim: "prominent nucleoli", concept: null, verdict: "NOT_CERTIFIABLE", reason: "cell/subcellular morphology — needs HistoPLUS/H0-mini nucleus-level embeddings, not present" },
    { id: "mitoses", claim: "frequent mitoses", concept: null, verdict: "NOT_CERTIFIABLE", reason: "mitotic-figure detection is a cell-level HistoPLUS concept — aspirational, unlocks with extraction" },
    { id: "cafs", claim: "cancer-associated fibroblasts (CAFs)", concept: null, verdict: "NOT_CERTIFIABLE", reason: "fibroblast cell-type concept — HistoPLUS substrate not extracted yet" },
    { id: "cd8", claim: "scattered CD8+ T-cells", concept: null, verdict: "NOT_CERTIFIABLE", reason: "immunophenotype/cell-population concept — out of scope for tile-level tissue labels" },
    { id: "cd4", claim: "scattered CD4+ T-cells", concept: null, verdict: "NOT_CERTIFIABLE", reason: "immunophenotype/cell-population concept — out of scope for tile-level tissue labels" },
  ],

  summary_trace: [
    { n: 1, step: "coverage", observation: "5 of 12 claims certifiable (3 GROUNDED, 2 WEAK-capped); 7 declined", interpretation: "claims of different epistemic status were asserted in identical prose; the tool separates the testable from the untestable" },
    { n: 2, step: "declined", observation: "7 cell/subcellular claims declined — needs HistoPLUS/H0-mini nucleus-level embeddings, absent today", interpretation: "declined rather than force-fit — an honest NOT_CERTIFIABLE" },
    { n: 3, step: "confound_badge", observation: "UNCHECKED (single-source data)", interpretation: "the one honest hole named up front; biological validity beyond model-internal use rests on this + encoder faithfulness" },
  ],
};

// ---------------------------------------------------------------------------
// Agent probe design run — Sonnet proposes contrasts for the open question,
// the deterministic gate decides certifiability. (docs/RESULTS.md, "Agent
// probe design" section, 2026-07-12.)
// ---------------------------------------------------------------------------
window.DESIGNED_PROBES = {
  question: "Characterize the tumor microenvironment.",
  designed_by: "claude-sonnet-4-6 (bedrock)",
  n_probes: 8,
  probes: [
    { concept: "immune infiltration in tumor", contrast: "LYM vs TUM", auroc: 1.0, intensity_r: 0.237, gate: "PASS", sufficiency: 1.0, random_null: 0.01 },
    { concept: "desmoplastic stromal reaction", contrast: "STR vs TUM", auroc: 1.0, intensity_r: 0.581, gate: "PASS", sufficiency: 1.0, random_null: 0.03 },
    { concept: "stromal vs immune composition", contrast: "STR vs LYM", auroc: 1.0, intensity_r: 0.646, gate: "REJECT", sufficiency: 1.0, random_null: 0.0 },
    { concept: "tumor necrosis burden", contrast: "DEB vs TUM", auroc: 0.999, intensity_r: 0.128, gate: "PASS", sufficiency: 1.0, random_null: 0.01 },
    { concept: "mucinous microenvironment", contrast: "MUC vs TUM", auroc: 0.996, intensity_r: 0.426, gate: "PASS", sufficiency: 1.0, random_null: 0.0 },
    { concept: "peritumoral immune exclusion", contrast: "LYM vs STR", auroc: 1.0, intensity_r: 0.646, gate: "REJECT", sufficiency: 1.0, random_null: 0.0 },
    { concept: "normal mucosa displacement", contrast: "NORM vs TUM", auroc: 1.0, intensity_r: 0.172, gate: "PASS", sufficiency: 1.0, random_null: 0.0 },
    { concept: "adipose infiltration of stroma", contrast: "ADI vs STR", auroc: 1.0, intensity_r: 0.229, gate: "PASS", sufficiency: 1.0, random_null: 0.0 },
  ],
  cap_threshold: 0.6,
  note:
    "Both rejects share the SAME phenotype (stromal-vs-immune / immune-exclusion) and the SAME " +
    "failure mode: perfect AUROC (1.000) riding the staining/intensity proxy (|r|=0.646>0.60). " +
    "Without the intensity guard a confounded probe would have been certified.",
};

// ---------------------------------------------------------------------------
// Cross-model readout battery (docs/RESULTS.md, "Measured results" section).
// ---------------------------------------------------------------------------
window.MODEL_COMPARISON = [
  {
    model: "Phikon-v2",
    dim: 1024,
    probe_acc: 1.0,
    necessity_concept: "1.000 -> 0.500 (chance)",
    necessity_random: "1.000 ± 0.000",
    sufficiency_concept: 1.0,
    sufficiency_random: 0.0,
    specificity_cos: 0.02,
  },
  {
    model: "H-optimus-0",
    dim: 1536,
    probe_acc: 0.998,
    necessity_concept: "0.998 -> 0.500 (chance)",
    necessity_random: "0.998 ± 0.000",
    sufficiency_concept: 1.0,
    sufficiency_random: 0.004,
    specificity_cos: 0.008,
  },
];

// ---------------------------------------------------------------------------
// MCP verb surface (biolayer/mcp/server.py) — what powers each UI panel.
// ---------------------------------------------------------------------------
window.MCP_VERBS = [
  { verb: "certify_answer", powers: "the whole evidence card: claims, verdicts, traces", status: "ship" },
  { verb: "design", powers: "agent probe-design workbench", status: "ship" },
  { verb: "rehypothesize", powers: "closed-loop 'next hypothesis' panel", status: "ship" },
  { verb: "steer_from_card / ablate_from_card", powers: "zero-recompute steer/ablate demo buttons", status: "ship" },
  { verb: "layered / attribution", powers: "layer curve + patch heat overlay", status: "partial" },
  { verb: "confound", powers: "confound gate badge", status: "data gap" },
  { verb: "biolayer.mil (stretch)", powers: "slide-level aggregation branch", status: "stretch" },
];

// ---------------------------------------------------------------------------
// Pipeline diagram — structurally modeled on docs/DESIGN_MIL_AGGREGATOR.md's
// ASCII pipeline (§3), adapted to the tile-level certify path instead of the
// slide-level MIL aggregator it documents.
// ---------------------------------------------------------------------------
window.PIPELINE = [
  { id: "tile", label: "H&E tile", detail: "224px NCT-CRC-HE, 9 native tissue classes" },
  { id: "encode", label: "Frozen encoder", detail: "Phikon-v2 ViT-L/24, hook encoder.layer[L]" },
  { id: "probe", label: "Concept probe", detail: "diff-of-means axis, held-out AUROC + intensity gate" },
  { id: "hook", label: "Live do()", detail: "project axis out of CLS at layer L, let L+1..final recompute" },
  { id: "null", label: "Matched-random null", detail: "same edit, n random unit directions — must stay inert" },
  { id: "score", label: "Scorecard", detail: "necessity / sufficiency / specificity vs null, Holm-Bonferroni" },
  { id: "card", label: "Evidence card", detail: "GROUNDED / WEAK / NOT_CERTIFIABLE + reasoning trace" },
];

// ---------------------------------------------------------------------------
// Substrate registry — docs/ARCHITECTURE.md §3. All three tracked encoders,
// not just the two with a live battery run, so the coverage story is honest
// about what's wired vs. extract-only.
// ---------------------------------------------------------------------------
window.TRACKS = [
  { track: "phikon", model: "owkin/phikon-v2", gated: false, backend: "transformers", dim: 1024, blocks: 24, layers: "8 / 16 / 24", objective: "TUM vs LYM", status: "live battery" },
  { track: "h0", model: "bioptimus/H0-mini", gated: true, backend: "timm", dim: 768, blocks: 12, layers: "3 / 7 / 11", objective: "TUM vs NORM", status: "live battery" },
  { track: "—", model: "bioptimus/H-optimus-0", gated: true, backend: "timm", dim: 1536, blocks: 40, layers: "13 / 27 / 39", objective: "—", status: "extract-only" },
];

// ---------------------------------------------------------------------------
// Module map — docs/ARCHITECTURE.md §4, one card per path.
// ---------------------------------------------------------------------------
window.MODULES = [
  { path: "biolayer/config.py", role: "Model registry, S3 key layout, dataset/split/class constants", group: "substrate" },
  { path: "biolayer/tracks/", role: "Per-track bundles (model + dataset + objective + layers) — phikon and h0 never share assumptions", group: "substrate" },
  { path: "biolayer/data/models.py", role: "Frozen encoder loading; multi-layer local+global embed()", group: "substrate" },
  { path: "biolayer/data/extract.py", role: "CLI: tile → .npz embeddings, optional S3 upload", group: "substrate" },
  { path: "biolayer/data/s3_utils.py", role: "Shared S3 artifact channel", group: "substrate" },
  { path: "biolayer/causal/", role: "The battery: probe · intervene · battery · confound · attribution · live.py (source-intervention) · certify.py (card assembly)", group: "battery" },
  { path: "biolayer/dynamic/", role: "Answer-bound path: claims.py (decompose) · concepts.py (registry) · contrast.py (validation gate) · probe_design.py (agent) · scorecard.py · trace.py · certify_answer.py", group: "battery" },
  { path: "biolayer/mcp/", role: "MCP server + verbs + card — the certify_answer / design / rehypothesize interface this dashboard reads from", group: "interface" },
  { path: "biolayer/mil/", role: "Stretch: slide-level aggregation by reusing a ViT's final block (docs/DESIGN_MIL_AGGREGATOR.md)", group: "stretch" },
];

// ---------------------------------------------------------------------------
// Compute & infra — docs/ARCHITECTURE.md §5.
// ---------------------------------------------------------------------------
window.INFRA = [
  {
    title: "Auth",
    body: "Locally: a workspace-scoped terminal profile sources .owkin_hack_aws.sh (gitignored) + activates owkin-env. On the box: the SageMaker execution role provides S3/GPU auth — no long-lived keys.",
  },
  {
    title: "Storage",
    body: "s3://bucketbiolayer/ with per-dataset/per-model prefixes: embeddings/, directions/, sae/, certificates/. Current role is ListBucket-only, so embeddings regenerate locally (--no-upload) and are gitignored (*.npz, artifacts/) until the write policy is fixed.",
  },
  {
    title: "GPU",
    body: "H-optimus-0 (ViT-g/14) needs a GPU — SageMaker ml.g5.2xlarge (A10G, ~22GiB), Studio JupyterLab or a Training Job (quota = 1 each). EKS was evaluated and dropped: 0 EC2 G/VT and 0 HyperPod g5 quota on the account, and a single extraction job + a stdio MCP server need no orchestration anyway.",
  },
];

// ---------------------------------------------------------------------------
// MIL aggregator (stretch) — docs/DESIGN_MIL_AGGREGATOR.md. Presented as an
// open design, NOT a measured result — §8 of that doc lists this as an open
// question. No invented performance numbers here.
// ---------------------------------------------------------------------------
window.MIL = {
  status: "designed, not run this weekend — open question in DESIGN_MIL_AGGREGATOR.md §8",
  claim:
    "A ViT block has no positional term, so running one block over a bag of pre-extracted tile CLS vectors " +
    "treats them as an unordered SET — exactly the inductive bias a MIL slide-aggregator needs.",
  pipeline: [
    { label: "N tiles", detail: "pre-extracted CLS features, [N, 1536]" },
    { label: "+ slide-CLS", detail: "prepend a learnable query token → [N+1, 1536]" },
    { label: "blocks[-1]", detail: "reused, frozen — permutation-invariant set attention" },
    { label: "row 0", detail: "read out as the slide embedding [1, 1536]" },
    { label: "linear head", detail: "the only always-trained surface besides the CLS token" },
  ],
  controls: [
    { name: "Frozen-block (default)", trainable: "slide-CLS token + Linear(1536→C)", when: "few slides; safest" },
    { name: "LoRA / unfreeze last block", trainable: "+ blocks[-1] weights", when: "more slides; if frozen underfits" },
  ],
  nonNegotiable: "matched-random baseline: mean-pool of the SAME tile features → same head. If attention doesn't beat mean-pool, the reused block earns nothing — and it gets reported that way, not hidden.",
  openQuestions: [
    "Does blocks[-1] frozen actually beat mean-pool on NCT-CRC slide-style bags?",
    "Register-token count for H-optimus-0 — confirm num_prefix_tokens empirically.",
    "Feed pre-final-norm hidden states instead of post-norm CLS to close the distribution gap?",
    "Slide labels: NCT-CRC is tile-labeled — synthetic bags from tile labels, or wait for TCGA slide labels?",
  ],
};

// ---------------------------------------------------------------------------
// Confound gate — the differentiator per CLAUDE.md / STRATEGY.md, presented
// honestly as UNCHECKED this weekend (NCT-CRC is single-source). The
// distributions below are an ILLUSTRATIVE SCHEMATIC of what a Kömen-style
// site-probe would look like on multi-site data — not a measured result.
// ---------------------------------------------------------------------------
window.CONFOUND = {
  status: "UNCHECKED",
  reason: "NCT-CRC-HE is single-source, single-scanner data — there is no site/batch variation to probe against this weekend.",
  question: "Is a GROUNDED claim's causal signal real biology, or a site/scanner artifact riding the same encoder axis?",
  method:
    "Kömen-style site-probe: train a linear classifier to predict SITE (not concept) from the same activations the concept axis " +
    "lives in. If the concept direction is highly correlated with the site-predictive subspace, the claim is confounded — " +
    "GROUNDED gets demoted regardless of how clean the necessity/sufficiency/specificity pillars look.",
  unlocks: "TCGA multi-site cohorts, or the Kömen 2024 multi-scanner setup — either gives the site labels this needs.",
  illustrative_note: "Schematic only — illustrates the check, not a measured result (no multi-site data wired yet).",
  illustrative_sites: [
    { site: "Site A (scanner 1)", n: 640, mean: -0.34, spread: 0.42 },
    { site: "Site B (scanner 2)", n: 512, mean: 0.28, spread: 0.38 },
  ],
  illustrative_site_probe_auroc: 0.52,
  gate_rule: "if site-probe AUROC on the concept axis > 0.65 → demote GROUNDED to WEAK regardless of pillar scores",
};

// ---------------------------------------------------------------------------
// rehypothesize() cycle — candidate "what to probe next" suggestions an
// agent would propose after seeing the current card (docs/RESULTS.md).
// ---------------------------------------------------------------------------
window.NEXT_HYPOTHESES = [
  { hypothesis: "Does the immune_infiltrate axis generalize to a held-out site, or is it also riding intensity like stroma did?", basedOn: "stroma got capped on intensity — check the flagship claim isn't next", priority: "high" },
  { hypothesis: "Run confound gate on immune_infiltrate the moment multi-site labels exist — it's the highest-necessity claim, worth protecting first.", basedOn: "necessity=0.572 is the strongest live-intervention read in the card", priority: "high" },
  { hypothesis: "Probe 'mucin pools' (MUC vs DEB) — adjacent to the mucinous-microenvironment probe that already passed the gate.", basedOn: "design() surfaced mucinous microenvironment as PASS; neighboring concept untested", priority: "medium" },
  { hypothesis: "Re-run desmoplastic stroma with a de-staining-normalized contrast pool to see if the intensity cap clears.", basedOn: "stroma WEAK-capped at |r|=0.913 — test whether normalization rescues it", priority: "medium" },
  { hypothesis: "Extract HistoPLUS embeddings for one demo slide to unlock 2 of the 7 declined claims (CAFs, mitoses).", basedOn: "7 declined claims share one root cause: missing cell-level substrate", priority: "low" },
];
