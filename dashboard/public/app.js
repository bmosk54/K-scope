/**
 * BioLayer dashboard rendering. Reads window.CARD / DESIGNED_PROBES /
 * TRACKS / MIL / CONFOUND from data.js. No build step.
 *
 * Structure: 3 views (Case / Proof / Verdict) sharing one persistent
 * claim-strip. Selecting a claim anywhere re-renders all three so tabs
 * stay in sync.
 */
(function () {
  "use strict";

  const COLOR = {
    GROUNDED: "#3ddc97",
    WEAK: "#e8b23e",
    NOT_CERTIFIABLE: "#7a7fa0",
    accent: "#6c8dfb",
    danger: "#ef6572",
  };

  let certifiable = window.CARD.claims.filter((c) => !!c.scores);
  let declinedCount = window.CARD.claims.filter((c) => !c.scores).length;
  let selectedIdx = certifiable.findIndex((c) => c.id === "tils");
  if (selectedIdx < 0) selectedIdx = 0;

  // Answer-Flow (Sankey + claim rail) state
  let railSelectedId = null;   // id of the rail row that's highlighted (certifiable or declined)
  let sankeyRaf = null;        // handle for the drifting-particle animation loop

  // Backend base URL. Default "" = same-origin (works when the whole Node server is
  // port-forwarded to localhost). Override to point the UI at a separately-forwarded
  // SageMaker API: set window.API_BASE, or open the page with ?api=http://localhost:4173
  const API_BASE = (window.API_BASE ||
    new URLSearchParams(location.search).get("api") || "").replace(/\/+$/, "");
  const apiUrl = (p) => (API_BASE ? API_BASE + "/" + p : p);

  // recompute the derived claim state after a live CARD override (fetch / Run button)
  function recomputeState() {
    certifiable = window.CARD.claims.filter((c) => !!c.scores);
    declinedCount = window.CARD.claims.filter((c) => !c.scores).length;
    selectedIdx = Math.min(selectedIdx, Math.max(0, certifiable.length - 1));
    railSelectedId = null; // re-derive from the fresh card on next rail render
  }

  // ---------------------------------------------------------------- utils
  function mulberry32(seed) {
    let a = seed;
    return function () {
      a |= 0;
      a = (a + 0x6d2b79f5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  function hashStr(s) {
    let h = 0;
    for (let i = 0; i < s.length; i++) h = (Math.imul(h, 31) + s.charCodeAt(i)) | 0;
    return h;
  }
  function pct(x, digits) {
    if (x === null || x === undefined) return "—";
    return (x * 100).toFixed(digits === undefined ? 0 : digits) + "%";
  }
  function currentClaim() {
    return certifiable[selectedIdx];
  }

  const tooltip = d3.select("#tooltip");
  function showTip(event, html) {
    tooltip.html(html).style("display", "block").style("opacity", 1);
    moveTip(event);
  }
  function moveTip(event) {
    const w = window.innerWidth, h = window.innerHeight;
    tooltip
      .style("left", Math.min(event.clientX + 16, w - 340) + "px")
      .style("top", Math.min(event.clientY + 16, h - 240) + "px");
  }
  function hideTip() {
    tooltip.style("opacity", 0).style("display", "none");
  }

  // ---------------------------------------------------------------- claim strip
  function renderClaimStrip() {
    const claim = currentClaim();
    document.getElementById("strip-idx").textContent = `${selectedIdx + 1} / ${certifiable.length}`;
    document.getElementById("strip-name").textContent = claim.claim;
    document.getElementById("strip-axis").textContent = `${claim.concept.replace(/_/g, " ")} · ${claim.contrast}`;
    const v = document.getElementById("strip-verdict");
    v.className = "strip-verdict " + claim.verdict;
    v.textContent = claim.verdict;
    // Caption reflects the actual declines, not a hardcoded "cell/subcellular" assumption.
    const declEl = document.getElementById("strip-declined");
    declEl.textContent = declinedCount ? `+ ${declinedCount} declined` : "";
  }

  function selectClaim(idx) {
    selectedIdx = ((idx % certifiable.length) + certifiable.length) % certifiable.length;
    railSelectedId = currentClaim().id;
    hideTip();
    renderClaimStrip();
    renderHistology();
    renderQuadrantSelection();
    renderNecessityChart();
    renderVerdict();
    highlightRail();
  }
  function selectClaimById(id) {
    const idx = certifiable.findIndex((c) => c.id === id);
    if (idx >= 0) selectClaim(idx);
  }

  // ---------------------------------------------------------------- histology theater (tissue layout)
  const TISSUE_LAYOUT = (function build() {
    const rand = mulberry32(42);
    const centers = [
      { x: 0.27, y: 0.32, r: 0.155 },
      { x: 0.64, y: 0.24, r: 0.135 },
      { x: 0.46, y: 0.7, r: 0.175 },
    ];
    const blobs = centers.map((c) => {
      const pts = [];
      const n = 14;
      for (let i = 0; i < n; i++) {
        const a = (i / n) * Math.PI * 2;
        const jitter = 0.72 + rand() * 0.56;
        pts.push([c.x + Math.cos(a) * c.r * jitter, c.y + Math.sin(a) * c.r * jitter]);
      }
      return { center: c, pts };
    });
    const nuclei = [];
    blobs.forEach((b) => {
      for (let i = 0; i < 46; i++) {
        const a = rand() * Math.PI * 2;
        const r = Math.sqrt(rand()) * b.center.r * 0.92;
        nuclei.push({ x: b.center.x + Math.cos(a) * r, y: b.center.y + Math.sin(a) * r, rr: 2.2 + rand() * 1.6 });
      }
    });
    const stromaCells = [];
    for (let i = 0; i < 160; i++) {
      const x = rand(), y = rand();
      const insideGland = centers.some((c) => Math.hypot(x - c.x, y - c.y) < c.r * 1.02);
      if (!insideGland) stromaCells.push({ x, y, rot: rand() * 360, len: 4 + rand() * 5 });
    }
    const lymphs = [];
    for (let i = 0; i < 90; i++) {
      const nearGland = centers[Math.floor(rand() * centers.length)];
      const a = rand() * Math.PI * 2;
      const r = nearGland.r * (0.9 + rand() * 0.45);
      lymphs.push({ x: nearGland.x + Math.cos(a) * r, y: nearGland.y + Math.sin(a) * r, rr: 1.6 + rand() * 1.1 });
    }
    return { centers, blobs, nuclei, stromaCells, lymphs };
  })();

  // NOTE: the Case tissue render is a MEASURED per-patch causal heatmap, precomputed by
  // dashboard/precompute_heatmaps.py. For a real NCT-CRC tile we re-forward it through
  // frozen Phikon-v2, capture the 14x14 patch-token grid at the readout layer (the grid
  // models.py normally mean-pools away), project each patch onto the concept axis, and
  // z-score that projection against a matched-random-direction null (Section-5-D control).
  // window.HEATMAPS[concept] = { tile, z_grid 14x14, norm_grid, top_z, verdict, ... }.
  // Concepts with no substrate (declined claims) get an honest "no map" panel — never a
  // fabricated schematic.
  let overlayOn = true;

  function renderReadoutStatic(claim) {
    const el = document.getElementById("readout-static");
    const pillars = [
      ["necessity", claim.scores.necessity],
      ["sufficiency", claim.scores.sufficiency],
      ["specificity", claim.scores.specificity],
    ];
    const cv = claim.contrast_validation;
    const num = (v, d) => (typeof v === "number" ? v.toFixed(d) : "—");
    el.innerHTML =
      `<div class="ro-group"><div class="ro-label">Axis</div><div class="ro-axis-value">${claim.contrast || "—"}</div></div>` +
      `<div class="ro-group"><div class="ro-label">Causal pillars</div>` +
      pillars
        .map(
          ([name, v]) =>
            `<div class="ro-pillar"><div class="ro-pillar-name">${name}</div><div class="ro-pillar-track"><div class="ro-pillar-fill" style="width:${pct(v, 0)};background:${COLOR[claim.verdict]}"></div></div><div class="ro-pillar-value">${num(v, 3)}</div></div>`
        )
        .join("") +
      `</div>` +
      // contrast_validation is optional on live claims — degrade to a neutral row, never throw.
      (cv
        ? `<div class="ro-group"><div class="ro-label">Contrast gate</div>` +
          `<div class="ro-gate-row"><span class="badge ${cv.valid ? "PASS" : "CAPPED"}">${cv.valid ? "PASS" : "CAPPED"}</span><span class="ro-gate-detail">AUROC ${num(cv.heldout_auroc, 3)} · |r| ${num(cv.intensity_collinearity, 3)} (cap 0.60)</span></div>` +
          ((cv.warnings && cv.warnings.length) ? `<div class="ro-gate-warn">⚠ ${cv.warnings.join("; ")}</div>` : "") +
          `</div>`
        : "");
  }

  // The real reasoning trace certify emitted for this claim — every battery step, verbatim.
  // Replaces the old fabricated per-patch readout (seeded-random field + hashed neuron #),
  // which implied a spatial signal the pooled-CLS substrate does not have.
  function renderReasoningTrace(claim) {
    const el = document.getElementById("readout-live");
    const trace = (claim.reasoning_trace || []).filter((t) => t.step !== "verdict");
    el.innerHTML =
      `<div class="ro-label">Reasoning trace <span class="ro-label-sub">— live from the substrate</span></div>` +
      (trace.length
        ? trace
            .map(
              (t) =>
                `<div class="ro-trace-step"><div class="ro-trace-name">${(t.step || "").replace(/_/g, " ")}</div>` +
                `<div class="ro-trace-obs">${t.observation || ""}</div></div>`
            )
            .join("")
        : `<div class="live-empty">No reasoning trace recorded for this claim.</div>`);
  }

  function renderHistology() {
    const claim = currentClaim();
    const container = document.getElementById("tile-stage");
    container.innerHTML = "";
    const hm = claim && claim.concept && (window.HEATMAPS || {})[claim.concept];
    if (hm) renderCausalMap(container, claim, hm);
    else renderNoMap(container, claim);
    renderReadoutStatic(claim);
    renderReasoningTrace(claim);
  }

  // A real tile + measured per-patch causal heatmap (14x14 patch-projection z vs a
  // matched-random-direction null). Cold patches are transparent so tissue shows through;
  // hot patches are the ones whose readout-layer activation carries the concept axis.
  function renderCausalMap(container, claim, hm) {
    const S = hm.grid_side || (hm.z_grid || []).length || 14;
    const CELL = 224 / S;
    const svg = d3
      .select(container)
      .append("svg")
      .attr("viewBox", "0 0 224 224")
      .attr("preserveAspectRatio", "xMidYMid meet");

    // the real NCT-CRC tile this map was measured on
    svg.append("image")
      .attr("href", apiUrl(hm.tile)).attr("xlink:href", apiUrl(hm.tile))
      .attr("x", 0).attr("y", 0).attr("width", 224).attr("height", 224);

    const g = svg.append("g").attr("class", "hm-overlay").style("display", overlayOn ? null : "none");
    const norm = hm.norm_grid || [];
    const MAX_DARK = 0.74;                 // dimmest patch is veiled, never fully black
    let topR = 0, topC = 0, topV = -1;
    for (let r = 0; r < S; r++) {
      for (let c = 0; c < S; c++) {
        const v = (norm[r] && norm[r][c]) || 0;
        if (v > topV) { topV = v; topR = r; topC = c; }
        // spotlight: a patch keeps the tile's REAL color where it aligns with the concept
        // axis; the less it aligns, the heavier the dark-grey veil (opacity ∝ 1 − importance).
        g.append("rect")
          .attr("x", c * CELL).attr("y", r * CELL)
          .attr("width", CELL + 0.4).attr("height", CELL + 0.4)
          .attr("fill", "#0b0a12")
          .attr("fill-opacity", MAX_DARK * (1 - v));
      }
    }

    // ring the single most concept-carrying patch — left fully un-veiled (real color)
    g.append("rect")
      .attr("x", topC * CELL).attr("y", topR * CELL)
      .attr("width", CELL).attr("height", CELL)
      .attr("fill", "none").attr("stroke", "#ffe08a").attr("stroke-width", 1.4);

    // measured badge (replaces the old "schematic · not measured" watermark)
    svg.append("rect").attr("x", 6).attr("y", 6).attr("width", 74).attr("height", 16)
      .attr("rx", 3).attr("fill", "rgba(12,10,24,0.72)");
    svg.append("text").attr("x", 43).attr("y", 17).attr("text-anchor", "middle")
      .attr("fill", "#9be7a8").style("font-size", "9px").style("font-family", "var(--mono)")
      .text("● measured");

    // header row above the tile: axis · layer · top-z · overlay toggle
    const head = document.createElement("div");
    head.className = "hm-head";
    const strong = hm.top_z >= 3;
    head.innerHTML =
      `<div class="hm-head-left">` +
        `<span class="hm-axis">${hm.pos} vs ${hm.neg}</span>` +
        `<span class="hm-layer">readout layer · ${hm.n_patches || S * S} patches</span>` +
      `</div>` +
      `<div class="hm-head-right">` +
        `<span class="hm-z ${strong ? "hot" : "flat"}">top-patch z ${(+hm.top_z).toFixed(1)} vs null</span>` +
        `<button class="hm-toggle" id="hm-toggle">${overlayOn ? "hide" : "show"} overlay</button>` +
      `</div>`;
    container.prepend(head);
    const btn = head.querySelector("#hm-toggle");
    btn.addEventListener("click", () => { overlayOn = !overlayOn; renderHistology(); });

    // z legend under the tile
    const leg = document.createElement("div");
    leg.className = "hm-legend";
    leg.innerHTML =
      `<span class="hm-leg-lab">shaded = low · clear = high</span>` +
      `<span class="hm-leg-bar"></span>` +
      `<span class="hm-leg-lab">z ${hm.z_min} → ${hm.z_max}</span>`;
    container.appendChild(leg);

    document.getElementById("case-caption").innerHTML =
      `<b>${claim.claim}</b> — real NCT-CRC ${hm.pos} tile. Each cell is one 16px patch: it keeps the ` +
      `tile's real color where its readout-layer activation aligns with the <b>${hm.pos} vs ${hm.neg}</b> ` +
      `concept axis, and is veiled darker the less it does (z-scored against ${hm.n_null} matched-random directions). ` +
      (strong
        ? `The concept <b>singles out specific patches above the null</b> (top-z ${(+hm.top_z).toFixed(1)}) — ` +
          `a measured saliency map, not a schematic.`
        : `No patch clears the null here — reported honestly.`) +
      ` <span class="hm-caveat">Projection saliency on the model's representation; the faithful ` +
      `mask-and-recompute variant is the live source-intervention behind the necessity curve.</span>`;
  }

  // Honest fallback for a claim whose concept has no per-patch substrate (declined claims,
  // or any concept without a precomputed heatmap). No fabricated tissue.
  function renderNoMap(container, claim) {
    const why = claim && claim.reason
      ? claim.reason
      : "no per-patch causal map for this concept on the wired substrate";
    container.innerHTML =
      `<div class="hm-nomap">` +
        `<div class="hm-nomap-mark">⊘</div>` +
        `<div class="hm-nomap-title">No per-patch causal map</div>` +
        `<div class="hm-nomap-sub">${claim ? claim.claim : "—"} — ${why}. ` +
        `The certified evidence (axis, pillars, contrast gate, trace) is on the right.</div>` +
      `</div>`;
    const cap = document.getElementById("case-caption");
    if (cap) cap.textContent = "";
  }

  // ---------------------------------------------------------------- proof: quadrant map
  function renderQuadrant() {
    const container = document.getElementById("quadrant-chart");
    container.innerHTML = "";
    const width = Math.max(420, container.clientWidth || 520);
    const height = 340;
    const margin = { top: 20, right: 20, bottom: 40, left: 50 };

    const svg = d3.select(container).append("svg").attr("width", width).attr("height", height);

    if (!certifiable.length) {
      svg.append("text").attr("x", width / 2).attr("y", height / 2).attr("text-anchor", "middle")
        .attr("fill", "var(--text-faint)").style("font-size", "12px").text("no certifiable claims to plot");
      return;
    }

    // Domains are DATA-DRIVEN so live scores (which can fall well below the mock's tight
    // 0.9–1.0 band) never render off-chart or collapse the sufficiency radius to nothing.
    const specVals = certifiable.map((d) => d.scores.specificity);
    const sufVals = certifiable.map((d) => d.scores.sufficiency);
    let [sLo, sHi] = d3.extent(specVals);
    if (sLo === sHi) { sLo -= 0.02; sHi += 0.02; }          // flat band -> give it height
    const sPad = (sHi - sLo) * 0.2 || 0.02;

    const x = d3.scaleLinear().domain([0, 1.05]).range([margin.left, width - margin.right]);
    const y = d3.scaleLinear().domain([Math.max(0, sLo - sPad), Math.min(1, sHi + sPad)]).nice()
      .range([height - margin.bottom, margin.top]);
    // clamp() so a low-sufficiency claim can't produce a sub-floor / negative (invisible) radius.
    let [rLo, rHi] = d3.extent(sufVals);
    if (rLo === rHi) rLo = Math.max(0, rLo - 0.1);
    const r = d3.scaleSqrt().domain([rLo, rHi]).range([7, 17]).clamp(true);

    svg
      .append("g")
      .selectAll("line")
      .data(y.ticks(5))
      .join("line")
      .attr("class", "gridline")
      .attr("x1", margin.left)
      .attr("x2", width - margin.right)
      .attr("y1", y)
      .attr("y2", y);

    svg.append("g").attr("class", "axis").attr("transform", `translate(0,${height - margin.bottom})`).call(d3.axisBottom(x).ticks(6).tickFormat(d3.format(".1f")));
    svg.append("g").attr("class", "axis").attr("transform", `translate(${margin.left},0)`).call(d3.axisLeft(y).ticks(5).tickFormat(d3.format(".2f")));

    svg
      .append("text")
      .attr("x", (margin.left + width - margin.right) / 2)
      .attr("y", height - 6)
      .attr("text-anchor", "middle")
      .attr("fill", "var(--text-faint)")
      .style("font-size", "11px")
      .text("necessity (live source-intervention)");
    svg
      .append("text")
      .attr("transform", `translate(14,${(margin.top + height - margin.bottom) / 2}) rotate(-90)`)
      .attr("text-anchor", "middle")
      .attr("fill", "var(--text-faint)")
      .style("font-size", "11px")
      .text("specificity");

    svg
      .append("g")
      .attr("id", "quadrant-dots")
      .selectAll("circle")
      .data(certifiable)
      .join("circle")
      .attr("class", (d, i) => "quad-dot" + (i === selectedIdx ? " is-selected" : ""))
      .attr("cx", (d) => x(d.scores.necessity))
      .attr("cy", (d) => y(d.scores.specificity))
      .attr("r", (d) => r(d.scores.sufficiency))
      .attr("fill", (d) => COLOR[d.verdict])
      .attr("fill-opacity", 0.8)
      .attr("stroke", "#0b0c13")
      .attr("stroke-width", (d, i) => (i === selectedIdx ? 3 : 1.5))
      .style("cursor", "pointer")
      .on("click", (event, d) => selectClaimById(d.id))
      .on("mouseover", function (event, d) {
        d3.select(this).attr("fill-opacity", 1);
        showTip(
          event,
          `<div class="tt-title">${d.claim}</div>` +
            `<div class="tt-row"><span>verdict</span><span class="badge ${d.verdict}">${d.verdict}</span></div>` +
            `<div class="tt-row"><span>necessity</span><span>${d.scores.necessity.toFixed(3)}</span></div>` +
            `<div class="tt-row"><span>sufficiency</span><span>${d.scores.sufficiency.toFixed(3)}</span></div>` +
            `<div class="tt-row"><span>specificity</span><span>${d.scores.specificity.toFixed(3)}</span></div>`
        );
      })
      .on("mousemove", moveTip)
      .on("mouseout", function () {
        d3.select(this).attr("fill-opacity", 0.8);
        hideTip();
      });

    // Labels: with many claims clustered in a tight specificity band the names collide, so
    // greedily stagger any that land too close and paint a halo so overlaps stay readable.
    const placed = [];
    certifiable.forEach((d, i) => {
      const px = x(d.scores.necessity);
      let py = y(d.scores.specificity) - r(d.scores.sufficiency) - 6;
      let below = false;
      // if a prior label sits within ~52px horizontally and ~11px vertically, flip below / nudge
      for (const p of placed) {
        if (Math.abs(p.px - px) < 52 && Math.abs(p.py - py) < 11) {
          py = y(d.scores.specificity) + r(d.scores.sufficiency) + 13;
          below = true;
          if (placed.some((q) => Math.abs(q.px - px) < 52 && Math.abs(q.py - py) < 11)) py += 11;
          break;
        }
      }
      py = Math.max(12, Math.min(height - margin.bottom - 2, py));
      placed.push({ px, py, text: (d.concept || "").replace(/_/g, " "), i, below });
    });
    svg
      .append("g")
      .selectAll("text.dot-label")
      .data(placed)
      .join("text")
      .attr("class", (p) => "dot-label" + (p.i === selectedIdx ? " is-selected" : ""))
      .attr("x", (p) => p.px)
      .attr("y", (p) => p.py)
      .attr("text-anchor", "middle")
      .style("font-size", "9.5px")
      .style("pointer-events", "none")
      .attr("paint-order", "stroke")
      .attr("stroke", "var(--panel)")
      .attr("stroke-width", 3)
      .attr("stroke-linejoin", "round")
      .attr("fill", (p) => (p.i === selectedIdx ? "var(--text)" : "var(--text-faint)"))
      .text((p) => p.text);
  }

  function renderQuadrantSelection() {
    d3.selectAll("#quadrant-dots circle")
      .attr("stroke-width", (d, i) => (i === selectedIdx ? 3 : 1.5))
      .classed("is-selected", (d, i) => i === selectedIdx);
  }

  // ---------------------------------------------------------------- proof: necessity curve
  function renderNecessityChart() {
    const claim = currentClaim();
    document.getElementById("necessity-claim-name").textContent = claim.claim;
    const container = document.getElementById("necessity-chart");
    container.innerHTML = "";
    const width = Math.max(340, container.clientWidth || 420);
    const height = 300;
    const margin = { top: 20, right: 20, bottom: 36, left: 44 };
    const svg = d3.select(container).append("svg").attr("width", width).attr("height", height);

    // Concept-level certify has no per-slide live intervention -> no curve to draw.
    if (!claim.live_necessity || !claim.live_necessity.curve || !claim.live_necessity.curve.length) {
      svg.append("text").attr("x", width / 2).attr("y", height / 2).attr("text-anchor", "middle")
        .attr("fill", "var(--text-faint)").attr("font-family", "var(--mono)").attr("font-size", 12)
        .text("no live per-slide intervention — concept-level scope");
      return;
    }
    const curve = claim.live_necessity.curve;
    const layers = curve.map((c) => c.layer);
    const x = d3.scalePoint().domain(layers).range([margin.left, width - margin.right]).padding(0.5);
    const maxGap = Math.max(...curve.map((c) => c.gap), 0.1) * 1.25;
    const y = d3.scaleLinear().domain([0, maxGap]).range([height - margin.bottom, margin.top]);

    svg
      .append("g")
      .selectAll("line")
      .data(y.ticks(4))
      .join("line")
      .attr("class", "gridline")
      .attr("x1", margin.left)
      .attr("x2", width - margin.right)
      .attr("y1", y)
      .attr("y2", y);

    svg.append("g").attr("class", "axis").attr("transform", `translate(0,${height - margin.bottom})`).call(d3.axisBottom(x));
    svg.append("g").attr("class", "axis").attr("transform", `translate(${margin.left},0)`).call(d3.axisLeft(y).ticks(4));

    const nullLine = d3.line().x((d) => x(d.layer)).y((d) => y(d.z !== 0 ? Math.abs(d.gap / d.z) : 0));
    const mainLine = d3.line().x((d) => x(d.layer)).y((d) => y(d.gap));

    svg
      .append("path")
      .datum(curve)
      .attr("d", nullLine)
      .attr("fill", "none")
      .attr("stroke", "var(--text-faint)")
      .attr("stroke-width", 1.5)
      .attr("stroke-dasharray", "5 4");

    svg
      .append("path")
      .datum(curve)
      .attr("d", mainLine)
      .attr("fill", "none")
      .attr("stroke", COLOR.accent)
      .attr("stroke-width", 2.6);

    svg
      .append("g")
      .selectAll("circle")
      .data(curve)
      .join("circle")
      .attr("cx", (d) => x(d.layer))
      .attr("cy", (d) => y(d.gap))
      .attr("r", 5.5)
      .attr("fill", (d) => (d.bites ? COLOR.accent : "var(--border-strong)"))
      .attr("stroke", "#0b0c13")
      .attr("stroke-width", 2)
      .style("cursor", "pointer")
      .on("mouseover", (event, d) =>
        showTip(
          event,
          `<div class="tt-title">layer: ${d.layer}</div><div class="tt-row"><span>margin drop</span><span>+${d.gap.toFixed(3)}</span></div><div class="tt-row"><span>z vs null</span><span>${d.z.toFixed(1)}</span></div><div class="tt-note">${d.bites ? "significant bite" : "no significant bite (redundancy)"}</div>`
        )
      )
      .on("mousemove", moveTip)
      .on("mouseout", hideTip);
  }

  // ---------------------------------------------------------------- answer flow: sankey + claim rail
  const VERDICT_ORDER = { GROUNDED: 0, WEAK: 1, NOT_CERTIFIABLE: 2 };
  const SANKEY_ACCENT = "#7c9dff"; // neutral source-node tone, distinct from all 3 verdict colors

  // Group certifiable claims by concept (splitting a concept into separate nodes if its
  // claims ever disagree on verdict), then append one synthetic node absorbing ALL declined
  // claims. Returns {groups, verdictCount, total}.
  function buildFlowModel() {
    const claims = window.CARD.claims;
    const cert = claims.filter((c) => !!c.scores);
    const declined = claims.filter((c) => !c.scores);

    const gmap = new Map();
    cert.forEach((c) => {
      const key = c.concept + "||" + c.verdict;
      if (!gmap.has(key)) gmap.set(key, { concept: c.concept, verdict: c.verdict, count: 0, synthetic: false });
      gmap.get(key).count++;
    });
    let groups = [...gmap.values()].sort(
      (a, b) => VERDICT_ORDER[a.verdict] - VERDICT_ORDER[b.verdict] || a.concept.localeCompare(b.concept)
    );
    if (declined.length) {
      groups.push({ concept: "cell/subcellular", verdict: "NOT_CERTIFIABLE", count: declined.length, synthetic: true });
    }

    const verdictCount = { GROUNDED: 0, WEAK: 0, NOT_CERTIFIABLE: 0 };
    groups.forEach((g) => (verdictCount[g.verdict] += g.count));
    return { groups, verdictCount, total: claims.length };
  }

  function conceptLabel(g) {
    if (g.synthetic) return `cell/subcellular (${g.count})`;
    const name = g.concept.replace(/_/g, " ");
    return g.count > 1 ? `${name} (${g.count} claims)` : name;
  }

  function renderSankey() {
    const container = document.getElementById("sankey-chart");
    if (!container) return;
    if (sankeyRaf) { cancelAnimationFrame(sankeyRaf); sankeyRaf = null; }
    container.innerHTML = "";

    const { groups, verdictCount, total } = buildFlowModel();
    const verdicts = ["GROUNDED", "WEAK", "NOT_CERTIFIABLE"];

    const W = Math.max(460, container.clientWidth || 640);
    const H = 344;
    const nodeW = 14;
    // tighten inter-node padding when the concept column is crowded so thin ribbons + their
    // labels don't collide (mock has 5-6 nodes; a live answer can have 10+).
    const pad = Math.max(1, groups.length, verdicts.length) > 7 ? 11 : 18;
    const margin = { left: 4, right: 152, top: 10, bottom: 10 };
    const availH = H - margin.top - margin.bottom;
    const maxNodes = Math.max(1, groups.length, verdicts.length);
    const ky = (availH - (maxNodes - 1) * pad) / total; // px per claim

    const x0 = margin.left;
    const x2 = W - margin.right;
    const x1 = x0 + (x2 - x0) * 0.4;

    // node objects
    const srcNode = { id: "src", label: "K-Pro answer", value: total, verdict: null, x: x0 };
    const conceptNodes = groups.map((g, i) => ({
      id: "g" + i, label: conceptLabel(g), value: g.count, verdict: g.verdict, x: x1,
    }));
    const verdictNodes = verdicts.map((v) => ({
      id: v, label: v, value: verdictCount[v], verdict: v, x: x2, isVerdict: true,
    }));

    // stack a column vertically, centered in the available height
    function layoutColumn(nodes) {
      const stackH = nodes.reduce((s, n) => s + n.value * ky, 0) + (nodes.length - 1) * pad;
      let y = margin.top + (availH - stackH) / 2;
      nodes.forEach((n) => { n.h = n.value * ky; n.y = y; n.sy = y; n.ty = y; y += n.h + pad; });
    }
    layoutColumn([srcNode]);
    layoutColumn(conceptNodes);
    layoutColumn(verdictNodes);

    // links: source -> concept, concept -> verdict (iterate in node order so ribbons stack cleanly)
    const links = [];
    conceptNodes.forEach((cn) => links.push({ s: srcNode, t: cn, value: cn.value, verdict: cn.verdict }));
    conceptNodes.forEach((cn) => {
      const vn = verdictNodes.find((v) => v.id === cn.verdict);
      links.push({ s: cn, t: vn, value: cn.value, verdict: cn.verdict });
    });
    links.forEach((l) => {
      l.w = l.value * ky;
      l.x0 = l.s.x + nodeW;
      l.x1 = l.t.x;
      l.y0 = l.s.sy + l.w / 2; l.s.sy += l.w;
      l.y1 = l.t.ty + l.w / 2; l.t.ty += l.w;
    });
    const linkPath = (l) => {
      const xm = (l.x0 + l.x1) / 2;
      return `M${l.x0},${l.y0} C${xm},${l.y0} ${xm},${l.y1} ${l.x1},${l.y1}`;
    };

    const svg = d3.select(container).append("svg").attr("viewBox", `0 0 ${W} ${H}`).attr("height", H);

    // links
    const linkSel = svg
      .append("g")
      .selectAll("path")
      .data(links)
      .join("path")
      .attr("class", "sankey-link")
      .attr("d", linkPath)
      .attr("stroke", (l) => COLOR[l.verdict])
      .attr("stroke-width", (l) => Math.max(1, l.w))
      .attr("stroke-opacity", 0.35)
      .on("mouseover", function (event, l) {
        d3.select(this).attr("stroke-opacity", 0.75);
        showTip(
          event,
          `<div class="tt-title">${l.s.label} → ${l.t.label}</div>` +
            `<div class="tt-row"><span>claims</span><span>${l.value}</span></div>`
        );
      })
      .on("mousemove", moveTip)
      .on("mouseout", function () { d3.select(this).attr("stroke-opacity", 0.35); hideTip(); });

    // nodes
    const allNodes = [srcNode, ...conceptNodes, ...verdictNodes];
    svg
      .append("g")
      .selectAll("rect")
      .data(allNodes)
      .join("rect")
      .attr("class", "sankey-node")
      .attr("x", (n) => n.x)
      .attr("y", (n) => n.y)
      .attr("width", nodeW)
      .attr("height", (n) => Math.max(1, n.h))
      .attr("rx", 2)
      .attr("fill", (n) => (n.verdict ? COLOR[n.verdict] : SANKEY_ACCENT))
      .on("mouseover", function (event, n) {
        showTip(event, `<div class="tt-title">${n.label}</div><div class="tt-row"><span>claims</span><span>${n.value}</span></div>`);
      })
      .on("mousemove", moveTip)
      .on("mouseout", hideTip);

    // labels — concept column (smaller), verdict column (larger/bolder), source (right of node w/ bg)
    function nodeLabel(sel, nodes, size, weight, fill, maxPx) {
      const budget = maxPx && parseFloat(size) ? Math.floor(maxPx / (parseFloat(size) * 0.56)) : 0;
      const trunc = (s) => (budget && s.length > budget ? s.slice(0, Math.max(1, budget - 1)) + "…" : s);
      const t = sel
        .selectAll("text")
        .data(nodes.filter((n) => n.h > 0.5))
        .join("text")
        .attr("class", "sankey-nlabel")
        .attr("x", (n) => n.x + nodeW + 8)
        .attr("y", (n) => n.y + n.h / 2)
        .attr("dominant-baseline", "middle")
        .style("font-size", size)
        .style("font-weight", weight)
        .attr("fill", fill)
        .text((n) => trunc(n.label));
      // keep the full name reachable on hover when truncated
      t.append("title").text((n) => n.label);
    }
    // concept labels live between their column and the verdict column — clip to that gap.
    nodeLabel(svg.append("g"), conceptNodes, "11.5px", 500, "var(--text-dim)", x2 - (x1 + nodeW + 8) - 6);
    nodeLabel(svg.append("g"), verdictNodes, "12.5px", 700, "var(--text)");

    // source label: sits over the link fan, so give it a small backing rect for legibility
    const srcG = svg.append("g");
    const srcLabelX = srcNode.x + nodeW + 8;
    const srcLabelY = srcNode.y + srcNode.h / 2;
    srcG
      .append("rect")
      .attr("x", srcLabelX - 5)
      .attr("y", srcLabelY - 11)
      .attr("width", 96)
      .attr("height", 22)
      .attr("rx", 4)
      .attr("fill", "var(--panel)")
      .attr("fill-opacity", 0.82);
    srcG
      .append("text")
      .attr("class", "sankey-nlabel")
      .attr("x", srcLabelX)
      .attr("y", srcLabelY)
      .attr("dominant-baseline", "middle")
      .style("font-size", "11.5px")
      .style("font-weight", 600)
      .attr("fill", SANKEY_ACCENT)
      .text(srcNode.label);

    // subtle drifting particles — one per link, "claims are flowing" (getPointAtLength loop)
    const pathEls = linkSel.nodes();
    const particles = pathEls.map((el, i) => ({ el, len: el.getTotalLength(), off: (i * 0.37) % 1, verdict: links[i].verdict }));
    const dotSel = svg
      .append("g")
      .selectAll("circle")
      .data(particles)
      .join("circle")
      .attr("r", 2.2)
      .attr("fill", (d) => COLOR[d.verdict])
      .attr("opacity", 0.9);
    let last = null;
    function tick(ts) {
      if (last === null) last = ts;
      const dt = Math.min(0.05, (ts - last) / 1000); // clamp to survive tab-switch pauses
      last = ts;
      dotSel.each(function (d) {
        d.off = (d.off + dt * 0.14) % 1;
        const p = d.el.getPointAtLength(d.off * d.len);
        d3.select(this).attr("cx", p.x).attr("cy", p.y);
      });
      sankeyRaf = requestAnimationFrame(tick);
    }
    sankeyRaf = requestAnimationFrame(tick);
  }

  function renderClaimRail() {
    const rail = document.getElementById("claim-rail");
    if (!rail) return;
    if (railSelectedId === null) railSelectedId = currentClaim().id;
    rail.innerHTML = "";
    const sel = d3.select(rail);

    const rows = sel
      .selectAll(".rail-row")
      .data(window.CARD.claims)
      .join("div")
      .attr("class", (c) => "rail-row" + (c.scores ? "" : " declined"))
      .attr("data-id", (c) => c.id)
      .on("click", (event, c) => {
        if (c.scores) {
          selectClaimById(c.id); // drives histology + quadrant + necessity + verdict
        } else {
          railSelectedId = c.id; // declined: highlight only — no battery ran, nothing to drive
          highlightRail();
        }
      });

    rows.append("div").attr("class", "rail-dot").style("background", (c) => COLOR[c.verdict]);

    const txt = rows.append("div").attr("class", "rail-text");
    txt.append("div").attr("class", "rail-claim").text((c) => c.claim);
    txt
      .append("div")
      .attr("class", "rail-sub")
      .text((c) => (c.scores ? (c.concept || "").replace(/_/g, " ") : c.reason || "not certifiable"));

    // mini battery: 3 bars for certifiable, one flat neutral bar for declined
    const bars = rows.append("div").attr("class", "rail-bars");
    bars.each(function (c) {
      const g = d3.select(this);
      if (c.scores) {
        [c.scores.necessity, c.scores.sufficiency, c.scores.specificity].forEach((v) => {
          g.append("div")
            .attr("class", "rail-bar")
            .style("height", Math.max(3, (v || 0) * 22) + "px")
            .style("background", COLOR[c.verdict]);
        });
      } else {
        g.append("div").attr("class", "rail-bar declined").style("height", "4px");
      }
    });

    highlightRail();
  }

  function highlightRail() {
    d3.selectAll("#claim-rail .rail-row").classed("selected", function () {
      return this.getAttribute("data-id") === railSelectedId;
    });
  }

  const NUMWORD = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen"];
  function renderAnswerFlow() {
    const badge = document.getElementById("aflow-badge");
    if (badge) badge.textContent = window.CARD.coverage.summary;
    // Headline must match the real claim count — never the hardcoded mock's "twelve".
    const title = document.getElementById("aflow-title");
    if (title) {
      const n = window.CARD.claims.length;
      const word = NUMWORD[n] || String(n);
      title.textContent = `One answer, ${word} claim${n === 1 ? "" : "s"}, one honest split`;
    }
    renderSankey();
    renderClaimRail();
  }

  // ---------------------------------------------------------------- proof: probe design appendix
  function renderProbeAppendix() {
    document.getElementById("probe-sub").textContent =
      `"${window.DESIGNED_PROBES.question}" → ${window.DESIGNED_PROBES.designed_by} proposed ${window.DESIGNED_PROBES.n_probes} candidate contrasts, validated against the same deterministic intensity gate as hand-authored probes.`;

    const probes = window.DESIGNED_PROBES.probes;
    const chosen = probes.filter((p) => p.gate === "PASS").sort((a, b) => b.auroc - a.auroc || b.sufficiency - a.sufficiency)[0];

    const list = d3.select("#probe-compact-list");
    list.selectAll("*").remove();
    const rows = list
      .selectAll(".probe-row")
      .data(probes)
      .join("div")
      .attr("class", (d) => "probe-row" + (d.gate === "PASS" ? " accepted" : " rejected") + (d === chosen ? " chosen" : ""));
    rows.append("div").attr("class", "mark").text((d) => (d.gate === "PASS" ? "✓" : "✗"));
    rows.append("div").attr("class", "probe-name").text((d) => d.concept + (d === chosen ? " — chosen" : ""));
    rows.append("div").attr("class", "probe-contrast").text((d) => d.contrast);
    rows.append("div").attr("class", "probe-flag").text((d) => (d.gate === "REJECT" ? `rides intensity |r|=${d.intensity_r.toFixed(2)}` : ""));
  }

  // ---------------------------------------------------------------- verdict
  function renderVerdict() {
    const claim = currentClaim();
    const badge = document.getElementById("verdict-badge-big");
    badge.className = "verdict-badge-big " + claim.verdict;
    badge.textContent = claim.verdict;
    // Prefer the explicit verdict step; fall back to the last step, then to the claim's
    // notes/reason. Live claims can ship an empty or truncated trace — never assume one.
    const tr = claim.reasoning_trace || [];
    const vstep = tr.find((t) => t.step === "verdict") || tr[tr.length - 1];
    document.getElementById("verdict-sentence").textContent =
      (vstep && vstep.interpretation) || (claim.notes && claim.notes[0]) || claim.reason || "—";
  }

  function renderCoverage() {
    const cov = window.CARD.coverage;
    const grounded = certifiable.filter((c) => c.verdict === "GROUNDED").length;
    const weak = certifiable.filter((c) => c.verdict === "WEAK").length;
    const notCert = (cov && cov.not_certifiable != null) ? cov.not_certifiable : declinedCount;
    // total from the parts we're drawing so the three segments always sum to exactly 100%
    // (a stale coverage.claims_total would otherwise under/overflow the rounded bar).
    const total = Math.max(1, grounded + weak + notCert);
    const bar = d3.select("#coverage-bar");
    bar.selectAll("*").remove();
    [
      [grounded, COLOR.GROUNDED],
      [weak, COLOR.WEAK],
      [notCert, COLOR.NOT_CERTIFIABLE],
    ].forEach(([n, c]) => {
      bar.append("div").attr("class", "coverage-seg").style("width", (n / total) * 100 + "%").style("background", c);
    });
    document.getElementById("coverage-line").textContent = cov.summary;
  }

  function renderConfound() {
    const cf = window.CONFOUND;
    document.getElementById("confound-badge").textContent = cf.status;
    document.getElementById("confound-reason").textContent = cf.reason;

    const container = document.getElementById("confound-schematic");
    container.innerHTML = "";
    const auroc = cf.illustrative_site_probe_auroc;
    const threshold = 0.65;
    const overlapPct = Math.max(0, Math.min(1, (auroc - 0.5) / 0.5));
    const crossed = auroc > threshold;

    const wrap = d3.select(container).append("div").attr("class", "axis-flow");

    const conceptRow = wrap.append("div").attr("class", "axis-row concept");
    conceptRow.append("div").attr("class", "axis-label").text("Concept axis");
    conceptRow.append("div").attr("class", "axis-track").append("div").attr("class", "axis-fill").style("width", "100%");
    conceptRow.append("div").html('<span style="color:var(--accent);font-size:14px">&#8594;</span>');

    const siteRow = wrap.append("div").attr("class", "axis-row site");
    siteRow.append("div").attr("class", "axis-label").text("Site axis");
    siteRow.append("div").attr("class", "axis-track").append("div").attr("class", "axis-fill").style("width", "100%");
    siteRow.append("div").html('<span style="color:var(--text-faint);font-size:14px">&#8594;</span>');

    const overlap = wrap.append("div").attr("class", "overlap-block");
    const head = overlap.append("div").attr("class", "ov-head");
    head.append("div").attr("class", "ov-title").text("Directional overlap (site-probe AUROC)");
    head.append("div").attr("class", "ov-value").style("color", crossed ? "var(--danger)" : "var(--warning)").text(auroc.toFixed(2));
    overlap.append("div").attr("class", "overlap-track").append("div").attr("class", "overlap-fill").style("width", Math.max(4, overlapPct * 100) + "%");
    overlap
      .append("div")
      .attr("class", "overlap-caption")
      .html(
        crossed
          ? `⚠ above the ${threshold.toFixed(2)} gate threshold — would demote GROUNDED to WEAK.`
          : `${auroc.toFixed(2)} is barely above chance (0.50) — no scanner artifact detected in this illustrative pass. Real check needs multi-site data (not wired this weekend).`
      );
  }

  function renderBuildNotes() {
    const table = d3.select("#tracks-table");
    table.selectAll("*").remove();
    table.append("thead").append("tr").selectAll("th").data(["Track", "Model", "Objective", "Status"]).join("th").text((d) => d);
    const rows = table.append("tbody").selectAll("tr").data(window.TRACKS).join("tr");
    rows.append("td").text((d) => d.track);
    rows.append("td").text((d) => d.model);
    rows.append("td").text((d) => d.objective);
    rows.append("td").text((d) => d.status);

    document.getElementById("mil-note").textContent =
      "MIL slide aggregator (stretch, not run this weekend): " + window.MIL.claim;
  }

  // ---------------------------------------------------------------- view router
  const VIEW_META = {
    prompt: { title: "Prompt", tag: "input slide → K-Pro answer → certify" },
    case: { title: "Case", tag: "one claim, followed pixel to concept axis" },
    proof: { title: "Proof", tag: "necessity × sufficiency × specificity, and where the axis came from" },
    verdict: { title: "Verdict", tag: "certified claim, confound check, honest coverage" },
    research: { title: "AutoResearch", tag: "autonomous causal-circuit discovery — probe · certify · locate · ablate · reflect" },
  };

  function goToView(id) {
    document.body.dataset.view = id;                 // drives Prompt-view / card-view CSS
    d3.selectAll(".view").classed("active", false);
    d3.select("#view-" + id).classed("active", true);
    d3.selectAll(".nav-item").classed("active", false);
    d3.select(`.nav-item[data-view="${id}"]`).classed("active", true);
    const meta = VIEW_META[id];
    if (meta) {
      document.getElementById("topbar-title").textContent = meta.title;
      document.getElementById("topbar-tag").textContent = meta.tag;
    }
    if (id === "proof") { renderAnswerFlow(); renderQuadrant(); renderQuadrantSelection(); renderNecessityChart(); }
  }

  function initNavigation() {
    d3.selectAll(".nav-item").on("click", function () {
      if (this.dataset.view) goToView(this.dataset.view);  // link tiles (e.g. Studio) navigate via href
    });
  }

  // ----------------------------------------- prompt console: score + reasoning trace
  // Shows, in the Prompt window, the certified claim's verdict/scores at the top and the
  // deterministic reasoning trace as boxes-with-arrows in causal order. Uses the same
  // per-claim `reasoning_trace` the certify_answer card returns (step OR pillar key).
  function renderPromptTrace() {
    const el = document.getElementById("c-trace");
    if (!el) return;
    const claim = (typeof currentClaim === "function") ? currentClaim() : null;
    const steps = claim && (claim.reasoning_trace || []);
    if (!claim || !steps || !steps.length) { el.hidden = true; el.innerHTML = ""; return; }
    const esc = (s) => String(s == null ? "" : s).replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));
    const col = COLOR[claim.verdict] || COLOR.accent;
    const sc = claim.scores || {};
    const num = (v) => (typeof v === "number" ? v.toFixed(2) : "—");
    // Composite "certify score": geomean of the three pillars — the same aggregate the
    // population certify uses (geomean(necessity, sufficiency, specificity)).
    const g = ["necessity", "sufficiency", "specificity"].map((k) => Math.max(0, sc[k] || 0));
    const total = Math.pow(g[0] * g[1] * g[2], 1 / 3);
    const boxes = steps
      .map((s, i) => {
        const name = esc((s.step || s.pillar || "").toString().toUpperCase());
        const arrow = i < steps.length - 1 ? `<span class="ct-arrow" aria-hidden="true">→</span>` : "";
        const why = s.interpretation ? `<span class="ct-why">${esc(s.interpretation)}</span>` : "";
        return `<div class="ct-box"><span class="ct-n">${esc(s.n)}</span><span class="ct-name">${name}</span><span class="ct-obs">${esc(s.observation)}</span>${why}</div>${arrow}`;
      })
      .join("");
    el.innerHTML =
      `<div class="ct-scoreband">` +
        `<div class="ct-total"><span class="ct-total-lab">certify score</span>` +
          `<span class="ct-total-val" style="color:${col}">${total.toFixed(2)}</span>` +
          `<span class="ct-total-sub">geomean · nec × suf × spec</span></div>` +
        `<div class="ct-meta"><span class="ct-concept">${esc(claim.concept)}</span>` +
          `<span class="ct-chip" style="color:${col};border-color:${col}">${esc(claim.verdict)}</span>` +
          `<span class="ct-sc">nec ${num(sc.necessity)}</span>` +
          `<span class="ct-sc">suf ${num(sc.sufficiency)}</span>` +
          `<span class="ct-sc">spec ${num(sc.specificity)}</span></div>` +
      `</div>` +
      `<div class="ct-lab2">reasoning trace · causal order</div>` +
      `<div class="ct-flow">${boxes}</div>`;
    el.hidden = false;
  }

  // ------------------------------------------------------------- render-all
  function renderAll() {
    renderPromptTrace();
    renderClaimStrip();
    renderHistology();
    renderAnswerFlow();
    renderQuadrant();
    renderNecessityChart();
    renderProbeAppendix();
    renderVerdict();
    renderCoverage();
    renderConfound();
    renderBuildNotes();
  }

  // ------------------------------------------------------ live data (bridge)
  // Pull the REAL globals from the certify infra (/api/all). On any failure keep the
  // static mock in data.js — the dashboard always renders. A small badge shows which.
  function setLiveBadge(live) {
    const el = document.getElementById("substrate-label");
    if (!el) return;
    el.dataset.live = live ? "1" : "0";
    el.textContent = (live ? "● LIVE · " : "○ mock · ") + el.textContent.replace(/^[●○] (LIVE|mock) · /, "");
  }
  async function bootstrapData() {
    try {
      const r = await fetch(apiUrl("api/all"), { headers: { Accept: "application/json" } });
      if (!r.ok) throw new Error("api " + r.status);
      const d = await r.json();
      if (d.error) throw new Error(d.error);
      if (d.CARD) window.CARD = d.CARD;
      if (d.DESIGNED_PROBES) window.DESIGNED_PROBES = d.DESIGNED_PROBES;
      if (d.MCP_VERBS) window.MCP_VERBS = d.MCP_VERBS;
      if (d.TRACKS) window.TRACKS = d.TRACKS;
      recomputeState();
      setLiveBadge(true);
      return true;
    } catch (e) {
      setLiveBadge(false); // static mock retained
      return false;
    }
  }

  // Measured per-patch causal heatmaps (dashboard/precompute_heatmaps.py). Static JSON so
  // the demo needs no GPU at runtime; absence just falls back to the honest "no map" panel.
  async function loadHeatmaps() {
    try {
      const r = await fetch(apiUrl("heatmaps/heatmaps.json"), { headers: { Accept: "application/json" } });
      if (!r.ok) return;
      window.HEATMAPS = await r.json();
      if (!document.body.classList.contains("no-card")) renderHistology();
    } catch (e) { /* no heatmaps -> renderNoMap fallback */ }
  }

  // ---------------------------------------------------------------- run button
  let running = false;
  async function runCertifyAnimation() {
    if (running) return;
    running = true;
    const btn = document.getElementById("btn-run");
    btn.disabled = true;
    btn.textContent = "running battery…";
    goToView("case");
    try {
      const r = await fetch(apiUrl("api/certify_answer"), {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt: window.CARD.prompt, answer: window.CARD.answer,
                               track: window.CARD.track, bedrock: true }),
      });
      if (r.ok) {
        const d = await r.json();
        if (d.CARD) { window.CARD = d.CARD; recomputeState(); setLiveBadge(true); renderAll(); }
      }
    } catch (e) { /* keep current card */ }
    btn.disabled = false;
    btn.textContent = "Run certify_answer()";
    running = false;
  }

  // ---------------------------------------------------------------- inference console
  const $c = (id) => document.getElementById(id);
  function setStatus(t) { $c("c-status").textContent = t; }
  function showOut(o) { const el = $c("c-out"); el.textContent = typeof o === "string" ? o : JSON.stringify(o, null, 2); el.classList.add("show"); }

  // Run certify on the current prompt+answer -> reveal + populate the Case/Proof/Verdict.
  async function runCertify() {
    if (!$c("c-answer").value.trim()) { setStatus("submit the slide + prompt first →"); return; }
    const btn = $c("c-certify"); btn.disabled = true; setStatus("running the causal battery…");
    try {
      const r = await fetch(apiUrl("api/certify_answer"), {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt: $c("c-prompt").value, answer: $c("c-answer").value,
                               track: $c("c-track").value, bedrock: $c("c-bedrock").checked }) });
      const d = await r.json();
      if (d.error) throw new Error(d.error);
      window.CARD = d.CARD; recomputeState(); setLiveBadge(true);
      document.body.classList.remove("no-card");        // reveal the evidence card
      // hand the REAL certified card to the Intervention Studio (separate page) so its
      // layer visualisation runs on this exact run, not the mock.
      try {
        localStorage.setItem("biolayer:lastCard", JSON.stringify({
          card: d.CARD, track: $c("c-track").value, at: new Date().toISOString() }));
      } catch (e) { /* storage unavailable — Studio falls back to /api/all */ }
      renderAll(); goToView("case");
      setStatus(d.CARD.coverage.summary + " · " + (d.CARD.certification_scope || {}).level);
    } catch (e) { setStatus("certify failed: " + e.message); }
    btn.disabled = false;
  }

  function initConsole() {
    // tile-mosaic metadata (each tile is classified independently — no spatial adjacency)
    fetch(apiUrl("slide_demo.json")).then((r) => r.json()).then((s) => {
      window._slide = s;
      $c("slide-meta").innerHTML = "<b>" + (s.substrate || "h_optimus_0") + "</b> · " +
        s.n_tiles + " independent tiles → composition <b>" + s.ho_composition + "</b>" +
        (s.sampling ? '<br><span class="dimlabel">' + s.sampling + "</span>" : "");
      if (s.prompt) $c("c-prompt").value = s.prompt;
    }).catch(() => { $c("slide-meta").textContent = "tiles unavailable"; });

    // Submit slide + prompt -> K-Pro (Claude) infers the answer
    $c("c-submit").addEventListener("click", async () => {
      const b = $c("c-submit"); b.disabled = true; setStatus("K-Pro inferring from the slide…");
      $c("c-answer").value = "";
      try {
        const r = await fetch(apiUrl("api/answer"), {
          method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ prompt: $c("c-prompt").value }) });
        const d = await r.json();
        if (d.error) throw new Error(d.error);
        $c("c-answer").value = d.answer;
        setStatus("answer ready — press Run certify →");
      } catch (e) { setStatus("submit failed: " + e.message); }
      b.disabled = false;
    });

    // Hypothesis -> optimize the prompt and refill the prompt box
    $c("c-hypothesis").addEventListener("click", async () => {
      const b = $c("c-hypothesis"); b.disabled = true; setStatus("optimizing the prompt…");
      try {
        const cov = window.CARD && window.CARD.coverage ? window.CARD.coverage.summary : "";
        const r = await fetch(apiUrl("api/optimize_prompt"), {
          method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ prompt: $c("c-prompt").value, coverage: cov }) });
        const d = await r.json();
        if (d.error) throw new Error(d.error);
        $c("c-prompt").value = d.prompt;
        setStatus("prompt optimized ↺ — Submit again to re-infer");
      } catch (e) { setStatus("optimize failed: " + e.message); }
      b.disabled = false;
    });

    $c("c-certify").addEventListener("click", runCertify);

    // MCP verb chips -> raw verb output in the console tray
    document.querySelectorAll(".verbchips button[data-verb]").forEach((b) => {
      b.addEventListener("click", async () => {
        const verb = b.dataset.verb; b.disabled = true; setStatus(verb + "…");
        try {
          const r = await fetch(apiUrl("api/verb/" + verb), {
            method: "POST", headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ track: $c("c-track").value, question: $c("c-prompt").value }) });
          const d = await r.json();
          if (d.error) throw new Error(d.error);
          showOut(d.result); setStatus(verb + " ✓");
        } catch (e) { setStatus(verb + " failed: " + e.message); }
        b.disabled = false;
      });
    });
  }

  // ---------------------------------------------------------------- init
  function init() {
    // Start BLANK: the evidence card is empty until the user submits + certifies.
    document.body.classList.add("no-card");
    setLiveBadge(false);

    initNavigation();
    initConsole();
    loadHeatmaps();

    document.getElementById("claim-prev").addEventListener("click", () => selectClaim(selectedIdx - 1));
    document.getElementById("claim-next").addEventListener("click", () => selectClaim(selectedIdx + 1));
    const _runBtn = document.getElementById("btn-run");
    if (_runBtn) _runBtn.addEventListener("click", runCertify);

    goToView("prompt");   // land on the Prompt section

    window.addEventListener("resize", () => {
      renderHistology();
      if (document.getElementById("view-proof").classList.contains("active")) {
        renderAnswerFlow();
        renderQuadrant();
        renderQuadrantSelection();
        renderNecessityChart();
      }
    });
  }

  document.addEventListener("DOMContentLoaded", init);
})();
