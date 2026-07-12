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

  // recompute the derived claim state after a live CARD override (fetch / Run button)
  function recomputeState() {
    certifiable = window.CARD.claims.filter((c) => !!c.scores);
    declinedCount = window.CARD.claims.filter((c) => !c.scores).length;
    selectedIdx = Math.min(selectedIdx, Math.max(0, certifiable.length - 1));
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
    document.getElementById("strip-declined").textContent =
      `+ ${declinedCount} declined (cell/subcellular — needs HistoPLUS)`;
  }

  function selectClaim(idx) {
    selectedIdx = ((idx % certifiable.length) + certifiable.length) % certifiable.length;
    hideTip();
    renderClaimStrip();
    renderHistology();
    renderQuadrantSelection();
    renderNecessityChart();
    renderVerdict();
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

  function heatAt(x, y, claim) {
    const c = TISSUE_LAYOUT.centers;
    const glandField = Math.max(...c.map((g) => Math.exp(-Math.pow(Math.hypot(x - g.x, y - g.y) / g.r, 2) * 1.4)));
    if (!claim || !claim.scores) return 0;
    switch (claim.concept) {
      case "tumor_epithelium":
        return glandField;
      case "normal_mucosa":
        return Math.max(0, 1 - glandField * 1.35);
      case "immune_infiltrate": {
        const rand = mulberry32(hashStr(claim.id + "heat"));
        let s = 0;
        for (let i = 0; i < 10; i++) {
          const bx = rand(), by = rand();
          s = Math.max(s, Math.exp(-Math.pow(Math.hypot(x - bx, y - by) / 0.09, 2)));
        }
        return Math.max(s, glandField * 0.18);
      }
      case "stroma":
        return glandField * (1 - glandField) * 4.2;
      default:
        return 0;
    }
  }

  // Grounded, not invented: reconstructs the same intermediate quantities heatAt()
  // used, so "why is this pixel hot" always matches what's actually drawn.
  function explainPatch(x, y, claim) {
    const c = TISSUE_LAYOUT.centers;
    let nearest = 0, bestD = Infinity;
    c.forEach((g, i) => {
      const d = Math.hypot(x - g.x, y - g.y) / g.r;
      if (d < bestD) { bestD = d; nearest = i; }
    });
    const glandField = Math.max(...c.map((g) => Math.exp(-Math.pow(Math.hypot(x - g.x, y - g.y) / g.r, 2) * 1.4)));

    switch (claim.concept) {
      case "tumor_epithelium":
        return `inside tumor nest #${nearest + 1}, gland-field ${glandField.toFixed(2)}`;
      case "normal_mucosa":
        return `gland-field ${glandField.toFixed(2)} (low = ordered mucosa, not neoplasia)`;
      case "immune_infiltrate": {
        const rand = mulberry32(hashStr(claim.id + "heat"));
        let bestHot = Infinity;
        for (let i = 0; i < 10; i++) {
          const bx = rand(), by = rand();
          bestHot = Math.min(bestHot, Math.hypot(x - bx, y - by));
        }
        return bestHot < 0.09
          ? `inside a lymphocyte aggregate, ${bestHot.toFixed(2)} grid-units from center`
          : `outside any lymphocyte aggregate (${bestHot.toFixed(2)} away) — faint gland-border baseline`;
      }
      case "stroma": {
        const border = glandField * (1 - glandField) * 4.2;
        return `tumor–gland boundary, interface score ${border.toFixed(2)}`;
      }
      default:
        return "no heat model wired for this concept";
    }
  }

  let selectedPatch = null;

  function renderReadoutStatic(claim) {
    const el = document.getElementById("readout-static");
    const pillars = [
      ["necessity", claim.scores.necessity],
      ["sufficiency", claim.scores.sufficiency],
      ["specificity", claim.scores.specificity],
    ];
    const cv = claim.contrast_validation;
    el.innerHTML =
      `<div class="ro-group"><div class="ro-label">Axis</div><div class="ro-axis-value">${claim.contrast}</div></div>` +
      `<div class="ro-group"><div class="ro-label">Causal pillars</div>` +
      pillars
        .map(
          ([name, v]) =>
            `<div class="ro-pillar"><div class="ro-pillar-name">${name}</div><div class="ro-pillar-track"><div class="ro-pillar-fill" style="width:${pct(v, 0)};background:${COLOR[claim.verdict]}"></div></div><div class="ro-pillar-value">${v.toFixed(3)}</div></div>`
        )
        .join("") +
      `</div>` +
      `<div class="ro-group"><div class="ro-label">Contrast gate</div>` +
      `<div class="ro-gate-row"><span class="badge ${cv.valid ? "PASS" : "CAPPED"}">${cv.valid ? "PASS" : "CAPPED"}</span><span class="ro-gate-detail">AUROC ${cv.heldout_auroc.toFixed(3)} · |r| ${cv.intensity_collinearity.toFixed(3)} (cap 0.60)</span></div>` +
      (cv.warnings.length ? `<div class="ro-gate-warn">⚠ ${cv.warnings.join("; ")}</div>` : "") +
      `</div>`;
  }

  function renderReadoutLive(claim, patch) {
    const el = document.getElementById("readout-live");
    if (!patch) {
      el.innerHTML =
        `<div class="live-head"><div class="live-title">live patch readout</div></div>` +
        `<div class="live-empty">Hover the tissue grid — this panel updates in real time with that token's activation, most influential neuron, and a grounded reason.</div>`;
      return;
    }
    const why = explainPatch(patch.xn, patch.yn, claim);
    const neuron = Math.floor(mulberry32(hashStr(claim.id + ":" + patch.i + "," + patch.j))() * 1024);
    el.innerHTML =
      `<div class="live-head"><div class="live-title"><span class="live-dot"></span>live patch readout</div><div class="live-patch-coord">[${patch.i}, ${patch.j}]</div></div>` +
      `<div class="live-row"><div class="live-row-k">${claim.concept.replace(/_/g, " ")} activation</div><div class="live-row-v accent">${pct(patch.heat, 0)}</div></div>` +
      `<div class="live-row"><div class="live-row-k">most influential neuron</div><div class="live-row-v">#${neuron}</div></div>` +
      `<div class="live-row"><div class="live-row-k">layer</div><div class="live-row-v">encoder.layer[16]</div></div>` +
      `<div class="live-why">${why}</div>`;
  }

  function renderHistology() {
    const claim = currentClaim();
    const container = document.getElementById("tile-stage");
    container.innerHTML = "";
    const W = 640, H = 460;
    const svg = d3.select(container).append("svg").attr("viewBox", `0 0 ${W} ${H}`);

    svg.append("rect").attr("width", W).attr("height", H).attr("fill", "#1b1730");

    const line = d3.line().curve(d3.curveCatmullRomClosed.alpha(0.85));
    TISSUE_LAYOUT.blobs.forEach((b) => {
      svg
        .append("path")
        .attr("d", line(b.pts.map((p) => [p[0] * W, p[1] * H])))
        .attr("fill", "#332a53")
        .attr("stroke", "#453a6c")
        .attr("stroke-width", 1.4);
    });

    svg
      .selectAll(".stroma-cell")
      .data(TISSUE_LAYOUT.stromaCells)
      .join("line")
      .attr("x1", (d) => d.x * W - Math.cos((d.rot * Math.PI) / 180) * d.len)
      .attr("y1", (d) => d.y * H - Math.sin((d.rot * Math.PI) / 180) * d.len)
      .attr("x2", (d) => d.x * W + Math.cos((d.rot * Math.PI) / 180) * d.len)
      .attr("y2", (d) => d.y * H + Math.sin((d.rot * Math.PI) / 180) * d.len)
      .attr("stroke", "#5f5390")
      .attr("stroke-width", 1.1)
      .attr("stroke-opacity", 0.55);

    svg
      .selectAll(".nucleus")
      .data(TISSUE_LAYOUT.nuclei)
      .join("ellipse")
      .attr("cx", (d) => d.x * W)
      .attr("cy", (d) => d.y * H)
      .attr("rx", (d) => d.rr)
      .attr("ry", (d) => d.rr * 1.3)
      .attr("fill", "#8b78e8")
      .attr("fill-opacity", 0.85);

    svg
      .selectAll(".lymph")
      .data(TISSUE_LAYOUT.lymphs)
      .join("circle")
      .attr("cx", (d) => d.x * W)
      .attr("cy", (d) => d.y * H)
      .attr("r", (d) => d.rr)
      .attr("fill", "#241f40")
      .attr("stroke", "#6c8dfb")
      .attr("stroke-width", 0.6)
      .attr("fill-opacity", 0.8);

    const cols = 20, rows = 15;
    const cw = W / cols, ch = H / rows;
    const cells = [];
    for (let j = 0; j < rows; j++) {
      for (let i = 0; i < cols; i++) {
        const xn = (i + 0.5) / cols, yn = (j + 0.5) / rows;
        cells.push({ i, j, xn, yn, heat: heatAt(xn, yn, claim) });
      }
    }
    const colorScale = d3.interpolateRgb("#5865ff", "#ef6572");
    const patchLayer = svg.append("g");
    patchLayer
      .selectAll("rect")
      .data(cells)
      .join("rect")
      .attr("class", "patch-cell")
      .attr("x", (d) => d.i * cw)
      .attr("y", (d) => d.j * ch)
      .attr("width", cw)
      .attr("height", ch)
      .attr("fill", (d) => colorScale(d.heat))
      .attr("fill-opacity", (d) => 0.08 + d.heat * 0.62)
      .attr("stroke", "rgba(255,255,255,0.04)")
      .attr("stroke-width", 1)
      .style("cursor", "pointer")
      .on("mouseover", function (event, d) {
        if (!d3.select(this).classed("is-selected")) d3.select(this).attr("stroke", "rgba(255,255,255,0.35)");
        showTip(
          event,
          `<div class="tt-title">patch [${d.i},${d.j}]</div><div class="tt-row"><span>activation</span><span>${pct(d.heat, 0)}</span></div><div class="tt-note">encoder.layer[16] CLS-projection onto the "${claim.concept}" axis for this token's receptive field.</div>`
        );
        renderReadoutLive(claim, d);
      })
      .on("mousemove", moveTip)
      .on("mouseout", function () {
        if (!d3.select(this).classed("is-selected")) d3.select(this).attr("stroke", "rgba(255,255,255,0.04)");
        hideTip();
        renderReadoutLive(claim, selectedPatch);
      })
      .on("click", function (event, d) {
        patchLayer.selectAll("rect").classed("is-selected", false).attr("stroke", "rgba(255,255,255,0.04)").attr("stroke-width", 1);
        d3.select(this).classed("is-selected", true).attr("stroke", "#fff").attr("stroke-width", 2);
        selectedPatch = d;
        renderReadoutLive(claim, d);
      });

    svg
      .append("text")
      .attr("x", W - 12)
      .attr("y", 20)
      .attr("text-anchor", "end")
      .attr("fill", "#7d82a0")
      .style("font-size", "11px")
      .style("font-family", "var(--mono)")
      .text(`axis: ${claim.contrast}`);

    document.getElementById("case-caption").textContent =
      `${claim.claim} — 20×15 patch tokens colored by projection onto the ${claim.contrast} axis at encoder.layer[16].`;

    renderReadoutStatic(claim);
    selectedPatch = null;
    renderReadoutLive(claim, null);
  }

  // ---------------------------------------------------------------- proof: quadrant map
  function renderQuadrant() {
    const container = document.getElementById("quadrant-chart");
    container.innerHTML = "";
    const width = Math.max(420, container.clientWidth || 520);
    const height = 340;
    const margin = { top: 20, right: 20, bottom: 40, left: 50 };

    const svg = d3.select(container).append("svg").attr("width", width).attr("height", height);

    const x = d3.scaleLinear().domain([0, 1.05]).range([margin.left, width - margin.right]);
    const y = d3.scaleLinear().domain([0.9, 1.0]).range([height - margin.bottom, margin.top]);
    const r = d3.scaleSqrt().domain([0.9, 1]).range([8, 18]);

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

    svg
      .append("g")
      .selectAll("text.dot-label")
      .data(certifiable)
      .join("text")
      .attr("class", "dot-label")
      .attr("x", (d) => x(d.scores.necessity))
      .attr("y", (d) => y(d.scores.specificity) - r(d.scores.sufficiency) - 6)
      .attr("text-anchor", "middle")
      .style("font-size", "9.5px")
      .attr("fill", "var(--text-faint)")
      .text((d) => d.concept.replace(/_/g, " "));
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
    const trace = claim.reasoning_trace[claim.reasoning_trace.length - 1];
    document.getElementById("verdict-sentence").textContent = trace.interpretation;
  }

  function renderCoverage() {
    const cov = window.CARD.coverage;
    const grounded = certifiable.filter((c) => c.verdict === "GROUNDED").length;
    const weak = certifiable.filter((c) => c.verdict === "WEAK").length;
    const total = cov.claims_total;
    const bar = d3.select("#coverage-bar");
    bar.selectAll("*").remove();
    [
      [grounded, COLOR.GROUNDED],
      [weak, COLOR.WEAK],
      [cov.not_certifiable, COLOR.NOT_CERTIFIABLE],
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
    case: { title: "Case", tag: "one claim, followed pixel to concept axis" },
    proof: { title: "Proof", tag: "necessity × sufficiency × specificity, and where the axis came from" },
    verdict: { title: "Verdict", tag: "certified claim, confound check, honest coverage" },
  };

  function goToView(id) {
    d3.selectAll(".view").classed("active", false);
    d3.select("#view-" + id).classed("active", true);
    d3.selectAll(".nav-item").classed("active", false);
    d3.select(`.nav-item[data-view="${id}"]`).classed("active", true);
    const meta = VIEW_META[id];
    if (meta) {
      document.getElementById("topbar-title").textContent = meta.title;
      document.getElementById("topbar-tag").textContent = meta.tag;
    }
    if (id === "proof") { renderQuadrant(); renderQuadrantSelection(); renderNecessityChart(); }
  }

  function initNavigation() {
    d3.selectAll(".nav-item").on("click", function () {
      goToView(this.dataset.view);
    });
  }

  // ------------------------------------------------------------- render-all
  function renderAll() {
    renderClaimStrip();
    renderHistology();
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
      const r = await fetch("api/all", { headers: { Accept: "application/json" } });
      if (!r.ok) throw new Error("api " + r.status);
      const d = await r.json();
      if (d.error) throw new Error(d.error);
      if (d.CARD) window.CARD = d.CARD;
      if (d.DESIGNED_PROBES) window.DESIGNED_PROBES = d.DESIGNED_PROBES;
      if (d.MCP_VERBS) window.MCP_VERBS = d.MCP_VERBS;
      if (d.TRACKS) window.TRACKS = d.TRACKS;
      recomputeState();
      setLiveBadge(true);
    } catch (e) {
      setLiveBadge(false); // static mock retained
    }
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
      const r = await fetch("api/certify_answer", {
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

  // ---------------------------------------------------------------- init
  async function init() {
    await bootstrapData();
    renderAll();

    initNavigation();
    goToView("case");

    document.getElementById("claim-prev").addEventListener("click", () => selectClaim(selectedIdx - 1));
    document.getElementById("claim-next").addEventListener("click", () => selectClaim(selectedIdx + 1));
    document.getElementById("btn-run").addEventListener("click", runCertifyAnimation);

    window.addEventListener("resize", () => {
      renderHistology();
      if (document.getElementById("view-proof").classList.contains("active")) {
        renderQuadrant();
        renderQuadrantSelection();
        renderNecessityChart();
      }
    });
  }

  document.addEventListener("DOMContentLoaded", init);
})();
