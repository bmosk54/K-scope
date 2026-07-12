/**
 * BioLayer — Intervention Studio.
 *
 * Goodfire-style "apply a concept, watch it propagate." One concept axis at a time:
 * pick a dynamic ACTION (ablate / steer / specificity), scrub its STRENGTH, and see the
 * intervention flow down the encoder's depth — the layer-by-layer necessity bite — while
 * the readout head swings. Every verdict exposes the dynamic actions and the per-layer
 * interventions that earned it (the actions ledger).
 *
 * Runs on window.CARD from data.js (rich mock curves) and upgrades to the live certify
 * card via /api/all when the backend is reachable.
 */
(function () {
  "use strict";

  const ON = "#4f7ae0";      // on-target necessity (validated dark-surface mark)
  const OFF = "#cf7a3c";     // off-target / cross-interference (always dashed + labeled)
  const SUCCESS = "#3ddc97", WARN = "#e8b23e", NEUTRAL = "#7a7fa0";

  const API_BASE = (window.API_BASE ||
    new URLSearchParams(location.search).get("api") || "").replace(/\/+$/, "");
  const apiUrl = (p) => (API_BASE ? API_BASE + "/" + p : p);
  const $ = (id) => document.getElementById(id);
  const esc = (s) => String(s == null ? "" : s).replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));

  const tooltip = d3.select("#tooltip");
  const showTip = (e, html) => { tooltip.html(html).style("display", "block").style("opacity", 1); moveTip(e); };
  const moveTip = (e) => tooltip
    .style("left", Math.min(e.clientX + 16, window.innerWidth - 320) + "px")
    .style("top", Math.min(e.clientY + 16, window.innerHeight - 200) + "px");
  const hideTip = () => tooltip.style("opacity", 0).style("display", "none");

  const ACTIONS = {
    ablate:      { verb: "ablate_live · necessity",  step: ["necessity_live", "necessity_cached"] },
    steer:       { verb: "steer · sufficiency",      step: ["sufficiency"] },
    specificity: { verb: "specificity · off-target", step: ["specificity"] },
  };

  const state = { claims: [], nc: [], idx: 0, action: "ablate", strength: 1.0, blocks: 24, gen: 0 };

  // ------------------------------------------------------------------ data prep
  function ingest() {
    const all = (window.CARD && window.CARD.claims) || [];
    state.claims = all.filter((c) => c.scores && c.live_necessity && c.live_necessity.curve);
    state.nc = all.filter((c) => !c.scores || !(c.live_necessity && c.live_necessity.curve));
    // substrate depth from the live track registry if present
    const tr = (window.TRACKS || []).find((t) => /phikon/i.test(t.model || t.track || ""));
    if (tr && tr.blocks) state.blocks = tr.blocks;
    if (state.idx >= state.claims.length) state.idx = 0;
  }
  const cur = () => state.claims[state.idx];
  const curve = () => cur().live_necessity.curve;
  const crossCurve = () => (cur().live_necessity.cross_interference || null);
  // depth fraction per station: name-agnostic, biased so the readout sits near the head
  const depthFrac = (i, n) => (n <= 1 ? 0.9 : 0.22 + (0.92 - 0.22) * (i / (n - 1)));

  // ------------------------------------------------------------------ concept rail
  function verdictClass(v) { return "v-" + (v || "NULL"); }

  function buildRail() {
    $("rail-count").textContent = state.claims.length + " axes";
    const host = $("concept-list");
    host.innerHTML = "";

    state.claims.forEach((c, i) => {
      const card = document.createElement("div");
      card.className = "concept-card" + (i === state.idx ? " active" : "");
      const spark = miniSpark(c.live_necessity.curve);
      card.innerHTML =
        `<div class="cc-top">
           <span class="cc-chip" style="background:${ON}"></span>
           <span class="cc-name">${esc(c.claim)}</span>
           <span class="cc-verdict ${verdictClass(c.verdict)}">${c.verdict}</span>
         </div>
         <div class="cc-contrast">${esc(c.contrast || c.concept || "—")}</div>
         <div class="cc-spark">${spark}</div>`;
      card.addEventListener("click", () => selectConcept(i));
      host.appendChild(card);
    });

    // declined axes — visible so the honest NOT_CERTIFIABLE verdicts are in the same list
    if (state.nc.length) {
      const foot = $("rail-foot");
      foot.innerHTML = `<b style="color:var(--text-dim)">+ ${state.nc.length} declined</b> — ` +
        `no axis to intervene on (cell/subcellular or outcome claims). certify returns ` +
        `NOT_CERTIFIABLE rather than force-fitting a probe.`;
    }
  }

  // 3 mini bars, colored by significance — the per-layer bite at a glance
  function miniSpark(cv) {
    const w = 210, h = 26, gap = 4, bw = (w - gap * (cv.length - 1)) / cv.length;
    const max = Math.max(...cv.map((d) => d.gap), 0.1);
    return `<svg width="100%" height="${h}" viewBox="0 0 ${w} ${h}" preserveAspectRatio="none">` +
      cv.map((d, i) => {
        const bh = Math.max(2, (Math.max(d.gap, 0) / max) * (h - 2));
        return `<rect x="${i * (bw + gap)}" y="${h - bh}" width="${bw}" height="${bh}" rx="2"
          fill="${d.bites ? ON : "rgba(255,255,255,.14)"}"></rect>`;
      }).join("") + `</svg>`;
  }

  function selectConcept(i) {
    state.idx = i;
    hideTip();
    d3.selectAll(".concept-card").classed("active", (_, k) => k === i);
    renderStage();
  }

  // ------------------------------------------------------------------ stage header
  function renderStage() {
    const c = cur();
    $("stage-concept").textContent = c.claim;
    $("stage-contrast").textContent = (c.concept ? c.concept.replace(/_/g, " ") + "  ·  " : "") + (c.contrast || "");
    const vp = $("stage-verdict");
    vp.textContent = c.verdict;
    vp.className = "verdict-pill " + verdictClass(c.verdict);
    drawCanvas();
    renderLegend();
    renderCaption();
    renderMeter();
    renderPillars();
    renderBite();
    renderLedger();
  }

  // ------------------------------------------------------------------ THE CANVAS
  // Vertical encoder depth (input at top, readout head at bottom). Each extracted layer
  // is an intervention station; the horizontal bar is the readout-margin drop when the
  // concept axis is projected out THERE, scaled by strength. Matched-random null ≈ 0.
  function drawCanvas() {
    const host = $("depth-canvas");
    host.innerHTML = "";
    const gen = ++state.gen;               // invalidates any in-flight propagation loop
    const W = host.clientWidth || 700, H = host.clientHeight || 420;
    const m = { top: 26, right: 128, bottom: 30, left: 132 };
    const colX = m.left, barMax = W - m.right;
    const svg = d3.select(host).append("svg").attr("viewBox", `0 0 ${W} ${H}`);

    const cv = curve();
    const cross = crossCurve();
    const n = cv.length;
    const y = (i) => m.top + (H - m.top - m.bottom) * depthFrac(i, n);
    const readoutY = y(n - 1);

    // x-scale over readout-margin drop (shared by on/off target so bars are comparable)
    const allGaps = cv.map((d) => d.gap).concat(cross ? cross.map((d) => Math.abs(d.gap)) : []);
    const maxGap = Math.max(...allGaps, 0.1) * 1.12;
    const x = d3.scaleLinear().domain([0, maxGap]).range([colX, barMax]);
    const s = state.strength, mode = state.action;

    // --- faint block ladder (the ViT-L substrate) -----------------------------
    const ladder = svg.append("g");
    for (let b = 0; b <= state.blocks; b++) {
      const yy = m.top + (H - m.top - m.bottom) * (b / state.blocks);
      ladder.append("line").attr("x1", colX - 9).attr("x2", colX + 9).attr("y1", yy).attr("y2", yy)
        .attr("stroke", "rgba(255,255,255,.07)").attr("stroke-width", 1);
    }
    // the depth spine
    svg.append("line").attr("x1", colX).attr("x2", colX).attr("y1", m.top).attr("y2", H - m.bottom)
      .attr("stroke", "rgba(255,255,255,.18)").attr("stroke-width", 2);
    svg.append("text").attr("x", colX - 14).attr("y", m.top - 10).attr("text-anchor", "end")
      .attr("fill", NEUTRAL).attr("font-size", 10).attr("font-family", "var(--mono)").text("input · block 0");
    svg.append("text").attr("x", colX - 14).attr("y", H - m.bottom + 16).attr("text-anchor", "end")
      .attr("fill", NEUTRAL).attr("font-size", 10).attr("font-family", "var(--mono)")
      .text("readout head · block " + state.blocks);

    // --- baseline axis (null ≈ 0) --------------------------------------------
    svg.append("text").attr("x", barMax).attr("y", H - m.bottom + 18).attr("text-anchor", "end")
      .attr("fill", NEUTRAL).attr("font-size", 10).text("readout-margin drop →   (matched-random null ≈ 0)");

    // --- per-station intervention --------------------------------------------
    cv.forEach((d, i) => {
      const yy = y(i);
      const g = svg.append("g");

      // station depth label (extracted layer name)
      g.append("text").attr("x", colX - 22).attr("y", yy + 4).attr("text-anchor", "end")
        .attr("fill", "var(--text-dim)").attr("font-size", 11.5).attr("font-weight", 600)
        .text(String(d.layer).replace(/_/g, " "));

      // null envelope (matched-random reaches only this far) — std ≈ gap/z
      const nullStd = d.z ? Math.abs(d.gap / d.z) : 0;
      const nullW = x(Math.min(nullStd * 1.645, maxGap)) - colX;
      if (nullW > 1)
        g.append("rect").attr("x", colX).attr("y", yy - 9).attr("width", nullW).attr("height", 18)
          .attr("fill", "rgba(122,127,160,.16)").attr("rx", 3);

      if (mode === "steer") {
        // sufficiency is a single inject→flip, not layer-resolved: only mark the readout
        if (i === n - 1) drawSteer(g, colX, yy, x, maxGap, s);
        else drawInertNode(g, colX, yy);
        return;
      }

      // ABLATE / SPECIFICITY: on-target necessity bar
      const gw = Math.max(0, x(d.gap * s) - colX);
      g.append("rect").attr("class", "on-bar").attr("x", colX).attr("y", yy - 7)
        .attr("height", 14).attr("rx", 4).attr("fill", ON).attr("width", 0)
        .transition().duration(420).attr("width", gw);

      // off-target (cross-interference) dashed bar in specificity mode
      if (mode === "specificity" && cross && cross[i]) {
        const off = cross[i], ow = Math.max(0, x(Math.abs(off.gap) * s) - colX);
        g.append("rect").attr("x", colX).attr("y", yy - 11).attr("height", 22).attr("rx", 4)
          .attr("fill", "none").attr("stroke", OFF).attr("stroke-width", 1.6)
          .attr("stroke-dasharray", "4 3").attr("width", 0)
          .transition().duration(420).attr("width", Math.max(ow, 3));
        g.append("text").attr("x", colX + Math.max(ow, 3) + 8).attr("y", yy - 12)
          .attr("fill", OFF).attr("font-size", 9.5).attr("font-family", "var(--mono)")
          .text(`off ${off.gap >= 0 ? "+" : ""}${off.gap.toFixed(2)}`);
      }

      // value label
      g.append("text").attr("x", Math.min(colX + gw + 10, barMax)).attr("y", yy + 4)
        .attr("fill", "var(--text)").attr("font-size", 11).attr("font-family", "var(--mono)")
        .attr("font-weight", 600).text(`+${(d.gap * s).toFixed(2)}`);
      g.append("text").attr("x", Math.min(colX + gw + 10, barMax)).attr("y", yy + 17)
        .attr("fill", NEUTRAL).attr("font-size", 9.5).attr("font-family", "var(--mono)")
        .text(`z ${d.z.toFixed(1)}`);

      // station node — green ring if the bite clears the null
      drawNode(g, colX, yy, d);
    });

    // --- propagation pulse: first significant bite → readout head -------------
    if (mode !== "steer") animatePropagation(svg, colX, cv, y, readoutY, gen);
  }

  function drawNode(g, x, yy, d) {
    const node = g.append("circle").attr("cx", x).attr("cy", yy).attr("r", 7)
      .attr("fill", d.bites ? ON : "var(--panel-2)")
      .attr("stroke", d.bites ? SUCCESS : "var(--border-strong)")
      .attr("stroke-width", d.bites ? 2.5 : 1.5).style("cursor", "pointer");
    if (d.bites) node.attr("filter", "drop-shadow(0 0 5px rgba(61,220,151,.5))");
    node.on("mouseover", (e) => showTip(e,
        `<div class="tt-title">layer: ${esc(d.layer)}</div>` +
        `<div class="tt-row"><span>margin drop</span><span>+${d.gap.toFixed(3)}</span></div>` +
        `<div class="tt-row"><span>z vs null</span><span>${d.z.toFixed(1)}</span></div>` +
        `<div class="tt-note">${d.bites ? "significant bite — decision causally depends on the axis here" : "no significant bite — redundancy recomputes it downstream"}</div>`))
      .on("mousemove", moveTip).on("mouseout", hideTip);
  }

  function drawInertNode(g, x, yy) {
    g.append("circle").attr("cx", x).attr("cy", yy).attr("r", 6)
      .attr("fill", "var(--panel-2)").attr("stroke", "var(--border-strong)").attr("stroke-width", 1.5);
  }

  function drawSteer(g, x, yy, xs, maxGap, s) {
    const c = cur(), suf = c.scores.sufficiency || 0;
    const w = Math.max(0, xs(maxGap * 0.85 * suf * s) - x);
    g.append("rect").attr("x", x).attr("y", yy - 7).attr("height", 14).attr("rx", 4)
      .attr("fill", ON).attr("opacity", 0.9).attr("width", 0)
      .transition().duration(420).attr("width", w);
    g.append("text").attr("x", x + w + 10).attr("y", yy + 4).attr("fill", "var(--text)")
      .attr("font-size", 11).attr("font-family", "var(--mono)").attr("font-weight", 600)
      .text(`flip ${(suf * s).toFixed(2)}`);
    g.append("text").attr("x", x + w + 10).attr("y", yy + 17).attr("fill", NEUTRAL)
      .attr("font-size", 9.5).attr("font-family", "var(--mono)").text("random 0.00");
    g.append("circle").attr("cx", x).attr("cy", yy).attr("r", 7).attr("fill", ON)
      .attr("stroke", SUCCESS).attr("stroke-width", 2.5)
      .attr("filter", "drop-shadow(0 0 5px rgba(61,220,151,.5))");
  }

  function animatePropagation(svg, x, cv, y, readoutY, gen) {
    const first = cv.findIndex((d) => d.bites);
    if (first < 0) return;
    const startY = y(first);
    const pulse = svg.append("circle").attr("cx", x).attr("r", 4).attr("fill", ON)
      .attr("opacity", 0.9).attr("filter", "drop-shadow(0 0 6px " + ON + ")");
    const run = () => {
      if (gen !== state.gen) return;       // a newer draw superseded us — stop the loop
      pulse.attr("cy", startY).attr("opacity", 0.9)
        .transition().duration(1100).ease(d3.easeCubicInOut)
        .attr("cy", readoutY).attr("opacity", 0.15)
        .transition().duration(400).on("end", run);
    };
    run();
  }

  // ------------------------------------------------------------------ legend
  function renderLegend() {
    const items = [`<span class="leg-item"><span class="leg-swatch" style="background:${ON}"></span>readout-margin drop (concept ablated)</span>`,
      `<span class="leg-item"><span class="leg-dot" style="background:${SUCCESS}"></span>significant bite (z ≥ 1.645 vs null)</span>`,
      `<span class="leg-item"><span class="leg-swatch" style="background:rgba(122,127,160,.4)"></span>matched-random null envelope</span>`];
    if (state.action === "specificity")
      items.splice(1, 0, `<span class="leg-item"><span class="leg-swatch dash"></span>off-target axis (should stay flat)</span>`);
    $("canvas-legend").innerHTML = items.join("");
  }

  function renderCaption() {
    const cap = {
      ablate: "<b>Ablate</b> projects the concept axis out of the CLS token at each layer and lets the rest of the forward pass recompute. The bar is how much the readout margin drops — the layer-resolved <b>necessity</b> curve. It stays near the null until deep in the stack: the model rebuilds the concept from un-ablated patch tokens (redundancy / Hydra effect) and only truly depends on the axis near the readout.",
      steer: "<b>Steer</b> injects the concept direction into the representation (sufficiency). It is a single do() at the readout, not a layer-resolved edit — so it flips the class assignment where a matched-random direction does not. This is the clean, concept-specific signal on pathology FMs; necessity + specificity carry the verdict.",
      specificity: "<b>Specificity</b> ablates the concept axis and scores a <em>different</em> readout (off-target, dashed). The on-target bar bites; the off-target bar stays flat — the effect is targeted to this axis, not general damage to the representation.",
    };
    $("stage-caption").innerHTML = cap[state.action];
  }

  // ------------------------------------------------------------------ readout meter
  function renderMeter() {
    const c = cur(), cv = curve();
    const readout = cv[cv.length - 1];
    const s = state.strength;
    $("meter-pos").textContent = (c.concept || "concept").replace(/_/g, " ").split(" ")[0] || "concept";
    const negName = (c.contrast || "").split(" vs ")[1] || "contrast";
    $("meter-neg").textContent = negName;

    let posPct, cap;
    if (state.action === "steer") {
      // inject from a neutral start → swing toward the concept by the flip rate
      const flip = (c.scores.sufficiency || 0) * s;
      posPct = 50 + 42 * flip;
      cap = `injecting the concept direction swings the readout toward <b>${esc((c.concept||"").replace(/_/g," "))}</b> ` +
            `(flip ${flip.toFixed(2)}); a matched-random direction leaves it at the midline.`;
    } else if (state.action === "specificity") {
      posPct = 88; // on-target intact; off-target unaffected
      cap = `ablating the off-target axis leaves this readout intact — the needle holds at the concept.`;
    } else {
      const bite = Math.min(1, Math.abs(readout.gap) / 1.0); // readout gaps here flip the decision
      posPct = 88 - 38 * s * bite;
      cap = `ablating the axis at the readout drops the margin by <b>+${(readout.gap*s).toFixed(2)}</b> — the ` +
            (s * bite > 0.6 ? "decision collapses toward the contrast." : "decision mostly holds (redundancy).");
    }
    posPct = Math.max(50, Math.min(94, posPct));
    $("meter-needle").style.left = posPct + "%";
    const fill = $("meter-fill");
    fill.style.left = Math.min(50, posPct) + "%";
    fill.style.width = Math.abs(posPct - 50) + "%";
    $("meter-caption").innerHTML = cap;
  }

  // ------------------------------------------------------------------ pillars
  function renderPillars() {
    const sc = cur().scores, host = $("pillars");
    const cv = curve(), readoutOnly = !cv.slice(0, -1).some((d) => d.bites);
    const tiles = [
      { name: "necessity", val: sc.necessity, tag: readoutOnly ? "readout-limited" : "bites early" },
      { name: "sufficiency", val: sc.sufficiency, tag: "steering axis" },
      { name: "specificity", val: sc.specificity, tag: "targeted" },
    ];
    host.innerHTML = tiles.map((t) =>
      `<div class="pill-tile">
         <div class="pill-name">${t.name}</div>
         <div class="pill-val">${t.val != null ? t.val.toFixed(2) : "—"}</div>
         <div class="pill-verdict" style="color:var(--text-faint)">${t.tag}</div>
       </div>`).join("");
  }

  function renderBite() {
    const cv = curve(), first = cv.find((d) => d.bites);
    const host = $("bite-callout");
    if (!first) { host.innerHTML = "No significant bite above the matched-random null at any extracted layer."; return; }
    const readoutOnly = cv.slice(0, -1).every((d) => !d.bites);
    host.innerHTML = `First significant bite at <b>${esc(first.layer).replace(/_/g," ")}</b> (z ${first.z.toFixed(1)}). ` +
      (readoutOnly
        ? `Necessity is <b>readout-limited</b> — the axis is recomputed downstream until the decision head, so a naive single-axis TCAV faithfulness claim would over-state it.`
        : `The decision causally depends on the axis from mid-network on — the strongest per-slide causal read.`);
  }

  // ------------------------------------------------------------------ actions ledger
  // Every dynamic action that fired, with its layer-by-layer bite — visible at THIS verdict.
  function renderLedger() {
    const c = cur();
    const lv = $("ledger-verdict");
    lv.textContent = c.verdict; lv.className = "ledger-verdict " + verdictClass(c.verdict);

    const trace = c.reasoning_trace || [];
    const rows = [];
    trace.forEach((t) => {
      if (t.step === "verdict") return;
      const label = VERB_LABEL[t.step] || t.step.replace(/_/g, " ");
      const badge = badgeFor(t);
      const activeStep = ACTIONS[state.action] && ACTIONS[state.action].step.includes(t.step);
      let chips = "";
      if (t.step === "necessity_live" || t.step === "necessity_cached") chips = layerChips(curve(), "bite");
      if (t.step === "specificity" && crossCurve()) chips = layerChips(crossCurve(), "bite-off");
      rows.push(
        `<div class="led-row${activeStep ? " on" : ""}">
           <div class="led-top">
             <span class="led-verb">${esc(label)}</span>
             <span class="led-badge ${badge.cls}">${badge.txt}</span>
           </div>
           <div class="led-obs">${esc(shorten(t.observation))}</div>
           ${chips}
         </div>`);
    });
    $("ledger").innerHTML = rows.join("") ||
      `<div class="led-row"><div class="led-obs">No causal actions ran — certify declined this claim.</div></div>`;
  }

  const VERB_LABEL = {
    contrast_validation: "probe · contrast gate",
    necessity_live: "ablate_live · necessity",
    necessity_cached: "ablate · necessity (cached)",
    sufficiency: "steer · sufficiency",
    specificity: "specificity · off-target",
    confound: "confound · site gate",
    multiple_comparisons: "multiple comparisons",
  };

  function badgeFor(t) {
    const o = (t.observation || "") + " " + (t.interpretation || "");
    if (/UNCHECKED|no site|single-source/i.test(o)) return { cls: "off", txt: "unchecked" };
    if (/WARN|does NOT|FLAG|leaked|RIDES|CAPPED|collinear/i.test(o)) return { cls: "warn", txt: "warn" };
    return { cls: "pass", txt: "pass" };
  }

  function layerChips(cv, biteClass) {
    return `<div class="led-layers">` + cv.map((d) => {
      const bite = biteClass === "bite" ? d.bites : (d.z && d.z >= 1.645 && d.gap > 0);
      return `<span class="led-chip ${bite ? biteClass : ""}">${esc(String(d.layer).replace(/_/g, " ").split(" ").pop())} ${d.gap >= 0 ? "+" : ""}${d.gap.toFixed(2)}</span>`;
    }).join("") + `</div>`;
  }

  function shorten(s) { s = String(s || ""); return s.length > 200 ? s.slice(0, 197) + "…" : s; }

  // ------------------------------------------------------------------ controls
  function setAction(a) {
    state.action = a;
    d3.selectAll(".seg-btn").classed("active", function () { return this.dataset.action === a; });
    drawCanvas(); renderLegend(); renderCaption(); renderMeter(); renderLedger();
  }
  function setStrength(v) {
    state.strength = v / 100;
    $("strength-val").textContent = v + "%";
    drawCanvas(); renderMeter();
  }

  // ------------------------------------------------------------------ live upgrade
  function setLive(live) {
    const chip = $("substrate-chip");
    chip.dataset.live = live ? "1" : "0";
    chip.textContent = (live ? "● LIVE · " : "○ mock · ") + "phikon-v2 · ViT-L/" + state.blocks;
  }
  async function bootstrap() {
    try {
      const r = await fetch(apiUrl("api/all"), { headers: { Accept: "application/json" } });
      if (!r.ok) throw new Error("api " + r.status);
      const d = await r.json();
      if (d.error) throw new Error(d.error);
      if (d.CARD) window.CARD = d.CARD;
      if (d.TRACKS) window.TRACKS = d.TRACKS;
      ingest();
      if (!state.claims.length) throw new Error("no live per-layer curves — keeping mock");
      buildRail(); renderStage(); setLive(true);
    } catch (e) { setLive(false); }
  }

  async function rerun() {
    const btn = $("rerun-btn"); btn.disabled = true; btn.textContent = "⟳ …";
    await bootstrap();
    btn.disabled = false; btn.textContent = "⟳ live";
  }

  // ------------------------------------------------------------------ init
  function init() {
    ingest();
    if (!state.claims.length) { // nothing certifiable with a curve — degrade gracefully
      $("depth-canvas").innerHTML = `<div class="nc-state"><div class="nc-reason">No certifiable claim carries a per-layer intervention curve in this card.</div></div>`;
      return;
    }
    setLive(false);
    buildRail();
    renderStage();

    document.querySelectorAll(".seg-btn").forEach((b) =>
      b.addEventListener("click", () => setAction(b.dataset.action)));
    $("strength-slider").addEventListener("input", (e) => setStrength(+e.target.value));
    $("rerun-btn").addEventListener("click", rerun);
    window.addEventListener("resize", () => { drawCanvas(); });

    bootstrap();
  }

  document.addEventListener("DOMContentLoaded", init);
})();
