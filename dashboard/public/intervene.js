/**
 * BioLayer — Intervention Studio.
 *
 * Rebuilt around the LIVE layer-by-layer intervention (card.claims[].live_necessity),
 * NOT the cached readout projection and NOT a spatial per-tile field. Our ablate acts on
 * the pooled CLS representation — there is no image location, so the degree of freedom we
 * expose is DEPTH:
 *
 *   LAYER STRIP   pick which block's CLS to project the concept axis out of.
 *   STRENGTH      scrub the ablation magnitude (interpolates the precomputed live_necessity).
 *   CURVE         accumulating decision-margin drop by depth — concept trace + matched-random
 *                 null band. The monotone rise IS the finding; the flat null is the control.
 *   READOUT       decision-margin drop at the selected site, two traces overlaid (concept vs null).
 *
 * The claim rail shows the FULL verdict spread — GROUNDED, WEAK (with the |r| that capped it),
 * and NOT_CERTIFIABLE (with the reason). A 9/9-green list is a rubber stamp; the failures are
 * the product.
 *
 * Data source: the exact card the cockpit just certified (localStorage) -> live /api/all ->
 * the data.js mock (offline, badged). live_necessity ships inside the card — no extra fetch.
 */
(function () {
  "use strict";

  const V = { GROUNDED: "#3ddc97", WEAK: "#e8b23e", NOT_CERTIFIABLE: "#7a7fa0" };
  const NULLC = "#7a7fa0";          // matched-random null trace/band
  const EPS = 1e-6;

  const API_BASE = (window.API_BASE ||
    new URLSearchParams(location.search).get("api") || "").replace(/\/+$/, "");
  const apiUrl = (p) => (API_BASE ? API_BASE + "/" + p : p);
  const $ = (id) => document.getElementById(id);
  const esc = (s) => String(s == null ? "" : s).replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));
  const clamp = (v, a, b) => Math.max(a, Math.min(b, v));

  const tooltip = d3.select("#tooltip");
  const showTip = (e, html) => { tooltip.html(html).style("display", "block").style("opacity", 1); moveTip(e); };
  const moveTip = (e) => tooltip
    .style("left", Math.min(e.clientX + 16, window.innerWidth - 320) + "px")
    .style("top", Math.min(e.clientY + 16, window.innerHeight - 200) + "px");
  const hideTip = () => tooltip.style("opacity", 0).style("display", "none");

  const state = {
    card: null, source: "mock", track: "phikon",
    all: [], selectedId: null, selLayer: 0, strength: 1.0, blocks: 24,
  };

  // ------------------------------------------------------------------ data load
  async function loadCard() {
    try {
      const stash = JSON.parse(localStorage.getItem("biolayer:lastCard") || "null");
      if (stash && stash.card && stash.card.claims) {
        state.card = stash.card; state.track = stash.track || "phikon"; state.source = "certify";
        return;
      }
    } catch (e) { /* ignore bad stash */ }
    try {
      const r = await fetch(apiUrl("api/all"), { headers: { Accept: "application/json" } });
      if (r.ok) {
        const d = await r.json();
        if (!d.error && d.CARD && d.CARD.claims) {
          state.card = d.CARD;
          state.track = (d._meta && d._meta.track) || "phikon";
          state.source = "live";
          if (d.TRACKS) window.TRACKS = d.TRACKS;
          return;
        }
      }
    } catch (e) { /* fall through to mock */ }
    state.card = window.CARD; state.source = "mock";
  }

  function ingest() {
    state.all = (state.card && state.card.claims) || [];
    const tr = (window.TRACKS || []).find((t) => /phikon/i.test(t.model || t.track || ""));
    if (tr && tr.blocks) state.blocks = tr.blocks;
    // default selection: first interventable claim (falls back to first claim)
    if (!state.all.some((c) => c.id === state.selectedId)) {
      const first = state.all.find(isInterventable) || state.all[0];
      state.selectedId = first ? first.id : null;
    }
  }

  const selected = () => state.all.find((c) => c.id === state.selectedId) || null;
  const isInterventable = (c) =>
    !!(c && c.scores && c.live_necessity && Array.isArray(c.live_necessity.curve) && c.live_necessity.curve.length);
  const curveOf = (c) => (c && c.live_necessity && c.live_necessity.curve) || [];
  const nullStd = (pt) => (pt.z ? Math.abs(pt.gap / pt.z) : 0);   // matched-random null σ ≈ gap/z (null mean ~0)
  const capR = (c) => c && c.contrast_validation && c.contrast_validation.intensity_collinearity;

  // block index for the i-th extracted layer (last = readout = B-1; gives 8/16/23 on ViT-L/24)
  function blockFor(i, n) {
    const B = state.blocks;
    return i === n - 1 ? B - 1 : Math.round(B * (i + 1) / n);
  }
  function axisOf(c) {
    const m = /^(.+?)\s+vs\s+(.+)$/.exec((c && c.contrast) || "");
    return m ? { pos: m[1].trim(), neg: m[2].trim() } : null;
  }
  const verdictClass = (v) => "v-" + (v || "NULL");

  // ------------------------------------------------------------------ concept rail
  function buildSummary() {
    const counts = { GROUNDED: 0, WEAK: 0, NOT_CERTIFIABLE: 0 };
    state.all.forEach((c) => { counts[c.verdict] = (counts[c.verdict] || 0) + 1; });
    const order = ["GROUNDED", "WEAK", "NOT_CERTIFIABLE"];
    $("verdict-summary").innerHTML = order.map((v) =>
      `<span class="vs-item ${verdictClass(v)}"><b>${counts[v] || 0}</b> ${v.replace("NOT_CERTIFIABLE", "NOT CERT")}</span>`
    ).join("");
    $("rail-count").textContent = state.all.length + " claims";
  }

  function buildRail() {
    const host = $("concept-list"); host.innerHTML = "";
    state.all.forEach((c) => {
      const inter = isInterventable(c);
      const card = document.createElement("div");
      card.className = "concept-card " + verdictClass(c.verdict) +
        (c.id === state.selectedId ? " active" : "") + (inter ? "" : " declined");
      card.dataset.id = c.id;

      let sub = "";
      if (c.verdict === "WEAK" && capR(c) != null) {
        sub = `<div class="cc-note warn">capped · rides intensity |r| = ${capR(c).toFixed(3)}</div>`;
      } else if (!inter) {
        sub = `<div class="cc-note dim">${esc(c.reason || "no axis to intervene on")}</div>`;
      }

      card.innerHTML =
        `<div class="cc-top">
           <span class="cc-name">${esc(c.claim)}</span>
           <span class="cc-verdict ${verdictClass(c.verdict)}">${c.verdict}</span>
         </div>
         <div class="cc-contrast">${esc(c.contrast || (c.concept ? c.concept.replace(/_/g, " ") : "—"))}</div>
         ${inter ? `<div class="cc-gaps" id="ccg-${c.id}"></div>` : ""}
         ${sub}`;
      card.addEventListener("click", () => selectClaim(c.id));
      host.appendChild(card);
      if (inter) paintCardGaps(c);
    });
  }

  // tiny 3-bar gap sparkline in the rail card — the monotone rise at a glance
  function paintCardGaps(c) {
    const host = $("ccg-" + c.id); if (!host) return;
    const curve = curveOf(c), max = Math.max(...curve.map((p) => p.gap), EPS);
    host.innerHTML = curve.map((p) =>
      `<span class="ccg-bar" style="height:${(4 + 18 * (p.gap / max)).toFixed(1)}px;background:${p.bites ? V[c.verdict] : "var(--border-strong)"}"></span>`
    ).join("");
  }

  function selectClaim(id) {
    state.selectedId = id; hideTip();
    const c = selected();
    // land on the layer that bites hardest (max gap) so the strongest signal reads first
    if (isInterventable(c)) {
      const curve = curveOf(c);
      state.selLayer = curve.reduce((best, p, i) => (p.gap > curve[best].gap ? i : best), 0);
    }
    d3.selectAll(".concept-card").classed("active", function () { return this.dataset.id === id; });
    renderStage();
  }

  // ------------------------------------------------------------------ stage
  function renderStage() {
    const c = selected();
    $("stage-concept").textContent = c ? c.claim : "—";
    $("stage-contrast").textContent = c ? ((c.concept ? c.concept.replace(/_/g, " ") + "  ·  " : "") + (c.contrast || "")) : "";
    const vp = $("stage-verdict"); vp.textContent = c ? c.verdict : "—"; vp.className = "verdict-pill " + verdictClass(c && c.verdict);

    const inter = isInterventable(c);
    const ioi = inter && c.live_necessity.intervened_on_input;
    $("intervened-badge").innerHTML = ioi
      ? `<span class="ioi on" title="the do() was applied on this slide's forward pass, not a cached embedding">✓ intervened_on_input · per-slide forward pass</span>`
      : "";

    // toggle the ablation UI vs a declined empty-state
    ["layerstrip-wrap", "control-bar", "curve-wrap"].forEach((idn) => { $(idn).style.display = inter ? "" : "none"; });

    if (!inter) { renderDeclined(c); renderPillarsEmpty(); renderLedger(); return; }

    renderLayerStrip();
    renderCurve();
    renderReadout();
    renderPillars();
    renderBite();
    renderLedger();
    renderCaption();
    renderCurveLegend();
  }

  function renderDeclined(c) {
    $("necessity-curve").innerHTML = "";
    $("readout-block").innerHTML =
      `<div class="nc-state">
         <div class="nc-badge v-NOT_CERTIFIABLE">NOT CERTIFIABLE</div>
         <div class="nc-reason">${esc((c && c.reason) || "No causal axis for this claim — certify returns NOT_CERTIFIABLE rather than force-fitting a probe.")}</div>
         <div class="nc-foot">No layer intervention runs: there is no concept axis on this substrate to project out.</div>
       </div>`;
    $("stage-caption").innerHTML =
      "This claim was <b>declined</b>. The battery never ran — no axis, no ablation, no null. Showing it (rather than dropping it) is the honesty the flow depends on.";
    $("bite-callout").innerHTML = "";
  }

  // ------------------------------------------------------------------ LAYER STRIP
  function renderLayerStrip() {
    const c = selected(), curve = curveOf(c), n = curve.length, s = state.strength;
    const host = $("layer-strip"); host.innerHTML = "";
    curve.forEach((p, i) => {
      const node = document.createElement("button");
      node.className = "layer-node" + (i === state.selLayer ? " sel" : "") + (p.bites ? " bites" : "");
      node.style.setProperty("--v", V[c.verdict]);
      node.innerHTML =
        `<div class="ln-block">block ${blockFor(i, n)}</div>
         <div class="ln-name">${esc(String(p.layer).replace(/_/g, " "))}</div>
         <div class="ln-gap">Δ ${(p.gap * s).toFixed(2)}</div>
         <div class="ln-z">z ${p.z.toFixed(1)} · ${p.bites ? "bites" : "n.s."}</div>`;
      node.addEventListener("click", () => setLayer(i));
      host.appendChild(node);
      if (i < n - 1) { const a = document.createElement("div"); a.className = "ln-arrow"; a.textContent = "→"; host.appendChild(a); }
    });
  }

  // ------------------------------------------------------------------ THE CURVE (depth × margin-drop)
  function renderCurve() {
    const c = selected(), curve = curveOf(c), n = curve.length, s = state.strength;
    const host = $("necessity-curve"); host.innerHTML = "";
    const W = Math.max(420, host.clientWidth || 620), H = 300;
    const m = { top: 26, right: 26, bottom: 46, left: 56 };
    const svg = d3.select(host).append("svg").attr("viewBox", `0 0 ${W} ${H}`).attr("width", "100%").attr("height", H);

    const x = d3.scalePoint().domain(d3.range(n)).range([m.left, W - m.right]).padding(0.5);
    // fixed y-domain to the FULL-strength curve so scrubbing animates within a stable frame
    const maxGap = Math.max(...curve.map((p) => p.gap), 0.1) * 1.15;
    const y = d3.scaleLinear().domain([-maxGap * 0.06, maxGap]).range([H - m.bottom, m.top]);

    const gap = (i) => curve[i].gap * s;
    const nsd = (i) => nullStd(curve[i]) * s;

    // gridlines
    svg.append("g").selectAll("line").data(y.ticks(5)).join("line")
      .attr("class", "gridline").attr("x1", m.left).attr("x2", W - m.right).attr("y1", y).attr("y2", y);

    // axes
    svg.append("g").attr("class", "axis").attr("transform", `translate(${m.left},0)`).call(d3.axisLeft(y).ticks(5).tickFormat(d3.format(".1f")));
    svg.append("text").attr("transform", `translate(16,${(m.top + H - m.bottom) / 2}) rotate(-90)`)
      .attr("text-anchor", "middle").attr("fill", "var(--text-faint)").style("font-size", "10.5px").text("Δ decision margin (logit)");

    // matched-random null band (±1σ around 0) — must be visible and flat
    const band = d3.area().x((d, i) => x(i)).y0((d, i) => y(-nsd(i))).y1((d, i) => y(nsd(i))).curve(d3.curveMonotoneX);
    svg.append("path").datum(curve).attr("d", band).attr("fill", NULLC).attr("fill-opacity", 0.3);
    svg.append("line").attr("x1", m.left).attr("x2", W - m.right).attr("y1", y(0)).attr("y2", y(0))
      .attr("stroke", NULLC).attr("stroke-width", 1).attr("stroke-dasharray", "4 4").attr("opacity", 0.65);

    // guide line at the selected ablation site
    svg.append("line").attr("class", "sel-guide")
      .attr("x1", x(state.selLayer)).attr("x2", x(state.selLayer)).attr("y1", m.top).attr("y2", H - m.bottom)
      .attr("stroke", "var(--border-strong)").attr("stroke-width", 1).attr("stroke-dasharray", "2 3");

    // concept ablation trace
    const line = d3.line().x((d, i) => x(i)).y((d, i) => y(gap(i))).curve(d3.curveMonotoneX);
    svg.append("path").datum(curve).attr("d", line).attr("fill", "none").attr("stroke", V[c.verdict]).attr("stroke-width", 2.6);

    // points + per-layer labels
    curve.forEach((p, i) => {
      const g = svg.append("g");
      g.append("circle").attr("cx", x(i)).attr("cy", y(gap(i))).attr("r", i === state.selLayer ? 7 : 5)
        .attr("fill", p.bites ? V[c.verdict] : "var(--panel)")
        .attr("stroke", i === state.selLayer ? "var(--text)" : V[c.verdict]).attr("stroke-width", 2)
        .style("cursor", "pointer")
        .on("click", () => setLayer(i))
        .on("mouseover", (e) => showTip(e,
          `<div class="tt-title">block ${blockFor(i, n)} · ${esc(p.layer)}</div>` +
          `<div class="tt-row"><span>margin drop</span><span>+${gap(i).toFixed(3)}</span></div>` +
          `<div class="tt-row"><span>matched-random null σ</span><span>${nsd(i).toFixed(3)}</span></div>` +
          `<div class="tt-row"><span>z vs null</span><span>${p.z.toFixed(1)}</span></div>` +
          `<div class="tt-note">${p.bites ? "significant necessity bite" : "no significant bite (redundancy)"}</div>`))
        .on("mousemove", moveTip).on("mouseout", hideTip);
      // z annotation above biting points
      g.append("text").attr("x", x(i)).attr("y", y(gap(i)) - (i === state.selLayer ? 13 : 11))
        .attr("text-anchor", "middle").attr("fill", "var(--text-faint)").style("font-size", "9.5px")
        .text(`z ${p.z.toFixed(0)}`);
      // x label: block + layer name
      g.append("text").attr("x", x(i)).attr("y", H - m.bottom + 18).attr("text-anchor", "middle")
        .attr("fill", "var(--text-dim)").style("font-size", "11px").text(`block ${blockFor(i, n)}`);
      g.append("text").attr("x", x(i)).attr("y", H - m.bottom + 32).attr("text-anchor", "middle")
        .attr("fill", "var(--text-faint)").style("font-size", "9.5px").text(String(p.layer).replace(/_/g, " "));
    });
  }

  function renderCurveLegend() {
    const c = selected();
    $("curve-legend").innerHTML =
      `<span class="leg-item"><span class="leg-swatch" style="background:${V[c.verdict]}"></span>concept ablation (this claim)</span>` +
      `<span class="leg-item"><span class="leg-swatch" style="background:${NULLC};opacity:.5"></span>matched-random null (±1σ)</span>` +
      `<span class="leg-item"><span class="leg-dot"></span>selected ablation site</span>`;
  }

  // ------------------------------------------------------------------ READOUT (two traces overlaid)
  function renderReadout() {
    const c = selected(), curve = curveOf(c), i = state.selLayer, p = curve[i], s = state.strength, n = curve.length;
    const conceptDrop = p.gap * s, nullDrop = nullStd(p) * s;
    const ref = Math.max(...curve.map((q) => q.gap), EPS);       // shared scale = full-strength max gap
    const ratio = nullDrop > EPS ? conceptDrop / nullDrop : Infinity;

    $("ro-sub").textContent = `block ${blockFor(i, n)} · ${String(p.layer).replace(/_/g, " ")} · strength ${(s * 100).toFixed(0)}%`;
    $("readout-block").innerHTML =
      `<div class="trace-row">
         <div class="trace-k"><span class="trace-dot" style="background:${V[c.verdict]}"></span>concept ablation</div>
         <div class="trace-bar"><div class="trace-fill" style="width:${clamp(conceptDrop / ref * 100, 0, 100)}%;background:${V[c.verdict]}"></div></div>
         <div class="trace-v">+${conceptDrop.toFixed(3)}</div>
       </div>
       <div class="trace-row">
         <div class="trace-k"><span class="trace-dot" style="background:${NULLC}"></span>matched-random null</div>
         <div class="trace-bar"><div class="trace-fill" style="width:${clamp(nullDrop / ref * 100, 0.4, 100)}%;background:${NULLC}"></div></div>
         <div class="trace-v">+${nullDrop.toFixed(3)}</div>
       </div>
       <div class="trace-caption">
         ${p.bites
           ? `The concept ablation drops the decision margin <b>${isFinite(ratio) ? Math.round(ratio) + "×" : "far"}</b> more than a matched-random edit of the same magnitude — a real necessity bite (z ${p.z.toFixed(1)}).`
           : `At this depth the drop sits in the null band — necessity is redundancy-limited here (Hydra effect). Step toward the readout to see it bite.`}
       </div>`;
  }

  // ------------------------------------------------------------------ pillars / bite / ledger / caption
  function renderPillars() {
    const sc = selected().scores;
    const tiles = [
      { name: "necessity", val: sc.necessity, tag: "layer-resolved" },
      { name: "sufficiency", val: sc.sufficiency, tag: "steering axis" },
      { name: "specificity", val: sc.specificity, tag: "targeted" },
    ];
    $("pillars").innerHTML = tiles.map((t) =>
      `<div class="pill-tile"><div class="pill-name">${t.name}</div>` +
      `<div class="pill-val">${t.val != null ? t.val.toFixed(2) : "—"}</div>` +
      `<div class="pill-verdict" style="color:var(--text-faint)">${t.tag}</div></div>`).join("");
  }
  function renderPillarsEmpty() {
    $("pillars").innerHTML = ["necessity", "sufficiency", "specificity"].map((nm) =>
      `<div class="pill-tile"><div class="pill-name">${nm}</div><div class="pill-val">—</div>` +
      `<div class="pill-verdict" style="color:var(--text-faint)">not run</div></div>`).join("");
  }

  function renderBite() {
    const c = selected(), curve = curveOf(c), n = curve.length;
    const best = curve.reduce((b, p, i) => (p.gap > curve[b].gap ? i : b), 0);
    const meanNull = curve.reduce((a, p) => a + nullStd(p), 0) / n;
    const rising = curve[n - 1].gap >= curve[0].gap;
    $("bite-callout").innerHTML =
      `Concept ablation bites hardest at <b>block ${blockFor(best, n)} · ${esc(String(curve[best].layer).replace(/_/g, " "))}</b> ` +
      `(Δ ${curve[best].gap.toFixed(2)}, z ${curve[best].z.toFixed(1)}). The matched-random null holds at ~${meanNull.toFixed(2)} across every depth. ` +
      (rising
        ? `The margin-drop <b>rises monotonically with depth</b> — necessity is redundancy-limited early and bites near the readout. That layer-resolved honesty is the point.`
        : `The drop peaks mid-network — consistent with mid-layer redundancy.`);
  }

  const VERB_LABEL = {
    contrast_validation: "probe · contrast gate", necessity_live: "ablate_live · necessity",
    necessity_cached: "ablate · necessity (cached)", sufficiency: "steer · sufficiency",
    specificity: "specificity · off-target", confound: "confound · site gate",
    multiple_comparisons: "multiple comparisons",
  };
  function badgeFor(t) {
    const o = (t.observation || "") + " " + (t.interpretation || "");
    if (/UNCHECKED|no site|single-source/i.test(o)) return { cls: "off", txt: "unchecked" };
    if (/WARN|does NOT|FLAG|leaked|RIDES|CAPPED|collinear/i.test(o)) return { cls: "warn", txt: "warn" };
    return { cls: "pass", txt: "pass" };
  }
  const shorten = (s) => { s = String(s || ""); return s.length > 220 ? s.slice(0, 217) + "…" : s; };

  // per-layer chips from the REAL live_necessity curve (block · Δgap), the bite emphasized
  function layerChips() {
    const c = selected(); if (!isInterventable(c)) return "";
    const curve = curveOf(c), n = curve.length;
    return `<div class="led-layers">` + curve.map((p, i) =>
      `<span class="led-chip ${p.bites ? "bite" : ""}">b${blockFor(i, n)} Δ${p.gap.toFixed(2)}</span>`).join("") + `</div>`;
  }

  function renderLedger() {
    const c = selected();
    const lv = $("ledger-verdict"); lv.textContent = c ? c.verdict : "—"; lv.className = "ledger-verdict " + verdictClass(c && c.verdict);
    const trace = (c && c.reasoning_trace) || [];
    const rows = [];
    trace.forEach((t) => {
      if (t.step === "verdict") return;
      const label = VERB_LABEL[t.step] || t.step.replace(/_/g, " ");
      const badge = badgeFor(t);
      const isNec = t.step === "necessity_live" || t.step === "necessity_cached";
      rows.push(
        `<div class="led-row${isNec ? " on" : ""}">
           <div class="led-top"><span class="led-verb">${esc(label)}</span>
             <span class="led-badge ${badge.cls}">${badge.txt}</span></div>
           <div class="led-obs">${esc(shorten(t.observation))}</div>${isNec ? layerChips() : ""}
         </div>`);
    });
    $("ledger").innerHTML = rows.join("") ||
      `<div class="led-row"><div class="led-obs">No causal actions ran — certify declined this claim.</div></div>`;
  }

  function renderCaption() {
    const c = selected();
    $("stage-caption").innerHTML =
      "<b>Live source-intervention:</b> the concept axis is projected out of the CLS at the selected block on <b>this slide's forward pass</b>, and blocks L+1…final recompute. " +
      "The margin-drop vs a matched-random null of the same magnitude is the necessity signal — reported layer by layer, not collapsed to one readout-space number." +
      (state.source === "mock"
        ? ' <span class="src-warn">· offline: illustrative card (start the warm backend for the live substrate)</span>' : "");
  }

  // ------------------------------------------------------------------ controls
  function setLayer(i) {
    state.selLayer = i;
    renderLayerStrip(); renderCurve(); renderReadout();
  }
  function setStrength(v) {
    state.strength = v / 100; $("strength-val").textContent = v + "%";
    renderLayerStrip(); renderCurve(); renderReadout();
  }

  // ------------------------------------------------------------------ live badge / rerun
  function setChip() {
    const chip = $("substrate-chip");
    const live = state.source !== "mock";
    chip.dataset.live = live ? "1" : "0";
    const label = state.source === "certify" ? "● LIVE certify" : state.source === "live" ? "● LIVE demo" : "○ mock";
    chip.textContent = label + " · phikon-v2 · ViT-L/" + state.blocks;
  }
  async function rerun() {
    const btn = $("rerun-btn"); btn.disabled = true; btn.textContent = "⟳ …";
    try { localStorage.removeItem("biolayer:lastCard"); } catch (e) {}
    await loadCard(); ingest(); setChip(); buildSummary(); buildRail(); renderStage();
    btn.disabled = false; btn.textContent = "⟳ live";
  }

  // ------------------------------------------------------------------ init
  async function init() {
    $("strength-slider").addEventListener("input", (e) => setStrength(+e.target.value));
    $("rerun-btn").addEventListener("click", rerun);
    window.addEventListener("resize", () => { if (isInterventable(selected())) renderCurve(); });

    await loadCard(); ingest(); setChip(); buildSummary(); buildRail();
    if (!state.all.length) {
      $("necessity-curve").innerHTML = `<div class="panels-loading">No claims in this card.</div>`;
      return;
    }
    // ensure a sensible default layer for the initial interventable claim
    selectClaim(state.selectedId);
  }

  document.addEventListener("DOMContentLoaded", init);
})();
