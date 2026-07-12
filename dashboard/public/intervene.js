/**
 * KScope — Intervention Studio.
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

  const V = { GROUNDED: "#12916A", WEAK: "#B5852A", NOT_CERTIFIABLE: "#8C8577" };
  const NULLC = "#8C8577";          // matched-random null trace/band
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
      const stash = JSON.parse(localStorage.getItem("kscope:lastCard") || "null");
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
      const first = state.all.find(hasCurve) || state.all.find(isCertifiable) || state.all[0];
      state.selectedId = first ? first.id : null;
    }
  }

  const selected = () => state.all.find((c) => c.id === state.selectedId) || null;
  // Two distinct states: a claim can have a LIVE per-block curve to scrub (the 3D deck),
  // OR merely be CERTIFIABLE (has pillar scores but no source-intervention curve — the
  // common case, since the dashboard certify runs without live_ctx). Certifiable-but-no-
  // curve claims must STILL render a useful stage (scores + gate + trace), never blank.
  const hasCurve = (c) =>
    !!(c && c.live_necessity && Array.isArray(c.live_necessity.curve) && c.live_necessity.curve.length);
  const isCertifiable = (c) => !!(c && c.scores && Object.keys(c.scores).length);
  const isInterventable = hasCurve;   // kept name: "interventable" == a live curve exists
  // Coerce every point's gap/z to a finite number here so downstream .toFixed() calls
  // (sheets, tooltips, ledger, caveat) can never throw on a live curve missing a field.
  const num = (v) => (typeof v === "number" && isFinite(v) ? v : 0);
  const curveOf = (c) => ((c && c.live_necessity && c.live_necessity.curve) || [])
    .map((p) => ({ ...p, gap: num(p.gap), z: num(p.z) }));
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
      const inter = hasCurve(c);
      const cert = isCertifiable(c);
      const card = document.createElement("div");
      card.className = "concept-card " + verdictClass(c.verdict) +
        (c.id === state.selectedId ? " active" : "") + (cert ? "" : " declined");
      card.dataset.id = c.id;

      let sub = "";
      if (c.verdict === "WEAK" && capR(c) != null) {
        sub = `<div class="cc-note warn">capped · rides intensity |r| = ${capR(c).toFixed(3)}</div>`;
      } else if (cert && !inter) {
        sub = `<div class="cc-note dim">cached necessity · no live layer curve</div>`;
      } else if (!cert) {
        sub = `<div class="cc-note dim">${esc(c.reason || "no axis to intervene on")}</div>`;
      }

      card.innerHTML =
        `<div class="cc-top">
           <span class="cc-name">${esc(c.claim)}</span>
           <span class="cc-verdict ${verdictClass(c.verdict)}">${c.verdict}</span>
         </div>
         <div class="cc-contrast">${esc(c.contrast || (c.concept ? c.concept.replace(/_/g, " ") : "—"))}</div>
         ${(inter || cert) ? `<div class="cc-gaps" id="ccg-${c.id}"></div>` : ""}
         ${sub}`;
      card.addEventListener("click", () => selectClaim(c.id));
      host.appendChild(card);
      if (inter) paintCardGaps(c);
      else if (cert) paintCardScores(c);
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

  // rail sparkline for a certifiable claim with NO live curve: the three pillar scores
  // (necessity / sufficiency / specificity), so the card is informative, not blank.
  function paintCardScores(c) {
    const host = $("ccg-" + c.id); if (!host) return;
    const sc = c.scores || {};
    const vals = [sc.necessity, sc.sufficiency, sc.specificity].map((v) => (typeof v === "number" ? v : 0));
    host.innerHTML = vals.map((v) => {
      const h = 4 + 18 * clamp(v, 0, 1);
      return `<span class="ccg-bar" style="height:${h.toFixed(1)}px;background:${v > 0.5 ? V[c.verdict] : "var(--border-strong)"}"></span>`;
    }).join("");
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

    const inter = hasCurve(c);
    const cert = isCertifiable(c);
    const ioi = inter && c.live_necessity.intervened_on_input;
    $("intervened-badge").innerHTML = ioi
      ? `<span class="ioi on" title="the do() was applied on this slide's forward pass, not a cached embedding">✓ intervened_on_input · per-slide forward pass</span>`
      : (cert ? `<span class="ioi off" title="certify ran without live_ctx — necessity is the cached readout-space score">○ cached necessity · no per-slide forward pass</span>` : "");

    // the strength scrubber only makes sense with a live curve; the deck area is shown for
    // both live and cached (cached fills it with the scores/gate panel), hidden only when declined.
    $("control-bar").style.display = inter ? "" : "none";
    $("deck-wrap").style.display = (inter || cert) ? "" : "none";

    if (inter) {
      renderDeck(); renderReadout(); renderPillars(); renderBite();
      renderLedger(); renderCaption(); renderDeckLegend();
      return;
    }
    if (cert) {
      renderCached(c); renderPillars(); renderLedger();
      return;
    }
    renderDeclined(c); renderPillarsEmpty(); renderLedger();
  }

  // Certifiable claim WITHOUT a live layer curve (certify ran without live_ctx). Show the
  // REAL pillar scores + the contrast-gate result + the reasoning ledger, honestly badged
  // "cached" — so a GROUNDED claim is never rendered as a blank / NOT-CERTIFIABLE box.
  function renderCached(c) {
    const sc = c.scores || {}, cv = c.contrast_validation || {}, rr = capR(c);
    const f2 = (v) => (v != null && isFinite(v) ? (+v).toFixed(2) : "—");
    $("deck3d").innerHTML =
      `<div class="cached-panel">
         <div class="cp-head">
           <span class="cp-verdict ${verdictClass(c.verdict)}">${c.verdict}</span>
           <span class="cp-contrast">${esc(c.contrast || (c.concept ? c.concept.replace(/_/g, " ") : "—"))}</span>
         </div>
         <div class="cp-grid">
           ${[["necessity", sc.necessity], ["sufficiency", sc.sufficiency], ["specificity", sc.specificity]]
             .map(([k, v]) => `<div class="cp-tile"><div class="cp-k">${k}</div><div class="cp-v">${f2(v)}</div></div>`).join("")}
         </div>
         <div class="cp-gate">
           <span>held-out AUROC <b>${f2(cv.heldout_auroc)}</b></span>
           <span>intensity |r| <b>${f2(rr)}</b>${rr != null && rr > 0.6 ? ' <i class="cp-warn">rides intensity → capped</i>' : ""}</span>
         </div>
         <div class="cp-note">Necessity here is the <b>cached readout-space</b> score. The per-block
           live source-intervention curve (the depth stack you can scrub) needs the GPU forward-pass
           path — re-run certify with <code>live_ctx</code> to get the layer-resolved intervention.</div>
       </div>`;
    $("readout-block").innerHTML =
      `<div class="trace-caption">Certified in <b>cached mode</b> — necessity ${f2(sc.necessity)},
       sufficiency ${f2(sc.sufficiency)}, specificity ${f2(sc.specificity)} (readout-space projection
       vs matched-random null). Run the live path for the graded, layer-by-layer curve.</div>`;
    $("bite-callout").innerHTML =
      `Verdict <b>${c.verdict}</b> from the cached battery. ` +
      (rr != null && rr > 0.6
        ? `The contrast rides the intensity proxy (|r| ${f2(rr)}) — capped regardless of pillar scores.`
        : `Contrast passed the validation gate (AUROC ${f2(cv.heldout_auroc)}, |r| ${f2(rr)}).`);
    $("stage-caption").innerHTML =
      "<b>Cached certification:</b> pillar scores come from the readout-space battery " +
      "(concept-axis projection vs matched-random null). The layer-resolved <b>live</b> intervention — " +
      "projecting the axis out at each block on this slide's forward pass — is not in this card; it needs " +
      "the GPU <code>live_ctx</code> path." +
      (state.source === "mock" ? ' <span class="src-warn">· offline mock card</span>' : "");
  }

  function renderDeclined(c) {
    $("deck3d").innerHTML = "";
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

  // ------------------------------------------------------------------ THE 3D LAYER STACK
  // The encoder's depth as a stack of glass sheets in perspective (Goodfire-style). One
  // sheet per probed block: input at the back, readout at the front. Each sheet shows, in
  // plane, the concept-ablation margin-drop bar vs the flat matched-random null — so the
  // monotone rise reads as the bars growing toward the front. Click a sheet to ablate there.
  function renderDeck() {
    const c = selected(), curve = curveOf(c), n = curve.length, s = state.strength;
    const host = $("deck3d"); host.innerHTML = "";
    const maxGap = Math.max(...curve.map((p) => p.gap), EPS);
    const deck = document.createElement("div"); deck.className = "deck";

    curve.forEach((p, i) => {
      const gap = p.gap * s, nsd = nullStd(p) * s;
      const gpct = clamp(gap / maxGap * 100, 0, 100), npct = clamp(nsd / maxGap * 100, 0.5, 100);
      const plane = document.createElement("div");
      plane.className = "plane" + (i === state.selLayer ? " sel" : "") + (p.bites ? " bites" : "");
      plane.style.setProperty("--z", (i * 46) + "px");   // deeper block = further forward
      plane.style.setProperty("--v", V[c.verdict]);
      plane.style.setProperty("--glow", (p.bites ? 0.12 + 0.5 * (gap / maxGap) : 0.05).toFixed(3));
      plane.innerHTML =
        `<div class="sheet">
           <div class="sheet-head">
             <span class="sh-block">block ${blockFor(i, n)}</span>
             <span class="sh-name">${esc(String(p.layer).replace(/_/g, " "))}</span>
             <span class="sh-flag ${p.bites ? "on" : ""}">${p.bites ? "bites" : "n.s."}</span>
           </div>
           <div class="sheet-bars">
             <div class="sbar-null" style="width:${npct}%"></div>
             <div class="sbar-gap" style="width:${gpct}%"></div>
           </div>
           <div class="sheet-foot"><span>Δ ${gap.toFixed(2)}</span><span>z ${p.z.toFixed(1)}</span></div>
         </div>`;
      plane.addEventListener("click", () => setLayer(i));
      plane.addEventListener("mouseover", (e) => showTip(e,
        `<div class="tt-title">block ${blockFor(i, n)} · ${esc(p.layer)}</div>` +
        `<div class="tt-row"><span>margin drop</span><span>+${gap.toFixed(3)}</span></div>` +
        `<div class="tt-row"><span>matched-random null σ</span><span>${nsd.toFixed(3)}</span></div>` +
        `<div class="tt-row"><span>z vs null</span><span>${p.z.toFixed(1)}</span></div>` +
        `<div class="tt-note">${p.bites ? "significant necessity bite — ablate here to collapse the readout" : "no significant bite — redundancy recomputes it downstream"}</div>`));
      plane.addEventListener("mousemove", moveTip);
      plane.addEventListener("mouseout", hideTip);
      deck.appendChild(plane);
    });
    host.appendChild(deck);
  }

  function renderDeckLegend() {
    const c = selected();
    $("curve-legend").innerHTML =
      `<span class="leg-item"><span class="leg-swatch" style="background:${V[c.verdict]}"></span>concept-ablation margin drop</span>` +
      `<span class="leg-item"><span class="leg-swatch" style="background:${NULLC};opacity:.5"></span>matched-random null</span>` +
      `<span class="leg-item"><span class="leg-dot"></span>selected ablation site (front → readout)</span>`;
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
    renderDeck(); renderReadout();
  }
  function setStrength(v) {
    state.strength = v / 100; $("strength-val").textContent = v + "%";
    renderDeck(); renderReadout();
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
    // Reload the latest certified card — do NOT discard the stash (that would silently drop
    // the just-certified run back to the mock if the backend is cold).
    await loadCard(); ingest(); setChip(); buildSummary(); buildRail(); renderStage();
    btn.disabled = false; btn.textContent = "⟳ live";
  }

  // ------------------------------------------------------------------ init
  async function init() {
    $("strength-slider").addEventListener("input", (e) => setStrength(+e.target.value));
    $("rerun-btn").addEventListener("click", rerun);
    window.addEventListener("resize", () => { if (isInterventable(selected())) renderDeck(); });

    await loadCard(); ingest(); setChip(); buildSummary(); buildRail();
    if (!state.all.length) {
      $("deck3d").innerHTML = `<div class="panels-loading">No claims in this card.</div>`;
      return;
    }
    // ensure a sensible default layer for the initial interventable claim
    selectClaim(state.selectedId);
  }

  document.addEventListener("DOMContentLoaded", init);
})();
