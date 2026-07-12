/* AutoResearch rail — drives the autonomous causal-circuit discovery loop.
 *
 * Self-contained: opens an EventSource to /api/autoresearch and renders one card per
 * streamed iteration (probe -> certify -> locate causal layer -> ablate -> reflect).
 * Touches only #research-rail DOM; never reaches into app.js state.
 */
(function () {
  "use strict";
  var $ = function (id) { return document.getElementById(id); };
  var shell = document.querySelector(".app-shell");
  var es = null;                 // active EventSource
  var agg = {};                  // layer -> {seen, bites, maxGap} across iterations

  var el = {
    rail: $("research-rail"), collapse: $("rr-collapse"), nav: $("nav-research"),
    problem: $("rr-problem"), track: $("rr-track"), iters: $("rr-iters"),
    bedrock: $("rr-bedrock"), live: $("rr-live"),
    start: $("rr-start"), stop: $("rr-stop"), status: $("rr-status"),
    feed: $("rr-feed"), summary: $("rr-circuit-summary"),
    aggBox: $("rr-circuit-agg"), mode: $("rr-circuit-mode"),
  };
  if (!el.rail) return;

  // ---- collapse / nav -----------------------------------------------------
  el.collapse.addEventListener("click", function () { shell.classList.toggle("rail-collapsed"); });
  if (el.nav) el.nav.addEventListener("click", function () {
    shell.classList.remove("rail-collapsed");
    el.problem.focus();
  });

  function setStatus(text, cls) {
    el.status.className = "rr-status" + (cls ? " " + cls : "");
    el.status.innerHTML = (cls === "running" ? '<span class="rr-dot"></span>' : "") +
      escapeHtml(text);
  }
  function escapeHtml(s) {
    return String(s == null ? "" : s).replace(/[&<>"]/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c];
    });
  }
  function fmt(x, d) { return (x == null || isNaN(x)) ? "—" : Number(x).toFixed(d == null ? 2 : d); }

  // ---- lifecycle ----------------------------------------------------------
  el.start.addEventListener("click", start);
  el.stop.addEventListener("click", function () { stop("stopped", "done"); });

  function start() {
    stop(null);                                  // clear any prior run
    el.feed.innerHTML = "";
    agg = {};
    el.summary.hidden = true; el.aggBox.innerHTML = "";
    var q = new URLSearchParams({
      problem: el.problem.value.trim() || "Characterize the tumor microenvironment.",
      track: el.track.value,
      iters: el.iters.value || "5",
      bedrock: el.bedrock.checked ? "1" : "0",
      live: el.live.checked ? "1" : "0",
    });
    el.start.disabled = true; el.stop.disabled = false;
    setStatus("connecting…", "running");
    try {
      es = new EventSource("/api/autoresearch?" + q.toString());
    } catch (e) { setStatus("EventSource failed: " + e, "error"); reset(); return; }

    es.onmessage = function (ev) {
      var rec;
      try { rec = JSON.parse(ev.data); } catch (e) { return; }
      if (rec.done) { finishDone(rec); return; }
      renderIter(rec);
      if (rec.circuit_mode && el.mode) el.mode.textContent = "· " + rec.circuit_mode;
      setStatus("iteration " + rec.iter + " / " + rec.max_iters + (el.live.checked ? " · live" : ""), "running");
    };
    // Named server-sent 'error' event (our SSE error frame carries JSON).
    es.addEventListener("error", function (ev) {
      var payload = ev && ev.data;
      if (payload) { try { setStatus("error: " + (JSON.parse(payload).error || payload), "error"); } catch (e) {} }
    });
    // Transport-level error / stream close. EventSource auto-reconnects by default —
    // we must close() to stop it re-running the whole loop after the server ends the stream.
    es.onerror = function () {
      if (es) { es.close(); es = null; }
      // a close right after the done frame is normal; only flag if nothing streamed at all
      if (el.feed.children.length === 0) setStatus("connection error — is app_server.py running?", "error");
      reset();
    };
  }

  function finishDone(rec) {
    var d = document.createElement("div");
    d.className = "rr-done" + (rec.reason === "max_iters" ? " max" : "");
    d.textContent = (rec.reason === "converged" ? "✓ converged — " : "● ") + (rec.note || rec.reason || "done");
    el.feed.appendChild(d);
    setStatus(rec.reason === "converged" ? "converged" : "done (" + (rec.reason || "") + ")", "done");
    stop(null);
    el.feed.scrollTop = el.feed.scrollHeight;
  }

  function stop(text, cls) {
    if (es) { es.close(); es = null; }
    reset();
    if (text) setStatus(text, cls);
  }
  function reset() { el.start.disabled = false; el.stop.disabled = true; }

  // ---- rendering ----------------------------------------------------------
  function renderIter(r) {
    var c = r.contrast || {}, ab = r.ablation || {}, pil = r.pillars || {};
    var card = document.createElement("div");
    card.className = "rr-iter";

    var pillHtml = ["necessity", "sufficiency", "specificity"].map(function (k) {
      var p = pil[k] || {}; var pass = p.passed;
      return '<div class="rr-pill ' + (pass ? "pass" : "fail") + '">' +
        '<div class="rr-pill-name">' + k.slice(0, 4) + '</div>' +
        '<div class="rr-pill-val">' + fmt(p.confidence, 2) + '</div></div>';
    }).join("");

    card.innerHTML =
      '<div class="rr-iter-top">' +
        '<span class="rr-iter-n">iter ' + r.iter + " / " + r.max_iters + '</span>' +
        '<span class="rr-vbadge ' + escapeHtml(r.verdict) + '">' + escapeHtml(r.verdict) +
          ' · ' + fmt(r.score, 3) + '</span>' +
      '</div>' +
      '<div class="rr-concept">' + escapeHtml(r.concept || (c.pos + " vs " + c.neg)) + '</div>' +
      '<div class="rr-hyp">' + escapeHtml(r.hypothesis || "") +
        ' <span class="rr-by">' + escapeHtml(r.proposed_by || "") + '</span></div>' +
      '<div class="rr-pillars">' + pillHtml + '</div>' +
      '<div class="rr-circuit">' +
        '<div class="rr-clabel">causal circuit — necessity by layer' +
          (r.circuit_mode ? ' (' + escapeHtml(r.circuit_mode) + ')' : '') + '</div>' +
        circuitRows(r.circuit || [], ab.layer) +
      '</div>' +
      '<div class="rr-ablate">⊘ <b>ablate</b> ' + escapeHtml(ab.note || "") + '</div>' +
      '<div class="rr-reflect">' +
        (r.diagnosis ? '<span class="rr-diag">↳ ' + escapeHtml(r.diagnosis) + '</span><br>' : '') +
        escapeHtml(r.next_hypothesis || "") +
        (r.next_probe ? '<div class="rr-next">next: <b>' +
          escapeHtml(r.next_probe.pos + " vs " + r.next_probe.neg) + '</b></div>' : '') +
      '</div>';

    el.feed.appendChild(card);
    el.feed.scrollTop = el.feed.scrollHeight;
    updateAgg(r.circuit || [], r.circuit_mode);
  }

  function circuitRows(circuit, loadLayer) {
    if (!circuit.length) return '<div class="rr-clabel">no layer curve</div>';
    var maxGap = Math.max.apply(null, circuit.map(function (n) { return Math.abs(n.necessity_gap || 0); }).concat([1e-6]));
    return circuit.map(function (n) {
      var w = Math.max(2, Math.round(100 * Math.abs(n.necessity_gap || 0) / maxGap));
      var cls = n.layer === loadLayer ? "load" : (n.bites ? "bites" : "");
      return '<div class="rr-crow">' +
        '<span class="rr-clayer">' + escapeHtml(n.layer) + '</span>' +
        '<span class="rr-cbar"><span class="rr-cfill ' + cls + '" style="width:' + w + '%"></span></span>' +
        '<span class="rr-cval">' + fmt(n.necessity_gap, 2) + '</span></div>';
    }).join("");
  }

  function updateAgg(circuit, mode) {
    circuit.forEach(function (n) {
      var a = agg[n.layer] || (agg[n.layer] = { seen: 0, bites: 0, maxGap: 0 });
      a.seen++; if (n.bites) a.bites++;
      a.maxGap = Math.max(a.maxGap, Math.abs(n.necessity_gap || 0));
    });
    var layers = Object.keys(agg);
    if (!layers.length) return;
    el.summary.hidden = false;
    if (mode && el.mode) el.mode.textContent = "· " + mode;
    var maxGap = Math.max.apply(null, layers.map(function (k) { return agg[k].maxGap; }).concat([1e-6]));
    el.aggBox.innerHTML = layers.map(function (k) {
      var a = agg[k]; var w = Math.max(3, Math.round(100 * a.maxGap / maxGap));
      var bites = a.bites > 0;
      return '<div class="rr-agg-row">' +
        '<span class="rr-agg-layer">' + escapeHtml(k) + '</span>' +
        '<span class="rr-agg-track"><span class="rr-agg-fill ' + (bites ? "bites" : "") +
          '" style="width:' + w + '%"></span></span>' +
        '<span class="rr-agg-val">' + fmt(a.maxGap, 2) + '</span></div>';
    }).join("");
  }
})();
