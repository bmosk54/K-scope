/* AutoResearch view — drives the autonomous causal-circuit discovery loop.
 *
 * Self-contained: opens an EventSource to /api/autoresearch and renders one card per
 * streamed iteration (probe -> certify -> locate causal layer -> ablate -> reflect).
 * Also drives the pipeline visualization (#rr-pipeline) — the five-stage loop diagram
 * that flows while running and fills each stage's live value from the latest iteration.
 * Touches only #view-research DOM; never reaches into app.js state.
 */
(function () {
  "use strict";
  var $ = function (id) { return document.getElementById(id); };
  var es = null;                 // active EventSource
  var agg = {};                  // layer -> {seen, bites, maxGap} across iterations
  var serverErr = false;         // a named server 'error' frame was shown this run

  var el = {
    problem: $("rr-problem"), track: $("rr-track"), iters: $("rr-iters"),
    bedrock: $("rr-bedrock"), live: $("rr-live"),
    start: $("rr-start"), stop: $("rr-stop"), status: $("rr-status"),
    feed: $("rr-feed"), summary: $("rr-circuit-summary"),
    aggBox: $("rr-circuit-agg"), mode: $("rr-circuit-mode"),
    pipeline: $("rr-pipeline"),
    cliBody: $("rr-cli-body"), cliMeta: $("rr-cli-meta"),
  };
  if (!el.feed) return;

  // ---- reasoning-trace CLI ------------------------------------------------
  function cliClear(problem) {
    if (!el.cliBody) return;
    el.cliBody.innerHTML =
      '<div class="rr-cli-line dim"><span class="rr-cli-prompt">$</span> autoresearch --loop' +
      (problem ? ' --problem "' + escapeHtml(shorten(problem, 60)) + '"' : "") + '</div>';
    cliCursor();
  }
  function cliCursor() {
    if (!el.cliBody) return;
    var cur = el.cliBody.querySelector(".rr-cli-cursorline");
    if (cur) { el.cliBody.appendChild(cur); return; }
    cur = document.createElement("div");
    cur.className = "rr-cli-line rr-cli-cursorline";
    cur.innerHTML = '<span class="rr-cli-cursor">▋</span>';
    el.cliBody.appendChild(cur);
  }
  function cliLine(html, cls) {
    if (!el.cliBody) return;
    var d = document.createElement("div");
    d.className = "rr-cli-line" + (cls ? " " + cls : "");
    d.innerHTML = html;
    var cur = el.cliBody.querySelector(".rr-cli-cursorline");
    if (cur) el.cliBody.insertBefore(d, cur); else el.cliBody.appendChild(d);
    el.cliBody.scrollTop = el.cliBody.scrollHeight;
  }
  function cliMeta(text, cls) {
    if (!el.cliMeta) return;
    el.cliMeta.className = "rr-cli-meta" + (cls ? " " + cls : "");
    el.cliMeta.textContent = text;
  }
  // Emit one iteration's reasoning as a block of terminal lines: probe → certify →
  // locate → ablate → reflect, mirroring the pipeline stages.
  function cliEmit(r) {
    var c = r.contrast || {}, ab = r.ablation || {}, pil = r.pillars || {};
    cliLine('<span class="rr-cli-k">── iter ' + r.iter + " / " + r.max_iters + " ─────────────────────</span>", "hr");
    var by = r.proposed_by ? '  <span class="rr-cli-k">by</span> <span class="rr-cli-v">' + escapeHtml(r.proposed_by) + "</span>" : "";
    cliLine('<span class="rr-cli-stage">probe  </span> <span class="rr-cli-k">concept=</span><span class="rr-cli-v">' +
      escapeHtml(r.concept || (c.pos + " vs " + c.neg)) + "</span>" + by);
    if (r.hypothesis) cliLine('           <span class="rr-cli-k">hyp:</span> ' + escapeHtml(r.hypothesis), "dim");
    var pills = ["necessity", "sufficiency", "specificity"].map(function (k) {
      var p = pil[k] || {};
      return '<span class="rr-cli-' + (p.passed ? "ok" : "no") + '">' + k.slice(0, 4) + (p.passed ? "✓" : "✕") + "</span>";
    }).join(" ");
    cliLine('<span class="rr-cli-stage">certify</span> <span class="rr-cli-' + escapeHtml(r.verdict) + '">' +
      escapeHtml(r.verdict) + '</span> <span class="rr-cli-k">score=</span><span class="rr-cli-v">' +
      fmt(r.score, 3) + "</span>   " + pills);
    cliLine('<span class="rr-cli-stage">locate </span> <span class="rr-cli-k">load layer=</span><span class="rr-cli-v">' +
      (ab.layer != null ? ab.layer : "—") + "</span>" +
      (r.circuit_mode ? ' <span class="rr-cli-k">mode=</span><span class="rr-cli-v">' + escapeHtml(r.circuit_mode) + "</span>" : ""));
    cliLine('<span class="rr-cli-stage">ablate </span> <span class="rr-cli-v">' + escapeHtml(ab.note || "—") + "</span>");
    if (r.diagnosis) cliLine('<span class="rr-cli-stage">reflect</span> ↳ ' + escapeHtml(r.diagnosis), "reflect");
    if (r.next_probe)
      cliLine('           <span class="rr-cli-next">→ next probe:</span> <span class="rr-cli-v">' +
        escapeHtml(r.next_probe.pos + " vs " + r.next_probe.neg) + "</span>", "reflect");
    else if (r.next_hypothesis)
      cliLine('           <span class="rr-cli-next">→</span> ' + escapeHtml(r.next_hypothesis), "reflect");
    cliCursor();
  }

  // ---- pipeline visualization --------------------------------------------
  var STAGES = ["probe", "certify", "locate", "ablate", "reflect"];
  var stageEl = {};   // stage key -> {node: .rr-stage, val: .rr-stage-val}
  STAGES.forEach(function (k) {
    var val = $("rr-stage-" + k);
    stageEl[k] = { node: val ? val.closest(".rr-stage") : null, val: val };
  });
  function pipeReset() {
    if (el.pipeline) el.pipeline.classList.remove("running");
    STAGES.forEach(function (k) {
      var s = stageEl[k]; if (!s.node) return;
      s.node.classList.remove("hot", "done");
      s.val.textContent = "—";
    });
  }
  function pipeRunning(on) {
    if (el.pipeline) el.pipeline.classList.toggle("running", !!on);
  }
  // Fill each stage's live value from the newest iteration, then sweep the "hot"
  // highlight left-to-right so the eye follows one datum through the loop.
  function pipeUpdate(r) {
    var c = r.contrast || {}, ab = r.ablation || {};
    var vals = {
      probe: r.concept || (c.pos && c.neg ? c.pos + " vs " + c.neg : "—"),
      certify: (r.verdict || "—") + " · " + fmt(r.score, 3),
      locate: (ab.layer != null ? "layer " + ab.layer : "—") +
              (r.circuit_mode ? " · " + r.circuit_mode : ""),
      ablate: shorten(ab.note, 46),
      reflect: r.next_probe ? "→ " + r.next_probe.pos + " vs " + r.next_probe.neg
               : (r.diagnosis ? shorten(r.diagnosis, 46) : "converged"),
    };
    STAGES.forEach(function (k, i) {
      var s = stageEl[k]; if (!s.node) return;
      window.setTimeout(function () {
        STAGES.forEach(function (kk) { if (stageEl[kk].node) stageEl[kk].node.classList.remove("hot"); });
        s.node.classList.add("hot", "done");
        s.val.textContent = vals[k];
      }, i * 130);
    });
  }
  function shorten(s, n) {
    s = String(s == null ? "" : s).trim();
    return s.length > n ? s.slice(0, n - 1) + "…" : (s || "—");
  }

  function setStatus(text, cls) {
    el.status.className = "rr-status" + (cls ? " " + cls : "");
    el.status.innerHTML = (cls === "running" ? '<span class="rr-dot"></span>' : "") +
      escapeHtml(text);
    cliMeta(text, cls);
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
    pipeReset(); pipeRunning(true);
    cliClear(el.problem.value.trim());
    cliLine('<span class="rr-cli-k">booting causal battery · opening SSE stream …</span>', "dim");
    var q = new URLSearchParams({
      problem: el.problem.value.trim() || "Characterize the tumor microenvironment.",
      track: el.track.value,
      iters: el.iters.value || "5",
      bedrock: el.bedrock.checked ? "1" : "0",
      live: el.live.checked ? "1" : "0",
    });
    el.start.disabled = true; el.stop.disabled = false;
    serverErr = false;
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
    // Named server-sent 'error' event (our SSE error frame carries JSON). Runs before
    // onerror for the same "error" event; flag it so onerror doesn't clobber the message.
    es.addEventListener("error", function (ev) {
      var payload = ev && ev.data;
      if (payload) {
        try { setStatus("error: " + (JSON.parse(payload).error || payload), "error"); serverErr = true; }
        catch (e) {}
      }
    });
    // Transport-level error / stream close. EventSource auto-reconnects by default —
    // we must close() to stop it re-running the whole loop after the server ends the stream.
    es.onerror = function () {
      if (es) { es.close(); es = null; }
      // don't overwrite a specific server error; only flag a real connection failure
      // (nothing streamed AND no server error frame was shown this run).
      if (!serverErr && el.feed.children.length === 0)
        setStatus("connection error — is app_server.py running?", "error");
      reset();
    };
  }

  function finishDone(rec) {
    var d = document.createElement("div");
    d.className = "rr-done" + (rec.reason === "max_iters" ? " max" : "");
    d.textContent = (rec.reason === "converged" ? "✓ converged — " : "● ") + (rec.note || rec.reason || "done");
    el.feed.appendChild(d);
    setStatus(rec.reason === "converged" ? "converged" : "done (" + (rec.reason || "") + ")", "done");
    cliLine('<span class="rr-cli-k">── ' + (rec.reason === "converged" ? "✓ converged" : "● " + (rec.reason || "done")) +
      " ──────────────────</span> " + escapeHtml(rec.note || ""), "hr");
    cliCursor();
    stop(null);
    el.feed.scrollTop = el.feed.scrollHeight;
  }

  function stop(text, cls) {
    if (es) { es.close(); es = null; }
    reset();
    if (text) setStatus(text, cls);
  }
  function reset() { el.start.disabled = false; el.stop.disabled = true; pipeRunning(false); }

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
    pipeUpdate(r);
    cliEmit(r);
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
