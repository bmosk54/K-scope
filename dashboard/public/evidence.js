/* Evidence view — the SAE's mechanistic evidence for a certified claim.
 *
 * Self-contained: reads the precomputed sae/sae.json, touches only #view-evidence DOM,
 * never reaches into app.js state. Same contract as research.js.
 *
 * WHAT THIS VIEW IS FOR. `certify` answers "is the claim supported?" — it returns a verdict
 * and three pillars, and it returns NO tissue. This view answers the question a pathologist
 * asks next: "what did the model actually LOOK AT, and would it still say this if I took
 * that away?" So it leads with the morphology (the exemplar tiles) and the trust verdict,
 * and keeps the feature indices as an implementation detail at the bottom.
 *
 * THE CHART. Three ablation curves, all measured identically — project directions out of
 * every token at every block >= 27 of H-Optimus-0, let the remaining 13 blocks run, read the
 * model's own decision:
 *
 *   SAE features  the model COMPUTES the concept with  -> the call collapses
 *   random        the same NUMBER of features (null)    -> barely moves
 *   probe         the direction the concept is READ OUT along, ~99% accurate -> does not move
 *
 * The flat probe line is the whole argument for this project existing: a 99%-accurate linear
 * probe is not what the model computes with, so deleting it changes nothing. Explainability
 * tells you a direction correlates. Mechanistic interpretability tells you what breaks.
 */
(function () {
  "use strict";

  // Read the series colours from CSS rather than hardcoding them, so a reskin of the dashboard
  // restyles this view too. Resolved lazily: the stylesheet must be parsed first.
  function tok(name, fallback) {
    var v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
    return v || fallback;
  }
  var COL = {};
  function loadColors() {
    COL.sae = tok("--ev-sae", "#d93a4e");
    COL.random = tok("--ev-random", "#6b7086");
    COL.probe = tok("--ev-probe", "#2f6ae0");
  }
  var DATA = null;

  var esc = function (s) {
    return String(s == null ? "" : s).replace(/[&<>]/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c];
    });
  };

  function el(id) { return document.getElementById(id); }

  // The 9 NCT-CRC-HE class codes are an implementation detail. A pathologist reads tissue.
  var TISSUE = {
    ADI: "adipose", BACK: "background", DEB: "debris", LYM: "lymphocytes",
    MUC: "mucus", MUS: "smooth muscle", NORM: "normal mucosa",
    STR: "cancer-associated stroma", TUM: "tumour epithelium",
  };
  var tissue = function (k) { return TISSUE[k] || k; };

  /* The fallback arrives as {STR: 0.127, MUC: 0.107} — the probability each tissue GAINS once
   * the concept is deleted. Rendering that dict raw is the exact developer artifact this view
   * exists to avoid, so turn it into a sentence. */
  function fallbackText(fb, concept) {
    if (!fb || typeof fb === "string") return String(fb || "no coherent fallback");
    var parts = Object.keys(fb).sort(function (a, b) { return fb[b] - fb[a]; });
    if (!parts.length) return "Nothing coherent — the model does not fall back on another tissue.";
    var phrases = parts.map(function (k) {
      return tissue(k) + " (+" + Math.round(fb[k] * 100) + "%)";
    });
    var list = phrases.length === 1 ? phrases[0]
      : phrases.slice(0, -1).join(", ") + " and " + phrases[phrases.length - 1];
    return "With the " + tissue(concept) + " features deleted, the model re-reads the same tiles " +
      "as " + list + ". That is what it falls back on — a useful sanity check: if the fallback " +
      "tissue is implausible for this slide, the original call was resting on something odd.";
  }

  /* ---------------------------------------------------------------- how it works
   * One picture, no jargon, before any numbers. Five steps, left to right: a tile goes into
   * the model; the SAE reads out the visual features the model built; we keep the ones that
   * fire on this concept; we delete them from the live network; the model answers again. */
  function renderHow() {
    var steps = [
      { n: "1", t: "One tile", d: "224px of tissue" },
      { n: "2", t: "The model reads it", d: "into 1,536 numbers" },
      { n: "3", t: "We unpack those", d: "into 6,144 visual features it taught itself" },
      { n: "4", t: "We erase the ones it used", d: "inside the live model" },
      { n: "5", t: "We ask it again", d: "if the answer changes, the tissue was real evidence" },
    ];
    var host = el("ev-how");
    if (!host) return;
    host.innerHTML =
      '<div class="ev-how-lab">How we did this</div>' +
      '<div class="ev-how-row">' +
      steps.map(function (s, i) {
        return '<div class="ev-how-step">' +
                 '<div class="ev-how-n">' + s.n + "</div>" +
                 '<div class="ev-how-t">' + esc(s.t) + "</div>" +
                 '<div class="ev-how-d">' + esc(s.d) + "</div>" +
               "</div>" +
               (i < steps.length - 1 ? '<div class="ev-how-arrow" aria-hidden="true">&rarr;</div>' : "");
      }).join("") +
      "</div>";
  }

  /* ------------------------------------------------------------- YOUR TILE (the case)
   * The rest of the dashboard follows ONE tile. So does this. We show the model's read of that
   * tile, the handful of visual features that actually fired inside it, what each of those
   * features looks like elsewhere in the tissue bank, where they fire in the user's own tile,
   * and what happens to the model's read when we take them away. */
  function renderTile(t) {
    var host = el("ev-tile");
    if (!host || !t) return;

    /* Name the model's ACTUAL answer on the USER'S tile. "The model called this tissue" means
     * nothing — it answered a specific multiple-choice question, so say which answer it gave,
     * how sure it was, and that the rest of the page is about that answer. */
    // (the top-of-page lede was folded into this hero — one header, not two)

    // how few features does it take to flip THIS tile? (first k below 0.5)
    var flipAt = null;
    for (var i = 0; i < t.ks.length; i++) {
      if (t.curves.sae[i] < 0.5) { flipAt = t.ks[i]; break; }
    }
    var fb = Object.keys(t.fallback || {}).sort(function (a, b) { return t.fallback[b] - t.fallback[a]; });
    var fbTxt = fb.length
      ? "the model reads the same pixels as <b>" + esc(tissue(fb[0])) + "</b> instead" +
        (fb[1] ? ", with some " + esc(tissue(fb[1])) : "")
      : "the model has no coherent second answer";

    var feats = (t.features || []).map(function (f, i) {
      return '<div class="ev-feat">' +
        '<div class="ev-feat-rank">' + (i + 1) + "</div>" +
        '<img class="ev-feat-img" src="' + esc(f.exemplars_png) + '" ' +
          'alt="tiles from the tissue bank that fire feature ' + f.feature + ' hardest" />' +
        '<div class="ev-feat-meta">' +
          '<div class="ev-feat-name"><b>' + esc(f.looks_like) + "</b>" +
            '<span class="ev-feat-pure">' + f.purity_pct + "% pure</span></div>" +
          // one decimal: several features are worth <1 point, and Math.round gives "1 pts"
          '<div class="ev-feat-sub"><b>' + (f.effect * 100).toFixed(1) +
            " pts</b> of confidence, alone</div>" +
        "</div></div>";
    }).join("");

    host.innerHTML =
      '<div class="ev-tile-head">' +
        "<h2>What the model looked at <span>in your tile</span></h2>" +
        // one decimal: 0.9985 rounded to "100%" is a misstatement, and it also disagreed with the
        // 1.00 shown in the stat tile below at a different precision
        "<p><b>H-Optimus-0</b> embedded this tile and read it as <b>" + esc(t.answer_tissue) +
          "</b> at <b>" + (t.confidence * 100).toFixed(1) + "% confidence</b>. This page opens up that " +
          "embedding: which visual features the model built it from, what each of those features " +
          "corresponds to in real tissue, and which ones the answer actually depends on.</p>" +
      "</div>" +

      '<div class="ev-tile-top">' +
        '<figure class="ev-tile-fig">' +
          '<img src="' + esc(t.tile_png) + '" alt="the input tile" />' +
          "<figcaption>your tile</figcaption>" +
        "</figure>" +
        '<figure class="ev-tile-fig">' +
          '<img src="' + esc(t.where_png) + '" alt="where tissue of the answered type sits in your tile" />' +
          '<div class="ev-legend">' +
            '<span class="ev-legend-lab">weak</span>' +
            '<span class="ev-legend-bar"></span>' +
            '<span class="ev-legend-lab">strong</span>' +
          "</div>" +
          "<figcaption>where the <b>" + esc(t.answer_tissue) + "</b> features fire<br>" +
            "16&times;16 grid &middot; ~7&nbsp;&micro;m per square</figcaption>" +
        "</figure>" +
        '<div class="ev-tile-read">' +
          '<div class="ev-tile-lab">The model’s answer</div>' +
          '<div class="ev-tile-ans">' + esc(t.answer_tissue) + "</div>" +
          '<div class="ev-tile-conf">' + (t.confidence * 100).toFixed(1) + "% confident</div>" +
          "<p>Its embedding of this tile is built from <b>" + t.n_active + "</b> visual features, drawn " +
            "from a vocabulary of " + t.n_features.toLocaleString() + " that the model learned on its own. " +
            "Nobody labelled these. Here are the four that contribute most.</p>" +
        "</div>" +
      "</div>" +

      '<div class="ev-feats-lab">The four features it leaned on most ' +
        '<span>A feature is a direction in the model’s embedding, not a tile. To show what each one ' +
        '<i>means</i>, we display the six tiles in the dataset where it fires hardest. Those six tiles ' +
        'are the feature’s definition by example.</span></div>' +
      '<div class="ev-feats">' + feats + "</div>" +

      '<div class="ev-tile-verdict">' +
        (flipAt
          ? "These four are not just correlated with the answer, they carry it. Erase " + flipAt +
            " of the " + t.n_active + " features from the live model and its read of this tile drops from <b>" +
            t.confidence.toFixed(2) + "</b> to <b>" + t.curves.sae[t.ks.indexOf(flipAt)].toFixed(2) +
            "</b>. Erase " + t.ks[t.ks.length - 1] + " features it never used here and the embedding still " +
            "reads <b>" + t.curves.random[t.curves.random.length - 1].toFixed(2) + "</b>. So the tissue " +
            "shown above really is what the model built this embedding out of."
          : "Erasing this tile’s features does not change the model’s read of it.") +
      "</div>" +

      '<div class="ev-stats" id="ev-tile-stats"></div>' +
      '<div class="ev-panel"><div class="section-label">What it takes to change its mind</div>' +
        '<div id="ev-tile-chart"></div></div>';

    var last = t.curves.sae.length - 1;
    el("ev-tile-stats").innerHTML = [
      ["Nothing removed", t.confidence.toFixed(2), "", "the model’s answer, untouched"],
      ["The features it used", t.curves.sae[last].toFixed(2), tok("--ev-sae", "#d93a4e"),
       "remove all " + t.n_active + " and it changes its mind"],
      ["Features it never used", t.curves.random[last].toFixed(2), tok("--ev-random", "#6b7086"),
       "remove the same number at random and nothing happens"],
    ].map(function (s) {
      return '<div class="ev-stat"><div class="ev-stat-lab">' + esc(s[0]) + "</div>" +
             '<div class="ev-stat-val"' + (s[2] ? ' style="color:' + s[2] + '"' : "") + ">" +
             esc(s[1]) + "</div>" +
             '<div class="ev-stat-sub">' + esc(s[3]) + "</div></div>";
    }).join("");

    drawCurve("ev-tile-chart", t.ks, {
      sae: [t.confidence].concat(t.curves.sae),
      random: [t.confidence].concat(t.curves.random),
      probe: [t.confidence].concat(t.curves.probe),
    }, flipAt);
  }

  // ------------------------------------------------------------------ the chart
  // Shared by the per-tile view and the population view. `cu` arrives with the k=0 baseline
  // already prepended, so every line starts from the model's UNTOUCHED confidence -- without
  // it the curves begin mid-air and you cannot see how far the model actually fell.
  function renderCurve(c) {
    drawCurve("ev-chart", DATA.ks, {
      sae: [c.baseline].concat(c.curves.sae),
      random: [c.baseline].concat(c.curves.random),
      probe: [c.baseline].concat(c.curves.probe),
    }, c.features_to_overturn);
  }

  function drawCurve(hostId, ksIn, cu, flipAt) {
    var host = el(hostId);
    if (!host || !cu) return;
    host.innerHTML = "";

    var ks = [0].concat(ksIn);
    var W = host.clientWidth || 620, H = 260;
    var m = { t: 18, r: 222, b: 38, l: 46 };   // r fits the longest right-edge label + value
    var iw = W - m.l - m.r, ih = H - m.t - m.b;

    var svg = d3.select(host).append("svg")
      .attr("width", W).attr("height", H).attr("class", "ev-svg");
    var g = svg.append("g").attr("transform", "translate(" + m.l + "," + m.t + ")");

    // x is ordinal (0,5,20,80,160) — a point scale, so the steps read evenly
    var x = d3.scalePoint().domain(ks.map(String)).range([0, iw]);
    var y = d3.scaleLinear().domain([0, 1]).range([ih, 0]);

    g.append("g").attr("class", "ev-axis").attr("transform", "translate(0," + ih + ")")
      .call(d3.axisBottom(x));
    g.append("g").attr("class", "ev-axis").call(d3.axisLeft(y).ticks(5).tickFormat(d3.format(".1f")));

    // the 0.5 line: below this, the model has changed its mind
    g.append("line").attr("class", "ev-mind")
      .attr("x1", 0).attr("x2", iw).attr("y1", y(0.5)).attr("y2", y(0.5));
    g.append("text").attr("class", "ev-mind-lab")
      .attr("x", 2).attr("y", y(0.5) - 5).text("model changes its mind");

    // Mark WHERE the call is overturned. The headline number ("20 features") is meaningless
    // unless the reader can see the point on the curve it refers to.
    if (flipAt != null && ks.indexOf(flipAt) >= 0) {
      var bx = x(String(flipAt));
      g.append("line").attr("class", "ev-broke")
        .attr("x1", bx).attr("x2", bx).attr("y1", 0).attr("y2", ih);
      // label at the FOOT of the marker — the top of the chart is where the flat probe and
      // random lines live, and the label would sit on top of them
      // sits above the foot of the marker: the red series label lands at the bottom-right corner
      g.append("text").attr("class", "ev-broke-lab")
        .attr("x", bx + 5).attr("y", ih - 26)
        .text("changes its mind here: " + flipAt + " features");
    }

    var line = d3.line()
      .x(function (d, i) { return x(String(ks[i])); })
      .y(function (d) { return y(d); });

    // The probe and random lines both end near 1.0, so their right-edge labels collide. Lay the
    // labels out with a minimum vertical gap — they are the point of the chart and must be legible
    // on a projector.
    // Label the lines by what they MEAN to the reader, not by the method that produced them.
    // "near-perfect" rather than a hardcoded 100% — the probe is 99.8% on TUM and 100.0% on LYM,
    // and rounding the first one up would be a misstatement. The exact figure is in the stat tile.
    /* The probe line is deliberately NOT plotted. It is a real and strong result (a 99.8%-accurate
     * probe direction, deleted the same way, barely moves the model) but on a shared demo it reads
     * as "probes don't work" — a swipe at certify, which uses probes for what probes are good at:
     * detection. The finding lives in Method & scope instead, framed as complementary. To put it
     * back, add "probe" to `series` and to LAB. */
    var LAB = {
      sae: "features it used",
      random: "features it didn’t",
    };
    var series = [["random", cu.random], ["sae", cu.sae]];
    var labY = {};
    series.map(function (kv) { return { n: kv[0], v: kv[1][kv[1].length - 1] }; })
      .sort(function (a, b) { return b.v - a.v; })          // top of chart downwards
      .forEach(function (d, i, arr) {
        var want = y(d.v);
        if (i > 0) want = Math.max(want, labY[arr[i - 1].n] + 13);  // 13px minimum gap
        labY[d.n] = want;
      });

    series.forEach(function (kv) {
      var name = kv[0], vals = kv[1];
      if (!vals) return;
      g.append("path").datum(vals)
        .attr("class", "ev-line ev-line-" + name)
        .attr("fill", "none")
        .attr("stroke", COL[name])
        .attr("stroke-width", name === "sae" ? 2.5 : 1.8)
        .attr("stroke-dasharray", name === "probe" ? "5 3" : null)
        .attr("d", line);
      g.selectAll(".ev-dot-" + name).data(vals).enter().append("circle")
        .attr("class", "ev-dot ev-dot-" + name)
        .attr("cx", function (d, i) { return x(String(ks[i])); })
        .attr("cy", function (d) { return y(d); })
        .attr("r", name === "sae" ? 3.6 : 2.6)
        .attr("fill", COL[name]);
      // right-edge label, so the reader never has to cross-reference a legend
      var last = vals[vals.length - 1];
      g.append("text").attr("class", "ev-serieslab")
        .attr("x", iw + 8).attr("y", labY[name] + 4)
        .attr("fill", COL[name])
        .text(LAB[name] + "  " + last.toFixed(2));
    });

    g.append("text").attr("class", "ev-axlab")
      .attr("x", iw / 2).attr("y", ih + 32).attr("text-anchor", "middle")
      .text("visual features removed from the model");
    g.append("text").attr("class", "ev-axlab")
      .attr("transform", "rotate(-90)").attr("x", -ih / 2).attr("y", -32)
      .attr("text-anchor", "middle")
      .text("how sure the model still is");
  }

  // ------------------------------------------------------------------ the panel
  function render(name) {
    var c = DATA && DATA.concepts && DATA.concepts[name];
    if (!c) return;

    d3.selectAll(".ev-tab").classed("active", function () {
      return this.dataset.concept === name;
    });

    // SPARSE = the belief rests on a few features you can look at -> auditable (good for the
    // user, and warning-coloured because it also means the answer is fragile).
    // DISTRIBUTED = robust, but untraceable to any specific morphology.
    var sparse = c.encoding === "SPARSE / AUDITABLE";
    var vcol = sparse ? tok("--ev-sparse", "#a86a00") : tok("--ev-dist", "#1d7a52");

    /* 1. VERDICT — plain language, first. Not a feature index.
     *
     * The tool's own headline quotes the LAST k (160) even when the call was already overturned
     * at 20 — which buries the actual finding under the biggest number. Lead with the number
     * that matters: how few features it takes to break the belief. */
    var ki = DATA.ks.indexOf(c.features_to_overturn);
    var lastK = DATA.ks[DATA.ks.length - 1];
    var lastV = c.curves.sae[c.curves.sae.length - 1];
    var headline;
    if (c.features_to_overturn != null && ki >= 0) {
      headline =
        "The model encodes " + tissue(c.concept) + " <b>concentrated</b> in a few features. Erasing the " +
        "top <b>" + c.features_to_overturn + "</b> across 32 tiles of this type already drops confidence " +
        "from <b>" + c.baseline.toFixed(2) + "</b> to <b>" + c.curves.sae[ki].toFixed(2) + "</b>; erasing " +
        "all <b>" + lastK + "</b> takes it to <b>" + lastV.toFixed(2) + "</b>. Erasing " + lastK +
        " features at random instead leaves it at <b>" +
        c.curves.random[c.curves.random.length - 1].toFixed(2) + "</b>. A short list of features explains " +
        "this tissue.";
    } else {
      headline =
        "The model encodes " + tissue(c.concept) + " <b>diffusely</b>. Erasing all <b>" + lastK +
        "</b> of its top features only moves confidence from <b>" + c.baseline.toFixed(2) + "</b> to <b>" +
        lastV.toFixed(2) + "</b> — against <b>" +
        c.curves.random[c.curves.random.length - 1].toFixed(2) + "</b> for the same number erased at " +
        "random. Different tiles of this type are built from different features, so there is no short list " +
        "to inspect. The per-tile view above still works.";
    }

    /* The tool's own `should_i_trust_it` string went here. It was cut: it talked about "the belief"
     * being "robust" without ever saying what question was asked, and it left the reader unable to
     * tell which outcome is the good one. State both explicitly instead. */
    var meaning = sparse
      ? "<b>This is the good case for interpretability.</b> A short list of features accounts for how " +
        "the model encodes " + tissue(c.concept) + ", so you can look at those features, see the tissue " +
        "behind them, and understand the encoding."
      : "<b>This is the harder case.</b> The model spreads " + tissue(c.concept) + " across many " +
        "features, so no short list explains it. You cannot summarise this tissue type in a handful of " +
        "features — but you can still explain any <i>individual</i> tile, which is what the panel above does.";

    el("ev-verdict").innerHTML =
      '<div class="ev-chip" style="color:' + vcol + ';border-color:' + vcol + '">' +
        (sparse ? "CONCENTRATED" : "SPREAD OUT") + "</div>" +
      '<div class="ev-headline">' + headline + "</div>" +
      '<div class="ev-trust">' + meaning + "</div>";

    /* The 4-panel evidence-card PNG used to render here. It was cut: it packed exemplar tiles, a
     * robustness curve, a fallback bar chart and a heat map into one image, duplicating what the
     * per-tile panel now shows properly and confusing everyone who looked at it. The per-tile view
     * is the explanation; this section is just "does the pattern hold across the dataset". */

    /* 3. THE NUMBERS. Anchor EVERY figure on the page to the same k (the last one), which is also
     * where the chart's endpoint labels sit. Previously the stat tile and the verdict quoted the
     * flip point (k=20) while the chart and the summary table quoted the last k (k=160) — three
     * different "confidence after erasing" numbers on one screen, all correct, none reconcilable
     * by the reader. The flip point still appears, but it is explicitly labelled with its own k. */
    var at = c.curves.sae.length - 1;
    var K = DATA.ks[at];
    el("ev-stats").innerHTML = [
      ["Nothing erased", c.baseline.toFixed(2), "", "the model’s confidence to begin with"],
      ["Its " + K + " features erased", c.curves.sae[at].toFixed(2), tok("--ev-sae", "#d93a4e"),
       "the " + K + " features most associated with this tissue"],
      [K + " random features erased", c.curves.random[at].toFixed(2), tok("--ev-random", "#6b7086"),
       "same number, chosen at random. The control."],
    ].map(function (s) {
      return '<div class="ev-stat"><div class="ev-stat-lab">' + esc(s[0]) + "</div>" +
             '<div class="ev-stat-val"' + (s[2] ? ' style="color:' + s[2] + '"' : "") + ">" +
             esc(s[1]) + "</div>" +
             '<div class="ev-stat-sub">' + esc(s[3]) + "</div></div>";
    }).join("");

    renderCurve(c);

    // "What it falls back on" was cut here: this section is about how the dataset is encoded, not
    // about auditing a single answer, so the fallback tissue is a distraction. It stays in the
    // per-tile panel, where it is about the user's own tile.

    // Implementation detail, last and collapsed. The K Pro user never needs this.
    el("ev-raw").innerHTML =
      "<summary>Technical detail</summary>" +
      "<div class='ev-rawbody'>" +
        "<p>" + esc(c.intervention) + "</p>" +
        "<p>baseline confidence " + c.baseline.toFixed(3) +
        " · features to overturn " + (c.features_to_overturn == null ? "not overturned" : c.features_to_overturn) +
        " · top SAE features [" + c.top_features.join(", ") + "]</p>" +
      "</div>";
  }

  // ------------------------------------------------------------------ boot
  function boot() {
    var view = el("view-evidence");
    if (!view) return;
    loadColors();
    renderHow();

    /* The charts bake their colours into SVG attributes at draw time, so the dashboard's theme
     * toggle (which stamps data-theme on <html>) would leave them stale — dark-theme reds on a
     * light ground. Re-read the tokens and redraw whenever the theme flips. */
    var TILE = null;
    if (window.MutationObserver) {
      new MutationObserver(function () {
        loadColors();
        if (TILE) renderTile(TILE);
        var active = document.querySelector(".ev-tab.active");
        if (active) render(active.dataset.concept);
      }).observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
    }

    // the case: the user's own tile. Independent of the population view — if one is missing the
    // other still renders.
    fetch("sae/tile.json", { cache: "no-store" })
      .then(function (r) { if (!r.ok) throw new Error("tile.json " + r.status); return r.json(); })
      .then(function (t) { TILE = t; renderTile(t); })
      .catch(function (e) {
        var h = el("ev-tile");
        if (h) h.innerHTML = '<div class="ev-missing">Per-tile evidence not precomputed: ' +
          esc(e.message) + ". Run <code>python biolayer/sae/precompute_tile_ui.py --tile " +
          "dashboard/public/input_tile.png --out dashboard/public/sae</code>.</div>";
      });

    fetch("sae/sae.json", { cache: "no-store" })
      .then(function (r) {
        if (!r.ok) throw new Error("sae.json " + r.status);
        return r.json();
      })
      .then(function (j) {
        DATA = j;
        var names = Object.keys(j.concepts || {});
        if (!names.length) throw new Error("no concepts in sae.json");

        // Tabs name the TISSUE, not the dataset's class code. "LYM" means nothing to a pathologist.
        var tabs = el("ev-tabs");
        tabs.innerHTML = names.map(function (n) {
          return '<button class="ev-tab" data-concept="' + esc(n) + '">' + esc(tissue(n)) +
                 ' <span class="ev-tab-code">' + esc(n) + "</span></button>";
        }).join("");
        d3.selectAll(".ev-tab").on("click", function () { render(this.dataset.concept); });

        /* The summary table. Three concepts, three genuinely different answers — that spread IS
         * the result, and it is invisible if the reader has to click each tab and remember. */
        var rows = names.map(function (n) {
          var c = j.concepts[n];
          var sparse = c.encoding === "SPARSE / AUDITABLE";
          var col = sparse ? tok("--ev-sparse", "#a86a00") : tok("--ev-dist", "#1d7a52");
          var overturn = c.features_to_overturn == null
            ? "more than " + DATA.ks[DATA.ks.length - 1]
            : String(c.features_to_overturn);
          var meaning = sparse
            ? "Easy to interpret: a short list of features explains it."
            : "Hard to summarise: no short list. Explain tile by tile instead.";
          return '<tr data-concept="' + esc(n) + '">' +
            "<td><b>" + esc(tissue(n)) + "</b></td>" +
            '<td><span class="ev-pill" style="color:' + col + ';border-color:' + col + '">' +
              (sparse ? "CONCENTRATED" : "SPREAD OUT") + "</span></td>" +
            '<td class="num">' + c.baseline.toFixed(2) + " &rarr; <b style=\"color:" +
              tok("--ev-sae", "#d93a4e") + '">' + c.curves.sae[c.curves.sae.length - 1].toFixed(2) +
              "</b></td>" +
            '<td class="num">' + esc(overturn) + "</td>" +
            "<td>" + esc(meaning) + "</td></tr>";
        }).join("");

        el("ev-summary").innerHTML =
          '<div class="ev-sum-lab">Not every tissue type is encoded the same way</div>' +
          '<div class="ev-sum-scroll"><table class="ev-sum-table">' +
          "<thead><tr><th>Tissue type</th><th>How it is encoded</th>" +
          "<th>Confidence after erasing its features</th><th>Features needed to flip it</th>" +
          "<th>What that tells you</th></tr></thead><tbody>" + rows + "</tbody></table></div>";

        d3.selectAll(".ev-sum-table tbody tr").on("click", function () {
          render(this.dataset.concept);
          el("ev-verdict").scrollIntoView({ behavior: "smooth", block: "nearest" });
        });

        render(names[0]);
        window.addEventListener("resize", function () {
          var a = document.querySelector(".ev-tab.active");
          if (a && document.body.dataset.view === "evidence") render(a.dataset.concept);
        });
      })
      .catch(function (e) {
        el("ev-verdict").innerHTML =
          '<div class="ev-missing">SAE evidence not available: ' + esc(e.message) +
          '. Run <code>python scripts/precompute_sae_ui.py --out dashboard/public/sae</code>.</div>';
      });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot);
  } else {
    boot();
  }
})();
