"""Render a self-contained interactive HTML gallery of several WSI patches.

One web view with three linked regions: a thumbnail rail (click to switch), a large
native-resolution stage, and a whole-slide map that highlights the active patch (warm
plum) and shows the others (cool blue). Arrow keys / number keys also switch. Images
are embedded as JPEG data URIs, so the .html is fully self-contained and CSP-safe.

    # the built-in 5-patch demo (morphologically distinct regions of the default slide)
    python -m biolayer.data.wsi_patch_gallery

    # your own patches: repeat --patch CX CY TITLE DESC (level-0 center coords)
    python -m biolayer.data.wsi_patch_gallery \
        --patch 24000 44000 "Cellular epithelium" "Crowded glandular nuclei" \
        --patch 20000 30000 "Adipose" "Honeycomb of fat cells" \
        --slide s3://bucketbiolayer/wsi/BRACS/BRACS_1003675.svs

Reads pyramid level 0 once and cuts every tile from it, so N patches cost one full-res
read (not N). Shares the S3 download + AccessDenied remediation with wsi_thumbnail.
"""
import argparse
import base64
import html
import io
import json
import os
import sys

from . import wsi_thumbnail as wt

# Built-in demo: five morphologically distinct regions of the default BRACS slide,
# picked from a low-pyramid scout sheet. (center_x, center_y, title, description)
DEMO_PATCHES = [
    (24000, 44000, "Cellular epithelium", "Dense epithelial/glandular nests with crowded hyperchromatic nuclei."),
    (47000, 6800,  "Fibrous band",        "Compact eosinophilic collagen — dense fibrous connective tissue."),
    (41000, 34000, "Loose stroma",        "Wispy collagen with scattered spindle fibroblast nuclei."),
    (8500,  9500,  "Tissue margin",       "Edge of the section meeting bare glass, with a tissue fold."),
    (20000, 30000, "Adipose",             "Honeycomb of empty adipocytes — fatty tissue, low cellularity."),
]


def _jpeg_uri(arr, quality: int, maxside=None):
    from PIL import Image
    import numpy as np
    im = Image.fromarray(np.asarray(arr)[..., :3].astype("uint8"))
    if maxside and max(im.size) > maxside:
        im.thumbnail((maxside, maxside), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=quality, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii"), im.size


def build_gallery(slide: str, patches, out_html: str, scratch: str,
                  size: int = 1024, quality: int = 86, display_max: int = 1024,
                  overview_max: int = 1400) -> dict:
    """Read each patch via random-access region reads and write the self-contained HTML.

    Every tile is fetched with the OpenSlide-backed reader, which decodes only the
    tiles overlapping the region — so N patches cost N small reads, and a slide whose
    level 0 exceeds RAM (e.g. a 125k×79k TCGA-BRCA slide) works the same as a small one.
    """
    from .wsi_reader import open_wsi

    local = slide if not slide.startswith("s3://") else wt._download(slide, scratch)
    stem = os.path.splitext(os.path.basename(local))[0]

    reader = open_wsi(local)
    try:
        l0w, l0h = reader.dimensions
        mpp = reader.mpp
        mag = round(10.0 / mpp) if mpp else None

        # whole-slide overview (bounded thumbnail, never a full-res plane) -> JPEG.
        over_arr, _ = reader.thumbnail(overview_max)
        over_uri, _ = _jpeg_uri(over_arr, quality=82, maxside=overview_max)

        print(f"[gallery] {reader.__class__.__name__}: reading {len(patches)} regions "
              f"from level 0 ({l0w}x{l0h})…", file=sys.stderr)
        out = []
        for cx, cy, title, desc in patches:
            ox = max(0, min(int(cx) - size // 2, l0w - size))
            oy = max(0, min(int(cy) - size // 2, l0h - size))
            crop = reader.read_region((ox, oy), 0, (size, size))   # RGB PIL.Image
            uri, _ = _jpeg_uri(crop, quality=quality, maxside=display_max)
            out.append({
                "title": title, "desc": desc,
                "cx": ox + size // 2, "cy": oy + size // 2, "ox": ox, "oy": oy,
                "w": size, "h": size, "img": uri,
                "box": {"l": round(100 * ox / l0w, 3), "t": round(100 * oy / l0h, 3),
                        "w": round(100 * size / l0w, 3), "h": round(100 * size / l0h, 3)},
            })
            print(f"  {title:20s} origin=({ox},{oy}) {len(uri)//1024}KB", file=sys.stderr)
    finally:
        reader.close()

    mpp_txt = f"{mpp:.4f} µm/px" if mpp else "unknown"
    mag_txt = f"≈ {mag}×" if mag else "unknown"
    page = (_TEMPLATE
            .replace("__STEM__", html.escape(stem))
            .replace("__SRCURI__", html.escape(slide))
            .replace("__OVERVIEW__", over_uri)
            .replace("__PATCHES__", json.dumps(out))
            .replace("__PATCHES_BOTTOM__", "null")
            .replace("__L0W__", str(l0w)).replace("__L0H__", str(l0h))
            .replace("__MPP_NUM__", repr(mpp if mpp else 0.0))
            .replace("__MPP_TXT__", mpp_txt).replace("__MAG_TXT__", mag_txt)
            .replace("__AXIS_NOTE__", "").replace("__SOURCES__", "null"))

    os.makedirs(os.path.dirname(os.path.abspath(out_html)), exist_ok=True)
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(page)
    return {"output_html": out_html, "n_patches": len(out),
            "level0_dimensions": [l0w, l0h], "mpp_um_per_px": mpp, "magnification": mag,
            "bytes": os.path.getsize(out_html)}


# Self-contained, CSP-safe (no external fonts/scripts). __TOKENS__ are filled in above.
_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__STEM__ — patch gallery</title>
<script>
  /* set the theme BEFORE first paint so dark mode never flashes light on reload */
  (function () {
    try {
      var m = localStorage.getItem('gallery-theme')
              || (matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');
      document.documentElement.setAttribute('data-theme', m);
    } catch (e) {}
  })();
</script>
<style>
  :root {
    --bg: #f4f1f4; --panel: #fbfafb; --ink: #241f27; --muted: #6c6172;
    --line: #e4dde5; --accent: #8d3f6e; --other: #3f6f9e; --stage: #1b141c;
    --stage-ink: #d9cfda; --stage-muted: #8f8091; --good: #2f7d55;
    --mono: ui-monospace, "SF Mono", "JetBrains Mono", Menlo, Consolas, monospace;
    --sans: "Inter", system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --bg: #161016; --panel: #1f1820; --ink: #ece5ed; --muted: #a596a6;
      --line: #322936; --accent: #cd7fb0; --other: #5f97c9; --stage: #0d090e;
      --stage-ink: #d9cfda; --stage-muted: #7c6e7e;
    }
  }
  :root[data-theme="light"] {
    --bg: #f4f1f4; --panel: #fbfafb; --ink: #241f27; --muted: #6c6172;
    --line: #e4dde5; --accent: #8d3f6e; --other: #3f6f9e; --stage: #1b141c;
    --stage-ink: #d9cfda; --stage-muted: #8f8091;
  }
  :root[data-theme="dark"] {
    --bg: #161016; --panel: #1f1820; --ink: #ece5ed; --muted: #a596a6;
    --line: #322936; --accent: #cd7fb0; --other: #5f97c9; --stage: #0d090e;
    --stage-ink: #d9cfda; --stage-muted: #7c6e7e;
  }
  * { box-sizing: border-box; }
  [hidden] { display: none !important; }   /* keep the hidden attr authoritative over flex/grid rules */
  body { margin: 0; background: var(--bg); color: var(--ink);
    font-family: var(--sans); line-height: 1.5; -webkit-font-smoothing: antialiased; }
  .wrap { max-width: 1180px; margin: 0 auto; padding: clamp(18px, 3.5vw, 44px); position: relative; }
  header { display: flex; flex-wrap: wrap; align-items: baseline; gap: 6px 16px; }
  .eyebrow { font-family: var(--mono); font-size: 12px; letter-spacing: .12em;
    text-transform: uppercase; color: var(--accent); font-weight: 600; }
  h1 { font-family: var(--mono); font-size: clamp(20px, 3.2vw, 30px); font-weight: 600;
    margin: 2px 0 0; letter-spacing: -0.01em; width: 100%; }
  .sub { color: var(--muted); font-size: 14px; margin: 4px 0 0; word-break: break-all; }
  .axis-note { width: 100%; margin: 10px 0 0; font-family: var(--mono); font-size: 12.5px;
    color: var(--ink); display: inline-flex; align-items: center; gap: 8px; }
  .axis-note .pill { background: color-mix(in srgb, var(--accent) 16%, transparent);
    color: var(--accent); border-radius: 6px; padding: 2px 8px; font-weight: 600; }

  /* source (WSI) switcher on the header link */
  .src-switch { position: relative; display: inline-flex; align-items: center; gap: 4px; }
  .src-switch.switchable { cursor: pointer; }
  .src-switch.switchable:hover code.k { text-decoration: underline; }
  .src-caret { color: var(--muted); font-size: 10px; }
  .src-menu { position: absolute; top: 100%; left: 0; margin-top: 7px; z-index: 30;
    background: var(--panel); border: 1px solid var(--line); border-radius: 11px; padding: 6px;
    min-width: 260px; box-shadow: 0 14px 34px -14px rgba(0,0,0,.55);
    display: flex; flex-direction: column; gap: 2px; }
  .src-item { text-align: left; background: none; border: 0; border-radius: 8px; padding: 8px 11px;
    font-family: var(--sans); font-size: 12.5px; color: var(--ink); cursor: pointer;
    display: flex; align-items: center; gap: 8px; }
  .src-item:hover:not(:disabled) { background: color-mix(in srgb, var(--accent) 12%, transparent); }
  .src-item.current { color: var(--accent); font-weight: 600; cursor: default; }
  .src-item.current::before { content: "●"; font-size: 8px; }
  .src-item:not(.current)::before { content: "○"; font-size: 8px; color: var(--muted); }

  /* rank-end toggle (top vs bottom of the axis) */
  .rank-toggle { display: flex; align-items: center; gap: 12px; margin-top: 14px; }
  .rt-label { font-family: var(--mono); font-size: 11px; letter-spacing: .1em; text-transform: uppercase;
    color: var(--muted); }
  .rt-seg { display: inline-flex; border: 1px solid var(--line); border-radius: 9px; overflow: hidden; }
  .rt-btn { background: var(--panel); color: var(--muted); border: 0; cursor: pointer;
    font-family: var(--mono); font-size: 12px; padding: 6px 13px; transition: background .12s, color .12s; }
  .rt-btn + .rt-btn { border-left: 1px solid var(--line); }
  .rt-btn[aria-pressed="true"] { background: var(--accent); color: #fff; }

  /* three regions: thumb rail · stage · aside */
  .layout { display: grid; gap: 20px; margin-top: 16px;
    grid-template-columns: 104px minmax(0,1fr); grid-template-areas: "rail stage" "rail aside"; }
  @media (min-width: 880px) {
    .layout { grid-template-columns: 108px minmax(0,1.5fr) minmax(240px,1fr);
      grid-template-areas: "rail stage aside"; align-items: start; }
  }
  @media (max-width: 879px) {
    .layout { grid-template-columns: 1fr; grid-template-areas: "rail" "stage" "aside"; }
  }

  /* thumbnail rail */
  .rail { grid-area: rail; display: flex; flex-direction: column; gap: 7px; }
  @media (max-width: 879px) { .rail { flex-direction: row; overflow-x: auto; padding-bottom: 4px; } }
  .rail-h { font-family: var(--mono); font-size: 10px; letter-spacing: .14em; text-transform: uppercase;
    color: var(--muted); margin: 2px 0 2px 2px; }
  @media (max-width: 879px) { .rail-h { display: none; } }
  .thumb { position: relative; border: 0; padding: 0; background: none; cursor: pointer;
    border-radius: 10px; overflow: hidden; flex: 0 0 auto; width: 100%;
    outline: 2px solid transparent; outline-offset: 2px; transition: outline-color .12s, transform .12s; }
  @media (max-width: 879px) { .thumb { width: 92px; } }
  .thumb img { display: block; width: 100%; height: clamp(46px, 9.5vh, 88px); object-fit: cover;
    border-radius: 9px; filter: saturate(.96); }
  @media (max-width: 879px) { .thumb img { height: 92px; } }
  .thumb .tcap { position: absolute; left: 0; right: 0; bottom: 0; padding: 10px 6px 4px;
    font-size: 10px; line-height: 1.15; color: #fff; text-align: left; font-weight: 500;
    background: linear-gradient(transparent, rgba(15,10,16,.82)); }
  .thumb[aria-current="true"] { outline-color: var(--accent); }
  .thumb[aria-current="true"] img { filter: saturate(1.05); }
  .thumb:not([aria-current="true"]) img { opacity: .82; }
  .thumb:hover:not([aria-current="true"]) { transform: translateY(-1px); }
  .thumb:hover:not([aria-current="true"]) img { opacity: 1; }
  .thumb:focus-visible { outline-color: var(--accent); }

  /* stage */
  .stage { grid-area: stage; background: var(--stage); border-radius: 14px;
    padding: clamp(12px,2vw,22px); display: flex; flex-direction: column; gap: 12px;
    box-shadow: 0 1px 0 var(--line), 0 18px 40px -24px rgba(0,0,0,.5); }
  .stage-title { display: flex; align-items: baseline; gap: 10px; color: var(--stage-ink);
    font-family: var(--mono); font-size: 13px; }
  .stage-title b { font-size: 15px; font-weight: 600; }
  .stage-title .dot { width: 9px; height: 9px; border-radius: 3px; background: var(--accent);
    box-shadow: 0 0 10px 1px color-mix(in srgb, var(--accent) 70%, transparent); flex: 0 0 auto; }
  .stage-img { width: 100%; aspect-ratio: 1/1; border-radius: 6px; display: block; object-fit: cover;
    background: #000; box-shadow: 0 0 0 1px rgba(255,255,255,.06), inset 0 0 60px rgba(0,0,0,.35); }
  .stage-foot { display: flex; justify-content: space-between; align-items: center; gap: 12px;
    font-family: var(--mono); font-size: 11.5px; color: var(--stage-muted); }
  .scalebar { display: flex; align-items: center; gap: 8px; color: var(--stage-ink); }
  .scalebar .bar { height: 3px; background: var(--stage-ink); border-radius: 2px; }

  /* aside */
  .aside { grid-area: aside; display: flex; flex-direction: column; gap: 12px; }
  .panel { background: var(--panel); border: 1px solid var(--line); border-radius: 14px; padding: 12px 15px; }
  .panel h2 { font-family: var(--mono); font-size: 11px; letter-spacing: .14em; text-transform: uppercase;
    color: var(--muted); font-weight: 600; margin: 0 0 9px; }
  .desc { font-size: 13px; color: var(--ink); margin: -2px 0 10px; }
  dl { margin: 0; display: grid; grid-template-columns: auto 1fr; gap: 6px 14px; }
  dt { color: var(--muted); font-size: 13px; }
  dd { margin: 0; font-family: var(--mono); font-size: 13px; text-align: right; font-variant-numeric: tabular-nums; }

  .loc { position: relative; border-radius: 8px; overflow: hidden; border: 1px solid var(--line); }
  .loc img { display: block; width: 100%; }
  .locbox { position: absolute; border-radius: 3px; transform: translate(-1px,-1px); min-width: 15px; min-height: 15px; }
  .locbox.other { outline: 1.5px solid var(--other);
    box-shadow: 0 0 0 1px rgba(255,255,255,.5); cursor: pointer; }
  .locbox.active { outline: 2.5px solid var(--accent); z-index: 3;
    box-shadow: 0 0 0 1.5px #fff, 0 0 16px 4px color-mix(in srgb, var(--accent) 75%, transparent); }
  .loc-legend { display: flex; gap: 16px; margin: 7px 2px 0; font-size: 11.5px; color: var(--muted);
    font-family: var(--mono); flex-wrap: wrap; }
  .loc-legend span { display: inline-flex; align-items: center; gap: 6px; }
  .swatch { width: 11px; height: 11px; border-radius: 3px; outline-offset: -1px; }
  .swatch.a { outline: 2.5px solid var(--accent); }
  .swatch.o { outline: 1.5px solid var(--other); }

  footer { margin-top: 24px; padding-top: 14px; border-top: 1px solid var(--line);
    font-family: var(--mono); font-size: 12px; color: var(--muted);
    display: flex; flex-wrap: wrap; gap: 6px 14px; justify-content: space-between; }
  code.k { color: var(--accent); }
  .hint { font-size: 11.5px; color: var(--muted); margin: 2px 2px 0; }

  /* dark-mode toggle */
  .theme-toggle { position: absolute; top: clamp(18px,3.5vw,44px); right: clamp(18px,3.5vw,44px);
    background: var(--panel); color: var(--ink); border: 1px solid var(--line); border-radius: 9px;
    font-family: var(--mono); font-size: 12px; padding: 6px 11px; cursor: pointer; z-index: 5;
    display: inline-flex; align-items: center; gap: 7px; transition: border-color .12s; }
  .theme-toggle:hover { border-color: var(--accent); }

  /* rail pager (max 6 thumbs per page) */
  .pager { display: flex; align-items: center; justify-content: space-between; gap: 6px;
    margin-top: 4px; font-family: var(--mono); font-size: 11px; color: var(--muted); }
  .pager button { background: var(--panel); color: var(--ink); border: 1px solid var(--line);
    border-radius: 7px; width: 26px; height: 24px; cursor: pointer; font-size: 13px; line-height: 1;
    transition: border-color .12s, opacity .12s; }
  .pager button:hover:not(:disabled) { border-color: var(--accent); }
  .pager button:disabled { opacity: .35; cursor: default; }
  .pager .pg-label { flex: 1; text-align: center; }
  @media (max-width: 879px) { .pager { grid-area: rail; } }
</style>
</head>
<body>

<div class="wrap">
  <button id="theme-toggle" class="theme-toggle" type="button" aria-label="Toggle dark mode">
    <span id="theme-icon">◐</span><span id="theme-label">Dark</span></button>
  <header>
    <span class="eyebrow">Whole-slide image · patch gallery</span>
    <h1>__STEM__</h1>
    <p class="sub">H&amp;E · Aperio SVS · __MAG_TXT__ ·
      <span class="src-switch" id="src-switch">
        <code class="k" id="src-uri">__SRCURI__</code><span class="src-caret" id="src-caret" hidden>▾</span>
        <span class="src-menu" id="src-menu" hidden></span>
      </span>
    </p>
    __AXIS_NOTE__
  </header>

  <div id="rank-toggle" class="rank-toggle" hidden>
    <span class="rt-label">Show along axis</span>
    <div class="rt-seg">
      <button id="rt-top" class="rt-btn" type="button" aria-pressed="true">▲ Top 24</button>
      <button id="rt-bottom" class="rt-btn" type="button" aria-pressed="false">▼ Bottom 24</button>
    </div>
  </div>

  <div class="layout">
    <nav class="rail" aria-label="Patches">
      <div class="rail-h">Patches</div>
      <div id="thumbs" style="display:contents"></div>
      <div id="pager" class="pager" hidden>
        <button id="pg-prev" type="button" aria-label="previous 6">&#8249;</button>
        <span class="pg-label" id="pg-label">—</span>
        <button id="pg-next" type="button" aria-label="next 6">&#8250;</button>
      </div>
    </nav>

    <section class="stage">
      <div class="stage-title"><span class="dot"></span><b id="st-title">—</b></div>
      <img class="stage-img" id="stage-img" alt="">
      <div class="stage-foot">
        <span id="st-note">—</span>
        <span class="scalebar"><span class="bar" id="st-bar" style="width:60px"></span><span id="st-scale">—</span></span>
      </div>
    </section>

    <aside class="aside">
      <div class="panel">
        <h2 id="md-title">Patch</h2>
        <p class="desc" id="md-desc">—</p>
        <dl>
          <dt>Center (level 0)</dt><dd id="md-center">—</dd>
          <dt>Origin (level 0)</dt><dd id="md-origin">—</dd>
          <dt>Size</dt><dd id="md-size">—</dd>
          <dt>Resolution</dt><dd>__MPP_TXT__</dd>
          <dt>Magnification</dt><dd>__MAG_TXT__</dd>
        </dl>
      </div>

      <div class="panel">
        <h2>Location on slide</h2>
        <div class="loc">
          <img src="__OVERVIEW__" alt="Whole-slide overview">
          <div id="boxes"></div>
        </div>
        <div class="loc-legend">
          <span><span class="swatch a"></span>active patch</span>
          <span><span class="swatch o"></span>other patches</span>
        </div>
        <p class="hint">Click a thumbnail or a box, or use &uarr; &darr; / 1&ndash;9.</p>
      </div>
    </aside>
  </div>
</div>

<script>
  const PATCHES_TOP = __PATCHES__;
  const PATCHES_BOTTOM = __PATCHES_BOTTOM__;        // array (opposite end of the axis) or null
  const MPP = __MPP_NUM__;
  const PAGE = 6;                                   // at most 6 thumbnails on the rail at once

  const thumbsEl = document.getElementById('thumbs');
  const boxesEl  = document.getElementById('boxes');
  const stageImg = document.getElementById('stage-img');
  const pagerEl  = document.getElementById('pager');
  const pgPrev   = document.getElementById('pg-prev');
  const pgNext   = document.getElementById('pg-next');
  const pgLabel  = document.getElementById('pg-label');
  let PATCHES = PATCHES_TOP;
  let nPages = Math.ceil(PATCHES.length / PAGE);
  let active = 0, page = 0, boxEls = [];

  // location boxes: ALL patches of the current set on the slide map (not paginated)
  function renderBoxes() {
    boxesEl.innerHTML = '';
    PATCHES.forEach((p, i) => {
      const d = document.createElement('div');
      d.className = 'locbox other';
      d.style.left = p.box.l + '%'; d.style.top = p.box.t + '%';
      d.style.width = p.box.w + '%'; d.style.height = p.box.h + '%';
      d.title = p.title;
      d.addEventListener('click', () => setActive(i));
      boxesEl.appendChild(d);
    });
    boxEls = [...boxesEl.children];
  }

  // rail: render only the current page's 6 thumbnails
  function renderThumbs() {
    thumbsEl.innerHTML = '';
    const start = page * PAGE, end = Math.min(start + PAGE, PATCHES.length);
    for (let i = start; i < end; i++) {
      const p = PATCHES[i];
      const b = document.createElement('button');
      b.className = 'thumb'; b.type = 'button'; b.dataset.idx = i;
      b.setAttribute('aria-current', i === active ? 'true' : 'false');
      b.innerHTML = `<img src="${p.img}" alt="${p.title}"><span class="tcap">${p.title}</span>`;
      b.addEventListener('click', () => setActive(i));
      thumbsEl.appendChild(b);
    }
    if (nPages > 1) {
      pagerEl.hidden = false;
      pgLabel.textContent = `${start + 1}–${end} of ${PATCHES.length}`;
    }
  }

  function setActive(i) {
    active = (i + PATCHES.length) % PATCHES.length;
    const p = PATCHES[active];
    const wantPage = Math.floor(active / PAGE);
    if (wantPage !== page) { page = wantPage; renderThumbs(); }
    else { [...thumbsEl.children].forEach(t => t.setAttribute('aria-current', +t.dataset.idx === active ? 'true' : 'false')); }
    boxEls.forEach((bx, k) => bx.className = 'locbox ' + (k === active ? 'active' : 'other'));
    stageImg.src = p.img;
    stageImg.alt = `${p.title} — level-0 tile at (${p.ox}, ${p.oy})`;
    document.getElementById('st-title').textContent = p.title;
    document.getElementById('st-note').textContent = `${p.w} × ${p.h} px · level 0`;
    document.getElementById('md-title').textContent = p.title;
    document.getElementById('md-desc').textContent = p.desc;
    document.getElementById('md-center').textContent = `${p.cx}, ${p.cy}`;
    document.getElementById('md-origin').textContent = `${p.ox}, ${p.oy}`;
    document.getElementById('md-size').textContent = `${p.w} × ${p.h}`;
    if (MPP > 0) {
      const tileUm = p.w * MPP;
      let target = tileUm / 5, mag = Math.pow(10, Math.floor(Math.log10(target)));
      let step = [1,2,5,10].find(s => s*mag >= target) || 10;
      const scaleUm = step * mag, frac = Math.max(0.04, Math.min(0.9, scaleUm / tileUm));
      document.getElementById('st-bar').style.width = (frac * 160).toFixed(0) + 'px';
      document.getElementById('st-scale').textContent = scaleUm.toFixed(0) + ' µm';
    } else {
      document.getElementById('st-scale').textContent = '';
    }
    const at = thumbsEl.querySelector(`[data-idx="${active}"]`);
    if (at) at.scrollIntoView({ block: 'nearest', inline: 'nearest' });
  }

  pgPrev.addEventListener('click', () => { page = (page - 1 + nPages) % nPages; renderThumbs(); });
  pgNext.addEventListener('click', () => { page = (page + 1) % nPages; renderThumbs(); });

  document.addEventListener('keydown', (e) => {
    if (e.key === 'ArrowDown' || e.key === 'ArrowRight') { setActive(active + 1); e.preventDefault(); }
    else if (e.key === 'ArrowUp' || e.key === 'ArrowLeft') { setActive(active - 1); e.preventDefault(); }
    else if (e.key >= '1' && e.key <= String(Math.min(9, PATCHES.length))) setActive(+e.key - 1);
  });

  // dark-mode toggle (CSS already ships [data-theme] palettes; persist the choice)
  const tRoot = document.documentElement, tBtn = document.getElementById('theme-toggle'),
        tIcon = document.getElementById('theme-icon'), tLabel = document.getElementById('theme-label');
  function applyTheme(mode) {
    tRoot.setAttribute('data-theme', mode);
    tIcon.textContent = mode === 'dark' ? '☀' : '◐';
    tLabel.textContent = mode === 'dark' ? 'Light' : 'Dark';
  }
  const savedTheme = localStorage.getItem('gallery-theme');
  applyTheme(savedTheme || (matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'));
  tBtn.addEventListener('click', () => {
    const next = tRoot.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
    applyTheme(next); localStorage.setItem('gallery-theme', next);
  });

  // source (WSI) switcher: make the header link open a menu of known slides
  const SOURCES = __SOURCES__;                      // [{label, href, current}] or null
  const srcSwitch = document.getElementById('src-switch'),
        srcMenu = document.getElementById('src-menu'), srcCaret = document.getElementById('src-caret');
  function buildSourceMenu(items) {
    if (items.length <= 1) return;                  // only the current slide exists -> no switcher
    srcCaret.hidden = false;
    srcSwitch.classList.add('switchable');
    items.forEach(s => {
      const it = document.createElement('button');
      it.type = 'button'; it.className = 'src-item' + (s.current ? ' current' : '');
      it.textContent = s.label; it.disabled = !!s.current;
      it.addEventListener('click', (e) => { e.stopPropagation(); if (!s.current) location.href = s.href; });
      srcMenu.appendChild(it);
    });
    srcSwitch.addEventListener('click', () => { srcMenu.hidden = !srcMenu.hidden; });
    document.addEventListener('click', (e) => { if (!srcSwitch.contains(e.target)) srcMenu.hidden = true; });
  }
  if (SOURCES && SOURCES.length > 1) {
    // only list sources whose gallery file actually exists (no 404s); current is always kept
    Promise.all(SOURCES.map(s => s.current
        ? Promise.resolve(true)
        : fetch(s.href, { method: 'HEAD' }).then(r => r.ok).catch(() => false)))
      .then(oks => buildSourceMenu(SOURCES.filter((s, i) => oks[i])));
  }

  // top / bottom-of-axis toggle (only if a bottom set was supplied)
  const rankToggle = document.getElementById('rank-toggle'),
        rtTop = document.getElementById('rt-top'), rtBottom = document.getElementById('rt-bottom');
  function switchSet(which) {
    PATCHES = which === 'bottom' ? PATCHES_BOTTOM : PATCHES_TOP;
    nPages = Math.ceil(PATCHES.length / PAGE);
    page = 0; active = 0;
    rtTop.setAttribute('aria-pressed', String(which !== 'bottom'));
    rtBottom.setAttribute('aria-pressed', String(which === 'bottom'));
    renderBoxes(); renderThumbs(); setActive(0);
  }
  if (PATCHES_BOTTOM) {
    rankToggle.hidden = false;
    rtTop.addEventListener('click', () => switchSet('top'));
    rtBottom.addEventListener('click', () => switchSet('bottom'));
  }

  renderBoxes();
  renderThumbs();
  setActive(0);
</script>
</body>
</html>
"""


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--patch", action="append", nargs=4, metavar=("CX", "CY", "TITLE", "DESC"),
                    help="a patch: level-0 center x, y, a title, a description (repeatable). "
                         "If omitted, the built-in 5-patch demo is used.")
    ap.add_argument("--slide", default=wt.DEFAULT_S3,
                    help=f"s3:// URI or local path to an .svs (default: {wt.DEFAULT_S3})")
    ap.add_argument("--size", type=int, default=1024, help="tile side in level-0 px (default 1024)")
    ap.add_argument("--quality", type=int, default=86, help="embedded-JPEG quality (default 86)")
    ap.add_argument("--display-max", type=int, default=1024, help="cap each tile's longest side (default 1024)")
    ap.add_argument("--out", default=None, help="output .html path (default: <scratch>/<stem>__gallery.html)")
    ap.add_argument("--scratch", default=wt.SCRATCH, help=f"download/output dir (default {wt.SCRATCH})")
    args = ap.parse_args(argv)

    patches = [(int(cx), int(cy), t, d) for cx, cy, t, d in args.patch] if args.patch else DEMO_PATCHES

    stem = os.path.splitext(os.path.basename(args.slide))[0]
    out = args.out or os.path.join(args.scratch, f"{stem}__gallery.html")

    meta = build_gallery(args.slide, patches, out, args.scratch,
                         size=args.size, quality=args.quality, display_max=args.display_max)

    print(json.dumps(meta, indent=2))
    print(f"\nGallery ready: {meta['output_html']} ({meta['bytes']//1024} KB)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
