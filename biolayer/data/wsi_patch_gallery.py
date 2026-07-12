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
            .replace("__SRC_TITLE__", html.escape(stem)).replace("__SRCURI__", html.escape(slide))
            .replace("__OVERVIEW__", over_uri)
            .replace("__PATCHES__", json.dumps(out))
            .replace("__PATCHES_BOTTOM__", "null")
            .replace("__PATCHES_TILE__", "null").replace("__PATCHES_TILE_BOTTOM__", "null")
            .replace("__L0W__", str(l0w)).replace("__L0H__", str(l0h))
            .replace("__MPP_NUM__", repr(mpp if mpp else 0.0))
            .replace("__MPP_TXT__", mpp_txt).replace("__MAG_TXT__", mag_txt)
            .replace("__AXIS_NOTE__", "").replace("__SOURCES__", "null").replace("__AXES__", "null"))

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
<title>__STEM__ — slide gallery</title>
<script>
  /* set the theme BEFORE first paint so dark mode never flashes light on reload */
  (function () {
    try {
      // shared key with the dashboard (same origin) so the whole UI themes uniformly
      var m = localStorage.getItem('kscope:theme') || localStorage.getItem('gallery-theme')
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
  /* kill all transitions during a theme flip so elements with their own transition (e.g. the
     top/bottom-24 buttons) recolor instantly instead of cross-fading — no "light up" flash */
  :root.theme-switch * { transition: none !important; }
  body { margin: 0; background: var(--bg); color: var(--ink);
    font-family: var(--sans); line-height: 1.5; -webkit-font-smoothing: antialiased; }
  .wrap { max-width: 1180px; margin: 0 auto; padding: clamp(18px, 3.5vw, 44px); position: relative; }
  header { display: flex; flex-wrap: wrap; align-items: baseline; gap: 6px 16px; }
  .eyebrow { font-family: var(--mono); font-size: 12px; letter-spacing: .12em;
    text-transform: uppercase; color: var(--accent); font-weight: 600; }
  h1 { font-family: var(--mono); font-size: clamp(20px, 3.2vw, 30px); font-weight: 600;
    margin: 2px 0 0; letter-spacing: -0.01em; width: 100%; }
  .sub { color: var(--muted); font-size: 14px; margin: 4px 0 0; word-break: break-all; }
  /* sits at the right edge of the header row (margin-left:auto) instead of a full-width line
     below the title — packs the title block horizontally rather than stacking it vertically */
  .axis-note { margin: 4px 0 0 auto; font-family: var(--mono); font-size: 12.5px;
    color: var(--ink); display: inline-flex; align-items: center; gap: 8px; text-align: right; }
  .axis-note .pill { background: color-mix(in srgb, var(--accent) 16%, transparent);
    color: var(--accent); border-radius: 6px; padding: 2px 8px; font-weight: 600; }

  /* source (WSI) + axis (concept) switchers */
  .src-switch, .ax-switch { position: relative; display: inline-flex; align-items: center; gap: 4px; }
  .ax-switch { gap: 3px; }
  .src-switch.switchable, .ax-switch.switchable { cursor: pointer; }
  .src-switch.switchable:hover code.k { text-decoration: underline; }
  .ax-switch.switchable:hover .pill { filter: brightness(1.08); }
  .src-caret, .ax-caret { color: var(--muted); font-size: 10px; }
  .ax-caret { font-size: 9px; }
  .src-menu, .ax-menu { position: absolute; top: 100%; left: 0; margin-top: 7px; z-index: 30;
    background: var(--panel); border: 1px solid var(--line); border-radius: 11px; padding: 6px;
    min-width: 260px; box-shadow: 0 14px 34px -14px rgba(0,0,0,.55);
    display: flex; flex-direction: column; gap: 2px; }
  .ax-menu { min-width: 210px; left: auto; right: 0; }  /* right-aligned: the switcher now hugs the right edge */
  .src-item { text-align: left; background: none; border: 0; border-radius: 8px; padding: 8px 11px;
    font-family: var(--sans); font-size: 12.5px; color: var(--ink); cursor: pointer;
    display: flex; align-items: center; gap: 8px; }
  .src-item:hover:not(:disabled) { background: color-mix(in srgb, var(--accent) 12%, transparent); }
  .src-item.current { color: var(--accent); font-weight: 600; cursor: default; }
  .src-item.current::before { content: "●"; font-size: 8px; }
  .src-item:not(.current)::before { content: "○"; font-size: 8px; color: var(--muted); }

  /* rank-unit ("Rank by") + rank-end (top/bottom) toggles share one wrapping row */
  .toggle-row { display: flex; flex-wrap: wrap; align-items: center; gap: 10px 26px; }
  .rank-toggle { display: flex; align-items: center; gap: 12px; margin-top: 14px; }
  .rt-label { font-family: var(--mono); font-size: 11px; letter-spacing: .1em; text-transform: uppercase;
    color: var(--muted); }
  .rt-seg { display: inline-flex; border: 1px solid var(--line); border-radius: 9px; overflow: hidden; }
  .rt-btn { background: var(--panel); color: var(--muted); border: 0; cursor: pointer;
    font-family: var(--mono); font-size: 12px; padding: 6px 13px; transition: background .12s, color .12s; }
  .rt-btn + .rt-btn { border-left: 1px solid var(--line); }
  .rt-btn[aria-pressed="true"] { background: var(--accent); color: #fff; }

  /* dashed unit boxes drawn on the stage view: subtle scale markers for the ranked unit.
     Centered on the crop (the crop is centered on the ranked patch/tile); sized in JS. */
  .stage-imgwrap { position: relative; display: block; width: 100%; }
  /* dashed marker for the ranked unit, sized in JS. Red reads strongly on H&E and matches
     the active-region red; a dark + light 1px outline keeps the dash sharp on any background. */
  .unit-box { position: absolute; left: 50%; top: 50%; transform: translate(-50%, -50%);
    pointer-events: none; box-sizing: border-box; border-radius: 2px; min-width: 6px; min-height: 6px;
    border: 2px dashed #ff1f2e;
    box-shadow: 0 0 0 1px rgba(0,0,0,.6), inset 0 0 0 1px rgba(255,255,255,.55); }
  /* bold red arrow that points at the (tiny) 14×14 patch box from the top-right. Toggled via
     the .show class, NOT the [hidden] attribute — SVGElement has no reflecting .hidden IDL prop,
     so `svg.hidden = false` would leave the attribute set and CSS would keep it display:none. */
  .unit-arrow { display: none; position: absolute; inset: 0; width: 100%; height: 100%;
    pointer-events: none; overflow: visible; filter: drop-shadow(0 0.5px 1px rgba(0,0,0,.4)); }
  .unit-arrow.show { display: block; }

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

  /* Desktop: fit the whole gallery to one screen — no page scrollbars. The .wrap becomes a
     100%-height flex column, the layout fills the leftover height, and the square stage image
     is sized to the largest square that fits the remaining box (min of the box's w/h, via
     container-query units). Guarded by @supports so unsupported browsers keep the scroll flow. */
  .stage-imgbox { flex: 1 1 auto; min-height: 0; display: flex; align-items: center; justify-content: center; }
  @media (min-width: 880px) {
    @supports (container-type: size) {
      html, body { height: 100%; }
      .wrap { height: 100%; display: flex; flex-direction: column; overflow: hidden; }
      .layout { flex: 1 1 auto; min-height: 0; align-items: stretch; }
      .stage { min-height: 0; overflow: hidden; }
      .stage-imgbox { container-type: size; }
      .stage-imgwrap { width: min(100cqw, 100cqh); height: auto; aspect-ratio: 1 / 1; }
      .stage-img { height: 100%; aspect-ratio: auto; }
      /* aside may scroll; the rail must NOT (overflow-y:auto would force overflow-x and clip
         the selected thumb's red outline) — pagination already caps it at 6 thumbs. */
      .aside { min-height: 0; overflow-y: auto; }
      .rail { min-height: 0; }
    }
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
  .thumb[aria-current="true"] { outline-color: #ff1f2e; }
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
  .locbox.active { outline: 2px solid #ff1f2e; z-index: 3; box-shadow: 0 0 0 1px #fff; }
  .loc-legend { display: flex; gap: 16px; margin: 7px 2px 0; font-size: 11.5px; color: var(--muted);
    font-family: var(--mono); flex-wrap: wrap; }
  .loc-legend span { display: inline-flex; align-items: center; gap: 6px; }
  .swatch { width: 11px; height: 11px; border-radius: 3px; outline-offset: -1px; }
  .swatch.a { outline: 2.5px solid #ff1f2e; }
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
    <span class="eyebrow">Whole-slide image · slide gallery</span>
    <h1>__STEM__</h1>
    <p class="sub">H&amp;E · Aperio SVS · __MAG_TXT__ ·
      <span class="src-switch" id="src-switch">
        <code class="k" id="src-uri" title="__SRCURI__">__SRC_TITLE__</code><span class="src-caret" id="src-caret" hidden>▾</span>
        <span class="src-menu" id="src-menu" hidden></span>
      </span>
    </p>
    __AXIS_NOTE__
  </header>

  <div class="toggle-row">
    <div id="unit-toggle" class="rank-toggle" hidden>
      <span class="rt-label">Rank by</span>
      <div class="rt-seg">
        <button id="ut-tile" class="rt-btn" type="button" aria-pressed="true">224×224 tile</button>
        <button id="ut-patch" class="rt-btn" type="button" aria-pressed="false">14×14 patch</button>
      </div>
    </div>
    <div id="rank-toggle" class="rank-toggle" hidden>
      <div class="rt-seg">
        <button id="rt-top" class="rt-btn" type="button" aria-pressed="true">▲ Top 24</button>
        <button id="rt-bottom" class="rt-btn" type="button" aria-pressed="false">▼ Bottom 24</button>
      </div>
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
      <div class="stage-imgbox">
        <div class="stage-imgwrap" id="unit-overlay">
          <img class="stage-img" id="stage-img" alt="">
          <div class="unit-box box-tile" id="box-tile" hidden></div>
          <div class="unit-box box-patch" id="box-patch" hidden></div>
          <!-- bold arrow that points at the tiny 14×14 box (shown only in patch mode) -->
          <svg class="unit-arrow" id="unit-arrow" viewBox="0 0 100 100" aria-hidden="true">
            <defs>
              <!-- refX = the tip's x so the arrowhead POINT sits exactly on the line's end;
                   butt cap (no round) so the stroke can't poke past the tip -->
              <marker id="ua-head" markerUnits="userSpaceOnUse" markerWidth="5" markerHeight="5"
                      refX="4.4" refY="2.2" orient="auto">
                <path d="M0,0 L4.4,2.2 L0,4.4 Z" fill="#ff1f2e"></path>
              </marker>
            </defs>
            <line x1="68" y1="27" x2="53" y2="46.2" stroke="#ff1f2e" stroke-width="1.3"
                  marker-end="url(#ua-head)"></line>
          </svg>
        </div>
      </div>
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
          <dt>View</dt><dd id="md-size">—</dd>
          <dt>Unit</dt><dd id="md-unit">—</dd>
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
  const PATCHES_TILE_TOP = __PATCHES_TILE__;        // 224×224 tile-embedding ranking (or null)
  const PATCHES_TILE_BOTTOM = __PATCHES_TILE_BOTTOM__;
  const MPP = __MPP_NUM__;
  // ranking unit (tile|patch) × axis end; default to the 224×224 tile when that ranking exists
  let unit = (PATCHES_TILE_TOP && PATCHES_TILE_TOP.length) ? 'tile' : 'patch', curEnd = 'top';
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
    document.getElementById('st-note').textContent = `${p.w} × ${p.h} px crop`;
    document.getElementById('md-title').textContent = p.title;
    document.getElementById('md-desc').textContent = p.desc;
    document.getElementById('md-center').textContent = `${p.cx}, ${p.cy}`;
    document.getElementById('md-origin').textContent = `${p.ox}, ${p.oy}`;
    document.getElementById('md-size').textContent = `${p.w} × ${p.h} px`;  // the view crop
    // the ranked unit: a 14×14 patch token (224px-tile footprint / 16) or the whole 224×224 tile.
    if (MPP > 0) {
      const tilePx = Math.round(224 * Math.max(0.5 / MPP, 1)), tokPx = Math.round(tilePx / 16);
      document.getElementById('md-unit').textContent = unit === 'tile'
        ? `224×224 tile · ~${tilePx} px · ${Math.round(tilePx * MPP)} µm`
        : `14×14 patch · ~${tokPx} px · ${(tokPx * MPP).toFixed(1)} µm`;
    } else {
      document.getElementById('md-unit').textContent = unit === 'tile' ? '224×224 tile' : '14×14 patch token';
    }
    updateUnitBoxes(p.w);
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
    // disable transitions, commit the new palette in one synchronous reflow, then re-enable
    // on the next frame — the recolor is instant, so nothing cross-fades / flashes.
    tRoot.classList.add('theme-switch');
    tRoot.setAttribute('data-theme', mode);
    void tRoot.offsetWidth;                                  // force the recalc while frozen
    requestAnimationFrame(() => tRoot.classList.remove('theme-switch'));
    if (tIcon) tIcon.textContent = mode === 'dark' ? '☀' : '◐';
    if (tLabel) tLabel.textContent = mode === 'dark' ? 'Light' : 'Dark';
  }
  // Expose applyTheme so the embedding dashboard can drive the gallery's theme from its own
  // (single, bottom-left) toggle — keeps the whole UI on one uniform switch.
  window.__applyTheme = applyTheme;
  // localStorage throws a SecurityError on opaque origins (e.g. file://); guard it so a
  // failed theme read/write can't halt the whole script and leave the gallery un-wired.
  // Shared key with the dashboard (same origin) so a choice made anywhere sticks everywhere.
  let savedTheme = null;
  try { savedTheme = localStorage.getItem('kscope:theme') || localStorage.getItem('gallery-theme'); } catch (e) {}
  applyTheme(savedTheme || (matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'));
  tBtn.addEventListener('click', () => {
    const next = tRoot.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
    applyTheme(next);
    try { localStorage.setItem('kscope:theme', next); } catch (e) {}
  });
  // When embedded in the dashboard, the single bottom-left toggle governs the theme — hide the
  // gallery's own top-right button so there's exactly one control for the whole UI.
  if (window.parent !== window) tBtn.style.display = 'none';

  // source (WSI) switcher: make the header link open a menu of known slides
  // Generic dropdown switcher (used for both the WSI source and the ranking axis): keep only
  // entries whose target gallery exists (HEAD-probe, no 404s), current always kept + on top.
  function wireSwitcher(switchEl, menuEl, caretEl, items, itemClass) {
    if (!switchEl || !items || items.length <= 1) return;
    Promise.all(items.map(s => s.current
        ? Promise.resolve(true)
        : fetch(s.href, { method: 'HEAD' }).then(r => r.ok).catch(() => false)))
      .then(oks => {
        const avail = items.filter((s, i) => oks[i])
                           .sort((a, b) => (b.current ? 1 : 0) - (a.current ? 1 : 0));
        if (avail.length <= 1) return;
        caretEl.hidden = false;
        switchEl.classList.add('switchable');
        avail.forEach(s => {
          const it = document.createElement('button');
          it.type = 'button'; it.className = itemClass + (s.current ? ' current' : '');
          it.textContent = s.label; it.disabled = !!s.current;
          it.addEventListener('click', (e) => { e.stopPropagation(); if (!s.current) location.href = s.href; });
          menuEl.appendChild(it);
        });
        switchEl.addEventListener('click', () => { menuEl.hidden = !menuEl.hidden; });
        document.addEventListener('click', (e) => { if (!switchEl.contains(e.target)) menuEl.hidden = true; });
      });
  }
  wireSwitcher(document.getElementById('src-switch'), document.getElementById('src-menu'),
               document.getElementById('src-caret'), __SOURCES__, 'src-item');
  wireSwitcher(document.getElementById('ax-switch'), document.getElementById('ax-menu'),
               document.getElementById('ax-caret'), __AXES__, 'src-item ax-item');

  // Ranked-set selection = unit (14×14 patch | 224×224 tile) × axis end (top | bottom).
  const rankToggle = document.getElementById('rank-toggle'),
        rtTop = document.getElementById('rt-top'), rtBottom = document.getElementById('rt-bottom');
  const unitToggle = document.getElementById('unit-toggle'),
        utPatch = document.getElementById('ut-patch'), utTile = document.getElementById('ut-tile');
  const boxTile = document.getElementById('box-tile'), boxPatch = document.getElementById('box-patch');
  const unitArrow = document.getElementById('unit-arrow');

  function currentSet() {
    if (unit === 'tile') return curEnd === 'bottom' ? PATCHES_TILE_BOTTOM : PATCHES_TILE_TOP;
    return curEnd === 'bottom' ? PATCHES_BOTTOM : PATCHES_TOP;
  }
  function applySet() {
    PATCHES = currentSet() || PATCHES_TOP;
    nPages = Math.ceil(PATCHES.length / PAGE);
    page = 0; active = 0;
    renderBoxes(); renderThumbs(); setActive(0);
  }
  // dashed scale boxes over the view: the 224×224 tile (the embedding square) is always drawn;
  // the 14×14 patch box only when ranking by patch (then the crop is centered on that patch).
  // Both centered — the crop is centered on the ranked unit — and sized as a % of the view width.
  function updateUnitBoxes(view) {
    if (!(MPP > 0) || !boxTile) return;
    view = view || (PATCHES.length ? PATCHES[active].w : 1536);
    const tilePx = Math.round(224 * Math.max(0.5 / MPP, 1)), tokPx = Math.round(tilePx / 16);
    // ONE box, sized to the SELECTED unit so toggling visibly changes it: a whole 224×224 tile
    // (~30% of the view) vs a single 14×14 patch (~2% — deliberately tiny, that's the true scale).
    const px = unit === 'tile' ? tilePx : tokPx;
    boxTile.hidden = false;
    boxTile.style.width = boxTile.style.height = (100 * px / view).toFixed(2) + '%';
    boxPatch.hidden = true;
    if (unitArrow) unitArrow.classList.toggle('show', unit === 'patch');  // arrow only for the tiny 14×14 box
  }

  function setEnd(which) {
    curEnd = which;
    rtTop.setAttribute('aria-pressed', String(which !== 'bottom'));
    rtBottom.setAttribute('aria-pressed', String(which === 'bottom'));
    applySet();
  }
  function setUnit(u) {
    unit = u;
    utPatch.setAttribute('aria-pressed', String(u === 'patch'));
    utTile.setAttribute('aria-pressed', String(u === 'tile'));
    applySet();
  }
  if (PATCHES_BOTTOM) {
    rankToggle.hidden = false;
    rtTop.addEventListener('click', () => setEnd('top'));
    rtBottom.addEventListener('click', () => setEnd('bottom'));
  }
  if (PATCHES_TILE_TOP && PATCHES_TILE_TOP.length) {
    unitToggle.hidden = false;
    utPatch.addEventListener('click', () => setUnit('patch'));
    utTile.addEventListener('click', () => setUnit('tile'));
  }

  applySet();   // initial paint through the same path as the toggles (picks tile/patch set)
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
