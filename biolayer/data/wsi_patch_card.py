"""Render a self-contained HTML "patch card" for an arbitrary (x, y) on a WSI.

Given a level-0 pixel coordinate, this cuts a native-resolution tile, renders the
whole-slide overview with a locator box marking where the tile came from, and writes
a single self-contained .html file (images embedded as data URIs) — the same visual
as the one-off viewer, but parameterized so you can pick any subpatch.

    # tile centered on a chosen point (the natural way to "click" a spot)
    python -m biolayer.data.wsi_patch_card 38512 30512

    # top-left origin instead of center, custom tile size, explicit slide
    python -m biolayer.data.wsi_patch_card 38000 30000 --size 1024 --anchor corner \
        --slide s3://bucketbiolayer/wsi/BRACS/BRACS_1003675.svs

The tile is read at pyramid level 0 (full resolution) and downsampled only for the
web view if it exceeds --display-max. Reuses biolayer.data.wsi_thumbnail for the S3
download + level/region extraction, so S3 auth and the AccessDenied remediation are
identical to that tool.
"""
import argparse
import base64
import html
import os
import sys

from . import wsi_thumbnail as wt


def _data_uri(png_path: str) -> str:
    with open(png_path, "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode("ascii")


def _nice_scale_um(tile_um: float):
    """A round bar length (1/2/5 x 10^n µm) that is roughly a fifth of the tile."""
    target = tile_um / 5.0
    if target <= 0:
        return 100.0
    import math
    mag = 10 ** math.floor(math.log10(target))
    for step in (1, 2, 5, 10):
        if step * mag >= target:
            return step * mag
    return 10 * mag


def build_card(slide: str, x: int, y: int, size: int, anchor: str,
               out_html: str, scratch: str, display_max: int) -> dict:
    """Extract tile + overview for one (x, y) and write the self-contained HTML card."""
    from PIL import Image
    from .wsi_reader import open_wsi

    local = slide if not slide.startswith("s3://") else wt._download(slide, scratch)
    stem = os.path.splitext(os.path.basename(local))[0]

    # center-anchored by default: (x, y) is the middle of the tile, which is how you
    # think about "show me this spot" rather than "this is a corner".
    x0, y0 = (x - size // 2, y - size // 2) if anchor == "center" else (x, y)

    tile_png = os.path.join(scratch, f"{stem}__card_tile_{x0}_{y0}_{size}.png")
    over_png = os.path.join(scratch, f"{stem}__card_overview.png")

    # Random-access reader: OpenSlide reads only the tiles overlapping the region, so
    # this never materializes a full level (works on slides whose level 0 exceeds RAM).
    reader = open_wsi(local)
    try:
        w0, h0 = reader.dimensions
        n_levels = reader.level_count
        mpp = reader.mpp
        mag = round(10.0 / mpp) if mpp else None
        # clamp the requested origin to the slide so the locator box + labels stay truthful.
        x0c = max(0, min(x0, w0 - 1))
        y0c = max(0, min(y0, h0 - 1))
        w_eff = min(size, w0 - x0c)
        h_eff = min(size, h0 - y0c)
        clamped = (x0c, y0c, w_eff, h_eff) != (x0, y0, size, size)

        tile_img = reader.read_region((x0c, y0c), 0, (w_eff, h_eff))   # RGB PIL.Image
        over_arr, _ = reader.thumbnail(1600)                          # bounded, never full-res
    finally:
        reader.close()

    os.makedirs(scratch, exist_ok=True)
    Image.fromarray(over_arr).save(over_png)
    # downscale the tile only for the web view; keep native res otherwise.
    if max(tile_img.size) > display_max:
        tile_img = tile_img.copy()
        tile_img.thumbnail((display_max, display_max), Image.LANCZOS)
    tile_img.save(tile_png)
    disp_w, disp_h = tile_img.size
    downsampled = (disp_w, disp_h) != (w_eff, h_eff)

    # locator box as a percentage of the overview (== percentage of level-0).
    box = {
        "left": 100.0 * x0c / w0, "top": 100.0 * y0c / h0,
        "w": 100.0 * w_eff / w0, "h": 100.0 * h_eff / h0,
    }

    # real scale bar: a round physical length as a fraction of the displayed tile width.
    if mpp:
        tile_um = w_eff * mpp
        scale_um = _nice_scale_um(tile_um)
        scale_frac = max(0.04, min(0.9, scale_um / tile_um))
        scale_label = (f"{scale_um:.0f} µm" if scale_um >= 1 else f"{scale_um:.2g} µm")
    else:
        scale_frac, scale_label = 0.25, "scale n/a"

    res_txt = f"{mpp:.4f} µm/px" if mpp else "unknown"
    mag_txt = f"≈ {mag}×" if mag else "unknown"
    tile_note = f"{w_eff} × {h_eff} px · level 0"
    if downsampled:
        tile_note += f" (shown at {disp_w}×{disp_h})"

    subj = html.escape(os.path.basename(local))
    src_uri = html.escape(slide)

    page = _TEMPLATE
    repl = {
        "__TITLE__": html.escape(stem),
        "__STEM__": html.escape(stem),
        "__SRCURI__": src_uri,
        "__SUBJ__": subj,
        "__PATCH__": _data_uri(tile_png),
        "__OVERVIEW__": _data_uri(over_png),
        "__BOXL__": f"{box['left']:.3f}", "__BOXT__": f"{box['top']:.3f}",
        "__BOXW__": f"{box['w']:.3f}", "__BOXH__": f"{box['h']:.3f}",
        "__ORIGX__": str(x0c), "__ORIGY__": str(y0c),
        "__CTRX__": str(x0c + w_eff // 2), "__CTRY__": str(y0c + h_eff // 2),
        "__SIZE__": f"{w_eff} × {h_eff}",
        "__RES__": res_txt, "__MAG__": mag_txt,
        "__TILENOTE__": tile_note,
        "__L0DIMS__": f"{w0} × {h0}",
        "__NLEV__": str(n_levels),
        "__SCALEFRAC__": f"{scale_frac * 100:.1f}", "__SCALELBL__": scale_label,
        "__CLAMPNOTE__": ("Requested region ran past the slide edge and was clamped."
                          if clamped else ""),
    }
    for k, v in repl.items():
        page = page.replace(k, v)

    os.makedirs(os.path.dirname(os.path.abspath(out_html)), exist_ok=True)
    with open(out_html, "w") as f:
        f.write(page)

    return {
        "output_html": out_html,
        "tile_origin_level0": [x0c, y0c],
        "tile_center_level0": [x0c + w_eff // 2, y0c + h_eff // 2],
        "tile_size": [w_eff, h_eff],
        "displayed_size": [disp_w, disp_h],
        "clamped": clamped,
        "mpp_um_per_px": mpp,
        "magnification": mag,
        "tile_png": tile_png,
        "overview_png": over_png,
    }


# Self-contained page: no external fonts/scripts (Artifact CSP-safe). Placeholders
# are __TOKENS__ so the CSS braces don't collide with str.format.
_TEMPLATE = """<title>__TITLE__ — WSI patch</title>
<style>
  :root {
    --bg: #f4f1f4; --panel: #fbfafb; --ink: #241f27; --muted: #6c6172;
    --line: #e4dde5; --accent: #8d3f6e; --stage: #1b141c;
    --stage-ink: #d9cfda; --stage-muted: #8f8091; --good: #2f7d55;
    --mono: ui-monospace, "SF Mono", "JetBrains Mono", Menlo, Consolas, monospace;
    --sans: "Inter", system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --bg: #161016; --panel: #1f1820; --ink: #ece5ed; --muted: #a596a6;
      --line: #322936; --accent: #cd7fb0; --stage: #0d090e;
      --stage-ink: #d9cfda; --stage-muted: #7c6e7e;
    }
  }
  :root[data-theme="light"] {
    --bg: #f4f1f4; --panel: #fbfafb; --ink: #241f27; --muted: #6c6172;
    --line: #e4dde5; --accent: #8d3f6e; --stage: #1b141c;
    --stage-ink: #d9cfda; --stage-muted: #8f8091;
  }
  :root[data-theme="dark"] {
    --bg: #161016; --panel: #1f1820; --ink: #ece5ed; --muted: #a596a6;
    --line: #322936; --accent: #cd7fb0; --stage: #0d090e;
    --stage-ink: #d9cfda; --stage-muted: #7c6e7e;
  }
  * { box-sizing: border-box; }
  body { margin: 0; background: var(--bg); color: var(--ink);
    font-family: var(--sans); line-height: 1.5; -webkit-font-smoothing: antialiased; }
  .wrap { max-width: 1080px; margin: 0 auto; padding: clamp(20px, 4vw, 48px); }
  header { display: flex; flex-wrap: wrap; align-items: baseline; gap: 8px 16px; margin-bottom: 4px; }
  .eyebrow { font-family: var(--mono); font-size: 12px; letter-spacing: .12em;
    text-transform: uppercase; color: var(--accent); font-weight: 600; }
  h1 { font-family: var(--mono); font-size: clamp(20px, 3.2vw, 30px); font-weight: 600;
    margin: 2px 0 0; letter-spacing: -0.01em; width: 100%; }
  .sub { color: var(--muted); font-size: 14px; margin: 4px 0 0; word-break: break-all; }
  .grid { display: grid; grid-template-columns: 1fr; gap: 22px; margin-top: 28px; }
  @media (min-width: 780px) { .grid { grid-template-columns: minmax(0,1.55fr) minmax(240px,1fr); align-items: start; } }
  .stage { background: var(--stage); border-radius: 14px; padding: clamp(14px,2.5vw,26px);
    display: flex; flex-direction: column; gap: 12px;
    box-shadow: 0 1px 0 var(--line), 0 18px 40px -24px rgba(0,0,0,.5); }
  .stage-img { width: 100%; aspect-ratio: 1/1; border-radius: 6px; display: block;
    object-fit: cover; background: #000;
    box-shadow: 0 0 0 1px rgba(255,255,255,.06), inset 0 0 60px rgba(0,0,0,.35); }
  .stage-foot { display: flex; justify-content: space-between; align-items: center; gap: 12px;
    font-family: var(--mono); font-size: 11.5px; color: var(--stage-muted); letter-spacing: .02em; }
  .scalebar { display: flex; align-items: center; gap: 8px; color: var(--stage-ink); }
  .scalebar .bar { height: 3px; width: __SCALEFRAC__%; max-width: 160px; min-width: 24px;
    background: var(--stage-ink); border-radius: 2px; }
  .panel { background: var(--panel); border: 1px solid var(--line); border-radius: 14px; padding: 18px 20px; }
  .panel h2 { font-family: var(--mono); font-size: 11px; letter-spacing: .14em; text-transform: uppercase;
    color: var(--muted); font-weight: 600; margin: 0 0 14px; }
  dl { margin: 0; display: grid; grid-template-columns: auto 1fr; gap: 9px 16px; }
  dt { color: var(--muted); font-size: 13px; }
  dd { margin: 0; font-family: var(--mono); font-size: 13px; text-align: right; font-variant-numeric: tabular-nums; }
  .pill { display: inline-flex; align-items: center; gap: 6px; font-family: var(--mono); font-size: 11px;
    padding: 2px 8px; border-radius: 999px;
    background: color-mix(in srgb, var(--good) 15%, transparent); color: var(--good); font-weight: 600; }
  .pill::before { content:""; width: 6px; height: 6px; border-radius: 999px; background: var(--good); }
  .divider { height: 1px; background: var(--line); margin: 18px 0; }
  .loc { position: relative; border-radius: 8px; overflow: hidden; border: 1px solid var(--line); }
  .loc img { display: block; width: 100%; }
  .locbox { position: absolute; left: __BOXL__%; top: __BOXT__%; width: __BOXW__%; height: __BOXH__%;
    outline: 2px solid var(--accent);
    box-shadow: 0 0 0 1px #fff, 0 0 14px 3px color-mix(in srgb, var(--accent) 70%, transparent);
    border-radius: 2px; min-width: 7px; min-height: 7px; }
  .loc-cap { font-size: 12px; color: var(--muted); margin: 9px 2px 0; }
  .warn { color: #b06a00; font-size: 12px; margin: 10px 2px 0; font-family: var(--mono); }
  footer { margin-top: 26px; padding-top: 16px; border-top: 1px solid var(--line);
    font-family: var(--mono); font-size: 12px; color: var(--muted);
    display: flex; flex-wrap: wrap; gap: 6px 14px; justify-content: space-between; }
  code.k { color: var(--accent); }
</style>

<div class="wrap">
  <header>
    <span class="eyebrow">Whole-slide image · native-resolution tile</span>
    <h1>__STEM__</h1>
    <p class="sub">H&amp;E · Aperio SVS · <code class="k">__SRCURI__</code></p>
  </header>

  <div class="grid">
    <section class="stage">
      <img class="stage-img" src="__PATCH__"
           alt="Native-resolution H&amp;E tile from __SUBJ__ at level-0 (__ORIGX__, __ORIGY__)">
      <div class="stage-foot">
        <span>tile · __TILENOTE__</span>
        <span class="scalebar"><span class="bar"></span>__SCALELBL__</span>
      </div>
    </section>

    <aside>
      <div class="panel">
        <h2>Tile provenance</h2>
        <dl>
          <dt>Center (level 0)</dt><dd>__CTRX__, __CTRY__</dd>
          <dt>Origin (level 0)</dt><dd>__ORIGX__, __ORIGY__</dd>
          <dt>Size</dt><dd>__SIZE__</dd>
          <dt>Resolution</dt><dd>__RES__</dd>
          <dt>Magnification</dt><dd>__MAG__</dd>
        </dl>
        <div class="divider"></div>
        <h2>Source slide</h2>
        <dl>
          <dt>Level-0 dims</dt><dd>__L0DIMS__</dd>
          <dt>Pyramid levels</dt><dd>__NLEV__</dd>
          <dt>Format</dt><dd>Aperio .svs</dd>
          <dt>S3 access</dt><dd><span class="pill">granted</span></dd>
        </dl>
      </div>

      <div style="margin-top:16px" class="panel">
        <h2>Location on slide</h2>
        <div class="loc">
          <img src="__OVERVIEW__" alt="Whole-slide overview of __SUBJ__">
          <div class="locbox" title="tile location"></div>
        </div>
        <p class="loc-cap">Plum box marks the tile within the whole slide.</p>
        <p class="warn">__CLAMPNOTE__</p>
      </div>
    </aside>
  </div>

  <footer>
    <span>tifffile + Pillow · no OpenSlide required</span>
    <span>biolayer.data.wsi_patch_card</span>
  </footer>
</div>
"""


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("x", type=int, help="level-0 x coordinate")
    ap.add_argument("y", type=int, help="level-0 y coordinate")
    ap.add_argument("--size", type=int, default=1024, help="tile side in level-0 px (default 1024)")
    ap.add_argument("--anchor", choices=("center", "corner"), default="center",
                    help="treat (x, y) as the tile center (default) or its top-left corner")
    ap.add_argument("--slide", default=wt.DEFAULT_S3,
                    help=f"s3:// URI or local path to an .svs (default: {wt.DEFAULT_S3})")
    ap.add_argument("--display-max", type=int, default=2048,
                    help="cap the embedded tile's longest side for the web view (default 2048)")
    ap.add_argument("--out", default=None, help="output .html path (default: <scratch>/<stem>__card_<x>_<y>.html)")
    ap.add_argument("--scratch", default=wt.SCRATCH, help=f"download/output dir (default {wt.SCRATCH})")
    args = ap.parse_args(argv)

    stem = os.path.splitext(os.path.basename(args.slide))[0]
    out = args.out or os.path.join(args.scratch, f"{stem}__card_{args.x}_{args.y}.html")

    meta = build_card(args.slide, args.x, args.y, args.size, args.anchor,
                      out, args.scratch, args.display_max)

    import json
    print(json.dumps(meta, indent=2))
    print(f"\nHTML card ready: {meta['output_html']}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
