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
            .replace("__L0W__", str(l0w)).replace("__L0H__", str(l0h))
            .replace("__MPP_NUM__", repr(mpp if mpp else 0.0))
            .replace("__MPP_TXT__", mpp_txt).replace("__MAG_TXT__", mag_txt))

    os.makedirs(os.path.dirname(os.path.abspath(out_html)), exist_ok=True)
    with open(out_html, "w") as f:
        f.write(page)
    return {"output_html": out_html, "n_patches": len(out),
            "level0_dimensions": [l0w, l0h], "mpp_um_per_px": mpp, "magnification": mag,
            "bytes": os.path.getsize(out_html)}


# Self-contained, CSP-safe (no external fonts/scripts). __TOKENS__ are filled in above.
_TEMPLATE = r"""<title>__STEM__ — patch gallery</title>
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
  body { margin: 0; background: var(--bg); color: var(--ink);
    font-family: var(--sans); line-height: 1.5; -webkit-font-smoothing: antialiased; }
  .wrap { max-width: 1180px; margin: 0 auto; padding: clamp(18px, 3.5vw, 44px); }
  header { display: flex; flex-wrap: wrap; align-items: baseline; gap: 6px 16px; }
  .eyebrow { font-family: var(--mono); font-size: 12px; letter-spacing: .12em;
    text-transform: uppercase; color: var(--accent); font-weight: 600; }
  h1 { font-family: var(--mono); font-size: clamp(20px, 3.2vw, 30px); font-weight: 600;
    margin: 2px 0 0; letter-spacing: -0.01em; width: 100%; }
  .sub { color: var(--muted); font-size: 14px; margin: 4px 0 0; word-break: break-all; }

  /* three regions: thumb rail · stage · aside */
  .layout { display: grid; gap: 20px; margin-top: 26px;
    grid-template-columns: 104px minmax(0,1fr); grid-template-areas: "rail stage" "rail aside"; }
  @media (min-width: 880px) {
    .layout { grid-template-columns: 108px minmax(0,1.5fr) minmax(240px,1fr);
      grid-template-areas: "rail stage aside"; align-items: start; }
  }
  @media (max-width: 879px) {
    .layout { grid-template-columns: 1fr; grid-template-areas: "rail" "stage" "aside"; }
  }

  /* thumbnail rail */
  .rail { grid-area: rail; display: flex; flex-direction: column; gap: 10px; }
  @media (max-width: 879px) { .rail { flex-direction: row; overflow-x: auto; padding-bottom: 4px; } }
  .rail-h { font-family: var(--mono); font-size: 10px; letter-spacing: .14em; text-transform: uppercase;
    color: var(--muted); margin: 2px 0 2px 2px; }
  @media (max-width: 879px) { .rail-h { display: none; } }
  .thumb { position: relative; border: 0; padding: 0; background: none; cursor: pointer;
    border-radius: 10px; overflow: hidden; flex: 0 0 auto; width: 100%;
    outline: 2px solid transparent; outline-offset: 2px; transition: outline-color .12s, transform .12s; }
  @media (max-width: 879px) { .thumb { width: 92px; } }
  .thumb img { display: block; width: 100%; aspect-ratio: 1/1; object-fit: cover;
    border-radius: 9px; filter: saturate(.96); }
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
  .aside { grid-area: aside; display: flex; flex-direction: column; gap: 16px; }
  .panel { background: var(--panel); border: 1px solid var(--line); border-radius: 14px; padding: 16px 18px; }
  .panel h2 { font-family: var(--mono); font-size: 11px; letter-spacing: .14em; text-transform: uppercase;
    color: var(--muted); font-weight: 600; margin: 0 0 12px; }
  .desc { font-size: 13px; color: var(--ink); margin: -2px 0 14px; }
  dl { margin: 0; display: grid; grid-template-columns: auto 1fr; gap: 8px 16px; }
  dt { color: var(--muted); font-size: 13px; }
  dd { margin: 0; font-family: var(--mono); font-size: 13px; text-align: right; font-variant-numeric: tabular-nums; }

  .loc { position: relative; border-radius: 8px; overflow: hidden; border: 1px solid var(--line); }
  .loc img { display: block; width: 100%; }
  .locbox { position: absolute; border-radius: 3px; transform: translate(-1px,-1px); min-width: 15px; min-height: 15px; }
  .locbox.other { outline: 1.5px solid var(--other);
    box-shadow: 0 0 0 1px rgba(255,255,255,.5); cursor: pointer; }
  .locbox.active { outline: 2.5px solid var(--accent); z-index: 3;
    box-shadow: 0 0 0 1.5px #fff, 0 0 16px 4px color-mix(in srgb, var(--accent) 75%, transparent); }
  .loc-legend { display: flex; gap: 16px; margin: 10px 2px 0; font-size: 11.5px; color: var(--muted);
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
</style>

<div class="wrap">
  <header>
    <span class="eyebrow">Whole-slide image · patch gallery</span>
    <h1>__STEM__</h1>
    <p class="sub">H&amp;E · Aperio SVS · __MAG_TXT__ · <code class="k">__SRCURI__</code></p>
  </header>

  <div class="layout">
    <nav class="rail" aria-label="Patches">
      <div class="rail-h">Patches</div>
      <div id="thumbs" style="display:contents"></div>
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

  <footer>
    <span>tifffile + Pillow · no OpenSlide required</span>
    <span>biolayer.data.wsi_patch_gallery</span>
  </footer>
</div>

<script>
  const PATCHES = __PATCHES__;
  const MPP = __MPP_NUM__;

  const thumbsEl = document.getElementById('thumbs');
  const boxesEl  = document.getElementById('boxes');
  const stageImg = document.getElementById('stage-img');
  let active = 0;

  PATCHES.forEach((p, i) => {
    const b = document.createElement('button');
    b.className = 'thumb'; b.type = 'button';
    b.setAttribute('aria-current', i === 0 ? 'true' : 'false');
    b.innerHTML = `<img src="${p.img}" alt="${p.title}"><span class="tcap">${p.title}</span>`;
    b.addEventListener('click', () => setActive(i));
    thumbsEl.appendChild(b);
  });

  PATCHES.forEach((p, i) => {
    const d = document.createElement('div');
    d.className = 'locbox other';
    d.style.left = p.box.l + '%'; d.style.top = p.box.t + '%';
    d.style.width = p.box.w + '%'; d.style.height = p.box.h + '%';
    d.title = p.title;
    d.addEventListener('click', () => setActive(i));
    boxesEl.appendChild(d);
  });

  const thumbEls = [...thumbsEl.children];
  const boxEls = [...boxesEl.children];

  function setActive(i) {
    active = (i + PATCHES.length) % PATCHES.length;
    const p = PATCHES[active];
    thumbEls.forEach((t, k) => t.setAttribute('aria-current', k === active ? 'true' : 'false'));
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
    thumbEls[active].scrollIntoView({ block: 'nearest', inline: 'nearest' });
  }

  document.addEventListener('keydown', (e) => {
    if (e.key === 'ArrowDown' || e.key === 'ArrowRight') { setActive(active + 1); e.preventDefault(); }
    else if (e.key === 'ArrowUp' || e.key === 'ArrowLeft') { setActive(active - 1); e.preventDefault(); }
    else if (e.key >= '1' && e.key <= String(Math.min(9, PATCHES.length))) setActive(+e.key - 1);
  });

  setActive(0);
</script>
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
