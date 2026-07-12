"""Build a real multi-tile `live_ctx` for the demo's LIVE source-intervention necessity.

The live per-slide necessity (biolayer.causal.intervene.live_necessity) derives each
claim's concept axis from THIS run's hidden states (diff-of-means on the slide's own
pos/neg tiles) and watches the readout-positive tiles. So the demo "slide" must be a
multi-class REGION carrying tiles of every tissue class the claims reference — not a
single tile. We sample a balanced slide + a DISJOINT reference set (for the non-circular
live-fit readout probe) straight from the cached NCT-CRC-HE dataset.

    from dashboard.live_slide import build_live_ctx
    ctx = build_live_ctx(per_class=10)          # warm encoder + tiles
    card = bridge.build_card(live_ctx=ctx)      # genuine live necessity per claim
"""
import io

import numpy as np
from PIL import Image

from biolayer import config
from biolayer.causal import live as _live

# Tissue classes the demo's tissue claims certify on (BACK = background, excluded — it is
# not a tissue concept and no claim derives an axis against it).
_SLIDE_CLASSES = ("ADI", "DEB", "LYM", "MUC", "MUS", "NORM", "STR", "TUM")


def build_live_ctx(per_class=10, n_null=12, model_key="phikon_v2", split="train",
                   seed=0, encoder=None, local_files_only=True):
    """Return a live_ctx dict: a balanced multi-class slide + disjoint reference set +
    a warm LiveEncoder, ready to pass to certify_answer/build_card.

    per_class tiles of each tissue class go to the SLIDE and another per_class (disjoint)
    to the REFERENCE set, so every tissue claim has >=2 pos and >=2 neg tiles to derive
    its axis and a clean readout probe. dataset_slug pins the label space so the
    certify guard only runs live on matching (NCT tissue) claims.
    """
    name_to_idx = {c: i for i, c in enumerate(config.CLASS_NAMES)}
    want = {name_to_idx[c]: 2 * per_class for c in _SLIDE_CLASSES}
    got = {k: [] for k in want}

    from datasets import Image as HFImage, load_dataset
    ds = load_dataset(config.DATASET_ID, split=config.SPLITS[split], streaming=True)
    ds = ds.cast_column(config.IMAGE_COLUMN, HFImage(decode=False))
    ds = ds.shuffle(seed=seed, buffer_size=5000)
    for row in ds:
        lbl = int(row[config.LABEL_COLUMN])
        if lbl in got and len(got[lbl]) < want[lbl]:
            got[lbl].append(Image.open(io.BytesIO(row[config.IMAGE_COLUMN]["bytes"])).convert("RGB"))
        if all(len(got[k]) >= want[k] for k in want):
            break

    slide_imgs, slide_lbls, ref_imgs, ref_lbls = [], [], [], []
    for k, imgs in got.items():
        half = len(imgs) // 2
        slide_imgs += imgs[:half];  slide_lbls += [k] * half
        ref_imgs += imgs[half:];    ref_lbls += [k] * (len(imgs) - half)

    enc = encoder or _live.make_live_encoder(model_key, local_files_only=local_files_only)
    return {
        "images": slide_imgs, "image_labels": np.array(slide_lbls),
        "ref_images": ref_imgs, "ref_labels": np.array(ref_lbls),
        "encoder": enc, "n_null": n_null,
        "dataset_slug": config.DATASET_SLUG,          # guard: live only on NCT tissue claims
        "n_tiles": len(slide_imgs),
        "classes_present": sorted({config.CLASS_NAMES[k] for k in got if got[k]}),
    }
