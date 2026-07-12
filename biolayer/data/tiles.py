"""Fetch a few labeled tiles (PIL images) for live per-input intervention.

The cached npz stores embeddings only; the live hook needs the actual images. This
pulls a small class-balanced set of tiles from the HF dataset the same way
data.extract does (streaming, decode only what we keep).
"""
import io

from PIL import Image

from .. import config


def fetch(class_codes, per_class=8, split="train", seed=0, shuffle_buffer=2000,
          dataset_id=None, class_names=None):
    """Return (images, labels) — labels are indices into `class_names` (default
    config.CLASS_NAMES). Only tiles whose class is in `class_codes` are kept."""
    from datasets import Image as HFImage, load_dataset

    class_names = class_names or config.CLASS_NAMES
    dataset_id = dataset_id or config.DATASET_ID
    want = {class_names.index(c): c for c in class_codes}
    hf_split = config.resolve_split(split)

    ds = load_dataset(dataset_id, split=hf_split, streaming=True)
    ds = ds.cast_column(config.IMAGE_COLUMN, HFImage(decode=False))
    if shuffle_buffer > 0:
        ds = ds.shuffle(seed=seed, buffer_size=shuffle_buffer)

    counts = {i: 0 for i in want}
    images, labels = [], []
    for ex in ds:
        lbl = int(ex[config.LABEL_COLUMN])
        if lbl not in want or counts[lbl] >= per_class:
            if all(c >= per_class for c in counts.values()):
                break
            continue
        raw = ex[config.IMAGE_COLUMN]
        images.append(Image.open(io.BytesIO(raw["bytes"])).convert("RGB"))
        labels.append(lbl)
        counts[lbl] += 1
    return images, labels
