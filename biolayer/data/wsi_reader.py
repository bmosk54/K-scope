"""Format-agnostic whole-slide reader — Aperio .svs and pyramidal/generic .tiff.

    reader = open_wsi("slide.svs")   # or .tiff/.tif — dispatched by CONTENT, not extension
    reader.mpp                        # microns/pixel at level 0 (or None if unknown)
    lvl = reader.level_for_mpp(0.5)   # pick the pyramid level closest to a target mpp
    tile = reader.read_region((x, y), lvl, (224, 224))   # RGB PIL.Image

The whole point: SVS and TIFF go through ONE interface, so the tiler never branches
on format. Backends, auto-selected:
  1. OpenSlide — Aperio SVS (pyramidal TIFF) + most WSI TIFFs (Philips/Hamamatsu/
     generic tiled-pyramidal). Preferred; sniffs format by content.
  2. tifffile — fallback for TIFFs OpenSlide can't open (OME-TIFF, some BigTIFF).

Container deps (add to the tiling/embed job): `openslide-python` + `openslide-bin`
(bundles libopenslide — no apt needed), `tifffile`, `zarr`, `Pillow`, `numpy`.
"""
import numpy as np
from PIL import Image


class WSIReader:
    """Uniform interface (mirrors the OpenSlide API the tiler expects)."""

    dimensions = (0, 0)          # (width, height) at level 0
    level_count = 1
    level_dimensions = ()        # [(w, h), ...] per level, largest first
    level_downsamples = (1.0,)   # downsample factor of each level vs level 0
    mpp = None                   # microns/pixel at level 0, or None if unknown

    def read_region(self, location, level, size):  # -> RGB PIL.Image
        raise NotImplementedError

    def close(self):
        pass

    # ---- shared helpers, format-independent --------------------------------
    def level_for_downsample(self, downsample: float) -> int:
        """Largest level whose downsample does not exceed the target."""
        best, bd = 0, 0.0
        for i, d in enumerate(self.level_downsamples):
            if d <= downsample + 1e-6 and d >= bd:
                best, bd = i, d
        return best

    def level_for_mpp(self, target_mpp: float) -> int:
        """Pyramid level whose effective mpp is closest to target_mpp.

        Raises if mpp is unknown — callers must decide (assume, or read at level 0).
        """
        if self.mpp is None:
            raise ValueError("slide has no MPP metadata; pass an explicit level or "
                             "assume a magnification")
        return self.level_for_downsample(target_mpp / self.mpp)

    def mpp_at(self, level: int):
        return None if self.mpp is None else self.mpp * self.level_downsamples[level]

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


# ---------------------------------------------------------------------------
# Backend 1: OpenSlide (SVS + most WSI TIFFs)
# ---------------------------------------------------------------------------
class _OpenSlideWSI(WSIReader):
    def __init__(self, os_obj):
        self._os = os_obj
        self.dimensions = os_obj.dimensions
        self.level_count = os_obj.level_count
        self.level_dimensions = os_obj.level_dimensions
        self.level_downsamples = os_obj.level_downsamples
        self.mpp = self._read_mpp(os_obj)

    @staticmethod
    def _read_mpp(o):
        p = o.properties
        for key in ("openslide.mpp-x", "aperio.MPP"):
            if p.get(key):
                try:
                    return float(p[key])
                except ValueError:
                    pass
        # Fall back to TIFF resolution tags (ResolutionUnit 1=none, 2=inch, 3=cm).
        try:
            xres = float(p["tiff.XResolution"])
            unit = p.get("tiff.ResolutionUnit", "").lower()
            if xres > 0 and unit in ("centimeter", "cm", "3"):
                return 10_000.0 / xres          # px/cm -> µm/px
            if xres > 0 and unit in ("inch", "2"):
                return 25_400.0 / xres          # px/inch -> µm/px
        except (KeyError, ValueError):
            pass
        return None

    def read_region(self, location, level, size):
        return self._os.read_region(location, level, size).convert("RGB")

    def close(self):
        self._os.close()


# ---------------------------------------------------------------------------
# Backend 2: tifffile fallback (OME-TIFF / exotic TIFF OpenSlide can't open)
# ---------------------------------------------------------------------------
class _TiffWSI(WSIReader):
    def __init__(self, path):
        import tifffile
        import zarr

        self._tif = tifffile.TiffFile(path)
        series = self._tif.series[0]
        # Pyramidal series expose sub-resolution levels; else a single level.
        store = series.aszarr()
        self._z = zarr.open(store, mode="r")
        if isinstance(self._z, zarr.hierarchy.Group):          # multi-level pyramid
            self._levels = [self._z[str(k)] for k in sorted(self._z.array_keys(), key=int)]
        else:                                                  # single array
            self._levels = [self._z]

        h0, w0 = self._levels[0].shape[:2]
        self.dimensions = (w0, h0)
        self.level_dimensions = tuple((lv.shape[1], lv.shape[0]) for lv in self._levels)
        self.level_count = len(self._levels)
        self.level_downsamples = tuple(w0 / lv.shape[1] for lv in self._levels)
        self.mpp = self._read_mpp(series)

    @staticmethod
    def _read_mpp(series):
        # OME metadata PhysicalSizeX, else TIFF XResolution/ResolutionUnit.
        try:
            page = series.pages[0]
            tags = page.tags
            if "XResolution" in tags and "ResolutionUnit" in tags:
                num, den = tags["XResolution"].value
                xres = num / den
                unit = tags["ResolutionUnit"].value  # 2=inch, 3=cm
                if xres > 0 and int(unit) == 3:
                    return 10_000.0 / xres
                if xres > 0 and int(unit) == 2:
                    return 25_400.0 / xres
        except Exception:
            pass
        return None

    def read_region(self, location, level, size):
        x0, y0 = location                       # level-0 coords, per OpenSlide contract
        ds = self.level_downsamples[level]
        lx, ly = int(x0 / ds), int(y0 / ds)
        w, h = size
        arr = self._levels[level][ly:ly + h, lx:lx + w]
        arr = np.asarray(arr)
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        return Image.fromarray(arr[..., :3].astype("uint8"), "RGB")

    def close(self):
        self._tif.close()


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------
def open_wsi(path: str) -> WSIReader:
    """Open an SVS or TIFF WSI, format detected by content. Prefers OpenSlide."""
    try:
        import openslide
        try:
            return _OpenSlideWSI(openslide.OpenSlide(path))
        except openslide.OpenSlideUnsupportedFormatError:
            pass  # e.g. an OME-TIFF OpenSlide won't take -> fall through
    except ImportError:
        pass
    return _TiffWSI(path)
