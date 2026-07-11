"""Shared H-optimus-0 model loading, preprocessing, and inference.

Imported by both server.py (HTTP) and mcp_server.py (MCP tool) so the model
is defined and loaded identically in both places.
"""

import base64
import io
import os

import timm
import torch
from PIL import Image
from torchvision import transforms

TILE_SIZE = 224
EMBED_DIM = 1536

_MEAN = (0.707223, 0.578729, 0.703617)
_STD = (0.211883, 0.230117, 0.177517)

_preprocess = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=_MEAN, std=_STD),
    ]
)


class ModelNotLoadedError(RuntimeError):
    pass


class InvalidTileError(ValueError):
    pass


class HOptimusModel:
    """Loads H-optimus-0 once and serves feature extraction."""

    def __init__(self) -> None:
        self.model: torch.nn.Module | None = None
        self.device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def load(self) -> None:
        hf_token = os.environ.get("HF_TOKEN")
        if not hf_token:
            raise RuntimeError("HF_TOKEN env var is required to load the gated H-optimus-0 repo")

        import huggingface_hub

        huggingface_hub.login(token=hf_token)

        model = timm.create_model(
            "hf-hub:bioptimus/H-optimus-0",
            pretrained=True,
            init_values=1e-5,
            dynamic_img_size=False,
        )
        model.eval()
        model.to(self.device)
        self.model = model

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    def _decode_tile(self, raw: bytes) -> Image.Image:
        try:
            image = Image.open(io.BytesIO(raw)).convert("RGB")
        except Exception as exc:
            raise InvalidTileError(f"could not decode image: {exc}") from exc

        if image.size != (TILE_SIZE, TILE_SIZE):
            raise InvalidTileError(
                f"expected a {TILE_SIZE}x{TILE_SIZE} tile, got {image.size[0]}x{image.size[1]}. "
                "This server does not resample MPP; resize/crop tiles to 224x224 "
                "at ~0.5 microns/pixel before sending."
            )
        return image

    def decode_base64_tile(self, b64: str) -> Image.Image:
        try:
            raw = base64.b64decode(b64)
        except Exception as exc:
            raise InvalidTileError(f"invalid base64: {exc}") from exc
        return self._decode_tile(raw)

    def decode_bytes_tile(self, raw: bytes) -> Image.Image:
        return self._decode_tile(raw)

    @torch.inference_mode()
    def embed_batch(self, images: list[Image.Image]) -> list[list[float]]:
        if not self.is_loaded:
            raise ModelNotLoadedError("model is not loaded")

        batch = torch.stack([_preprocess(img) for img in images]).to(self.device)

        autocast_enabled = self.device == "cuda"
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=autocast_enabled):
            features = self.model(batch)

        return features.float().cpu().tolist()

    def embed_one(self, image: Image.Image) -> list[float]:
        return self.embed_batch([image])[0]


_model_singleton: HOptimusModel | None = None


def get_model() -> HOptimusModel:
    global _model_singleton
    if _model_singleton is None:
        _model_singleton = HOptimusModel()
    return _model_singleton
