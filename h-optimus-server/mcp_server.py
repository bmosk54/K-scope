"""MCP server exposing H-optimus-0 tile-level feature extraction as a tool (stdio transport)."""

from mcp.server.fastmcp import FastMCP

from model import get_model

mcp = FastMCP("h-optimus-0")


@mcp.tool()
def extract_features(tile_base64: str) -> list[float]:
    """Extract a 1536-dim CLS embedding from a single 224x224 RGB histology tile.

    Args:
        tile_base64: base64-encoded image bytes (PNG/JPEG) of a 224x224 RGB tile
            at ~0.5 microns/pixel. Not resampled server-side.
    """
    model = get_model()
    if not model.is_loaded:
        model.load()

    image = model.decode_base64_tile(tile_base64)
    return model.embed_one(image)


@mcp.tool()
def extract_features_batch(tiles_base64: list[str]) -> list[list[float]]:
    """Extract 1536-dim CLS embeddings for a batch of 224x224 RGB histology tiles.

    Args:
        tiles_base64: list of base64-encoded 224x224 RGB tile images.
    """
    model = get_model()
    if not model.is_loaded:
        model.load()

    images = [model.decode_base64_tile(b64) for b64 in tiles_base64]
    return model.embed_batch(images)


if __name__ == "__main__":
    mcp.run(transport="stdio")
