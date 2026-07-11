"""FastAPI server exposing H-optimus-0 tile-level feature extraction over HTTP."""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from model import InvalidTileError, get_model

MAX_BATCH_SIZE = int(os.environ.get("MAX_BATCH_SIZE", "64"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_model().load()
    yield


app = FastAPI(title="H-optimus-0 inference server", lifespan=lifespan)


class EmbedBase64Request(BaseModel):
    image_base64: str


class EmbedBatchRequest(BaseModel):
    images_base64: list[str]


class EmbedResponse(BaseModel):
    features: list[float]


class EmbedBatchResponse(BaseModel):
    features: list[list[float]]


@app.get("/health")
def health():
    model = get_model()
    return {"status": "ok", "device": model.device, "model_loaded": model.is_loaded}


@app.post("/embed", response_model=EmbedResponse)
async def embed(file: UploadFile | None = File(default=None), body: EmbedBase64Request | None = None):
    model = get_model()

    if file is not None:
        raw = await file.read()
        try:
            image = model.decode_bytes_tile(raw)
        except InvalidTileError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    elif body is not None:
        try:
            image = model.decode_base64_tile(body.image_base64)
        except InvalidTileError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    else:
        raise HTTPException(status_code=400, detail="provide either a multipart file upload or a JSON body with image_base64")

    features = model.embed_one(image)
    return EmbedResponse(features=features)


@app.post("/embed_batch", response_model=EmbedBatchResponse)
async def embed_batch(request: EmbedBatchRequest):
    if not request.images_base64:
        raise HTTPException(status_code=400, detail="images_base64 must be a non-empty list")
    if len(request.images_base64) > MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"batch of {len(request.images_base64)} exceeds MAX_BATCH_SIZE={MAX_BATCH_SIZE}",
        )

    model = get_model()
    try:
        images = [model.decode_base64_tile(b64) for b64 in request.images_base64]
    except InvalidTileError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    features = model.embed_batch(images)
    return EmbedBatchResponse(features=features)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "8000")),
    )
