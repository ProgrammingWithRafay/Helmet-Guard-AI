import base64
import logging

import cv2
import numpy as np
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.concurrency import run_in_threadpool

from app.schemas import DetectionResponse, FrameRequest, ErrorResponse
from app.inference import run_inference, load_model, ModelNotAvailableError

logger = logging.getLogger("helmetguard")

# Reject uploads larger than this to avoid memory exhaustion (DoS).
MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the model once at startup so the first request isn't slow and the
    # (thread-safe) singleton is warm before we serve traffic. If no validated
    # model exists, load_model() raises -- we log a clear message and re-raise so
    # the app REFUSES TO START rather than serving a deprecated/unvalidated model.
    logger.info("Warming up detection model...")
    try:
        load_model()
    except ModelNotAvailableError as e:
        logger.error("Startup aborted: %s", e)
        raise
    logger.info("Model ready.")
    yield


app = FastAPI(title="HelmetGuard AI", lifespan=lifespan)

# CORS: allow only the known frontend origin(s). A wildcard "*" cannot be
# combined with allow_credentials=True per the CORS spec, so we list origins
# explicitly. Override via the ALLOWED_ORIGINS env var (comma-separated).
import os

_default_origins = "http://localhost:3000,http://127.0.0.1:3000"
ALLOWED_ORIGINS = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", _default_origins).split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _error(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": {"code": code, "message": message}})


@app.post("/api/v1/detect/image", response_model=DetectionResponse, responses={400: {"model": ErrorResponse}})
async def detect_image(
    file: UploadFile = File(...),
    confidence_threshold: float = Form(0.25),
):
    # content_type may be None for some clients -- guard before .startswith().
    if not (file.content_type or "").startswith("image/"):
        return _error(400, "INVALID_FILE_TYPE", "Only images are supported.")

    contents = await file.read()
    if not contents:
        return _error(400, "EMPTY_FILE", "The uploaded file is empty or corrupted.")

    if len(contents) > MAX_UPLOAD_BYTES:
        return _error(
            413, "FILE_TOO_LARGE",
            f"File exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit.",
        )

    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        return _error(400, "DECODE_FAILED", "Could not decode image.")

    try:
        # Inference is CPU-bound and blocking -- run it off the event loop so
        # concurrent requests (e.g. the webcam frame stream) aren't stalled.
        return await run_in_threadpool(run_inference, image, confidence_threshold)
    except Exception:
        logger.exception("Inference failed for uploaded image")
        return _error(500, "INFERENCE_FAILED", "Inference failed while processing the image.")


@app.post("/api/v1/detect/frame", response_model=DetectionResponse, responses={400: {"model": ErrorResponse}})
async def detect_frame(request: FrameRequest):
    # --- Decode stage: any failure here is a 400 (bad input) ---
    try:
        raw = request.frame_base64
        encoded = raw.split(",", 1)[1] if "," in raw else raw
        decoded = base64.b64decode(encoded)
        nparr = np.frombuffer(decoded, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except Exception:
        logger.warning("Failed to decode base64 frame", exc_info=True)
        return _error(400, "NO_FRAME_DATA", "Invalid base64 frame data.")

    if image is None:
        return _error(400, "NO_FRAME_DATA", "Could not decode frame image.")

    if image.nbytes > MAX_UPLOAD_BYTES:
        return _error(413, "FILE_TOO_LARGE", "Decoded frame exceeds the size limit.")

    # --- Inference stage: failures here are a real 500 (server error) ---
    try:
        return await run_in_threadpool(run_inference, image, request.confidence_threshold)
    except Exception:
        logger.exception("Inference failed for frame")
        return _error(500, "INFERENCE_FAILED", "Inference failed while processing the frame.")


@app.get("/api/v1/health")
async def health():
    return {"status": "ok"}
