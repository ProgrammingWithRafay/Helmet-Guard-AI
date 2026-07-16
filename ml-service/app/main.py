from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import base64
import cv2
import numpy as np

from app.schemas import DetectionResponse, FrameRequest, ErrorResponse
from app.inference import run_inference

app = FastAPI(title="HelmetGuard AI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/api/v1/detect/image", response_model=DetectionResponse, responses={400: {"model": ErrorResponse}})
async def detect_image(
    file: UploadFile = File(...),
    confidence_threshold: float = Form(0.5)
):
    if not file.content_type.startswith("image/"):
        return JSONResponse(
            status_code=400,
            content={"error": {"code": "INVALID_FILE_TYPE", "message": "Only images are supported."}}
        )
    
    contents = await file.read()
    if not contents:
        return JSONResponse(
            status_code=400,
            content={"error": {"code": "EMPTY_FILE", "message": "The uploaded file is empty or corrupted."}}
        )

    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if image is None:
        return JSONResponse(
            status_code=400,
            content={"error": {"code": "INFERENCE_FAILED", "message": "Could not decode image."}}
        )

    try:
        result = run_inference(image, confidence_threshold)
        return result
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": {"code": "INFERENCE_FAILED", "message": str(e)}}
        )

@app.post("/api/v1/detect/frame", response_model=DetectionResponse, responses={400: {"model": ErrorResponse}})
async def detect_frame(request: FrameRequest):
    try:
        if "," in request.frame_base64:
            header, encoded = request.frame_base64.split(",", 1)
        else:
            encoded = request.frame_base64
            
        decoded = base64.b64decode(encoded)
        nparr = np.frombuffer(decoded, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise ValueError("Failed to decode frame.")
            
        result = run_inference(image, request.confidence_threshold)
        return result
    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={"error": {"code": "NO_FRAME_DATA", "message": "Invalid base64 frame data."}}
        )

@app.get("/api/v1/health")
async def health():
    return {"status": "ok"}
