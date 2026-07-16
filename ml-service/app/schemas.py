from pydantic import BaseModel, Field
from typing import List, Optional

class BBox(BaseModel):
    x_min: int
    y_min: int
    x_max: int
    y_max: int

class Detection(BaseModel):
    id: str
    class_name: str = Field(alias="class")
    confidence: float
    bbox: BBox
    
    class Config:
        populate_by_name = True

class DetectionSummary(BaseModel):
    total_riders: int
    compliant: int
    violations: int

class DetectionResponse(BaseModel):
    request_id: str
    image_width: int
    image_height: int
    inference_time_ms: int
    detections: List[Detection]
    summary: DetectionSummary

class FrameRequest(BaseModel):
    frame_base64: str
    confidence_threshold: float = 0.5

class ErrorDetail(BaseModel):
    code: str
    message: str

class ErrorResponse(BaseModel):
    error: ErrorDetail
