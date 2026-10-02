"""
Pydantic Schemas for FastAPI Bone Fracture Service
===================================================
Defines request and response data contracts strictly adhering
to required machine-readable JSON formats.
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field

class HealthResponse(BaseModel):
    status: str = "ok"

class BoundingBoxInfo(BaseModel):
    box: List[float] = Field(..., description="[x1, y1, x2, y2] pixel coordinates in native resolution")
    box_2d: Optional[List[float]] = None
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence score")

class PredictionResponse(BaseModel):
    success: bool = True
    anatomical_region: str = Field(..., description="Predicted anatomical site/bone")
    anatomical_confidence: float = Field(..., ge=0.0, le=1.0, description="Softmax confidence score for region")
    fracture: bool = Field(..., description="Binary fracture presence status")
    fracture_confidence: float = Field(..., ge=0.0, le=1.0, description="Sigmoid probability of fracture")
    localization_available: bool = Field(..., description="Whether spatial lesion coordinates are supported")
    localization: Optional[List[Dict[str, Any]]] = Field(None, description="List of detected lesion bounding boxes or null")
    caption: str = Field(..., description="Disciplined factual natural-language caption")
    visualization_url: Optional[str] = Field(None, description="Safe relative URL to download annotated image")
    top_predictions: Optional[List[Dict[str, Any]]] = Field(None, description="Top-3 predicted anatomical regions with confidences")
    input_width: Optional[int] = Field(None, description="Native width of input image before preprocessing")
    input_height: Optional[int] = Field(None, description="Native height of input image before preprocessing")
    routing_resolution: Optional[str] = Field(None, description="Routing resolution ('224' or '448')")
    selected_model: Optional[str] = Field(None, description="Selected checkpoint filename")

class ErrorResponse(BaseModel):
    success: bool = False
    error: str = Field(..., description="Client-safe error description")
