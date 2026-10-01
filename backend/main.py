"""
FastAPI Bone Fracture Diagnostic Backend
=========================================
Production REST API serving the deep-learning inference pipeline.

Endpoints:
- POST /predict: Processes uploaded radiograph and returns structured diagnostic prediction
- GET /health: Health probe endpoint
- GET /visualizations/{filename}: Securely serves generated visual diagnostic overlays
"""

import os
import sys
import logging
from contextlib import asynccontextmanager
from typing import Optional

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fastapi import FastAPI, UploadFile, File, HTTPException, status
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware

from backend.schemas import HealthResponse, PredictionResponse, ErrorResponse
from backend.services.inference_service import InferenceService, ImageValidationError, InferenceExecutionError

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("bonefracture.api")

VISUALIZATIONS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "static", "visualizations"))
os.makedirs(VISUALIZATIONS_DIR, exist_ok=True)

DOWNLOADS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "static", "downloads"))
os.makedirs(DOWNLOADS_DIR, exist_ok=True)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load models once into memory
    logger.info("Starting Bone Fracture API service...")
    try:
        InferenceService.get_instance(vis_output_dir=VISUALIZATIONS_DIR)
        logger.info("Models loaded and inference pipeline ready.")
    except Exception as e:
        logger.critical(f"Failed to initialize models during startup: {e}")
    yield
    # Shutdown
    logger.info("Shutting down Bone Fracture API service.")

app = FastAPI(
    title="Bone Fracture Image Captioning & Detection API",
    description="Production REST API providing multi-task anatomical classification, fracture detection, lesion localization, and factual clinical captioning.",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for future frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """
    Liveness and readiness probe.
    """
    return HealthResponse(status="ok")

@app.post(
    "/predict",
    response_model=PredictionResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid, corrupted, or unsupported image file"},
        500: {"model": ErrorResponse, "description": "Internal inference execution error"},
        503: {"model": ErrorResponse, "description": "Model service temporarily unavailable"}
    },
    tags=["Inference"]
)
async def predict_fracture(image: UploadFile = File(..., description="Uploaded bone X-ray image file")):
    """
    Accepts an uploaded X-ray image (multipart/form-data) and returns a structured prediction:
    - anatomical region & confidence
    - binary fracture status & probability
    - lesion localization bounding boxes when supported
    - disciplined factual natural-language caption
    - relative URL to retrieve generated visualization
    """
    # 1. Basic Content Type / Filename Validation
    filename = image.filename or "unknown.jpg"
    content_type = image.content_type or ""
    
    # Check allowed image extensions
    allowed_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
    ext = os.path.splitext(filename)[1].lower()
    if ext and ext not in allowed_exts and not content_type.startswith("image/"):
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"success": False, "error": f"Unsupported file format '{ext}'. Please upload a standard image file (PNG, JPG, BMP)."}
        )
        
    # 2. Read bytes into memory
    try:
        file_bytes = await image.read()
    except Exception:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"success": False, "error": "Unable to read uploaded file payload."}
        )
        
    # 3. Get inference service singleton
    try:
        service = InferenceService.get_instance(vis_output_dir=VISUALIZATIONS_DIR)
    except Exception:
        logger.error("Inference service singleton could not be obtained.")
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"success": False, "error": "Model inference service is currently initializing or unavailable."}
        )
        
    # 4. Process image and run inference
    try:
        prediction = service.process_image(file_bytes=file_bytes, original_filename=filename)
        return PredictionResponse(**prediction)
    except ImageValidationError as e:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"success": False, "error": str(e)}
        )
    except InferenceExecutionError:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"success": False, "error": "Inference processing failed. Please check image quality and try again."}
        )
    except Exception as e:
        logger.error(f"Unexpected error in /predict: {e}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"success": False, "error": "An unexpected server error occurred during analysis."}
        )

@app.get("/visualizations/{filename}", tags=["Visualizations"])
async def get_visualization(filename: str):
    """
    Safely serves an annotated diagnostic image overlay.
    Guards against path traversal attacks.
    """
    # Prevent path traversal
    clean_filename = os.path.basename(filename)
    if clean_filename != filename:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid visualization identifier.")
        
    filepath = os.path.join(VISUALIZATIONS_DIR, clean_filename)
    if not os.path.exists(filepath):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Visualization image not found or expired.")
        
    return FileResponse(filepath, media_type="image/png")
    
@app.get("/downloads/{filename}", tags=["Downloads"])
async def download_file(filename: str):
    """
    Serves downloadable architectural specifications, diagrams, PDFs, and Word documents.
    Protected against path traversal attacks.
    """
    clean_filename = os.path.basename(filename)
    if clean_filename != filename:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid file identifier.")
        
    filepath = os.path.join(DOWNLOADS_DIR, clean_filename)
    if not os.path.exists(filepath):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Requested document not found.")
        
    media_types = {
        ".pdf": "application/pdf",
        ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ".png": "image/png"
    }
    ext = os.path.splitext(clean_filename)[1].lower()
    media_type = media_types.get(ext, "application/octet-stream")
    
    return FileResponse(filepath, media_type=media_type, filename=clean_filename)
