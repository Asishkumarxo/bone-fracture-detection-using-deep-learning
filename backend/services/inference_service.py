"""
Backend Inference Service
=========================
Coordinates singleton model loading, memory-safe image validation,
execution through the validated inference pipeline, and safe visualization generation.
"""

import os
import sys
import io
import uuid
import logging
from typing import Dict, Any, Optional
from PIL import Image
import torch

# Ensure workspace root is accessible
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from inference.model_registry import ModelRegistry
from predict import BoneFractureInferencePipeline
from inference.visualize import create_annotated_visualization

logger = logging.getLogger("bonefracture.inference")

class ImageValidationError(Exception):
    """Raised when an uploaded file cannot be parsed as a valid radiograph image."""
    pass

class InferenceExecutionError(Exception):
    """Raised when the deep learning inference pipeline encounters an unrecoverable error."""
    pass

class InferenceService:
    _instance: Optional['InferenceService'] = None

    def __init__(self, vis_output_dir: str = "backend/static/visualizations"):
        self.vis_output_dir = vis_output_dir
        os.makedirs(self.vis_output_dir, exist_ok=True)
        self.device = ModelRegistry.get_device()
        self.pipeline: Optional[BoneFractureInferencePipeline] = None
        self._initialize_pipeline()

    def _initialize_pipeline(self):
        try:
            logger.info("Initializing BoneFractureInferencePipeline (loading models once)...")
            self.pipeline = BoneFractureInferencePipeline(device=self.device)
            logger.info("Pipeline loaded successfully and cached in memory.")
        except Exception as e:
            logger.error(f"Critical error loading model checkpoints: {e}")
            self.pipeline = None
            raise RuntimeError("Model checkpoint initialization failed") from e

    @classmethod
    def get_instance(cls, vis_output_dir: str = "backend/static/visualizations") -> 'InferenceService':
        if cls._instance is None:
            cls._instance = cls(vis_output_dir=vis_output_dir)
        return cls._instance

    def process_image(self, file_bytes: bytes, original_filename: str) -> Dict[str, Any]:
        """
        Validates uploaded bytes in-memory and executes the frozen inference pipeline.
        
        Parameters:
        -----------
        file_bytes : bytes
            Raw binary content of the uploaded file.
        original_filename : str
            User-provided filename for safe identification.
            
        Returns:
        --------
        dict:
            Safe prediction dictionary conforming strictly to API contract.
        """
        if not file_bytes or len(file_bytes) == 0:
            raise ImageValidationError("Uploaded image file is empty.")
            
        # 1. Validate image format in-memory
        try:
            bio = io.BytesIO(file_bytes)
            with Image.open(bio) as raw_im:
                raw_im.verify()
            
            # Re-read verified image buffer
            bio.seek(0)
            pil_img = Image.open(bio)
            pil_img.load()
        except Exception as e:
            raise ImageValidationError("Uploaded file is corrupted or not a valid image format.")
            
        if self.pipeline is None:
            raise InferenceExecutionError("Inference service model is unavailable.")
            
        # 2. Inspect original dimensions BEFORE preprocessing and perform resolution routing
        orig_w, orig_h = pil_img.size
        routing_info = ModelRegistry.route_image(orig_w, orig_h)
        selected_model_name = routing_info["selected_model"]
        routing_res = routing_info["routing_resolution"]
        target_size = routing_info["target_size"]
        
        logger.info(
            f"Resolution routing: {orig_w}x{orig_h} (min={min(orig_w, orig_h)}) "
            f"-> {routing_res}x{routing_res} ({selected_model_name})"
        )

        # Select appropriate model branch
        if selected_model_name == ModelRegistry.EXP3_CHECKPOINT_FILENAME and self.pipeline.exp3_model is not None:
            reg_predictor = self.pipeline.exp3_region_predictor
            frac_predictor = self.pipeline.exp3_fracture_predictor
        else:
            reg_predictor = self.pipeline.region_predictor
            frac_predictor = self.pipeline.fracture_predictor

        # 3. Execute inference pipeline with routed resolution
        try:
            # Predict anatomical region
            reg_res = reg_predictor.predict(pil_img, target_size=target_size)
            anatomical_region = reg_res['predicted_region']
            anatomical_confidence = reg_res['confidence']
            
            # Predict fracture status (decision threshold = 0.50)
            frac_res = frac_predictor.predict(pil_img, target_size=target_size)
            is_fracture = frac_res['fracture']
            fracture_confidence = frac_res['fracture_probability']
            
            # Optional localization
            loc_res = self.pipeline.localization_predictor.predict(pil_img, fracture_detected=is_fracture)
            localization_available = loc_res['localization_available']
            localization = loc_res['localization']
            
            # Generate factual caption
            from inference.caption_generator import generate_caption
            caption = generate_caption({
                'anatomical_region': anatomical_region,
                'fracture': is_fracture,
                'localization_available': localization_available,
                'localization': localization
            })
            
            # Compute top-3 predictions for uncertainty inspection
            dist = reg_res.get('probability_distribution', {})
            top_3 = sorted(dist.items(), key=lambda x: x[1], reverse=True)[:3]
            top_predictions = [{"region": r, "confidence": round(p, 4)} for r, p in top_3]
            
            # 4. Assemble response without exposing server filesystem paths
            result = {
                "success": True,
                "anatomical_region": anatomical_region,
                "anatomical_confidence": anatomical_confidence,
                "fracture": is_fracture,
                "fracture_confidence": fracture_confidence,
                "localization_available": localization_available,
                "localization": localization,
                "caption": caption,
                "top_predictions": top_predictions,
                "input_width": orig_w,
                "input_height": orig_h,
                "routing_resolution": routing_res,
                "selected_model": selected_model_name
            }
            
            # 5. Generate annotated visualization safely
            unique_id = uuid.uuid4().hex[:12]
            vis_filename = f"pred_{unique_id}.png"
            create_annotated_visualization(
                image_input=pil_img,
                prediction_result=result,
                output_dir=self.vis_output_dir,
                filename=vis_filename
            )
            result["visualization_url"] = f"/visualizations/{vis_filename}"
            
            return result
            
        except ImageValidationError:
            raise
        except Exception as e:
            logger.error(f"Inference execution failed: {e}", exc_info=True)
            raise InferenceExecutionError("Internal inference processing failed.")
