"""
Binary Fracture Predictor Component
===================================
Produces calibrated fracture probability, binary status prediction,
and prediction confidence using the exact validation threshold (0.50).
"""

import os
from typing import Union, Dict, Any, Optional, Tuple
from PIL import Image
import torch
import numpy as np

from src.models import MultiTaskModel
from .model_registry import ModelRegistry
from .preprocessing import preprocess_for_classifier

class FracturePredictor:
    """
    Dedicated predictor for binary fracture presence/absence.
    """
    def __init__(self, checkpoint_path: Optional[str] = None, model: Optional[torch.nn.Module] = None,
                 threshold: Optional[float] = None, device: Optional[torch.device] = None):
        self.device = device or ModelRegistry.get_device()
        self.threshold = threshold if threshold is not None else ModelRegistry.FRACTURE_THRESHOLD
        
        if model is not None:
            self.model = model
        else:
            ckpt_path = checkpoint_path or ModelRegistry.get_fracture_path() or ModelRegistry.get_classifier_path()
            if not os.path.exists(ckpt_path):
                raise FileNotFoundError(f"Classifier checkpoint not found: {ckpt_path}")
            checkpoint = torch.load(ckpt_path, map_location=self.device, weights_only=False)
            
            if 'model_state_dict' in checkpoint:
                from src.models import DedicatedFractureClassifier
                self.model = DedicatedFractureClassifier(pretrained=False)
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                backbone = checkpoint.get('backbone', ModelRegistry.CLASSIFIER_BACKBONE)
                num_regions = len(checkpoint.get('region_to_idx', ModelRegistry.REGION_TO_IDX))
                self.model = MultiTaskModel(backbone=backbone, num_regions=num_regions, pretrained=False)
                self.model.load_state_dict(checkpoint['state_dict'])
            
        self.model.to(self.device)
        self.model.eval()

    def predict(
        self,
        image_input: Union[str, Image.Image],
        target_size: Optional[Tuple[int, int]] = None
    ) -> Dict[str, Any]:
        """
        Processes one image and returns fracture status, probability, and confidence.
        
        Returns:
        --------
        dict:
            - fracture: bool (True if prob >= threshold)
            - fracture_label: str ('Positive' or 'Negative')
            - fracture_probability: float [0.0, 1.0]
            - confidence: float [0.0, 1.0] (probability towards predicted class)
        """
        tensor = preprocess_for_classifier(image_input, target_size=target_size).to(self.device)
        
        with torch.no_grad():
            out = self.model(tensor)
            if isinstance(out, tuple):
                _, fracture_logits = out
            else:
                fracture_logits = out
            prob = float(torch.sigmoid(fracture_logits).view(-1)[0].cpu().item())
            
        is_fracture = bool(prob >= self.threshold)
        label = "Positive" if is_fracture else "Negative"
        confidence = prob if is_fracture else (1.0 - prob)
        
        return {
            'fracture': is_fracture,
            'fracture_label': label,
            'fracture_probability': round(prob, 4),
            'confidence': round(confidence, 4)
        }
