"""
Anatomical Region Predictor Component
=====================================
Loads the multi-task model and produces predicted anatomical region,
confidence score, and complete class probability distribution.
"""

import os
from typing import Union, Dict, Any, Optional, Tuple
from PIL import Image
import torch
import numpy as np

from src.models import MultiTaskModel
from .model_registry import ModelRegistry
from .preprocessing import preprocess_for_classifier

class RegionPredictor:
    """
    Dedicated predictor for anatomical region classification.
    """
    def __init__(self, checkpoint_path: Optional[str] = None, model: Optional[torch.nn.Module] = None, device: Optional[torch.device] = None):
        self.device = device or ModelRegistry.get_device()
        self.idx_to_region = ModelRegistry.IDX_TO_REGION
        self.num_regions = ModelRegistry.NUM_REGIONS
        
        if model is not None:
            self.model = model
        else:
            ckpt_path = checkpoint_path or ModelRegistry.CLASSIFIER_CHECKPOINT_PATH
            if not os.path.exists(ckpt_path):
                raise FileNotFoundError(f"Classifier checkpoint not found: {ckpt_path}")
            checkpoint = torch.load(ckpt_path, map_location=self.device)
            backbone = checkpoint.get('backbone', ModelRegistry.CLASSIFIER_BACKBONE)
            
            self.model = MultiTaskModel(backbone=backbone, num_regions=self.num_regions, pretrained=False)
            self.model.load_state_dict(checkpoint['state_dict'])
            
        self.model.to(self.device)
        self.model.eval()

    def predict(
        self,
        image_input: Union[str, Image.Image],
        target_size: Optional[Tuple[int, int]] = None
    ) -> Dict[str, Any]:
        """
        Processes one image and returns predicted region, confidence, and full probability distribution.
        
        Returns:
        --------
        dict:
            - predicted_region: str
            - confidence: float
            - probability_distribution: dict {region_name: prob}
        """
        tensor = preprocess_for_classifier(image_input, target_size=target_size).to(self.device)
        
        with torch.no_grad():
            region_logits, _ = self.model(tensor)
            probs = torch.softmax(region_logits, dim=-1)[0].cpu().numpy()
            
        pred_idx = int(np.argmax(probs))
        pred_region = self.idx_to_region[pred_idx]
        conf = float(probs[pred_idx])
        
        distribution = {self.idx_to_region[i]: round(float(probs[i]), 4) for i in range(len(probs))}
        
        return {
            'predicted_region': pred_region,
            'confidence': round(conf, 4),
            'probability_distribution': distribution
        }
