"""
Fracture Spatial Localization Predictor Component
==================================================
Runs object detection on confirmed fracture cases using Faster R-CNN,
maps bounding coordinates back to native image dimensions, and avoids
fabricating localization when unsupported.
"""

import os
from typing import Union, Dict, Any, Optional, List
from PIL import Image
import torch
import torchvision.models.detection as detection
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from .model_registry import ModelRegistry
from .preprocessing import preprocess_for_detector

class LocalizationPredictor:
    """
    Dedicated predictor for spatial lesion bounding box localization.
    """
    def __init__(self, checkpoint_path: Optional[str] = None, score_thresh: Optional[float] = None, device: Optional[torch.device] = None):
        self.device = device or ModelRegistry.get_device()
        self.score_thresh = score_thresh or ModelRegistry.DETECTOR_CONFIDENCE_THRESHOLD
        self.target_size = ModelRegistry.DETECTOR_TARGET_SIZE
        self.model = None
        self.is_available = False
        
        ckpt_path = checkpoint_path or ModelRegistry.DETECTOR_CHECKPOINT_PATH
        if os.path.exists(ckpt_path):
            try:
                ckpt = torch.load(ckpt_path, map_location=self.device)
                self.target_size = ckpt.get('target_size', self.target_size)
                
                model = detection.fasterrcnn_mobilenet_v3_large_fpn(weights=None)
                in_features = model.roi_heads.box_predictor.cls_score.in_features
                model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=ModelRegistry.DETECTOR_NUM_CLASSES)
                model.load_state_dict(ckpt['model_state'])
                model.to(self.device)
                model.eval()
                
                self.model = model
                self.is_available = True
            except Exception as e:
                print(f"Warning: Could not load detector checkpoint ({e}). Localization disabled.")
                self.is_available = False

    def predict(self, image_input: Union[str, Image.Image], fracture_detected: bool = True) -> Dict[str, Any]:
        """
        Runs localization if fracture is detected and detector is loaded.
        
        Returns:
        --------
        dict:
            - localization_available: bool
            - localization: list of dict or None
        """
        if not fracture_detected or not self.is_available or self.model is None:
            return {
                'localization_available': False,
                'localization': None
            }
            
        tensor, (orig_w, orig_h) = preprocess_for_detector(image_input)
        tensor = tensor.to(self.device)
        
        with torch.no_grad():
            preds = self.model(tensor)[0]
            boxes = preds['boxes'].cpu().numpy()
            scores = preds['scores'].cpu().numpy()
            labels = preds['labels'].cpu().numpy()
            
        valid_mask = (scores >= self.score_thresh) & (labels == 1)
        f_boxes = boxes[valid_mask]
        f_scores = scores[valid_mask]
        
        if len(f_boxes) == 0:
            return {
                'localization_available': False,
                'localization': None
            }
            
        scale_w = orig_w / float(self.target_size[0])
        scale_h = orig_h / float(self.target_size[1])
        
        detected_lesions: List[Dict[str, Any]] = []
        for box, score in zip(f_boxes, f_scores):
            x1 = round(float(box[0] * scale_w), 1)
            y1 = round(float(box[1] * scale_h), 1)
            x2 = round(float(box[2] * scale_w), 1)
            y2 = round(float(box[3] * scale_h), 1)
            detected_lesions.append({
                'box': [x1, y1, x2, y2],
                'box_2d': [x1, y1, x2, y2],
                'confidence': round(float(score), 4)
            })
            
        return {
            'localization_available': True,
            'localization': detected_lesions
        }
