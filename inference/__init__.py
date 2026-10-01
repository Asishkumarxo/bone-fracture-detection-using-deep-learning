"""
Production Bone Fracture Inference Package
==========================================
Modular components for anatomical classification, fracture detection,
lesion localization, factual captioning, and diagnostic visualization.
"""

from .model_registry import ModelRegistry
from .preprocessing import preprocess_for_classifier, preprocess_for_detector
from .region_predictor import RegionPredictor
from .fracture_predictor import FracturePredictor
from .localization_predictor import LocalizationPredictor
from .caption_generator import generate_caption
from .visualize import create_annotated_visualization

__all__ = [
    'ModelRegistry',
    'preprocess_for_classifier',
    'preprocess_for_detector',
    'RegionPredictor',
    'FracturePredictor',
    'LocalizationPredictor',
    'generate_caption',
    'create_annotated_visualization'
]
