"""
Centralized Inference Model Registry & Configuration
=====================================================
Contains all verified model paths, architectures, label mappings,
and preprocessing parameters directly extracted from validated checkpoints.
"""

import os
from typing import Dict, Tuple, List, Optional
import torch

def _resolve_classifier_path() -> str:
    candidates = [
        os.environ.get("CLASSIFIER_CHECKPOINT"),
        os.path.join("models", "checkpoints", "best_model.pt"),
        "best_model.pt",
        os.path.join("models", "best_model.pt")
    ]
    for p in candidates:
        if p and os.path.exists(p):
            return p
    return os.environ.get("CLASSIFIER_CHECKPOINT", os.path.join("models", "checkpoints", "best_model.pt"))

def _resolve_detector_path() -> str:
    candidates = [
        os.environ.get("DETECTOR_CHECKPOINT"),
        os.path.join("models", "checkpoints", "best_detector.pt"),
        "best_detector.pt",
        os.path.join("models", "best_detector.pt")
    ]
    for p in candidates:
        if p and os.path.exists(p):
            return p
    return os.environ.get("DETECTOR_CHECKPOINT", os.path.join("models", "checkpoints", "best_detector.pt"))

class ModelRegistry:
    # --- Multi-Task Classifier Configuration ---
    CLASSIFIER_BACKBONE: str = "resnet50"
    CLASSIFIER_IMAGE_SIZE: Tuple[int, int] = (224, 224)
    CLASSIFIER_NORMALIZATION_MEAN: List[float] = [0.5, 0.5, 0.5]
    CLASSIFIER_NORMALIZATION_STD: List[float] = [0.5, 0.5, 0.5]

    # Anatomical Region Mappings (verified from best_model.pt)
    REGION_TO_IDX: Dict[str, int] = {
        'Arm': 0,
        'Foot': 1,
        'Hand': 2,
        'Lower leg': 3,
        'Thigh': 4,
        'pelvis': 5,
        'wrist': 6
    }
    IDX_TO_REGION: Dict[int, str] = {idx: reg for reg, idx in REGION_TO_IDX.items()}
    NUM_REGIONS: int = len(REGION_TO_IDX)

    # Uncertainty handling threshold for anatomical region classification
    REGION_UNCERTAINTY_THRESHOLD: float = 0.40

    # Binary Fracture Mappings (verified from training convention)
    FRACTURE_TO_IDX: Dict[str, int] = {'Negative': 0, 'Positive': 1}
    IDX_TO_FRACTURE: Dict[int, str] = {0: 'Negative', 1: 'Positive'}
    FRACTURE_THRESHOLD: float = 0.50 # Exact threshold established during validation

    # --- Fracture Localization Detector Configuration ---
    DETECTOR_ARCHITECTURE: str = "fasterrcnn_mobilenet_v3_large_fpn"
    DETECTOR_TARGET_SIZE: Tuple[int, int] = (384, 384)
    DETECTOR_NUM_CLASSES: int = 2 # 0: background, 1: fracture
    DETECTOR_CONFIDENCE_THRESHOLD: float = 0.10 # Calibrated operating threshold to capture subtle fracture lesions

    @classmethod
    def get_classifier_path(cls) -> str:
        return _resolve_classifier_path()

    @classmethod
    def get_detector_path(cls) -> str:
        return _resolve_detector_path()

    # Backwards-compatible attributes
    @property
    def CLASSIFIER_CHECKPOINT_PATH(self) -> str:
        return _resolve_classifier_path()

    @property
    def DETECTOR_CHECKPOINT_PATH(self) -> str:
        return _resolve_detector_path()

    @classmethod
    def get_device(cls) -> torch.device:
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    @classmethod
    def verify_checkpoints(cls) -> Dict[str, bool]:
        """Checks if checkpoint files exist on disk."""
        return {
            'classifier': os.path.exists(cls.get_classifier_path()),
            'detector': os.path.exists(cls.get_detector_path())
        }

# Module-level alias for backward compatibility
CLASSIFIER_CHECKPOINT_PATH = _resolve_classifier_path()
DETECTOR_CHECKPOINT_PATH = _resolve_detector_path()
ModelRegistry.CLASSIFIER_CHECKPOINT_PATH = _resolve_classifier_path()
ModelRegistry.DETECTOR_CHECKPOINT_PATH = _resolve_detector_path()
