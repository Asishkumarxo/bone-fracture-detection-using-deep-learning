"""
Test Suite: Model Architecture & Loading Verification
=====================================================
Tests:
1. MultiTaskModel architecture initialization (ResNet-50 backbone)
2. Dual-task head output shapes (7-dim region logits, 1-dim fracture logit)
3. Backbone layer freeze and unfreeze methods
4. Localization detector architecture instantiation
5. ModelRegistry checkpoint discovery paths
"""

import os
import sys
import unittest
import torch

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models import MultiTaskModel
from inference.model_registry import ModelRegistry

class TestModelLoading(unittest.TestCase):
    def test_multitask_model_init(self):
        """Verify MultiTaskModel instantiates without pretrained weights for offline testing."""
        model = MultiTaskModel(backbone='resnet50', num_regions=7, pretrained=False)
        self.assertIsNotNone(model)
        self.assertEqual(model.num_regions, 7)

    def test_forward_pass_output_shapes(self):
        """Verify model produces correct output dimensions for batched input."""
        model = MultiTaskModel(backbone='resnet50', num_regions=7, pretrained=False)
        model.eval()
        
        # Batch of 2 dummy images: (B, C, H, W) = (2, 3, 224, 224)
        dummy_input = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            region_logits, fracture_logits = model(dummy_input)

        self.assertEqual(region_logits.shape, (2, 7))
        self.assertEqual(fracture_logits.shape, (2,))

    def test_freeze_and_unfreeze_behavior(self):
        """Verify transfer learning parameter freezing behaves as designed."""
        model = MultiTaskModel(backbone='resnet50', num_regions=7, pretrained=False)
        
        # Freeze backbone
        model.freeze_backbone()
        for param in model.backbone.parameters():
            self.assertFalse(param.requires_grad)
        # Heads must remain trainable
        for param in model.region_head.parameters():
            self.assertTrue(param.requires_grad)
        for param in model.fracture_head.parameters():
            self.assertTrue(param.requires_grad)

        # Unfreeze all
        model.unfreeze_all()
        for param in model.parameters():
            self.assertTrue(param.requires_grad)

    def test_checkpoint_discovery_methods(self):
        """Verify dynamic discovery returns valid string paths without runtime errors."""
        clf_path = ModelRegistry.get_classifier_path()
        det_path = ModelRegistry.get_detector_path()
        self.assertIsInstance(clf_path, str)
        self.assertIsInstance(det_path, str)
        
        status = ModelRegistry.verify_checkpoints()
        self.assertIn('classifier', status)
        self.assertIn('detector', status)

if __name__ == '__main__':
    unittest.main()
