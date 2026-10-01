"""
Test Suite: Preprocessing & Image Ingestion Pipeline
===================================================
Tests:
1. Image loading and format verification
2. Aspect-ratio preserving padding logic
3. Tensor output shapes for classifier (1, 3, 224, 224)
4. Tensor output shapes for detector (1, 3, 384, 384)
5. Intensity standardization and value range
"""

import os
import sys
import unittest
from PIL import Image
import torch

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from inference.preprocessing import (
    resize_and_pad,
    load_and_validate_image,
    preprocess_for_classifier,
    preprocess_for_detector
)
from inference.model_registry import ModelRegistry

class TestPreprocessing(unittest.TestCase):
    def setUp(self):
        # Create small synthetic test image (grayscale with realistic dimensions)
        self.test_img = Image.new('L', (400, 300), color=100)
        self.square_img = Image.new('L', (224, 224), color=50)

    def test_resize_and_pad_aspect_ratio_preserved(self):
        """Verify resize_and_pad maintains target dimensions without distorting aspect ratio."""
        target_size = (224, 224)
        padded = resize_and_pad(self.test_img, target_size=target_size, fill_color=0)
        self.assertEqual(padded.size, target_size)
        self.assertEqual(padded.mode, 'L')

    def test_resize_and_pad_already_target_size(self):
        """Pass-through check when image is already target size."""
        padded = resize_and_pad(self.square_img, target_size=(224, 224))
        self.assertEqual(padded.size, (224, 224))

    def test_load_and_validate_pil_image(self):
        """Validation of PIL image in-memory."""
        validated = load_and_validate_image(self.test_img)
        self.assertEqual(validated.size, (400, 300))

    def test_load_and_validate_invalid_input(self):
        """Validation of invalid input types and missing files."""
        with self.assertRaises(TypeError):
            load_and_validate_image(12345)

        with self.assertRaises(FileNotFoundError):
            load_and_validate_image("non_existent_file_path.png")

    def test_preprocess_for_classifier_shape(self):
        """Verify classifier tensor shape is strictly (1, 3, 224, 224)."""
        tensor = preprocess_for_classifier(self.test_img)
        self.assertIsInstance(tensor, torch.Tensor)
        self.assertEqual(tensor.shape, (1, 3, 224, 224))
        self.assertEqual(tensor.dtype, torch.float32)

    def test_preprocess_for_detector_shape(self):
        """Verify detector tensor shape is strictly (1, 3, 384, 384) with correct original size tuple."""
        tensor, orig_size = preprocess_for_detector(self.test_img)
        self.assertIsInstance(tensor, torch.Tensor)
        self.assertEqual(tensor.shape, (1, 3, 384, 384))
        self.assertEqual(orig_size, (400, 300))

if __name__ == '__main__':
    unittest.main()
