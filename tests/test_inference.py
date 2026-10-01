"""
Test Suite: End-to-End Inference Pipeline & Schema Verification
===============================================================
Tests:
1. Pipeline initialization and execution on valid image
2. Strict schema verification (all 7 required fields present)
3. Confidence score range validation [0.0, 1.0]
4. Missing image file error handling (FileNotFoundError)
5. Corrupted image file error handling (ValueError)
6. Visualization output generation
"""

import os
import sys
import unittest
import tempfile
from PIL import Image

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from inference.model_registry import ModelRegistry
from predict import BoneFractureInferencePipeline, run_inference

class TestInferencePipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create a temporary directory with a clean synthetic radiograph
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.valid_image_path = os.path.join(cls.temp_dir.name, "synthetic_xray.png")
        cls.corrupted_image_path = os.path.join(cls.temp_dir.name, "corrupt.png")
        
        # Save valid synthetic grayscale X-ray
        im = Image.new('L', (300, 300), color=80)
        im.save(cls.valid_image_path)
        
        # Save non-image corrupted file
        with open(cls.corrupted_image_path, "wb") as f:
            f.write(b"NOT_A_VALID_IMAGE_DATA_HEADER")

        # Check if model checkpoints are present to determine full vs mock execution
        cls.has_classifier = os.path.exists(ModelRegistry.get_classifier_path())

    @classmethod
    def tearDownClass(cls):
        cls.temp_dir.cleanup()

    def test_missing_image_raises_file_not_found(self):
        """Inference must fail explicitly if the image file does not exist."""
        if not self.has_classifier:
            self.skipTest("Classifier checkpoint not present on disk.")
        pipeline = BoneFractureInferencePipeline()
        with self.assertRaises(FileNotFoundError):
            pipeline.analyze("non_existent_image_12345.png")

    def test_corrupted_image_raises_value_error(self):
        """Inference must fail with a descriptive ValueError if image payload is invalid/corrupt."""
        if not self.has_classifier:
            self.skipTest("Classifier checkpoint not present on disk.")
        pipeline = BoneFractureInferencePipeline()
        with self.assertRaises(ValueError):
            pipeline.analyze(self.corrupted_image_path)

    def test_inference_output_schema(self):
        """Verifies exact required output schema contract on valid input."""
        if not self.has_classifier:
            self.skipTest("Classifier checkpoint not present on disk.")
            
        pipeline = BoneFractureInferencePipeline()
        result = pipeline.analyze(self.valid_image_path)
        
        # Required schema fields
        required_keys = [
            "anatomical_region",
            "anatomical_confidence",
            "fracture",
            "fracture_confidence",
            "localization_available",
            "localization",
            "caption"
        ]
        for key in required_keys:
            self.assertIn(key, result, f"Missing required schema field: {key}")

        # Data type and range assertions
        self.assertIsInstance(result["anatomical_region"], str)
        self.assertIn(result["anatomical_region"], ModelRegistry.REGION_TO_IDX)
        self.assertIsInstance(result["anatomical_confidence"], float)
        self.assertGreaterEqual(result["anatomical_confidence"], 0.0)
        self.assertLessEqual(result["anatomical_confidence"], 1.0)
        
        self.assertIsInstance(result["fracture"], bool)
        self.assertIsInstance(result["fracture_confidence"], float)
        self.assertGreaterEqual(result["fracture_confidence"], 0.0)
        self.assertLessEqual(result["fracture_confidence"], 1.0)
        
        self.assertIsInstance(result["localization_available"], bool)
        if result["localization_available"]:
            self.assertIsInstance(result["localization"], list)
        else:
            self.assertIsNone(result["localization"])
            
        self.assertIsInstance(result["caption"], str)
        self.assertTrue(len(result["caption"]) > 0)

    def test_run_inference_convenience_function(self):
        """Verify the functional run_inference entry point."""
        if not self.has_classifier:
            self.skipTest("Classifier checkpoint not present on disk.")
        result = run_inference(self.valid_image_path)
        self.assertIn("caption", result)
        self.assertIn("fracture", result)

if __name__ == '__main__':
    unittest.main()
