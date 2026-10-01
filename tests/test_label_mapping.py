"""
Test Suite: Anatomical & Pathological Label Mappings
===================================================
Tests:
1. Exact anatomical region class count and names (7 classes)
2. Verified class taxonomy (Arm, Foot, Hand, Lower leg, Thigh, pelvis, wrist)
3. Absence of unauthorized 'Hip' class (mapped under 'pelvis' in BoneFract)
4. Bidirectional mapping consistency (forward and inverse dictionaries)
5. Binary fracture status encoding (Negative=0, Positive=1)
6. Decision and uncertainty thresholds
"""

import os
import sys
import unittest

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from inference.model_registry import ModelRegistry

class TestLabelMapping(unittest.TestCase):
    def test_anatomical_classes(self):
        """Verify the exact 7 anatomical classes defined in the trained model."""
        expected_classes = {'Arm', 'Foot', 'Hand', 'Lower leg', 'Thigh', 'pelvis', 'wrist'}
        actual_classes = set(ModelRegistry.REGION_TO_IDX.keys())
        self.assertEqual(actual_classes, expected_classes)
        self.assertEqual(ModelRegistry.NUM_REGIONS, 7)

    def test_pelvis_vs_hip_taxonomy(self):
        """
        Verify dataset convention: 'pelvis' is the ground truth class;
        'Hip' must NOT be an independent class in this model registry.
        """
        self.assertIn('pelvis', ModelRegistry.REGION_TO_IDX)
        self.assertNotIn('Hip', ModelRegistry.REGION_TO_IDX)
        self.assertNotIn('hip', ModelRegistry.REGION_TO_IDX)

    def test_bidirectional_region_mapping(self):
        """Verify 1-to-1 bijection between REGION_TO_IDX and IDX_TO_REGION."""
        for name, idx in ModelRegistry.REGION_TO_IDX.items():
            self.assertEqual(ModelRegistry.IDX_TO_REGION[idx], name)

        for idx, name in ModelRegistry.IDX_TO_REGION.items():
            self.assertEqual(ModelRegistry.REGION_TO_IDX[name], idx)

    def test_fracture_mapping(self):
        """Verify binary fracture label convention."""
        self.assertEqual(ModelRegistry.FRACTURE_TO_IDX['Negative'], 0)
        self.assertEqual(ModelRegistry.FRACTURE_TO_IDX['Positive'], 1)
        self.assertEqual(ModelRegistry.IDX_TO_FRACTURE[0], 'Negative')
        self.assertEqual(ModelRegistry.IDX_TO_FRACTURE[1], 'Positive')

    def test_calibrated_thresholds(self):
        """Verify validation-established operating thresholds."""
        self.assertEqual(ModelRegistry.FRACTURE_THRESHOLD, 0.50)
        self.assertEqual(ModelRegistry.DETECTOR_CONFIDENCE_THRESHOLD, 0.10)
        self.assertGreater(ModelRegistry.REGION_UNCERTAINTY_THRESHOLD, 0.0)
        self.assertLess(ModelRegistry.REGION_UNCERTAINTY_THRESHOLD, 1.0)

if __name__ == '__main__':
    unittest.main()
