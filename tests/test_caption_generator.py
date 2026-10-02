"""
Unit Tests for Factual Medical Description Generator
====================================================
Validates all 10 clinical rules and exact formatting specifications.
"""

import os
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
from src.caption_generator import generate_description, generate_factual_caption
from inference.caption_generator import generate_caption

class TestCaptionGenerator(unittest.TestCase):

    def test_example_1_negative(self):
        """
        If:
        region = wrist
        fracture = false

        Output:
        "X-ray of the wrist with no fracture detected by the model."
        """
        pred = {
            "anatomical_region": "wrist",
            "fracture": False
        }
        caption = generate_description(pred)
        self.assertEqual(caption, "X-ray of the wrist with no fracture detected by the model.")

    def test_example_2_positive_no_location(self):
        """
        If:
        region = wrist
        fracture = true

        Output:
        "X-ray of the wrist with a fracture detected by the model."
        """
        pred = {
            "anatomical_region": "wrist",
            "fracture": True
        }
        caption = generate_description(pred)
        self.assertEqual(caption, "X-ray of the wrist with a fracture detected by the model.")

    def test_example_3_positive_with_location(self):
        """
        If:
        region = wrist
        fracture = true
        location = distal radius

        Output:
        "X-ray of the wrist showing a fracture involving the distal radius."
        """
        pred = {
            "anatomical_region": "wrist",
            "fracture": True,
            "fracture_location": "distal radius"
        }
        caption = generate_description(pred)
        self.assertEqual(caption, "X-ray of the wrist showing a fracture involving the distal radius.")

    def test_location_with_leading_the(self):
        """
        If location already starts with 'the', do not duplicate 'the'.
        """
        pred = {
            "anatomical_region": "wrist",
            "fracture": True,
            "fracture_location": "the distal radius"
        }
        caption = generate_description(pred)
        self.assertEqual(caption, "X-ray of the wrist showing a fracture involving the distal radius.")

    def test_region_with_leading_the(self):
        pred = {
            "anatomical_region": "the wrist",
            "fracture": False
        }
        caption = generate_description(pred)
        self.assertEqual(caption, "X-ray of the wrist with no fracture detected by the model.")

    def test_capitalized_and_spaced_region(self):
        pred = {
            "anatomical_region": " Lower leg ",
            "fracture": True
        }
        caption = generate_description(pred)
        self.assertEqual(caption, "X-ray of the lower leg with a fracture detected by the model.")

    def test_unavailable_localization_ignored(self):
        """
        If localization is unavailable (None, empty, N/A), do not mention a precise location.
        """
        for unavail in [None, "", "N/A", "unavailable", "None", "unknown"]:
            pred = {
                "anatomical_region": "wrist",
                "fracture": True,
                "fracture_location": unavail
            }
            caption = generate_description(pred)
            self.assertEqual(caption, "X-ray of the wrist with a fracture detected by the model.")

    def test_negative_fracture_ignores_location(self):
        """
        If fracture is false, location should not be mentioned even if passed.
        """
        pred = {
            "anatomical_region": "wrist",
            "fracture": False,
            "fracture_location": "distal radius"
        }
        caption = generate_description(pred)
        self.assertEqual(caption, "X-ray of the wrist with no fracture detected by the model.")

    def test_determinism(self):
        """
        The caption generator must be deterministic for identical structured inputs.
        """
        pred = {
            "anatomical_region": "wrist",
            "fracture": True,
            "fracture_confidence": 0.94,
            "fracture_location": "distal radius"
        }
        results = [generate_description(pred) for _ in range(100)]
        self.assertTrue(all(r == results[0] for r in results))

    def test_prohibited_terms_absent(self):
        """
        Ensures none of the prohibited clinical hallucination terms appear.
        """
        prohibited = [
            "displaced", "displacement", "comminuted", "transverse", "spiral",
            "greenstick", "hairline", "oblique", "severity", "severe", "mild",
            "moderate", "male", "female", "year-old", "pediatric", "adult",
            "fall", "trauma", "foosh", "collision", "surgery", "cast", "referral",
            "definitive diagnosis", "diagnosed with"
        ]
        
        test_inputs = [
            {"anatomical_region": "wrist", "fracture": False},
            {"anatomical_region": "wrist", "fracture": True},
            {"anatomical_region": "wrist", "fracture": True, "fracture_location": "distal radius"},
            {"anatomical_region": "foot", "fracture": True, "fracture_location": "5th metatarsal"},
            {"anatomical_region": "hand", "fracture": False},
        ]
        
        for inp in test_inputs:
            caption = generate_description(inp).lower()
            for term in prohibited:
                self.assertNotIn(term, caption, f"Prohibited term '{term}' found in '{caption}'")

    def test_backwards_compatibility_wrapper(self):
        # Dict signature
        c1 = generate_factual_caption({"anatomical_region": "wrist", "fracture": False})
        self.assertEqual(c1, "X-ray of the wrist with no fracture detected by the model.")
        
        # Positional signature
        c2 = generate_factual_caption("wrist", 0.99, True, 0.95, None, "distal radius")
        self.assertEqual(c2, "X-ray of the wrist showing a fracture involving the distal radius.")

    def test_fourteen_combinations_all_seven_regions(self):
        """
        Tests all 7 anatomical regions across both fracture=True and fracture=False (14 combinations)
        for both inference.caption_generator and src.caption_generator.
        """
        regions = ["Arm", "Foot", "Hand", "Lower leg", "Thigh", "pelvis", "wrist"]
        for region in regions:
            # Negative case
            neg_pred = {"anatomical_region": region, "fracture": False}
            inf_neg = generate_caption(neg_pred)
            src_neg = generate_description(neg_pred)
            expected_neg = f"X-ray of the {region.lower()} with no fracture detected by the model."
            self.assertEqual(inf_neg, expected_neg)
            self.assertEqual(src_neg, expected_neg)

            # Positive case
            pos_pred = {"anatomical_region": region, "fracture": True}
            inf_pos = generate_caption(pos_pred)
            src_pos = generate_description(pos_pred)
            self.assertEqual(inf_pos, f"X-ray of the {region.lower()} showing a fracture.")
            self.assertEqual(src_pos, f"X-ray of the {region.lower()} with a fracture detected by the model.")

    def test_edge_cases_and_invalid_missing_values(self):
        """
        Tests missing anatomy, unknown anatomy, missing fracture, None values, and invalid inputs.
        """
        edge_cases = [
            ({}, "X-ray with no fracture detected by the model."),
            ({"fracture": False}, "X-ray with no fracture detected by the model."),
            ({"fracture": True}, "X-ray showing a fracture."),
            ({"anatomical_region": None, "fracture": False}, "X-ray with no fracture detected by the model."),
            ({"anatomical_region": None, "fracture": True}, "X-ray showing a fracture."),
            ({"anatomical_region": "unknown", "fracture": False}, "X-ray with no fracture detected by the model."),
            ({"anatomical_region": "unknown", "fracture": True}, "X-ray showing a fracture."),
            ({"anatomical_region": "N/A", "fracture": False}, "X-ray with no fracture detected by the model."),
            ({"anatomical_region": "N/A", "fracture": True}, "X-ray showing a fracture."),
            ({"anatomical_region": "wrist", "fracture": None}, "X-ray of the wrist with no fracture detected by the model."),
            ({"anatomical_region": "wrist", "fracture": True, "fracture_confidence": 0.0001}, "X-ray of the wrist showing a fracture."),
            ({"anatomical_region": "wrist", "fracture": False, "fracture_confidence": 0.4999}, "X-ray of the wrist with no fracture detected by the model."),
            ({"anatomical_region": "wrist", "fracture": True, "fracture_confidence": 0.5001}, "X-ray of the wrist showing a fracture."),
            ({"anatomical_region": "wrist", "fracture": True, "fracture_confidence": 0.9999}, "X-ray of the wrist showing a fracture."),
        ]
        for pred, expected in edge_cases:
            res = generate_caption(pred)
            self.assertEqual(res, expected)

if __name__ == "__main__":
    unittest.main()
