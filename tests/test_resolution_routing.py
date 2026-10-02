"""
Test Suite: Resolution-Aware Inference Routing & Verification
============================================================
Validates:
1. Decision boundary min(width, height) >= 300 routing logic.
2. Mandatory test dimensions:
   - 102x102  -> best_model.pt (224x224)
   - 800x800  -> exp3_resnet50_448.pt (448x448)
   - 373x454  -> exp3_resnet50_448.pt (448x448)
   - 1920x876 -> exp3_resnet50_448.pt (448x448)
   - 809x1296 -> exp3_resnet50_448.pt (448x448)
3. Selected model strictly matches preprocessing resolution.
4. FastAPI /predict integration returns routing metadata without breaking schema.
5. End-to-end outputs: anatomy, fracture status, confidence, caption.
"""

import os
import io
import sys
import pytest
from PIL import Image
from fastapi.testclient import TestClient

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from inference.model_registry import ModelRegistry
from inference.preprocessing import preprocess_for_classifier, get_original_dimensions
from predict import BoneFractureInferencePipeline
from backend.main import app

client = TestClient(app)

def create_synthetic_xray(width: int, height: int, fill: int = 70) -> Image.Image:
    """Creates an in-memory synthetic radiograph of specified native dimensions."""
    return Image.new('L', (width, height), color=fill)

def pil_to_png_bytes(pil_img: Image.Image) -> bytes:
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return buf.getvalue()

# -------------------------------------------------------------
# 1. Routing Rule Unit Tests
# -------------------------------------------------------------
class TestRoutingLogic:
    def test_routing_102x102(self):
        decision = ModelRegistry.route_image(102, 102)
        assert decision["selected_model"] == "best_model.pt"
        assert decision["routing_resolution"] == "224"
        assert decision["target_size"] == (224, 224)
        assert decision["input_width"] == 102
        assert decision["input_height"] == 102
        assert decision["branch"] == "low_resolution"

    def test_routing_800x800(self):
        decision = ModelRegistry.route_image(800, 800)
        assert decision["selected_model"] == "exp3_resnet50_448.pt"
        assert decision["routing_resolution"] == "448"
        assert decision["target_size"] == (448, 448)
        assert decision["input_width"] == 800
        assert decision["input_height"] == 800
        assert decision["branch"] == "high_resolution"

    def test_routing_373x454(self):
        decision = ModelRegistry.route_image(373, 454)
        assert decision["selected_model"] == "exp3_resnet50_448.pt"
        assert decision["routing_resolution"] == "448"
        assert decision["target_size"] == (448, 448)
        assert min(373, 454) >= 300
        assert decision["branch"] == "high_resolution"

    def test_routing_1920x876(self):
        decision = ModelRegistry.route_image(1920, 876)
        assert decision["selected_model"] == "exp3_resnet50_448.pt"
        assert decision["routing_resolution"] == "448"
        assert decision["target_size"] == (448, 448)
        assert min(1920, 876) >= 300
        assert decision["branch"] == "high_resolution"

    def test_routing_809x1296(self):
        decision = ModelRegistry.route_image(809, 1296)
        assert decision["selected_model"] == "exp3_resnet50_448.pt"
        assert decision["routing_resolution"] == "448"
        assert decision["target_size"] == (448, 448)
        assert min(809, 1296) >= 300
        assert decision["branch"] == "high_resolution"

    def test_boundary_conditions(self):
        # min is 299 -> low res
        dec_299 = ModelRegistry.route_image(299, 1000)
        assert dec_299["selected_model"] == "best_model.pt"
        assert dec_299["routing_resolution"] == "224"

        dec_299_b = ModelRegistry.route_image(1000, 299)
        assert dec_299_b["selected_model"] == "best_model.pt"
        assert dec_299_b["routing_resolution"] == "224"

        # min is 300 -> high res
        dec_300 = ModelRegistry.route_image(300, 300)
        assert dec_300["selected_model"] == "exp3_resnet50_448.pt"
        assert dec_300["routing_resolution"] == "448"

        dec_300_tall = ModelRegistry.route_image(300, 600)
        assert dec_300_tall["selected_model"] == "exp3_resnet50_448.pt"
        assert dec_300_tall["routing_resolution"] == "448"

# -------------------------------------------------------------
# 2. Pipeline End-to-End Inference Tests
# -------------------------------------------------------------
class TestPipelineResolutionRouting:
    @classmethod
    def setup_class(cls):
        cls.pipeline = BoneFractureInferencePipeline()

    @pytest.mark.parametrize("width,height,expected_model,expected_res", [
        (102, 102, "best_model.pt", "224"),
        (800, 800, "exp3_resnet50_448.pt", "448"),
        (373, 454, "exp3_resnet50_448.pt", "448"),
        (1920, 876, "exp3_resnet50_448.pt", "448"),
        (809, 1296, "exp3_resnet50_448.pt", "448"),
    ])
    def test_pipeline_routing_dimensions(self, width, height, expected_model, expected_res):
        img = create_synthetic_xray(width, height)
        result = self.pipeline.analyze(img)

        # Check metadata
        assert result["input_width"] == width
        assert result["input_height"] == height
        assert result["selected_model"] == expected_model
        assert result["routing_resolution"] == expected_res

        # Confirm selected model matches preprocessing resolution
        if expected_model == "exp3_resnet50_448.pt":
            assert result["routing_resolution"] == "448"
        else:
            assert result["routing_resolution"] == "224"

        # Check predictions valid
        assert result["anatomical_region"] in ModelRegistry.REGION_TO_IDX
        assert 0.0 <= result["anatomical_confidence"] <= 1.0
        assert isinstance(result["fracture"], bool)
        assert 0.0 <= result["fracture_confidence"] <= 1.0
        assert isinstance(result["caption"], str)
        assert len(result["caption"]) > 0

# -------------------------------------------------------------
# 3. FastAPI /predict Integration Tests
# -------------------------------------------------------------
class TestFastAPIRoutingEndpoint:
    @pytest.mark.parametrize("width,height,expected_model,expected_res", [
        (102, 102, "best_model.pt", "224"),
        (800, 800, "exp3_resnet50_448.pt", "448"),
        (373, 454, "exp3_resnet50_448.pt", "448"),
        (1920, 876, "exp3_resnet50_448.pt", "448"),
        (809, 1296, "exp3_resnet50_448.pt", "448"),
    ])
    def test_api_predict_routing(self, width, height, expected_model, expected_res):
        img = create_synthetic_xray(width, height)
        img_bytes = pil_to_png_bytes(img)
        fname = f"test_{width}x{height}.png"

        response = client.post(
            "/predict",
            files={"image": (fname, img_bytes, "image/png")}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["input_width"] == width
        assert data["input_height"] == height
        assert data["selected_model"] == expected_model
        assert data["routing_resolution"] == expected_res
        assert data["anatomical_region"] in ModelRegistry.REGION_TO_IDX
        assert isinstance(data["fracture"], bool)
        assert "caption" in data
        assert data["visualization_url"] is not None
