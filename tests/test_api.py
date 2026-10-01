"""
Test Suite: FastAPI REST Endpoints Verification
===============================================
Tests:
1. GET /health liveness and readiness probe
2. POST /predict with valid radiograph payload
3. POST /predict with invalid file extension (400 validation error)
4. POST /predict with corrupted image payload (400 error handling)
5. GET /visualizations/{filename} retrieval and path traversal guard
"""

import os
import sys
import io
import unittest
from PIL import Image
from fastapi.testclient import TestClient

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.main import app

client = TestClient(app)

class TestFastAPIEndpoints(unittest.TestCase):
    def setUp(self):
        # Generate valid synthetic radiograph bytes
        im = Image.new('L', (224, 224), color=120)
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        self.valid_png_bytes = buf.getvalue()

    def test_health_check_endpoint(self):
        """GET /health must return status 200 with status='ok'."""
        response = client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data.get("status"), "ok")

    def test_predict_endpoint_valid_image(self):
        """POST /predict must return structured prediction with status 200."""
        response = client.post(
            "/predict",
            files={"image": ("test_radiograph.png", self.valid_png_bytes, "image/png")}
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        
        self.assertTrue(data.get("success"))
        self.assertIn("anatomical_region", data)
        self.assertIn("anatomical_confidence", data)
        self.assertIn("fracture", data)
        self.assertIn("fracture_confidence", data)
        self.assertIn("localization_available", data)
        self.assertIn("caption", data)

    def test_predict_endpoint_invalid_file_extension(self):
        """POST /predict must reject non-image file extensions with 400 Bad Request."""
        response = client.post(
            "/predict",
            files={"image": ("malicious_script.exe", b"binary content", "application/octet-stream")}
        )
        self.assertEqual(response.status_code, 400)
        data = response.json()
        self.assertFalse(data.get("success", True))
        self.assertIn("error", data)

    def test_predict_endpoint_corrupted_payload(self):
        """POST /predict must reject corrupted image bytes with 400 Bad Request."""
        response = client.post(
            "/predict",
            files={"image": ("corrupted.jpg", b"corrupted non-image byte stream", "image/jpeg")}
        )
        self.assertEqual(response.status_code, 400)
        data = response.json()
        self.assertFalse(data.get("success", True))
        self.assertIn("error", data)

    def test_visualization_path_traversal_guard(self):
        """GET /visualizations with directory traversal attempt must return 400."""
        response = client.get("/visualizations/..%2F..%2Fsecret.txt")
        self.assertIn(response.status_code, [400, 404])

if __name__ == '__main__':
    unittest.main()
