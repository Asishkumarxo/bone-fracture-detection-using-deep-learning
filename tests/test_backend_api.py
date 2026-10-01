"""
Automated Test Suite for FastAPI Backend API
============================================
Tests:
1. GET /health
2. POST /predict with normal radiograph
3. POST /predict with acute fracture radiograph
4. POST /predict with FracAtlas lesion image
5. POST /predict with corrupted/invalid image (verify 400 error)
6. GET /visualizations/{filename} (verify retrieval of generated overlay)
"""

import os
import sys

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data == {"status": "ok"}
    print("\n[PASSED] GET /health returned 200 OK")

def _get_test_image_bytes(preferred_paths, fallback_filename="synth_xray.png"):
    from PIL import Image
    import io
    for p in preferred_paths:
        if p and os.path.exists(p):
            with open(p, "rb") as f:
                return f.read(), os.path.basename(p), "image/png"
    # Create valid synthetic bone radiograph in-memory
    im = Image.new('L', (224, 224), color=40)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue(), fallback_filename, "image/png"

def test_predict_normal_image():
    candidates = [
        "processed/test/Arm_patient04158_Negative_001.png",
        "tests/fixtures/sample_normal.png"
    ]
    img_bytes, filename, ctype = _get_test_image_bytes(candidates, "sample_normal.png")
    
    response = client.post(
        "/predict",
        files={"image": (filename, img_bytes, ctype)}
    )
        
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "anatomical_region" in data
    assert "fracture" in data
    assert "caption" in data
    assert "visualization_url" in data
    print(f"\n[PASSED] POST /predict (Normal): {data['anatomical_region']} | Fracture={data['fracture']} | \"{data['caption']}\"")

def test_predict_fractured_image():
    candidates = [
        "processed/test/Thigh_patient16212_Positive_001.png",
        "tests/fixtures/sample_fractured.png"
    ]
    img_bytes, filename, ctype = _get_test_image_bytes(candidates, "sample_fractured.png")
    
    response = client.post(
        "/predict",
        files={"image": (filename, img_bytes, ctype)}
    )
        
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "anatomical_region" in data
    assert "fracture" in data
    assert "caption" in data
    assert data["visualization_url"] is not None
    print(f"\n[PASSED] POST /predict (Fracture): {data['anatomical_region']} | Fracture={data['fracture']} | \"{data['caption']}\"")
    
    # Test visualization retrieval
    vis_url = data["visualization_url"]
    vis_resp = client.get(vis_url)
    assert vis_resp.status_code == 200
    assert vis_resp.headers["content-type"] == "image/png"
    print(f"[PASSED] GET {vis_url} successfully returned annotated image ({len(vis_resp.content):,} bytes)")

def test_predict_fracatlas_image():
    fracatlas_dir = os.environ.get("FRACATLAS_DIR", "data/FracAtlas")
    candidates = [
        os.path.join(fracatlas_dir, "images", "Fractured", "IMG0003297.jpg"),
        "E:/FracAtlas/images/Fractured/IMG0003297.jpg",
        "tests/fixtures/sample_localized.png"
    ]
    img_bytes, filename, ctype = _get_test_image_bytes(candidates, "sample_fracatlas.jpg")
    
    response = client.post(
        "/predict",
        files={"image": (filename, img_bytes, ctype)}
    )
        
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "anatomical_region" in data
    assert "caption" in data
    print(f"\n[PASSED] POST /predict (FracAtlas/Localized): {data['anatomical_region']} | Fracture={data['fracture']} | Loc={data['localization_available']}")

def test_predict_corrupted_file():
    corrupt_path = "tests/invalid_test_file.jpg"
    if not os.path.exists(corrupt_path):
        os.makedirs("tests", exist_ok=True)
        with open(corrupt_path, "wb") as f:
            f.write(b"not an image file content")
            
    with open(corrupt_path, "rb") as f:
        response = client.post(
            "/predict",
            files={"image": ("corrupted.jpg", f, "image/jpeg")}
        )
        
    assert response.status_code == 400
    data = response.json()
    assert data["success"] is False
    assert "error" in data
    print(f"\n[PASSED] POST /predict (Corrupted File): Trapped 400 Bad Request with message: \"{data['error']}\"")

if __name__ == "__main__":
    test_health_endpoint()
    test_predict_normal_image()
    test_predict_fractured_image()
    test_predict_fracatlas_image()
    test_predict_corrupted_file()
    print("\n==========================================")
    print("ALL FASTAPI BACKEND TESTS PASSED!")
    print("==========================================")
