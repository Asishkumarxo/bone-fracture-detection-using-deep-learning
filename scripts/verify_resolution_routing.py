"""
Standalone Verification Script for Resolution-Aware Inference Routing
=====================================================================
Tests:
1. 102x102 real dataset radiograph
2. 800x800 real dataset radiograph
3. 373x454 real dataset radiograph
4. 1920x876 real external radiograph
5. 809x1296 real external radiograph (hero_xray.jpg / sample_wrist.jpg)
6. Checkpoint integrity (SHA-256 hash validation)
7. FastAPI endpoint verification
"""

import os
import sys
import hashlib
from PIL import Image
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from predict import BoneFractureInferencePipeline
from backend.main import app

def sha256_hash(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest().upper()

def main():
    print("=" * 70)
    print("RESOLUTION-AWARE INFERENCE ROUTING VERIFICATION SUITE")
    print("=" * 70)

    # 1. Verify Checkpoint Hashes
    baseline_path = "models/checkpoints/best_model.pt"
    exp3_path = "models/checkpoints/exp3_resnet50_448.pt"

    baseline_expected = "480B21A5E07E2F3A36E453FA70F0964EF944E13FE010644FD551770769121FC1"
    exp3_expected = "F31533C52D58AE756D0ED93691513E19F5F29442498A0882FE467D5D91986DFB"

    baseline_actual = sha256_hash(baseline_path)
    exp3_actual = sha256_hash(exp3_path)

    print(f"\n[CHECKPOINT INTEGRITY]")
    print(f"  best_model.pt:        {baseline_actual} (Match: {baseline_actual == baseline_expected})")
    print(f"  exp3_resnet50_448.pt: {exp3_actual} (Match: {exp3_actual == exp3_expected})")
    assert baseline_actual == baseline_expected, "best_model.pt hash changed!"
    assert exp3_actual == exp3_expected, "exp3_resnet50_448.pt hash changed!"

    # 2. Pipeline Initialization
    pipeline = BoneFractureInferencePipeline()

    # 3. Test Cases Configuration
    p1 = r"C:\Users\arbaz\Downloads\BoneFract A Bone Fracture Dataset\BoneFract A Bone Fracture Dataset\BoneFract A Bone Fracture Dataset\BoneFract A Bone Fracture Dataset\test\Arm\patient03647\Negative\Arm_patient03647_Negative_001.png"
    p2 = r"C:\Users\arbaz\Downloads\BoneFract A Bone Fracture Dataset\BoneFract A Bone Fracture Dataset\BoneFract A Bone Fracture Dataset\BoneFract A Bone Fracture Dataset\test\Arm\patient26948\Positive\Arm_patient26948_Positive_001.png"
    p3 = r"C:\Users\arbaz\Downloads\BoneFract A Bone Fracture Dataset\BoneFract A Bone Fracture Dataset\BoneFract A Bone Fracture Dataset\BoneFract A Bone Fracture Dataset\test\Arm\patient29529\Positive\Arm_patient29529_Positive_001.png"
    
    # Create external 1920x876 test image from real wrist X-ray
    os.makedirs("tests/fixtures", exist_ok=True)
    p4 = "tests/fixtures/external_1920x876.png"
    if not os.path.exists(p4):
        with Image.open("frontend/assets/hero_xray.jpg") as im_hero:
            im_1920 = im_hero.resize((1920, 876), Image.Resampling.BILINEAR)
            im_1920.save(p4)

    p5 = "frontend/assets/sample_wrist.jpg"

    test_cases = [
        ("1. 102x102 Native Image", p1, 102, 102, "best_model.pt", "224"),
        ("2. 800x800 Native Image", p2, 800, 800, "exp3_resnet50_448.pt", "448"),
        ("3. 373x454 Native Image", p3, 373, 454, "exp3_resnet50_448.pt", "448"),
        ("4. 1920x876 External X-Ray", p4, 1920, 876, "exp3_resnet50_448.pt", "448"),
        ("5. 809x1296 External X-Ray", p5, 809, 1296, "exp3_resnet50_448.pt", "448"),
    ]

    print("\n" + "=" * 70)
    print("PIPELINE ROUTING & DIAGNOSTIC VERIFICATION")
    print("=" * 70)

    client = TestClient(app)

    for label, path, exp_w, exp_h, exp_model, exp_res in test_cases:
        with Image.open(path) as im:
            actual_w, actual_h = im.size

        assert (actual_w, actual_h) == (exp_w, exp_h), f"Size mismatch for {label}: expected ({exp_w},{exp_h}), got ({actual_w},{actual_h})"

        # A. Test BoneFractureInferencePipeline
        res = pipeline.analyze(path)
        assert res["input_width"] == exp_w
        assert res["input_height"] == exp_h
        assert res["selected_model"] == exp_model
        assert res["routing_resolution"] == exp_res

        # B. Test FastAPI /predict endpoint
        with open(path, "rb") as f:
            resp = client.post("/predict", files={"image": (os.path.basename(path), f, "image/png" if path.endswith(".png") else "image/jpeg")})
        assert resp.status_code == 200, f"API failed with {resp.status_code}: {resp.text}"
        api_data = resp.json()
        assert api_data["selected_model"] == exp_model
        assert api_data["routing_resolution"] == exp_res
        assert api_data["input_width"] == exp_w
        assert api_data["input_height"] == exp_h

        status_str = "POSITIVE (Fracture)" if res['fracture'] else "NEGATIVE (Normal)"
        print(f"\n--- {label} ---")
        print(f"  Path:                {path}")
        print(f"  Native Dimensions:   {actual_w} x {actual_h} px (min = {min(actual_w, actual_h)})")
        print(f"  Routing Decision:    {res['routing_resolution']}x{res['routing_resolution']} Preprocessing -> {res['selected_model']}")
        print(f"  Model Selection:     {'[MATCH]' if res['selected_model'] == exp_model else '[FAIL]'}")
        print(f"  Resolution Match:    {'[MATCH]' if res['routing_resolution'] == exp_res else '[FAIL]'}")
        print(f"  Anatomical Region:   {res['anatomical_region']} (Conf: {res['anatomical_confidence']*100:.1f}%)")
        print(f"  Fracture Status:     {status_str} (Conf: {res['fracture_confidence']*100:.1f}%)")
        print(f"  Caption:             \"{res['caption']}\"")
        print(f"  FastAPI API Status:  HTTP 200 OK | Model: {api_data['selected_model']} | Res: {api_data['routing_resolution']}")

    print("\n" + "=" * 70)
    print("ALL VERIFICATIONS COMPLETED SUCCESSFULLY!")
    print("=" * 70)

if __name__ == "__main__":
    main()
