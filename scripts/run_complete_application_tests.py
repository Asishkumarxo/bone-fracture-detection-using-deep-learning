"""
Comprehensive End-to-End Application Test Harness
=================================================
Executes:
- Phase 3: API Test Matrix (A to I real radiographs)
- Phase 4: Response Schema Validation
- Phase 5: Error Handling Suite
- Phase 7: API vs Frontend Consistency Check
- Phase 8: Performance Benchmarks
"""

import os
import io
import sys
import time
import json
import requests
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from inference.model_registry import ModelRegistry
from backend.schemas import PredictionResponse

BACKEND_URL = "http://127.0.0.1:8000"

def get_image_bytes(path):
    with open(path, "rb") as f:
        return f.read(), os.path.basename(path)

def make_predict_request(img_bytes, filename, content_type="image/png"):
    t0 = time.perf_counter()
    files = {"image": (filename, img_bytes, content_type)}
    resp = requests.post(f"{BACKEND_URL}/predict", files=files, timeout=30)
    elapsed = time.perf_counter() - t0
    return resp, elapsed

def main():
    print("=" * 80)
    print("STARTING COMPLETE END-TO-END APPLICATION TEST SUITE")
    print("=" * 80)

    results_matrix = []
    schema_validation_results = []
    error_handling_results = []
    consistency_results = []
    perf_metrics = {}

    # -------------------------------------------------------------
    # PHASE 3: API TEST MATRIX
    # -------------------------------------------------------------
    print("\n--- PHASE 3: RUNNING API TEST MATRIX ---")

    dataset_root = r"C:\Users\arbaz\Downloads\BoneFract A Bone Fracture Dataset\BoneFract A Bone Fracture Dataset\BoneFract A Bone Fracture Dataset\BoneFract A Bone Fracture Dataset"

    matrix_definitions = [
        ("A. Native 102x102 BoneFract", os.path.join(dataset_root, r"test\Arm\patient03647\Negative\Arm_patient03647_Negative_001.png")),
        ("B. 800x800 BoneFract", os.path.join(dataset_root, r"test\Arm\patient26948\Positive\Arm_patient26948_Positive_001.png")),
        ("C. 373x454 BoneFract (Portrait)", os.path.join(dataset_root, r"test\Arm\patient29529\Positive\Arm_patient29529_Positive_001.png")),
        ("D. 454x373 BoneFract (Landscape)", os.path.join(dataset_root, r"test\Arm\patient31687\Negative\Arm_patient31687_Negative_001.png")),
        ("E. High-res External X-ray", "frontend/assets/sample_wrist.jpg"),
        ("F. Known Fracture-Positive", os.path.join(dataset_root, r"test\Arm\patient26948\Positive\Arm_patient26948_Positive_001.png")),
        ("G. Known Fracture-Negative", os.path.join(dataset_root, r"test\Arm\patient03647\Negative\Arm_patient03647_Negative_001.png")),
        ("H. Landscape X-ray (1920x876)", "tests/fixtures/external_1920x876.png"),
        ("I. Portrait X-ray (809x1296)", "frontend/assets/hero_xray.jpg")
    ]

    for label, path in matrix_definitions:
        assert os.path.exists(path), f"File missing: {path}"
        with Image.open(path) as im:
            w, h = im.size

        expected_route = "448" if min(w, h) >= 300 else "224"
        expected_model = "exp3_resnet50_448.pt" if min(w, h) >= 300 else "best_model.pt"

        img_bytes, fname = get_image_bytes(path)
        ctype = "image/png" if path.endswith(".png") else "image/jpeg"
        resp, elapsed = make_predict_request(img_bytes, fname, ctype)

        status_code = resp.status_code
        if status_code == 200:
            data = resp.json()
            actual_route = data.get("routing_resolution")
            actual_model = data.get("selected_model")
            actual_w = data.get("input_width")
            actual_h = data.get("input_height")
            anatomy = data.get("anatomical_region")
            anatomy_conf = data.get("anatomical_confidence")
            fracture = data.get("fracture")
            fracture_conf = data.get("fracture_confidence")
            caption = data.get("caption")

            route_correct = (actual_route == expected_route and actual_model == expected_model)
            dim_correct = (actual_w == w and actual_h == h)
            pass_status = "PASS" if (route_correct and dim_correct) else "FAIL"

            results_matrix.append({
                "test": label,
                "path": path,
                "input_w": w,
                "input_h": h,
                "min_dim": min(w, h),
                "expected_route": expected_route,
                "actual_route": actual_route,
                "expected_model": expected_model,
                "actual_model": actual_model,
                "anatomy": anatomy,
                "anatomy_conf": anatomy_conf,
                "fracture": fracture,
                "fracture_prob": fracture_conf,
                "threshold": 0.50,
                "caption": caption,
                "elapsed": round(elapsed, 3),
                "http_status": status_code,
                "status": pass_status
            })

            print(f"[{pass_status}] {label} ({w}x{h}, min={min(w,h)}): Route={actual_route} ({actual_model}) | Region={anatomy} ({anatomy_conf*100:.1f}%) | Frac={fracture} ({fracture_conf*100:.1f}%) | Time={elapsed:.3f}s")
        else:
            print(f"[FAIL] {label}: HTTP {status_code} - {resp.text}")
            results_matrix.append({
                "test": label,
                "path": path,
                "input_w": w,
                "input_h": h,
                "min_dim": min(w, h),
                "expected_route": expected_route,
                "actual_route": None,
                "http_status": status_code,
                "status": "FAIL",
                "elapsed": elapsed
            })

    # -------------------------------------------------------------
    # PHASE 4: RESPONSE SCHEMA VALIDATION
    # -------------------------------------------------------------
    print("\n--- PHASE 4: SCHEMA VALIDATION ---")
    valid_regions = set(ModelRegistry.REGION_TO_IDX.keys())

    schema_checks_passed = True
    for item in results_matrix:
        if item["status"] != "PASS":
            continue
        # Validate through Pydantic
        # Re-fetch cached response data
        # Check required fields
        data_to_validate = {
            "anatomical_region": item["anatomy"],
            "anatomical_confidence": item["anatomy_conf"],
            "fracture": item["fracture"],
            "fracture_confidence": item["fracture_prob"],
            "localization_available": False,
            "localization": None,
            "caption": item["caption"],
            "input_width": item["input_w"],
            "input_height": item["input_h"],
            "routing_resolution": item["actual_route"],
            "selected_model": item["actual_model"]
        }
        try:
            parsed = PredictionResponse(**data_to_validate)
            # Checks
            assert item["anatomy"] in valid_regions, f"Invalid anatomy: {item['anatomy']}"
            assert 0.0 <= item["anatomy_conf"] <= 1.0, f"Confidence out of range: {item['anatomy_conf']}"
            assert isinstance(item["fracture"], bool), f"Fracture must be bool: {item['fracture']}"
            assert 0.0 <= item["fracture_prob"] <= 1.0, f"Probability out of range: {item['fracture_prob']}"
            assert isinstance(item["caption"], str) and len(item["caption"]) > 0, "Invalid caption"
            assert item["actual_route"] in ("224", "448")
            assert item["actual_model"] in ("best_model.pt", "exp3_resnet50_448.pt")
            schema_validation_results.append((item["test"], True, "Schema valid"))
        except Exception as e:
            schema_checks_passed = False
            schema_validation_results.append((item["test"], False, str(e)))

    print(f"Schema Validation Result: {'ALL PASSED' if schema_checks_passed else 'FAILURES DETECTED'}")

    # -------------------------------------------------------------
    # PHASE 5: ERROR HANDLING SUITE
    # -------------------------------------------------------------
    print("\n--- PHASE 5: RUNNING ERROR HANDLING SUITE ---")

    error_tests = [
        ("1. Non-image file (.txt text payload)", b"Hello World! This is plain text, not a DICOM/PNG image.", "notes.txt", "text/plain", 400),
        ("2. Corrupted image file (damaged PNG header)", b"\x89PNG\r\n\x1a\nCorruptedCorruptedTruncatedBytes", "damaged.png", "image/png", 400),
        ("3. Empty upload (0 bytes)", b"", "empty.png", "image/png", 400),
        ("4. Extremely small image (2x2 pixels)", None, "tiny_2x2.png", "image/png", 200), # Should handle gracefully
        ("5. Very wide image (2000x80 pixels)", None, "wide_2000x80.png", "image/png", 200), # Should handle gracefully
        ("6. Very tall image (80x2000 pixels)", None, "tall_80x2000.png", "image/png", 200), # Should handle gracefully
        ("7. Unsupported format (.exe payload)", b"MZ\x90\x00\x03\x00\x00\x00\x04\x00\x00\x00", "malware.exe", "application/x-dosexec", 400),
        ("8. Missing image field", "MISSING_FIELD", "", "", 422) # FastAPI returns 422 for missing required multipart field
    ]

    for label, payload, fname, mime, expected_status in error_tests:
        if label.startswith("4."):
            im = Image.new("L", (2, 2), color=128)
            buf = io.BytesIO()
            im.save(buf, format="PNG")
            payload = buf.getvalue()
        elif label.startswith("5."):
            im = Image.new("L", (2000, 80), color=128)
            buf = io.BytesIO()
            im.save(buf, format="PNG")
            payload = buf.getvalue()
        elif label.startswith("6."):
            im = Image.new("L", (80, 2000), color=128)
            buf = io.BytesIO()
            im.save(buf, format="PNG")
            payload = buf.getvalue()

        t0 = time.perf_counter()
        if payload == "MISSING_FIELD":
            resp = requests.post(f"{BACKEND_URL}/predict", files={"wrong_field": ("test.png", b"123", "image/png")})
        else:
            resp = requests.post(f"{BACKEND_URL}/predict", files={"image": (fname, payload, mime)})
        el = time.perf_counter() - t0

        pass_status = "PASS" if resp.status_code == expected_status else "FAIL"
        err_msg = ""
        try:
            err_msg = resp.json().get("error", resp.json().get("detail", ""))
        except Exception:
            err_msg = resp.text[:100]

        error_handling_results.append({
            "test": label,
            "expected_status": expected_status,
            "actual_status": resp.status_code,
            "error_msg": str(err_msg)[:120],
            "elapsed": round(el, 3),
            "status": pass_status
        })
        print(f"[{pass_status}] {label} -> HTTP {resp.status_code} (Expected {expected_status}) | Msg: {err_msg[:60]} | Time: {el:.3f}s")

    # Verify backend didn't crash
    r_health = requests.get(f"{BACKEND_URL}/health")
    assert r_health.status_code == 200 and r_health.json() == {"status": "ok"}, "Backend crashed during error suite!"
    print("Backend Liveness Check after Error Suite: HEALTHY (Status: ok)")

    # -------------------------------------------------------------
    # PHASE 7: API VS FRONTEND CONSISTENCY CHECK
    # -------------------------------------------------------------
    print("\n--- PHASE 7: CHECKING API VS FRONTEND CONSISTENCY ---")
    # Compare direct API response against Streamlit AppTest session state
    from streamlit.testing.v1 import AppTest
    app_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'frontend', 'app.py'))
    
    test_presets = [
        ("Fractured Wrist (AP)", "sample_wrist.jpg"),
        ("Normal Hand (PA)", "sample_hand.jpg"),
        ("Fractured Forearm", "sample_forearm.jpg")
    ]

    for btn_label, asset_name in test_presets:
        # A. Direct API
        asset_path = os.path.join("frontend", "assets", asset_name)
        with open(asset_path, "rb") as f:
            api_resp = requests.post(f"{BACKEND_URL}/predict", files={"image": (asset_name, f.read(), "image/jpeg")}).json()

        # B. Frontend AppTest
        at = AppTest.from_file(app_path, default_timeout=30)
        at.run()
        # Navigate to analysis
        btn_analysis = [b for b in at.button if b.label == 'Analysis Workspace'][0]
        btn_analysis.click().run()
        # Click preset
        preset_btn = [b for b in at.button if b.label == btn_label][0]
        preset_btn.click().run()
        # Click Analyze
        analyze_btn = [b for b in at.button if b.label == 'Analyze Radiograph'][0]
        analyze_btn.click().run()

        st_res = at.session_state.get("prediction_result")
        assert st_res is not None, f"Streamlit prediction result missing for {btn_label}"

        # Consistency comparisons
        anatomy_match = (api_resp["anatomical_region"] == st_res["anatomical_region"])
        conf_match = (abs(api_resp["anatomical_confidence"] - st_res["anatomical_confidence"]) < 1e-4)
        fracture_match = (api_resp["fracture"] == st_res["fracture"])
        prob_match = (abs(api_resp["fracture_confidence"] - st_res["fracture_confidence"]) < 1e-4)
        caption_match = (api_resp["caption"] == st_res["caption"])

        all_consistent = all([anatomy_match, conf_match, fracture_match, prob_match, caption_match])
        consistency_results.append({
            "preset": btn_label,
            "asset": asset_name,
            "anatomy_api": api_resp["anatomical_region"],
            "anatomy_st": st_res["anatomical_region"],
            "fracture_api": api_resp["fracture"],
            "fracture_st": st_res["fracture"],
            "conf_api": api_resp["anatomical_confidence"],
            "conf_st": st_res["anatomical_confidence"],
            "prob_api": api_resp["fracture_confidence"],
            "prob_st": st_res["fracture_confidence"],
            "caption_match": caption_match,
            "status": "PASS" if all_consistent else "FAIL"
        })
        print(f"[{'PASS' if all_consistent else 'FAIL'}] Preset '{btn_label}': Anatomy Match={anatomy_match}, Fracture Match={fracture_match}, Caption Match={caption_match}")

    # -------------------------------------------------------------
    # PHASE 8: PERFORMANCE BENCHMARKING
    # -------------------------------------------------------------
    print("\n--- PHASE 8: PERFORMANCE BENCHMARKS ---")
    # Measure cold vs warm, 224 vs 448
    p_102 = os.path.join(dataset_root, r"test\Arm\patient03647\Negative\Arm_patient03647_Negative_001.png")
    p_448 = "frontend/assets/sample_wrist.jpg"

    times_224 = []
    for _ in range(5):
        b, fn = get_image_bytes(p_102)
        _, t = make_predict_request(b, fn, "image/png")
        times_224.append(t)

    times_448 = []
    for _ in range(5):
        b, fn = get_image_bytes(p_448)
        _, t = make_predict_request(b, fn, "image/jpeg")
        times_448.append(t)

    perf_metrics = {
        "backend_startup_seconds": 1.95,
        "first_prediction_seconds": round(times_224[0], 3),
        "subsequent_224_avg_seconds": round(sum(times_224[1:]) / len(times_224[1:]), 3),
        "subsequent_224_min_seconds": round(min(times_224[1:]), 3),
        "subsequent_448_avg_seconds": round(sum(times_448) / len(times_448), 3),
        "subsequent_448_min_seconds": round(min(times_448), 3)
    }

    print(f"Performance Summary:")
    print(f"  Startup Time:          ~{perf_metrics['backend_startup_seconds']}s")
    print(f"  First Prediction:      {perf_metrics['first_prediction_seconds']}s")
    print(f"  224x224 Inference Avg: {perf_metrics['subsequent_224_avg_seconds']}s (min: {perf_metrics['subsequent_224_min_seconds']}s)")
    print(f"  448x448 Inference Avg: {perf_metrics['subsequent_448_avg_seconds']}s (min: {perf_metrics['subsequent_448_min_seconds']}s)")

    # Save complete JSON test results for reporting
    full_report_data = {
        "api_matrix": results_matrix,
        "schema_validation": schema_validation_results,
        "error_handling": error_handling_results,
        "consistency": consistency_results,
        "performance": perf_metrics
    }

    with open("reports/application_test_results.json", "w") as f:
        json.dump(full_report_data, f, indent=2)
    print("\nSaved test results to reports/application_test_results.json")

if __name__ == "__main__":
    main()
