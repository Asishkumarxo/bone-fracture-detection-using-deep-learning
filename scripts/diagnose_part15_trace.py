"""
Part 15: End-to-End Pipeline Trace Diagnostic
=============================================
Traces the exact flow of localization data through all 5 system layers:
  1. MODEL: Raw PyTorch Faster R-CNN inference
  2. PREDICT: predict.py pipeline contracts
  3. FASTAPI: REST API POST /predict JSON response
  4. STREAMLIT: Frontend state consumption logic
  5. VISUALIZATION: Static image overlay generation & retrieval
Logs the localization object at each stage to reports/pipeline_trace_log.json
"""

import os
import sys
import json
import requests
import torch
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from inference.model_registry import ModelRegistry
from inference.localization_predictor import LocalizationPredictor
from inference.preprocessing import preprocess_for_detector
from predict import BoneFractureInferencePipeline
from backend.services.inference_service import InferenceService

def main():
    # Pick a fractured image with known detection: IMG0000453.jpg
    test_img = "E:/FracAtlas/images/Fractured/IMG0000453.jpg"
    if not os.path.exists(test_img):
        print(f"Error: {test_img} not found.")
        return

    print("="*65)
    print("      PART 15: 5-STAGE END-TO-END PIPELINE TRACE       ")
    print("="*65)
    print(f"Input Radiograph: {test_img}\n")

    trace_log = {}

    # -------------------------------------------------------------
    # STAGE 1: RAW MODEL OUTPUT
    # -------------------------------------------------------------
    print(">>> STAGE 1: MODEL (Faster R-CNN MobileNetV3 FPN)")
    device = ModelRegistry.get_device()
    detector = LocalizationPredictor(score_thresh=0.10, device=device)
    tensor, (w, h) = preprocess_for_detector(test_img)
    tensor = tensor.to(device)

    with torch.no_grad():
        raw_pred = detector.model(tensor)[0]
        raw_boxes = raw_pred['boxes'].cpu().numpy()
        raw_scores = raw_pred['scores'].cpu().numpy()
        raw_labels = raw_pred['labels'].cpu().numpy()

    stage1_obj = {
        "raw_proposals_count": len(raw_scores),
        "top_scores": [float(round(s, 4)) for s in raw_scores[:5]],
        "top_boxes": [b.round(1).tolist() for b in raw_boxes[:5]],
        "top_labels": [int(l) for l in raw_labels[:5]]
    }
    print(json.dumps(stage1_obj, indent=2))
    trace_log["stage1_model"] = stage1_obj

    # -------------------------------------------------------------
    # STAGE 2: PREDICT.PY PIPELINE
    # -------------------------------------------------------------
    print("\n>>> STAGE 2: PREDICT.PY (BoneFractureInferencePipeline)")
    pipeline = BoneFractureInferencePipeline()
    predict_res = pipeline.analyze(test_img)
    
    stage2_obj = {
        "anatomical_region": predict_res["anatomical_region"],
        "anatomical_confidence": predict_res["anatomical_confidence"],
        "fracture": predict_res["fracture"],
        "fracture_confidence": predict_res["fracture_confidence"],
        "localization_available": predict_res["localization_available"],
        "localization": predict_res["localization"],
        "caption": predict_res["caption"]
    }
    print(json.dumps(stage2_obj, indent=2))
    trace_log["stage2_predict_py"] = stage2_obj

    # -------------------------------------------------------------
    # STAGE 3: FASTAPI JSON RESPONSE
    # -------------------------------------------------------------
    print("\n>>> STAGE 3: FASTAPI REST API (POST /predict)")
    backend_url = "http://127.0.0.1:8000"
    stage3_obj = {}
    try:
        with open(test_img, "rb") as f:
            resp = requests.post(f"{backend_url}/predict", files={"image": ("xray.jpg", f, "image/jpeg")}, timeout=30.0)
        
        stage3_obj["http_status"] = resp.status_code
        stage3_obj["response_json"] = resp.json()
        print(f"HTTP Status: {resp.status_code}")
        print(json.dumps(resp.json(), indent=2))
    except Exception as e:
        stage3_obj["error"] = str(e)
        print(f"FastAPI request failed: {e}")
    trace_log["stage3_fastapi"] = stage3_obj

    # -------------------------------------------------------------
    # STAGE 4: STREAMLIT FRONTEND DATA BINDING
    # -------------------------------------------------------------
    print("\n>>> STAGE 4: STREAMLIT FRONTEND DATA BINDING")
    api_json = stage3_obj.get("response_json", {})
    is_fracture = api_json.get("fracture", False)
    loc_avail = api_json.get("localization_available", False)
    loc_boxes = api_json.get("localization", None)
    vis_url = api_json.get("visualization_url", None)

    stage4_obj = {
        "received_is_fracture": is_fracture,
        "received_loc_available": loc_avail,
        "received_loc_count": len(loc_boxes) if loc_boxes else 0,
        "frontend_branch": (
            "Render bounding boxes & overlay" if (is_fracture and loc_avail and loc_boxes)
            else "Fracture localization is not available for this prediction." if is_fracture
            else "Normal bone structure — not applicable"
        ),
        "visualization_url": vis_url
    }
    print(json.dumps(stage4_obj, indent=2))
    trace_log["stage4_streamlit"] = stage4_obj

    # -------------------------------------------------------------
    # STAGE 5: VISUALIZATION RETRIEVAL
    # -------------------------------------------------------------
    print("\n>>> STAGE 5: VISUALIZATION RETRIEVAL")
    stage5_obj = {}
    if vis_url:
        full_vis_url = f"{backend_url}{vis_url}"
        try:
            v_resp = requests.get(full_vis_url, timeout=10.0)
            stage5_obj["vis_http_status"] = v_resp.status_code
            stage5_obj["content_type"] = v_resp.headers.get("content-type")
            stage5_obj["image_size_bytes"] = len(v_resp.content)
            print(f"GET {full_vis_url} -> Status {v_resp.status_code}, Bytes: {len(v_resp.content):,}")
        except Exception as e:
            stage5_obj["error"] = str(e)
            print(f"Visualization fetch failed: {e}")
    else:
        stage5_obj["status"] = "No visualization URL generated because localization was unavailable or not requested"
        print(stage5_obj["status"])
    trace_log["stage5_visualization"] = stage5_obj

    # Save complete trace log
    os.makedirs('reports', exist_ok=True)
    trace_file = 'reports/pipeline_trace_log.json'
    with open(trace_file, 'w') as f:
        json.dump(trace_log, f, indent=2)
    print(f"\nTrace completed! Saved complete log to {trace_file}")

if __name__ == '__main__':
    main()
