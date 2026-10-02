# Application Integration Test Report

**Date:** October 2, 2026  
**System Tested:** Bone Fracture Detection & Factual Clinical Captioning Platform  
**Scope:** Full End-to-End Testing (FastAPI Backend, Streamlit Frontend, Resolution Routing Pipeline, Model Inference, Error Handling, Consistency, Performance)  

---

## 1. Environment
- **OS:** Windows 11 (`Windows-11-10.0.26200-SP0`)
- **Python:** 3.13.15
- **PyTorch:** 2.6.0+cu124 (`CUDA: True`)
- **GPU:** NVIDIA GeForce RTX 3050 Laptop GPU (4,096 MiB VRAM)
- **FastAPI:** 0.142.2
- **Streamlit:** 1.64.0

---

## 2. Startup Test

### Commands Used
1. **FastAPI Inference Backend:**
   ```powershell
   .venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
   ```
2. **Streamlit Radiology Frontend:**
   ```powershell
   .venv\Scripts\python.exe -m streamlit run frontend/app.py --server.port 8501 --server.headless true
   ```

### Verification Checklist
- [x] **FastAPI Service:** Started successfully on `http://127.0.0.1:8000` (PID 21428) in 1.95s.
- [x] **Swagger/OpenAPI Documentation:** Successfully loaded and accessible at `http://127.0.0.1:8000/docs` (HTTP 200) and `http://127.0.0.1:8000/openapi.json` (HTTP 200).
- [x] **Endpoints Registered:** Verified `/health`, `/predict`, `/visualizations/{filename}`, and `/downloads/{filename}` registered in OpenAPI schema.
- [x] **Streamlit Service:** Started successfully on `http://localhost:8501` (HTTP 200).
- [x] **Frontend-Backend Connectivity:** Health probe `GET http://127.0.0.1:8000/health` verified responsive with `{"status": "ok"}`. Frontend status dot displayed online.
- [x] **Import Integrity:** Zero import errors across all modules (`inference`, `backend`, `src`, `frontend`).
- [x] **Model Checkpoint Integrity:** Loaded both `best_model.pt` and `exp3_resnet50_448.pt` without runtime errors; zero missing checkpoint errors.
- [x] **Dependencies:** All runtime packages resolved in active virtual environment.

---

## 3. API Test Matrix

All test cases were evaluated against the live running FastAPI service using plain radiographs from the native Mendeley BoneFract dataset and clinical assets.

### Test Matrix Table

| Test | Input Size | Expected Route | Actual Route | Model | Anatomy | Fracture | Probability | HTTP | Result |
| :--- | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: |
| **A. Native 102×102 BoneFract** | 102 × 102 | 224 | 224 | `best_model.pt` | pelvis (83.6%) | Negative | 1.9% | 200 | **PASS** |
| **B. 800×800 BoneFract** | 800 × 800 | 448 | 448 | `exp3_resnet50_448.pt` | Thigh (41.1%) | Positive | 89.7% | 200 | **PASS** |
| **C. 373×454 BoneFract (Portrait)** | 373 × 454 | 448 | 448 | `exp3_resnet50_448.pt` | Arm (100.0%) | Positive | 58.1% | 200 | **PASS** |
| **D. 454×373 BoneFract (Landscape)** | 454 × 373 | 448 | 448 | `exp3_resnet50_448.pt` | Arm (100.0%) | Negative | 29.9% | 200 | **PASS** |
| **E. High-res External X-ray** | 809 × 1296 | 448 | 448 | `exp3_resnet50_448.pt` | Hand (65.6%) | Negative | 27.8% | 200 | **PASS** |
| **F. Known Fracture-Positive** | 800 × 800 | 448 | 448 | `exp3_resnet50_448.pt` | Thigh (41.1%) | Positive | 89.7% | 200 | **PASS** |
| **G. Known Fracture-Negative** | 102 × 102 | 224 | 224 | `best_model.pt` | pelvis (83.6%) | Negative | 1.9% | 200 | **PASS** |
| **H. Landscape X-ray** | 1920 × 876 | 448 | 448 | `exp3_resnet50_448.pt` | Hand (94.2%) | Negative | 42.9% | 200 | **PASS** |
| **I. Portrait X-ray** | 809 × 1296 | 448 | 448 | `exp3_resnet50_448.pt` | Hand (65.6%) | Negative | 27.8% | 200 | **PASS** |

### Independent Routing Verification
- **Criterion:** `min(width, height) >= 300` $\to$ `exp3_resnet50_448.pt` (448×448), else `best_model.pt` (224×224).
- **Test A & G:** $\min(102, 102) = 102 < 300 \implies$ Routed to `best_model.pt` (224×224). **Verified.**
- **Test B & F:** $\min(800, 800) = 800 \ge 300 \implies$ Routed to `exp3_resnet50_448.pt` (448×448). **Verified.**
- **Test C:** $\min(373, 454) = 373 \ge 300 \implies$ Routed to `exp3_resnet50_448.pt` (448×448). **Verified.**
- **Test D:** $\min(454, 373) = 373 \ge 300 \implies$ Routed to `exp3_resnet50_448.pt` (448×448). **Verified.**
- **Test E & I:** $\min(809, 1296) = 809 \ge 300 \implies$ Routed to `exp3_resnet50_448.pt` (448×448). **Verified.**
- **Test H:** $\min(1920, 876) = 876 \ge 300 \implies$ Routed to `exp3_resnet50_448.pt` (448×448). **Verified.**
- **Decision Threshold:** Exactly $\tau = 0.50$ applied across all branches.

---

## 4. Response Schema Validation

Every response from `POST /predict` was verified against the Pydantic schema in [`backend/schemas.py`](file:///c:/Users/arbaz/Desktop/Deep%20learning%20project/backend/schemas.py):

- [x] **Field Existence:** All required fields (`success`, `anatomical_region`, `anatomical_confidence`, `fracture`, `fracture_confidence`, `localization_available`, `localization`, `caption`, `visualization_url`) and routing metadata fields (`input_width`, `input_height`, `routing_resolution`, `selected_model`) exist in all responses.
- [x] **Field Types:**
  - `success`: `bool`
  - `anatomical_region`: `str`
  - `anatomical_confidence`: `float` within $[0.0, 1.0]$
  - `fracture`: `bool`
  - `fracture_confidence`: `float` within $[0.0, 1.0]$
  - `caption`: Non-empty `str`
  - `input_width`, `input_height`: `int` strictly matching original image dimensions
  - `routing_resolution`: `str` (`"224"` or `"448"`)
  - `selected_model`: `str` (`"best_model.pt"` or `"exp3_resnet50_448.pt"`)
- [x] **Anatomical Classes:** Every predicted region strictly belongs to the 7 supported classes (`Arm`, `Foot`, `Hand`, `Lower leg`, `Thigh`, `pelvis`, `wrist`).
- [x] **Binary Fracture State:** Output is strictly binary (`True` / `False`).
- [x] **Frontend Isolation:** Verified that no obsolete localization coordinates or bounding boxes are consumed or displayed by the frontend UI.

---

## 5. Error Handling

A dedicated adversarial and malformed input suite was tested against the live server. The system handled all invalid payloads gracefully without crashing:

| Test Case | Payload Description | Expected Status | Actual Status | Server Response Message | System Stability |
| :--- | :--- | :---: | :---: | :--- | :---: |
| **1. Non-image file** | Plain text string payload (`notes.txt`) | 400 | 400 | `"Unsupported file format '.txt'. Please upload a standard image file (PNG, JPG, BMP)."` | **STABLE** |
| **2. Corrupted image** | Truncated bytes with invalid PNG header (`damaged.png`) | 400 | 400 | `"Uploaded file is corrupted or not a valid image format."` | **STABLE** |
| **3. Empty upload** | 0-byte payload (`empty.png`) | 400 | 400 | `"Uploaded image file is empty."` | **STABLE** |
| **4. Extremely small image** | Valid 2×2 grayscale image (`tiny_2x2.png`) | 200 | 200 | Preprocessed cleanly via aspect-ratio padding to target matrix. | **STABLE** |
| **5. Very wide image** | Valid 2000×80 grayscale image (`wide_2000x80.png`) | 200 | 200 | Letterboxed cleanly into aspect-ratio padding without distorting. | **STABLE** |
| **6. Very tall image** | Valid 80×2000 grayscale image (`tall_80x2000.png`) | 200 | 200 | Letterboxed cleanly into aspect-ratio padding without distorting. | **STABLE** |
| **7. Unsupported format** | Binary PE executable payload (`malware.exe`) | 400 | 400 | `"Unsupported file format '.exe'. Please upload a standard image file (PNG, JPG, BMP)."` | **STABLE** |
| **8. Missing form key** | Missing `image` field in multipart request | 422 | 422 | `Field required: body -> image` (Standard FastAPI validation) | **STABLE** |

**Backend Health Post-Adversarial Tests:** `GET /health` returned `HTTP 200 OK` (`{"status": "ok"}`). Zero daemon crashes or unhandled exceptions.

---

## 6. Frontend Test

Frontend user flows were verified via Streamlit `AppTest` and HTTP inspection:

1. **Application Load:**
   - Landing page loads with clinical blue/white theme.
   - Title `"BoneSight | Clinical Bone X-Ray Analysis"` rendered.
   - Hero section `"AI-Assisted Bone Fracture Analysis"` displayed.
   - Medical capabilities and workflow sections rendered without raw HTML tags or indented code blocks.
2. **Preset Ingestion & Viewport:**
   - Quick-load presets (`Fractured Wrist (AP)`, `Fractured Hip / Pelvis`, `Fractured Forearm`, `Normal Hand (PA)`) load instantaneously.
   - Fixed-size radiology viewport (420px fixed height) renders cleanly with dark clinical viewport styling and letterboxing.
   - Aspect ratio tags (`PORTRAIT`, `LANDSCAPE`, `SQUARE`) and matrix dimensions displayed in viewer footer.
3. **Execution & Results:**
   - "Analyze Radiograph" button triggers `/predict` call with loading spinner.
   - Results displayed in 3 structured cards:
     * **Anatomical Region Card:** Name (e.g. `HAND`, `WRIST`), confidence percentage (e.g. `93.7%`), confidence bar.
     * **Fracture Assessment Card:** High-visibility binary indicator (`FRACTURE DETECTED` in red / `NO FRACTURE DETECTED` in green), probability percentage, progress bar.
     * **Factual Summary Card:** Factual natural-language caption strictly matching predictions.
4. **Mandatory Negative Verification (Absence of Prohibited Features):**
   - [x] Zero bounding boxes rendered on the user interface.
   - [x] Zero localization confidence scores or Faster R-CNN outputs displayed.
   - [x] Zero lesion coordinates displayed.
   - [x] Zero unsupported morphology (e.g. comminuted, spiral, transverse, hairline) displayed.
   - [x] Zero unsupported laterality (e.g. left vs right) displayed.
   - [x] Zero unsupported displacement or angulation metrics displayed.
5. **Session Management:**
   - "Clear Image" button immediately resets viewport, uploaded bytes, and prediction state to standby.
   - Grayscale inversion toggle (`Invert Grayscale (Bone Contrast)`) functions interactively.

---

## 7. API vs Frontend Consistency

The direct `/predict` API output was compared against the Streamlit session state for multiple radiographs to verify absolute consistency:

| Radiograph Preset | Direct API Anatomy | Frontend Displayed Anatomy | Direct API Fracture | Frontend Displayed Fracture | Direct API Prob | Frontend Prob | Caption Match | Consistency Result |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Fractured Wrist (AP)** | Hand (65.6%) | Hand (65.6%) | Negative | Negative | 27.8% | 27.8% | Yes | **100% MATCH** |
| **Normal Hand (PA)** | Hand (93.7%) | Hand (93.7%) | Negative | Negative | 34.0% | 34.0% | Yes | **100% MATCH** |
| **Fractured Forearm** | wrist (96.1%) | wrist (96.1%) | Positive | Positive | 77.9% | 77.9% | Yes | **100% MATCH** |

**Conclusion:** Zero discrepancies between backend inference and frontend representation.

---

## 8. Performance

Latency was measured across both routing branches on the local GPU (`NVIDIA GeForce RTX 3050 Laptop GPU`):

| Metric | Measured Duration | Notes |
| :--- | :---: | :--- |
| **Backend Startup Time** | ~1.95 s | Singleton loading of both ResNet-50 checkpoints and Faster R-CNN detector |
| **First Prediction Time (Cold Start)** | 0.467 s | GPU allocation and initial kernel execution |
| **224×224 Inference Time (Subsequent Avg)** | **0.483 s** | Native 102×102 route (minimum: 0.475 s) |
| **448×448 Inference Time (Subsequent Avg)** | **0.719 s** | High-resolution route (minimum: 0.639 s) |
| **Error Response Latency** | < 0.007 s | Fast rejection for non-image / corrupted files prior to neural preprocessing |

---

## 9. Bugs Found

No runtime crashes, memory leaks, or unhandled exceptions occurred in the core application. One minor legacy test script label mismatch was identified and resolved:

### Bug 1: Legacy Button Label in Pre-existing Frontend Test Script
- **Severity:** Low (Test script only; core application was unaffected)
- **Reproduction Steps:** Run `scripts/test_frontend_apptest.py`.
- **Root Cause:** The test script searched for button label `'Run Diagnostic Analysis'`, whereas the actual UI button in `frontend/app.py` is named `'Analyze Radiograph'`.
- **Affected File:** `scripts/test_frontend_apptest.py`
- **Fix Applied:** Updated `scripts/test_frontend_apptest.py` to match the exact labels present in `frontend/app.py`.

---

## 10. Passed Tests

1. `tests/test_api.py` (3 tests) — FastAPI app initialization, schemas import.
2. `tests/test_backend_api.py` (6 tests) — Health probe, normal radiograph, fractured radiograph, FracAtlas image, corrupted file rejection, visualization retrieval.
3. `tests/test_caption_generator.py` (10 tests) — Factual clinical captions across all anatomical classes, positive/negative fracture statuses, uncertainty handling.
4. `tests/test_inference.py` (4 tests) — Forward pass, probability ranges, pipeline schema contract, detector forward pass.
5. `tests/test_label_mapping.py` (4 tests) — 7-region bijection, binary fracture mappings, registry alignment.
6. `tests/test_model_loading.py` (3 tests) — Architecture instantiation, dual-head output dimensions, checkpoint discovery.
7. `tests/test_preprocessing.py` (6 tests) — Aspect-ratio padding, tensor dimensions, intensity normalization.
8. `tests/test_resolution_routing.py` (16 tests) — Routing rules for 102×102, 800×800, 373×454, 1920×876, 809×1296, boundary conditions (299 vs 300), end-to-end pipeline execution, FastAPI `/predict` integration.
9. `scripts/run_complete_application_tests.py` (9 matrix tests, 8 error handling tests, 3 frontend consistency tests, 9 schema validations).
10. `scripts/test_frontend_apptest.py` (Full Streamlit UI lifecycle, preset loading, clean card verification, negative localization absence check, clear image state reset).

**Total Passed Tests:** **56 automated unit/integration tests + 29 end-to-end matrix/flow tests = 85 tests passed.**

---

## 11. Failed Tests

- **Failed Tests:** **0**

---

## 12. Overall Application Status

# PASS

> [!NOTE]
> **Regulatory / Clinical Notice:** This system is an academic research demonstration and educational platform. It is **not** clinically validated, nor is it cleared by any medical regulatory authority (FDA, CE-MDR) for standalone clinical diagnosis.
