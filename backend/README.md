# Bone Fracture Diagnostic Backend API

Production FastAPI service for deep-learning bone X-ray analysis, providing multi-task anatomical classification, fracture detection, spatial lesion localization, and factual natural-language captioning.

---

## 1. Quick Start

### Prerequisites
Ensure the project virtual environment is active and dependencies are installed:
```powershell
# From project root
.venv\Scripts\Activate.ps1
```

### Start the Server
Start the production server on port 8000 using Uvicorn:

```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

For hot-reloading during development:
```bash
uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```

Interactive OpenAPI documentation is automatically available at:
* **Swagger UI:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
* **ReDoc:** [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

---

## 2. API Endpoints

### `GET /health`
Liveness and readiness health probe.

**Response:**
```json
{
  "status": "ok"
}
```

---

### `POST /predict`
Analyzes an uploaded bone radiograph.

* **Content-Type:** `multipart/form-data`
* **Form Field:** `image` (binary file payload: PNG, JPG, BMP)

#### Example Request (`curl`)
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
     -F "image=@processed/test/Arm_patient04158_Negative_001.png"
```

#### Example Response (Normal Bone)
```json
{
  "success": true,
  "anatomical_region": "Lower leg",
  "anatomical_confidence": 0.544,
  "fracture": false,
  "fracture_confidence": 0.133,
  "localization_available": false,
  "localization": null,
  "caption": "X-ray of the lower leg with no fracture detected by the model.",
  "visualization_url": "/visualizations/pred_a1b2c3d4e5f6.png"
}
```

#### Example Response (Fracture Detected)
```json
{
  "success": true,
  "anatomical_region": "Thigh",
  "anatomical_confidence": 0.986,
  "fracture": true,
  "fracture_confidence": 0.924,
  "localization_available": false,
  "localization": null,
  "caption": "X-ray of the thigh showing a fracture.",
  "visualization_url": "/visualizations/pred_f7e8d9c0b1a2.png"
}
```

---

### `GET /visualizations/{filename}`
Retrieves the non-destructive annotated visual overlay with bounding boxes, confidence badges, and factual captions.

* **Path Traversal Protection:** Validates filenames to prevent unauthorized filesystem access.
* **Content-Type:** `image/png`

---

## 3. Architecture & Security Guardrails

1. **Singleton Model Loading:** Checkpoint weights (`best_model.pt` and `best_detector.pt`) are loaded once during application startup in the FastAPI lifespan handler, eliminating per-request loading latency.
2. **Inference Mode Execution:** All predictions execute within `torch.no_grad()` contexts.
3. **In-Memory Validation:** Uploaded bytes are validated using PIL integrity checks before passing to preprocessing.
4. **Client-Safe Error Handling:**
   - Corrupted or non-image files return `400 Bad Request` with client-safe messages.
   - Internal filesystem paths and stack traces are never exposed in API responses.
5. **Factual Caption Policy:** Captions strictly adhere to validated predictions, prohibiting hallucinated clinical attributes.
