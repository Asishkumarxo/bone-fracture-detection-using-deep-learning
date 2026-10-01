# REST API Documentation

The Bone Fracture AI service provides a production-grade FastAPI backend service.

## Server Execution

```bash
uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```

Interactive OpenAPI documentation is available at `http://127.0.0.1:8000/docs`.

---

## Endpoints

### 1. Health Probe
* **Method:** `GET`
* **Path:** `/health`
* **Description:** Liveness and readiness probe for container orchestrators and monitoring.
* **Response (200 OK):**
```json
{
  "status": "ok"
}
```

---

### 2. Predict Fracture
* **Method:** `POST`
* **Path:** `/predict`
* **Content-Type:** `multipart/form-data`
* **Parameters:**
  * `image` (UploadFile, required): Bone radiograph image (PNG, JPG, JPEG, BMP).
* **Response (200 OK):**
```json
{
  "success": true,
  "anatomical_region": "Lower leg",
  "anatomical_confidence": 0.884,
  "fracture": true,
  "fracture_confidence": 0.912,
  "localization_available": true,
  "localization": [
    {
      "box": [120.5, 340.0, 210.0, 430.5],
      "box_2d": [120.5, 340.0, 210.0, 430.5],
      "confidence": 0.784
    }
  ],
  "caption": "X-ray of the lower leg showing a fracture.",
  "visualization_url": "/visualizations/pred_a1b2c3d4e5f6.png"
}
```
* **Error Responses:**
  * `400 Bad Request`: Corrupted image or unsupported file extension.
  * `503 Service Unavailable`: Models initializing or checkpoint unavailable.

---

### 3. Retrieve Visualization
* **Method:** `GET`
* **Path:** `/visualizations/{filename}`
* **Description:** Safely returns the annotated visual overlay image. Protected against directory traversal attacks.
