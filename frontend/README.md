# Bone Fracture Detection & Image Captioning — Frontend UI

Modern, interactive Streamlit web application for bone X-ray fracture detection, anatomical classification, and factual clinical captioning.

---

## 1. Quick Start Guide

To run the complete system, launch the backend and frontend in **two separate terminal windows**.

### Terminal 1: Launch FastAPI Backend
```powershell
# In project root
.venv\Scripts\Activate.ps1

# Start FastAPI server on port 8000
uvicorn backend.main:app --host 127.0.0.1 --port 8000
```
*Verify backend health:* Open [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health) (returns `{"status": "ok"}`).

---

### Terminal 2: Launch Streamlit Frontend
```powershell
# In project root
.venv\Scripts\Activate.ps1

# Start Streamlit application
streamlit run frontend/app.py
```

The web dashboard will automatically open in your default browser at:  
👉 **[http://localhost:8501](http://localhost:8501)**

---

## 2. User Workflow & Features

1. **Upload Plain Radiograph:**
   - Supports PNG, JPG, JPEG, BMP, and WEBP formats.
   - Immediately renders the original radiograph preview upon selection.
2. **Execute Diagnosis ("Analyze X-ray"):**
   - Dispatches payload to `POST http://127.0.0.1:8000/predict`.
   - Displays real-time progress spinner during neural forward pass.
3. **Diagnostic Results Display:**
   - **Anatomical Region Card:** Predicted bone site (*wrist, pelvis, lower leg, thigh, arm, hand, foot*) with softmax confidence.
   - **Fracture Status Card:** High-visibility binary indicator (`FRACTURE DETECTED` vs `NO FRACTURE`) with calibrated probability.
   - **Spatial Lesion Localization:** Displays annotated visual bounding boxes when supported, or *"Fracture localization is not available for this prediction."*
   - **Generated Factual Caption:** Clinically disciplined description strictly bounded by validated model findings without clinical speculation.
4. **Interactive Controls:**
   - **Reset / New Image Button:** Instantly resets the session state to analyze subsequent radiographs.
   - **Raw JSON Inspector:** Expandable viewer for machine-readable verification during academic/college demonstrations.

---

## 3. Architecture Isolation Guardrails

* **Zero Duplicated Inference:** The frontend contains no model weights, no PyTorch forward loops, and no training code. All computation is handled by the FastAPI service.
* **Client-Safe Error Handling:** Network disruptions or corrupted uploads trigger clear, friendly notifications without exposing internal server stack traces.
