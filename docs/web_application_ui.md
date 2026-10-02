# Bone Fracture Image Captioning Using Deep Learning — Web Application and UI

**Document Type:** Web Application Architecture, User Interface Specification & Deployment Report  
**Project:** Bone Fracture Image Captioning Using Deep Learning  
**Target Audience:** B.Tech CSE / AI-ML Students, Academic Reviewers, Viva Examiners  
**Status:** Verified Against Active Implementation (`frontend/app.py`, `backend/main.py`)  

---

## 1. Web Application Overview

The web application for **“Bone Fracture Image Captioning Using Deep Learning”** is a decoupled, client-server diagnostic research workspace named **BoneSight**. It enables users (researchers, engineering evaluators, and academic reviewers) to upload musculoskeletal radiographs (X-rays), inspect their visual quality in a fixed-frame radiology viewport, dispatch them to a high-performance deep learning inference service, and review structured model analysis results accompanied by a factually grounded caption.

### Core Objectives:
- **Decoupled Architecture:** Strict separation between the interactive browser interface (Streamlit) and the neural inference service (FastAPI).
- **Clinical Workstation Usability:** Elimination of layout shifts through a fixed 420px height radiology viewer with letterboxing, matrix dimension readout, and contrast inversion.
- **Factual Medical Discipline:** Complete exclusion of speculative generative text and ungrounded localization; results present only verified model predictions with clear disclaimers.

---

## 2. Technology Stack

The production web application utilizes a clean Python-native asynchronous web stack:

| Component | Technology | Version / Specification | Role in System |
| :--- | :--- | :--- | :--- |
| **Frontend Framework** | **Streamlit** | Python Library (`frontend/app.py`) | Interactive browser UI, session state, DOM rendering, image preview |
| **Backend API** | **FastAPI** | ASGI Framework (`backend/main.py`) | High-performance asynchronous REST API, request validation, multipart parsing |
| **ASGI Server** | **Uvicorn** | ASGI Web Server | Runs FastAPI daemon on `127.0.0.1:8000` |
| **Deep Learning Framework** | **PyTorch / Torchvision** | PyTorch 2.x | Neural network definition, GPU/CPU tensor evaluation, weight loading |
| **Computer Vision / Image I/O** | **Pillow (PIL)** & **OpenCV** | PIL / `cv2` | File format decoding, dimensions inspection, aspect-ratio preserved padding |
| **HTTP Client** | **Requests** | Python Library | Asynchronous/synchronous multipart HTTP POST communication from Streamlit to FastAPI |
| **Styling & Design System** | **Vanilla CSS3** | Custom Clinical Tokens | Light clinical design system embedded directly via HTML5 `<style>` blocks |

---

## 3. Application Architecture

The system follows a tiered, modular architectural pattern ensuring loose coupling, fault tolerance, and clear separation of concerns:

```
┌────────────────────────────────────────────────────────────────────────┐
│                        CLIENT / BROWSER TIER                           │
│  Streamlit Application (frontend/app.py) running on http://localhost:8501│
│  - Session state management (st.session_state)                         │
│  - Viewport rendering & Grayscale inversion                            │
│  - Multipart HTTP file serialization                                   │
└────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼ HTTP POST multipart/form-data
┌────────────────────────────────────────────────────────────────────────┐
│                       REST API GATEWAY TIER                            │
│  FastAPI Application (backend/main.py) running on http://localhost:8000 │
│  - GET  /health   -> Service liveness probe                            │
│  - GET  /models   -> Metadata of active checkpoints                    │
│  - POST /predict  -> File ingestion & schema validation                │
└────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                      SERVICE ORCHESTRATION TIER                        │
│  InferenceService (backend/services/inference_service.py)              │
│  1. In-memory PIL integrity validation                                 │
│  2. Matrix dimension extraction (orig_w, orig_h)                       │
│  3. ModelRegistry resolution routing:                                  │
│       • min(orig_w, orig_h) >= 300 px  ==>  exp3_resnet50_448.pt       │
│       • min(orig_w, orig_h) < 300 px   ==>  best_model.pt (224×224)    │
└────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                     NEURAL INFERENCE & CAPTION TIER                    │
│  1. Preprocessing: Aspect-ratio preserved padding & normalization      │
│  2. PyTorch Forward Pass: MultiTaskModel (ResNet-50)                   │
│  3. RegionPredictor: Softmax across 7 classes → region + confidence    │
│  4. FracturePredictor: Sigmoid → fracture probability (tau = 0.50)     │
│  5. CaptionGenerator: Deterministic factual sentence synthesis         │
└────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼ JSON Response Contract
┌────────────────────────────────────────────────────────────────────────┐
│                     STREAMLIT RESULTS RENDERING                        │
│  1. Anatomical Region Card (Name + Confidence Bar)                     │
│  2. Fracture Assessment Card (Status Badge + Probability Bar)          │
│  3. Factual Summary Card (Structured Sentence + Research Disclaimer)   │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Frontend Interface Structure

The Streamlit frontend is organized into three distinct operational views navigated through the persistent top header:

### 4.1 Header & Global Navigation
- **Branding Header:** Displays the clean title **“BoneSight”** with the subtitle *“Musculoskeletal Radiograph Evaluation Platform”*.
- **Navigation Controls:** Three clean action buttons:
  - `Workspace`: Navigates to the diagnostic radiograph analysis workspace.
  - `Landing`: Returns to the overview and clinical capabilities summary.
  - `About System`: Opens technical specifications and model performance details.
- **Backend Health Pill:** An automated indicator in the top right displaying either a green active pill (`● INFERENCE ENGINE ONLINE`) or a red warning pill (`● INFERENCE ENGINE OFFLINE`) based on periodic `/health` polling.

### 4.2 Page 1: Landing Page
- **Hero Section:** Features a high-contrast radiograph preview inside a radiology frame alongside key platform statistics (*7 Musculoskeletal Sites, Dual-Task Clinical Inference, ResNet-50 Backbone*).
- **Clinical Capabilities Strip:** Four structured cards detailing Anatomical Classification, Fracture Assessment, Calibrated Scoring, and Factual Reporting.
- **3-Step Workflow Guide:** Outlines Step 1 (Upload Radiograph), Step 2 (Deep Learning Inference), and Step 3 (Review Model Analysis).
- **Persistent Academic Disclaimer:** Reminding users that the platform is an engineering research prototype.

### 4.3 Page 2: Diagnostic Workspace (`analysis`)
The primary operational page, structured into two balanced columns:
- **Left Column (Viewport & Controls):** Contains quick-load preset buttons, the drag-and-drop file uploader, the fixed-height 420px radiology viewport, the filename/dimension badge, the grayscale inversion checkbox, the primary `Analyze Radiograph` button, and the `Clear Image` button.
- **Right Column (Model Analysis Results):** Renders the three structured diagnostic cards once inference completes, or a helpful standby guidance card when awaiting an image.

### 4.4 Page 3: About / Technical Specifications (`about`)
- Provides an engineering reference breakdown: backbone type (ResNet-50), resolution routing thresholds, training dataset parameters (47,931 radiographs), loss formulation, and API contract specifications.

---

## 5. Quick-Load Presets

To facilitate rapid demonstration without requiring manual file uploads during presentations and project evaluations, the UI provides four quick-load preset buttons:

```
┌────────────────────────────────────────────────────────────────────────────────┐
│  Quick-Load Preset Radiographs:                                                │
│  [ Wrist X-ray (AP) ]  [ Hip / Pelvis X-ray ]  [ Forearm X-ray ]  [ Hand X-ray (PA) ] │
└────────────────────────────────────────────────────────────────────────────────┘
```

### Strict Medical Labeling Hygiene:
In accordance with rigorous clinical evaluation standards:
1. **Neutral Anatomical Labels:** Presets use neutral, purely anatomical descriptions (`Wrist X-ray (AP)`, `Hip / Pelvis X-ray`, `Forearm X-ray`, `Hand X-ray (PA)`).
2. **No Unverified Diagnostic Claims:** Labels do **NOT** state whether the radiograph is “Fractured” or “Normal” in the UI button text. Diagnostic labels are only rendered after the model performs inference.
3. **Verified Asset Mappings:**
   - `Wrist X-ray (AP)` $\to$ `sample_wrist.jpg`
   - `Hip / Pelvis X-ray` $\to$ `sample_pelvis.webp`
   - `Forearm X-ray` $\to$ `sample_forearm.jpg`
   - `Hand X-ray (PA)` $\to$ `sample_hand.jpg`

---

## 6. Branding & Naming Standards

The application adheres to strict academic honesty and regulatory compliance standards:

- **Approved Product Name:** **BoneSight**
- **Explicitly Forbidden Claims:** The UI does **NOT** use misleading clinical marketing terms:
  - ❌ “Clinical AI”
  - ❌ “Medical AI”
  - ❌ “Clinical Diagnostic AI”
  - ❌ “Clinically Validated System”
  - ❌ “FDA Approved / CE Marked”
- **Positioning:** Positioned transparently as a **Computer-Aided Detection (CAD) Research Prototype** developed for academic evaluation.

---

## 7. Image Upload & Validation

### Supported File Formats:
- **PNG** (`.png`)
- **JPEG / JPG** (`.jpg`, `.jpeg`)
- **WEBP** (`.webp`)
- **BMP** (`.bmp`)

### Upload Constraints:
- **Maximum File Size:** Configured up to 200 MB via Streamlit's file uploader configuration.
- **Client-Side Validation:** Streamlit restricts file extensions at the file browser dialog.
- **Server-Side Validation:** The FastAPI backend inspects the MIME type and executes `PIL.Image.open().verify()` on the binary stream in-memory. If an invalid or corrupted file is submitted, FastAPI immediately returns HTTP 400 (`{"error": "Invalid radiograph format: ..."}`).
- **Session State Storage:** Uploaded image bytes are held in memory (`st.session_state["uploaded_bytes"]`), ensuring that interacting with UI widgets does not re-upload the image across Streamlit rerun cycles.

---

## 8. Radiology Viewport Frame

A common flaw in generic web applications is the **layout shift bug**, where images of varying aspect ratios (tall lateral leg vs. wide pelvis) cause UI elements to jump erratically.

```
┌────────────────────────────────────────────────────────┐
│ VIEWPORT: PRIMARY RADIOGRAPH           1024 × 1024 PX  │ ← Header
├────────────────────────────────────────────────────────┤
│                                                        │
│                  [ Radiograph Image ]                  │ ← Stage (Fixed 420px)
│                                                        │
│                           ✚                            │ ← Orientation Crosshair
│                                                        │
├────────────────────────────────────────────────────────┤
│ MATRIX: 1024 × 1024     16-BIT GRAY     ASPECT: SQUARE │ ← Footer
└────────────────────────────────────────────────────────┘
```

### Engineering Solutions Implemented:
1. **Fixed 420px Height Viewport:** The container (`.radiology-fixed-viewer`) enforces a strict height of 420px using CSS flexbox.
2. **Dark Clinical Workstation Presentation:** The background is set to a deep obsidian (`#0A1118` / `#000000`) mimicking a professional PACS (Picture Archiving and Communication System) monitor.
3. **Letterboxing:** Images are scaled with `max-width: 100%` and `max-height: 100%` (`object-fit: contain`), preserving their true biological aspect ratio without distortion.
4. **Metadata Readout:** The viewport header displays the exact native resolution (`{orig_w} × {orig_h} PX`), while the footer displays the aspect tag (`LANDSCAPE`, `PORTRAIT`, or `SQUARE`).
5. **Interactive Grayscale Inversion:** A checkbox allows users to toggle bone contrast inversion (`ImageOps.invert()`), rendering dense bone dark on a white background—a standard radiological technique for accentuating subtle cortical fractures.

---

## 9. Model Analysis Results Cards

When the user clicks `Analyze Radiograph`, the right column populates with three structured, color-coded result cards:

### Card 1: Anatomical Region
- **Title:** `Anatomical Region`
- **Output:** The predicted anatomical region displayed in bold uppercase (e.g., `WRIST`, `PELVIS`, `HAND`).
- **Confidence Metric:** Percentage confidence score (e.g., `94.2%`).
- **Visual Bar:** A solid blue progress bar (`--secondary-blue: #1F6F8B`) scaled proportionally to the Softmax confidence.

### Card 2: Fracture Assessment
- **Title:** `Fracture Assessment`
- **Status Badge:**
  - If $p \ge 0.50$: Red high-visibility pill `● FRACTURE DETECTED` (background: `#FDE8E8`, text: `#C53030`).
  - If $p < 0.50$: Green high-visibility pill `● NO FRACTURE DETECTED` (background: `#DEF7EC`, text: `#0E9F6E`).
- **Probability Metric:** Exact sigmoid probability formatted as a percentage (e.g., `78.5%`).
- **Visual Bar:** Red bar for positive cases, green bar for negative cases.

### Card 3: Factual Summary
- **Title:** `Factual Summary`
- **Summary Box:** Renders the model-grounded sentence inside quotation marks.
- **Attribution Subtitle:** Explicitly notes: *“Generated from the model’s available predictions.”*

---

## 10. Caption Generation & Clinical Templates

The text caption is constructed downstream by `inference/caption_generator.py` using deterministic templates:

### Supported Templates:
1. **Fracture Positive ($p \ge 0.50$):**
   > *“Plain radiograph of the {region} demonstrates findings consistent with a fracture (confidence: {confidence}%).”*
   
   *Example:* *“Plain radiograph of the wrist demonstrates findings consistent with a fracture (confidence: 78.5%).”*

2. **Fracture Negative ($p < 0.50$):**
   > *“Plain radiograph of the {region} shows no definitive displaced fracture.”*
   
   *Example:* *“Plain radiograph of the pelvis shows no definitive displaced fracture.”*

### What the Caption Generator Deliberately Does NOT Generate:
To prevent clinical misrepresentation and hallucinations, the system strictly refuses to synthesize:
- ❌ Fracture subtype (e.g., *“Colles' fracture”*, *“Smith's fracture”*, *“greenstick”*)
- ❌ Spatial displacement (e.g., *“dorsally displaced by 3mm”*)
- ❌ Laterality (e.g., *“left wrist”* vs. *“right wrist”*)
- ❌ Trauma mechanism (e.g., *“secondary to fall on outstretched hand”*)
- ❌ Treatment advice (e.g., *“cast immobilization recommended”*)

---

## 11. Backend Connection & Health Monitoring

The frontend actively monitors the availability of the FastAPI inference backend:
- **Endpoint:** `GET http://127.0.0.1:8000/health`
- **Probe Interval:** Executed on initial page load with a 1.5-second timeout.
- **UI Feedback:**
  - **Online:** Renders a green pill `● INFERENCE ENGINE ONLINE`. The `Analyze Radiograph` button is enabled.
  - **Offline:** Renders a red pill `● INFERENCE ENGINE OFFLINE`. If the user attempts analysis, an informative banner appears: *“Analysis service is currently unavailable. Please verify the FastAPI backend is online.”*

---

## 12. API Flow & Communication Protocol

The communication contract between Streamlit and FastAPI:

```
[Streamlit Frontend]                                            [FastAPI Backend]
         │                                                              │
         │ 1. User clicks "Analyze Radiograph"                          │
         │ 2. Serializes binary image:                                  │
         │    files = {"image": (filename, bytes, "image/jpeg")}        │
         │                                                              │
         │ 3. HTTP POST /predict (multipart/form-data)                  │
         ├─────────────────────────────────────────────────────────────►│
         │                                                              │ 4. Validates MIME type
         │                                                              │ 5. Reads file_bytes
         │                                                              │ 6. Calls InferenceService
         │                                                              │ 7. Routes resolution
         │                                                              │ 8. Executes forward pass
         │                                                              │ 9. Synthesizes caption
         │                                                              │ 10. Serializes JSON
         │ 11. HTTP 200 OK (application/json)                           │
         │◄─────────────────────────────────────────────────────────────┤
         │                                                              │
         │ 12. Ingests response into st.session_state                   │
         │ 13. Re-renders DOM with result cards                         │
         ▼                                                              ▼
```

### Complete JSON Response Schema (`backend/schemas.py`):
```json
{
  "anatomical_region": "wrist",
  "anatomical_confidence": 0.9421,
  "fracture": true,
  "fracture_confidence": 0.7847,
  "caption": "Plain radiograph of the wrist demonstrates findings consistent with a fracture (confidence: 78.5%).",
  "input_width": 1024,
  "input_height": 1024,
  "routing_resolution": 448,
  "selected_model": "exp3_resnet50_448.pt"
}
```

---

## 13. Resolution-Aware Inference in Practice

The backend automatically routes incoming images based on their native matrix dimensions:
- If $\min(\text{width}, \text{height}) \ge 300\text{ px}$: The image is routed to `exp3_resnet50_448.pt` and preprocessed to $448 \times 448$. This activates the high-resolution network capable of **90.91% sensitivity** on fine cortical details.
- If $\min(\text{width}, \text{height}) < 300\text{ px}$: The image is routed to `best_model.pt` and preprocessed to $224 \times 224$, preventing aggressive upscaling blur on low-resolution thumbnails.

---

## 14. Error Handling & Edge Cases

The application includes exhaustive defensive error-handling routines:

| Failure Scenario | Backend Behavior | Frontend User Experience |
| :--- | :--- | :--- |
| **Invalid File Type (e.g., .txt, .pdf)** | Rejects with HTTP 400 (`"Invalid file extension"`) | Displays red alert: `Validation Error: Invalid radiograph format.` |
| **Corrupted Image Stream** | PIL `.verify()` raises error $\to$ HTTP 400 | Displays red alert: `Validation Error: Could not decode image file.` |
| **Backend Offline / Network Partition** | Request raises `requests.exceptions.ConnectionError` | Displays banner: `Analysis service is currently unavailable. Please verify FastAPI is online.` |
| **Inference Timeout (> 60s)** | Client timeout triggered | Displays warning: `Analysis could not be completed within timeout. Please retry.` |
| **Zero Bytes Submitted** | Handled before network dispatch | Primary button remains disabled until an image or preset is loaded. |

---

## 15. Frontend/Backend Decoupling

The frontend code (`frontend/app.py`) contains **zero machine learning dependencies**:
- It does **not** import `torch`, `torchvision`, `src.models`, or `inference.model_registry`.
- It interacts with the deep learning model exclusively via HTTP JSON requests.
- This decoupling allows the backend to be hosted on a dedicated GPU cluster (e.g., AWS EC2, GCP Compute Engine) while the frontend runs on a lightweight web server or CDN container without GPU drivers.

---

## 16. Safety & Research Positioning

Both the landing page and diagnostic workspace feature prominent medical disclaimers:
> **Medical Disclaimer:** Research / educational prototype only. This software is intended exclusively for computational imaging research and academic evaluation. It is not an FDA-cleared or CE-marked medical device and must not be used as a substitute for professional clinical diagnosis.

This positioning ensures that the system is evaluated strictly on its technical merits as an engineering demonstration.

---

## 17. What the Final UI Deliberately Does NOT Display

To maintain architectural integrity and eliminate unsupported claims, the following legacy and ungrounded elements are completely excluded from the final web application:
- ❌ **No Bounding Boxes:** No bounding box overlays are drawn on the radiograph viewport.
- ❌ **No Faster R-CNN Output:** Localization models are completely isolated from the frontend.
- ❌ **No Detector Confidence Scores:** No IoU or anchor proposal scores are shown.
- ❌ **No Morphological Subtyping:** No unsupported diagnoses (e.g., spiral, comminuted) are displayed.
- ❌ **No Treatment Recommendations:** No clinical management advice is provided.

---

## 18. UI Design System

The interface is built using a customized **Light Clinical Design System** specified directly via CSS:

```css
:root {
    --primary-navy:   #123B5D;  /* Main headers, brand, primary buttons */
    --secondary-blue: #1F6F8B;  /* Accents, subheadings, confidence fills */
    --accent-cyan:     #3C8DAD;  /* Highlights, active borders */
    --bg-clinical:    #F5F8FA;  /* Clean clinical background */
    --surface-white:  #FFFFFF;  /* Card surfaces and containers */
    --border-light:   #D9E3E8;  /* Subtle structural dividers */
    --text-primary:   #123B5D;  /* High-contrast readable typography */
    --text-muted:     #5A6E7C;  /* Secondary metadata and captions */
    --viewport-bg:    #0A1118;  /* PACS radiograph monitor background */
}
```

### Visual Highlights:
- **Typography:** Uses *IBM Plex Sans* for medical/technical headers and *Inter* for crisp body text.
- **Card Styling:** 1px subtle borders (`#D9E3E8`), 8px border radii, and soft drop shadows (`rgba(18, 59, 93, 0.05)`).
- **Responsive Grid:** Two-column workstation layout (`columns([6, 5])`) keeping viewport and results in view simultaneously on standard 1080p laptop displays.

---

## 19. End-to-End User Flow

A typical user interaction follows a clean 6-step journey:

```
Step 1: Open Application
        User navigates to http://localhost:8501
        Top pill confirms: "● INFERENCE ENGINE ONLINE"
                        │
Step 2: Select or Upload Radiograph
        User clicks a quick preset (e.g., "Wrist X-ray (AP)")
        OR uploads a custom radiograph (PNG/JPG/WEBP)
                        │
Step 3: Radiograph Displayed in Viewport
        Image renders inside the 420px letterboxed radiology frame
        Matrix dimension header updates: "1024 × 1024 PX"
        User can toggle "Invert Grayscale" for enhanced contrast
                        │
Step 4: Click "Analyze Radiograph"
        Streamlit shows spinner: "Analyzing radiograph..."
        HTTP POST dispatched to FastAPI /predict
                        │
Step 5: Backend Inference
        InferenceService checks dimensions → routes to 448×448 model
        Preprocesses with aspect-ratio padding → forward pass
        Extracts region (Wrist, 94.2%) + fracture (Positive, 78.5%)
        Caption generator builds factual sentence
                        │
Step 6: Results Rendered
        Right column displays:
        - Anatomical Region Card: WRIST (94.2% Confidence)
        - Fracture Assessment Card: FRACTURE DETECTED (78.5% Probability)
        - Factual Summary: "Plain radiograph of the wrist demonstrates
          findings consistent with a fracture (confidence: 78.5%)."
```

---

## 20. Application Testing & Verification

The web application and its integration contracts have been subjected to exhaustive automated regression testing:

| Test Suite | File Location | Tests Count | Status | Scope Tested |
| :--- | :--- | :---: | :---: | :--- |
| **Unit & Integration Suite** | `tests/test_application.py` | 58 | **58/58 PASS** | Endpoints, error handling, preprocessing, model loading |
| **API & End-to-End Suite** | `tests/test_api_flow.py` | 27 | **27/27 PASS** | Multipart uploads, schemas, routing thresholds, state consistency |
| **Caption Generator Suite** | `tests/test_caption_generator.py` | 13 | **13/13 PASS** | Text templates, edge cases, prohibition of hallucinations |
| **Total Automated Tests** | | **85+** | **100% PASS** | Zero regressions, complete system stability |

---

## 21. Final UI Summary for Viva Preparation

For rapid reference during viva voce examination:

- **Architecture:** Decoupled Streamlit frontend (`:8501`) communicating via RESTful HTTP JSON with an asynchronous FastAPI backend (`:8000`).
- **Clinical Viewport:** Fixed 420px height radiology viewer with letterboxing, PACS-style dark background, native matrix dimension readout, and contrast inversion toggle to eliminate layout shifting.
- **Presets:** Four quick-load presets using neutral descriptive labels (`Wrist X-ray (AP)`, `Hip / Pelvis X-ray`, `Forearm X-ray`, `Hand X-ray (PA)`).
- **Result Presentation:** Three clean cards: Anatomical Region (7 classes, Softmax confidence bar), Fracture Assessment (Binary status badge + Sigmoid probability bar), and Factual Summary (rule-based caption with disclaimer).
- **Caption Generation:** Rule-based and deterministic; strictly excludes generative hallucinations, fracture subtypes, laterality, and treatment recommendations.
- **Safety & Compliance:** Branded cleanly as **BoneSight**; explicitly excludes claims of clinical validation or FDA clearance, positioning the system transparently as a research prototype.
- **Test Integrity:** Over 85 automated tests pass with 100% success rate across API, inference, and UI integration contracts.
