# Bone fracture image captioning using deep learning (ResNet-50)

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-EE4C2C.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B.svg)](https://streamlit.io/)
[![Tests: 35 Passed](https://img.shields.io/badge/Tests-35%20Passed-brightgreen.svg)](tests/)

---

## 1. Overview

This repository provides an end-to-end, reproducible deep learning system for **Bone Fracture X-Ray Analysis and Factual Image Description**. 

Plain bone radiography (X-ray) is the standard initial imaging modality in musculoskeletal trauma assessment. However, high clinical caseloads and subtle fracture margins can lead to delayed or missed diagnoses. This project addresses these challenges by processing input bone radiographs through a multi-stage deep learning pipeline that outputs:

1. **Anatomical Region Classification:** Identifies the bone or joint site (7 classes).
2. **Binary Fracture Detection:** Predicts fracture presence with calibrated probability scores.
4. **Disciplined Factual Captions:** Synthesizes deterministic natural-language descriptions strictly bounded by verified model findings, eliminating hallucinations.
5. **Interactive Interfaces:** Provides a high-performance **FastAPI** REST backend and a **Streamlit** diagnostic web dashboard.

---

## 2. Features

* **Multi-Task Architecture:** Shared ResNet-50 backbone with independent anatomical and pathological heads.
* **Spatial Lesion Localization:** Faster R-CNN with MobileNetV3-Large FPN for lesion bounding box detection.
* **Disciplined Caption Synthesizer:** Strict, hallucination-free factual caption generator adhering to 10 clinical validation rules.
* **Uncertainty Quantification:** Flags low-confidence anatomical predictions below calibrated operating thresholds ($\tau < 0.40$).
* **Cross-Platform CLI Inference:** Single-command execution returning structured, machine-readable JSON.
* **Production FastAPI Service:** Asynchronous REST API with `/health`, `/predict`, and secure `/visualizations/{filename}` endpoints.
* **Streamlit Diagnostic UI:** Upload radiographs, inspect classification metrics, render visual overlays, and review JSON responses.
* **Automated Test Suite:** Comprehensive suite (35 automated tests) covering preprocessing, label taxonomy, model loading, pipeline execution, and REST endpoints.

---

## 3. System Architecture

```
X-RAY RADIOGRAPH
      │
      ▼
Deterministic Preprocessing (Aspect-Ratio Preserved Pad to 224×224 / 384×384)
      │
      ▼
Multi-Task Backbone (ResNet-50 Feature Extractor)
      ├──► Head 1: Anatomical Region Classification (7 classes)
      └──► Head 2: Fracture Presence Detection (Binary BCE with pos_weight)
               │
               ├─► If Fracture = True  ──► Faster R-CNN MobileNetV3 FPN Detector
               └─► If Fracture = False ──► Bypass Localization
                        │
                        ▼
            Structured Findings Assembly
                        │
                        ├─► Factual Caption Generator (Hallucination-free template)
                        ├─► Annotated Visual Overlay Renderer
                        └─► Dual Interfaces (FastAPI REST API / Streamlit Dashboard)
```

For detailed specifications, see the [Architecture Document](docs/architecture.md).

---

## 4. Datasets

This project utilizes two primary open-access radiographic datasets:

| Dataset | Modality & Scope | Role in Pipeline | Labels & Annotations |
| :--- | :--- | :--- | :--- |
| **BoneFract** | ~9,260 radiographs across 7 anatomical regions | Multi-task anatomical classification & fracture detection | 7 regions (`Arm`, `Foot`, `Hand`, `Lower leg`, `Thigh`, `pelvis`, `wrist`), binary fracture status |
| **FracAtlas** | 4,083 radiographs with detailed lesions | Fracture lesion spatial localization | Bounding boxes, polygon segmentation masks (COCO JSON) |

> [!IMPORTANT]
> **Dataset files are intentionally not included in this Git repository** due to multi-gigabyte storage limits and medical data governance practices. Complete download instructions, source citations, and directory schemas are documented in [`data/README.md`](data/README.md).

---

## 5. Project Structure

```
bone-fracture-ai/
│
├── README.md                          <- Main project documentation
├── LICENSE                            <- MIT License (with third-party data notices)
├── .gitignore                         <- Comprehensive ML/Python ignore rules
├── .env.example                       <- Environment variable template
├── requirements.txt                   <- Core production dependencies
├── requirements-dev.txt               <- Testing & development tooling
├── predict.py                         <- Root CLI inference entry point
├── preprocess.py                      <- Patient-cluster preprocessing pipeline
├── train.py                           <- Multi-task classifier training script
├── train_detector.py                  <- Faster R-CNN detector training script
├── evaluate.py                        <- Multi-task test set evaluation
├── evaluate_detector.py               <- Lesion localization evaluation
│
├── data/                              <- Dataset storage instructions
│   └── README.md
│
├── models/                            <- Model architecture & weight instructions
│   ├── README.md
│   └── checkpoints/
│       └── .gitkeep                   <- Place best_model.pt and best_detector.pt here
│
├── inference/                         <- Modular inference package
│   ├── __init__.py
│   ├── predict.py                     <- Inference CLI entry point
│   ├── model_registry.py              <- Dynamic checkpoint resolver & configurations
│   ├── preprocessing.py               <- Deterministic resizing & normalization
│   ├── region_predictor.py            <- Anatomical site classification
│   ├── fracture_predictor.py          <- Binary fracture presence prediction
│   ├── localization_predictor.py      <- Faster R-CNN bounding box predictor
│   ├── caption_generator.py           <- Disciplined factual caption generator
│   └── visualize.py                   <- Diagnostic overlay rendering
│
├── backend/                           <- FastAPI REST API service
│   ├── main.py                        <- Application routes and lifecycle
│   ├── schemas.py                     <- Pydantic request/response schemas
│   └── services/
│       └── inference_service.py       <- Singleton model loader & pipeline coordinator
│
├── frontend/                          <- Streamlit web dashboard
│   ├── app.py                         <- Interactive user interface
│   └── README.md
│
├── preprocessing/                     <- Preprocessing entry points
│   ├── preprocess.py
│   └── dataset_audit.py
│
├── training/                          <- Training entry points & configurations
│   ├── train.py
│   ├── train_detector.py
│   └── configs/
│       └── default_config.json
│
├── evaluation/                        <- Evaluation scripts
│   ├── evaluate.py
│   └── evaluate_detector.py
│
├── tests/                             <- Automated test suite (35 tests)
│   ├── test_preprocessing.py          <- Image loading and transformation tests
│   ├── test_label_mapping.py          <- Class taxonomy and mapping tests
│   ├── test_model_loading.py          <- Model init and tensor shape tests
│   ├── test_inference.py              <- End-to-end inference and schema tests
│   ├── test_caption_generator.py      <- Clinical captioning rule validation
│   ├── test_api.py                    <- FastAPI endpoint tests
│   └── test_backend_api.py            <- Backend service tests
│
├── docs/                              <- Detailed design specifications
│   ├── architecture.md
│   ├── data_pipeline.md
│   ├── model_pipeline.md
│   ├── inference_pipeline.md
│   └── api.md
│
├── reports/                           <- Evaluation metrics, plots, and RCA reports
│   ├── github_readiness_report.md
│   ├── dataset_audit.md
│   ├── model_evaluation.md
│   └── ROOT_CAUSE_ANALYSIS.md
│
└── assets/                            <- Architecture diagrams and resources
    └── README.md
```

---

## 6. Requirements

* **Operating System:** Windows 10/11, macOS (Apple Silicon or Intel), or Linux (Ubuntu 20.04+).
* **Python Version:** Python **3.10**, **3.11**, **3.12**, or **3.13** (tested on Python 3.13).
* **Hardware Requirements:**
  * **CPU:** Multi-core processor (Intel Core i5/i7/i9, AMD Ryzen, or Apple Silicon). Minimum 8 GB RAM (16 GB recommended for full dataset preprocessing).
  * **GPU (Optional):** NVIDIA GPU with CUDA 11.8+ or 12.1+ for accelerated training. All inference and tests run cleanly on standard CPU.

---

## 7. Installation

### Step 1: Clone Repository
```bash
git clone https://github.com/tanmay0212-cls/bone-fracture-detection-using-deep-learning.git
cd bone-fracture-detection-using-deep-learning
```

### Step 2: Create Virtual Environment
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> [!TIP]
> If using an NVIDIA GPU with CUDA on Linux or Windows, install the matching PyTorch CUDA build first:
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
> ```

### Step 4: Configure Environment
```bash
# Windows PowerShell
Copy-Item .env.example .env

# Linux / macOS
cp .env.example .env
```

---

## 8. Dataset Setup

1. Review [`data/README.md`](data/README.md) for download links.
2. Download `BoneFract A Bone Fracture Dataset.zip` and extract to `data/BoneFract/`.
3. Download FracAtlas and place images and annotations into `data/FracAtlas/`.

---

## 9. Model Setup

Place trained checkpoints into `models/checkpoints/` (or repository root):

```
models/checkpoints/
├── best_model.pt       <- Multi-task classifier
└── best_detector.pt    <- Fracture localization detector
```

The system automatically detects checkpoints in `models/checkpoints/`, repository root, or via environment variables (`CLASSIFIER_CHECKPOINT`, `DETECTOR_CHECKPOINT`).

---

## 10. Preprocessing

Execute the patient-cluster, leak-free preprocessing pipeline:

```bash
python preprocess.py
```

Or via the modular entry point:
```bash
python preprocessing/preprocess.py
```

This creates the partitioned datasets in `processed/` and metadata CSVs: `train.csv`, `validation.csv`, and `test.csv`.

---

## 11. Training

### Train Multi-Task Classifier (ResNet-50):
```bash
# Full training
python train.py --backbone resnet50 --epochs_stage1 3 --epochs_stage2 2 --batch_size 32

# Fast sample-limited training on CPU
python train.py --backbone resnet50 --max_train_samples 1000 --max_val_samples 250
```

### Train Fracture Localization Detector (Faster R-CNN):
```bash
python train_detector.py --img_size 384 --epochs 2 --batch_size 2
```

---

## 12. Evaluation

### Evaluate Multi-Task Classifier:
```bash
python evaluate.py --checkpoint models/checkpoints/best_model.pt --test_csv test.csv
```

Outputs `metrics.json`, `classification_report.csv`, and `confusion_matrix.png`.

### Evaluate Fracture Detector:
```bash
python evaluate_detector.py --checkpoint models/checkpoints/best_detector.pt
```

---

## 13. CLI Inference

Run inference on any radiograph:

```bash
# Standard formatted terminal output
python predict.py --image path/to/xray.jpg

# Or via inference package entry point
python inference/predict.py --image path/to/xray.jpg

# Export annotated visual overlay
python predict.py --image path/to/xray.jpg --output-dir visualizations

# Strict machine-readable JSON output
python predict.py --image path/to/xray.jpg --json
```

---

## 14. Backend Service (FastAPI)

Launch the REST backend service:

```bash
uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```

* **Health Probe:** `http://127.0.0.1:8000/health`
* **Interactive Docs (Swagger UI):** `http://127.0.0.1:8000/docs`

---

## 15. Frontend Dashboard (Streamlit)

Launch the web dashboard in a separate terminal:

```bash
streamlit run frontend/app.py
```

The browser will automatically open `http://localhost:8501`.

---

## 16. API Documentation

### POST `/predict`
Accepts `multipart/form-data` with an `image` field.

#### Example Request (cURL):
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "image=@sample_xray.png"
```

#### Example Response (200 OK):
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
  "visualization_url": "/visualizations/pred_b363441d5f0a.png"
}
```

---

## 17. Example Output

```
=================================================================
         BONE FRACTURE STRUCTURED DIAGNOSTIC PREDICTION
=================================================================
Input File:              sample_wrist_fracture.png
Anatomical Region:       WRIST (94.20%)
Fracture Status:         POSITIVE (Fracture Detected)
Fracture Confidence:     91.20%
Localization Supported:  True
Detected Lesion Locations:
  [1] BBox: [142.0, 310.5, 235.0, 412.0] | Conf: 78.40%

Generated Factual Caption:
  "X-ray of the wrist showing a fracture."

Annotated Visualization: visualizations/annotated_wrist.png
=================================================================
```

---

## 18. Troubleshooting

* **Checkpoint not found:** Ensure `best_model.pt` is located in `models/checkpoints/` or set `CLASSIFIER_CHECKPOINT` in `.env`.
* **FastAPI Backend Offline in Streamlit:** Start the FastAPI service first with `uvicorn backend.main:app --reload` before launching Streamlit.
* **Localization returns null / false:** Expected when primary model predicts `fracture = False` or lesion confidence is below operating threshold ($\tau < 0.10$).
* **Pelvis vs. Hip Joint Prediction:** In the BoneFract dataset, hip examinations are classified under `pelvis`.
* **Out of Memory during training:** Lower `--batch_size` to 16 or 8, or use `--max_train_samples`.

---

## 19. Limitations

* **Domain Restriction:** Trained exclusively on plain X-ray radiographs; not suitable for CT, MRI, or ultrasound.
* **Taxonomy Constraints:** Supports the 7 anatomical regions in BoneFract; does not support cranial, facial, spine, or rib cage radiographs.
* **Subtle Lesions:** Subtle non-displaced or stress fractures may require cross-sectional imaging (CT/MRI).

---

## 20. Medical Disclaimer

> [!CAUTION]
> **This software and associated models are intended solely for academic research, educational demonstrations, and engineering development.** The predictions generated by this system are AI-assisted outputs and **are not a substitute for professional clinical diagnosis, radiologist interpretation, or emergency medical triage**. Do not make patient-care decisions based on outputs from this system.

---

## 21. Future Improvements

* Vision-Language Foundation Model integration (e.g., BioViL, Med-Flamingo) for dense clinical report generation.
* Multi-view radiograph fusion (AP + Lateral projections).
* Uncertainty calibration via Monte Carlo Dropout and conformal prediction.
* DICOM native format ingestion and PACS integration.
