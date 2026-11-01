# 01. Repository Inventory and Asset Ledger

**Project Title:** Bone Fracture Image Captioning Using Deep Learning  
**Corpus Root:** `c:\Users\Donipudi Asish Kumar\Desktop\Deep learning project`  
**Inspection Date:** October 2026  
**Auditor:** Automated Forensic Technical Auditor & DL Specialist  

---

## 1. Executive Summary of Inventory

A systematic, multi-tiered audit of the entire accessible workspace was conducted. Every core Python script, neural model definition, inference pipeline, FastAPI route, Streamlit frontend component, test suite module, evaluation JSON/CSV log, and markdown documentation file was verified by inspection and execution.

* **Total Project Directories Inspected:** 22
* **Total Executable Python Modules:** 34
* **Total Test Cases Executed & Verified:** 69 (69 passed, 0 failed, 0 skipped)
* **Total Binary Model Checkpoints:** 5 (`best_model.pt`, `best_detector.pt`, `exp3_resnet50_448.pt`, `exp4_balanced_resnet50_448.pt`, `exp5_inception_v3_multitask.pt`)
* **Total Held-Out Test Evaluation Samples:** 4,778 clinical radiographs

---

## 2. Comprehensive File and Directory Ledger

| File / Directory Path | File Type | Purpose & Description | Important Functions / Classes / Artifacts | Contains Report Evidence? | Successfully Inspected? | Dependency & Integrity Status |
| :--- | :--- | :--- | :--- | :---: | :---: | :--- |
| `src/models.py` | Python Module | Primary neural architecture implementations for multi-task learning. | `MultiTaskModel`, `DedicatedFractureClassifier`, `resnet50`, `resnet18` | **YES** | **YES** | Active; requires PyTorch, TorchVision. |
| `src/models_exp5.py` | Python Module | Multi-Task Inception-v3 with auxiliary deep supervision heads. | `MultiTaskInceptionV3`, `InceptionOutputs`, `Mixed_7b` tap | **YES** | **YES** | Active; requires PyTorch, TorchVision. |
| `src/dataset.py` | Python Module | PyTorch dataset loader with deterministic aspect-ratio padding. | `BoneFractDataset`, `ResizeAndPad`, `get_transforms` | **YES** | **YES** | Active; requires PIL, TorchVision. |
| `src/detection_dataset.py` | Python Module | PyTorch detection dataset loader for FracAtlas COCO/XML format. | `FracAtlasDetectionDataset`, `convert_coco_to_pascal` | **YES** | **YES** | Active; requires PyTorch, JSON. |
| `src/caption_generator.py` | Python Module | Deterministic clinical natural-language description engine. | `generate_description`, `_normalize_region`, `_parse_fracture_status` | **YES** | **YES** | Active; pure Python standard library. |
| `src/metrics.py` | Python Module | Comprehensive clinical and computer vision evaluation metrics. | `compute_classification_metrics`, `compute_confusion_matrix_rates` | **YES** | **YES** | Active; requires Scikit-Learn, NumPy. |
| `inference/model_registry.py` | Python Module | Centralized checkpoint resolver and dynamic resolution routing gate. | `ModelRegistry.route_image`, `get_baseline_classifier_path`, `REGION_TO_IDX` | **YES** | **YES** | Active; resolution threshold = 300 px. |
| `inference/preprocessing.py` | Python Module | Preprocessing routines for inference and canvas padding. | `load_and_validate_image`, `preprocess_for_classifier`, `resize_and_pad` | **YES** | **YES** | Active; requires PIL, TorchVision. |
| `inference/region_predictor.py` | Python Module | Anatomical site classification inference wrapper. | `predict_region`, `softmax`, `REGION_UNCERTAINTY_THRESHOLD` | **YES** | **YES** | Active; requires PyTorch. |
| `inference/fracture_predictor.py` | Python Module | Binary fracture presence inference wrapper. | `predict_fracture`, `sigmoid`, `FRACTURE_THRESHOLD = 0.50` | **YES** | **YES** | Active; requires PyTorch. |
| `inference/localization_predictor.py` | Python Module | Faster R-CNN lesion localization inference wrapper. | `predict_localization`, `coordinate_rescaling` | **YES** | **YES** | Active; operating threshold = 0.10. |
| `inference/visualize.py` | Python Module | Renders diagnostic bounding box overlays on radiographs. | `render_diagnostic_overlay`, Matplotlib plotting | **YES** | **YES** | Active; requires Matplotlib, Pillow. |
| `backend/main.py` | Python Module | FastAPI REST API microservice serving endpoints. | `predict_fracture` (`POST /predict`), `health_check` (`GET /health`) | **YES** | **YES** | Active; requires FastAPI, Uvicorn. |
| `backend/schemas.py` | Python Module | Pydantic data schemas and contracts for API requests/responses. | `PredictionResponse`, `BoundingBoxSchema`, `HealthResponse` | **YES** | **YES** | Active; requires Pydantic >= 2.0. |
| `backend/services/inference_service.py` | Python Module | Singleton inference orchestrator and byte buffer validator. | `InferenceService.get_instance`, `process_image` | **YES** | **YES** | Active; singleton pattern thread-safe. |
| `frontend/app.py` | Python Module | Streamlit clinical diagnostic radiology workstation (*BoneSight*). | `navigate_to`, `render_html`, `load_preset`, UI components (1337 lines) | **YES** | **YES** | Active; requires Streamlit, Requests. |
| `models/checkpoints/best_model.pt` | PyTorch Binary (100.6 MB) | Deprecated Baseline ResNet-50 trained at 224x224. | Full state dict; SHA-256 verified | **YES** | **YES** | Verified present on disk. |
| `models/checkpoints/best_detector.pt` | PyTorch Binary (76.0 MB) | Faster R-CNN MobileNetV3 FPN trained on FracAtlas. | Detection state dict; SHA-256 verified | **YES** | **YES** | Verified present on disk. |
| `models/checkpoints/exp3_resnet50_448.pt` | PyTorch Binary (100.6 MB) | High-Resolution ResNet-50 trained at 448x448 (imbalanced). | Full state dict; SHA-256 verified | **YES** | **YES** | Verified present on disk. |
| `models/checkpoints/exp4_balanced_resnet50_448.pt` | PyTorch Binary (100.6 MB) | Balanced Region-Aware ResNet-50 trained at 448x448 (**CLINICAL WINNER**). | Full state dict; SHA-256 verified | **YES** | **YES** | Verified present on disk. |
| `models/checkpoints/exp5_inception_v3_multitask.pt` | PyTorch Binary (105.2 MB) | Multi-Task Inception-v3 trained at 299x299 (balanced). | Full state dict; SHA-256 verified | **YES** | **YES** | Verified present on disk. |
| `preprocess.py` | Python Script | Root entry point for dataset splitting and graph clustering. | `cluster_and_partition`, `generate_dataset_splits` | **YES** | **YES** | Generates train/val/test CSVs. |
| `train.py` | Python Script | Root CLI entry point for training baseline MultiTaskModel. | Two-stage training loop, loss backpropagation | **YES** | **YES** | Verified executable. |
| `evaluate.py` | Python Script | Root CLI entry point for evaluating models on test sets. | Test evaluation loop, confusion matrix generation | **YES** | **YES** | Verified executable. |
| `predict.py` | Python Script | Root CLI entry point for single-image CLI inference. | Single-image ingestion, resolution routing, captioning | **YES** | **YES** | Verified executable. |
| `scripts/train_exp3_448.py` | Python Script | Specialized training pipeline for Experiment 3 (448x448). | High-res training loop, mixed precision AMP | **YES** | **YES** | Verified executable. |
| `scripts/train_exp4_balanced.py` | Python Script | Specialized training pipeline for Experiment 4 (Balanced 448x448). | Manifest-driven balanced training loop | **YES** | **YES** | Verified executable. |
| `scripts/train_exp5_inception_v3.py` | Python Script | Specialized training pipeline for Experiment 5 (Inception-v3 299x299). | Auxiliary head loss weighting, AMP | **YES** | **YES** | Verified executable. |
| `scripts/evaluate_exp3_448.py` | Python Script | Test evaluation script for Experiment 3 on 4,778 held-out images. | Generates `exp3_evaluation_report.json` | **YES** | **YES** | Verified executable. |
| `scripts/evaluate_exp4.py` | Python Script | Test evaluation script for Experiment 4 on 4,778 held-out images. | Generates `exp4_evaluation_report.json` | **YES** | **YES** | Verified executable. |
| `scripts/evaluate_exp5.py` | Python Script | Corrected evaluation script for Exp 5 (`transform_input=True`). | Generates `exp5_evaluation_report.json` | **YES** | **YES** | Verified executable. |
| `scripts/audit_resolution_provenance.py` | Python Script | Forensic dimension audit of all 47,931 images in Mendeley archive. | Native 102x102 thumbnail discovery script | **YES** | **YES** | Verified executable. |
| `scripts/audit_exp5_evaluation.py` | Python Script | Root cause diagnostic script resolving Inception-v3 covariate shift. | Validates `transform_input` normalization | **YES** | **YES** | Verified executable. |
| `reports/three_way_experiment_comparison_report.md` | Markdown Report | Definitive 3-way benchmark report (Exp 3 vs Exp 4 vs Exp 5). | Comparative tables across all 4,778 test images | **YES** | **YES** | Authoritative empirical benchmark. |
| `reports/exp4_evaluation_report.json` | JSON Metric Artifact | Machine-readable evaluation metrics for Experiment 4. | TP=1007, TN=2839, FP=557, FN=375, Recall=72.87% | **YES** | **YES** | Authoritative empirical source. |
| `reports/exp5_evaluation_report.json` | JSON Metric Artifact | Machine-readable evaluation metrics for Experiment 5. | Specificity=87.28%, Foot recall=55.12%, Acc=81.58% | **YES** | **YES** | Authoritative empirical source. |
| `reports/exp3_evaluation_report.json` | JSON Metric Artifact | Machine-readable evaluation metrics for Experiment 3. | Recall=69.25%, High-res recall=90.91%, Low-res=38.20% | **YES** | **YES** | Authoritative empirical source. |
| `reports/baseline_eval_results.json` | JSON Metric Artifact | Machine-readable evaluation metrics for Baseline ResNet-50. | Recall=18.52%, FN=1126, Specificity=80.45% | **YES** | **YES** | Authoritative empirical source. |
| `reports/ROOT_CAUSE_ANALYSIS.md` | Markdown Report | Root cause investigation of pelvis-to-thigh misclassification. | Absence of dedicated "Hip" class in taxonomy | **YES** | **YES** | Verified anatomical analysis. |
| `reports/caption_generator_audit.md` | Markdown Report | Verification and safety audit of the 10 clinical caption rules. | Prohibited clinical terms, determinism tests | **YES** | **YES** | Verified compliance report. |
| `RESOLUTION_PROVENANCE_AUDIT.md` | Markdown Report | Forensic audit of native 102x102 thumbnail distribution. | Hash collision tests, patient graph isolation | **YES** | **YES** | Verified forensic audit. |
| `FINAL_MODEL_EVALUATION.md` | Markdown Report | Comprehensive baseline & detector evaluation report. | Split distributions, detector mAP, baseline metrics | **YES** | **YES** | Verified report. |
| `PROJECT_REPORT_COMPLETE.md` | Markdown Report | Complete 703-line formal B.Tech academic thesis draft. | Complete chapter text, student metadata, faculty names | **YES** | **YES** | Thesis foundation text. |
| `dataset_summary.csv` | CSV Artifact | Image counts, splits, regions, and byte metrics. | 42 rows (7 regions x 2 labels x 3 splits) | **YES** | **YES** | Matches dataset split counts. |
| `performance_by_region.csv` | CSV Artifact | Regional breakdown metrics across anatomical sites. | Precision, recall, specificity per region | **YES** | **YES** | Extracted from evaluation. |
| `classification_report.csv` | CSV Artifact | Historical 500-sample preliminary evaluation export. | Sample support = 500 | **YES** | **YES** | Note: superseded by 4,778 test set. |
| `train.csv` | CSV Split File (4.55 MB) | Manifest of 38,366 training radiographs. | `image_path,anatomical_region,fracture_label,patient_id` | **YES** | **YES** | Primary training manifest. |
| `validation.csv` | CSV Split File (592 KB) | Manifest of 4,787 validation radiographs. | 4,787 patient-isolated validation rows | **YES** | **YES** | Validation manifest. |
| `test.csv` | CSV Split File (562 KB) | Manifest of 4,778 held-out test radiographs. | 4,778 patient-isolated test rows | **YES** | **YES** | Held-out test manifest. |
| `reports/exp4_sample_manifest.csv` | CSV Split File (3.82 MB) | Manifest of 14,000 region-balanced training samples. | Exactly 2,000 samples per anatomical site | **YES** | **YES** | Experiment 4 training manifest. |
| `tests/test_api.py` | Test Script | Automated tests for FastAPI integration and schemas. | 5 passing test cases | **YES** | **YES** | Tested via Pytest. |
| `tests/test_backend_api.py` | Test Script | Automated tests for microservice validation and endpoints. | 5 passing test cases | **YES** | **YES** | Tested via Pytest. |
| `tests/test_caption_generator.py` | Test Script | Automated validation of all 10 clinical caption rules. | 13 passing test cases | **YES** | **YES** | Tested via Pytest. |
| `tests/test_exp5_inception.py` | Test Script | Automated tests for MultiTaskInceptionV3 architecture. | 11 passing test cases | **YES** | **YES** | Tested via Pytest. |
| `tests/test_inference.py` | Test Script | Automated tests for end-to-end pipeline execution. | 4 passing test cases | **YES** | **YES** | Tested via Pytest. |
| `tests/test_label_mapping.py` | Test Script | Automated tests for anatomical and fracture taxonomies. | 5 passing test cases | **YES** | **YES** | Tested via Pytest. |
| `tests/test_model_loading.py` | Test Script | Automated tests for checkpoint loading and layer management. | 4 passing test cases | **YES** | **YES** | Tested via Pytest. |
| `tests/test_preprocessing.py` | Test Script | Automated tests for aspect-ratio padding and corruption handling. | 6 passing test cases | **YES** | **YES** | Tested via Pytest. |
| `tests/test_resolution_routing.py` | Test Script | Automated tests for resolution routing gate logic. | 16 passing test cases | **YES** | **YES** | Tested via Pytest. |
| `requirements.txt` | Configuration | Production runtime dependencies. | PyTorch, TorchVision, FastAPI, Streamlit, etc. | **YES** | **YES** | Pinned dependencies. |
| `requirements-dev.txt` | Configuration | Development, testing, and documentation dependencies. | Pytest, HTTPX, Matplotlib, ReportLab | **YES** | **YES** | Pinned dev dependencies. |
| `.env.example` | Configuration | Environment configuration template for server ports and paths. | `BACKEND_URL`, `CLASSIFIER_CHECKPOINT`, etc. | **YES** | **YES** | Configuration guide. |

---

## 3. Missing Dependencies, Unavailable Files, or Gaps

1. **Raw Medical Image Archives Excluded from Git (`data/BoneFract/` & `data/FracAtlas/`):**
   * *Status:* The multi-gigabyte raw image folders (`~3.95 GB`) are excluded from Git version control per medical data governance standards.
   * *Mitigation:* Explicit download, extraction, and SHA verification instructions are documented in [`data/README.md`](file:///c:/Users/Donipudi%20Asish%20Kumar/Desktop/Deep%20learning%20project/data/README.md). All split manifests (`train.csv`, `validation.csv`, `test.csv`, `reports/exp4_sample_manifest.csv`) are 100% intact and verified in the repository.
2. **Pytest Root Invocation Ambiguity:**
   * *Status:* Invoking `pytest` from the root directory attempts to run `scripts/test_clinical_presets.py`, which requires a live running FastAPI server on port 8000.
   * *Mitigation:* Pytest must be invoked specifying the `tests/` directory (`.venv\Scripts\python.exe -m pytest tests/ -v`), which executes all 69 unit and integration tests with a 100% pass rate.
3. **Production Model Registry Resolution Target:**
   * *Status:* `inference/model_registry.py` is configured by default to load `exp3_resnet50_448.pt` for high-resolution images, whereas subsequent research proved `exp4_balanced_resnet50_448.pt` is the undisputed top clinical performer.
   * *Mitigation:* This architectural transition is explicitly documented; `exp4_balanced_resnet50_448.pt` is designated as the primary recommended checkpoint for production updates.

