# 13. Missing Information, Discrepancies, and Open Questions

**Project Title:** Bone Fracture Image Captioning Using Deep Learning  
**Scope:** Transparent audit of unresolved issues, resolved discrepancies, missing external files, and open academic administrative questions.  
**Document Version:** 1.0 (Audited October 2026)  

---

## 1. Resolved Discrepancies During Technical Audit

### 1.1 Discrepancy 1: Inception-v3 Evaluation Covariate Shift
* **Initial Observation:** An early evaluation run of Inception-v3 (`scripts/evaluate_exp5.py`) reported an anomalous accuracy collapse to 28.34%.
* **Audit Investigation:** Inspection of `src/models_exp5.py` and TorchVision's source code ([`reports/exp5_evaluation_audit.md`](file:///c:/Users/Donipudi%20Asish%20Kumar/Desktop/Deep%20learning%20project/reports/exp5_evaluation_audit.md)) showed that `transform_input = False` was passed during evaluation, whereas PyTorch's native Inception-v3 model was trained with internal scaling enabled (`transform_input = True`).
* **Resolution Applied:** Setting `transform_input = True` restored the native training distribution. Re-evaluating the frozen checkpoint `exp5_inception_v3_multitask.pt` confirmed the model operates at **81.58% accuracy, 87.28% specificity, 67.58% recall, and 80.24% region accuracy**, fully aligning with validation metrics.

### 1.2 Discrepancy 2: Active Production Routing vs. Experimental Winner
* **Initial Observation:** Documentation references Experiment 4 as the undisputed top clinical performer, yet `inference/model_registry.py` defines `EXP3_CHECKPOINT_FILENAME = "exp3_resnet50_448.pt"`.
* **Audit Investigation:** Chronologically, Experiment 3 was the initial high-resolution baseline deployed to `model_registry.py`. Experiment 4 was subsequently trained and evaluated as a controlled research experiment, proving superior clinical sensitivity (72.87% recall, catching 1,007 fractures vs. 957 in Exp 3).
* **Resolution:** Clearly documented that `exp4_balanced_resnet50_448.pt` is the definitive recommended model for production deployment, while `model_registry.py` currently loads `exp3_resnet50_448.pt` by default.

### 1.3 Discrepancy 3: Pytest Root Collection Failure
* **Initial Observation:** Running `pytest` from the root directory triggered an error collecting `scripts/test_clinical_presets.py`.
* **Audit Investigation:** `scripts/test_clinical_presets.py` is an end-to-end client script that requires a live, running FastAPI server on port 8000. Pytest's default file discovery collected it because of the `test_` prefix.
* **Resolution:** Pytest should be invoked targeting the dedicated test suite directory: `.venv\Scripts\python.exe -m pytest tests/ -v`, which executes all 69 unit, integration, and API tests with 100% pass rate.

---

## 2. Missing External Artifacts & Data Gaps

### 2.1 Multi-Gigabyte Raw Image Archives (`data/BoneFract/` & `data/FracAtlas/`)
* **Status:** Raw image archives (~3.95 GB) are intentionally excluded from the Git repository due to GitHub file size limits and medical data governance practices.
* **Impact:** The code cannot retrain models from scratch without downloading the archives. However, all trained checkpoints (`best_model.pt`, `best_detector.pt`, `exp3_resnet50_448.pt`, `exp4_balanced_resnet50_448.pt`, `exp5_inception_v3_multitask.pt`) and split manifests (`train.csv`, `validation.csv`, `test.csv`, `reports/exp4_sample_manifest.csv`) are 100% present and verified on disk.
* **Mitigation:** Complete acquisition and extraction commands are documented in [`data/README.md`](file:///c:/Users/Donipudi%20Asish%20Kumar/Desktop/Deep%20learning%20project/data/README.md).

### 2.2 Ground-Truth Bounding Boxes for Mendeley BoneFract
* **Status:** The Mendeley BoneFract dataset contains image-level labels only (anatomical site and fracture presence), lacking bounding box coordinates.
* **Impact:** Faster R-CNN object detection can only be trained on the FracAtlas dataset ($N=719$ positive images with 924 bounding boxes).
* **Mitigation:** The system successfully uses FracAtlas for localization, and the Streamlit UI explicitly notes that localization on non-FracAtlas images is under development.

### 2.3 External Prospective Hospital Validation
* **Status:** Testing has been conducted exclusively on retrospective held-out splits ($N=4,778$). Prospective validation on live hospital emergency DICOM streams has not yet been conducted.
* **Mitigation:** Formally cataloged as a known limitation in [`08_error_analysis_and_limitations.md`](file:///c:/Users/Donipudi%20Asish%20Kumar/Desktop/Deep%20learning%20project/project_report_evidence/08_error_analysis_and_limitations.md).

---

## 3. Administrative Questions for Final University Submission

To finalize the academic report for formal university submission, the following institutional parameters should be confirmed by the candidate:
1. **Official Submission Date:** Is the final date set for April 2026 or an earlier interim evaluation date?
2. **Supervisor & Examiner Signatures:** Are digital signature blocks required for Dr. Chandra Sekhar Dash and Mr. Rajkumar Mohanta, or will physical wet signatures be obtained on the printed copies?
3. **Target Print Format:** Will the university require standard double-sided spiral binding or hardbound black embossed book binding?

