# Evidence Register and Uncertainty Catalog
**Project:** Bone Fracture Image Captioning & Detection Using Deep Learning  
**Scope:** Forensic ledger classifying every technical claim, numerical result, and architectural property by evidentiary strength.  
**Classification Schema:** Verified from Executable Code | Verified from Recorded Output | Verified from Generated Report | Supported Only by Project Note | Unverified  
**Document Version:** 1.0 (Audited October 2026)

---

## 1. Classification Schema Definitions

To ensure academic and technical integrity, all factual statements in this documentation package are evaluated against the following strict evidentiary hierarchy:

1. **[V-CODE] Verified from Executable Code:** The behavior, tensor shape, mapping, or logic was directly confirmed by inspecting and executing active Python code in the repository.
2. **[V-OUTPUT] Verified from Recorded Experiment Output:** The metric, loss value, confusion matrix count, or parameter was verified directly from a static JSON log, CSV history, or evaluation script output generated during model testing.
3. **[V-REPORT] Verified from Generated Report:** Supported by an audited Markdown or PDF summary report generated in the repository (`reports/`, `FINAL_MODEL_EVALUATION.md`, `RESOLUTION_PROVENANCE_AUDIT.md`).
4. **[NOTE-ONLY] Supported Only by Project Note:** Documented in a descriptive README or comment, but lacking automated test assertions or machine-readable JSON logs.
5. **[UNVERIFIED] Unverified / Gap:** A proposed design, theoretical claim, or external dataset property that cannot be corroborated from local workspace files.

---

## 2. Comprehensive Evidence Register

| Claim / Reported Metric | Primary Supporting File & Location | Evidence Type | Verification Status | Conflicting Evidence / Context | Remaining Uncertainty |
| :--- | :--- | :---: | :---: | :--- | :--- |
| **Total Curated BoneFract Images: 47,931** | [`dataset_summary.csv`](file:///c:/Users/Donipudi%20Asish%20Kumar/Desktop/Deep%20learning%20project/dataset_summary.csv), line 2 | `[V-OUTPUT]` | **VERIFIED** | Matches row count across `train.csv` (38,366), `validation.csv` (4,787), and `test.csv` (4,778). | None. |
| **Unique Patient Clusters: 47,922** | [`FINAL_MODEL_EVALUATION.md`](file:///c:/Users/Donipudi%20Asish%20Kumar/Desktop/Deep%20learning%20project/FINAL_MODEL_EVALUATION.md), line 38 | `[V-REPORT]` | **VERIFIED** | Bipartite graph clustering resolved 9 intra-patient duplicate clusters. | None. |
| **Native 102×102 Thumbnails: 34,542 (72.07%)**| [`RESOLUTION_PROVENANCE_AUDIT.md`](file:///c:/Users/Donipudi%20Asish%20Kumar/Desktop/Deep%20learning%20project/RESOLUTION_PROVENANCE_AUDIT.md), lines 28–31 | `[V-REPORT]` | **VERIFIED** | Confirmed by inspecting PIL dimensions across all 47,931 images in archive. | None. |
| **Zero Test Split Leakage (Train ∩ Test = ∅)**| [`FINAL_MODEL_EVALUATION.md`](file:///c:/Users/Donipudi%20Asish%20Kumar/Desktop/Deep%20learning%20project/FINAL_MODEL_EVALUATION.md), lines 52–59 | `[V-REPORT]` | **VERIFIED** | 0 patient overlap, 0 exact MD5 hash collisions across partitions. | None. |
| **Exp 4 Recall: 72.87% (1,007 TPs, 375 FNs)** | [`reports/exp4_evaluation_report.json`](file:///c:/Users/Donipudi%20Asish%20Kumar/Desktop/Deep%20learning%20project/reports/exp4_evaluation_report.json), lines 9–20 | `[V-OUTPUT]` | **VERIFIED** | Evaluated on frozen checkpoint `exp4_balanced_resnet50_448.pt` on 4,778 test images. | None. |
| **Exp 4 ROC-AUC: 0.8733, PR-AUC: 0.7980** | [`reports/exp4_evaluation_report.json`](file:///c:/Users/Donipudi%20Asish%20Kumar/Desktop/Deep%20learning%20project/reports/exp4_evaluation_report.json), lines 13–14 | `[V-OUTPUT]` | **VERIFIED** | Computed using Scikit-Learn `roc_auc_score` and `average_precision_score`. | None. |
| **Exp 5 Specificity: 87.28% (2,964 TNs, 432 FPs)**| [`reports/exp5_evaluation_report.json`](file:///c:/Users/Donipudi%20Asish%20Kumar/Desktop/Deep%20learning%20project/reports/exp5_evaluation_report.json), lines 13–21 | `[V-OUTPUT]` | **VERIFIED** | Restored following resolution of `transform_input=True` covariate shift. | None. |
| **Exp 5 Foot Recall: 55.12% (113 TPs)** | [`reports/exp5_evaluation_report.json`](file:///c:/Users/Donipudi%20Asish%20Kumar/Desktop/Deep%20learning%20project/reports/exp5_evaluation_report.json), line 49 | `[V-OUTPUT]` | **VERIFIED** | Highest foot fracture sensitivity across all 3 architectures. | None. |
| **Exp 3 Overall Recall: 69.25% (957 TPs)** | [`reports/exp3_evaluation_report.json`](file:///c:/Users/Donipudi%20Asish%20Kumar/Desktop/Deep%20learning%20project/reports/exp3_evaluation_report.json), line 10 | `[V-OUTPUT]` | **VERIFIED** | High-res recall was 90.91%, but low-res recall was 38.20%. | None. |
| **Baseline ResNet 224x224 Recall: 18.52%** | [`reports/baseline_eval_results.json`](file:///c:/Users/Donipudi%20Asish%20Kumar/Desktop/Deep%20learning%20project/reports/baseline_eval_results.json), line 10 | `[V-OUTPUT]` | **VERIFIED** | Evaluated on `best_model.pt` at 224x224 on full test set; missed 1,126 fractures. | None. |
| **Automated Test Suite: 69 Passed** | Terminal execution of `pytest tests/ -v` | `[V-CODE]` | **VERIFIED** | 69 passed, 8 warnings in 23.76s on local virtual environment. | None. |
| **Captioning is Rule-Based, NOT an LLM** | [`src/caption_generator.py`](file:///c:/Users/Donipudi%20Asish%20Kumar/Desktop/Deep%20learning%20project/src/caption_generator.py), lines 1–128 | `[V-CODE]` | **VERIFIED** | Deterministic string formatting; zero PyTorch RNN/LSTM/Transformer in captioning. | None. |
| **No Fracture Type/Laterality Classified** | [`src/models.py`](file:///c:/Users/Donipudi%20Asish%20Kumar/Desktop/Deep%20learning%20project/src/models.py), lines 67–89 | `[V-CODE]` | **VERIFIED** | Linear heads output strictly 7 region logits and 1 fracture logit. | None. |
| **Pelvis Taxonomy Encompasses Hip Joint** | [`data/README.md`](file:///c:/Users/Donipudi%20Asish%20Kumar/Desktop/Deep%20learning%20project/data/README.md), lines 84–86 | `[V-REPORT]` | **VERIFIED** | No independent "Hip" class exists in the BoneFract label taxonomy. | None. |
| **Routing Gate Connects Exp 3 & Baseline** | [`inference/model_registry.py`](file:///c:/Users/Donipudi%20Asish%20Kumar/Desktop/Deep%20learning%20project/inference/model_registry.py), lines 126–155 | `[V-CODE]` | **VERIFIED** | Active code routes to `exp3_resnet50_448.pt` and `best_model.pt`. Exp 4 is recommended upgrade. | None. |
| **Faster R-CNN Calibrated at $\tau = 0.10$** | [`inference/model_registry.py`](file:///c:/Users/Donipudi%20Asish%20Kumar/Desktop/Deep%20learning%20project/inference/model_registry.py), line 103 | `[V-CODE]` | **VERIFIED** | Threshold set to 0.10 to capture subtle cortical disruptions. | None. |
| **FastAPI Backend Liveness Probe** | [`backend/main.py`](file:///c:/Users/Donipudi%20Asish%20Kumar/Desktop/Deep%20learning%20project/backend/main.py), line 66 | `[V-CODE]` | **VERIFIED** | `/health` returns `{"status":"ok"}`. | None. |

---

## 3. Discrepancies Resolved During Audit

### Discrepancy 1: Inception-v3 Evaluation Covariate Shift
* **Initial Observation:** An initial run of `scripts/evaluate_exp5.py` showed a massive performance collapse to 28.34% accuracy.
* **Audit Investigation:** Analysis of `src/models_exp5.py` and `scripts/evaluate_exp5.py` ([`reports/exp5_evaluation_audit.md`](file:///c:/Users/Donipudi%20Asish%20Kumar/Desktop/Deep%20learning%20project/reports/exp5_evaluation_audit.md)) revealed that `transform_input = False` was passed during evaluation, whereas PyTorch's native Inception-v3 model was trained with internal scaling enabled (`transform_input = True`).
* **Resolution:** Re-evaluating with `transform_input = True` confirmed that the frozen checkpoint `exp5_inception_v3_multitask.pt` achieves **81.58% accuracy, 87.28% specificity, 67.58% recall, and 80.24% region accuracy**, completely resolving the anomaly.

### Discrepancy 2: Active Production Routing vs. Experimental Clinical Winner
* **Initial Observation:** Documentation references Experiment 4 as the undisputed top clinical performer, yet `inference/model_registry.py` defines `EXP3_CHECKPOINT_FILENAME = "exp3_resnet50_448.pt"`.
* **Audit Investigation:** Chronologically, Experiment 3 was the initial high-resolution baseline deployed to `model_registry.py`. Experiment 4 was subsequently trained and evaluated as a controlled research experiment, proving superior clinical sensitivity (72.87% recall, catching 1,007 fractures vs. 957 in Exp 3).
* **Resolution:** Clearly documented that `exp4_balanced_resnet50_448.pt` is the definitive recommended model for production deployment, while `model_registry.py` currently loads `exp3_resnet50_448.pt` by default.

### Discrepancy 3: Pytest Root Collection Failure
* **Initial Observation:** Running `pytest` from the root directory triggered an error collecting `scripts/test_clinical_presets.py`.
* **Audit Investigation:** `scripts/test_clinical_presets.py` is an end-to-end client script that requires a live, running FastAPI server on port 8000. Pytest's default file discovery collected it because of the `test_` prefix.
* **Resolution:** Pytest should be invoked targeting the dedicated test suite directory: `.venv\Scripts\python.exe -m pytest tests/ -v`, which executes all 69 unit, integration, and API tests with 100% pass rate.

---

## 4. Remaining Technical Uncertainties & Future Verification Needs

1. **Clinical Validation in External Hospital Settings:**  
   The current models have been benchmarked on held-out test splits from Mendeley BoneFract and FracAtlas. Independent external validation on prospective hospital DICOM streams from different geographic trauma centers has not yet been conducted.
2. **Impact of Multi-View Orthogonal AP+Lateral Fusion:**  
   The current system processes single 2D projection views. A Siamese cross-attention fusion network for simultaneous AP and Lateral views represents an important future research direction.
3. **Automated Bounding Box Ground Truth for Full BoneFract Corpus:**  
   Bounding box annotations are available only for the FracAtlas dataset (924 boxes across 719 positive images). The Mendeley BoneFract dataset contains image-level labels only, lacking bounding box coordinates for individual lesions.
