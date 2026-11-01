# 12. Master Evidence Index

**Project Title:** Bone Fracture Image Captioning Using Deep Learning  
**Scope:** Forensic evidence ledger establishing the authoritative source, verification method, and location for every critical technical claim, numerical metric, and architectural property in this project.  
**Classification Schema:**  
- `[V-CODE]`: Verified directly from executable source code.  
- `[V-OUTPUT]`: Verified directly from recorded experimental output (JSON, CSV).  
- `[V-REPORT]`: Verified from an audited technical report in the repository.  
- `[NOTE-ONLY]`: Supported by project notes / comments.  
- `[UNVERIFIED]`: Unverified or pending external data.  
**Document Version:** 1.0 (Audited October 2026)  

---

## Master Evidentiary Ledger

| Claim / Key Finding | Authoritative Source File & Location | Evidence Type | Verification Status | Detailed Context & Verification Notes |
| :--- | :--- | :---: | :---: | :--- |
| **Total Curated BoneFract Images: 47,931** | [`dataset_summary.csv`](file:///c:/Users/Donipudi%20Asish%20Kumar/Desktop/Deep%20learning%20project/dataset_summary.csv), line 2 | `[V-OUTPUT]` | **VERIFIED** | Matches sum of `train.csv` (38,366), `validation.csv` (4,787), and `test.csv` (4,778). |
| **Unique Patient Clusters: 47,922** | [`FINAL_MODEL_EVALUATION.md`](file:///c:/Users/Donipudi%20Asish%20Kumar/Desktop/Deep%20learning%20project/FINAL_MODEL_EVALUATION.md), line 38 | `[V-REPORT]` | **VERIFIED** | Bipartite graph clustering resolved 9 intra-patient duplicate clusters. |
| **Native 102×102 Thumbnails: 34,542 (72.07%)** | [`RESOLUTION_PROVENANCE_AUDIT.md`](file:///c:/Users/Donipudi%20Asish%20Kumar/Desktop/Deep%20learning%20project/RESOLUTION_PROVENANCE_AUDIT.md), lines 28–31 | `[V-REPORT]` | **VERIFIED** | Confirmed by inspecting PIL dimensions across all 47,931 images in archive. |
| **Zero Test Split Leakage (Train ∩ Test = ∅)** | [`FINAL_MODEL_EVALUATION.md`](file:///c:/Users/Donipudi%20Asish%20Kumar/Desktop/Deep%20learning%20project/FINAL_MODEL_EVALUATION.md), lines 52–59 | `[V-REPORT]` | **VERIFIED** | 0 patient overlap, 0 exact MD5 hash collisions across partitions. |
| **Exp 4 Recall: 72.87% (1,007 TPs, 375 FNs)** | [`reports/exp4_evaluation_report.json`](file:///c:/Users/Donipudi%20Asish%20Kumar/Desktop/Deep%20learning%20project/reports/exp4_evaluation_report.json), lines 9–20 | `[V-OUTPUT]` | **VERIFIED** | Evaluated on frozen checkpoint `exp4_balanced_resnet50_448.pt` on 4,778 test images. |
| **Exp 4 ROC-AUC: 0.8733, PR-AUC: 0.7980** | [`reports/exp4_evaluation_report.json`](file:///c:/Users/Donipudi%20Asish%20Kumar/Desktop/Deep%20learning%20project/reports/exp4_evaluation_report.json), lines 13–14 | `[V-OUTPUT]` | **VERIFIED** | Computed using Scikit-Learn `roc_auc_score` and `average_precision_score`. |
| **Exp 5 Specificity: 87.28% (2,964 TNs, 432 FPs)** | [`reports/exp5_evaluation_report.json`](file:///c:/Users/Donipudi%20Asish%20Kumar/Desktop/Deep%20learning%20project/reports/exp5_evaluation_report.json), lines 13–21 | `[V-OUTPUT]` | **VERIFIED** | Restored following resolution of `transform_input=True` covariate shift. |
| **Exp 5 Foot Recall: 55.12% (113 TPs)** | [`reports/exp5_evaluation_report.json`](file:///c:/Users/Donipudi%20Asish%20Kumar/Desktop/Deep%20learning%20project/reports/exp5_evaluation_report.json), line 49 | `[V-OUTPUT]` | **VERIFIED** | Highest foot fracture sensitivity across all 3 architectures. |
| **Exp 3 Overall Recall: 69.25% (957 TPs)** | [`reports/exp3_evaluation_report.json`](file:///c:/Users/Donipudi%20Asish%20Kumar/Desktop/Deep%20learning%20project/reports/exp3_evaluation_report.json), line 10 | `[V-OUTPUT]` | **VERIFIED** | High-res recall was 90.91%, but low-res recall was 38.20%. |
| **Baseline ResNet 224x224 Recall: 18.52%** | [`reports/baseline_eval_results.json`](file:///c:/Users/Donipudi%20Asish%20Kumar/Desktop/Deep%20learning%20project/reports/baseline_eval_results.json), line 10 | `[V-OUTPUT]` | **VERIFIED** | Evaluated on `best_model.pt` at 224x224 on full test set; missed 1,126 fractures. |
| **Automated Test Suite: 69 Passed** | Pytest CLI execution output (`tests/`) | `[V-CODE]` | **VERIFIED** | 69 passed, 8 warnings in 98.51s on local environment. |
| **Captioning is Rule-Based, NOT an LLM** | [`src/caption_generator.py`](file:///c:/Users/Donipudi%20Asish%20Kumar/Desktop/Deep%20learning%20project/src/caption_generator.py), lines 1–128 | `[V-CODE]` | **VERIFIED** | Deterministic string formatting; zero PyTorch RNN/LSTM/Transformer in captioning. |
| **No Fracture Type/Laterality Classified** | [`src/models.py`](file:///c:/Users/Donipudi%20Asish%20Kumar/Desktop/Deep%20learning%20project/src/models.py), lines 67–89 | `[V-CODE]` | **VERIFIED** | Linear heads output strictly 7 region logits and 1 fracture logit. |
| **Pelvis Taxonomy Encompasses Hip Joint** | [`data/README.md`](file:///c:/Users/Donipudi%20Asish%20Kumar/Desktop/Deep%20learning%20project/data/README.md), lines 84–86 | `[V-REPORT]` | **VERIFIED** | No independent "Hip" class exists in the BoneFract label taxonomy. |
| **Routing Gate Connects Exp 3 & Baseline** | [`inference/model_registry.py`](file:///c:/Users/Donipudi%20Asish%20Kumar/Desktop/Deep%20learning%20project/inference/model_registry.py), lines 126–155 | `[V-CODE]` | **VERIFIED** | Active code routes to `exp3_resnet50_448.pt` and `best_model.pt`. Exp 4 is recommended upgrade. |
| **Faster R-CNN Calibrated at $\tau = 0.10$** | [`inference/model_registry.py`](file:///c:/Users/Donipudi%20Asish%20Kumar/Desktop/Deep%20learning%20project/inference/model_registry.py), line 103 | `[V-CODE]` | **VERIFIED** | Threshold set to 0.10 to capture subtle cortical disruptions. |
| **FastAPI Backend Liveness Probe** | [`backend/main.py`](file:///c:/Users/Donipudi%20Asish%20Kumar/Desktop/Deep%20learning%20project/backend/main.py), line 66 | `[V-CODE]` | **VERIFIED** | `/health` returns `{"status":"ok"}`. |
| **Streamlit Excludes Experimental Localization**| [`frontend/app.py`](file:///c:/Users/Donipudi%20Asish%20Kumar/Desktop/Deep%20learning%20project/frontend/app.py), line 1300 | `[V-CODE]` | **VERIFIED** | UI explicitly states: *"Under active development (excluded from UI)"*. |
| **In-Memory Buffer Validation** | [`backend/services/inference_service.py`](file:///c:/Users/Donipudi%20Asish%20Kumar/Desktop/Deep%20learning%20project/backend/services/inference_service.py), lines 50–65 | `[V-CODE]` | **VERIFIED** | Enforces `raw_im.verify()` to reject malformed payloads with HTTP 400. |
| **Path Traversal Protection** | [`backend/main.py`](file:///c:/Users/Donipudi%20Asish%20Kumar/Desktop/Deep%20learning%20project/backend/main.py), lines 152–154 | `[V-CODE]` | **VERIFIED** | Enforces `os.path.basename` equality check on visualization and download routes. |

