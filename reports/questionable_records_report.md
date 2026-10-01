# Questionable Records Analysis Report
**Dataset:** BoneFract (Mendeley Data)  
**Date of Audit:** September 2026  
**Auditor:** Lead Medical ML Data Engineer  

---

## 1. Executive Summary

During the rigorous automated dataset audit across all **47,931 images**, a total of **6,599 questionable records** were identified and cataloged into [`reports/questionable_records.csv`](reports/questionable_records.csv).

These questionable records fall into four primary critical categories:
1. **Severe Cross-Split Data Leakage (4,989 instances):** Identical image files present in the training set also appear in the test set (1,154 images, 24.07% of the test set) and validation set (1,157 images, 24.14% of the validation set).
2. **Ground-Truth Label Inconsistency / Cross-Label Conflicts (2,509 instances):** Exactly identical images (bit-for-bit MD5 collisions) assigned conflicting labels—one instance marked `Positive` and another marked `Negative`.
3. **Synthetic Patient ID Multi-Assignment (14,253 duplicate instances):** Identical images assigned different patient IDs, proving patient IDs are not unique clinical patient entities.
4. **Folder vs. Filename Anatomical Mismatch (1 instance):** File `Arm_patient26944_Positive_001.png` located inside the `wrist/` directory.

---

## 2. Category Breakdown

### 2.1 Cross-Split Leakage (4,989 Records)
* **Finding:** 1,150 identical image hashes are shared between the `train` and `test` splits; 1,148 hashes are shared between `train` and `valid`.
* **Impact:** Evaluating models on the pre-existing splits yields falsely inflated test accuracy because **~24% of the test set is already in the training set**.
* **Remediation Required in Step 03 / Step 04:** Deduplicate the dataset based on image hash before splitting, or ensure all duplicates of an image are restricted to the training set or discarded.

### 2.2 Cross-Label Conflicts (2,509 Records)
* **Finding:** 2,509 records share identical image hashes across opposing fracture classes (`Positive` vs `Negative`).
  * *Example:* 
    * `Arm_patient29529_Positive_001.png` (Label: Positive)
    * `Arm_patient31682_Negative_001.png` (Label: Negative)
    * Both share MD5: `f425541dda85a5884f508515fd16d33e`
* **Impact:** Conflicting ground-truth labels directly inject gradient noise and corrupt model convergence for fracture classification.
* **Remediation Required in Step 03:** These conflicting pairs must be purged or reviewed by clinical radiologists before training.

### 2.3 Anatomical Region & Metadata Mismatch (1 Record)
* **Image Path:** `E:/BoneFract A Bone Fracture Dataset/train/wrist/patient26944/Positive/Arm_patient26944_Positive_001.png`
* **Issue:** Folder location is `wrist/`, but filename is prefixed with `Arm_`.
* **Remediation Required in Step 03:** Resolve anatomical ground truth for this image (wrist vs. distal forearm/arm) or exclude from training.

### 2.4 Channel Inconsistencies (647 Records)
* **Finding:** While 47,284 images are 3-channel RGB:
  * **645 images** are 1-channel Grayscale (`L` mode).
  * **2 images** are 4-channel RGBA (`RGBA` mode).
* **Impact:** Unhandled channel disparities will trigger shape-mismatch runtime exceptions during PyTorch/TensorFlow tensor batching.
* **Remediation Required in Step 05:** Enforce unified channel conversion (e.g., standardizing all inputs to 3-channel RGB or 1-channel Grayscale via a deterministic preprocessing transform).

---

## 3. Data Integrity Artifacts
* Full audit log: [`dataset_audit.csv`](dataset_audit.csv)
* Questionable records: [`reports/questionable_records.csv`](reports/questionable_records.csv)
* Aggregate summary: [`dataset_summary.csv`](dataset_summary.csv)
