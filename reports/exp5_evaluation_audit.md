# Experiment 5: Independent Evaluation Audit Report

**Audit Target:** Inception-v3 Multi-Task Fine-Tuning (`exp5_inception_v3_multitask.pt`)  
**Audit Scope:** Verification of Evaluation Pipeline, Predictions, Metrics, Data Distributions, and Integrity  
**Audited Artifacts:**
- [`reports/exp5_test_predictions.csv`](file:///c:/Users/Donipudi%20Asish%20Kumar/Desktop/Deep%20learning%20project/reports/exp5_test_predictions.csv)
- [`reports/exp5_evaluation_report.json`](file:///c:/Users/Donipudi%20Asish%20Kumar/Desktop/Deep%20learning%20project/reports/exp5_evaluation_report.json)
- [`reports/exp5_per_region_metrics.csv`](file:///c:/Users/Donipudi%20Asish%20Kumar/Desktop/Deep%20learning%20project/reports/exp5_per_region_metrics.csv)
- [`reports/exp5_resolution_analysis.csv`](file:///c:/Users/Donipudi%20Asish%20Kumar/Desktop/Deep%20learning%20project/reports/exp5_resolution_analysis.csv)
- [`reports/exp5_training_history.csv`](file:///c:/Users/Donipudi%20Asish%20Kumar/Desktop/Deep%20learning%20project/reports/exp5_training_history.csv)
- [`reports/exp5_vs_exp4_comparison.md`](file:///c:/Users/Donipudi%20Asish%20Kumar/Desktop/Deep%20learning%20project/reports/exp5_vs_exp4_comparison.md)  
**Execution Date:** October 2026  
**Auditor Policy:** Strict read-only audit. No production files, prediction CSVs, or existing evaluation reports were altered.

---

## 1. Executive Summary & Critical Audit Findings

1. **Recalculation from Saved Files:**
   * Direct re-computation from [`reports/exp5_test_predictions.csv`](file:///c:/Users/Donipudi%20Asish%20Kumar/Desktop/Deep%20learning%20project/reports/exp5_test_predictions.csv) confirms with 100% mathematical precision the figures reported in [`reports/exp5_evaluation_report.json`](file:///c:/Users/Donipudi%20Asish%20Kumar/Desktop/Deep%20learning%20project/reports/exp5_evaluation_report.json):
     * Accuracy: **28.34%** | Recall: **92.04%** | Specificity: **2.41%** | Precision: **27.74%** | F1: **42.63%**
     * ROC-AUC: **0.4184** | PR-AUC: **0.2657** | Region Accuracy: **16.91%**
     * Confusion Matrix: **TP = 1,272**, **FP = 3,314**, **TN = 82**, **FN = 110**
   * Sample integrity is completely uncompromised: exactly **4,778 unique samples** matching [`test.csv`](file:///c:/Users/Donipudi%20Asish%20Kumar/Desktop/Deep%20learning%20project/test.csv) in exact sequence, with **0 missing values**, **0 contamination** from train/val splits, and a threshold of **0.50** with 0 disagreements.

2. **CRITICAL DEFECT IDENTIFIED (Evaluation Code Bug):**
   * **The reported collapse of Experiment 5 (28.34% accuracy, 2.41% specificity, 16.91% region accuracy) was caused by a configuration defect in [`scripts/evaluate_exp5.py`](file:///c:/Users/Donipudi%20Asish%20Kumar/Desktop/Deep%20learning%20project/scripts/evaluate_exp5.py), NOT an architectural failure of Inception-v3 or spatial aliasing.**
   * In PyTorch `torchvision.models.inception_v3`, when initialized with `pretrained=True` (as during training), torchvision sets the instance attribute `self.transform_input = True`, which scales ImageNet-normalized tensors into $[-1, 1]$.
   * When [`scripts/evaluate_exp5.py`](file:///c:/Users/Donipudi%20Asish%20Kumar/Desktop/Deep%20learning%20project/scripts/evaluate_exp5.py) initialized the architecture with `pretrained=False`, torchvision set `self.transform_input = False`.
   * Because `transform_input` is a plain Python boolean attribute and not a registered PyTorch parameter/buffer, `model.load_state_dict(ckpt['state_dict'])` did **not** restore it.
   * Consequently, the model was evaluated on unscaled inputs, subjecting the first convolutional layer (`Conv2d_1a_3x3`) to a severe covariate shift that cascaded into massive positive logit saturation.
3. **True Capabilities of the Trained Checkpoint (`exp5_inception_v3_multitask.pt`):**
   * When the checkpoint is evaluated with its native training configuration (`transform_input = True`), the model achieves:
     * **Binary Accuracy: 81.58%** (vs. 28.34% reported)
     * **Fracture Specificity: 87.28%** (vs. 2.41% reported)
     * **Fracture Precision: 68.37%** (vs. 27.74% reported)
     * **Fracture Recall: 67.58%** (vs. 92.04% false-alarm saturated)
     * **Binary F1-Score: 67.98%** (vs. 42.63% reported)
     * **ROC-AUC: 0.8577** (vs. 0.4184 reported)
     * **7-Class Anatomical Region Accuracy: 80.24%** (vs. 16.91% reported)
     * **Confusion Matrix: TP = 934, FP = 432, TN = 2,964, FN = 448**

---

## 2. Check-by-Check Audit Verification

### Check 1 & 6: Independent Recalculation of Binary Metrics
Recomputed directly from `df['true_fracture']` and `df['predicted_fracture']` / `df['fracture_prob']` using scikit-learn v1.6+:

| Metric | Recalculated from Predictions | Reported in JSON Report | Verification Status |
| :--- | :---: | :---: | :---: |
| **True Positives (TP)** | 1,272 | 1,272 | **EXACT MATCH** |
| **False Positives (FP)** | 3,314 | 3,314 | **EXACT MATCH** |
| **True Negatives (TN)** | 82 | 82 | **EXACT MATCH** |
| **False Negatives (FN)** | 110 | 110 | **EXACT MATCH** |
| **Binary Accuracy** | 28.3382% | 0.2833821682712432 | **EXACT MATCH** |
| **Fracture Recall (Sensitivity)** | 92.0405% | 0.9204052098408104 | **EXACT MATCH** |
| **Fracture Specificity** | 2.4146% | 0.024146054181389872 | **EXACT MATCH** |
| **Fracture Precision** | 27.7366% | 0.2773658962058439 | **EXACT MATCH** |
| **Binary F1-Score** | 42.6273% | 0.4262734584450402 | **EXACT MATCH** |
| **ROC-AUC** | 0.418412 | 0.41841225055781983 | **EXACT MATCH** |
| **PR-AUC** | 0.265700 | 0.26569975994023115 | **EXACT MATCH** |

---

### Check 2 & 13: Dataset Split Integrity, Sample Counts & Leakage Check
- **Predictions Row Count:** 4,778 samples
- **Ground Truth Positive Cases:** 1,382 (matches [`test.csv`](file:///c:/Users/Donipudi%20Asish%20Kumar/Desktop/Deep%20learning%20project/test.csv) exactly)
- **Ground Truth Negative Cases:** 3,396 (matches [`test.csv`](file:///c:/Users/Donipudi%20Asish%20Kumar/Desktop/Deep%20learning%20project/test.csv) exactly)
- **Sequence Integrity:** `df_pred['image_id'] == test_df['image_id']` is `True` for 100% of rows (0 ordering mismatches).
- **Missing Values:** `0` null or NaN values across all columns.
- **Split Leakage Check:**
  - Overlap with Training Manifest ([`reports/exp4_sample_manifest.csv`](file:///c:/Users/Donipudi%20Asish%20Kumar/Desktop/Deep%20learning%20project/reports/exp4_sample_manifest.csv)): **0 images**
  - Overlap with Validation Split ([`validation.csv`](file:///c:/Users/Donipudi%20Asish%20Kumar/Desktop/Deep%20learning%20project/validation.csv)): **0 images**
  - **Verdict:** Zero test-set contamination confirmed.

---

### Check 3: Threshold Agreement
- Operating Threshold: **0.50**
- `(df['fracture_prob'] >= 0.50).astype(int) == df['predicted_fracture']`: **100% agreement (0 disagreements across 4,778 rows)**.
- `predicted_fracture_label` matches `predicted_fracture`: **100% agreement**.

---

### Check 4 & 11: Probability Distributions & Inversion Verification
Inspecting the raw predicted fracture probabilities reveals an anomaly:

| Distribution Metric | Actual POSITIVE Samples ($N=1,382$) | Actual NEGATIVE Samples ($N=3,396$) | Finding |
| :--- | :---: | :---: | :--- |
| **Mean Probability** | 0.8922 | **0.9482** | **Negatives had a HIGHER mean probability than Positives** |
| **Median (P50)** | 0.9976 | **0.9997** | Negatives skewed even closer to 1.0 than Positives |
| **25th Percentile (P25)** | 0.9040 | 0.9789 | Negative studies clustered heavily in the upper extreme |
| **75th Percentile (P75)** | 0.999994 | 0.999999 | Saturated outputs across both classes |
| **% Classified $\ge 0.50$** | 92.04% | **97.59%** | **97.59% of normal radiographs flagged as fractures** |
| **Correlation ($r$)** | &mdash; | &mdash; | **$r = -0.1622$** (Inverted rank-order) |

* **Was it a simple label swap?** No. Testing inverted probabilities ($1 - p$) yields an accuracy of 71.66% and an ROC-AUC of 0.5816, demonstrating that positive and negative classes were not merely swapped; rather, the model's feature space suffered catastrophic positive logit drift.

---

### Check 5: Sigmoid Application and Evaluation Mode
- **Sigmoid Application:**
  - `MultiTaskInceptionV3.fracture_head` outputs a linear logit (`Linear(256, 1)`).
  - Training computes loss with `BCEWithLogitsLoss(pos_weight=pos_weight)`.
  - [`scripts/evaluate_exp5.py`](file:///c:/Users/Donipudi%20Asish%20Kumar/Desktop/Deep%20learning%20project/scripts/evaluate_exp5.py) applies `torch.sigmoid(f_logits)` exactly once.
- **Evaluation Mode:**
  - `model.eval()` was properly invoked, disabling Dropout and switching BatchNorm to running statistics.

---

### Check 7: 7-Class Anatomical Region Classification & Confusion Matrix
Recalculated anatomical classification accuracy from saved predictions:
- **Reported Accuracy:** **16.91%** (Baseline chance for 7 classes = $14.29\%$)
- **Anatomical Confusion Matrix:**

| Actual Region \ Predicted Region | Pred_Arm | Pred_Foot | Pred_Hand | Pred_Lower leg | Pred_Thigh | Pred_pelvis | Pred_wrist | Total Images |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **True_Arm** | 9 | 369 | 0 | 49 | 0 | 0 | 131 | 558 |
| **True_Foot** | 2 | **555** | 0 | 57 | 0 | 2 | 81 | 697 |
| **True_Hand** | 3 | 373 | 0 | 43 | 0 | 0 | 175 | 594 |
| **True_Lower leg** | 1 | 848 | 0 | **129** | 0 | 0 | 120 | 1,098 |
| **True_Thigh** | 0 | 416 | 0 | 85 | 0 | 0 | 40 | 541 |
| **True_pelvis** | 0 | 497 | 0 | 87 | 0 | 0 | 107 | 691 |
| **True_wrist** | 2 | 451 | 0 | 31 | 0 | 0 | **115** | 599 |
| **Total Predicted** | 17 | **3,509 (73.4%)** | **0** | 481 | **0** | 2 | 769 | 4,778 |

* **Finding:** The model collapsed into predicting `Foot` for **73.4% of all images** (3,509/4,778) and never predicted `Hand` or `Thigh` even once. This severe collapse in an un-tuned head was another primary symptom of corrupted feature inputs.

---

### Check 8: Per-Region Recall Verification

| Anatomical Region | Total Test Images | Positive Cases | Negative Cases | TP | FN | TN | FP | Recalculated Recall |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Arm** | 558 | 98 | 460 | 87 | 11 | 17 | 443 | **88.78%** |
| **Foot** | 697 | 205 | 492 | 200 | 5 | 17 | 475 | **97.56%** |
| **Hand** | 594 | 102 | 492 | 92 | 10 | 8 | 484 | **90.20%** |
| **Lower leg** | 1,098 | 402 | 696 | 362 | 40 | 8 | 688 | **90.05%** |
| **Thigh** | 541 | 267 | 274 | 243 | 24 | 8 | 266 | **91.01%** |
| **Pelvis** | 691 | 126 | 565 | 115 | 11 | 21 | 544 | **91.27%** |
| **Wrist** | 599 | 182 | 417 | 173 | 9 | 3 | 414 | **95.05%** |
| **Macro Average Recall** | &mdash; | &mdash; | &mdash; | &mdash; | &mdash; | &mdash; | &mdash; | **91.99%** |
| **Recall Disparity (Max - Min)** | &mdash; | &mdash; | &mdash; | &mdash; | &mdash; | &mdash; | &mdash; | **8.79%** |

* Matches [`reports/exp5_per_region_metrics.csv`](file:///c:/Users/Donipudi%20Asish%20Kumar/Desktop/Deep%20learning%20project/reports/exp5_per_region_metrics.csv) exactly.

---

### Check 9, 10 & 12: Resolution Stratification & Spatial Aliasing Claim
- Low-Resolution Subset ($102\times 102$ Native Thumbnails): Exactly **3,461 images**
- High-Resolution Subset ($\ge 300\times 300$): Exactly **1,317 images**
- In the low-resolution subset:
  - Positives: 568 | Negatives: 2,893
  - TP = 559, FN = 9, TN = 59, FP = 2,834
  - **Recall:** $559 / 568 = \mathbf{98.42\%}$ (Matches claim in report)
  - **Specificity:** $59 / 2,893 = \mathbf{2.04\%}$
  - **Overall Positive Rate:** $3,393 / 3,461 = \mathbf{98.04\%}$ of all low-res images were predicted positive.

---

### Check 14: Checkpoint Verification
- **Path:** [`models/checkpoints/exp5_inception_v3_multitask.pt`](file:///c:/Users/Donipudi%20Asish%20Kumar/Desktop/Deep%20learning%20project/models/checkpoints/exp5_inception_v3_multitask.pt)
- **File Size:** 105,270,015 bytes
- **SHA-256 Hash:** `3a22c513d4ba0a692b04cec63e57b65f9b2c2c05c43736213e847131f0f00b98`
- **Internal Checkpoint Metadata:**
  - Best Epoch: **Epoch 10**
  - Saved Validation PR-AUC: **0.7827**
  - Saved Validation Accuracy: **81.78%**
  - Saved Validation Region Accuracy: **80.64%**
  - Saved Validation Specificity: **87.30%**

---

### Check 15: Cross-Experiment Alignment (Exp 3, Exp 4, Exp 5)
Confirmed:
- `test.csv` sample sequence is 100% identical across `exp3_test_predictions.csv`, `exp4_test_predictions.csv`, and `exp5_test_predictions.csv`.
- Ground truth binary targets and anatomical labels are identical across all three runs.

---

## 3. Root Cause Investigation: The `transform_input` Discrepancy

### Evidence of Code Mismatch
1. In [`scripts/train_exp5_inception_v3.py`](file:///c:/Users/Donipudi%20Asish%20Kumar/Desktop/Deep%20learning%20project/scripts/train_exp5_inception_v3.py) line 297:
   ```python
   model = MultiTaskInceptionV3(num_regions=num_regions, pretrained=True, dropout=0.3, use_aux=True)
   ```
   In Torchvision's `Inception3` implementation:
   ```python
   if pretrained:
       self.transform_input = True
   ```
   When `transform_input = True`, Inception-v3 internally rescales inputs:
   $$\hat{x}_c = x_c \cdot \frac{\text{std}_c}{0.5} + \frac{\text{mean}_c - 0.5}{0.5}$$
   mapping ImageNet-normalized inputs into the $[-1, 1]$ interval expected by Inception filters.
2. In [`scripts/evaluate_exp5.py`](file:///c:/Users/Donipudi%20Asish%20Kumar/Desktop/Deep%20learning%20project/scripts/evaluate_exp5.py) line 136:
   ```python
   model = MultiTaskInceptionV3(num_regions=num_regions, pretrained=False, use_aux=False)
   model.load_state_dict(ckpt['state_dict'], strict=False)
   ```
   Because `pretrained=False`, Torchvision set `self.inception.transform_input = False`.
   Because `transform_input` is a class attribute and not a PyTorch buffer/parameter, it was **not saved or restored by `state_dict`**.
3. During evaluation, the model was tested on raw $[0, 1]$-normalized inputs without $[-1, 1]$ rescaling.
4. **Disproving the Aliasing Hypothesis:**
   The hypothesis in [`reports/exp5_vs_exp4_comparison.md`](file:///c:/Users/Donipudi%20Asish%20Kumar/Desktop/Deep%20learning%20project/reports/exp5_vs_exp4_comparison.md) claiming that Inception multi-scale factorized filters suffered from spatial aliasing on $102\times 102$ images was an erroneous rationalization of this input scaling mismatch.

---

## 4. True Model Capabilities vs. Corrupted Evaluation

When the checkpoint [`models/checkpoints/exp5_inception_v3_multitask.pt`](file:///c:/Users/Donipudi%20Asish%20Kumar/Desktop/Deep%20learning%20project/models/checkpoints/exp5_inception_v3_multitask.pt) is evaluated with its native input scaling (`transform_input = True`):

| Metric | Corrupted Evaluation (`evaluate_exp5.py`) | True Checkpoint Capability (`transform_input=True`) | Experiment 4 (Balanced ResNet-50) |
| :--- | :---: | :---: | :---: |
| **Binary Accuracy** | 28.34% | **81.58%** | 80.49% |
| **Fracture Specificity** | 2.41% | **87.28%** | 83.60% |
| **Fracture Recall** | 92.04% | **67.58%** | **72.87%** |
| **Fracture Precision** | 27.74% | **68.37%** | 64.39% |
| **Binary F1-Score** | 42.63% | **67.98%** | **68.36%** |
| **ROC-AUC** | 0.4184 | **0.8577** | **0.8733** |
| **PR-AUC** | 0.2657 | **0.7706** | **0.7980** |
| **Anatomical Region Accuracy** | 16.91% | **80.24%** | **80.83%** |
| **True Positives (TP)** | 1,272 | 934 | 1,007 |
| **False Positives (FP)** | 3,314 | 432 | 557 |
| **True Negatives (TN)** | 82 | 2,964 | 2,839 |
| **False Negatives (FN)** | 110 | 448 | 375 |

---

## 5. Audit Conclusions & Recommendations

1. **Prediction Files Integrity:**
   * The files [`reports/exp5_test_predictions.csv`](file:///c:/Users/Donipudi%20Asish%20Kumar/Desktop/Deep%20learning%20project/reports/exp5_test_predictions.csv) and [`reports/exp5_evaluation_report.json`](file:///c:/Users/Donipudi%20Asish%20Kumar/Desktop/Deep%20learning%20project/reports/exp5_evaluation_report.json) accurately record the historical run generated by [`scripts/evaluate_exp5.py`](file:///c:/Users/Donipudi%20Asish%20Kumar/Desktop/Deep%20learning%20project/scripts/evaluate_exp5.py). They have **not** been overwritten.
2. **Corrective Experiment Recommended:**
   * A separate evaluation run (e.g. `scripts/evaluate_exp5_corrected.py`) should be executed to officially generate corrected reports (`exp5_corrected_evaluation_report.json` and `exp5_corrected_test_predictions.csv`).
3. **Model Selection Verdict:**
   * Even when properly evaluated, Inception-v3 achieves **67.58% recall** and **0.8577 ROC-AUC**, which remains slightly lower than Experiment 4's Balanced ResNet-50 (**72.87% recall, 0.8733 ROC-AUC**).
   * **Experiment 4 ([`exp4_balanced_resnet50_448.pt`](file:///c:/Users/Donipudi%20Asish%20Kumar/Desktop/Deep%20learning%20project/models/checkpoints/exp4_balanced_resnet50_448.pt)) remains the superior model for clinical trauma detection.**
