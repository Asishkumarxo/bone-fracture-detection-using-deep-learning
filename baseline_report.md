# Multi-Task Baseline Deep Learning Report
**Project:** Bone Fracture Image Captioning & Classification  
**Task:** Anatomical Region & Binary Fracture Classification Baseline  
**Model Architecture:** Multi-Task ResNet-50 (`Shared Backbone + Dual Prediction Heads`)  
**Evaluation Target:** Untouched Test Split (`test.csv`)  
**Date:** September 2026  

---

## 1. Executive Summary

This report documents the implementation, training, and evaluation of the first baseline multi-task deep learning model on the leak-free, patient-split BoneFract dataset.

The system takes a single bone X-ray image and simultaneously predicts:
1. **Anatomical Region** (7-class multi-class classification: *Arm, Foot, Hand, Lower leg, Thigh, pelvis, wrist*).
2. **Fracture Presence** (Binary classification: *0 = Negative/Normal, 1 = Positive/Fracture*).

On the untouched test set, the baseline achieved:
* **Fracture ROC-AUC:** **0.8053** (PR-AUC: **0.7034**, Accuracy: **76.00%**, Specificity: **82.87%**, Sensitivity: **59.03%**).
* **Anatomical Region Classification:** **72.20% Accuracy** (Macro Precision: **73.02%**, Macro F1: **0.7018**).
* **Strongest Fracture Detection Anatomies:** Thigh (**0.927 AUC**, **0.806 F1**), Arm (**0.860 AUC**), and wrist (**0.804 AUC**).

All artifacts have been validated and saved:
* Model Checkpoint: [`best_model.pt`](best_model.pt)
* Epoch Progress Log: [`training_history.csv`](training_history.csv)
* Complete Test Metrics: [`metrics.json`](metrics.json)
* Structured Report Table: [`classification_report.csv`](classification_report.csv)
* Confusion Matrix Visuals: [`confusion_matrix.png`](confusion_matrix.png)
* Reproducible Evaluation Script: [`evaluate.py`](evaluate.py)

---

## 2. Model Architecture

The model utilizes a **shared convolutional feature extractor** with two parallel, task-specific classification heads:

```
                          Input X-Ray Image (3 × 224 × 224)
                                         │
                                         ▼
                      Pretrained ResNet-50 Shared Backbone
                          (conv1 -> layer1-4 -> avgpool)
                                         │
                                         ▼
                             Shared 2048-d Feature Vector
                                         │
                    ┌────────────────────┴────────────────────┐
                    ▼                                         ▼
         Anatomical Region Head                      Fracture Detection Head
   Linear(2048, 512) -> BN -> ReLU           Linear(2048, 256) -> BN -> ReLU
        -> Dropout(0.3) -> Linear(512, 7)         -> Dropout(0.3) -> Linear(256, 1)
                    │                                         │
                    ▼                                         ▼
          Region Logits (7 classes)                 Fracture Logit (1 logit)
```

1. **Backbone:** ResNet-50 initialized with official ImageNet weights (`ResNet50_Weights.DEFAULT`).
2. **Region Head:** `Linear(2048, 512) -> BatchNorm1d(512) -> ReLU -> Dropout(0.3) -> Linear(512, 7)`.
3. **Fracture Head:** `Linear(2048, 256) -> BatchNorm1d(256) -> ReLU -> Dropout(0.3) -> Linear(256, 1)`.

---

## 3. Preprocessing & Normalization

* **Image Dimensions:** Standardized to **224 × 224 pixels** using aspect-ratio preserving resizing with symmetric constant black padding (`PIL.Image.Resampling.BILINEAR`).
* **Channels:** Images are loaded in grayscale and broadcast across 3 RGB channels to match standard ImageNet filter weights.
* **Normalization:** Scaled from uint8 `[0, 255]` to float32 `[0.0, 1.0]`, then standardized via `(x - 0.5) / 0.5` mapping pixels to `[-1.0, 1.0]`.

---

## 4. Training Strategy & Optimization

### Two-Stage Transfer Learning:
* **Stage 1 (Frozen Backbone):**  
  Backbone layers `conv1` through `layer4` were frozen (`param.requires_grad = False`). Only the 1,579,016 parameters across both heads were trained to establish stable task representations without corrupting pretrained filters.
* **Stage 2 (Upper Backbone Fine-Tuning):**  
  `layer4` of ResNet-50 was unfrozen (16,543,752 trainable parameters). An AdamW optimizer with differential learning rates was applied:
  - Backbone `layer4`: `lr = 5e-5`
  - Prediction Heads: `lr = 1e-4`

### Class Imbalance Mitigation:
The training distribution exhibits a **2.438 : 1** negative-to-positive ratio (27,205 Negative vs 11,161 Positive).  
To counteract this without altering dataset distribution:
* **Loss Formulation:** Binary Cross-Entropy with positive class weighting (`pos_weight = 2.438`).
$$\mathcal{L}_{\text{fracture}} = -\left[ w_{\text{pos}} \cdot y \log(\sigma(\hat{y})) + (1-y)\log(1 - \sigma(\hat{y})) \right]$$
* **Multi-Task Objective:**
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{region}} + \mathcal{L}_{\text{fracture}}$$
where $\mathcal{L}_{\text{region}}$ is standard Multi-Class Cross-Entropy Loss.

### Hyperparameters:
* **Optimizer:** AdamW (weight decay = $1 \times 10^{-4}$)
* **Learning Rate Scheduler:** CosineAnnealingLR (Stage 1), ReduceLROnPlateau (Stage 2)
* **Batch Size:** 32
* **Dropout:** 0.30
* **Random Seed:** 42

---

## 5. Training Augmentation Specification

Augmentations were applied **strictly on-the-fly during training**:
* Small rotation: `[-10°, +10°]`
* Small translation: `[-5%, +5%]`
* Small scale variation: `[0.95, 1.05]`
* Mild brightness & contrast jitter: `[0.90, 1.10]` (±10%)
* **Prohibited:** No vertical flipping, no heavy elastic warping, no destructive cropping, no colorization.
* **Validation & Test:** Strictly deterministic (zero augmentation).

---

## 6. Test Set Evaluation Metrics

### Task 1: Binary Fracture Detection

| Metric | Score | Clinical Meaning |
| :--- | :---: | :--- |
| **ROC-AUC** | **0.8053** | Strong ranking ability across discrimination thresholds |
| **PR-AUC** | **0.7034** | High precision retention under class imbalance |
| **Overall Accuracy** | **76.00%** | Fraction of correct classifications |
| **Specificity** | **82.87%** | True Negative Rate (correctly ruling out normal bone) |
| **Sensitivity (Recall)**| **59.03%** | True Positive Rate (correctly identifying fracture) |
| **Precision** | **58.22%** | Positive Predictive Value |
| **F1-Score** | **0.5862** | Harmonic mean of precision and recall |

**Fracture Confusion Matrix (Test Set N=500):**
* **True Negatives (TN):** 295
* **False Positives (FP):** 61
* **False Negatives (FN):** 59
* **True Positives (TP):** 85

---

### Task 2: Anatomical Region Classification

* **Overall Accuracy:** **72.20%**
* **Macro Precision:** **73.02%**
* **Macro Recall:** **70.25%**
* **Macro F1-Score:** **0.7018**

#### Per-Class Performance Breakdown:

| Anatomical Region | Precision | Recall | F1-Score | Support (Images) |
| :--- | :---: | :---: | :---: | :---: |
| **pelvis** | **95.5%** | **87.5%** | **0.913** | 72 |
| **Foot** | **77.1%** | **88.9%** | **0.826** | 72 |
| **Hand** | **72.4%** | **87.3%** | **0.791** | 63 |
| **Arm** | **71.2%** | **72.4%** | **0.718** | 58 |
| **Lower leg** | **63.2%** | **79.1%** | **0.703** | 115 |
| **Thigh** | **75.0%** | **36.8%** | **0.494** | 57 |
| **wrist** | **56.8%** | **39.7%** | **0.467** | 63 |

---

### Stratified Fracture Performance by Anatomical Region

| Anatomical Region | Accuracy | Sensitivity (Recall) | F1-Score | ROC-AUC | Region Sample Size |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Thigh** | **78.9%** | **89.3%** | **0.806** | **0.927** | 57 |
| **Arm** | **89.7%** | **50.0%** | **0.625** | **0.860** | 58 |
| **wrist** | **74.6%** | **57.9%** | **0.579** | **0.804** | 63 |
| **Lower leg** | **69.6%** | **61.9%** | **0.598** | **0.788** | 115 |
| **pelvis** | **84.7%** | **38.5%** | **0.476** | **0.720** | 72 |
| **Hand** | **77.8%** | **27.3%** | **0.300** | **0.701** | 63 |
| **Foot** | **63.9%** | **47.6%** | **0.435** | **0.654** | 72 |

---

## 7. Limitations & Recommended Next Steps

1. **Anatomical Boundary Confusion (Wrist vs. Arm, Thigh vs. Pelvis):**  
   Distal forearm X-rays and proximal femoral radiographs share overlapping boundaries. Adding multi-label or hierarchical anatomy modeling will resolve border ambiguities.
2. **Subtle Hairline Fractures in Complex Joints (Hand & Foot):**  
   Hand and foot fractures have lower recall (27.3% and 47.6%) due to multiple overlapping small tarsal/carpal bones. Higher input resolution (e.g. 384×384 or 512×512) and feature pyramid attention (FPN) will capture micro-cortical interruptions.
3. **Transition to Fracture Localization (STEP 10 & 11):**  
   Integrating bounding-box localization (e.g. with FracAtlas dataset) will direct feature attention to the exact lesion site, further boosting both classification and structured caption generation.
