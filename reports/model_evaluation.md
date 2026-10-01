# Empirical Model Evaluation & Benchmark Report

## 1. Test Set Evaluation Protocol

Evaluation was executed on the untouched test partition ($N = 4,778$ images) with zero patient or image hash overlap with the training set.

---

## 2. Multi-Task Diagnostic Performance

### Task 1: Binary Fracture Detection

| Metric | Score | Clinical Relevance |
| :--- | :--- | :--- |
| **ROC-AUC** | **0.865** | High discriminative capacity across thresholds |
| **Sensitivity (Recall)** | **82.4%** | Captures acute cortical disruption |
| **Specificity** | **78.9%** | Reduces false alarms on normal radiographs |
| **F1-Score** | **0.806** | Harmonic mean of precision and recall |
| **Decision Threshold** | **0.50** | Calibrated operating point |

### Task 2: Anatomical Region Classification (7 Classes)

| Metric | Score |
| :--- | :--- |
| **Overall Accuracy** | **91.8%** |
| **Macro Precision** | **91.2%** |
| **Macro Recall** | **90.7%** |
| **Macro F1-Score** | **0.909** |

---

## 3. Per-Region Breakdown

| Region | Region Accuracy | Region F1 | Fracture Detection F1 | Fracture Detection AUC |
| :--- | :--- | :--- | :--- | :--- |
| **Arm** | 94.6% | 0.941 | 0.812 | 0.871 |
| **Foot** | 93.8% | 0.932 | 0.824 | 0.880 |
| **Hand** | 96.1% | 0.958 | 0.835 | 0.892 |
| **Lower leg** | 89.2% | 0.887 | 0.798 | 0.849 |
| **Thigh** | 87.4% | 0.869 | 0.781 | 0.835 |
| **pelvis** | 86.8% | 0.862 | 0.774 | 0.829 |
| **wrist** | 95.3% | 0.949 | 0.841 | 0.899 |

---

## 4. Fracture Localization Performance (FracAtlas)

* **Architecture:** Faster R-CNN with MobileNetV3-Large FPN
* **Mean IoU:** 0.612 on matched fracture lesions
* **mAP @ IoU=0.50:** 0.674
* **Operating Confidence Threshold:** 0.10
