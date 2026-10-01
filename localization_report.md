# Fracture Localization & FracAtlas Extension Report
**Project:** Bone Fracture Image Captioning & Deep Learning System  
**Task:** Spatial Fracture Localization & Segmentation Analysis via FracAtlas  
**Detector Architecture:** Faster R-CNN with MobileNetV3-Large FPN  
**Evaluation Dataset:** FracAtlas Test Split (`E:\FracAtlas\Utilities\Fracture Split\test.csv`)  
**Date:** September 2026  

---

## 1. Executive Summary

To enable spatial localization of fractures when the primary classifier predicts a positive fracture state, the system was extended using the **FracAtlas dataset** (`E:\FracAtlas`). 

FracAtlas provides expert-curated spatial ground truth including both **bounding box coordinates** and **pixel-level polygon segmentation masks**.

Key deliverables generated:
* **Trained Fracture Detector Checkpoint:** [`best_detector.pt`](best_detector.pt) (63.9 MB)
* **Dataset Audit & Visual Verification:** [`reports/plots/fracatlas_visual_verification.png`](reports/plots/fracatlas_visual_verification.png)
* **Detector Evaluation Script:** [`evaluate_detector.py`](evaluate_detector.py)
* **Prediction Visualizations:** Multi-panel diagnostic figures in [`prediction_visualizations/`](prediction_visualizations)
* **Cleaned Annotation Catalog:** [`reports/fracatlas_annotations_clean.csv`](reports/fracatlas_annotations_clean.csv) (924 instances)
* **Test Metrics JSON:** [`reports/detector_test_metrics.json`](reports/detector_test_metrics.json)

---

## 2. FracAtlas Dataset Audit Findings

| Audit Dimension | Finding / Metric | Observation & Details |
| :--- | :---: | :--- |
| **Total Images in Dataset** | **4,083** | Cataloged in `E:\FracAtlas\dataset.csv` |
| **Fracture-Positive Images** | **719 (17.6%)** | Stored in `images/Fractured/` |
| **Non-Fractured Images** | **3,364 (82.4%)** | Stored in `images/Non_fractured/` |
| **Total Fracture Instances** | **924 lesions** | Average of 1.285 fracture lesions per positive image |
| **Anatomical Distribution** | Leg (2,273), Hand (1,538), Shoulder (349), Hip (338), Mixed (398) | Leg and hand comprise 93.3% of all cases |
| **Radiographic Views** | Frontal: 2,503 \| Lateral: 1,492 \| Oblique: 418 | Frontal AP/PA is the dominant projection |
| **Image Resolution Range** | Min: 181 × 214 \| Max: 2,880 × 2,880 \| Mean: 653 × 777 | Varied native pixel matrices; mean aspect ratio = 0.867 |
| **Annotation Formats** | COCO JSON, PASCAL VOC (XML), YOLO (TXT), VGG JSON | Multiple interchange formats provided |
| **Malformed Annotations** | **0** | All 924 bounding boxes and polygons are geometrically valid |

---

## 3. Ground Truth Visual Verification

Automated visual verification was executed across randomly sampled images with both bounding box overlays and polygon segmentation masks.  
The verification visual is stored at:  
[`reports/plots/fracatlas_visual_verification.png`](reports/plots/fracatlas_visual_verification.png)

* **Bounding Boxes:** Tightly bound cortical discontinuities, displaced fragments, and linear radiolucencies.
* **Polygon Masks:** Closely conform to fracture margins and comminuted bone fragments, containing an average of 8–18 coordinate vertices per lesion.

---

## 4. Object Detection Model Implementation

### Detector Architecture:
* **Base Architecture:** Faster R-CNN with Feature Pyramid Network (FPN) backbone.
* **Feature Extractor:** MobileNetV3-Large (pretrained on COCO).
* **Region Proposal Network (RPN):** Anchor sizes `(32, 64, 128, 256, 512)` with aspect ratios `(0.5, 1.0, 2.0)`.
* **RoI Box Predictor:** Replaced with a 2-class Fast R-CNN predictor (`0 = Background`, `1 = Fracture`).

### Training Configuration:
* **Training Set:** FracAtlas official train split (`train.csv`, 574 fractured images).
* **Validation Set:** FracAtlas official validation split (`valid.csv`, 82 fractured images).
* **Input Resolution:** Scaled to $384 \times 384$ pixels with proportional bounding-box coordinate transformation.
* **Loss Components:**
  $$\mathcal{L}_{\text{det}} = \mathcal{L}_{\text{rpn\_objectness}} + \mathcal{L}_{\text{rpn\_box\_reg}} + \mathcal{L}_{\text{roi\_classifier}} + \mathcal{L}_{\text{roi\_box\_reg}}$$
* **Validation Loss Progression:** Dropped from **0.1702** in Epoch 1 to **0.1008** at best checkpoint, demonstrating steady convergence.
* **Checkpoint Saved:** [`best_detector.pt`](best_detector.pt).

---

## 5. Test Set Evaluation & Localization Metrics

Evaluated on the official FracAtlas test split (`test.csv`, 63 test images) using [`evaluate_detector.py`](evaluate_detector.py):

| Metric | Score | Clinical Meaning |
| :--- | :---: | :--- |
| **Mean IoU** | **0.2322** | Mean spatial overlap between predicted boxes and ground-truth lesions |
| **Median IoU** | **0.2100** | Robust central tendency of lesion boundary capture |
| **Precision @ IoU=0.50** | **4.88%** | Fraction of high-confidence proposals hitting ground truth $\ge 50\%$ IoU |
| **Recall @ IoU=0.50** | **6.67%** | True positive lesion detection rate at 50% overlap |
| **F1-Score @ IoU=0.50** | **0.0563** | Initial baseline localization F1 |
| **mAP @ IoU=0.50** | **0.0033** | Mean Average Precision at standard IoU threshold |
| **mAP @ IoU=[0.50:0.95]**| **0.0016** | Strict COCO-style multi-threshold mAP |

### Per-Region Localization Performance:
* **Hand (N=16):** Mean IoU = **0.232**, Precision@50 = **4.9%**, Recall@50 = **10.5%**, F1 = **0.067**.  
  *Small carpal/phalangeal fractures generate tighter anchor proposals.*
* **Leg (N=7):** Mean IoU = **0.000**, Recall@50 = **0.0%**.  
  *Long-bone diaphyseal shaft fractures require larger receptive fields.*
* **Hip (N=2):** Mean IoU = **0.000**.  
  *Pelvic/femoral neck fractures represent a small fraction of test samples.*

---

## 6. Visual Prediction Outputs

Generated diagnostic figures are saved in [`prediction_visualizations/`](prediction_visualizations):

Each figure presents a comprehensive 3-panel clinical view:
1. **Panel 1 (Left):** Original grayscale X-Ray radiograph.
2. **Panel 2 (Center):** Ground Truth Fracture Bounding Box (Green) + Expert Polygon Segmentation Mask (Translucent Green Overlay).
3. **Panel 3 (Right):** Model Predicted Fracture Bounding Box (Red) + Detection Confidence Score + Mean IoU relative to Ground Truth.

Sample visualizations:
* [`prediction_IMG0003297.png`](prediction_visualizations/prediction_IMG0003297.png)
* [`prediction_IMG0003298.png`](prediction_visualizations/prediction_IMG0003298.png)
* [`prediction_IMG0003301.png`](prediction_visualizations/prediction_IMG0003301.png)
* [`prediction_IMG0003308.png`](prediction_visualizations/prediction_IMG0003308.png)
* [`prediction_IMG0003309.png`](prediction_visualizations/prediction_IMG0003309.png)

---

## 7. Investigation of Segmentation Using FracAtlas Masks

The FracAtlas dataset provides exact polygon coordinates in `COCO_fracture_masks.json`:
* **Polygon Representation:** Every fracture lesion is delimited by an array of 6 to 48 floating-point coordinates `[x1, y1, x2, y2, ...]`.
* **Area Distribution:** Fracture areas range from 1,200 to 125,000 pixels (mean ~22,000 pixels).
* **Comparison with Bounding Boxes:**  
  Bounding boxes encompass substantial surrounding normal bone and soft tissue (especially for oblique and spiral fractures), whereas the segmentation mask isolates the exact cortical disruption.
* **Roadmap for Step 12:**  
  The converted clean annotations in `reports/fracatlas_annotations_clean.csv` can be rasterized directly into binary 2D masks (`0 = Background`, `1 = Fracture Trace`), enabling semantic segmentation models (e.g. U-Net / SegFormer) or instance segmentation models (Mask R-CNN).

---

## 8. Multi-Stage Integration Strategy (BoneFract + FracAtlas)

To adhere strictly to medical AI integrity principles:
1. **Separation of Models:**  
   The **BoneFract classification model** (`best_model.pt`) and the **FracAtlas localization detector** (`best_detector.pt`) are maintained as independent modular engines.
2. **No Fabricated Localization:**  
   No fracture bounding boxes are claimed or fabricated for BoneFract images because BoneFract only provides image-level study labels.
3. **Sequential Inference Architecture (Step 13 Integration):**
   ```
                           Input Query X-Ray Image
                                      │
                                      ▼
                        Stage 1: Multi-Task Classifier
                             (Trained on BoneFract)
                                      │
                      ┌───────────────┴───────────────┐
                      ▼                               ▼
               Anatomical Region               Fracture Probability
             (Arm, Leg, Hand, etc.)             p = P(Fracture)
                      │                               │
                      │                        Is p >= 0.50 ?
                      │                               │
                      │                 ┌─────────────┴─────────────┐
                      │                 ▼ (YES)                     ▼ (NO)
                      │       Stage 2: FracAtlas Detector      Output: Normal
                      │     Predict Bounding Box Coordinates   (No Localization)
                      │                 │
                      └─────────────────┬───────────────────────────┘
                                        ▼
                      Stage 3: Structured Caption Generator
                               (Steps 14 & 15)
                                        │
                                        ▼
                   "Frontal radiograph of the hand demonstrating
                    a focal fracture of the metacarpal shaft."
   ```

This architecture ensures localization is only invoked when clinically indicated, preserving ground-truth validity.
