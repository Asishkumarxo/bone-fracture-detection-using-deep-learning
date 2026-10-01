# Preprocessing & Data Cleaning Report
**Dataset:** BoneFract (Mendeley Data)  
**Execution Date:** September 2026  
**Pipeline Script:** [`preprocess.py`](preprocess.py)  

---

## 1. Summary of Ingestion & Exclusion

| Metric | Value | Notes |
| :--- | :--- | :--- |
| **Original Images Audited** | **47,931** | Direct count from source archive |
| **Number Retained** | **47,931** | 100% of images verified and retained |
| **Number Excluded** | **0** | Only corrupted/unreadable images qualify for removal |
| **Exclusion Reasons** | **None** | All 47,931 images passed raster loading and byte validation |

---

## 2. Patient-Level Splitting & Leakage Prevention

* **Total Patients:** **47,922**
* **Train Patients:** **38,357** (80.0%)
* **Validation Patients:** **4,787** (10.0%)
* **Test Patients:** **4,778** (10.0%)
* **Cross-Split Patient Overlap:** **0** (Zero patient overlap between any splits)
* **Cross-Split Duplicate Image Overlap:** **0** (Zero image hash collisions across splits)

> [!IMPORTANT]
> **Cluster-Level Partitioning:**  
> The 47,931 images were partitioned into 40,663 connected patient clusters linked by image hash equivalence. By treating each cluster as an indivisible atomic unit, the pipeline eradicated the **24.07% test-to-train data leakage** present in the dataset's original default splits.

---

## 3. Distribution by Split

| Split | Total Images | Unique Patients | Negative (Normal) | Positive (Fracture) | % Positive |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Train** | **38,366** | 38,357 | 27,205 | 11,161 | 29.09% |
| **Validation** | **4,787** | 4,787 | 3,409 | 1,378 | 28.79% |
| **Test** | **4,778** | 4,778 | 3,396 | 1,382 | 28.92% |
| **Total** | **47,931** | **47,922** | **34,010** | **13,921** | **29.04%** |

### Images per Anatomical Region (Overall)
* **Lower leg:** 10,996 images (22.9%)
* **pelvis:** 6,983 images (14.6%)
* **Foot:** 6,840 images (14.3%)
* **Hand:** 6,032 images (12.6%)
* **wrist:** 6,020 images (12.6%)
* **Arm:** 5,597 images (11.7%)
* **Thigh:** 5,463 images (11.4%)

---

## 4. Image Processing Specification

* **Final Image Resolution:** **224 × 224 pixels**
* **Channels:** **1-channel Grayscale** (standardized from mixed RGB, Grayscale, and RGBA sources)
* **Aspect Ratio Preservation:** Resized proportionally with symmetric constant black padding (letterbox pad).
* **Interpolation:** Bilinear interpolation (`PIL.Image.Resampling.BILINEAR`)
* **Storage Format:** 8-bit PNG (`L` mode, lossless dynamic range preservation)

---

## 5. Normalization & Augmentation Strategy

### Normalization Method:
* **Tensor Conversion:** Rescaled from `[0, 255]` uint8 to `[0.0, 1.0]` float32.
* **Standardization:** Normalized via `(x - mean) / std` with `mean = [0.5]` and `std = [0.5]` (maps pixel dynamic range to `[-1.0, 1.0]`). For transfer learning models expecting 3 channels, the grayscale channel is broadcast across RGB channels.

### Training Augmentation Configuration (Conservative):
Applied **strictly and exclusively** to training samples on-the-fly during training DataLoader iteration:
* **Small Rotation:** Uniform random rotation within **[-10°, +10°]**.
* **Small Translation:** Uniform random translation within **[-5%, +5%]** of height and width.
* **Small Scale Variation:** Uniform random scaling within **[0.95, 1.05]**.
* **Mild Brightness & Contrast Jitter:** Factor range **[0.90, 1.10]** (±10% variation).

### Prohibited Augmentations:
* **Vertical Flipping:** PROHIBITED (violates anatomical gravity and clinical projection axes).
* **Extreme Rotations / Heavy Warping:** PROHIBITED (alters cortical bone alignment).
* **Destructive Cropping:** PROHIBITED (prevents cutting off subtle periosteal reactions or fracture lines).
* **Arbitrary Colorization:** PROHIBITED (preserves radiologic attenuation characteristics).
* **Validation / Test Augmentation:** STRICTLY PROHIBITED (deterministic evaluation only).

---

## 6. Generated Metadata Artifacts
* [`train.csv`](train.csv)
* [`validation.csv`](validation.csv)
* [`test.csv`](test.csv)
* Processed image directory: [`processed/`](processed)
