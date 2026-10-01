# Dataset Directory Guide

This directory holds the raw and structured medical datasets utilized for training and evaluating the bone fracture detection, classification, and localization models.

> [!IMPORTANT]
> **Dataset files and raw radiographs are intentionally excluded from the Git repository** due to size constraints (multi-gigabyte archives) and medical data governance practices. Follow the instructions below to acquire and configure the datasets locally.

---

## 1. Required Datasets Overview

| Dataset | Primary Source | Tasks in this Project | Labels / Annotations |
| :--- | :--- | :--- | :--- |
| **BoneFract** | Mendeley Data | Multi-task Classification: Anatomical region (7 classes) & Binary fracture detection | 7 anatomical regions, binary fracture status (`Positive`, `Negative`) |
| **FracAtlas** | Figshare / Official Repository | Lesion Localization: Spatial bounding box object detection | Bounding boxes, polygon segmentation masks (COCO JSON format) |

---

## 2. Dataset Acquisition

### A. BoneFract
* **Title:** BoneFract: A Bone Fracture Dataset
* **Official Repository:** Mendeley Data (`doi:10.17632/jhqrxmff7v`)
* **Download:** Download the archive `BoneFract A Bone Fracture Dataset.zip` (~3.95 GB).
* **Extraction:** Extract the archive directly into `data/BoneFract/`.

### B. FracAtlas
* **Title:** FracAtlas: A Dataset for Bone Fracture Classification, Detection and Segmentation
* **Official Repository:** Figshare (`doi:10.6084/m9.figshare.22634356`) / Nature Scientific Data
* **Download:** Download the images directory and utilities/annotations folder.
* **Extraction:** Place files into `data/FracAtlas/`.

---

## 3. Expected Local Directory Structure

After downloading and extracting, your `data/` directory should look as follows:

```
data/
├── README.md                          <- This guide
│
├── BoneFract/                         <- Extracted BoneFract dataset
│   └── BoneFract A Bone Fracture Dataset/
│       ├── Arm/
│       │   ├── fractured/
│       │   └── not fractured/
│       ├── Foot/
│       ├── Hand/
│       ├── Lower leg/
│       ├── Thigh/
│       ├── pelvis/
│       └── wrist/
│
└── FracAtlas/                         <- Extracted FracAtlas dataset
    ├── images/
    │   ├── Fractured/
    │   └── Non_fractured/
    ├── Annotations/
    │   ├── COCO JSON/
    │   │   └── COCO_fracture_masks.json
    │   └── XML/
    └── Utilities/
        └── Fracture Split/
            ├── train.csv
            ├── valid.csv
            └── test.csv
```

---

## 4. Anatomical Classes & Taxonomy (BoneFract)

The multi-task classifier is trained on 7 distinct anatomical categories:

1. `Arm` (Humerus, radius, ulna)
2. `Foot` (Tarsals, metatarsals, phalanges)
3. `Hand` (Carpals, metacarpals, phalanges)
4. `Lower leg` (Tibia, fibula)
5. `Thigh` (Femoral diaphysis and shaft)
6. `pelvis` (Pelvic girdle, acetabulum, and hip joints)
7. `wrist` (Distal radius, ulna, carpal joints)

> [!NOTE]
> **Pelvis vs. Hip Joint Taxonomy:** In BoneFract, radiographs of the hip joint are cataloged under `pelvis` (or proximal femur under `Thigh`). The system does not use an independent "Hip" class because that label does not exist in the training dataset.

---

## 5. Preprocessing & Split Generation

Once `BoneFract` is placed in `data/BoneFract/`, execute the leak-free patient-cluster preprocessing pipeline:

```bash
python preprocess.py
```

This performs:
1. Patient-level clustering to prevent cross-split patient or duplicate image leakage.
2. Stratified 80% train / 10% validation / 10% test partitioning.
3. Deterministic aspect-ratio preserved padding to 224×224 8-bit grayscale PNGs.
4. Generates root metadata files: `train.csv`, `validation.csv`, and `test.csv`.
