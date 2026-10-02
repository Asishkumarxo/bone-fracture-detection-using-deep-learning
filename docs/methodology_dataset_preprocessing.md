# Bone Fracture Image Captioning Using Deep Learning — Methodology, Dataset and Data Preprocessing

**Project Title:** Bone Fracture Image Captioning Using Deep Learning  
**Target Level:** B.Tech Computer Science & Engineering / Artificial Intelligence & Machine Learning  
**Document Purpose:** Comprehensive technical reference detailing the problem formulation, complete methodology, dataset architecture, empirical dataset audits, resolution provenance analysis, and preprocessing pipeline.

---

## 1. Project Overview

Musculoskeletal trauma accounts for a significant portion of worldwide emergency department visits. In clinical practice, plain projection radiography (X-ray) remains the first-line, most accessible, and cost-effective diagnostic imaging modality for assessing skeletal injury. However, interpretation of radiographs requires specialized radiological expertise. In high-volume emergency centers, rural clinics, or off-hours settings, dedicated musculoskeletal radiologists may not be immediately available, potentially leading to delayed triage, missed fractures (particularly subtle cortical disruptions), or diagnostic variability.

This project, **“Bone Fracture Image Captioning Using Deep Learning”**, establishes an end-to-end, reproducible deep learning system that analyzes musculoskeletal radiographs and synthesizes concise, factual, and clinically disciplined natural-language descriptions. 

Unlike black-box generative language models that are susceptible to clinical hallucinations, this system employs a decoupled, multi-task convolutional architecture combined with a rule-grounded caption synthesis engine:
1. **Multi-Task ResNet-50 Classifier:** Concurrently identifies the anatomical region across 7 skeletal sites and detects fracture presence.
2. **Resolution-Aware Inference Router:** Dynamically routes radiographs to optimal input resolutions based on unresized native matrix dimensions.
3. **Disciplined Factual Caption Generator:** Synthesizes natural-language sentences strictly bounded by validated model outputs, deliberately eliminating medical hallucinations.
4. **Decoupled Web Architecture:** Features a high-performance **FastAPI** backend and an intuitive **Streamlit** clinical workstation interface.

---

## 2. Problem Statement

Automating bone radiograph interpretation poses several foundational challenges:

1. **High Visual Similarity Across Skeletal Regions:** Musculoskeletal radiographs vary significantly in projection angles (Anteroposterior [AP], Lateral, Oblique) and scale, yet bones share similar grayscale densities, cortices, and trabecular patterns.
2. **Severe Fracture Subtlety:** Hairline cracks, non-displaced fissures, and buckle fractures manifest as millimeter-scale cortical interruptions against complex skeletal anatomy, making downsampling destructive.
3. **Severe Resolution Inhomogeneity in Medical Datasets:** Public datasets often suffer from extreme bimodal resolution distributions (e.g., legacy low-resolution thumbnails alongside high-resolution digital X-rays), causing single-resolution models to either miss fine fractures or fail on blurry inputs.
4. **The Danger of Hallucinatory Captions:** Generic Vision-Language Models (VLMs) or Large Language Models (LLMs) frequently fabricate medical attributes—such as inventing fracture subtypes (spiral, comminuted), non-existent lateralities (left vs. right), or speculative surgical recommendations—which presents severe safety hazards in medical systems.
5. **Class Imbalance:** In general clinical screening, intact (normal) bones outnumber acute traumatic fractures, introducing gradient bias during training.

---

## 3. Project Objective

The primary objectives of this engineering project are:

1. **Dual-Task Convolutional Learning:** Build and train a multi-task deep neural network using a shared **ResNet-50** convolutional backbone to simultaneously predict:
   - **Anatomical Region:** 7-class classification (`Arm`, `Foot`, `Hand`, `Lower leg`, `Thigh`, `pelvis`, `wrist`).
   - **Fracture Presence:** Binary assessment (`Positive` [Fracture Present] vs. `Negative` [Fracture Absent]).
2. **Patient-Stratified Data Architecture:** Curate and verify a leak-free partition of the **Mendeley BoneFract** dataset ensuring zero patient overlap across training, validation, and test splits.
3. **Resolution-Aware Inference Pipeline:** Design and implement an automated resolution router that evaluates native image matrix dimensions before resizing, dynamically dispatching images to either a $224 \times 224$ baseline branch or a $448 \times 448$ high-resolution branch.
4. **Hallucination-Free Natural Language Captioning:** Implement a deterministic, template-constrained caption generation module that translates structured neural predictions into disciplined sentences without fabricating unsupported medical details.
5. **Production Deployment & Evaluation:** Build and validate an interactive, decoupled web platform (**FastAPI** + **Streamlit**) that provides rapid diagnostic inference, clear probability metrics, and strict error handling.

---

## 4. Overall Methodology

The overall methodology follows a strictly decoupled, sequential data and inference pipeline:

```
[ Input Radiograph (PNG, JPG, WEBP, BMP) ]
                   │
                   ▼
       [ 1. Input Integrity Validation ]
        - In-memory PIL integrity verification
        - MIME type and format validation
                   │
                   ▼
       [ 2. Native Dimension Detection ]
        - Extract unresized (orig_w, orig_h)
                   │
                   ▼
       [ 3. Resolution-Aware Router ]
        - Decision Rule: min(orig_w, orig_h) >= 300 px?
             ├── YES ──► Route to 448×448 Pipeline (exp3_resnet50_448.pt)
             └── NO  ──► Route to 224×224 Pipeline (best_model.pt)
                   │
                   ▼
       [ 4. Deterministic Preprocessing ]
        - Grayscale standardization ('L')
        - Aspect-ratio preserving padding (target 224 or 448)
        - 3-channel broadcast for ImageNet backbone
        - Standardization: mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
                   │
                   ▼
     [ 5. ResNet-50 Convolutional Backbone ]
        - Feature vector extraction: 2048-dimensional embedding
                   │
         ┌─────────┴─────────┐
         ▼                   ▼
[ 6A. Anatomy Head ]  [ 6B. Fracture Head ]
 - Linear(2048->512)   - Linear(2048->256)
 - BatchNorm + ReLU    - BatchNorm + ReLU
 - Dropout(0.3)        - Dropout(0.3)
 - Linear(512->7)      - Linear(256->1)
 - Softmax Activation  - Sigmoid Activation (tau = 0.50)
         │                   │
         └─────────┬─────────┘
                   ▼
      [ 7. Prediction Integration ]
       - Predicted Anatomical Region (str) + Confidence (float)
       - Fracture Status (bool) + Fracture Probability (float)
                   │
                   ▼
      [ 8. Disciplined Caption Generator ]
       - Evaluates validated prediction dictionary
       - Deterministic rule synthesis (e.g., "X-ray of the wrist showing a fracture.")
       - Zero hallucinated clinical features
                   │
                   ▼
      [ 9. FastAPI Response Serialization ]
       - Validated JSON payload adhering to Pydantic schema
                   │
                   ▼
      [ 10. Streamlit Radiology Workstation ]
       - 3-Card Summary: Anatomical Region, Fracture Assessment, Factual Caption
       - Clear progress bars, probability scaling, and research disclaimer
```

---

## 5. Dataset Description

The primary dataset utilized for classifier training, validation, and testing is the **Mendeley BoneFract Dataset** (*BoneFract: A Bone Fracture Dataset*), collected and published by the Peoples University of Medical and Health Sciences for Women (PUMHSW).

### Summary Statistics
- **Total Images:** 47,931 radiographs
- **Total Unique Patients:** 47,922 patients
- **Image Format:** 100% PNG images
- **Color Space:** Grayscale radiographs represented as 3-channel RGB (47,284 images) or single-channel L (645 images), 8-bit depth.
- **Task Structure:** Joint multi-class anatomical classification (7 categories) and binary fracture detection (Positive vs. Negative).

### Anatomical Categories
The dataset strictly covers **seven anatomical regions**. Unsupported categories (such as a distinct Knee or Hip class) are not present in the dataset taxonomy:

| Anatomical Class | Total Images | Dataset Percentage | Description |
| :--- | :---: | :---: | :--- |
| **Lower leg** | 10,996 | 22.94% | Tibia and fibula views; patellar/knee views fall under this class. |
| **Pelvis** | 6,983 | 14.57% | Pelvic ring and hip joint examinations. |
| **Foot** | 6,840 | 14.27% | Tarsals, metatarsals, and phalanges. |
| **Hand** | 6,032 | 12.58% | Metacarpals, phalanges, and carpal-metacarpal joints. |
| **Wrist** | 6,020 | 12.56% | Distal radius, distal ulna, and proximal carpal bones. |
| **Arm** | 5,597 | 11.68% | Humerus, radius, and ulna shaft projections. |
| **Thigh** | 5,463 | 11.40% | Femur shaft and proximal femoral structures. |
| **Total** | **47,931** | **100.0%** | Seven anatomical sites. |

### Binary Fracture Distribution (Class Imbalance)
- **Negative (No Fracture Detected):** 34,010 images (**70.96%**)
- **Positive (Fracture Present):** 13,921 images (**29.04%**)
- **Imbalance Ratio:** $34,010 / 13,921 \approx 2.443 : 1$. To prevent gradient dominance by normal cases during training, this exact ratio was used as the positive-class loss weight (`pos_weight = 2.4375`).

---

## 6. Dataset Verification & Forensic Audit

A comprehensive forensic audit of the dataset was conducted to verify its integrity before and during model training.

### 1. Data Integrity and Corruption Audit
- **Zero File Corruption:** In-memory verification of all 47,931 image headers confirmed 0 unreadable or corrupted files.
- **Zero Label Corruption:** All 47,931 records strictly mapped to one of the 7 anatomical classes and one of the 2 binary fracture states.

### 2. Patient-Level Leakage Audit
A common pitfall in medical deep learning is **data leakage**, which occurs when multiple radiographs of the same patient are distributed across both the training set and the test set. In such scenarios, the model memorizes idiosyncratic patient features (bone mineral density, anatomical landmarks, surgical hardware) rather than generalizable pathology, leading to artificially inflated test metrics.

- **Unique Patient Identifiers:** The dataset encodes patient identity directly in directory structures (`patientXXXXX`).
- **Audit Result:** Zero patient overlap was verified across all three splits:
  - $\text{Train} \cap \text{Validation} = \emptyset$ (0 patient overlap)
  - $\text{Train} \cap \text{Test} = \emptyset$ (0 patient overlap)
  - $\text{Validation} \cap \text{Test} = \emptyset$ (0 patient overlap)
  - **Verdict:** The dataset splitting is strictly patient-stratified and completely leak-free.

---

## 7. Resolution Distribution & The Native Resolution Divide

A pivotal discovery made during the **Resolution Provenance Audit** was the extreme bimodal resolution distribution in the Mendeley BoneFract dataset.

### The Two Cohorts:
1. **Native 102×102 Thumbnail Cohort:**
   - **Count:** 34,542 images (**72.07%** of the entire dataset).
   - **Dimensions:** Exactly $102 \times 102$ pixels.
   - **Provenance:** These are native, standalone image captures published directly by PUMHSW. Forensic inspection of the original 3.25 GB distribution archive (`BoneFract A Bone Fracture Dataset.zip`) confirmed that **zero higher-resolution originals exist** for these images.
2. **Native Higher-Resolution Cohort:**
   - **Count:** 13,389 images (**27.93%** of the dataset).
   - **Dimensions:** Ranging from $373 \times 454$, $454 \times 373$, $800 \times 800$, up to $1920 \times 1080$ pixels.

### Why Upscaling Cannot Restore Lost Information
In digital signal and image processing, bilinear or bicubic interpolation calculates pixel values between existing sample points. When a $102 \times 102$ radiograph is enlarged by $4.4\times$ to $448 \times 448$, interpolation smooths pixel gradients over a larger area. 

In bone radiographs, fractures appear as **fine, sharp cortical discontinuities** (sub-millimeter linear lucencies). Upscaling blurs these sharp boundaries, converting high-frequency edges into diffuse gray gradients. 

When a model is trained at $448 \times 448$, its convolutional kernels specialize in detecting crisp, fine-grained structural disruptions. When fed upscaled $102 \times 102$ images, the model fails to detect the blurred fracture lines, resulting in a dramatic drop in fracture sensitivity (recall dropped to 38.20% on thumbnails in Experiment 3). Conversely, on genuinely high-resolution images, the $448 \times 448$ model achieved **90.91% recall**.

This empirical finding proved that a single uniform resolution cannot serve both cohorts effectively, necessitating **Resolution-Aware Routing**.

---

## 8. Data Splitting

The dataset was partitioned using strict patient-level stratification into three distinct subsets:

| Split Partition | Total Images | Unique Patients | % of Dataset | Primary Purpose |
| :--- | :---: | :---: | :---: | :--- |
| **Training Set** | 38,366 | 38,337 | 80.04% | Model parameter optimization via backpropagation and loss minimization. |
| **Validation Set** | 4,787 | 4,793 | 9.99% | Epoch-by-epoch model checkpointing, hyperparameter tuning, and early stopping. |
| **Test Set** | 4,778 | 4,792 | 9.97% | **Completely untouched** benchmark used exclusively for unbiased final evaluation. |
| **Total** | **47,931** | **47,922** | **100.0%** | Full dataset manifest. |

### Preservation of the Test Set:
The 4,778-image test set was strictly locked throughout all experiments. No test images were ever used for gradient updates, learning rate scheduling, threshold selection, or architectural decisions.

---

## 9. Image Preprocessing

Image preprocessing in medical computer vision must preserve anatomical geometry. Conventional naive resizing stretches images to square dimensions, artificially distorting bone morphology (e.g., turning an elongated femur into a squashed, unnatural shape).

To prevent this, a deterministic **aspect-ratio-preserving padding** pipeline was implemented in [`inference/preprocessing.py`](file:///c:/Users/arbaz/Desktop/Deep%20learning%20project/inference/preprocessing.py):

```
[ Input Radiograph (W × H) ]
              │
              ▼
    [ 1. Grayscale Conversion ]
    - Converts RGB or RGBA to single-channel 'L'
              │
              ▼
    [ 2. Aspect Ratio Preserving Scaling ]
    - Scaling ratio: r = min(target_w / orig_w, target_h / orig_h)
    - New dimensions: new_w = int(orig_w * r), new_h = int(orig_h * r)
    - Resize image using PIL.Image.Resampling.BILINEAR
              │
              ▼
    [ 3. Center Padding ]
    - Create blank canvas of size (target_w, target_h) with fill_color = 0 (black)
    - Calculate offsets: pad_x = (target_w - new_w) // 2, pad_y = (target_h - new_h) // 2
    - Paste scaled image at (pad_x, pad_y)
              │
              ▼
    [ 4. 3-Channel Broadcast ]
    - Replicate single-channel grayscale across 3 channels (RGB)
    - Required by pretrained ImageNet backbones
              │
              ▼
    [ 5. Tensor Normalization ]
    - Convert pixel intensities from [0, 255] to [0.0, 1.0] via ToTensor()
    - Normalize: mean = [0.5, 0.5, 0.5], std = [0.5, 0.5, 0.5]
    - Maps pixel distribution to range [-1.0, 1.0]
```

### Preprocessing Paths:
- **Baseline Path (Target: 224×224):** Applied to images where $\min(\text{width}, \text{height}) < 300\text{ px}$. Downsamples or pads images to a $224 \times 224$ matrix, optimizing feature extraction for native low-resolution thumbnails without introducing extreme scaling blur.
- **High-Resolution Path (Target: 448×448):** Applied to images where $\min(\text{width}, \text{height}) \ge 300\text{ px}$. Preserves fine cortical structures and trabecular patterns for native high-resolution radiographs.

---

## 10. Training Data Augmentation

Data augmentation must be clinically disciplined. While common in general object recognition, aggressive geometric distortions can alter medical meaning (e.g., severe shearing can mimic pathological angulation, and horizontal flipping alters anatomical laterality).

Inspecting [`src/dataset.py`](file:///c:/Users/arbaz/Desktop/Deep%20learning%20project/src/dataset.py#L8-L27) confirms the **exact augmentations** implemented:

```python
if split == 'train':
    return T.Compose([
        T.RandomRotation(degrees=(-10, 10)),
        T.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        T.ColorJitter(brightness=0.1, contrast=0.1),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
else:
    return T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
```

### Augmented Operations Verified:
1. **`RandomRotation(degrees=(-10, 10))`:** Small rotational variations simulate slight differences in patient positioning on the X-ray table without inverting anatomy.
2. **`RandomAffine(translate=(0.05, 0.05), scale=(0.95, 1.05))`:** Minor translation (up to 5% shift) and minor scaling (95% to 105%) simulate subtle variations in collimator framing and patient-to-detector distance.
3. **`ColorJitter(brightness=0.1, contrast=0.1)`:** Simulates variations in tube exposure (kVp/mAs) and digital detector dynamic range.
4. **Deliberately Excluded Augmentations:**
   - **RandomHorizontalFlip is NOT used:** Radiologists interpret radiographs based on standardized anatomical orientation. Flipping can invert lateral anatomical relationships.
   - **Elastic deformations, shearing, and severe cropping are NOT used:** These can fabricate artificial fracture lines or crop out focal cortical lesions.

---

## 11. Resolution-Aware Inference Routing

The resolution-aware routing logic is implemented in [`inference/model_registry.py`](file:///c:/Users/arbaz/Desktop/Deep%20learning%20project/inference/model_registry.py#L126-L155):

```python
ROUTING_MIN_DIMENSION_THRESHOLD = 300

def route_image(cls, width: int, height: int) -> Dict[str, Any]:
    min_dim = min(width, height)
    if min_dim >= 300:
        return {
            "selected_model": "exp3_resnet50_448.pt",
            "routing_resolution": "448",
            "target_size": (448, 448),
            "model_path": cls.get_exp3_classifier_path(),
            "branch": "high_resolution"
        }
    else:
        return {
            "selected_model": "best_model.pt",
            "routing_resolution": "224",
            "target_size": (224, 224),
            "model_path": cls.get_baseline_classifier_path(),
            "branch": "low_resolution"
        }
```

### Why $\min(\text{width}, \text{height}) \ge 300\text{ px}$ Was Selected:
1. The low-resolution thumbnail cohort has fixed dimensions of $102 \times 102$ pixels ($\min = 102 < 300$).
2. The high-resolution cohort features minimum dimensions starting at 373 pixels ($373 \times 454$, $800 \times 800$, etc.), with $\min \ge 373 > 300$.
3. A threshold of 300 pixels cleanly separates the two cohorts with a wide 271-pixel buffer, ensuring that every radiograph is processed by the model checkpoint optimized for its native visual fidelity.

---

## 12. Complete Methodology Flow (Step-by-Step)

1. **Client Submission:** The user selects a radiograph via the Streamlit interface and clicks `"Analyze Radiograph"`.
2. **Network Transmission:** The frontend dispatches a `multipart/form-data` POST request containing the binary image buffer to `http://127.0.0.1:8000/predict`.
3. **API Validation:** FastAPI intercepts the payload, validates MIME headers (`image/png`, `image/jpeg`, etc.), and passes the bytes to `InferenceService`.
4. **Dimension Extraction:** The service opens the image in-memory via PIL and extracts unresized pixel dimensions `(orig_w, orig_h)`.
5. **Resolution Routing:** `ModelRegistry.route_image()` determines whether `min(orig_w, orig_h) >= 300` and assigns the target size and model checkpoint.
6. **Aspect-Ratio Preprocessing:** The image is converted to grayscale, scaled, padded to the target matrix (224 or 448), replicated to 3 channels, and normalized.
7. **Neural Forward Pass:** The tensor is passed through the routed **ResNet-50 Multi-Task** network:
   - The shared backbone extracts a 2048-dimensional feature embedding.
   - The **Anatomy Head** outputs 7 raw logits, converted via Softmax to class probabilities.
   - The **Fracture Head** outputs a single binary logit, converted via Sigmoid to a probability in $[0.0, 1.0]$.
8. **Decision Thresholding:** The fracture probability is evaluated against decision threshold $\tau = 0.50$ (`is_fracture = prob >= 0.50`).
9. **Caption Generation:** `generate_caption()` evaluates the region name and fracture status against strict factual templates to synthesize an English sentence.
10. **Response Serialization:** FastAPI serializes all findings into a structured `PredictionResponse` JSON payload.
11. **Frontend Presentation:** Streamlit renders the results into 3 clean clinical cards (Anatomical Region, Fracture Assessment, and Factual Summary).

---

## 13. Methodology Summary

The data and methodology pipeline establishes a robust, scientifically disciplined foundation for medical image analysis:
- **Data Integrity:** Strict patient-level stratification eliminates data leakage across 47,931 radiographs.
- **Pre-processing Rigor:** Aspect-ratio preserving padding prevents geometric skeletal distortion.
- **Resolution-Aware Routing:** Eliminates the trade-off between low-resolution blur artifacts and high-resolution feature loss.
- **Anti-Hallucination Guardrails:** Decouples classification from language synthesis, guaranteeing deterministic, factual descriptions.
