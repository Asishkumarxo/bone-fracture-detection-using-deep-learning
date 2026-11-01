# 00. Project Overview & Institutional Identity

**Project Title:** Bone Fracture Image Captioning Using Deep Learning  
**Short Title / Working Name:** Musculoskeletal Deep Learning Diagnostic System (*BoneSight*)  
**Corpus / Repository Root:** `c:\Users\Donipudi Asish Kumar\Desktop\Deep learning project`  
**GitHub Repository:** `Asishkumarxo/bone-fracture-detection-using-deep-learning`  
**Document Type:** Technical Evidence Dossier (Academic Report Foundation)  
**Date of Audit:** October 2026  
**Auditor / Senior DL Engineer:** Automated Forensic Technical Auditor & DL Specialist  

---

## 1. Academic & Institutional Metadata

* **Candidate / Primary Investigator:** Donipudi Asish Kumar
* **University Registration Number:** 240301120035
* **Degree Program:** Bachelor of Technology (B.Tech) in Computer Science and Engineering
* **Department:** Department of Computer Science and Engineering
* **Faculty / School:** School of Engineering and Technology (SoET)
* **Institution:** Centurion University of Technology and Management (CUTM)
* **Campus:** Bhubaneswar Campus, Odisha, India
* **Project Supervisor:** Dr. Chandra Sekhar Dash, Associate Professor, Department of Computer Science and Engineering
* **Head of Department:** Mr. Rajkumar Mohanta, Professor and Head of Department, CSE
* **Dean of School:** Prof. Sujata Chakravarty, Dean, School of Engineering and Technology
* **Submission Session / Target Date:** Academic Year 2025–2026 / April 2026

---

## 2. Problem Statement & Clinical Context

### 2.1 The Clinical Challenge
Musculoskeletal trauma is one of the most common reasons for emergency department admissions globally. According to World Health Organization (WHO) trauma registries and emergency radiology audits, **plain radiography (X-ray)** remains the standard initial imaging modality due to its rapid acquisition, low radiation dose, and high availability.

However, interpreting trauma radiographs is fraught with diagnostic error:
1. **High Diagnostic Discrepancy Rate:** Between **60% and 80%** of all diagnostic discrepancies in emergency departments involve musculoskeletal radiograph interpretations (Berkowitz, 2018).
2. **Clinical Fatigue and Cognitive Bias:** High clinical caseloads, circadian fatigue during night shifts, and satisfaction-of-search errors cause clinicians to overlook subtle or secondary fractures.
3. **Anatomical Complexity:** Cancellous bone regions with complex overlapping trabecular architectures (e.g., foot tarsals/metatarsals, carpal bones of the wrist) frequently present with faint hairline or non-displaced occult fractures that lack gross cortical step-off.
4. **Severe Consequences of Missed Diagnosis:** Undetected fractures result in non-union, malunion, avascular necrosis, permanent chronic pain, mobility loss, and substantial medicolegal liability.

### 2.2 The Technical & Research Gap
While automated fracture detection models have been explored in computer vision literature, existing open-source and academic implementations suffer from critical deficiencies:
1. **Severe Data Contamination (Patient Leakage):** Most published benchmarks randomly split datasets by image file, ignoring patient clustering. When multiple radiographs of the same patient exist across training and test partitions, reported performance is heavily inflated.
2. **Resolution Blindness:** Standard CNN pipelines naively downsample all images to $224 \times 224$, inadvertently washing out hairline fracture margins. Conversely, upscaling low-resolution thumbnails causes bilinear interpolation blur that degrades edge detection.
3. **Clinical Hallucinations in Medical Captioning:** Conventional vision-language models (e.g., LSTM or Transformer-based image captioning models like Show-and-Tell or BLIP) generate grammatically fluent sentences but regularly **hallucinate unverified clinical findings**, such as inventing trauma mechanisms ("patient fell from ladder"), fracture subtypes ("severe comminuted spiral fracture"), or treatment directives ("proceed to open reduction internal fixation").
4. **Lack of Integrated Multi-Task Deployment:** Prior work largely separates anatomical site prediction, fracture classification, and lesion localization into siloed models, failing to provide unified clinical decision support.

---

## 3. Project Objectives

### 3.1 Primary Objective
To design, train, audit, benchmark, and deploy an end-to-end, reproducible deep learning system that accurately classifies anatomical skeletal regions, detects the presence of bone fractures with calibrated operating thresholds, localizes fracture lesions via bounding boxes, and generates disciplined, factual natural-language clinical descriptions completely free of medical hallucinations.

### 3.2 Secondary Objectives
1. **Zero-Contamination Graph Partitioning:** Engineer an automated bipartite graph-connected component partitioning protocol ensuring $0\%$ patient overlap and $0$ MD5 hash collisions between training, validation, and held-out test splits.
2. **Dynamic Resolution-Aware Routing:** Implement an intelligent inference gate that dynamically directs high-resolution diagnostic studies ($\min(W, H) \ge 300\text{ px}$) to high-resolution networks ($448 \times 448$) and native thumbnails to baseline networks ($224 \times 224$).
3. **Multi-Task Neural Architecture:** Implement a multi-task learning architecture based on ResNet-50 and Inception-v3 with a shared latent feature representation (2048 dimensions) driving simultaneous 7-class anatomical classification and calibrated binary fracture detection.
4. **Class & Regional Imbalance Mitigation:** Counteract natural class imbalance via positive loss weighting ($w_{\text{pos}} = 2.438$) and regional feature competition via stratified regional balancing (2,000 samples per anatomical region across 14,000 fine-tuning samples).
5. **Spatial Lesion Localization:** Integrate Faster R-CNN with MobileNetV3-Large Feature Pyramid Networks (FPN) trained on expert-annotated clinical datasets (FracAtlas) to provide interpretable lesion bounding boxes.
6. **Disciplined Factual Natural-Language Captioning:** Formulate and implement a 10-rule deterministic clinical sentence synthesis engine bounded strictly by verified computer vision predictions.
7. **Full-Stack Decoupled Production Serving:** Deploy an asynchronous FastAPI REST microservice paired with an interactive Streamlit clinical radiology dashboard (*BoneSight*).

---

## 4. Research Questions

1. **RQ1 (Resolution Impact):** Does increasing the spatial input resolution from $224 \times 224$ to $448 \times 448$ in a residual network significantly improve sensitivity to fine cortical bone fractures?
2. **RQ2 (Regional Balancing Impact):** How does stratified regional balancing (equalizing anatomical site representations in training) affect overall sensitivity, particularly in under-represented and complex anatomical regions like the pelvis and hand?
3. **RQ3 (Architectural Comparison):** How does multi-scale factorized convolution (Inception-v3 with asymmetric $1 \times 7$ and $7 \times 1$ filters) compare with uniform residual convolutions (ResNet-50) in detecting cancellous fractures in complex joint spaces (e.g., foot)?
4. **RQ4 (Hallucination Elimination):** Can a rule-bounded deterministic clinical caption engine eliminate false medical claims while maintaining concise radiological utility?
5. **RQ5 (Data Leakage Verification):** Does patient-level graph partitioning prevent optimistic bias in medical image classification benchmarks?

---

## 5. Scope & Boundary Conditions

### 5.1 Confirmed Implemented Capabilities
* **7-Class Anatomical Site Identification:** Accurately classifies plain radiographs into `Arm`, `Foot`, `Hand`, `Lower leg`, `Thigh`, `pelvis`, and `wrist`.
* **Calibrated Binary Fracture Presence:** Outputs a continuous probability $\sigma(z) \in [0, 1]$ thresholded at the canonical operating point $\tau = 0.50$.
* **Uncertainty Flagging:** Automatically flags anatomical uncertainty (`uncertain_anatomy = True`) when the top predicted region confidence falls below $\tau_{\text{unc}} = 0.40$.
* **Spatial Lesion Bounding Boxes:** Faster R-CNN detector with MobileNetV3 FPN trained on FracAtlas producing coordinate boxes $[x_1, y_1, x_2, y_2]$ at calibrated threshold $\tau_{\text{det}} = 0.10$.
* **Deterministic Factual Captioning:** Synthesizes structured findings into standard radiological sentences adhering to 10 strict clinical safety boundaries.
* **Dual-Service Production Architecture:** Asynchronous FastAPI backend on port 8000 and Streamlit diagnostic UI on port 8501.

### 5.2 Explicit Non-Capabilities (What the System Does NOT Do)
* **No Fracture Subtype Classification:** Does *not* predict transverse, oblique, spiral, comminuted, greenstick, or avulsion subtypes.
* **No Laterality Classification:** Does *not* predict left vs. right extremity.
* **No Displacement or Angulation Measurement:** Does *not* measure millimeter displacement or angular deformity.
* **No Patient Demographic Inference:** Does *not* infer age, sex, bone mineral density, or pediatric physis status.
* **No Generative Free-Form Reporting:** Does *not* use LLMs, GPT, or LSTMs to invent narrative clinical histories or trauma mechanisms.
* **No Independent "Hip" Class:** In the BoneFract dataset taxonomy, hip radiographs are cataloged under `pelvis` or proximal `Thigh`.

---

## 6. Functional & Non-Functional Requirements

### 6.1 Functional Requirements (FR)
* **FR-1:** Ingest arbitrary single-view plain radiographs in standard formats (PNG, JPG, BMP, TIFF).
* **FR-2:** Validate in-memory byte buffers to reject corrupted or non-image payloads with HTTP 400.
* **FR-3:** Perform aspect-ratio preserving padding onto a square black canvas to eliminate geometric distortion.
* **FR-4:** Route images based on native dimensions ($\ge 300\text{ px} \rightarrow 448\times 448$; $< 300\text{ px} \rightarrow 224\times 224$).
* **FR-5:** Output top-1 and distribution probabilities across 7 anatomical sites.
* **FR-6:** Output binary fracture status and probability calibrated at $\tau = 0.50$.
* **FR-7:** Generate deterministic natural-language clinical impressions.
* **FR-8:** Render diagnostic overlay visualizations with bounded boxes and regional tags.

### 6.2 Non-Functional Requirements (NFR)
* **NFR-1 (Inference Latency):** Mean end-to-end inference latency on GPU $< 100\text{ ms}$ per image ($< 500\text{ ms}$ on CPU).
* **NFR-2 (Memory Footprint):** GPU VRAM footprint during inference $< 1.5\text{ GB}$, permitting execution on entry-level laptop GPUs (e.g., NVIDIA RTX 3050 4GB).
* **NFR-3 (Software Reliability):** 100% test pass rate across unit, integration, and API tests (69/69 automated tests verified).
* **NFR-4 (Security):** Path traversal protection on all static file and visualization delivery endpoints.
* **NFR-5 (Zero Hallucination):** 0% rate of ungrounded clinical text generation.

---

## 7. Key Project Contributions

1. **Empirical Demonstration of Region Balancing (Exp 4):** Proved that stratified regional balancing (2,000 samples/region, $N=14,000$) significantly outperforms full-dataset imbalanced training ($N=38,366$), increasing fracture recall from 69.25% to **72.87%**, catching **1,007 True Positives**, and reducing missed fractures by **11.8%** (from 425 to 375 misses) on 4,778 held-out test samples.
2. **Discovery & Architectural Resolution of Native 102×102 Dataset Cohort:** Conducted the first forensic dimension audit of the Mendeley BoneFract dataset, proving that 72.07% (34,542 images) are native hospital thumbnails, and engineered the dual-resolution routing gate.
3. **First-of-Kind ResNet vs. Inception Trauma Benchmark:** Established that while ResNet-50 achieves superior global sensitivity (72.87%), Inception-v3 delivers superior Foot sensitivity (**55.12%**) and higher normal specificity (**87.28%**), demonstrating the clinical utility of multi-scale asymmetric filters.
4. **Disciplined Factual Sentence Generation:** Replaced vulnerable generative models with an auditable 10-rule medical synthesis engine that guarantees factual fidelity.
5. **Production-Ready Microservice & Workstation:** Delivered a fully decoupled, tested, and container-ready web ecosystem serving clinicians without third-party cloud dependencies.

