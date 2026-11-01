# 08. Error Analysis, Failure Investigation, and Limitations

**Project Title:** Bone Fracture Image Captioning Using Deep Learning  
**Scope:** Forensic error analysis, root causes of misclassification, dataset biases, clinical safety boundaries, and proposed future improvements.  
**Source Code References:** [`reports/ROOT_CAUSE_ANALYSIS.md`](file:///c:/Users/Donipudi%20Asish%20Kumar/Desktop/Deep%20learning%20project/reports/ROOT_CAUSE_ANALYSIS.md), [`RESOLUTION_PROVENANCE_AUDIT.md`](file:///c:/Users/Donipudi%20Asish%20Kumar/Desktop/Deep%20learning%20project/RESOLUTION_PROVENANCE_AUDIT.md), [`reports/caption_generator_audit.md`](file:///c:/Users/Donipudi%20Asish%20Kumar/Desktop/Deep%20learning%20project/reports/caption_generator_audit.md)  
**Document Version:** 1.0 (Audited October 2026)  

---

## 1. Forensic Root Cause Analysis: Pelvis vs. Thigh Misclassification

`[Source: reports/ROOT_CAUSE_ANALYSIS.md]`

In test evaluations across all models, while overall anatomical classification accuracy is high (80.83%), Pelvis radiographs exhibited recurring misclassification as `Thigh` or `Lower leg`:

### 1.1 Empirical Observations
* Out of 74 misclassified test pelvis radiographs, **58 (78.4%) were predicted as `Thigh`** (proximal femur) and 8 as `Lower leg`.
* Only 8 pelvis radiographs were confused with other anatomical regions (`Arm`, `Foot`, `Hand`, `wrist`).

### 1.2 Radiological & Dataset Root Cause
1. **Absence of Dedicated "Hip" Class in BoneFract Taxonomy:**  
   The BoneFract dataset categorizes all lower-quarter trauma into either `pelvis` or `Thigh`. It does not contain an independent label for the hip joint or femoral neck.
2. **Prominent Femoral Diaphysis Feature Representation:**  
   In emergency hip and pelvis radiographs, the proximal third of the femur (greater/lesser trochanter and upper diaphysis) is heavily featured. The cylindrical cortical shaft of the proximal femur strongly activates the tubular bone convolutional filters that the network learned for the `Thigh` class (femoral diaphysis).
3. **Field of View (FOV) Ambiguity:**  
   Collimated views focused tightly on the femoral neck exclude the broader pelvic ring (iliac crest and sacrum), causing the network to classify the image based entirely on the femoral shaft.

### 1.3 Engineered Mitigation
To prevent silent misclassification, the platform implements an **anatomical uncertainty gate** (`inference/region_predictor.py`):
$$\text{If } \max_{c} P(\text{region} = c \mid \mathbf{x}) < \tau_{\text{unc}} \quad (\tau_{\text{unc}} = 0.40) \implies \text{flag } \texttt{uncertain\_anatomy = True}$$
The UI highlights this uncertainty with an alert badge, prompting the clinician to confirm the anatomical site.

---

## 2. Occult & Hairline Fractures in Complex Cancellous Bone

`[Source: reports/three_way_experiment_comparison_report.md]`

### 2.1 The Cortical vs. Cancellous Disparity
A major finding of this project is the marked sensitivity disparity between long tubular bones and complex cancellous joint spaces:
* **Tubular Cortical Bones:** High sensitivity (Thigh: **92.13%**, Lower leg: **78.11%** in Exp 4).
* **Cancellous Joint Bones:** Moderate sensitivity (Foot: **53.17%**, Hand: **53.92%** in Exp 4).

### 2.2 Radiological Cause
* **Tubular Bones (Femur, Tibia):** Possess dense, thick cortical bone surrounded by soft tissue. Traumatic fractures create sharp cortical discontinuities, angular deformities, or displaced step-offs that generate high-contrast visual edges easily detected by $3\times 3$ convolutional filters.
* **Cancellous Bones (Tarsals, Metatarsals, Carpals):** Consist of delicate, overlapping, porous trabecular lattices. Occult hairline stress fractures present as subtle, faint lucent lines spanning only 1–2 pixels without gross displacement. Overlapping bone shadows in standard 2D views obscure these faint lines.

### 2.3 Architectural Solution (Inception-v3)
Experiment 5 demonstrated that **Inception-v3 factorized convolutions ($1\times 7, 7\times 1$)** improved Foot sensitivity to **55.12%** (+14.63% over baseline), proving that multi-scale asymmetric receptive fields capture intricate cancellous joint margins better than standard residual blocks.

---

## 3. The Resolution Horizon: Native 102×102 Thumbnails

`[Source: RESOLUTION_PROVENANCE_AUDIT.md]`

### 3.1 The Bottleneck
* **72.07% of the BoneFract dataset (34,542 images)** exists natively at $102 \times 102$ pixels.
* Forensic hash and dimension audits proved that no high-resolution originals exist for this cohort.

### 3.2 Failure Mechanism (Interpolation Blur)
When a $102 \times 102$ thumbnail is upscaled to $448 \times 448$ via bilinear or bicubic interpolation:
1. Pixel values are smoothed using weighted linear averaging.
2. A hairline fracture that spans 1–2 pixels in native resolution is broadened into a diffuse, 6–8 pixel gray gradient.
3. This eliminates the sharp high-frequency spatial gradients required by convolutional kernels to distinguish a true cortical crack from normal osteopenia or vascular grooves.
4. **Impact:** On the held-out test split, fracture recall on native high-res studies ($\ge 300\text{ px}$) is **86.86%**, but drops to **52.82%** on native $102 \times 102$ thumbnails.

---

## 4. Single-View (2D Projection) Clinical Limitation

* **The Clinical Standard:** Clinical emergency guidelines mandate a minimum of two orthogonal radiographic views (Anteroposterior [AP] and Lateral). Many non-displaced or hairline fractures are completely occult on the AP view due to superimposition and only visible on the lateral projection.
* **Current System Constraint:** The platform ingests single 2D projection views. If a fracture is obscured on that view, the model will classify it as normal, having no access to the complementary orthogonal projection.

---

## 5. Software Correctness vs. Clinical Reliability

A fundamental distinction must be maintained for university examination:
* **Automated Tests Passing (69/69):** Proves that tensor shapes match, Pydantic schemas validate, endpoints return expected HTTP codes, the 10 caption rules assemble without syntax errors, and models load into GPU memory without crashing.
* **Does NOT Prove Clinical Superiority:** Passing software tests does not guarantee that the model will catch every occult fracture in a trauma bay. Clinical diagnostic sensitivity is bounded by dataset resolution, single-view projection limits, and image quality.

---

## 6. Comprehensive Risk Mitigation Matrix

| Failure Mode | Root Cause | Clinical Consequence | Engineered Software Mitigation |
| :--- | :--- | :--- | :--- |
| **Missed Occult Fracture** | Faint hairline fracture in low-res or cancellous bone. | False Negative; untreated fracture, malunion. | Exp 4 region balancing caught 1,007 fractures (-11.8% misses); resolution gate directs high-res scans to 448x448. |
| **False Positive Alarm** | Nutrient canals, overlapping joint lines, artifacts. | Unnecessary splinting or secondary CT scan. | Operating threshold calibrated at $\tau=0.50$; Exp 5 achieved 87.28% specificity to suppress false alarms. |
| **Pelvis / Thigh Confusion** | Absence of dedicated "Hip" class in BoneFract taxonomy. | Incorrect anatomical labeling on diagnostic report. | Uncertainty threshold $\tau_{\text{unc}} < 0.40$ raises automated `uncertain_anatomy` warning. |
| **Interpolation Blur** | Bilinear upsampling of native $102\times 102$ thumbnails. | Suppressed high-frequency edge gradients. | Dynamic resolution routing gate segregates studies $\ge 300\text{ px}$ from low-res thumbnails. |
| **Clinical Hallucinations** | Generative language models inventing unsupported text. | Misleading treatment advice or fictitious trauma mechanisms. | Factual rule-based caption engine strictly bounded by 10 clinical safety rules. |
| **Corrupted Payload Attack** | Malformed or malicious file uploaded to REST API. | Server crash or unhandled 500 error. | In-memory `raw_im.verify()` validation returns clean HTTP 400 Bad Request. |

---

## 7. Concrete Recommended Future Work

1. **Biplanar Orthogonal Fusion Network (AP + Lateral):**  
   Implement a dual-stream Siamese architecture with cross-attention fusion to jointly evaluate AP and Lateral projections, mirroring clinical standard practice.
2. **Dedicated Hip Joint Sub-Taxonomy:**  
   Re-annotate pelvic and femoral radiographs to include an explicit "Hip" class with localized sub-regions (femoral neck, intertrochanteric, acetabulum).
3. **Vision-Language Pre-Training (CLIP / BiomedCLIP):**  
   Explore medical contrastive pre-training on matched musculoskeletal image-text pairs while maintaining deterministic guardrails at the generation head.
4. **DICOM PACS & HL7 FHIR Integration:**  
   Implement a DICOM C-STORE SCP listener to ingest 16-bit uncompressed hospital DICOM streams directly from trauma radiology worklists.

