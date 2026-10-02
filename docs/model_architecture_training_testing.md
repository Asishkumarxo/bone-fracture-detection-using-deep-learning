# Bone Fracture Image Captioning Using Deep Learning — Model Architecture, Training and Testing

**Document Type:** Technical Architecture, Training Specification & Testing Report  
**Project:** Bone Fracture Image Captioning Using Deep Learning  
**Target Audience:** B.Tech CSE / AI-ML Students, Academic Reviewers, Viva Examiners  
**Status:** Verified Against Active Implementation & Checkpoints  

---

## 1. Model Overview

The core computational engine of this project is a **Multi-Task Deep Convolutional Neural Network (CNN)** built upon a **ResNet-50** backbone. Rather than utilizing separate networks for different radiographic assessment tasks or relying on an ungrounded natural language generative model, the architecture unifies two distinct diagnostic objectives into a single forward pass:

1. **Anatomical Region Classification:** Identifying the body part imaged across **7 mutually exclusive musculoskeletal regions** (*Arm, Foot, Hand, Lower leg, Pelvis, Thigh, Wrist*).
2. **Binary Fracture Detection:** Detecting the presence or absence of a cortical bone disruption (*Positive / Fracture* vs. *Negative / Normal*).

```
                      Input Plain Radiograph
              [3 × 224 × 224]  or  [3 × 448 × 448]
                                 │
                                 ▼
                     ResNet-50 Shared Backbone
                 (conv1 → layer1-4 → avgpool)
                                 │
                                 ▼
                    Shared 2,048-d Feature Vector
                                 │
         ┌───────────────────────┴───────────────────────┐
         ▼                                               ▼
Anatomical Head (Linear-512)                    Fracture Head (Linear-256)
         │                                               │
         ▼                                               ▼
   7-Class Logits                                  1 Binary Logit
         │                                               │
   Softmax (τ_reg)                                 Sigmoid (τ=0.50)
         │                                               │
         ▼                                               ▼
Predicted Anatomical Region                     Fracture Status & Probability
   (e.g., "Wrist", 94.2%)                           (e.g., "Positive", 78.5%)
         │                                               │
         └───────────────────────┬───────────────────────┘
                                 │
                                 ▼
                 Rule-Based Caption Generator
                                 │
                                 ▼
       Factual Clinical Sentence (No Hallucinations)
```

The multi-task model outputs structured numerical predictions (region index, region confidence, fracture probability, and binary label). These predictions are subsequently transformed into a factual, medically disciplined English sentence by a deterministic, rule-based caption generator.

---

## 2. Why ResNet-50

### 2.1 Convolutional Neural Networks for Radiography
Plain radiographs (X-rays) are projectional shadowgraphs where tissue density dictates beam attenuation: dense cortical bone appears radiopaque (white/light gray), while soft tissue and air appear radiolucent (dark). Identifying fractures requires detecting subtle visual anomalies:
- Sharp cortical discontinuities (hairline fissures, crack lines)
- Step-offs, impactions, or angular bone deformities
- Disrupted trabecular structural architecture

Standard fully connected artificial neural networks fail on high-resolution images because they do not preserve spatial 2D relationships and suffer from combinatorial parameter explosion. CNNs solve this via **local receptive fields**, **shared convolutional kernels (weights)**, and **translation equivariance**, capturing hierarchical visual primitives: edges and intensity gradients in early layers, texture and bone boundaries in middle layers, and complex joint/shaft morphology in deep layers.

### 2.2 Deep Feature Extraction & The Degradation Problem
As CNN architectures become deeper (e.g., VGG-16, VGG-19), they can learn richer hierarchical representations. However, historically, stacking additional convolutional layers led to the **degradation problem**: as network depth increases, accuracy saturates and then rapidly degrades. This is not caused by overfitting (training error increases as well) but by vanishing or exploding gradients during backpropagation:

$$\frac{\partial \mathcal{L}}{\partial W_1} = \frac{\partial \mathcal{L}}{\partial z_L} \prod_{l=2}^{L} \frac{\partial z_l}{\partial z_{l-1}} \cdot \frac{\partial z_1}{\partial W_1}$$

When multiple Jacobian matrices have eigenvalues strictly less than 1, repeated multiplication causes gradients flowing back to early layers to decay exponentially toward zero.

### 2.3 Residual Learning & Skip Connections
ResNet (He et al., 2015) resolves the degradation problem by introducing **shortcut (skip or residual) connections**. Instead of forcing stacked layers to fit an underlying mapping $\mathcal{H}(\mathbf{x})$ directly, ResNet reformulates the objective so that the stacked layers approximate a residual function:

$$\mathcal{F}(\mathbf{x}) = \mathcal{H}(\mathbf{x}) - \mathbf{x} \implies \mathcal{H}(\mathbf{x}) = \mathcal{F}(\mathbf{x}) + \mathbf{x}$$

```
                x (Input to Residual Block)
                │───────┐ (Identity Shortcut)
                ▼       │
           [Conv 1×1]   │
                ▼       │
           [Conv 3×3]   │
                ▼       │
           [Conv 1×1]   │
                │       │
                ▼       │
              ( + ) ◄───┘
                │
                ▼
             ReLU(y)
```

**Why this is crucial for X-ray analysis:**
1. **Identity Preconditioning:** If identity mapping is optimal for a layer, the network simply drives the weights of $\mathcal{F}(\mathbf{x})$ toward zero ($\mathcal{F}(\mathbf{x}) = 0$), which is significantly easier to learn than fitting $\mathcal{H}(\mathbf{x}) = \mathbf{x}$ from scratch.
2. **Gradient Highway:** During backpropagation, the identity term $\mathbf{x}$ creates an uninterrupted gradient highway:
   $$\frac{\partial \mathcal{L}}{\partial \mathbf{x}} = \frac{\partial \mathcal{L}}{\partial \mathcal{H}} \left( \frac{\partial \mathcal{F}}{\partial \mathbf{x}} + \mathbf{I} \right)$$
   Even if the residual gradient $\frac{\partial \mathcal{F}}{\partial \mathbf{x}}$ becomes negligible, the identity matrix $\mathbf{I}$ guarantees that gradients propagate backward to earlier layers without diminishing.
3. **ResNet-50 Bottleneck Architecture:** ResNet-50 utilizes 3-layer bottleneck building blocks ($1\times 1$ conv for dimension reduction $\to$ $3\times 3$ conv $\to$ $1\times 1$ conv for restoration), providing high representational capacity (25.6M parameters) while maintaining computational tractability on modest GPU hardware.

---

## 3. Multi-Task Learning Architecture

In standard computer vision pipelines, anatomical site classification and pathology detection are frequently implemented as separate networks. In this project, they are unified into a **Multi-Task Learning (MTL)** network with a **hard parameter-sharing** backbone.

```
                              Input Image x
                        [B, 3, H, W] (224 or 448)
                                    │
                                    ▼
                          ResNet-50 Backbone
                     conv1: 7×7, 64, stride 2
                     bn1, relu, maxpool 3×3
                     layer1: 3 Bottleneck blocks (256-d)
                     layer2: 4 Bottleneck blocks (512-d)
                     layer3: 6 Bottleneck blocks (1024-d)
                     layer4: 3 Bottleneck blocks (2048-d)
                     avgpool: AdaptiveAvgPool2d((1, 1))
                     Flatten()
                                    │
                                    ▼
                       Shared Embedding Vector
                            [B, 2048]
                                    │
                    ┌───────────────┴───────────────┐
                    ▼                               ▼
          Task 1: Region Head             Task 2: Fracture Head
          Linear(2048 → 512)              Linear(2048 → 256)
          BatchNorm1d(512)                BatchNorm1d(256)
          ReLU(inplace=True)              ReLU(inplace=True)
          Dropout(p=0.3)                  Dropout(p=0.3)
          Linear(512 → 7)                 Linear(256 → 1)
                    │                               │
                    ▼                               ▼
              Region Logits                  Fracture Logit
                [B, 7]                          [B, 1]
```

### Why Share the Backbone?
1. **Biological and Radiographic Feature Synergy:** Determining whether a radiograph contains a fracture is fundamentally dependent on recognizing the bone's anatomy. The normal cortex of a femoral shaft is thick and dense; the carpal bones of the wrist are small, complex, and overlap extensively. A shared backbone forces the lower and middle convolutional representations to capture anatomical geometry and tissue contrast simultaneously.
2. **Regularization and Reduced Overfitting:** Training on two tasks concurrently acts as a regularizer. The gradient updates from the anatomical classification head prevent the backbone from overfitting to spurious high-frequency artifacts that might correlate with fractures in a small subset of images.
3. **Computational Efficiency:** A single forward pass through ResNet-50 extracts the 2,048-dimensional feature vector once, reducing GPU memory footprint and inference latency by approximately 50% compared to running two distinct models.

---

## 4. Anatomical Classification Head

The anatomical region head maps the 2,048-dimensional shared feature embedding to class logits across 7 verified musculoskeletal categories:

$$\mathbf{z}_{\text{region}} = \mathbf{W}_2 \cdot \text{Dropout}_{0.3}\left(\text{ReLU}\left(\text{BN}\left(\mathbf{W}_1 \mathbf{h} + \mathbf{b}_1\right)\right)\right) + \mathbf{b}_2$$

Where:
- $\mathbf{h} \in \mathbb{R}^{2048}$: Global average pooled feature vector from the backbone.
- $\mathbf{W}_1 \in \mathbb{R}^{512 \times 2048}, \mathbf{b}_1 \in \mathbb{R}^{512}$: Dimensionality reduction linear layer.
- $\text{BN}$: One-dimensional batch normalization over the 512 intermediate features.
- $\text{Dropout}(p=0.3)$: 30% inverted dropout to prevent co-adaptation of hidden units.
- $\mathbf{W}_2 \in \mathbb{R}^{7 \times 512}, \mathbf{b}_2 \in \mathbb{R}^{7}$: Final classification projection.

### Softmax Normalization
The raw logits $\mathbf{z}_{\text{region}} = [z_1, z_2, \dots, z_7]$ are mapped to a normalized probability distribution using the Softmax function:

$$P(\text{Region} = k \mid \mathbf{x}) = \frac{\exp(z_k)}{\sum_{j=1}^{7} \exp(z_j)}, \quad k \in \{0, 1, \dots, 6\}$$

The predicted anatomical site $\hat{y}_{\text{region}}$ is selected by the argmax operator, and its corresponding probability serves as the confidence score:

$$\hat{y}_{\text{region}} = \arg\max_{k \in \{0, \dots, 6\}} P(\text{Region} = k \mid \mathbf{x}), \quad c_{\text{region}} = \max_{k} P(\text{Region} = k \mid \mathbf{x})$$

The 7 supported anatomical categories and their internal indices:
- `0`: Arm
- `1`: Foot
- `2`: Hand
- `3`: Lower leg
- `4`: Pelvis
- `5`: Thigh
- `6`: Wrist

---

## 5. Binary Fracture Classification Head

The fracture detection head maps the 2,048-dimensional shared embedding to a single scalar logit:

$$z_{\text{fracture}} = \mathbf{w}_2^T \cdot \text{Dropout}_{0.3}\left(\text{ReLU}\left(\text{BN}\left(\mathbf{W}_1 \mathbf{h} + \mathbf{b}_1\right)\right)\right) + b_2$$

Where:
- $\mathbf{W}_1 \in \mathbb{R}^{256 \times 2048}, \mathbf{b}_1 \in \mathbb{R}^{256}$: Compression layer.
- $\text{BN}$: Batch normalization across 256 feature channels.
- $\mathbf{w}_2 \in \mathbb{R}^{256}, b_2 \in \mathbb{R}$: Final scalar logit projection.

### Sigmoid Activation & Probability Output
The scalar logit $z_{\text{fracture}} \in (-\infty, +\infty)$ is mapped to the unit interval $[0, 1]$ via the standard logistic sigmoid function:

$$p_{\text{fracture}} = \sigma(z_{\text{fracture}}) = \frac{1}{1 + \exp(-z_{\text{fracture}})}$$

### Decision Threshold
In binary medical classification, the choice of decision threshold $\tau$ determines the trade-off between Sensitivity (Recall) and Specificity. In this project:
- **Verified Decision Threshold:** $\tau = 0.50$
- **Classification Rule:**
  $$\hat{y}_{\text{fracture}} = \begin{cases} 1 \ (\text{Positive / Fracture Detected}), & \text{if } p_{\text{fracture}} \ge 0.50 \\ 0 \ (\text{Negative / No Fracture Detected}), & \text{if } p_{\text{fracture}} < 0.50 \end{cases}$$

---

## 6. Model Output vs. Caption Generation

A critical architectural distinction must be understood for technical examinations and viva defense:

> [!IMPORTANT]
> The ResNet-50 neural network does **NOT** generate English text. It does **NOT** contain an RNN, LSTM, Transformer decoder, or LLM token generator.

### Direct Neural Network Outputs:
The PyTorch model `MultiTaskModel.forward(x)` returns strictly two numeric tensors:
1. `region_logits`: Tensor of shape `[B, 7]`
2. `fracture_logits`: Tensor of shape `[B]`

From these tensors, the inference service extracts four deterministic attributes:
- `anatomical_region` (str, e.g., `"wrist"`)
- `anatomical_confidence` (float, e.g., `0.942`)
- `fracture` (bool, `True` or `False`)
- `fracture_confidence` (float, e.g., `0.785`)

### Separate Caption Generator (`inference/caption_generator.py`):
The text caption is constructed downstream by a **rule-based, deterministic caption generator**. This module maps the model's verified outputs into standardized clinical templates:
- When $p \ge 0.50$: `"Plain radiograph of the {region} demonstrates findings consistent with a fracture (confidence: {confidence}%)."`
- When $p < 0.50$: `"Plain radiograph of the {region} shows no definitive displaced fracture."`

This design prevents natural language hallucinations (such as inventing non-existent fracture subtypes, lateralities, or treatment plans) and ensures that every phrase in the generated sentence is strictly grounded in the neural network's numeric outputs.

---

## 7. Loss Function

During training, the multi-task model optimizes a combined objective function comprising multi-class Cross-Entropy loss for anatomy and weighted Binary Cross-Entropy loss for fracture presence:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{region}} + \lambda \cdot \mathcal{L}_{\text{fracture}}$$

Where $\lambda = 1.0$ in baseline training and $1.5$ in fine-tuning runs.

### 7.1 Anatomical Loss ($\mathcal{L}_{\text{region}}$)
Standard categorical Cross-Entropy loss across the $C=7$ mutually exclusive anatomical classes:

$$\mathcal{L}_{\text{region}} = -\sum_{k=0}^{6} y_{\text{region}, k} \log\left(\frac{\exp(z_k)}{\sum_{j=0}^{6} \exp(z_j)}\right)$$

Where $y_{\text{region}, k} \in \{0, 1\}$ is the one-hot encoded ground-truth anatomical target.

### 7.2 Fracture Loss ($\mathcal{L}_{\text{fracture}}$) with Positive Class Weighting
The training split exhibits significant class imbalance: **27,205 Negative (70.91%)** vs. **11,161 Positive (29.09%)** cases (an imbalance ratio of approximately $2.44:1$). Without correction, standard binary cross-entropy encourages the model to predict the majority negative class.

To counteract this, the fracture loss utilizes `nn.BCEWithLogitsLoss` parameterized with a positive class weight $w_{\text{pos}}$:

$$w_{\text{pos}} = \frac{N_{\text{negative}}}{N_{\text{positive}}} = \frac{27,205}{11,161} \approx 2.4375$$

The weighted loss formulation for sample $i$ is:

$$\mathcal{L}_{\text{fracture}}(z_i, y_i) = - \left[ w_{\text{pos}} \cdot y_i \cdot \log(\sigma(z_i)) + (1 - y_i) \cdot \log(1 - \sigma(z_i)) \right]$$

- For **Positive** cases ($y_i = 1$), the gradient is scaled up by $2.4375\times$, heavily penalizing false negatives.
- For **Negative** cases ($y_i = 0$), the gradient scaling is standard ($1.0\times$).

---

## 8. Training Procedure

The multi-task model was trained using a disciplined **two-stage transfer learning protocol** designed to adapt ImageNet visual features to radiographic patterns while preventing catastrophic forgetting.

```
┌────────────────────────────────────────────────────────────────────────┐
│ STAGE 1: Linear Probing / Feature Extraction (Frozen Backbone)         │
│ • Backbone (conv1 → layer4) is completely frozen (requires_grad=False) │
│ • Only region_head and fracture_head parameters are optimized          │
│ • Trainable Parameters: ~1,577,000                                     │
│ • Optimizer: AdamW, lr = 1e-3, weight_decay = 1e-4                     │
│ • Scheduler: CosineAnnealingLR (eta_min = 1e-5)                        │
└────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ STAGE 2: Differential Fine-Tuning (Upper Backbone Unfrozen)            │
│ • Lower backbone (conv1 → layer3) remains frozen                       │
│ • Upper residual block (layer4) and both heads are unfrozen            │
│ • Differential Learning Rates:                                         │
│     - layer4 parameters: lr = 5e-5 (cautious domain adaptation)        │
│     - classification heads: lr = 1e-4                                  │
│ • Optimizer: AdamW, weight_decay = 1e-4                                │
│ • Scheduler: ReduceLROnPlateau (mode='min', factor=0.5, patience=1)    │
└────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ Checkpoint Selection:                                                  │
│ • Monitored strictly on untouched Validation Split (4,787 images)      │
│ • Selection criterion: Minimum combined validation loss                │
│ • Test set (4,778 images) strictly locked until evaluation             │
└────────────────────────────────────────────────────────────────────────┘
```

### Verified Training Hyperparameters

| Parameter | Baseline Model (224×224) | Experiment 3 Model (448×448) |
| :--- | :--- | :--- |
| **Script** | `train.py` | `scripts/train_exp3_448.py` |
| **Input Resolution** | $224 \times 224$ pixels | $448 \times 448$ pixels |
| **Backbone Pretraining** | ImageNet-1k (`DEFAULT`) | ImageNet-1k (`DEFAULT`) |
| **Batch Size** | 32 | 16 (adjusted for 4GB VRAM) |
| **Optimizer** | AdamW ($\beta_1=0.9, \beta_2=0.999$) | AdamW ($\beta_1=0.9, \beta_2=0.999$) |
| **Weight Decay** | $1 \times 10^{-4}$ | $1 \times 10^{-4}$ |
| **Stage 1 Epochs / LR** | 5 epochs / $1 \times 10^{-3}$ | 3 epochs / $1 \times 10^{-3}$ |
| **Stage 2 Epochs / LR** | 5 epochs / $\eta_{\text{layer4}}=5\times 10^{-5}, \eta_{\text{head}}=1\times 10^{-4}$ | 3 epochs / $\eta_{\text{layer4}}=5\times 10^{-5}, \eta_{\text{head}}=1\times 10^{-4}$ |
| **Best Checkpoint** | `models/checkpoints/best_model.pt` | `models/checkpoints/exp3_resnet50_448.pt` |

---

## 9. Baseline Model (224×224)

The baseline model (`best_model.pt`) operates at the standard convolutional vision resolution of $224 \times 224$ pixels. It serves three vital roles:
1. **Architectural Benchmark:** Provides the reference baseline against which all subsequent experiments and high-resolution scaling are compared.
2. **Low-Latency Engine:** Executes in ~15 ms on modern GPUs, requiring minimal memory overhead.
3. **Small-Image Specialist:** Highly effective on radiographs whose native dimensions are small (e.g., $102 \times 102$ thumbnails), where upscaling to $448 \times 448$ introduces interpolation blur without adding diagnostic information.

---

## 10. Experiment 3 — 448×448 High-Resolution Model

Experiment 3 was executed to test the hypothesis that increasing classifier resolution from $224 \times 224$ to $448 \times 448$ preserves micro-trabecular patterns and hairline cortical disruptions that are otherwise lost during downsampling.

### Comparative Test Set Evaluation ($N=4,778$ Untouched Images, $\tau=0.50$)

| Evaluation Metric | Baseline Model (224×224) | Experiment 3 Model (448×448) | Absolute Improvement | Relative Improvement |
| :--- | :---: | :---: | :---: | :---: |
| **Accuracy** | 71.31% | **81.88%** | **+10.57%** | +14.82% |
| **Precision (PPV)** | 50.28% | **68.45%** | **+18.17%** | +36.14% |
| **Recall (Sensitivity)** | 71.49% | **69.25%** | -2.24% | -3.13% |
| **Specificity (TNR)** | 71.23% | **87.01%** | **+15.78%** | +22.15% |
| **F1-Score** | 0.5904 | **0.6885** | **+0.0981** | +16.62% |
| **ROC-AUC** | 0.7889 | **0.8770** | **+0.0881** | +11.17% |
| **PR-AUC** | 0.6790 | **0.7998** | **+0.1208** | +17.79% |

### Confusion Matrix Comparison on 4,778 Test Images

```
BASELINE MODEL (224×224):                EXPERIMENT 3 MODEL (448×448):
Total Positives = 1,382                  Total Positives = 1,382
Total Negatives = 3,396                  Total Negatives = 3,396

           Predicted                                Predicted
         Neg        Pos                           Neg        Pos
Actual                    Actual
 Neg    2,419 (TN)  977 (FP)        Neg    2,955 (TN)  441 (FP)
 Pos      394 (FN)  988 (TP)        Pos      425 (FN)  957 (TP)
```

**Key Takeaways from the Confusion Matrix:**
- **Massive Reduction in False Positives:** False positive predictions dropped from **977 down to 441** (a 54.9% reduction, eliminating 536 false alarms).
- **Substantial Specificity Surge:** Specificity climbed from 71.23% to 87.01%, meaning the high-resolution network is far less prone to misinterpreting normal bone contours or radiographic shadows as fractures.
- **Controlled Sensitivity:** Sensitivity remained stable (69.25% vs 71.49%), while Precision increased dramatically from 50.28% to 68.45%.

---

## 11. Resolution-Specific Results: The Resolution Bottleneck

When the Experiment 3 evaluation results are partitioned by the **native resolution** of the original test radiographs, a critical empirical finding emerges:

```
                    4,778 Test Radiographs
                              │
         ┌────────────────────┴────────────────────┐
         ▼                                         ▼
102×102 Thumbnail Cohort                 Genuinely High-Res Cohort
    (N = 3,461 images)                      (N = 1,317 images)
  • Positive Cases: 568                   • Positive Cases: 814
  • Negative Cases: 2,893                 • Negative Cases: 503
  • True Positives: 217                   • True Positives: 740
  • False Negatives: 351                  • False Negatives: 74
  • Recall: 38.20%                        • Recall: 90.91%
  • Specificity: 93.16%                   • Specificity: 51.69%
```

### Verified Resolution Partition Breakdown (From `reports/exp3_evaluation_report.json`)

| Partition | Total Images | Positive Cases | True Positives (TP) | False Negatives (FN) | Sensitivity (Recall) | Specificity |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Genuinely High-Res Radiographs** | 1,317 | 814 | **740** | **74** | **90.91%** | 51.69% |
| **Native 102×102 Thumbnail Radiographs** | 3,461 | 568 | **217** | **351** | **38.20%** | 93.16% |

### Scientific Significance
1. **The 102×102 Bottleneck:** 72.4% of the test set consists of native $102 \times 102$ thumbnails. When a $102 \times 102$ image is upscaled by $4.39\times$ to fit a $448 \times 448$ grid, bilinear interpolation blurs fine boundaries. A hairline fissure occupying 1–2 pixels in the thumbnail becomes an indistinct blur. Consequently, the 448×448 model fails to detect them, missing 351 out of 568 fractures (38.20% recall).
2. **True High-Res Diagnostic Power:** On genuine high-resolution radiographs where native resolution is preserved, the 448×448 model achieves an outstanding **90.91% Sensitivity (740 / 814 fractures detected)**.
3. **Engineering Implication:** This empirical finding directly led to the implementation of **Resolution-Aware Inference Routing** in the production pipeline, ensuring that images are dispatched to the model best suited for their native pixel density.

---

## 12. Testing Methodology

The testing protocol followed strict clinical machine learning hygiene to guarantee zero data leakage and uncompromised evaluation integrity:

```
[Mendeley BoneFract Corpus (47,931 Radiographs)]
                        │
      Graph-Connected Component Decomposition
           (Patient & Duplicate Clustered)
                        │
         ┌──────────────┼──────────────┐
         ▼              ▼              ▼
    Train (80%)      Val (10%)     Test (10%)
   38,366 Images   4,787 Images   4,778 Images
         │              │              │
         │ (Training)   │ (Tuning)     │ (LOCKED)
         ▼              ▼              │
    Model Training → Checkpoint        │
                       Selection       │
                           │           │
                           ▼           ▼
                   Final Benchmark Evaluation
                     (Single-Pass Execution)
```

1. **Validation Split Independence:** Checkpoint selection and early stopping monitored only the 4,787-image validation set. The test set was locked.
2. **Zero Split Leakage:** Exact MD5 hash collision audits verified 0 duplicate images across partitions, and patient-level clustering ensured 0 cross-partition patient overlap.
3. **Deterministic Operating Threshold:** The test evaluation utilized the fixed default threshold $\tau = 0.50$. No post-hoc threshold tuning was applied to the test split.
4. **Standardized Confusion Terminology:**
   - **True Positive (TP):** Ground-truth fracture correctly classified as fracture ($p \ge 0.50$).
   - **True Negative (TN):** Ground-truth normal radiograph correctly classified as normal ($p < 0.50$).
   - **False Positive (FP):** Normal radiograph incorrectly flagged as fractured ($p \ge 0.50$).
   - **False Negative (FN):** Fractured radiograph incorrectly classified as normal ($p < 0.50$).

---

## 13. Evaluation Metrics Explained

To ensure precise viva defense, each metric is defined mathematically with its specific clinical engineering interpretation:

### 13.1 Accuracy
The overall proportion of correct predictions across all cases:
$$\text{Accuracy} = \frac{\text{TP} + \text{TN}}{\text{TP} + \text{TN} + \text{FP} + \text{FN}}$$
*Interpretation:* While straightforward, accuracy can be misleading in imbalanced datasets. Here, accuracy reached 81.88%.

### 13.2 Sensitivity / Recall
The ability of the model to correctly identify patients who actually have a fracture:
$$\text{Sensitivity} = \frac{\text{TP}}{\text{TP} + \text{FN}}$$
*Interpretation:* In a clinical triage setting, sensitivity is critical because a false negative (missed fracture) can lead to non-union or long-term disability. On high-resolution radiographs, sensitivity reached 90.91%.

### 13.3 Specificity (True Negative Rate)
The ability of the model to correctly identify normal radiographs:
$$\text{Specificity} = \frac{\text{TN}}{\text{TN} + \text{FP}}$$
*Interpretation:* High specificity (87.01% in Exp 3) minimizes unnecessary hospital admissions, unnecessary immobilization, and redundant CT scans.

### 13.4 Positive Predictive Value (PPV) / Precision
The probability that a patient flagged as having a fracture actually has one:
$$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}$$
*Interpretation:* In Exp 3, precision improved by +18.17% (reaching 68.45%), meaning over two-thirds of positive alerts are genuine fractures.

### 13.5 F1-Score
The harmonic mean of precision and recall:
$$\text{F1} = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}} = \frac{2\text{TP}}{2\text{TP} + \text{FP} + \text{FN}}$$
*Interpretation:* Provides a balanced single metric when precision and recall must be weighed equally under class imbalance (0.6885 in Exp 3).

### 13.6 Receiver Operating Characteristic — Area Under Curve (ROC-AUC)
Measures the model's ability to discriminate between fractured and normal cases across all possible decision thresholds $\tau \in [0, 1]$:
$$\text{ROC-AUC} = \int_{0}^{1} \text{TPR}(\text{FPR}) \, d(\text{FPR})$$
*Interpretation:* An ROC-AUC of 0.8770 indicates an 87.7% probability that the model will score a randomly chosen fractured radiograph higher than a randomly chosen normal radiograph.

### 13.7 Precision-Recall Area Under Curve (PR-AUC)
Evaluates the trade-off between precision and recall across operating thresholds, particularly informative in low-prevalence settings:
$$\text{PR-AUC} = \int_{0}^{1} \text{Precision}(\text{Recall}) \, d(\text{Recall})$$
*Interpretation:* Reached 0.7998 in Exp 3 (compared to baseline prevalence of 28.92%), demonstrating substantial discriminative power.

---

## 14. Results Interpretation

A candid technical appraisal of the final experimental results reveals both significant capabilities and important engineering boundaries:

### Strengths:
1. **High Diagnostic Discrimination:** An ROC-AUC of **0.8770** and PR-AUC of **0.7998** confirm robust feature representation across diverse anatomical regions.
2. **Outstanding Performance on Large Bones:** Regions with substantial bone volume and clear cortical margins achieve high diagnostic metrics (Thigh Recall: **93.26%**, Specificity: 78.47%; Lower Leg Recall: **81.59%**, Specificity: 83.33%).
3. **False Positive Suppression:** The transition to $448 \times 448$ eliminated 536 false positives, raising specificity to 87.01%.

### Limitations:
1. **Small Complex Bones Remain Challenging:** Small bones with numerous overlapping articulations (Foot Recall: **40.49%**, Hand Recall: **43.14%**) remain difficult for a global classification backbone without localized attention.
2. **Thumbnail Image Degradation:** As proven in Section 11, the model cannot reliably detect hairline fractures in low-resolution $102 \times 102$ images due to irreversible interpolation blur.

---

## 15. External Image Testing (Secondary Evaluation)

To assess out-of-distribution generalization, 5 external bone radiographs sourced from web repositories were evaluated using the frozen pipeline (documented in `reports/exp3_evaluation_report.json`):

| Radiograph | Expected Ground Truth | Predicted Region | Fracture Prob | Predicted Label | Clinical Assessment |
| :--- | :--- | :---: | :---: | :---: | :--- |
| `2.jpg` | Positive (Distal radius fracture) | Hand | 78.47% | **Positive** | Correct fracture detection; slight region boundary overlap (Hand vs Wrist) |
| `wrist crack image.jpg` | Positive (Hairline wrist crack) | Hand | 27.82% | **Negative** | **False Negative**: Fine hairline fissure missed by global classifier |
| `AdobeStock_594927383-1.jpeg` | Positive (Forearm/Wrist fracture) | Wrist | 12.87% | **Negative** | **False Negative**: Hairline fracture missed due to contrast difference |
| `istockphoto-471457370-612x612.jpg`| Negative (Normal pelvis) | Wrist | 77.88% | **Positive** | **False Positive**: Contrast and cropping artifacts misclassified |
| `AdobeStock_200285274.webp` | Negative (Normal wrist) | Pelvis | 68.65% | **Positive** | **False Positive**: Out-of-distribution projection artifact |

### Technical Interpretation
This qualitative external benchmark illustrates standard **domain shift**: models trained on specific hospital cohorts encounter unfamiliar contrast curves, cropping geometries, and noise distributions in external images. This outcome provides strong empirical justification for presenting the system as a research and educational prototype rather than a clinically validated diagnostic tool.

---

## 16. Model Limitations

Documented and verified limitations supported by empirical testing:
1. **No Spatial Localization in Final Frontend:** The classifier outputs whole-image probabilities. It does not output bounding boxes or pixel-level segmentation masks on the web interface.
2. **No Subtype or Severity Characterization:** The model performs binary fracture classification. It cannot determine whether a fracture is transverse, oblique, spiral, comminuted, greenstick, displaced, or pathological.
3. **No Laterality Detection:** The model does not differentiate left from right limbs.
4. **Resolution Sensitivity:** Performance degrades significantly on images with native resolution below 300 pixels along either dimension.

---

## 17. Final Model Inference Flow

The complete end-to-end execution flow for a single input radiograph:

```
Step 1: Input Radiograph Received
        Format: PNG, JPG, JPEG, WEBP, or BMP
                        │
Step 2: Input Validation & Image Inspection
        PIL verifies decodability; dimensions (W, H) inspected
                        │
Step 3: Resolution-Aware Routing
        if min(W, H) >= 300 px:
            target_size = (448, 448), checkpoint = exp3_resnet50_448.pt
        else:
            target_size = (224, 224), checkpoint = best_model.pt
                        │
Step 4: Aspect-Ratio Preserved Preprocessing
        1. Aspect-ratio preserving resize (scale by min(target/W, target/H))
        2. Symmetric zero-padding to target_size
        3. Single-channel grayscale conversion
        4. Broadcast to 3-channel RGB tensor
        5. Normalization with mean=0.5, std=0.5
                        │
Step 5: ResNet-50 Multi-Task Forward Pass
        Tensors pass through shared backbone → 2,048-d embedding
        → Region Head (Linear-512) → 7 logits
        → Fracture Head (Linear-256) → 1 logit
                        │
Step 6: Metric Extraction
        Region: Softmax → predicted region string + confidence score
        Fracture: Sigmoid → probability p; label = (p >= 0.50)
                        │
Step 7: Rule-Based Caption Generation
        Inputs (region, p, confidence) injected into deterministic template
                        │
Step 8: Structured JSON Output
        Returned to FastAPI backend and rendered in Streamlit UI
```

---

## 18. Model Summary for Viva Preparation

For rapid reference during viva voce examination:

- **Architecture:** ResNet-50 shared backbone with two task-specific heads: a 7-class linear projection head for anatomical classification and a 1-class linear projection head for binary fracture detection.
- **Why ResNet-50:** Solves the degradation problem via residual skip connections ($\mathcal{H}(\mathbf{x}) = \mathcal{F}(\mathbf{x}) + \mathbf{x}$), preserving gradient flow and extracting rich hierarchical spatial features.
- **Loss Formulation:** Multi-task combined loss ($\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CE}} + \mathcal{L}_{\text{BCE}}$) with positive class weight $w_{\text{pos}} = 2.4375$ to balance the 71% normal vs 29% fracture class distribution.
- **Training Strategy:** Two-stage transfer learning using AdamW optimizer. Stage 1 freezes the backbone to train classification heads; Stage 2 unfreezes the upper `layer4` residual block with differential learning rates ($5\times 10^{-5}$ vs $1\times 10^{-4}$).
- **Experiment 3 Performance:** Retraining at $448 \times 448$ increased accuracy from 71.31% to **81.88%**, precision from 50.28% to **68.45%**, specificity from 71.23% to **87.01%**, and ROC-AUC from 0.7889 to **0.8770** across 4,778 untouched test images.
- **Resolution Bottleneck:** On genuine high-resolution images, sensitivity is **90.91%**; on upscaled $102 \times 102$ thumbnails, sensitivity drops to **38.20%** because interpolation cannot restore missing micro-fracture details.
- **Caption Generation:** Purely rule-based and deterministic; converts model-predicted region and fracture status into a factual sentence without natural language hallucinations.
