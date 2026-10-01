# Model Pipeline & Training Methodology

## 1. Multi-Task Classifier Architecture

The primary diagnostic classifier implements hard parameter sharing via a multi-task learning paradigm:

```
Input: Tensor (3, 224, 224)
  │
  ▼
[ResNet-50 Feature Extractor] ─── (2048-dim representation)
  ├──► [Anatomical Head: Linear(2048, 512) -> BN -> ReLU -> Dropout(0.3) -> Linear(512, 7)]
  └──► [Fracture Head:   Linear(2048, 256) -> BN -> ReLU -> Dropout(0.3) -> Linear(256, 1)]
```

### Two-Stage Transfer Learning:
* **Stage 1 (Frozen Backbone):** Backbone weights are frozen; only the task-specific heads are optimized using AdamW ($\eta = 10^{-3}$, weight decay $10^{-4}$) for 3 epochs with Cosine Annealing.
* **Stage 2 (Upper Unfrozen):** Layer 4 of ResNet-50 is unfrozen for fine-tuning at a reduced learning rate ($\eta = 10^{-4}$) for 2 epochs.

### Multi-Task Loss Formulation:
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{region}} + \lambda \mathcal{L}_{\text{fracture}}$$
* $\mathcal{L}_{\text{region}}$: Cross-Entropy Loss over 7 anatomical classes.
* $\mathcal{L}_{\text{fracture}}$: Binary Cross-Entropy with Logits, applying positive class weighting ($\text{pos\_weight} \approx 1.05$) to balance fracture occurrence.

---

## 2. Fracture Lesion Detector Architecture

* **Framework:** Faster R-CNN with MobileNetV3-Large Feature Pyramid Network (FPN).
* **Target Resolution:** 384×384 input resolution.
* **Anchor Generator:** Multi-scale aspect ratios $[0.5, 1.0, 2.0]$ optimized for linear and comminuted cortical fractures.
* **Loss Components:**
  $$\mathcal{L}_{\text{det}} = \mathcal{L}_{\text{rpn\_cls}} + \mathcal{L}_{\text{rpn\_box}} + \mathcal{L}_{\text{roi\_cls}} + \mathcal{L}_{\text{roi\_box}}$$
* **Calibrated Operating Threshold:** $\tau_{\text{det}} = 0.10$.

---

## 3. Class Taxonomy & Hip/Pelvis Discussion

The model classifies 7 anatomical regions:
`Arm`, `Foot`, `Hand`, `Lower leg`, `Thigh`, `pelvis`, `wrist`.

### Hip vs. Pelvis Clarification:
* **Dataset Truth:** The BoneFract training dataset catalogs pelvic and hip examinations under `pelvis`.
* **Clinical Relationship:** Hip radiographs feature the femoral head, neck, and proximal diaphysis. The classifier maps hip examinations to `pelvis`. When the proximal femoral diaphysis dominates the field of view, tubular bone features may also trigger similarity to `Thigh`.
* The system never fabricates an unsupported "Hip" class.
