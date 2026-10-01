# Model Checkpoints & Architecture Guide

This directory documents the neural network architectures, checkpoint conventions, and training procedures used by the Bone Fracture AI system.

> [!IMPORTANT]
> **Trained checkpoint weights (`.pt`, `.pth`, `.bin`) are excluded from GitHub** to maintain a lean repository and comply with version control best practices. Users can either place pretrained weights into `models/checkpoints/` or train new models using the provided training scripts.

---

## 1. Required Checkpoint Files

| Model Identifier | Filename | Target Location | Framework | Description |
| :--- | :--- | :--- | :--- | :--- |
| **Multi-Task Classifier** | `best_model.pt` | `models/checkpoints/best_model.pt` | PyTorch | ResNet-50 joint anatomical classifier (7 classes) & binary fracture detector |
| **Fracture Detector** | `best_detector.pt` | `models/checkpoints/best_detector.pt` | TorchVision | Faster R-CNN MobileNetV3-Large FPN lesion localization detector |

---

## 2. Checkpoint Resolution Hierarchy

The inference pipeline and REST backend automatically resolve checkpoint locations in the following order:

1. **Environment Variables:**
   * `CLASSIFIER_CHECKPOINT`: Custom path to classifier weights.
   * `DETECTOR_CHECKPOINT`: Custom path to detector weights.
2. **Canonical Project Folder:**
   * `models/checkpoints/best_model.pt`
   * `models/checkpoints/best_detector.pt`
3. **Repository Root Fallback:**
   * `./best_model.pt`
   * `./best_detector.pt`

---

## 3. Architecture Details

### Model A: Multi-Task Classifier (`best_model.pt`)
* **Backbone:** ResNet-50 feature extractor (2048-dimensional feature embedding).
* **Input Resolution:** `(1, 3, 224, 224)` (Grayscale image with aspect-ratio preserved padding, broadcast to 3 channels, normalized with mean `[0.5, 0.5, 0.5]` and std `[0.5, 0.5, 0.5]`).
* **Head 1 (Anatomical Region):** Linear(2048, 512) -> BatchNorm1d -> ReLU -> Dropout(0.3) -> Linear(512, 7).
  * Output: 7-class logits (`Arm`, `Foot`, `Hand`, `Lower leg`, `Thigh`, `pelvis`, `wrist`).
* **Head 2 (Fracture Detection):** Linear(2048, 256) -> BatchNorm1d -> ReLU -> Dropout(0.3) -> Linear(256, 1).
  * Output: 1-dim scalar logit passed through Sigmoid. Operating threshold: `0.50`.

### Model B: Fracture Lesion Localization (`best_detector.pt`)
* **Architecture:** `fasterrcnn_mobilenet_v3_large_fpn`.
* **Input Resolution:** `(1, 3, 384, 384)` (RGB resized bilinearly to 384×384).
* **Classes:** 2 classes (Class 0: Background, Class 1: Fracture).
* **Coordinate Mapping:** Bounding box coordinates generated on the 384×384 grid are automatically scaled back to the native image resolution `(orig_w, orig_h)` during inference.
* **Calibrated Confidence Threshold:** `0.10` operating threshold (captures subtle cortical disruption).

---

## 4. How to Train Models from Scratch

### Training the Multi-Task Classifier:
Ensure `train.csv` and `validation.csv` are generated via `python preprocess.py`, then run:

```bash
# Full two-stage training
python train.py --backbone resnet50 --epochs_stage1 3 --epochs_stage2 2 --batch_size 32

# Fast baseline training on CPU (with sample limit)
python train.py --backbone resnet50 --max_train_samples 1000 --max_val_samples 250
```

### Training the Fracture Lesion Detector:
Ensure FracAtlas is downloaded in `data/FracAtlas`, then run:

```bash
python train_detector.py --img_size 384 --epochs 5 --batch_size 4
```
