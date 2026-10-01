# Inference Pipeline & Caption Generation Specification

## 1. Unified Inference Flow

The inference pipeline takes an uploaded radiograph image and executes the following sequential steps:

1. **Integrity Validation:** Validates image format and loadability using PIL.
2. **Preprocessing:** Applies aspect-ratio preserved padding and normalization to produce a $(1, 3, 224, 224)$ tensor for classification and a $(1, 3, 384, 384)$ tensor for detection.
3. **Anatomical Region Classification:** Computes Softmax probabilities over all 7 classes; extracts top predicted class and confidence.
4. **Uncertainty Evaluation:** If winning confidence falls below $\tau_{\text{uncertainty}} = 0.40$, an uncertainty flag is appended to the diagnostic findings.
5. **Fracture Detection:** Applies Sigmoid to the scalar fracture logit. If $p \ge 0.50$, fracture is predicted as `True`.
6. **Conditional Localization:** If fracture is detected AND detector weights are loaded, Faster R-CNN runs on the 384×384 tensor. Boxes above $\tau_{\text{det}} = 0.10$ are mapped back to native dimensions. If no fracture is detected, localization is bypassed.
7. **Factual Caption Generation:** Synthesizes a clinically disciplined natural-language description using verified findings.
8. **Optional Visualization:** Renders bounding box overlays, diagnostic banners, and caption callouts.

---

## 2. Disciplined Medical Caption Generation

The caption synthesizer adheres strictly to 10 clinical validation rules:

| Condition | Generated Caption Template |
| :--- | :--- |
| **No Fracture Detected** | `"X-ray of the {region} with no fracture detected by the model."` |
| **Fracture Detected (No sub-location)** | `"X-ray of the {region} showing a fracture."` |
| **Fracture Detected (With sub-location)** | `"X-ray of the {region} showing a fracture involving the {location}."` |

### Strict Hallucination Prohibitions:
1. **Never infers fracture displacement** (e.g., displaced, angulated, rotated).
2. **Never infers fracture subtypes** (e.g., transverse, spiral, comminuted, hairline, greenstick).
3. **Never infers clinical severity** (e.g., mild, severe, acute).
4. **Never hallucinates patient demographics** (e.g., age, sex).
5. **Never provides medical recommendations** (e.g., surgery, cast, immobilization).
