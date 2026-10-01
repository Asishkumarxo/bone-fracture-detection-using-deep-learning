"""
Part 10: Orientation, Cropping, and Preprocessing Side-by-Side Visualization
============================================================================
Visualizes:
  - Original Image
  - Preprocessed Model Input (224x224 padded tensor inverted back to [0, 1] RGB)
Saves to reports/preprocessing_comparison.png
"""

import os
import sys
import numpy as np
import torch
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from inference.preprocessing import preprocess_for_classifier, load_and_validate_image
from inference.model_registry import ModelRegistry

def main():
    img_path = 'processed/test/pelvis_patient16064_Negative_001.png'
    raw_img = Image.open(img_path)
    orig_w, orig_h = raw_img.size

    # Pass through inference preprocessor
    tensor = preprocess_for_classifier(img_path) # Shape: (1, 3, 224, 224)
    
    # Invert normalization: tensor * std + mean
    # std = [0.5, 0.5, 0.5], mean = [0.5, 0.5, 0.5] -> tensor * 0.5 + 0.5
    img_tensor = tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)
    img_tensor = np.clip(img_tensor * 0.5 + 0.5, 0.0, 1.0)

    # Plot side-by-side
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Panel 1: Original
    axes[0].imshow(raw_img, cmap='gray')
    axes[0].set_title(f"1. Original Radiograph ({orig_w}x{orig_h})\nTrue Anatomy: Pelvis / Hip", fontsize=12, fontweight='bold')
    axes[0].axis('off')

    # Panel 2: Preprocessed Model Input
    axes[1].imshow(img_tensor)
    axes[1].set_title(f"2. Preprocessed Model Input (224x224)\nAspect-Ratio Preserved + Symmetric Zero-Padding", fontsize=12, fontweight='bold')
    axes[1].axis('off')

    plt.suptitle("Preprocessing Integrity Check: Original vs Model Input", fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()

    os.makedirs('reports', exist_ok=True)
    out_path = 'reports/preprocessing_comparison.png'
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved {out_path} successfully.")

if __name__ == '__main__':
    main()
