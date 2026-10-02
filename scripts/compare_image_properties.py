import os
from PIL import Image
import numpy as np

def inspect_img_properties(filepath):
    with Image.open(filepath) as im:
        w, h = im.size
        arr = np.array(im.convert('L'))
        print(f"\n--- {os.path.basename(filepath)} ---")
        print(f"  Dimensions: {w}x{h} (Aspect ratio: {w/h:.2f})")
        print(f"  Dynamic Range: [{arr.min()}, {arr.max()}]")
        print(f"  Mean: {arr.mean():.1f}, Std: {arr.std():.1f}")
        print(f"  Histogram quantiles [p5, p50, p95]: [{np.percentile(arr, 5):.1f}, {np.percentile(arr, 50):.1f}, {np.percentile(arr, 95):.1f}]")
        # Check border pixels (background)
        top = arr[0, :].mean()
        bottom = arr[-1, :].mean()
        left = arr[:, 0].mean()
        right = arr[:, -1].mean()
        print(f"  Border brightness: Top={top:.1f}, Bottom={bottom:.1f}, Left={left:.1f}, Right={right:.1f}")

# Dataset samples
dataset_samples = [
    'error_analysis/false_negatives/fn_Hand_patient08635_Positive_001.png.png',
    'error_analysis/false_negatives/fn_Arm_patient05411_Positive_001.png.png',
    'data/training_curated/Hand/Hand_Negative_002.jpg',
    'data/training_curated/Arm/Arm_Positive_000.png'
]

# External samples
external_samples = [
    'c:/Users/arbaz/Downloads/2.jpg',
    'c:/Users/arbaz/Downloads/istockphoto-471457370-612x612.jpg',
    'c:/Users/arbaz/Downloads/AdobeStock_594927383-1.jpeg',
    'c:/Users/arbaz/Downloads/wrist crack image.jpg'
]

print("=== DATASET SAMPLES ===")
for p in dataset_samples:
    if os.path.exists(p):
        inspect_img_properties(p)

print("\n=== EXTERNAL SAMPLES ===")
for p in external_samples:
    if os.path.exists(p):
        inspect_img_properties(p)
