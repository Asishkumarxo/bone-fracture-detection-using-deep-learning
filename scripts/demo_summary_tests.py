import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from predict import BoneFractureInferencePipeline

p = BoneFractureInferencePipeline()

samples = {
    'WRIST': 'c:/Users/arbaz/Downloads/wrist crack image.jpg',
    'PELVIS/HIP': 'c:/Users/arbaz/Downloads/AdobeStock_200285274.webp',
    'HAND': 'data/training_curated/Hand/Hand_Negative_002.jpg',
    'FOOT': 'data/training_curated/Foot/Foot_Negative_003.jpg',
    'ARM': 'data/training_curated/Arm/Arm_Negative_005.jpg',
    'LOWER LEG': 'error_analysis/false_positives/fp_Lower leg_patient35448_Negative_001.png.png',
    'THIGH': 'error_analysis/false_positives/fp_Thigh_patient35490_Negative_001.png.png'
}

for name, path in samples.items():
    res = p.analyze(path)
    print(f"=== {name} TEST ===")
    print(f"File: {path}")
    print(f"Predicted Region: {res['anatomical_region']}")
    print(f"Region Confidence: {res['anatomical_confidence']*100:.2f}%")
    print(f"Fracture Detected: {res['fracture']}")
    print(f"Fracture Probability: {res['fracture_confidence']*100:.2f}%")
    print(f"Top-3: {res.get('top_predictions')}")
    print(f"Caption: {res['caption']}")
    print()
