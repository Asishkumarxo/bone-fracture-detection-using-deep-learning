import os
import glob
import shutil
import pandas as pd
import numpy as np
from PIL import Image
import torch
import torchvision.transforms.functional as TF

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from predict import BoneFractureInferencePipeline
from inference.model_registry import ModelRegistry

def main():
    eval_dir = 'data/controlled_eval_35'
    os.makedirs(eval_dir, exist_ok=True)

    # 1. Pelvis (5 images)
    pelvis_imgs = sorted([f for f in glob.glob('error_analysis/pelvis_misclassified/*.png') if os.path.getsize(f) > 5000])[:5]
    
    # 2. Wrist (5 images)
    wrist_imgs = sorted([f for f in glob.glob('error_analysis/region_errors/reg_wrist_*.png') if os.path.getsize(f) > 5000])[:4]
    if os.path.exists('c:/Users/arbaz/Downloads/wrist crack image.jpg'):
        wrist_imgs.append('c:/Users/arbaz/Downloads/wrist crack image.jpg')
    else:
        wrist_imgs.append(sorted(glob.glob('error_analysis/region_errors/reg_wrist_*.png'))[4])

    # 3. Thigh (5 images)
    thigh_imgs = sorted([f for f in glob.glob('error_analysis/false_positives/fp_Thigh_*.png') if os.path.getsize(f) > 5000])[:4]
    thigh_imgs.append('error_analysis/region_errors/reg_Thigh_patient20798_Negative_001.png.png')

    # 4. Lower leg (5 images)
    leg_imgs = sorted([f for f in glob.glob('error_analysis/false_positives/fp_Lower leg_*.png') if os.path.getsize(f) > 5000])[:5]

    # 5. Hand (5 images)
    hand_imgs = []
    if os.path.exists('error_analysis/false_negatives/fn_Hand_patient08635_Positive_001.png.png'):
        hand_imgs.append('error_analysis/false_negatives/fn_Hand_patient08635_Positive_001.png.png')
    for hf in sorted(glob.glob('data/training_curated/Hand/*'))[:4]:
        hand_imgs.append(hf)
    hand_imgs = hand_imgs[:5]

    # 6. Foot (5 images)
    foot_imgs = []
    for ff in sorted(glob.glob('data/training_curated/Foot/*')):
        if os.path.getsize(ff) > 5000:
            foot_imgs.append(ff)
    # If fewer than 5, create distinct augmented versions
    while len(foot_imgs) < 5:
        src = foot_imgs[len(foot_imgs) % len(foot_imgs)]
        base, ext = os.path.splitext(src)
        aug_p = f"{base}_variant_{len(foot_imgs)}{ext}"
        with Image.open(src) as im:
            TF.hflip(im).save(aug_p)
        foot_imgs.append(aug_p)

    # 7. Arm (5 images)
    arm_imgs = []
    for af in sorted(glob.glob('data/training_curated/Arm/*')):
        if os.path.getsize(af) > 5000:
            arm_imgs.append(af)
    while len(arm_imgs) < 5:
        src = arm_imgs[len(arm_imgs) % len(arm_imgs)]
        base, ext = os.path.splitext(src)
        aug_p = f"{base}_variant_{len(arm_imgs)}{ext}"
        with Image.open(src) as im:
            TF.rotate(im, angle=5).save(aug_p)
        arm_imgs.append(aug_p)

    test_manifest = [
        ('wrist', wrist_imgs),
        ('Hand', hand_imgs),
        ('Foot', foot_imgs),
        ('pelvis', pelvis_imgs),
        ('Thigh', thigh_imgs),
        ('Lower leg', leg_imgs),
        ('Arm', arm_imgs)
    ]

    records = []
    pipeline = BoneFractureInferencePipeline()
    region_to_idx = ModelRegistry.REGION_TO_IDX

    print("\n" + "="*80)
    print("PHASE 6: RUNNING CONTROLLED OFFLINE TEST (BEFORE FIX)")
    print("="*80)

    for gt, img_list in test_manifest:
        print(f"\n--- Ground Truth: {gt} ({len(img_list)} images) ---")
        for idx, img_path in enumerate(img_list, 1):
            fname = os.path.basename(img_path)
            res = pipeline.analyze(img_path)
            dist = pipeline.region_predictor.predict(img_path)
            
            pred_region = res['anatomical_region']
            pred_idx = region_to_idx.get(pred_region, -1)
            conf = res['anatomical_confidence']
            probs = dist['probability_distribution']
            
            print(f"[{idx}/5] File: {fname[:35]:<35} | Pred: {pred_region:<10} | Conf: {conf*100:.2f}% | Correct: {pred_region.lower() == gt.lower()}")

            row = {
                'filename': fname,
                'ground_truth': gt,
                'predicted_region': pred_region,
                'predicted_index': pred_idx,
                'confidence': round(conf, 4),
                'probability_Arm': probs.get('Arm', 0.0),
                'probability_Foot': probs.get('Foot', 0.0),
                'probability_Hand': probs.get('Hand', 0.0),
                'probability_Lower_leg': probs.get('Lower leg', 0.0),
                'probability_Thigh': probs.get('Thigh', 0.0),
                'probability_pelvis': probs.get('pelvis', 0.0),
                'probability_wrist': probs.get('wrist', 0.0),
                'fracture_prediction': res['fracture'],
                'fracture_confidence': res['fracture_confidence'],
                'caption': res['caption']
            }
            records.append(row)

    df_out = pd.DataFrame(records)
    out_csv = 'reports/controlled_offline_test_before.csv'
    os.makedirs('reports', exist_ok=True)
    df_out.to_csv(out_csv, index=False)
    
    correct_count = sum(df_out['predicted_region'].str.lower() == df_out['ground_truth'].str.lower())
    total_count = len(df_out)
    acc = correct_count / total_count * 100.0
    print("\n" + "="*80)
    print(f"BEFORE EVALUATION SUMMARY: {correct_count}/{total_count} correct ({acc:.1f}% accuracy)")
    print(f"Saved complete results to: {out_csv}")
    print("="*80 + "\n")

if __name__ == '__main__':
    main()
