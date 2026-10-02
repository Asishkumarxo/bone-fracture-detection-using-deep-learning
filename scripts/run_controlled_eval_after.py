import os
import glob
import pandas as pd
import numpy as np
from PIL import Image
import torch

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from predict import BoneFractureInferencePipeline
from inference.model_registry import ModelRegistry

def main():
    before_csv = 'reports/controlled_offline_test_before.csv'
    if not os.path.exists(before_csv):
        print(f"Error: {before_csv} not found.")
        return

    df_before = pd.read_csv(before_csv)
    
    # We will re-run the exact same files evaluated in before_csv
    pipeline = BoneFractureInferencePipeline()
    region_to_idx = ModelRegistry.REGION_TO_IDX

    print("\n" + "="*80)
    print("PHASE 6: RUNNING CONTROLLED OFFLINE TEST (AFTER FIX)")
    print("="*80)

    # Map filename to file paths by searching
    all_candidate_dirs = [
        'error_analysis/pelvis_misclassified',
        'error_analysis/region_errors',
        'error_analysis/false_positives',
        'error_analysis/false_negatives',
        'data/training_curated/Hand',
        'data/training_curated/Foot',
        'data/training_curated/Arm',
        'data/controlled_eval_35',
        'c:/Users/arbaz/Downloads'
    ]

    records = []
    correct_count = 0
    total_count = len(df_before)

    for idx, row in df_before.iterrows():
        fname = row['filename']
        gt = row['ground_truth']
        
        # Locate file
        target_path = None
        for d in all_candidate_dirs:
            p = os.path.join(d, fname)
            if os.path.exists(p):
                target_path = p
                break
        
        if not target_path:
            # Try searching recursively in error_analysis and data
            matches = glob.glob(f"**/{fname}", recursive=True)
            if matches:
                target_path = matches[0]

        if not target_path or not os.path.exists(target_path):
            print(f"Warning: could not find {fname}, skipping")
            continue

        res = pipeline.analyze(target_path)
        dist = pipeline.region_predictor.predict(target_path)
        
        pred_region = res['anatomical_region']
        pred_idx = region_to_idx.get(pred_region, -1)
        conf = res['anatomical_confidence']
        probs = dist['probability_distribution']
        is_correct = pred_region.lower() == gt.lower()
        if is_correct:
            correct_count += 1

        print(f"[{idx+1:02d}/{total_count}] File: {fname[:32]:<32} | GT: {gt:<10} | Pred: {pred_region:<10} | Conf: {conf*100:5.1f}% | Correct: {is_correct}")

        rec = {
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
        records.append(rec)

    df_out = pd.DataFrame(records)
    out_csv = 'reports/controlled_offline_test_after.csv'
    df_out.to_csv(out_csv, index=False)

    acc = correct_count / len(records) * 100.0 if records else 0.0
    print("\n" + "="*80)
    print(f"AFTER EVALUATION SUMMARY: {correct_count}/{len(records)} correct ({acc:.1f}% accuracy)")
    print(f"Saved complete results to: {out_csv}")
    print("="*80 + "\n")

if __name__ == '__main__':
    main()
