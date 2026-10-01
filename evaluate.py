"""
Untouched Test Set Evaluation Script
====================================
Loads best_model.pt and performs rigorous evaluation on the untouched test set.

Generates:
- metrics.json
- classification_report.csv
- confusion_matrix.png
"""

import os
import sys
import argparse
import json
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

from src.dataset import BoneFractDataset, get_transforms
from src.models import MultiTaskModel
from src.metrics import compute_all_metrics, plot_confusion_matrices, export_classification_report_csv

def evaluate_test_set(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Evaluating on device: {device}')
    
    if not os.path.exists(args.checkpoint):
        print(f"Error: Model checkpoint {args.checkpoint} not found.")
        sys.exit(1)
        
    print(f"Loading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    region_to_idx = checkpoint['region_to_idx']
    idx_to_region = {idx: reg for reg, idx in region_to_idx.items()}
    backbone = checkpoint.get('backbone', 'resnet50')
    num_regions = len(region_to_idx)
    
    print(f"Backbone: {backbone} | Number of anatomical regions: {num_regions}")
    print(f"Regions: {list(region_to_idx.keys())}")
    
    # Instantiate and load model
    model = MultiTaskModel(backbone=backbone, num_regions=num_regions, pretrained=False)
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    model.eval()
    
    # Load untouched test set
    print(f"Loading test set from {args.test_csv}...")
    test_df = pd.read_csv(args.test_csv)
    
    if args.max_test_samples and args.max_test_samples < len(test_df):
        print(f"Subsampling {args.max_test_samples} stratified test samples...")
        from sklearn.model_selection import train_test_split
        strat_test = test_df['anatomical_region'] + '_' + test_df['fracture_label']
        test_df, _ = train_test_split(test_df, train_size=args.max_test_samples, stratify=strat_test, random_state=42)
        test_df = test_df.reset_index(drop=True)
        
    print(f"Test samples for evaluation: {len(test_df):,}")
    
    test_dataset = BoneFractDataset(test_df, region_to_idx=region_to_idx, transform=get_transforms('test'))
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    all_region_targets = []
    all_region_preds = []
    all_region_probs = []
    all_fracture_targets = []
    all_fracture_preds = []
    all_fracture_probs = []
    
    print("Running inference over test set...")
    with torch.no_grad():
        for batch in test_loader:
            images = batch['image'].to(device)
            region_targets = batch['region_target'].to(device)
            fracture_targets = batch['fracture_target'].to(device)
            
            region_logits, fracture_logits = model(images)
            
            probs_r = torch.softmax(region_logits, dim=-1).cpu().numpy()
            preds_r = np.argmax(probs_r, axis=-1)
            
            probs_f = torch.sigmoid(fracture_logits).cpu().numpy()
            preds_f = (probs_f >= 0.5).astype(int)
            
            all_region_targets.extend(region_targets.cpu().numpy())
            all_region_preds.extend(preds_r)
            all_region_probs.extend(probs_r)
            all_fracture_targets.extend(fracture_targets.cpu().numpy())
            all_fracture_preds.extend(preds_f)
            all_fracture_probs.extend(probs_f)
            
    # Compute all metrics
    print("Computing metrics...")
    metrics = compute_all_metrics(
        all_region_targets, all_region_preds, all_region_probs,
        all_fracture_targets, all_fracture_preds, all_fracture_probs,
        idx_to_region
    )
    
    # Save metrics.json
    with open(args.output_metrics, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Saved test evaluation metrics to {args.output_metrics}")
    
    # Save classification_report.csv
    export_classification_report_csv(metrics, save_path=args.output_report)
    
    # Plot and save confusion_matrix.png
    plot_confusion_matrices(metrics, save_path=args.output_cm)
    
    # Print formatted terminal summary
    fm = metrics['fracture_detection']
    rm = metrics['anatomical_classification']
    
    print("\n" + "="*60)
    print("TEST SET EVALUATION SUMMARY REPORT")
    print("="*60)
    print(f"FRACTURE DETECTION:")
    print(f"  Accuracy:            {fm['accuracy']*100:.2f}%")
    print(f"  Precision:           {fm['precision']*100:.2f}%")
    print(f"  Recall (Sensitivity):{fm['recall_sensitivity']*100:.2f}%")
    print(f"  Specificity:         {fm['specificity']*100:.2f}%")
    print(f"  F1-Score:            {fm['f1_score']:.4f}")
    print(f"  ROC-AUC:             {fm['roc_auc']:.4f}")
    print(f"  PR-AUC:              {fm['pr_auc']:.4f}")
    print(f"  Confusion Matrix:    TN={fm['counts']['TN']}, FP={fm['counts']['FP']}, FN={fm['counts']['FN']}, TP={fm['counts']['TP']}")
    
    print(f"\nANATOMICAL REGION CLASSIFICATION:")
    print(f"  Overall Accuracy:    {rm['accuracy']*100:.2f}%")
    print(f"  Macro Precision:     {rm['macro_precision']*100:.2f}%")
    print(f"  Macro Recall:        {rm['macro_recall']*100:.2f}%")
    print(f"  Macro F1:            {rm['macro_f1']:.4f}")
    
    print(f"\nPER-REGION BREAKDOWN:")
    for reg, st in rm['per_class'].items():
        print(f"  {reg:<12} | Precision: {st['precision']*100:5.1f}% | Recall: {st['recall']*100:5.1f}% | F1: {st['f1_score']:.3f} | Support: {st['support']}")
        
    print(f"\nFRACTURE DETECTION PERFORMANCE BY ANATOMICAL REGION:")
    for reg, st in metrics['fracture_by_anatomical_region'].items():
        auc_str = f"{st['roc_auc']:.3f}" if st['roc_auc'] >= 0 else "N/A"
        print(f"  {reg:<12} | Acc: {st['accuracy']*100:5.1f}% | Rec: {st['recall']*100:5.1f}% | F1: {st['f1_score']:.3f} | AUC: {auc_str} (N={st['count']})")
    print("="*60 + "\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate Multi-Task Model on Untouched Test Set')
    parser.add_argument('--checkpoint', type=str, default='best_model.pt')
    parser.add_argument('--test_csv', type=str, default='test.csv')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_test_samples', type=int, default=None)
    parser.add_argument('--output_metrics', type=str, default='metrics.json')
    parser.add_argument('--output_cm', type=str, default='confusion_matrix.png')
    parser.add_argument('--output_report', type=str, default='classification_report.csv')
    args = parser.parse_args()
    
    evaluate_test_set(args)
