"""
Comprehensive Audit Script for Experiment 5 (Inception-v3 Multi-Task)
====================================================================
Executes the 15 evaluation audit checks specified in the prompt.
"""

import os
import sys
import json
import hashlib
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, precision_recall_curve, auc, confusion_matrix
)

def compute_sha256(filepath):
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def main():
    print("=" * 80)
    print("EXPERIMENT 5 EVALUATION AUDIT EXECUTION")
    print("=" * 80)

    # File paths
    test_csv_path = "test.csv"
    val_csv_path = "validation.csv"
    train_manifest_path = "reports/exp4_sample_manifest.csv"
    ckpt_path = "models/checkpoints/exp5_inception_v3_multitask.pt"
    exp5_pred_path = "reports/exp5_test_predictions.csv"
    exp5_rep_path = "reports/exp5_evaluation_report.json"
    exp5_reg_path = "reports/exp5_per_region_metrics.csv"
    exp5_res_path = "reports/exp5_resolution_analysis.csv"
    exp5_hist_path = "reports/exp5_training_history.csv"
    exp4_pred_path = "reports/exp4_test_predictions.csv"
    exp3_pred_path = "reports/exp3_test_predictions.csv"

    # Load data
    df_pred = pd.read_csv(exp5_pred_path)
    df_test = pd.read_csv(test_csv_path)

    # -------------------------------------------------------------
    # CHECK 14: Checkpoint Confirmation & SHA-256
    # -------------------------------------------------------------
    print("\n--- CHECK 14: Checkpoint Confirmation & SHA-256 ---")
    if os.path.exists(ckpt_path):
        ckpt_size = os.path.getsize(ckpt_path)
        ckpt_hash = compute_sha256(ckpt_path)
        print(f"Checkpoint Path: {ckpt_path}")
        print(f"Checkpoint File Size: {ckpt_size:,} bytes")
        print(f"Checkpoint SHA-256: {ckpt_hash}")
    else:
        print(f"ERROR: Checkpoint not found at {ckpt_path}")

    # -------------------------------------------------------------
    # CHECK 2 & 13: Test Sample Integrity, Counts & Overlap
    # -------------------------------------------------------------
    print("\n--- CHECK 2 & 13: Sample Counts, Integrity & Leakage Verification ---")
    n_pred = len(df_pred)
    n_test = len(df_test)
    n_pred_unique = df_pred['image_id'].nunique()
    n_test_unique = df_test['image_id'].nunique()
    print(f"Predictions Rows: {n_pred:,} | Unique image_ids: {n_pred_unique:,}")
    print(f"Test.csv Rows:    {n_test:,} | Unique image_ids: {n_test_unique:,}")

    # Check missing values
    missing_vals = df_pred.isnull().sum().to_dict()
    print(f"Missing Values in Predictions: {missing_vals}")

    # Row-by-row alignment with test.csv
    same_order = (df_pred['image_id'] == df_test['image_id']).all()
    print(f"Row ordering strictly identical to test.csv: {same_order}")

    # Ground truth label matching
    gt_map = {'Negative': 0, 'Positive': 1}
    test_gt_num = df_test['fracture_label'].map(gt_map).values
    gt_match = (df_pred['true_fracture'].values == test_gt_num).all()
    print(f"Ground truth binary labels match test.csv exactly: {gt_match}")
    reg_match = (df_pred['anatomical_region'].values == df_test['anatomical_region'].values).all()
    print(f"Ground truth anatomical regions match test.csv exactly: {reg_match}")

    # Class distribution in ground truth
    pos_count = int((df_pred['true_fracture'] == 1).sum())
    neg_count = int((df_pred['true_fracture'] == 0).sum())
    print(f"Ground Truth Distribution: Positive = {pos_count:,}, Negative = {neg_count:,} (Expected: 1,382 Pos, 3,396 Neg)")

    # Check for accidental contamination from train or validation sets
    df_train = pd.read_csv(train_manifest_path)
    df_val = pd.read_csv(val_csv_path)
    pred_ids = set(df_pred['image_id'])
    train_ids = set(df_train['image_id'])
    val_ids = set(df_val['image_id'])
    print(f"Overlap with Training Manifest:   {len(pred_ids.intersection(train_ids))} images")
    print(f"Overlap with Validation Split:    {len(pred_ids.intersection(val_ids))} images")

    # -------------------------------------------------------------
    # CHECK 3: Threshold Verification & Probability Agreement
    # -------------------------------------------------------------
    print("\n--- CHECK 3: Decision Threshold (0.50) & Probability Agreement ---")
    threshold = 0.50
    recomputed_preds = (df_pred['fracture_prob'] >= threshold).astype(int)
    disagreements = (recomputed_preds != df_pred['predicted_fracture']).sum()
    print(f"Disagreements between fracture_prob >= 0.50 and predicted_fracture: {disagreements}")
    label_disagreements = (
        (df_pred['predicted_fracture'] == 1) & (df_pred['predicted_fracture_label'] != 'Positive')
    ).sum() + (
        (df_pred['predicted_fracture'] == 0) & (df_pred['predicted_fracture_label'] != 'Negative')
    ).sum()
    print(f"Disagreements between predicted_fracture and predicted_fracture_label: {label_disagreements}")

    # -------------------------------------------------------------
    # CHECK 1 & 6: Recalculate Binary Fracture Metrics
    # -------------------------------------------------------------
    print("\n--- CHECK 1 & 6: Recalculate Confusion Matrix & Binary Metrics ---")
    y_true = df_pred['true_fracture'].values
    y_pred = df_pred['predicted_fracture'].values
    y_prob = df_pred['fracture_prob'].values

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = f1_score(y_true, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_true, y_prob)
    prec_c, rec_c, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(rec_c, prec_c)

    print(f"Confusion Matrix: TP = {tp:,}, FP = {fp:,}, TN = {tn:,}, FN = {fn:,}")
    print(f"Accuracy:    {acc*100:.4f}% ({acc:.6f})")
    print(f"Precision:   {prec*100:.4f}% ({prec:.6f})")
    print(f"Recall:      {rec*100:.4f}% ({rec:.6f})")
    print(f"Specificity: {spec*100:.4f}% ({spec:.6f})")
    print(f"F1-Score:    {f1*100:.4f}% ({f1:.6f})")
    print(f"ROC-AUC:     {roc_auc:.6f}")
    print(f"PR-AUC:      {pr_auc:.6f}")

    # Compare with reports/exp5_evaluation_report.json
    with open(exp5_rep_path, 'r') as f:
        rep_json = json.load(f)
    json_metrics = rep_json['metrics']
    print("\nComparison with reported metrics in exp5_evaluation_report.json:")
    for m_name, calc_val in [
        ('accuracy', acc), ('precision', prec), ('recall', rec), ('specificity', spec),
        ('f1', f1), ('roc_auc', roc_auc), ('pr_auc', pr_auc), ('tp', tp), ('tn', tn), ('fp', fp), ('fn', fn)
    ]:
        rep_val = json_metrics.get(m_name)
        match_str = "MATCH" if np.isclose(calc_val, rep_val, atol=1e-5) else f"MISMATCH (Reported: {rep_val})"
        print(f"  {m_name:12s}: Recalculated = {calc_val:<10.6f} | Reported = {rep_val} | {match_str}")

    # -------------------------------------------------------------
    # CHECK 4: Label Inversion / Probability Inversion Check
    # -------------------------------------------------------------
    print("\n--- CHECK 4: Label Inversion / Class Mapping Inversion Check ---")
    # If probabilities were inverted (1 - y_prob):
    inv_prob = 1.0 - y_prob
    inv_roc = roc_auc_score(y_true, inv_prob)
    inv_pred = (inv_prob >= 0.50).astype(int)
    inv_acc = accuracy_score(y_true, inv_pred)
    print(f"If probabilities were inverted (1 - prob): ROC-AUC = {inv_roc:.4f}, Accuracy = {inv_acc*100:.2f}%")
    # Point-biserial correlation
    corr = np.corrcoef(y_prob, y_true)[0, 1]
    print(f"Correlation between fracture_prob and true_fracture: {corr:.4f}")

    # -------------------------------------------------------------
    # CHECK 11: Probability Distributions by Actual Class
    # -------------------------------------------------------------
    print("\n--- CHECK 11: Fracture Probability Distributions by True Class ---")
    pos_probs = y_prob[y_true == 1]
    neg_probs = y_prob[y_true == 0]

    def summarize_dist(arr, name):
        qs = np.percentile(arr, [0, 5, 10, 25, 50, 75, 90, 95, 100])
        print(f"{name} (N = {len(arr):,}):")
        print(f"  Mean: {np.mean(arr):.6f} | Std: {np.std(arr):.6f}")
        print(f"  Min:  {qs[0]:.6f} | P5:  {qs[1]:.6f} | P10: {qs[2]:.6f}")
        print(f"  P25:  {qs[3]:.6f} | Median (P50): {qs[4]:.6f} | P75: {qs[5]:.6f}")
        print(f"  P90:  {qs[6]:.6f} | P95: {qs[7]:.6f} | Max: {qs[8]:.6f}")
        frac_above_50 = (arr >= 0.50).mean()
        frac_above_90 = (arr >= 0.90).mean()
        frac_above_99 = (arr >= 0.99).mean()
        print(f"  % >= 0.50: {frac_above_50*100:.2f}% | % >= 0.90: {frac_above_90*100:.2f}% | % >= 0.99: {frac_above_99*100:.2f}%")

    summarize_dist(pos_probs, "Actual POSITIVE Cases")
    summarize_dist(neg_probs, "Actual NEGATIVE Cases")

    # -------------------------------------------------------------
    # CHECK 7: Anatomical Region Classification & Confusion Matrix
    # -------------------------------------------------------------
    print("\n--- CHECK 7: Anatomical Region Classification & Confusion Matrix ---")
    regions = sorted(df_pred['anatomical_region'].unique())
    reg_true = df_pred['anatomical_region'].values
    reg_pred = df_pred['predicted_region'].values
    reg_acc = accuracy_score(reg_true, reg_pred)
    print(f"7-Class Anatomical Region Accuracy: {reg_acc*100:.4f}% ({reg_acc:.6f})")
    print(f"Random Guess Baseline (1/7):         {100/7:.2f}% (0.142857)")

    cm_reg = confusion_matrix(reg_true, reg_pred, labels=regions)
    df_cm_reg = pd.DataFrame(cm_reg, index=[f"True_{r}" for r in regions], columns=[f"Pred_{r}" for r in regions])
    print("\nAnatomical Region Confusion Matrix:")
    print(df_cm_reg.to_string())

    print("\nPredicted Region Counts:")
    print(df_pred['predicted_region'].value_counts())

    # -------------------------------------------------------------
    # CHECK 8: Per-Region Recall & Metrics
    # -------------------------------------------------------------
    print("\n--- CHECK 8: Reproduce Per-Region Fracture Recall ---")
    reg_rows = []
    for r in regions:
        sub = df_pred[df_pred['anatomical_region'] == r]
        s_yt = sub['true_fracture'].values
        s_yp = sub['predicted_fracture'].values
        r_tp = int(((s_yt == 1) & (s_yp == 1)).sum())
        r_fn = int(((s_yt == 1) & (s_yp == 0)).sum())
        r_tn = int(((s_yt == 0) & (s_yp == 0)).sum())
        r_fp = int(((s_yt == 0) & (s_yp == 1)).sum())
        r_rec = r_tp / (r_tp + r_fn) if (r_tp + r_fn) > 0 else 0.0
        r_prec = r_tp / (r_tp + r_fp) if (r_tp + r_fp) > 0 else 0.0
        r_spec = r_tn / (r_tn + r_fp) if (r_tn + r_fp) > 0 else 0.0
        r_f1 = 2 * r_prec * r_rec / (r_prec + r_rec) if (r_prec + r_rec) > 0 else 0.0
        reg_rows.append({
            'Region': r, 'Total': len(sub), 'Pos': (s_yt == 1).sum(), 'Neg': (s_yt == 0).sum(),
            'TP': r_tp, 'FN': r_fn, 'TN': r_tn, 'FP': r_fp,
            'Recall': r_rec, 'Precision': r_prec, 'Specificity': r_spec, 'F1': r_f1
        })
    df_reg_metrics = pd.DataFrame(reg_rows)
    print(df_reg_metrics.to_string(index=False))

    macro_recall = df_reg_metrics['Recall'].mean()
    min_rec = df_reg_metrics['Recall'].min()
    max_rec = df_reg_metrics['Recall'].max()
    print(f"\nMacro Average Recall: {macro_recall*100:.2f}%")
    print(f"Recall Disparity (Max - Min): {(max_rec - min_rec)*100:.2f}% (Min: {min_rec*100:.2f}%, Max: {max_rec*100:.2f}%)")

    # -------------------------------------------------------------
    # CHECK 9, 10, 12: Resolution Stratification & Aliasing Check
    # -------------------------------------------------------------
    print("\n--- CHECK 9, 10, 12: Native Resolution Stratification & Low-Res Aliasing ---")
    thumb_mask = (df_pred['is_102x102'] == True)
    high_mask = (df_pred['is_102x102'] == False)

    n_thumb = int(thumb_mask.sum())
    n_high = int(high_mask.sum())
    print(f"Low-Res (102x102 Thumbnail) Count: {n_thumb:,} (Expected: 3,461)")
    print(f"High-Res (>=300x300) Count:         {n_high:,} (Expected: 1,317)")

    # Thumbnail metrics
    thumb_yt = y_true[thumb_mask]
    thumb_yp = y_pred[thumb_mask]
    thumb_tn, thumb_fp, thumb_fn, thumb_tp = confusion_matrix(thumb_yt, thumb_yp, labels=[0, 1]).ravel()
    thumb_rec = thumb_tp / (thumb_tp + thumb_fn)
    thumb_spec = thumb_tn / (thumb_tn + thumb_fp)
    thumb_acc = accuracy_score(thumb_yt, thumb_yp)
    thumb_prec = precision_score(thumb_yt, thumb_yp, zero_division=0)
    thumb_f1 = f1_score(thumb_yt, thumb_yp, zero_division=0)
    thumb_pred_pos_rate = (thumb_yp == 1).mean()

    print(f"\nThumbnail Tier ($102\\times 102$):")
    print(f"  Total: {n_thumb:,} | Positives: {(thumb_yt == 1).sum():,} | Negatives: {(thumb_yt == 0).sum():,}")
    print(f"  TP = {thumb_tp:,}, FN = {thumb_fn:,}, TN = {thumb_tn:,}, FP = {thumb_fp:,}")
    print(f"  Accuracy:    {thumb_acc*100:.2f}%")
    print(f"  Recall:      {thumb_rec*100:.2f}% (Claimed: 98.42%)")
    print(f"  Precision:   {thumb_prec*100:.2f}%")
    print(f"  Specificity: {thumb_spec*100:.2f}%")
    print(f"  F1-Score:    {thumb_f1*100:.2f}%")
    print(f"  Overall Predicted Positive Rate: {thumb_pred_pos_rate*100:.2f}% ({(thumb_yp == 1).sum():,} of {n_thumb:,})")

    # High-Res metrics
    high_yt = y_true[high_mask]
    high_yp = y_pred[high_mask]
    high_tn, high_fp, high_fn, high_tp = confusion_matrix(high_yt, high_yp, labels=[0, 1]).ravel()
    high_rec = high_tp / (high_tp + high_fn)
    high_spec = high_tn / (high_tn + high_fp)
    high_acc = accuracy_score(high_yt, high_yp)
    high_prec = precision_score(high_yt, high_yp, zero_division=0)
    high_f1 = f1_score(high_yt, high_yp, zero_division=0)
    high_pred_pos_rate = (high_yp == 1).mean()

    print(f"\nHigh-Res Tier ($\\ge 300\\times 300$):")
    print(f"  Total: {n_high:,} | Positives: {(high_yt == 1).sum():,} | Negatives: {(high_yt == 0).sum():,}")
    print(f"  TP = {high_tp:,}, FN = {high_fn:,}, TN = {high_tn:,}, FP = {high_fp:,}")
    print(f"  Accuracy:    {high_acc*100:.2f}%")
    print(f"  Recall:      {high_rec*100:.2f}%")
    print(f"  Precision:   {high_prec*100:.2f}%")
    print(f"  Specificity: {high_spec*100:.2f}%")
    print(f"  F1-Score:    {high_f1*100:.2f}%")
    print(f"  Overall Predicted Positive Rate: {high_pred_pos_rate*100:.2f}% ({(high_yp == 1).sum():,} of {n_high:,})")

    # -------------------------------------------------------------
    # CHECK 15: Cross-Experiment Alignment (Exp 3, 4, 5)
    # -------------------------------------------------------------
    print("\n--- CHECK 15: Cross-Experiment Alignment (Exp 3, 4, 5) ---")
    df_exp4 = pd.read_csv(exp4_pred_path)
    df_exp3 = pd.read_csv(exp3_pred_path)

    print(f"Exp 3 rows: {len(df_exp3):,}, Exp 4 rows: {len(df_exp4):,}, Exp 5 rows: {len(df_pred):,}")
    exp3_ids = list(df_exp3['image_id'])
    exp4_ids = list(df_exp4['image_id'])
    exp5_ids = list(df_pred['image_id'])

    same_ids_3_5 = (exp3_ids == exp5_ids)
    same_ids_4_5 = (exp4_ids == exp5_ids)
    print(f"Identical image_id sequence between Exp 3 and Exp 5: {same_ids_3_5}")
    print(f"Identical image_id sequence between Exp 4 and Exp 5: {same_ids_4_5}")

    # Ground truth labels consistency
    gt3 = df_exp3['true_fracture'].values
    gt4 = df_exp4['true_fracture'].values
    gt5 = df_pred['true_fracture'].values
    same_gt_3_5 = (gt3 == gt5).all()
    same_gt_4_5 = (gt4 == gt5).all()
    print(f"Identical ground truth labels between Exp 3 and Exp 5: {same_gt_3_5}")
    print(f"Identical ground truth labels between Exp 4 and Exp 5: {same_gt_4_5}")

    print("\nCross-Experiment Summary Table:")
    summary_data = [
        {"Model": "Exp 3 (ResNet-50 448)", "Recall": recall_score(gt3, df_exp3['predicted_fracture']),
         "Specificity": confusion_matrix(gt3, df_exp3['predicted_fracture'])[0, 0] / (gt3 == 0).sum(),
         "Precision": precision_score(gt3, df_exp3['predicted_fracture']),
         "Accuracy": accuracy_score(gt3, df_exp3['predicted_fracture']),
         "F1": f1_score(gt3, df_exp3['predicted_fracture']),
         "TP": int(((gt3 == 1) & (df_exp3['predicted_fracture'] == 1)).sum()),
         "FP": int(((gt3 == 0) & (df_exp3['predicted_fracture'] == 1)).sum())},
        {"Model": "Exp 4 (Balanced ResNet-50 448)", "Recall": recall_score(gt4, df_exp4['predicted_fracture']),
         "Specificity": confusion_matrix(gt4, df_exp4['predicted_fracture'])[0, 0] / (gt4 == 0).sum(),
         "Precision": precision_score(gt4, df_exp4['predicted_fracture']),
         "Accuracy": accuracy_score(gt4, df_exp4['predicted_fracture']),
         "F1": f1_score(gt4, df_exp4['predicted_fracture']),
         "TP": int(((gt4 == 1) & (df_exp4['predicted_fracture'] == 1)).sum()),
         "FP": int(((gt4 == 0) & (df_exp4['predicted_fracture'] == 1)).sum())},
        {"Model": "Exp 5 (Inception-v3 299)", "Recall": rec, "Specificity": spec,
         "Precision": prec, "Accuracy": acc, "F1": f1, "TP": tp, "FP": fp}
    ]
    print(pd.DataFrame(summary_data).to_string(index=False))

    print("\nAudit checks completed successfully.")

if __name__ == '__main__':
    main()
