import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix, brier_score_loss
)

def evaluate_test_predictions():
    df = pd.read_csv('test_predictions.csv')
    print(f"Total test rows: {len(df)}")
    print(df.head())

    y_true = df['true_fracture'].astype(int).values
    y_pred = df['predicted_fracture'].astype(int).values
    y_prob = df['fracture_confidence'].values

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)
    brier = brier_score_loss(y_true, y_prob)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    spec = tn / (tn + fp)
    fpr = fp / (tn + fp)
    fnr = fn / (fn + tp)

    print("\n" + "="*60)
    print("PHASE 5: ORIGINAL PROJECT TEST SET EVALUATION")
    print("="*60)
    print(f"Accuracy:            {acc:.4f} ({acc*100:.2f}%)")
    print(f"Precision:           {prec:.4f} ({prec*100:.2f}%)")
    print(f"Recall (Sensitivity): {rec:.4f} ({rec*100:.2f}%)")
    print(f"Specificity:         {spec:.4f} ({spec*100:.2f}%)")
    print(f"F1 Score:            {f1:.4f}")
    print(f"ROC-AUC:             {roc_auc:.4f}")
    print(f"PR-AUC:              {pr_auc:.4f}")
    print(f"Brier Score:         {brier:.4f}")
    print(f"False Positive Rate: {fpr:.4f}")
    print(f"False Negative Rate: {fnr:.4f}")
    print(f"Confusion Matrix:    TN={tn}, FP={fp}, FN={fn}, TP={tp}")

    # Probability percentiles (Phase 6)
    print("\n" + "="*60)
    print("PHASE 6: PROBABILITY DISTRIBUTION (TEST SET)")
    print("="*60)
    frac_probs = y_prob[y_true == 1]
    norm_probs = y_prob[y_true == 0]

    for name, p_arr in [('TRUE FRACTURE (Positive)', frac_probs), ('TRUE NORMAL (Negative)', norm_probs)]:
        print(f"\n--- {name} (N={len(p_arr)}) ---")
        print(f"  Min:    {np.min(p_arr):.4f}")
        print(f"  25th %: {np.percentile(p_arr, 25):.4f}")
        print(f"  Median: {np.median(p_arr):.4f}")
        print(f"  75th %: {np.percentile(p_arr, 75):.4f}")
        print(f"  Max:    {np.max(p_arr):.4f}")
        print(f"  Mean:   {np.mean(p_arr):.4f} +/- {np.std(p_arr):.4f}")

    # Per-region breakdown
    print("\n" + "="*60)
    print("FRACTURE PERFORMANCE BY ANATOMICAL REGION")
    print("="*60)
    for reg, grp in df.groupby('true_region'):
        r_yt = grp['true_fracture'].astype(int).values
        r_yp = grp['predicted_fracture'].astype(int).values
        r_prob = grp['fracture_confidence'].values
        r_rec = recall_score(r_yt, r_yp) if sum(r_yt) > 0 else 0.0
        r_prec = precision_score(r_yt, r_yp) if sum(r_yp) > 0 else 0.0
        r_fn = ((r_yt == 1) & (r_yp == 0)).sum()
        r_tot_pos = sum(r_yt)
        print(f"{reg:<12} | Total: {len(grp):4d} | Pos: {r_tot_pos:3d} | Recall: {r_rec*100:5.1f}% | Precision: {r_prec*100:5.1f}% | FN: {r_fn:3d}/{r_tot_pos}")

if __name__ == '__main__':
    evaluate_test_predictions()
