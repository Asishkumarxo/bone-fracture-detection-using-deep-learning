import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, precision_recall_curve, auc, confusion_matrix,
    classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns

def compute_all_metrics(all_region_targets, all_region_preds, all_region_probs,
                        all_fracture_targets, all_fracture_preds, all_fracture_probs,
                        idx_to_region, df_metadata=None):
    """
    Computes all specified evaluation metrics for both tasks,
    including stratified per-region fracture performance.
    """
    # 1. Fracture Detection Metrics
    y_true_f = np.array(all_fracture_targets)
    y_pred_f = np.array(all_fracture_preds)
    y_prob_f = np.array(all_fracture_probs)
    
    cm_fracture = confusion_matrix(y_true_f, y_pred_f, labels=[0, 1])
    # Extract TN, FP, FN, TP
    tn, fp, fn, tp = cm_fracture.ravel() if cm_fracture.shape == (2, 2) else (0, 0, 0, 0)
    
    acc_f = accuracy_score(y_true_f, y_pred_f)
    prec_f = precision_score(y_true_f, y_pred_f, zero_division=0)
    rec_f = recall_score(y_true_f, y_pred_f, zero_division=0) # Sensitivity
    spec_f = tn / (tn + fp) if (tn + fp) > 0 else 0.0 # Specificity
    f1_f = f1_score(y_true_f, y_pred_f, zero_division=0)
    
    try:
        roc_auc_f = roc_auc_score(y_true_f, y_prob_f)
    except Exception:
        roc_auc_f = 0.5
        
    try:
        precision_curve, recall_curve, _ = precision_recall_curve(y_true_f, y_prob_f)
        pr_auc_f = auc(recall_curve, precision_curve)
    except Exception:
        pr_auc_f = 0.0

    fracture_metrics = {
        'accuracy': float(acc_f),
        'precision': float(prec_f),
        'recall_sensitivity': float(rec_f),
        'specificity': float(spec_f),
        'f1_score': float(f1_f),
        'roc_auc': float(roc_auc_f),
        'pr_auc': float(pr_auc_f),
        'confusion_matrix': cm_fracture.tolist(),
        'counts': {'TN': int(tn), 'FP': int(fp), 'FN': int(fn), 'TP': int(tp)}
    }
    
    # 2. Anatomical Region Classification Metrics
    y_true_r = np.array(all_region_targets)
    y_pred_r = np.array(all_region_preds)
    region_labels = sorted(list(idx_to_region.keys()))
    region_names = [idx_to_region[i] for i in region_labels]
    
    acc_r = accuracy_score(y_true_r, y_pred_r)
    macro_prec_r = precision_score(y_true_r, y_pred_r, average='macro', zero_division=0)
    macro_rec_r = recall_score(y_true_r, y_pred_r, average='macro', zero_division=0)
    macro_f1_r = f1_score(y_true_r, y_pred_r, average='macro', zero_division=0)
    
    # Per-class metrics
    per_class_prec = precision_score(y_true_r, y_pred_r, labels=region_labels, average=None, zero_division=0)
    per_class_rec = recall_score(y_true_r, y_pred_r, labels=region_labels, average=None, zero_division=0)
    per_class_f1 = f1_score(y_true_r, y_pred_r, labels=region_labels, average=None, zero_division=0)
    
    cm_region = confusion_matrix(y_true_r, y_pred_r, labels=region_labels)
    
    per_class_dict = {}
    for idx, name in enumerate(region_names):
        per_class_dict[name] = {
            'precision': float(per_class_prec[idx]),
            'recall': float(per_class_rec[idx]),
            'f1_score': float(per_class_f1[idx]),
            'support': int((y_true_r == idx).sum())
        }
        
    region_metrics = {
        'accuracy': float(acc_r),
        'macro_precision': float(macro_prec_r),
        'macro_recall': float(macro_rec_r),
        'macro_f1': float(macro_f1_r),
        'per_class': per_class_dict,
        'confusion_matrix': cm_region.tolist(),
        'region_names': region_names
    }
    
    # 3. Stratified Fracture Performance by Anatomical Region
    per_region_fracture = {}
    for reg_idx, reg_name in idx_to_region.items():
        mask = (y_true_r == reg_idx)
        if mask.sum() > 0:
            reg_y_true = y_true_f[mask]
            reg_y_pred = y_pred_f[mask]
            reg_y_prob = y_prob_f[mask]
            
            reg_acc = accuracy_score(reg_y_true, reg_y_pred)
            reg_prec = precision_score(reg_y_true, reg_y_pred, zero_division=0)
            reg_rec = recall_score(reg_y_true, reg_y_pred, zero_division=0)
            reg_f1 = f1_score(reg_y_true, reg_y_pred, zero_division=0)
            
            try:
                reg_auc = roc_auc_score(reg_y_true, reg_y_prob) if len(np.unique(reg_y_true)) > 1 else None
            except Exception:
                reg_auc = None
                
            per_region_fracture[reg_name] = {
                'count': int(mask.sum()),
                'positive_count': int((reg_y_true == 1).sum()),
                'negative_count': int((reg_y_true == 0).sum()),
                'accuracy': float(reg_acc),
                'precision': float(reg_prec),
                'recall': float(reg_rec),
                'f1_score': float(reg_f1),
                'roc_auc': float(reg_auc) if reg_auc is not None else -1.0
            }
            
    return {
        'fracture_detection': fracture_metrics,
        'anatomical_classification': region_metrics,
        'fracture_by_anatomical_region': per_region_fracture
    }

def plot_confusion_matrices(metrics_dict, save_path='confusion_matrix.png'):
    """
    Renders high-resolution multi-panel confusion matrices and stratified metrics.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), dpi=300)
    
    # 1. Fracture Detection Confusion Matrix
    cm_f = np.array(metrics_dict['fracture_detection']['confusion_matrix'])
    sns.heatmap(cm_f, annot=True, fmt='d', cmap='Blues', cbar=False, ax=axes[0],
                xticklabels=['Negative (Normal)', 'Positive (Fracture)'],
                yticklabels=['Negative (Normal)', 'Positive (Fracture)'],
                annot_kws={'size': 14, 'weight': 'bold'})
    axes[0].set_title(f"Fracture Confusion Matrix\nAccuracy: {metrics_dict['fracture_detection']['accuracy']:.3f} | F1: {metrics_dict['fracture_detection']['f1_score']:.3f} | AUC: {metrics_dict['fracture_detection']['roc_auc']:.3f}",
                      fontsize=13, fontweight='bold', pad=12)
    axes[0].set_xlabel('Predicted Label', fontsize=11, fontweight='bold')
    axes[0].set_ylabel('True Ground Truth', fontsize=11, fontweight='bold')
    
    # 2. Anatomical Region Confusion Matrix
    cm_r = np.array(metrics_dict['anatomical_classification']['confusion_matrix'])
    region_names = metrics_dict['anatomical_classification']['region_names']
    sns.heatmap(cm_r, annot=True, fmt='d', cmap='Greens', cbar=False, ax=axes[1],
                xticklabels=region_names, yticklabels=region_names,
                annot_kws={'size': 10, 'weight': 'bold'})
    axes[1].set_title(f"Anatomical Region Confusion Matrix\nAccuracy: {metrics_dict['anatomical_classification']['accuracy']:.3f} | Macro F1: {metrics_dict['anatomical_classification']['macro_f1']:.3f}",
                      fontsize=13, fontweight='bold', pad=12)
    axes[1].set_xlabel('Predicted Region', fontsize=11, fontweight='bold')
    axes[1].set_ylabel('True Region', fontsize=11, fontweight='bold')
    axes[1].tick_params(axis='x', rotation=30)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f'Saved confusion matrix visualization to {save_path}')

def export_classification_report_csv(metrics_dict, save_path='classification_report.csv'):
    """
    Exports a structured CSV table containing per-class and summary metrics for both tasks.
    """
    rows = []
    # Fracture task
    fm = metrics_dict['fracture_detection']
    rows.append({
        'Task': 'Fracture Detection',
        'Category': 'Binary Overall',
        'Accuracy': fm['accuracy'],
        'Precision': fm['precision'],
        'Recall_Sensitivity': fm['recall_sensitivity'],
        'Specificity': fm['specificity'],
        'F1_Score': fm['f1_score'],
        'ROC_AUC': fm['roc_auc'],
        'PR_AUC': fm['pr_auc'],
        'Support': sum(sum(r) for r in fm['confusion_matrix'])
    })
    
    # Anatomical task
    am = metrics_dict['anatomical_classification']
    rows.append({
        'Task': 'Anatomical Classification',
        'Category': 'Macro Average',
        'Accuracy': am['accuracy'],
        'Precision': am['macro_precision'],
        'Recall_Sensitivity': am['macro_recall'],
        'Specificity': np.nan,
        'F1_Score': am['macro_f1'],
        'ROC_AUC': np.nan,
        'PR_AUC': np.nan,
        'Support': sum(sum(r) for r in am['confusion_matrix'])
    })
    
    for reg_name, stats in am['per_class'].items():
        rows.append({
            'Task': 'Anatomical Classification',
            'Category': f'Region: {reg_name}',
            'Accuracy': np.nan,
            'Precision': stats['precision'],
            'Recall_Sensitivity': stats['recall'],
            'Specificity': np.nan,
            'F1_Score': stats['f1_score'],
            'ROC_AUC': np.nan,
            'PR_AUC': np.nan,
            'Support': stats['support']
        })
        
    # Fracture by Region
    for reg_name, stats in metrics_dict['fracture_by_anatomical_region'].items():
        rows.append({
            'Task': 'Fracture by Region',
            'Category': f'Fracture in {reg_name}',
            'Accuracy': stats['accuracy'],
            'Precision': stats['precision'],
            'Recall_Sensitivity': stats['recall'],
            'Specificity': np.nan,
            'F1_Score': stats['f1_score'],
            'ROC_AUC': stats['roc_auc'] if stats['roc_auc'] >= 0 else np.nan,
            'PR_AUC': np.nan,
            'Support': stats['count']
        })
        
    df_rep = pd.DataFrame(rows)
    df_rep.to_csv(save_path, index=False)
    print(f'Saved structured classification report to {save_path}')
    return df_rep
