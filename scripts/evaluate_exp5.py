"""
Experiment 5: Held-Out Test Set Evaluation & Analysis
======================================================
Evaluates models/checkpoints/exp5_inception_v3_multitask.pt on the untouched
test set (4,778 samples) and computes:
- Overall Binary Fracture Metrics (Accuracy, Precision, Recall, Specificity, F1, ROC-AUC, PR-AUC, TP, TN, FP, FN)
- Anatomical Region Classification Accuracy (7 classes)
- Per-Region Fracture Recall, Precision, Specificity, F1
- Disparity between strongest and weakest regions
- Resolution-stratified analysis (102x102 thumbnails vs High-Res >=300x300)

Outputs:
- reports/exp5_test_predictions.csv
- reports/exp5_evaluation_report.json
- reports/exp5_per_region_metrics.csv
- reports/exp5_resolution_analysis.csv
"""

import os
import sys
import json
import time
import hashlib
import torch
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, precision_recall_curve, auc, confusion_matrix
)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.models_exp5 import MultiTaskInceptionV3

def compute_sha256(filepath):
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def resize_and_pad(img, target_size=(299, 299), fill_color=0):
    target_w, target_h = target_size
    orig_w, orig_h = img.size
    if orig_w == target_w and orig_h == target_h:
        return img
    ratio = min(target_w / orig_w, target_h / orig_h)
    new_w = max(1, int(round(orig_w * ratio)))
    new_h = max(1, int(round(orig_h * ratio)))
    resized_img = img.resize((new_w, new_h), Image.Resampling.BILINEAR)
    padded_img = Image.new('L', (target_w, target_h), color=fill_color)
    pad_x = (target_w - new_w) // 2
    pad_y = (target_h - new_h) // 2
    padded_img.paste(resized_img, (pad_x, pad_y))
    return padded_img

class EvalDataset(Dataset):
    def __init__(self, df, path_index, region_to_idx, target_size=(299, 299)):
        self.df = df.reset_index(drop=True)
        self.path_index = path_index
        self.region_to_idx = region_to_idx
        self.target_size = target_size
        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.fracture_to_idx = {'Negative': 0, 'Positive': 1}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_id = row['image_id']
        path = self.path_index[img_id]
        with Image.open(path) as img:
            orig_size = img.size
            gray = img.convert('L')
            padded = resize_and_pad(gray, self.target_size)
            rgb = padded.convert('RGB')
        tensor = self.transform(rgb)
        return {
            'image': tensor,
            'region_target': torch.tensor(self.region_to_idx[row['anatomical_region']], dtype=torch.long),
            'fracture_target': torch.tensor(self.fracture_to_idx[row['fracture_label']], dtype=torch.float32),
            'image_id': img_id,
            'anatomical_region': row['anatomical_region'],
            'fracture_label': row['fracture_label'],
            'orig_width': orig_size[0],
            'orig_height': orig_size[1]
        }

def evaluate_exp5():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 80, flush=True)
    print("EXPERIMENT 5: TEST SET EVALUATION (INCEPTION-v3)", flush=True)
    print("=" * 80, flush=True)
    print(f"Device: {device}", flush=True)

    ckpt_path = 'models/checkpoints/exp5_inception_v3_multitask.pt'
    if not os.path.exists(ckpt_path):
        print(f"Error: Checkpoint {ckpt_path} not found!", flush=True)
        sys.exit(1)

    ckpt_hash = compute_sha256(ckpt_path)
    print(f"Evaluating Checkpoint: {ckpt_path} (SHA-256: {ckpt_hash})", flush=True)

    # Load path index
    with open('data/bonefract_path_index.json') as f:
        old_idx = json.load(f)
    dataset_root = r"C:\Users\Donipudi Asish Kumar\Desktop\BoneFract A Bone Fracture Dataset"
    path_index = {}
    for img_id, old_p in old_idx.items():
        norm = old_p.replace("\\", "/")
        parts = norm.split("/")
        sub_idx = -1
        for i, p in enumerate(parts):
            if p in ("train", "valid", "test"):
                sub_idx = i
                break
        rel = os.path.join(*parts[sub_idx:])
        path_index[img_id] = os.path.join(dataset_root, rel)

    test_df = pd.read_csv('test.csv')
    print(f"Loaded untouched test.csv: {len(test_df):,} samples", flush=True)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    region_to_idx = ckpt['region_to_idx']
    idx_to_region = {v: k for k, v in region_to_idx.items()}
    num_regions = len(region_to_idx)
    target_size = (299, 299)

    model = MultiTaskInceptionV3(num_regions=num_regions, pretrained=False, use_aux=False)
    # Crucial: Inception-v3 was trained with torchvision's internal transform_input=True
    model.inception.transform_input = True
    model.load_state_dict(ckpt['state_dict'], strict=False)
    model.to(device)
    model.eval()

    ds = EvalDataset(test_df, path_index, region_to_idx, target_size=target_size)
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)

    print(f"Running inference on {len(ds):,} test samples...", flush=True)
    t0 = time.time()
    results = []
    with torch.no_grad():
        for batch in loader:
            imgs = batch['image'].to(device)
            r_logits, f_logits = model(imgs)
            r_probs = torch.softmax(r_logits, dim=-1).cpu().numpy()
            r_preds = np.argmax(r_probs, axis=-1)
            f_probs = torch.sigmoid(f_logits).cpu().numpy()
            f_preds = (f_probs >= 0.50).astype(int)

            for i in range(len(imgs)):
                results.append({
                    'image_id': batch['image_id'][i],
                    'anatomical_region': batch['anatomical_region'][i],
                    'true_fracture': int(batch['fracture_target'][i].item()),
                    'true_fracture_label': batch['fracture_label'][i],
                    'predicted_region': idx_to_region[r_preds[i]],
                    'region_correct': bool(idx_to_region[r_preds[i]] == batch['anatomical_region'][i]),
                    'fracture_prob': float(f_probs[i]),
                    'predicted_fracture': int(f_preds[i]),
                    'predicted_fracture_label': 'Positive' if f_preds[i] == 1 else 'Negative',
                    'orig_width': int(batch['orig_width'][i]),
                    'orig_height': int(batch['orig_height'][i]),
                    'is_102x102': bool(int(batch['orig_width'][i]) == 102 and int(batch['orig_height'][i]) == 102)
                })

    eval_time = time.time() - t0
    print(f"Inference complete in {eval_time:.1f}s ({len(ds)/eval_time:.1f} img/s)", flush=True)

    df_results = pd.DataFrame(results)
    df_results.to_csv('reports/exp5_test_predictions.csv', index=False)
    print("Saved reports/exp5_test_predictions.csv", flush=True)

    # Compute Overall Binary Fracture Metrics
    y_true = df_results['true_fracture'].values
    y_pred = df_results['predicted_fracture'].values
    y_prob = df_results['fracture_prob'].values

    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    spec_tn, spec_fp, spec_fn, spec_tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    spec = float(spec_tn / (spec_tn + spec_fp)) if (spec_tn + spec_fp) > 0 else 0.0
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    roc_auc = float(roc_auc_score(y_true, y_prob))
    prec_curve, rec_curve, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = float(auc(rec_curve, prec_curve))
    reg_acc = float(df_results['region_correct'].mean())

    print("\n--- OVERALL BINARY FRACTURE TEST METRICS ---")
    print(f"Accuracy:    {acc*100:.2f}%")
    print(f"Precision:   {prec*100:.2f}%")
    print(f"Recall:      {rec*100:.2f}%")
    print(f"Specificity: {spec*100:.2f}%")
    print(f"F1-Score:    {f1*100:.2f}%")
    print(f"ROC-AUC:     {roc_auc:.4f}")
    print(f"PR-AUC:      {pr_auc:.4f}")
    print(f"Region Acc:  {reg_acc*100:.2f}%")
    print(f"Confusion Matrix: TP={spec_tp:,}, TN={spec_tn:,}, FP={spec_fp:,}, FN={spec_fn:,}")

    # Per-Region Stats
    region_rows = []
    region_stats = {}
    recalls_list = []

    for r in sorted(df_results['anatomical_region'].unique()):
        sub = df_results[df_results['anatomical_region'] == r]
        sub_y_true = sub['true_fracture'].values
        sub_y_pred = sub['predicted_fracture'].values
        sub_y_prob = sub['fracture_prob'].values

        r_tp = int(((sub_y_true == 1) & (sub_y_pred == 1)).sum())
        r_fn = int(((sub_y_true == 1) & (sub_y_pred == 0)).sum())
        r_tn = int(((sub_y_true == 0) & (sub_y_pred == 0)).sum())
        r_fp = int(((sub_y_true == 0) & (sub_y_pred == 1)).sum())

        r_recall = float(r_tp / (r_tp + r_fn)) if (r_tp + r_fn) > 0 else 0.0
        r_prec = float(r_tp / (r_tp + r_fp)) if (r_tp + r_fp) > 0 else 0.0
        r_spec = float(r_tn / (r_tn + r_fp)) if (r_tn + r_fp) > 0 else 0.0
        r_f1 = float(2 * r_prec * r_recall / (r_prec + r_recall)) if (r_prec + r_recall) > 0 else 0.0

        recalls_list.append(r_recall)
        region_stats[r] = {
            'count': int(len(sub)),
            'positive_count': int((sub_y_true == 1).sum()),
            'negative_count': int((sub_y_true == 0).sum()),
            'tp': r_tp,
            'fn': r_fn,
            'tn': r_tn,
            'fp': r_fp,
            'recall': r_recall,
            'specificity': r_spec,
            'precision': r_prec,
            'f1': r_f1
        }
        region_rows.append({
            'anatomical_region': r,
            'total_images': int(len(sub)),
            'positive_cases': int((sub_y_true == 1).sum()),
            'negative_cases': int((sub_y_true == 0).sum()),
            'tp': r_tp,
            'fn': r_fn,
            'tn': r_tn,
            'fp': r_fp,
            'recall': r_recall,
            'precision': r_prec,
            'specificity': r_spec,
            'f1_score': r_f1
        })

    df_region = pd.DataFrame(region_rows)
    df_region.to_csv('reports/exp5_per_region_metrics.csv', index=False)
    print("Saved reports/exp5_per_region_metrics.csv", flush=True)

    macro_recall = float(np.mean(recalls_list))
    min_recall = float(np.min(recalls_list))
    max_recall = float(np.max(recalls_list))
    recall_disparity = float(max_recall - min_recall)

    print(f"\nMacro Average Per-Region Recall: {macro_recall*100:.2f}%")
    print(f"Recall Disparity (Max - Min):    {recall_disparity*100:.2f}% (Min: {min_recall*100:.2f}%, Max: {max_recall*100:.2f}%)")

    # Resolution Stratification
    res_rows = []
    res_stats = {}
    for tier_name, is_102 in [("Thumbnail Tier (102x102)", True), ("High-Res Tier (>=300x300)", False)]:
        sub = df_results[df_results['is_102x102'] == is_102]
        sub_y_true = sub['true_fracture'].values
        sub_y_pred = sub['predicted_fracture'].values
        sub_y_prob = sub['fracture_prob'].values

        res_tp = int(((sub_y_true == 1) & (sub_y_pred == 1)).sum())
        res_fn = int(((sub_y_true == 1) & (sub_y_pred == 0)).sum())
        res_tn = int(((sub_y_true == 0) & (sub_y_pred == 0)).sum())
        res_fp = int(((sub_y_true == 0) & (sub_y_pred == 1)).sum())

        res_recall = float(res_tp / (res_tp + res_fn)) if (res_tp + res_fn) > 0 else 0.0
        res_prec = float(res_tp / (res_tp + res_fp)) if (res_tp + res_fp) > 0 else 0.0
        res_spec = float(res_tn / (res_tn + res_fp)) if (res_tn + res_fp) > 0 else 0.0
        res_f1 = float(2 * res_prec * res_recall / (res_prec + res_recall)) if (res_prec + res_recall) > 0 else 0.0
        res_acc = float(accuracy_score(sub_y_true, sub_y_pred))

        res_stats[tier_name] = {
            'total_samples': int(len(sub)),
            'positive_cases': int((sub_y_true == 1).sum()),
            'negative_cases': int((sub_y_true == 0).sum()),
            'tp': res_tp,
            'fn': res_fn,
            'tn': res_tn,
            'fp': res_fp,
            'accuracy': res_acc,
            'recall': res_recall,
            'precision': res_prec,
            'specificity': res_spec,
            'f1': res_f1
        }
        res_rows.append({
            'resolution_tier': tier_name,
            'total_samples': int(len(sub)),
            'positive_cases': int((sub_y_true == 1).sum()),
            'negative_cases': int((sub_y_true == 0).sum()),
            'tp': res_tp,
            'fn': res_fn,
            'tn': res_tn,
            'fp': res_fp,
            'accuracy': res_acc,
            'recall': res_recall,
            'precision': res_prec,
            'specificity': res_spec,
            'f1_score': res_f1
        })

    df_res = pd.DataFrame(res_rows)
    df_res.to_csv('reports/exp5_resolution_analysis.csv', index=False)
    print("Saved reports/exp5_resolution_analysis.csv", flush=True)

    # Save comprehensive JSON report
    report = {
        'model': 'exp5_inception_v3_multitask.pt',
        'checkpoint_sha256': ckpt_hash,
        'architecture': 'Inception-v3',
        'input_resolution': [299, 299],
        'metrics': {
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'specificity': spec,
            'f1': f1,
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'region_accuracy': reg_acc,
            'tp': int(spec_tp),
            'tn': int(spec_tn),
            'fp': int(spec_fp),
            'fn': int(spec_fn),
            'macro_region_recall': macro_recall,
            'recall_disparity': recall_disparity,
            'min_region_recall': min_recall,
            'max_region_recall': max_recall
        },
        'region_stats': region_stats,
        'resolution_stats': res_stats
    }

    with open('reports/exp5_evaluation_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    print("Saved reports/exp5_evaluation_report.json", flush=True)
    print("=" * 80)
    return report

if __name__ == '__main__':
    evaluate_exp5()
