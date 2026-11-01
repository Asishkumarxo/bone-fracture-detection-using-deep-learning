"""
Experiment 4: Test Set Evaluation & Comparison with Experiment 3
================================================================
Evaluates models/checkpoints/exp4_balanced_resnet50_448.pt on the untouched
test set (4,778 samples) and generates direct side-by-side comparison with
Experiment 3 (exp3_resnet50_448.pt).

Outputs:
- reports/exp4_test_predictions.csv
- reports/exp4_evaluation_report.json
- reports/exp4_vs_exp3_comparison.md
"""

import os
import sys
import json
import time
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
from src.models import MultiTaskModel

def resize_and_pad(img, target_size=(448, 448), fill_color=0):
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
    def __init__(self, df, path_index, region_to_idx, target_size=(448, 448)):
        self.df = df.reset_index(drop=True)
        self.path_index = path_index
        self.region_to_idx = region_to_idx
        self.target_size = target_size
        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
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

def evaluate_exp4():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 80, flush=True)
    print("EXPERIMENT 4: TEST SET EVALUATION & COMPARISON WITH EXPERIMENT 3", flush=True)
    print("=" * 80, flush=True)
    print(f"Device: {device}", flush=True)

    ckpt_path = 'models/checkpoints/exp4_balanced_resnet50_448.pt'
    if not os.path.exists(ckpt_path):
        print(f"Error: Checkpoint {ckpt_path} not found!", flush=True)
        sys.exit(1)

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
    target_size = (448, 448)

    model = MultiTaskModel(backbone='resnet50', num_regions=num_regions, pretrained=False)
    model.load_state_dict(ckpt['state_dict'])
    model.to(device)
    model.eval()

    ds = EvalDataset(test_df, path_index, region_to_idx, target_size=target_size)
    loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=0)

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
                    'orig_height': int(batch['orig_height'][i])
                })

    eval_time = time.time() - t0
    print(f"Inference complete in {eval_time:.1f}s ({len(ds)/eval_time:.1f} img/s)", flush=True)

    df_results = pd.DataFrame(results)
    df_results.to_csv('reports/exp4_test_predictions.csv', index=False)
    print("Saved reports/exp4_test_predictions.csv", flush=True)

    # Compute Overall Binary Fracture Metrics
    y_true = df_results['true_fracture'].values
    y_pred = df_results['predicted_fracture'].values
    y_prob = df_results['fracture_prob'].values

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    spec_tn, spec_fp, spec_fn, spec_tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    spec = spec_tn / (spec_tn + spec_fp) if (spec_tn + spec_fp) > 0 else 0.0
    f1 = f1_score(y_true, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_true, y_prob)
    prec_curve, rec_curve, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(rec_curve, prec_curve)
    reg_acc = df_results['region_correct'].mean()

    # Per-Region Stats
    region_stats = {}
    for r in sorted(df_results['anatomical_region'].unique()):
        sub = df_results[df_results['anatomical_region'] == r]
        s_true = sub['true_fracture'].values
        s_pred = sub['predicted_fracture'].values
        s_prob = sub['fracture_prob'].values
        s_tp = int(np.sum((s_true == 1) & (s_pred == 1)))
        s_fn = int(np.sum((s_true == 1) & (s_pred == 0)))
        s_tn = int(np.sum((s_true == 0) & (s_pred == 0)))
        s_fp = int(np.sum((s_true == 0) & (s_pred == 1)))
        s_rec = float(s_tp / (s_tp + s_fn)) if (s_tp + s_fn) > 0 else 0.0
        s_spec = float(s_tn / (s_tn + s_fp)) if (s_tn + s_fp) > 0 else 0.0
        s_prec = float(s_tp / (s_tp + s_fp)) if (s_tp + s_fp) > 0 else 0.0
        s_f1 = float(2 * s_prec * s_rec / (s_prec + s_rec)) if (s_prec + s_rec) > 0 else 0.0
        region_stats[r] = {
            'count': len(sub),
            'positive_count': int(np.sum(s_true == 1)),
            'negative_count': int(np.sum(s_true == 0)),
            'tp': s_tp,
            'fn': s_fn,
            'tn': s_tn,
            'fp': s_fp,
            'recall': s_rec,
            'specificity': s_spec,
            'precision': s_prec,
            'f1': s_f1
        }

    # Resolution Breakdown
    df_results['is_102'] = (df_results['orig_width'] == 102) & (df_results['orig_height'] == 102)
    res_stats = {}
    for gname, gdf in [('102x102', df_results[df_results['is_102']]), ('higher_resolution', df_results[~df_results['is_102']])]:
        g_true = gdf['true_fracture'].values
        g_pred = gdf['predicted_fracture'].values
        g_tp = int(np.sum((g_true == 1) & (g_pred == 1)))
        g_fn = int(np.sum((g_true == 1) & (g_pred == 0)))
        g_tn = int(np.sum((g_true == 0) & (g_pred == 0)))
        g_fp = int(np.sum((g_true == 0) & (g_pred == 1)))
        g_rec = float(g_tp / (g_tp + g_fn)) if (g_tp + g_fn) > 0 else 0.0
        g_spec = float(g_tn / (g_tn + g_fp)) if (g_tn + g_fp) > 0 else 0.0
        res_stats[gname] = {
            'count': len(gdf),
            'positive_count': int(np.sum(g_true == 1)),
            'negative_count': int(np.sum(g_true == 0)),
            'tp': g_tp,
            'fn': g_fn,
            'tn': g_tn,
            'fp': g_fp,
            'recall': g_rec,
            'specificity': g_spec
        }

    output_data = {
        'model': 'exp4_balanced_resnet50_448.pt',
        'input_resolution': [448, 448],
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
            'fn': int(spec_fn)
        },
        'region_stats': region_stats,
        'resolution_stats': res_stats
    }

    with open('reports/exp4_evaluation_report.json', 'w') as f:
        json.dump(output_data, f, indent=2)
    print("Saved reports/exp4_evaluation_report.json", flush=True)

    # Load Exp3 report for direct comparison
    with open('reports/exp3_evaluation_report.json') as f:
        exp3_data = json.load(f)

    # Generate Comparison Markdown
    e3_m = exp3_data['metrics']
    e4_m = output_data['metrics']

    def diff(val4, val3):
        d = (val4 - val3) * 100
        sign = "+" if d >= 0 else ""
        return f"{sign}{d:.2f}%"

    md = []
    md.append("# Experiment 4 vs Experiment 3: Comparative Evaluation Report\n")
    md.append("**Experiment 4:** Balanced Region-Aware ResNet-50 (`exp4_balanced_resnet50_448.pt`)  ")
    md.append("**Experiment 3:** Baseline ResNet-50 at 448x448 (`exp3_resnet50_448.pt`)  ")
    md.append("**Evaluation Scope:** 100% Untouched Test Split ($N=4,778$)  \n")
    md.append("---\n")
    md.append("## 1. Overall Binary Fracture Classification Performance\n")
    md.append("| Metric | Experiment 3 (Exp3) | Experiment 4 (Exp4) | Absolute Difference (Exp4 - Exp3) |")
    md.append("| :--- | :---: | :---: | :---: |")
    md.append(f"| **Accuracy** | {e3_m['accuracy']*100:.2f}% | {e4_m['accuracy']*100:.2f}% | **{diff(e4_m['accuracy'], e3_m['accuracy'])}** |")
    md.append(f"| **Precision** | {e3_m['precision']*100:.2f}% | {e4_m['precision']*100:.2f}% | **{diff(e4_m['precision'], e3_m['precision'])}** |")
    md.append(f"| **Recall (Sensitivity)** | {e3_m['recall']*100:.2f}% | {e4_m['recall']*100:.2f}% | **{diff(e4_m['recall'], e3_m['recall'])}** |")
    md.append(f"| **Specificity** | {e3_m['specificity']*100:.2f}% | {e4_m['specificity']*100:.2f}% | **{diff(e4_m['specificity'], e3_m['specificity'])}** |")
    md.append(f"| **F1-Score** | {e3_m['f1']*100:.2f}% | {e4_m['f1']*100:.2f}% | **{diff(e4_m['f1'], e3_m['f1'])}** |")
    md.append(f"| **ROC-AUC** | {e3_m['roc_auc']:.4f} | {e4_m['roc_auc']:.4f} | **{e4_m['roc_auc'] - e3_m['roc_auc']:+.4f}** |")
    md.append(f"| **PR-AUC** | {e3_m['pr_auc']:.4f} | {e4_m['pr_auc']:.4f} | **{e4_m['pr_auc'] - e3_m['pr_auc']:+.4f}** |")
    md.append(f"| **True Positives (TP)** | {e3_m['tp']} | {e4_m['tp']} | {e4_m['tp'] - e3_m['tp']:+d} |")
    md.append(f"| **False Negatives (FN)** | {e3_m['fn']} | {e4_m['fn']} | {e4_m['fn'] - e3_m['fn']:+d} |")
    md.append(f"| **False Positives (FP)** | {e3_m['fp']} | {e4_m['fp']} | {e4_m['fp'] - e3_m['fp']:+d} |\n")

    md.append("---\n")
    md.append("## 2. Per-Region Fracture Recall Comparison\n")
    md.append("| Anatomical Region | Test Positives | Exp3 Recall | Exp4 Recall | Absolute Recall Change |")
    md.append("| :--- | :---: | :---: | :---: | :---: |")
    for r in sorted(region_stats.keys()):
        r3 = exp3_data['region_stats'][r]
        r4 = output_data['region_stats'][r]
        md.append(f"| **{r}** | {r4['positive_count']} | {r3['recall']*100:.2f}% | {r4['recall']*100:.2f}% | **{diff(r4['recall'], r3['recall'])}** |")

    md.append("\n---\n")
    md.append("## 3. Performance Stratified by Native Resolution\n")
    md.append("| Resolution Tier | Test Images | Test Positives | Exp3 Recall | Exp4 Recall | Recall Change |")
    md.append("| :--- | :---: | :---: | :---: | :---: | :---: |")
    for res_tier in ['102x102', 'higher_resolution']:
        res3 = exp3_data['resolution_stats'][res_tier]
        res4 = output_data['resolution_stats'][res_tier]
        md.append(f"| **{res_tier}** | {res4['count']} | {res4['positive_count']} | {res3['recall']*100:.2f}% | {res4['recall']*100:.2f}% | **{diff(res4['recall'], res3['recall'])}** |")

    md.append("\n---\n")
    md.append("## 4. Key Takeaways & Clinical Interpretation\n")
    md.append("Detailed analysis of improvements, trade-offs, and consistency across regions.\n")

    report_content = "\n".join(md)
    with open('reports/exp4_vs_exp3_comparison.md', 'w') as f:
        f.write(report_content)
    print("Saved reports/exp4_vs_exp3_comparison.md", flush=True)

if __name__ == '__main__':
    evaluate_exp4()
