"""
Experiment 3: Full Test Set Evaluation & Direct Baseline Comparison
===================================================================
Evaluates models/checkpoints/exp3_resnet50_448.pt at 448x448 resolution
on the complete, untouched 4,778-image test set and performs comprehensive
comparisons against the reference baseline.
"""

import os
import sys
import time
import json
import torch
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
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

def evaluate_exp3():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("="*70, flush=True)
    print("EXPERIMENT 3: COMPREHENSIVE TEST SET EVALUATION", flush=True)
    print("="*70, flush=True)
    print(f"Device: {device}", flush=True)

    ckpt_path = 'models/checkpoints/exp3_resnet50_448.pt'
    if not os.path.exists(ckpt_path):
        print(f"Error: Checkpoint {ckpt_path} not found!", flush=True)
        sys.exit(1)

    with open('data/bonefract_path_index.json') as f:
        path_index = json.load(f)

    test_df = pd.read_csv('test.csv')
    print(f"Loaded untouched test.csv: {len(test_df):,} samples", flush=True)

    ckpt = torch.load(ckpt_path, map_location=device)
    region_to_idx = ckpt['region_to_idx']
    idx_to_region = {v: k for k, v in region_to_idx.items()}
    num_regions = len(region_to_idx)
    target_size = tuple(ckpt.get('target_size', (448, 448)))
    print(f"Loaded checkpoint from {ckpt_path} (Trained resolution: {target_size})", flush=True)

    model = MultiTaskModel(backbone='resnet50', num_regions=num_regions, pretrained=False)
    model.load_state_dict(ckpt['state_dict'])
    model.to(device)
    model.eval()

    ds = EvalDataset(test_df, path_index, region_to_idx=region_to_idx, target_size=target_size)
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)

    all_r_t, all_r_p, all_r_prob = [], [], []
    all_f_t, all_f_p, all_f_prob = [], [], []
    all_regions = []
    all_widths = []
    all_heights = []
    all_image_ids = []

    t0 = time.time()
    with torch.no_grad():
        for b in loader:
            imgs = b['image'].to(device)
            r_logits, f_logits = model(imgs)
            probs_r = torch.softmax(r_logits, dim=-1).cpu().numpy()
            probs_f = torch.sigmoid(f_logits).cpu().numpy().flatten()
            preds_r = np.argmax(probs_r, axis=-1)
            preds_f = (probs_f >= 0.50).astype(int)

            all_r_t.extend(b['region_target'].numpy())
            all_r_p.extend(preds_r)
            all_r_prob.extend(probs_r)
            all_f_t.extend(b['fracture_target'].numpy())
            all_f_p.extend(preds_f)
            all_f_prob.extend(probs_f)
            all_regions.extend(b['anatomical_region'])
            all_widths.extend(b['orig_width'].numpy())
            all_heights.extend(b['orig_height'].numpy())
            all_image_ids.extend(b['image_id'])

    elapsed = time.time() - t0
    print(f"Test inference completed in {elapsed:.1f}s ({len(test_df)/elapsed:.1f} img/s)", flush=True)

    y_true = np.array(all_f_t, dtype=int)
    y_pred = np.array(all_f_p, dtype=int)
    y_prob = np.array(all_f_prob, dtype=float)

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    accuracy = float(accuracy_score(y_true, y_pred))
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred))
    specificity = float(tn / (tn + fp))
    f1 = float(f1_score(y_true, y_pred))
    roc_auc = float(roc_auc_score(y_true, y_prob))
    pr_auc = float(average_precision_score(y_true, y_prob))
    fnr = float(fn / (fn + tp))

    # Save detailed test predictions CSV
    df_results = pd.DataFrame({
        'image_id': all_image_ids,
        'anatomical_region': all_regions,
        'predicted_region': [idx_to_region[p] for p in all_r_p],
        'orig_width': all_widths,
        'orig_height': all_heights,
        'true_fracture': y_true,
        'predicted_fracture': y_pred,
        'fracture_prob': y_prob
    })
    df_results.to_csv('reports/exp3_test_predictions.csv', index=False)
    print("Saved reports/exp3_test_predictions.csv", flush=True)

    # Region-wise breakdown
    region_stats = {}
    for r in sorted(df_results['anatomical_region'].unique()):
        sub = df_results[df_results['anatomical_region'] == r]
        sub_true = sub['true_fracture'].values
        sub_pred = sub['predicted_fracture'].values
        sub_prob = sub['fracture_prob'].values
        sub_tp = int(np.sum((sub_true == 1) & (sub_pred == 1)))
        sub_fn = int(np.sum((sub_true == 1) & (sub_pred == 0)))
        sub_tn = int(np.sum((sub_true == 0) & (sub_pred == 0)))
        sub_fp = int(np.sum((sub_true == 0) & (sub_pred == 1)))
        sub_rec = float(sub_tp / (sub_tp + sub_fn)) if (sub_tp + sub_fn) > 0 else 0.0
        sub_spec = float(sub_tn / (sub_tn + sub_fp)) if (sub_tn + sub_fp) > 0 else 0.0
        region_stats[r] = {
            'count': len(sub),
            'positive_count': int(np.sum(sub_true == 1)),
            'negative_count': int(np.sum(sub_true == 0)),
            'tp': sub_tp,
            'fn': sub_fn,
            'tn': sub_tn,
            'fp': sub_fp,
            'recall': sub_rec,
            'specificity': sub_spec
        }

    # Native resolution breakdown
    df_results['is_102'] = (df_results['orig_width'] == 102) & (df_results['orig_height'] == 102)
    res_groups = {
        '102x102': df_results[df_results['is_102']],
        'higher_resolution': df_results[~df_results['is_102']]
    }
    res_stats = {}
    for gname, gdf in res_groups.items():
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

    # External images evaluation
    external_images = [
        ('AdobeStock_594927383-1.jpeg', 'c:/Users/arbaz/Downloads/AdobeStock_594927383-1.jpeg', 'Positive (Forearm/Wrist fracture)'),
        ('wrist crack image.jpg', 'c:/Users/arbaz/Downloads/wrist crack image.jpg', 'Positive (Wrist fracture)'),
        ('2.jpg', 'c:/Users/arbaz/Downloads/2.jpg', 'Positive (Distal radius fracture)'),
        ('istockphoto-471457370-612x612.jpg', 'c:/Users/arbaz/Downloads/istockphoto-471457370-612x612.jpg', 'Negative (Normal pelvis)'),
        ('AdobeStock_200285274.webp', 'c:/Users/arbaz/Downloads/AdobeStock_200285274.webp', 'Negative (Normal wrist)')
    ]
    external_results = []
    for name, p, expected in external_images:
        if os.path.exists(p):
            with Image.open(p) as img:
                g = img.convert('L')
                pad = resize_and_pad(g, target_size)
                rgb = pad.convert('RGB')
            tensor = ds.transform(rgb).unsqueeze(0).to(device)
            with torch.no_grad():
                r_logits, f_logits = model(tensor)
                p_frac = float(torch.sigmoid(f_logits).cpu().item())
                r_idx = int(torch.argmax(r_logits, dim=-1).cpu().item())
            external_results.append({
                'name': name,
                'path': p,
                'expected': expected,
                'predicted_region': idx_to_region[r_idx],
                'fracture_prob': p_frac,
                'predicted_label': 'Positive' if p_frac >= 0.50 else 'Negative'
            })

    output_data = {
        'model': 'exp3_resnet50_448.pt',
        'input_resolution': list(target_size),
        'metrics': {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'specificity': specificity,
            'f1': f1,
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'tp': int(tp),
            'tn': int(tn),
            'fp': int(fp),
            'fn': int(fn),
            'fnr': fnr
        },
        'region_stats': region_stats,
        'resolution_stats': res_stats,
        'external_results': external_results
    }

    with open('reports/exp3_evaluation_report.json', 'w') as f:
        json.dump(output_data, f, indent=2)
    print("Saved reports/exp3_evaluation_report.json", flush=True)

    print("\n" + "="*70, flush=True)
    print("EXPERIMENT 3 RESULTS SUMMARY", flush=True)
    print("="*70, flush=True)
    print(f"Accuracy:            {accuracy*100:.2f}%", flush=True)
    print(f"Precision:           {precision*100:.2f}%", flush=True)
    print(f"Recall:              {recall*100:.2f}%", flush=True)
    print(f"Specificity:         {specificity*100:.2f}%", flush=True)
    print(f"F1-Score:            {f1:.4f}", flush=True)
    print(f"ROC-AUC:             {roc_auc:.4f}", flush=True)
    print(f"PR-AUC:              {pr_auc:.4f}", flush=True)
    print(f"Confusion Matrix:    TP={tp}, TN={tn}, FP={fp}, FN={fn}, FNR={fnr*100:.2f}%", flush=True)

    print("\nRegion Recall:", flush=True)
    for r, st in region_stats.items():
        print(f"  {r:<12}: Recall={st['recall']*100:5.2f}% ({st['tp']}/{st['tp']+st['fn']}), Spec={st['specificity']*100:5.2f}%", flush=True)

    print("\nNative Resolution Analysis:", flush=True)
    for g, st in res_stats.items():
        print(f"  {g:<18}: Count={st['count']}, Recall={st['recall']*100:5.2f}%, Spec={st['specificity']*100:5.2f}%, FN={st['fn']}", flush=True)

    print("\nExternal Images:", flush=True)
    for ext in external_results:
        print(f"  {ext['name']:<35} Prob={ext['fracture_prob']*100:5.2f}% -> {ext['predicted_label']} (Expected: {ext['expected']})", flush=True)

if __name__ == '__main__':
    evaluate_exp3()
