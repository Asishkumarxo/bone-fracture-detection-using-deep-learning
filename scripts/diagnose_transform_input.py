import torch
import os
import json
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, recall_score, precision_score, f1_score,
    roc_auc_score, confusion_matrix, precision_recall_curve, auc
)
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.evaluate_exp5 import EvalDataset
from src.models_exp5 import MultiTaskInceptionV3

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt = torch.load('models/checkpoints/exp5_inception_v3_multitask.pt', map_location=device, weights_only=False)

    mandatory_root = r"C:\Users\Donipudi Asish Kumar\Desktop\BoneFract A Bone Fracture Dataset"
    with open('data/bonefract_path_index.json') as f:
        old_idx = json.load(f)
    path_index = {}
    for img_id, old_p in old_idx.items():
        norm = old_p.replace(chr(92), '/')
        parts = norm.split('/')
        sub_idx = -1
        for i, p in enumerate(parts):
            if p in ('train', 'valid', 'test'):
                sub_idx = i
                break
        rel = os.path.join(*parts[sub_idx:])
        path_index[img_id] = os.path.join(mandatory_root, rel)

    region_to_idx = ckpt['region_to_idx']
    idx_to_region = {v: k for k, v in region_to_idx.items()}
    test_df = pd.read_csv('test.csv')
    test_dataset = EvalDataset(test_df, path_index, region_to_idx, target_size=(299, 299))
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    model = MultiTaskInceptionV3(num_regions=len(region_to_idx), pretrained=False, use_aux=False)
    model.inception.transform_input = True
    model.load_state_dict(ckpt['state_dict'], strict=False)
    model.to(device)
    model.eval()

    results = []
    with torch.no_grad():
        for b in test_loader:
            img = b['image'].to(device)
            r_logits, f_logits = model(img)
            r_preds = torch.argmax(r_logits, dim=-1).cpu().numpy()
            f_probs = torch.sigmoid(f_logits).cpu().numpy()
            f_preds = (f_probs >= 0.50).astype(int)
            for i in range(len(img)):
                results.append({
                    'image_id': b['image_id'][i],
                    'region': b['anatomical_region'][i],
                    'pred_region': idx_to_region[r_preds[i]],
                    'true_f': int(b['fracture_target'][i].item()),
                    'pred_f': int(f_preds[i]),
                    'prob_f': float(f_probs[i]),
                    'is_102': bool(int(b['orig_width'][i]) == 102 and int(b['orig_height'][i]) == 102)
                })

    df = pd.DataFrame(results)
    y_t = df['true_f'].values
    y_p = df['pred_f'].values
    y_prob = df['prob_f'].values

    tn, fp, fn, tp = confusion_matrix(y_t, y_p).ravel()
    prec_c, rec_c, _ = precision_recall_curve(y_t, y_prob)

    print("=" * 70)
    print("CORRECTED INCEPTION-V3 EVALUATION (WITH transform_input=True)")
    print("=" * 70)
    print(f"Confusion Matrix: TP={tp:,}, FP={fp:,}, TN={tn:,}, FN={fn:,}")
    print(f"Accuracy:    {accuracy_score(y_t, y_p)*100:.4f}%")
    print(f"Recall:      {recall_score(y_t, y_p)*100:.4f}%")
    print(f"Specificity: {tn/(tn+fp)*100:.4f}%")
    print(f"Precision:   {precision_score(y_t, y_p)*100:.4f}%")
    print(f"F1-Score:    {f1_score(y_t, y_p)*100:.4f}%")
    print(f"ROC-AUC:     {roc_auc_score(y_t, y_prob):.4f}")
    print(f"PR-AUC:      {auc(rec_c, prec_c):.4f}")
    reg_acc = (df['region'] == df['pred_region']).mean()
    print(f"Region Acc:  {reg_acc*100:.4f}%")

    print("\nResolution Stratified:")
    for is_t, name in [(True, '102x102 Thumbnails'), (False, 'High-Res >=300x300')]:
        sub = df[df['is_102'] == is_t]
        s_yt = sub['true_f'].values
        s_yp = sub['pred_f'].values
        s_tn, s_fp, s_fn, s_tp = confusion_matrix(s_yt, s_yp).ravel()
        print(f"  {name} (N={len(sub):,}):")
        print(f"    TP={s_tp:,}, FP={s_fp:,}, TN={s_tn:,}, FN={s_fn:,}")
        print(f"    Accuracy:    {accuracy_score(s_yt, s_yp)*100:.2f}%")
        print(f"    Recall:      {recall_score(s_yt, s_yp)*100:.2f}%")
        print(f"    Specificity: {s_tn/(s_tn+s_fp)*100:.2f}%")
        print(f"    Precision:   {precision_score(s_yt, s_yp)*100:.2f}%")
        print(f"    F1-Score:    {f1_score(s_yt, s_yp)*100:.2f}%")

    print("\nPer-Region Breakdown:")
    for r in sorted(df['region'].unique()):
        sub = df[df['region'] == r]
        s_yt = sub['true_f'].values
        s_yp = sub['pred_f'].values
        r_rec = ((s_yt == 1) & (s_yp == 1)).sum() / (s_yt == 1).sum()
        r_spec = ((s_yt == 0) & (s_yp == 0)).sum() / (s_yt == 0).sum()
        r_prec = precision_score(s_yt, s_yp, zero_division=0)
        print(f"  {r:10s} (N={len(sub):,} | Pos: {(s_yt==1).sum():,}): Recall={r_rec*100:.2f}%, Spec={r_spec*100:.2f}%, Prec={r_prec*100:.2f}%")

if __name__ == '__main__':
    main()
