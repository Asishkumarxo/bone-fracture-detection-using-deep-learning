import torch
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import torchvision.transforms as T
from PIL import Image

import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.train_improved_fracture_classifier import DedicatedFractureClassifier
from inference.preprocessing import resize_and_pad

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt = torch.load('models/checkpoints/fracture_model_improved.pth', map_location=device, weights_only=False)
    model = DedicatedFractureClassifier(pretrained=False).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    val_df = pd.read_csv('data/fracture_splits/val_fracture.csv')
    print(f"Validation set size: {len(val_df)} (Pos: {sum(val_df['fracture_target']==1)}, Neg: {sum(val_df['fracture_target']==0)})")

    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    val_probs = []
    val_targets = val_df['fracture_target'].values

    for idx, row in val_df.iterrows():
        p = row['path']
        with Image.open(p) as im:
            gray = im.convert('L')
            padded = resize_and_pad(gray, target_size=(224, 224))
            rgb = padded.convert('RGB')
            t = transform(rgb).unsqueeze(0).to(device)
        with torch.no_grad():
            logit = model(t).item()
            prob = torch.sigmoid(torch.tensor(logit)).item()
        val_probs.append(prob)

    val_probs = np.array(val_probs)

    thresholds = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]

    records = []
    print("\n" + "="*80)
    print("PHASE 18: THRESHOLD OPTIMIZATION ON VALIDATION SET")
    print("="*80)
    print(f"{'Threshold':<10} | {'Precision':<10} | {'Recall/Sens':<12} | {'Specificity':<12} | {'F1 Score':<10} | {'TN':<4} {'FP':<4} {'FN':<4} {'TP':<4}")
    print("-" * 80)

    for th in thresholds:
        preds = (val_probs >= th).astype(int)
        prec = precision_score(val_targets, preds, zero_division=0)
        rec = recall_score(val_targets, preds, zero_division=0)
        f1 = f1_score(val_targets, preds, zero_division=0)
        tn, fp, fn, tp = confusion_matrix(val_targets, preds).ravel()
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        records.append({
            'threshold': th,
            'precision': round(prec, 4),
            'recall_sensitivity': round(rec, 4),
            'specificity': round(spec, 4),
            'f1_score': round(f1, 4),
            'TN': tn, 'FP': fp, 'FN': fn, 'TP': tp
        })
        print(f"{th:<10.2f} | {prec*100:8.2f}% | {rec*100:10.2f}% | {spec*100:10.2f}% | {f1:10.4f} | {tn:<4} {fp:<4} {fn:<4} {tp:<4}")

    df_res = pd.DataFrame(records)
    best_row = df_res.sort_values(by=['f1_score', 'precision'], ascending=False).iloc[0]
    print("\n" + "="*80)
    print(f"OPTIMAL VALIDATION THRESHOLD: {best_row['threshold']:.2f} (F1 = {best_row['f1_score']:.4f}, Recall = {best_row['recall_sensitivity']*100:.1f}%, Precision = {best_row['precision']*100:.1f}%)")
    print("="*80)

    df_res.to_csv('reports/threshold_optimization_validation.csv', index=False)

if __name__ == '__main__':
    main()
