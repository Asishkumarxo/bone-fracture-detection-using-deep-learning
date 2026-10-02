import glob, os, re, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
import numpy as np
from predict import BoneFractureInferencePipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def main():
    p = BoneFractureInferencePipeline()
    
    # Collect all labeled images on disk
    all_imgs = []
    
    # 1. error_analysis/false_negatives (True label = Positive)
    for f in glob.glob('error_analysis/false_negatives/*.*'):
        all_imgs.append((f, 1, 'error_analysis_fn'))
        
    # 2. error_analysis/false_positives (True label = Negative)
    for f in glob.glob('error_analysis/false_positives/*.*'):
        all_imgs.append((f, 0, 'error_analysis_fp'))
        
    # 3. error_analysis/pelvis_misclassified
    for f in glob.glob('error_analysis/pelvis_misclassified/*.*'):
        label = 1 if 'Positive' in os.path.basename(f) else 0
        all_imgs.append((f, label, 'pelvis_misclassified'))

    # 4. error_analysis/region_errors
    for f in glob.glob('error_analysis/region_errors/*.*'):
        label = 1 if 'Positive' in os.path.basename(f) else 0
        all_imgs.append((f, label, 'region_errors'))

    # 5. data/training_curated
    for f in glob.glob('data/training_curated/**/*.*', recursive=True):
        if os.path.isfile(f) and f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
            label = 1 if 'Positive' in os.path.basename(f) else 0
            all_imgs.append((f, label, 'training_curated'))

    # Deduplicate by absolute path
    seen = set()
    unique_samples = []
    for path, lbl, grp in all_imgs:
        canon = os.path.normpath(os.path.abspath(path))
        if canon not in seen:
            seen.add(canon)
            unique_samples.append((path, lbl, grp))

    print(f"Total unique labeled images on disk: {len(unique_samples)}")
    pos_count = sum(lbl for _, lbl, _ in unique_samples)
    neg_count = len(unique_samples) - pos_count
    print(f"Positive: {pos_count}, Negative: {neg_count}")

    records = []
    for path, y_true, grp in unique_samples:
        fname = os.path.basename(path)
        try:
            res = p.analyze(path)
            y_pred = int(res['fracture'])
            prob = res['fracture_confidence']
            records.append({
                'path': path,
                'filename': fname,
                'group': grp,
                'true_label': y_true,
                'pred_label': y_pred,
                'pred_prob': prob,
                'region': res['anatomical_region']
            })
        except Exception as e:
            print(f"Error on {path}: {e}")

    df = pd.DataFrame(records)
    y_t = df['true_label'].values
    y_p = df['pred_label'].values
    probs = df['pred_prob'].values

    acc = accuracy_score(y_t, y_p)
    rec = recall_score(y_t, y_p, zero_division=0)
    prec = precision_score(y_t, y_p, zero_division=0)
    f1 = f1_score(y_t, y_p, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_t, y_p).ravel()

    print("\n" + "="*60)
    print("CURRENT MODEL EVALUATION ON DISK IMAGES (BEFORE FRACTURE FIX)")
    print("="*60)
    print(f"Accuracy:  {acc:.4f} ({acc*100:.2f}%)")
    print(f"Recall:    {rec:.4f} ({rec*100:.2f}%)")
    print(f"Precision: {prec:.4f} ({prec*100:.2f}%)")
    print(f"F1 Score:  {f1:.4f}")
    print(f"Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    print(f"False Negatives: {fn}/{pos_count} ({fn/pos_count*100:.1f}%)")

    # Probabilities for Positive vs Negative
    pos_probs = probs[y_t == 1]
    neg_probs = probs[y_t == 0]
    print(f"\nPositive Images (N={len(pos_probs)}): Median={np.median(pos_probs):.4f}, Mean={np.mean(pos_probs):.4f}, Min={np.min(pos_probs):.4f}, Max={np.max(pos_probs):.4f}")
    print(f"Negative Images (N={len(neg_probs)}): Median={np.median(neg_probs):.4f}, Mean={np.mean(neg_probs):.4f}, Min={np.min(neg_probs):.4f}, Max={np.max(neg_probs):.4f}")

    # Inspect the False Negatives!
    fn_df = df[(df['true_label'] == 1) & (df['pred_label'] == 0)]
    print(f"\nSample False Negatives ({len(fn_df)} total):")
    for _, row in fn_df.head(15).iterrows():
        print(f"  {row['filename'][:40]:<40} | Region: {row['region']:<10} | Prob: {row['pred_prob']*100:5.2f}%")

if __name__ == '__main__':
    main()
