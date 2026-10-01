"""
Part 6 & Part 7: Confusion Analysis and Pelvis Stratified Performance
====================================================================
Generates:
  - reports/anatomical_confusion_matrix.png (7x7 matrix)
  - reports/anatomical_classification_report.csv (Precision, Recall, F1, Support)
  - error_analysis/pelvis_misclassified/ (Curated misclassified pelvis images)
  - error_analysis/pelvis_misclassified/pelvis_misclassified_catalog.csv
"""

import os
import shutil
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

def main():
    test_pred_path = 'reports/test_predictions.csv'
    if not os.path.exists(test_pred_path):
        print(f"Error: {test_pred_path} does not exist.")
        return

    df = pd.read_csv(test_pred_path)
    print(f"Loaded {len(df)} predictions from {test_pred_path}")

    # Standard region order
    region_order = ['Arm', 'Foot', 'Hand', 'Lower leg', 'Thigh', 'pelvis', 'wrist']
    
    # --- PART 6: Confusion Matrix & Classification Report ---
    cm = confusion_matrix(df['true_region'], df['predicted_region'], labels=region_order)
    
    # Plot 7x7 confusion matrix
    plt.figure(figsize=(9, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=region_order,
        yticklabels=region_order,
        cbar=True
    )
    plt.title("7x7 Anatomical-Region Confusion Matrix (Test Set N=4,778)", fontsize=13, fontweight='bold', pad=15)
    plt.xlabel("Predicted Region", fontsize=11, fontweight='bold')
    plt.ylabel("Ground Truth Region", fontsize=11, fontweight='bold')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    os.makedirs('reports', exist_ok=True)
    cm_fig_path = 'reports/anatomical_confusion_matrix.png'
    plt.savefig(cm_fig_path, dpi=200)
    plt.close()
    print(f"Saved {cm_fig_path}")

    # Key transition pairs of interest:
    r2i = {r: i for i, r in enumerate(region_order)}
    pelvis_to_lower_leg = cm[r2i['pelvis'], r2i['Lower leg']]
    pelvis_to_thigh = cm[r2i['pelvis'], r2i['Thigh']]
    thigh_to_lower_leg = cm[r2i['Thigh'], r2i['Lower leg']]
    lower_leg_to_thigh = cm[r2i['Lower leg'], r2i['Thigh']]

    print("\n--- KEY CONFUSION PAIRS OF INTEREST ---")
    print(f"Pelvis -> Lower Leg: {pelvis_to_lower_leg} cases")
    print(f"Pelvis -> Thigh:     {pelvis_to_thigh} cases")
    print(f"Thigh -> Lower Leg:  {thigh_to_lower_leg} cases")
    print(f"Lower Leg -> Thigh:  {lower_leg_to_thigh} cases")

    # Classification report
    clf_dict = classification_report(
        df['true_region'],
        df['predicted_region'],
        labels=region_order,
        output_dict=True,
        zero_division=0
    )
    
    report_rows = []
    for reg in region_order:
        stats = clf_dict[reg]
        report_rows.append({
            'anatomical_region': reg,
            'precision': round(stats['precision'], 4),
            'recall': round(stats['recall'], 4),
            'f1_score': round(stats['f1-score'], 4),
            'support': int(stats['support'])
        })

    # Add macro and weighted averages
    report_rows.append({
        'anatomical_region': 'macro_avg',
        'precision': round(clf_dict['macro avg']['precision'], 4),
        'recall': round(clf_dict['macro avg']['recall'], 4),
        'f1_score': round(clf_dict['macro avg']['f1-score'], 4),
        'support': int(clf_dict['macro avg']['support'])
    })
    report_rows.append({
        'anatomical_region': 'weighted_avg',
        'precision': round(clf_dict['weighted avg']['precision'], 4),
        'recall': round(clf_dict['weighted avg']['recall'], 4),
        'f1_score': round(clf_dict['weighted avg']['f1-score'], 4),
        'support': int(clf_dict['weighted avg']['support'])
    })

    rep_df = pd.DataFrame(report_rows)
    clf_csv_path = 'reports/anatomical_classification_report.csv'
    rep_df.to_csv(clf_csv_path, index=False)
    print(f"Saved {clf_csv_path}")
    print("\nAnatomical Classification Metrics:")
    print(rep_df.to_string())

    # --- PART 7: Analyze Pelvis Performance ---
    pelvis_df = df[df['true_region'] == 'pelvis'].copy()
    n_pelvis = len(pelvis_df)
    n_correct = (pelvis_df['predicted_region'] == 'pelvis').sum()
    n_incorrect = n_pelvis - n_correct
    
    pelvis_prec = clf_dict['pelvis']['precision']
    pelvis_rec = clf_dict['pelvis']['recall']
    pelvis_f1 = clf_dict['pelvis']['f1-score']

    print("\n==========================================")
    print("       PART 7: PELVIS PERFORMANCE         ")
    print("==========================================")
    print(f"Number of Pelvis Images:        {n_pelvis}")
    print(f"Correctly Classified as Pelvis: {n_correct} ({n_correct/n_pelvis*100:.2f}%)")
    print(f"Incorrectly Classified:         {n_incorrect} ({n_incorrect/n_pelvis*100:.2f}%)")
    print(f"Pelvis Recall:                  {pelvis_rec:.4f} ({pelvis_rec*100:.2f}%)")
    print(f"Pelvis Precision:               {pelvis_prec:.4f} ({pelvis_prec*100:.2f}%)")
    print(f"Pelvis F1-Score:                {pelvis_f1:.4f}")
    print("\nBreakdown of Misclassified Pelvis Images:")
    print(pelvis_df[pelvis_df['predicted_region'] != 'pelvis']['predicted_region'].value_counts())

    # Save incorrectly classified pelvis images
    out_dir = 'error_analysis/pelvis_misclassified'
    os.makedirs(out_dir, exist_ok=True)
    
    misclassified_pelvis = pelvis_df[pelvis_df['predicted_region'] != 'pelvis']
    catalog_rows = []

    for idx, row in misclassified_pelvis.iterrows():
        img_id = row['image_id']
        src_path = os.path.join('processed/test', img_id)
        dst_path = os.path.join(out_dir, img_id)
        
        if os.path.exists(src_path):
            shutil.copy2(src_path, dst_path)
            
        catalog_rows.append({
            'image_id': img_id,
            'true_class': row['true_region'],
            'predicted_class': row['predicted_region'],
            'confidence': row['region_confidence'],
            'probability_pelvis': row['probability_pelvis'],
            'probability_thigh': row['probability_thigh'],
            'probability_lower_leg': row['probability_lower_leg'],
            'true_fracture': row['true_fracture'],
            'predicted_fracture': row['predicted_fracture']
        })

    cat_df = pd.DataFrame(catalog_rows)
    cat_path = os.path.join(out_dir, 'pelvis_misclassified_catalog.csv')
    cat_df.to_csv(cat_path, index=False)
    print(f"\nSaved {len(cat_df)} misclassified pelvis images and metadata catalog to {cat_path}")

if __name__ == '__main__':
    main()
