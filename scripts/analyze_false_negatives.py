import pandas as pd
import numpy as np
import os
from PIL import Image

def analyze_false_negatives():
    df = pd.read_csv('test_predictions.csv')
    fn_df = df[(df['true_fracture'] == True) & (df['predicted_fracture'] == False)].copy()
    fn_df = fn_df.sort_values('fracture_confidence', ascending=True)

    print(f"Total False Negatives in untouched test set: {len(fn_df)} out of {sum(df['true_fracture'] == True)} fractures ({len(fn_df)/sum(df['true_fracture'] == True)*100:.1f}%)")
    print("\nTop 25 Lowest-Confidence False Negatives:")
    print(f"{'Filename':<42} | {'Region':<10} | {'Prob':<8} | {'Pred Region':<12}")
    print("-" * 80)
    for idx, row in fn_df.head(25).reset_index().iterrows():
        print(f"{row['image_id']:<42} | {row['true_region']:<10} | {row['fracture_confidence']*100:5.2f}%  | {row['predicted_region']:<12}")

    print("\nFalse Negatives by Anatomical Region:")
    fn_by_reg = fn_df['true_region'].value_counts()
    tot_by_reg = df[df['true_fracture'] == True]['true_region'].value_counts()
    for reg in tot_by_reg.index:
        n_fn = fn_by_reg.get(reg, 0)
        n_tot = tot_by_reg[reg]
        print(f"  {reg:<12}: {n_fn:3d} / {n_tot:3d} false negatives ({n_fn/n_tot*100:5.1f}%)")

if __name__ == '__main__':
    analyze_false_negatives()
