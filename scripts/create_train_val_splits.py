import pandas as pd
import numpy as np

def create_splits():
    df = pd.read_csv('data/fracture_splits/all_unique_images.csv')
    print(f"Total unique images: {len(df)}")
    
    # We want a stratified 80% train / 20% validation split
    # Stratified by fracture_label and region where possible
    np.random.seed(42)
    
    # Create stratification key
    df['strat_key'] = df['region'] + '_' + df['fracture_label']
    
    train_indices = []
    val_indices = []
    
    for key, group in df.groupby('strat_key'):
        indices = group.index.tolist()
        np.random.shuffle(indices)
        if len(indices) == 1:
            # If only 1 sample, put in train
            train_indices.extend(indices)
        else:
            n_val = max(1, int(round(len(indices) * 0.20)))
            val_indices.extend(indices[:n_val])
            train_indices.extend(indices[n_val:])
            
    train_df = df.loc[train_indices].reset_index(drop=True)
    val_df = df.loc[val_indices].reset_index(drop=True)
    
    print("\n--- TRAIN SPLIT (N = {}) ---".format(len(train_df)))
    print(train_df['fracture_label'].value_counts())
    print(pd.crosstab(train_df['region'], train_df['fracture_label']))
    
    print("\n--- VALIDATION SPLIT (N = {}) ---".format(len(val_df)))
    print(val_df['fracture_label'].value_counts())
    print(pd.crosstab(val_df['region'], val_df['fracture_label']))
    
    train_df.to_csv('data/fracture_splits/train_fracture.csv', index=False)
    val_df.to_csv('data/fracture_splits/val_fracture.csv', index=False)
    print("\nSaved train_fracture.csv and val_fracture.csv successfully.")

if __name__ == '__main__':
    create_splits()
