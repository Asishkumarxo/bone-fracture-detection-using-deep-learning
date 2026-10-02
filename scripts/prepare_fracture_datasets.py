import os, glob, re, hashlib
import pandas as pd
import numpy as np
from PIL import Image

def get_md5(filepath):
    hasher = hashlib.md5()
    with open(filepath, 'rb') as f:
        hasher.update(f.read())
    return hasher.hexdigest()

def main():
    print("Collecting and auditing all available labeled images for fracture training...")
    
    # Candidate source directories
    sources = [
        'error_analysis/false_negatives',
        'error_analysis/false_positives',
        'error_analysis/pelvis_misclassified',
        'error_analysis/region_errors',
        'data/training_curated'
    ]
    
    raw_records = []
    
    # 1. Collect from error_analysis and training_curated
    for src_dir in sources:
        for f in glob.glob(f"{src_dir}/**/*.*", recursive=True):
            if not f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                continue
            if 'prediction_' in f or 'gradcam' in f or 'catalog' in f:
                continue
            fname = os.path.basename(f)
            
            # Determine label strictly from verified filename and directory semantics
            fname_lower = fname.lower()
            if 'positive' in fname_lower or 'crack' in fname_lower:
                label = 1
                label_str = 'Positive'
            elif 'negative' in fname_lower or 'normal' in fname_lower:
                label = 0
                label_str = 'Negative'
            elif 'false_negatives' in f.lower():
                # False negatives are ground-truth positive fractures
                label = 1
                label_str = 'Positive'
            elif 'false_positives' in f.lower():
                # False positives are ground-truth normal/negative images
                label = 0
                label_str = 'Negative'
            else:
                continue
                
            # Determine region
            m = re.match(r'^(?:fn_|fp_|hce_|hiconf_|reg_)?([A-Za-z ]+?)_', fname)
            region = m.group(1).strip() if m else 'pelvis'
            if region not in ['Arm', 'Foot', 'Hand', 'Lower leg', 'Thigh', 'pelvis', 'wrist']:
                # fallback by folder
                for r in ['Arm', 'Foot', 'Hand', 'Lower leg', 'Thigh', 'pelvis', 'wrist']:
                    if r.lower() in f.lower():
                        region = r
                        break
            
            # Verify image can be opened and get size
            try:
                with Image.open(f) as im:
                    im.verify()
                with Image.open(f) as im:
                    w, h = im.size
                md5 = get_md5(f)
                raw_records.append({
                    'path': os.path.normpath(f).replace('\\', '/'),
                    'filename': fname,
                    'region': region,
                    'fracture_label': label_str,
                    'fracture_target': label,
                    'md5': md5,
                    'width': w,
                    'height': h
                })
            except Exception as e:
                print(f"Skipping corrupt image {f}: {e}")
                
    df_all = pd.DataFrame(raw_records)
    print(f"Total candidate files read: {len(df_all)}")
    
    # Deduplicate strictly by md5 hash so no image is duplicated
    df_unique = df_all.drop_duplicates(subset=['md5']).reset_index(drop=True)
    print(f"Total unique images after MD5 deduplication: {len(df_unique)}")
    print(df_unique['fracture_label'].value_counts())
    print("\nBreakdown by region and fracture status:")
    print(pd.crosstab(df_unique['region'], df_unique['fracture_label']))
    
    # Save curated catalog
    os.makedirs('data/fracture_splits', exist_ok=True)
    df_unique.to_csv('data/fracture_splits/all_unique_images.csv', index=False)
    print("\nSaved catalog to data/fracture_splits/all_unique_images.csv")

if __name__ == '__main__':
    main()
