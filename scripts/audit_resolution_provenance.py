"""
Resolution Provenance Audit for BoneFract Dataset
==================================================
Investigates whether 102x102 BoneFract images are genuinely native low-resolution
images or whether higher-resolution originals exist locally.

Tasks:
1. Scan all 47,931 images across train.csv, validation.csv, test.csv to identify all 102x102 images.
2. Check for original high-resolution versions across archive, metadata, and local directories.
3. Compute perceptual/difference hashes (dHash) to check for resized duplicates.
4. Compute full native-resolution distribution table.
5. Perform patient-level check for multi-resolution images within the same patient.
"""

import os
import sys
import json
import zipfile
import hashlib
import xml.etree.ElementTree as ET
from collections import defaultdict, Counter
import pandas as pd
import numpy as np
from PIL import Image

def get_dhash(image, hash_size=8):
    """
    Computes difference hash (dHash) of an image.
    Invariant to scale and aspect ratio.
    """
    # Resize to (hash_size + 1, hash_size)
    resized = image.convert('L').resize((hash_size + 1, hash_size), Image.Resampling.BILINEAR)
    pixels = np.array(resized, dtype=np.float32)
    # Compare adjacent pixels
    diff = pixels[:, 1:] > pixels[:, :-1]
    # Convert bool array to integer bitstring
    return sum([2 ** i for (i, v) in enumerate(diff.flatten()) if v])

def hamming_distance(h1, h2):
    return bin(h1 ^ h2).count('1')

def read_xlsx_fast(xlsx_path):
    """Reads sheet names and basic strings from xlsx using standard library zipfile/xml."""
    info = {'sheets': [], 'sample_text': []}
    if not os.path.exists(xlsx_path):
        return info
    try:
        with zipfile.ZipFile(xlsx_path, 'r') as z:
            for name in z.namelist():
                if name.startswith('xl/worksheets/'):
                    info['sheets'].append(name)
            if 'xl/sharedStrings.xml' in z.namelist():
                tree = ET.fromstring(z.read('xl/sharedStrings.xml'))
                strings = [elem.text for elem in tree.iter() if elem.text]
                info['sample_text'] = strings[:50]
                info['total_strings'] = len(strings)
    except Exception as e:
        info['error'] = str(e)
    return info

def run_provenance_audit():
    print("="*75)
    print("EXPERIMENT 4 PREPARATION: RESOLUTION PROVENANCE AUDIT")
    print("="*75)

    with open('data/bonefract_path_index.json') as f:
        path_index = json.load(f)

    # ----------------------------------------------------
    # TASK 1 & TASK 4: Resolution Distribution Across Splits
    # ----------------------------------------------------
    print("\nScanning resolutions across project splits...")
    splits = ['train', 'validation', 'test']
    split_dfs = {}
    split_sizes = {}
    split_102_counts = {}

    all_records = []

    for s in splits:
        df = pd.read_csv(f'{s}.csv')
        split_dfs[s] = df
        sizes = []
        n_102 = 0
        for idx, row in df.iterrows():
            iid = row['image_id']
            p = path_index[iid]
            with Image.open(p) as img:
                w, h = img.size
            sizes.append((w, h))
            is_102 = (w == 102 and h == 102)
            if is_102:
                n_102 += 1
            all_records.append({
                'split': s,
                'image_id': iid,
                'patient_id': row['patient_id'],
                'anatomical_region': row['anatomical_region'],
                'fracture_label': row['fracture_label'],
                'width': w,
                'height': h,
                'is_102': is_102,
                'path': p
            })
        split_sizes[s] = sizes
        split_102_counts[s] = n_102
        print(f"  {s:<12}: {len(df):,} total images | {n_102:,} are 102x102 ({n_102/len(df)*100:.2f}%)")

    total_images = len(all_records)
    total_102 = sum(split_102_counts.values())
    print(f"\nTASK 1 SUMMARY:")
    print(f"  Train 102x102:      {split_102_counts['train']:,} / {len(split_dfs['train']):,} ({split_102_counts['train']/len(split_dfs['train'])*100:.2f}%)")
    print(f"  Validation 102x102: {split_102_counts['validation']:,} / {len(split_dfs['validation']):,} ({split_102_counts['validation']/len(split_dfs['validation'])*100:.2f}%)")
    print(f"  Test 102x102:       {split_102_counts['test']:,} / {len(split_dfs['test']):,} ({split_102_counts['test']/len(split_dfs['test'])*100:.2f}%)")
    print(f"  TOTAL 102x102:      {total_102:,} / {total_images:,} ({total_102/total_images*100:.2f}%)")

    # Group complete resolution distribution
    df_all = pd.DataFrame(all_records)
    df_all['res_str'] = df_all['width'].astype(str) + 'x' + df_all['height'].astype(str)

    res_pivot = pd.crosstab(df_all['res_str'], df_all['split'])
    res_pivot['Total'] = res_pivot.sum(axis=1)
    res_pivot['Pct'] = res_pivot['Total'] / total_images * 100
    res_pivot = res_pivot.sort_values(by='Total', ascending=False)

    print("\nTASK 4: COMPLETE RESOLUTION DISTRIBUTION (TOP 15)")
    print(f"{'Resolution':<15} | {'Train':<8} | {'Validation':<10} | {'Test':<8} | {'Total':<8} | {'Pct':<6}")
    print("-" * 65)
    for res, row in res_pivot.head(15).iterrows():
        t = row.get('train', 0)
        v = row.get('validation', 0)
        te = row.get('test', 0)
        tot = row['Total']
        pct = row['Pct']
        print(f"{res:<15} | {t:<8} | {v:<10} | {te:<8} | {tot:<8} | {pct:5.2f}%")

    other_rows = res_pivot.iloc[15:]
    if len(other_rows) > 0:
        o_t = other_rows['train'].sum()
        o_v = other_rows['validation'].sum()
        o_te = other_rows['test'].sum()
        o_tot = other_rows['Total'].sum()
        o_pct = other_rows['Pct'].sum()
        print(f"{'Other ('+str(len(other_rows))+' sizes)':<15} | {o_t:<8} | {o_v:<10} | {o_te:<8} | {o_tot:<8} | {o_pct:5.2f}%")

    # ----------------------------------------------------
    # TASK 5: Patient-Level Resolution Heterogeneity
    # ----------------------------------------------------
    print("\nTASK 5: PATIENT-LEVEL RESOLUTION HETEROGENEITY CHECK")
    patient_res = defaultdict(set)
    patient_res_counts = defaultdict(Counter)

    for idx, row in df_all.iterrows():
        pid = row['patient_id']
        r = row['res_str']
        patient_res[pid].add(r)
        patient_res_counts[pid][r] += 1

    total_patients = len(patient_res)
    single_res_patients = sum(1 for pid, s in patient_res.items() if len(s) == 1)
    multi_res_patients = sum(1 for pid, s in patient_res.items() if len(s) > 1)

    print(f"Total Unique Patients across dataset: {total_patients:,}")
    print(f"Patients with exactly 1 image resolution: {single_res_patients:,} ({single_res_patients/total_patients*100:.2f}%)")
    print(f"Patients with MULTIPLE image resolutions: {multi_res_patients:,} ({multi_res_patients/total_patients*100:.2f}%)")

    # Check specifically if 102x102 coexists with higher-res in same patient
    patients_with_102_and_high = 0
    sample_mixed_patients = []

    for pid, s in patient_res.items():
        has_102 = '102x102' in s
        has_other = any(r != '102x102' for r in s)
        if has_102 and has_other:
            patients_with_102_and_high += 1
            if len(sample_mixed_patients) < 5:
                sample_mixed_patients.append((pid, dict(patient_res_counts[pid])))

    print(f"Patients having BOTH 102x102 and higher-res images: {patients_with_102_and_high:,} ({patients_with_102_and_high/total_patients*100:.2f}%)")
    if sample_mixed_patients:
        print("Sample mixed-resolution patients:")
        for pid, counts in sample_mixed_patients:
            print(f"  {pid}: {counts}")

    # ----------------------------------------------------
    # TASK 2: Archive & Metadata Inspection
    # ----------------------------------------------------
    print("\nTASK 2: CHECKING ARCHIVE, METADATA, & LOCAL REPOSITORIES")
    zip_path = r'C:\Users\arbaz\Downloads\BoneFract A Bone Fracture Dataset\BoneFract A Bone Fracture Dataset\BoneFract A Bone Fracture Dataset.zip'
    print(f"Inspecting raw zip archive: {zip_path}")
    if os.path.exists(zip_path):
        with zipfile.ZipFile(zip_path, 'r') as z:
            zip_members = z.namelist()
            zip_image_members = [m for m in zip_members if m.lower().endswith(('.png', '.jpg', '.jpeg'))]
            zip_other_members = [m for m in zip_members if not m.lower().endswith(('.png', '.jpg', '.jpeg'))]
            print(f"  Total zip entries: {len(zip_members):,}")
            print(f"  Image entries in zip: {len(zip_image_members):,}")
            print(f"  Non-image entries in zip ({len(zip_other_members)}): {zip_other_members}")

            # Check if zip contains any images not in extracted folder
            extracted_ids = set(df_all['image_id'])
            zip_ids = {os.path.basename(m) for m in zip_image_members}
            diff_zip = zip_ids - extracted_ids
            print(f"  Images in zip but not in project index: {len(diff_zip)}")
    else:
        print(f"  Zip file not found at {zip_path}")

    # Metadata Spreadsheets
    meta_dir = r'C:\Users\arbaz\Downloads\BoneFract A Bone Fracture Dataset\BoneFract A Bone Fracture Dataset\BoneFract A Bone Fracture Dataset\BoneFract A Bone Fracture Dataset'
    for xlsx in ['Class_Summary_Report.xlsx', 'PUMHSW-Xray_Dataset_Sheets.xlsx']:
        xp = os.path.join(meta_dir, xlsx)
        res_info = read_xlsx_fast(xp)
        print(f"\nMetadata inspection: {xlsx}")
        print(f"  Sheets found: {len(res_info.get('sheets', []))}")
        print(f"  Sample text strings: {res_info.get('sample_text', [])[:10]}")

    # Check local project directories
    print("\nLocal project directory search for alternative versions...")
    local_data_dirs = ['data', 'assets', 'models', 'reports']
    for d in local_data_dirs:
        if os.path.exists(d):
            n_files = sum(len(files) for _, _, files in os.walk(d))
            print(f"  Directory '{d}': {n_files} files")

    # ----------------------------------------------------
    # TASK 3: Hash / Duplicate / Resizing Verification
    # ----------------------------------------------------
    print("\nTASK 3: HASH & PERCEPTUAL DUPLICATE CHECK")
    print("Testing if 102x102 images are downsampled copies of existing higher-res images...")

    # Sample 1,000 102x102 images and all higher-res images in same anatomical region/patient
    df_102 = df_all[df_all['is_102']].copy()
    df_high = df_all[~df_all['is_102']].copy()

    print(f"  Total 102x102 pool: {len(df_102):,} images")
    print(f"  Total higher-res pool: {len(df_high):,} images")

    # 1. Filename overlap check
    names_102 = set(df_102['image_id'])
    names_high = set(df_high['image_id'])
    shared_names = names_102.intersection(names_high)
    print(f"  Exact duplicate filenames between 102x102 and higher-res: {len(shared_names)}")

    # 2. Patient-constrained visual similarity check
    # Check patients that have both 102x102 and higher-res images
    print("  Checking image similarity for patients with mixed resolutions...")
    mixed_duplicate_pairs = []

    for pid in list(patient_res.keys()):
        s = patient_res[pid]
        if '102x102' in s and len(s) > 1:
            p_sub_102 = df_102[df_102['patient_id'] == pid]
            p_sub_high = df_high[df_high['patient_id'] == pid]

            for _, r102 in p_sub_102.iterrows():
                with Image.open(r102['path']) as im102:
                    h102 = get_dhash(im102)

                for _, rhigh in p_sub_high.iterrows():
                    with Image.open(rhigh['path']) as imhigh:
                        hhigh = get_dhash(imhigh)

                    dist = hamming_distance(h102, hhigh)
                    if dist <= 3:  # Very close perceptual match (near duplicate)
                        mixed_duplicate_pairs.append({
                            'patient_id': pid,
                            'img_102': r102['image_id'],
                            'img_high': rhigh['image_id'],
                            'high_res': rhigh['res_str'],
                            'hamming_dist': dist
                        })

    print(f"  Found {len(mixed_duplicate_pairs)} near-identical image pairs within mixed-resolution patients!")
    for pair in mixed_duplicate_pairs[:5]:
        print(f"    Match: {pair['img_102']} (102x102) <-> {pair['img_high']} ({pair['high_res']}) [Hamming dist={pair['hamming_dist']}]")

    # 3. Global sample check across random 102x102 images vs higher-res images of same anatomical region
    print("\n  Running cross-patient perceptual hash check on sample of 200 102x102 images...")
    np.random.seed(42)
    sample_102 = df_102.sample(200, random_state=42)
    global_matches = []

    # Precompute hashes for higher-res images grouped by region
    high_hashes_by_region = defaultdict(list)
    for _, rhigh in df_high.iterrows():
        try:
            with Image.open(rhigh['path']) as im:
                h = get_dhash(im)
            high_hashes_by_region[rhigh['anatomical_region']].append((rhigh['image_id'], rhigh['res_str'], rhigh['path'], h))
        except Exception:
            pass

    for _, r102 in sample_102.iterrows():
        reg = r102['anatomical_region']
        with Image.open(r102['path']) as im:
            h102 = get_dhash(im)

        for iid_h, res_h, path_h, h_high in high_hashes_by_region[reg]:
            d = hamming_distance(h102, h_high)
            if d <= 2: # virtually identical image
                global_matches.append((r102['image_id'], iid_h, res_h, d))

    print(f"  Global matches found in 200 sample: {len(global_matches)}")
    for m in global_matches[:5]:
        print(f"    102x102: {m[0]} <-> High-Res: {m[1]} ({m[2]}) [dist={m[3]}]")

    # Save complete audit dictionary
    audit_results = {
        'total_images': total_images,
        'total_102_images': total_102,
        'split_102_counts': split_102_counts,
        'split_totals': {s: len(split_dfs[s]) for s in splits},
        'top_resolutions': res_pivot.head(20).to_dict(orient='index'),
        'total_patients': total_patients,
        'single_res_patients': single_res_patients,
        'multi_res_patients': multi_res_patients,
        'patients_with_102_and_high': patients_with_102_and_high,
        'mixed_duplicate_pairs_count': len(mixed_duplicate_pairs),
        'sample_mixed_pairs': mixed_duplicate_pairs[:10]
    }

    os.makedirs('reports', exist_ok=True)
    with open('reports/resolution_provenance_audit.json', 'w') as f:
        json.dump(audit_results, f, indent=2)
    print("\nSaved reports/resolution_provenance_audit.json")

if __name__ == '__main__':
    run_provenance_audit()
