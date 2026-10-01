import os
import sys
import glob
import time
import hashlib
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
from PIL import Image

def get_channel_and_depth(mode):
    mode_map = {
        '1': (1, '1-bit'),
        'L': (1, '8-bit grayscale'),
        'P': (1, '8-bit palette'),
        'RGB': (3, '8-bit per channel (24-bit RGB)'),
        'RGBA': (4, '8-bit per channel (32-bit RGBA)'),
        'CMYK': (4, '8-bit per channel (32-bit CMYK)'),
        'YCbCr': (3, '8-bit per channel (24-bit YCbCr)'),
        'I': (1, '32-bit signed integer'),
        'F': (1, '32-bit floating point'),
        'I;16': (1, '16-bit unsigned integer'),
        'I;16L': (1, '16-bit little-endian'),
        'I;16B': (1, '16-bit big-endian'),
    }
    return mode_map.get(mode, (len(mode) if mode else 1, f'Unknown ({mode})'))

def audit_single_image(filepath, base_dir):
    # Relpath normalization
    rel_path = os.path.relpath(filepath, base_dir).replace('\\', '/')
    parts = rel_path.split('/')
    
    # Expected structure: {split}/{region}/{patient_id}/{label}/{filename}
    split = parts[0] if len(parts) > 0 else 'Unknown'
    region = parts[1] if len(parts) > 1 else 'Unknown'
    folder_patient = parts[2] if len(parts) > 2 else 'Unknown'
    folder_label = parts[3] if len(parts) > 3 else 'Unknown'
    filename = os.path.basename(filepath)
    image_id = filename
    
    # Filename parsing: e.g. Arm_patient02324_Positive_001.png
    # Pattern check
    file_patient_match = re.search(r'(patient\d+)', filename, re.IGNORECASE)
    filename_patient = file_patient_match.group(1).lower() if file_patient_match else 'None'
    
    file_label_match = re.search(r'_(Positive|Negative)_', filename, re.IGNORECASE)
    filename_label = file_label_match.group(1).capitalize() if file_label_match else 'None'
    
    # Region in filename check
    filename_region = filename.split('_')[0] if '_' in filename else 'None'
    
    patient_id = folder_patient if folder_patient != 'Unknown' else filename_patient
    fracture_label = folder_label if folder_label != 'Unknown' else filename_label
    
    # Stat file
    file_stat = os.stat(filepath)
    file_size = file_stat.st_size
    
    # MD5 Checksum
    hasher = hashlib.md5()
    with open(filepath, 'rb') as f:
        # Read in 64kb chunks
        while chunk := f.read(65536):
            hasher.update(chunk)
    md5_hash = hasher.hexdigest()
    
    # Image inspection
    is_corrupt = False
    width = None
    height = None
    channels = None
    file_format = None
    bit_depth = None
    mode = None
    
    try:
        with Image.open(filepath) as img:
            file_format = img.format
            width, height = img.size
            mode = img.mode
            ch, depth = get_channel_and_depth(mode)
            channels = ch
            bit_depth = depth
            # Verify structure
            img.verify()
        
        # In PIL, verify() invalidates the image object; test actual loading
        with Image.open(filepath) as img:
            img.load()
    except Exception as e:
        is_corrupt = True
    
    aspect_ratio = round(width / height, 4) if (width and height and height > 0) else None
    
    # Suspicious filename / metadata flags
    suspicious = False
    suspicion_reasons = []
    
    if folder_patient.lower() != filename_patient.lower() and filename_patient != 'None':
        suspicious = True
        suspicion_reasons.append(f'Patient mismatch: folder={folder_patient} vs file={filename_patient}')
        
    if folder_label.lower() != filename_label.lower() and filename_label != 'None':
        suspicious = True
        suspicion_reasons.append(f'Label mismatch: folder={folder_label} vs file={filename_label}')
        
    if filename_region.lower() != region.lower() and filename_region != 'None':
        suspicious = True
        suspicion_reasons.append(f'Region mismatch: folder={region} vs file={filename_region}')
        
    if file_size == 0:
        suspicious = True
        suspicion_reasons.append('Zero file size')
        
    if is_corrupt:
        suspicious = True
        suspicion_reasons.append('Corrupted image')
        
    if width and height:
        if width < 50 or height < 50:
            suspicious = True
            suspicion_reasons.append(f'Extremely small dimension ({width}x{height})')
        if width > 4000 or height > 4000:
            suspicious = True
            suspicion_reasons.append(f'Extremely large dimension ({width}x{height})')
        if aspect_ratio and (aspect_ratio < 0.25 or aspect_ratio > 4.0):
            suspicious = True
            suspicion_reasons.append(f'Unusual aspect ratio ({aspect_ratio:.2f})')
            
    return {
        'image_path': filepath.replace('\\', '/'),
        'image_id': image_id,
        'patient_id_if_available': patient_id,
        'anatomical_region': region,
        'fracture_label': fracture_label,
        'width': width,
        'height': height,
        'channels': channels,
        'file_format': file_format,
        'file_size': file_size,
        'is_corrupt': is_corrupt,
        'is_duplicate': False, # Populated later across dataset
        'split': split,
        # Additional fields
        'bit_depth': bit_depth,
        'mode': mode,
        'aspect_ratio': aspect_ratio,
        'md5_hash': md5_hash,
        'is_suspicious': suspicious,
        'suspicion_reasons': '; '.join(suspicion_reasons) if suspicion_reasons else 'None'
    }

def main():
    base_dir = 'E:/BoneFract A Bone Fracture Dataset'
    print(f'Starting audit on: {base_dir}')
    
    # Gather image files
    all_files = []
    for split in ['train', 'valid', 'test']:
        split_dir = os.path.join(base_dir, split)
        if os.path.exists(split_dir):
            for root, dirs, files in os.walk(split_dir):
                for f in files:
                    if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
                        all_files.append(os.path.join(root, f))
                        
    total_images = len(all_files)
    print(f'Found {total_images} images across train, valid, test.')
    
    t0 = time.time()
    results = []
    
    # Thread pool for fast I/O & MD5
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(audit_single_image, f, base_dir): f for f in all_files}
        count = 0
        for future in as_completed(futures):
            res = future.result()
            results.append(res)
            count += 1
            if count % 5000 == 0 or count == total_images:
                elapsed = time.time() - t0
                rate = count / elapsed
                print(f'Audited {count}/{total_images} ({(count/total_images)*100:.1f}%) in {elapsed:.1f}s ({rate:.0f} img/s)')
                
    print('Converting to DataFrame...')
    df = pd.DataFrame(results)
    
    # Duplicate Analysis via MD5
    print('Analyzing duplicates and hash collisions...')
    hash_counts = df['md5_hash'].value_counts()
    duplicate_hashes = set(hash_counts[hash_counts > 1].index)
    df['is_duplicate'] = df['md5_hash'].isin(duplicate_hashes)
    
    # Mark duplicate groups
    hash_to_first_id = df.drop_duplicates(subset=['md5_hash'], keep='first').set_index('md5_hash')['image_id'].to_dict()
    df['duplicate_of'] = df.apply(lambda r: hash_to_first_id[r['md5_hash']] if (r['is_duplicate'] and r['image_id'] != hash_to_first_id[r['md5_hash']]) else '', axis=1)
    
    # Cross-split duplicate leakage check
    split_leakage = []
    grouped_hash = df.groupby('md5_hash')
    for h, group in grouped_hash:
        splits = group['split'].unique()
        if len(splits) > 1:
            split_leakage.append(h)
    df['cross_split_duplicate'] = df['md5_hash'].isin(set(split_leakage))
    
    # Cross-patient duplicate check
    patient_dup = []
    for h, group in grouped_hash:
        pats = group['patient_id_if_available'].unique()
        if len(pats) > 1:
            patient_dup.append(h)
    df['cross_patient_duplicate'] = df['md5_hash'].isin(set(patient_dup))
    
    # Cross-label duplicate check (same image with Positive AND Negative label!)
    label_conflict = []
    for h, group in grouped_hash:
        labels = group['fracture_label'].unique()
        if len(labels) > 1:
            label_conflict.append(h)
    df['cross_label_conflict'] = df['md5_hash'].isin(set(label_conflict))
    
    # Sort deterministically
    df.sort_values(by=['split', 'anatomical_region', 'patient_id_if_available', 'image_id'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    # Save dataset_audit.csv
    os.makedirs('reports', exist_ok=True)
    audit_csv_path = 'dataset_audit.csv'
    audit_csv_reports = os.path.join('reports', 'dataset_audit.csv')
    df.to_csv(audit_csv_path, index=False)
    df.to_csv(audit_csv_reports, index=False)
    print(f'Saved {audit_csv_path} and {audit_csv_reports}')
    
    # Questionable records
    questionable_mask = (
        df['is_corrupt'] |
        df['is_suspicious'] |
        df['cross_split_duplicate'] |
        df['cross_label_conflict']
    )
    df_questionable = df[questionable_mask].copy()
    quest_csv_path = os.path.join('reports', 'questionable_records.csv')
    df_questionable.to_csv(quest_csv_path, index=False)
    print(f'Identified {len(df_questionable)} questionable records. Saved to {quest_csv_path}')
    
    # Generate dataset_summary.csv
    print('Generating summary...')
    # Region x Fracture Label counts overall and by split
    summary = df.groupby(['split', 'anatomical_region', 'fracture_label']).agg(
        image_count=('image_id', 'count'),
        unique_patients=('patient_id_if_available', 'nunique'),
        duplicate_count=('is_duplicate', lambda x: x.sum()),
        corrupt_count=('is_corrupt', lambda x: x.sum()),
        avg_width=('width', 'mean'),
        avg_height=('height', 'mean'),
        avg_size_kb=('file_size', lambda x: x.mean() / 1024)
    ).reset_index()
    
    summary_csv_path = 'dataset_summary.csv'
    summary_csv_reports = os.path.join('reports', 'dataset_summary.csv')
    summary.to_csv(summary_csv_path, index=False)
    summary.to_csv(summary_csv_reports, index=False)
    print(f'Saved {summary_csv_path} and {summary_csv_reports}')
    
    # Overall summary by region and fracture status
    region_summary = df.groupby(['anatomical_region', 'fracture_label']).agg(
        image_count=('image_id', 'count'),
        unique_patients=('patient_id_if_available', 'nunique'),
        duplicate_count=('is_duplicate', lambda x: x.sum())
    ).reset_index()
    region_summary_path = os.path.join('reports', 'region_fracture_summary.csv')
    region_summary.to_csv(region_summary_path, index=False)
    
    print('Audit run finished successfully!')

if __name__ == '__main__':
    main()
