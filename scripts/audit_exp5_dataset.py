"""
Experiment 5: Mandatory Dataset Audit & Leakage Verification
=============================================================
Strictly verifies the dataset for Experiment 5 against the mandatory root:
C:\\Users\\Donipudi Asish Kumar\\Desktop\\BoneFract A Bone Fracture Dataset

Audits:
1. Dataset root accessibility and subfolder structure
2. Total images on disk
3. Training manifest (reports/exp4_sample_manifest.csv) integrity & SHA-256
4. Validation split (validation.csv) integrity
5. Held-out test split (test.csv) integrity
6. File existence on disk for every sample
7. Data distributions (region, fracture, resolution)
8. Zero train/val/test image leakage check
9. Outputs reports/exp5_dataset_audit.json
"""

import os
import sys
import json
import hashlib
import pandas as pd
from PIL import Image

def compute_sha256(filepath):
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def resolve_bonefract_path(image_id, old_path, dataset_root):
    norm = old_path.replace("\\", "/")
    parts = norm.split("/")
    sub_idx = -1
    for i, p in enumerate(parts):
        if p in ("train", "valid", "test"):
            sub_idx = i
            break
    if sub_idx != -1:
        rel = os.path.join(*parts[sub_idx:])
        return os.path.join(dataset_root, rel)
    return None

def run_audit():
    dataset_root = r"C:\Users\Donipudi Asish Kumar\Desktop\BoneFract A Bone Fracture Dataset"
    print("=" * 80)
    print("EXPERIMENT 5: MANDATORY DATASET AUDIT & LEAKAGE VERIFICATION")
    print("=" * 80)
    print(f"Mandatory Dataset Root: {dataset_root}")

    if not os.path.exists(dataset_root):
        raise FileNotFoundError(f"CRITICAL ERROR: Mandatory dataset root does not exist: {dataset_root}")

    # 1. Scan directory structure
    root_entries = os.listdir(dataset_root)
    print(f"Root entries: {root_entries}")

    expected_subdirs = ['train', 'valid', 'test']
    for sd in expected_subdirs:
        sd_path = os.path.join(dataset_root, sd)
        if not os.path.exists(sd_path):
            raise FileNotFoundError(f"Missing expected directory: {sd_path}")

    # 2. Verify bonefract_path_index.json
    path_index_file = "data/bonefract_path_index.json"
    if not os.path.exists(path_index_file):
        raise FileNotFoundError(f"Required path index missing: {path_index_file}")

    with open(path_index_file, "r", encoding="utf-8") as f:
        bonefract_path_index = json.load(f)
    print(f"Loaded bonefract_path_index: {len(bonefract_path_index):,} mapped images.")

    # 3. Audit Training Manifest (reports/exp4_sample_manifest.csv)
    manifest_path = "reports/exp4_sample_manifest.csv"
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Experiment 4 sample manifest missing: {manifest_path}")

    manifest_hash = compute_sha256(manifest_path)
    train_df = pd.read_csv(manifest_path)
    print(f"\n[1/4] Auditing Training Manifest: {manifest_path}")
    print(f"  SHA-256 Hash: {manifest_hash}")
    print(f"  Total Rows: {len(train_df):,}")
    print(f"  Unique Images: {train_df['image_id'].nunique():,}")

    if len(train_df) != 14000:
        raise ValueError(f"Expected 14,000 samples in balanced manifest, found {len(train_df)}")

    # Verify every training file exists on disk inside dataset_root
    train_missing = []
    train_outside_root = []
    norm_root = os.path.normpath(dataset_root).lower()

    for idx, row in train_df.iterrows():
        p = row['resolved_path']
        norm_p = os.path.normpath(p).lower()
        if not norm_p.startswith(norm_root):
            train_outside_root.append((row['image_id'], p))
        if not os.path.exists(p):
            train_missing.append((row['image_id'], p))

    if train_outside_root:
        raise ValueError(f"Found {len(train_outside_root)} images outside permitted root!")
    if train_missing:
        raise FileNotFoundError(f"Found {len(train_missing)} missing training images on disk!")

    print(f"  -> All 14,000 training images exist on disk strictly inside permitted root.")

    # Regional distribution
    train_region_counts = train_df['anatomical_region'].value_counts().to_dict()
    print("  -> Regional Distribution:")
    for reg, cnt in sorted(train_region_counts.items()):
        print(f"     {reg}: {cnt:,}")

    # Fracture label distribution
    train_fracture_counts = train_df['fracture_label'].value_counts().to_dict()
    print(f"  -> Fracture Label Distribution: {train_fracture_counts}")

    # Resolution distribution
    train_res_102 = int((train_df['is_102x102'] == True).sum())
    train_res_high = int((train_df['is_102x102'] == False).sum())
    print(f"  -> Native Resolution: 102x102={train_res_102:,}, High-Res={train_res_high:,}")

    # 4. Audit Validation Split (validation.csv)
    print(f"\n[2/4] Auditing Validation Split: validation.csv")
    val_df = pd.read_csv("validation.csv")
    val_missing = []
    val_paths = {}
    for idx, row in val_df.iterrows():
        img_id = row['image_id']
        if img_id not in bonefract_path_index:
            val_missing.append(img_id)
            continue
        p = resolve_bonefract_path(img_id, bonefract_path_index[img_id], dataset_root)
        if not p or not os.path.exists(p):
            val_missing.append(img_id)
        else:
            val_paths[img_id] = p

    if val_missing:
        raise FileNotFoundError(f"Found {len(val_missing)} missing validation images!")
    print(f"  -> Total Validation Samples: {len(val_df):,}")
    print(f"  -> All {len(val_df):,} validation images exist strictly inside permitted root.")
    val_region_counts = val_df['anatomical_region'].value_counts().to_dict()
    val_fracture_counts = val_df['fracture_label'].value_counts().to_dict()

    # 5. Audit Test Split (test.csv)
    print(f"\n[3/4] Auditing Held-Out Test Split: test.csv")
    test_df = pd.read_csv("test.csv")
    test_missing = []
    test_paths = {}
    test_resolutions = {'102x102': 0, 'high_res': 0}

    for idx, row in test_df.iterrows():
        img_id = row['image_id']
        if img_id not in bonefract_path_index:
            test_missing.append(img_id)
            continue
        p = resolve_bonefract_path(img_id, bonefract_path_index[img_id], dataset_root)
        if not p or not os.path.exists(p):
            test_missing.append(img_id)
        else:
            test_paths[img_id] = p

    if test_missing:
        raise FileNotFoundError(f"Found {len(test_missing)} missing test images!")
    print(f"  -> Total Test Samples: {len(test_df):,}")
    print(f"  -> All {len(test_df):,} test images exist strictly inside permitted root.")
    test_region_counts = test_df['anatomical_region'].value_counts().to_dict()
    test_fracture_counts = test_df['fracture_label'].value_counts().to_dict()

    # 6. Leakage & Overlap Check
    print(f"\n[4/4] Checking Train / Validation / Test Leakage...")
    train_ids = set(train_df['image_id'])
    val_ids = set(val_df['image_id'])
    test_ids = set(test_df['image_id'])

    train_val_overlap = train_ids.intersection(val_ids)
    train_test_overlap = train_ids.intersection(test_ids)
    val_test_overlap = val_ids.intersection(test_ids)

    print(f"  -> Train and Validation overlap: {len(train_val_overlap)} images")
    print(f"  -> Train and Test overlap: {len(train_test_overlap)} images")
    print(f"  -> Validation and Test overlap: {len(val_test_overlap)} images")

    if train_val_overlap or train_test_overlap or val_test_overlap:
        raise ValueError(
            f"DATA LEAKAGE DETECTED! Overlaps: "
            f"train_val={len(train_val_overlap)}, "
            f"train_test={len(train_test_overlap)}, "
            f"val_test={len(val_test_overlap)}"
        )
    print("  -> ZERO IMAGE LEAKAGE CONFIRMED: All three splits are mutually disjoint.")

    # Compile Audit Report Dict
    audit_report = {
        "dataset_root": dataset_root,
        "resolved_root_confirmed": True,
        "root_directories": root_entries,
        "training_manifest": {
            "path": manifest_path,
            "sha256": manifest_hash,
            "total_samples": len(train_df),
            "unique_images": train_df['image_id'].nunique(),
            "region_counts": train_region_counts,
            "fracture_counts": train_fracture_counts,
            "resolution_counts": {
                "thumbnail_102x102": train_res_102,
                "high_res": train_res_high
            },
            "all_files_exist": True
        },
        "validation_split": {
            "path": "validation.csv",
            "total_samples": len(val_df),
            "region_counts": val_region_counts,
            "fracture_counts": val_fracture_counts,
            "all_files_exist": True
        },
        "test_split": {
            "path": "test.csv",
            "total_samples": len(test_df),
            "region_counts": test_region_counts,
            "fracture_counts": test_fracture_counts,
            "all_files_exist": True
        },
        "leakage_verification": {
            "train_val_overlap": len(train_val_overlap),
            "train_test_overlap": len(train_test_overlap),
            "val_test_overlap": len(val_test_overlap),
            "is_leakage_free": True
        },
        "data_integrity_status": "PASSED"
    }

    os.makedirs("reports", exist_ok=True)
    out_audit_path = "reports/exp5_dataset_audit.json"
    with open(out_audit_path, "w", encoding="utf-8") as f:
        json.dump(audit_report, f, indent=2)
    print(f"\nAudit complete. Successfully saved report to {out_audit_path}")
    print("=" * 80)
    return audit_report

if __name__ == '__main__':
    run_audit()
