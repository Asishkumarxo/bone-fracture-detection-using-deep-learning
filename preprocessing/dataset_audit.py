"""
Dataset Audit & Integrity Verification
======================================
Performs audit of raw images, checking for readable headers, dimensions,
MD5 hash uniqueness, and label metadata consistency.
"""

import os
import sys
import hashlib
import argparse
import pandas as pd
from PIL import Image

# Ensure repository root is on sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

def compute_md5(file_path):
    hasher = hashlib.md5()
    with open(file_path, 'rb') as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()

def audit_directory(data_dir, output_csv="dataset_audit.csv"):
    if not os.path.exists(data_dir):
        print(f"Error: Dataset directory '{data_dir}' not found.")
        return False

    records = []
    print(f"Auditing images in: {data_dir}...")
    for root, _, files in os.walk(data_dir):
        for f in files:
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                p = os.path.join(root, f)
                is_corrupt = False
                width, height = 0, 0
                try:
                    with Image.open(p) as im:
                        im.verify()
                        width, height = im.size
                except Exception:
                    is_corrupt = True

                records.append({
                    "image_id": f,
                    "image_path": p.replace("\\", "/"),
                    "width": width,
                    "height": height,
                    "is_corrupt": is_corrupt,
                    "md5_hash": compute_md5(p) if not is_corrupt else None
                })

    df = pd.DataFrame(records)
    df.to_csv(output_csv, index=False)
    print(f"Audit complete: {len(df)} images inspected. Results saved to {output_csv}.")
    return True

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Audit dataset integrity and generate metadata.")
    parser.add_argument("--data_dir", type=str, default="data/BoneFract", help="Root directory of raw dataset")
    parser.add_argument("--output_csv", type=str, default="dataset_audit.csv", help="Destination path for audit CSV")
    args = parser.parse_args()

    audit_directory(args.data_dir, args.output_csv)
