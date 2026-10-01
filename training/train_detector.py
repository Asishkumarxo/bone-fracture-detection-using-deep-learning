"""
Fracture Detector Training Entry Point
======================================
Usage:
    python training/train_detector.py [--img_size 384] [--epochs 5] [--batch_size 4]
"""

import os
import sys
import argparse

# Ensure repository root is on sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from train_detector import train_detector

if __name__ == '__main__':
    fracatlas_dir = os.environ.get('FRACATLAS_DIR', 'data/FracAtlas')
    default_train = os.path.join(fracatlas_dir, 'Utilities', 'Fracture Split', 'train.csv')
    if not os.path.exists(default_train) and os.path.exists('E:/FracAtlas/Utilities/Fracture Split/train.csv'):
        default_train = 'E:/FracAtlas/Utilities/Fracture Split/train.csv'
        
    default_val = os.path.join(fracatlas_dir, 'Utilities', 'Fracture Split', 'valid.csv')
    if not os.path.exists(default_val) and os.path.exists('E:/FracAtlas/Utilities/Fracture Split/valid.csv'):
        default_val = 'E:/FracAtlas/Utilities/Fracture Split/valid.csv'

    parser = argparse.ArgumentParser(description='Train Fracture Object Detector on FracAtlas')
    parser.add_argument('--fracatlas_dir', type=str, default=fracatlas_dir, help='Base directory for FracAtlas dataset')
    parser.add_argument('--train_split', type=str, default=default_train, help='Path to train split CSV')
    parser.add_argument('--val_split', type=str, default=default_val, help='Path to validation split CSV')
    parser.add_argument('--annotations_csv', type=str, default='reports/fracatlas_annotations_clean.csv')
    parser.add_argument('--img_size', type=int, default=384, help='Image resolution')
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--max_train_samples', type=int, default=80)
    parser.add_argument('--max_val_samples', type=int, default=20)
    parser.add_argument('--output_checkpoint', type=str, default='best_detector.pt')
    args = parser.parse_args()
    
    train_detector(args)
