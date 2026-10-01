"""
Fracture Detector Evaluation Entry Point
========================================
Evaluates spatial lesion detector on test bounding boxes, computing
mAP@0.50, mAP@0.75, and per-region localization performance.

Usage:
    python evaluation/evaluate_detector.py [--checkpoint best_detector.pt]
"""

import os
import sys
import argparse

# Ensure repository root is on sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from evaluate_detector import evaluate_detector

if __name__ == '__main__':
    fracatlas_dir = os.environ.get('FRACATLAS_DIR', 'data/FracAtlas')
    default_test = os.path.join(fracatlas_dir, 'Utilities', 'Fracture Split', 'test.csv')
    if not os.path.exists(default_test) and os.path.exists('E:/FracAtlas/Utilities/Fracture Split/test.csv'):
        default_test = 'E:/FracAtlas/Utilities/Fracture Split/test.csv'

    parser = argparse.ArgumentParser(description='Evaluate Fracture Detector on FracAtlas')
    parser.add_argument('--fracatlas_dir', type=str, default=fracatlas_dir, help='Base directory for FracAtlas dataset')
    parser.add_argument('--checkpoint', type=str, default='best_detector.pt')
    parser.add_argument('--test_split', type=str, default=default_test)
    parser.add_argument('--annotations_csv', type=str, default='reports/fracatlas_annotations_clean.csv')
    parser.add_argument('--score_thresh', type=float, default=0.25)
    parser.add_argument('--max_test_samples', type=int, default=30)
    parser.add_argument('--vis_dir', type=str, default='prediction_visualizations')
    parser.add_argument('--num_visualizations', type=int, default=5)
    args = parser.parse_args()
    
    evaluate_detector(args)
