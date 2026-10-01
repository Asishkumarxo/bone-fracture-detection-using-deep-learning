"""
Multi-Task Classifier Evaluation Entry Point
============================================
Evaluates model on the test partition, computing accuracy, sensitivity,
specificity, macro-F1, ROC-AUC, and generating confusion matrices.

Usage:
    python evaluation/evaluate.py [--checkpoint best_model.pt] [--test_csv test.csv]
"""

import os
import sys
import argparse

# Ensure repository root is on sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from evaluate import evaluate_test_set

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate Multi-Task Model on Test Set')
    parser.add_argument('--checkpoint', type=str, default='best_model.pt')
    parser.add_argument('--test_csv', type=str, default='test.csv')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_test_samples', type=int, default=None)
    parser.add_argument('--output_metrics', type=str, default='metrics.json')
    parser.add_argument('--output_cm', type=str, default='confusion_matrix.png')
    parser.add_argument('--output_report', type=str, default='classification_report.csv')
    args = parser.parse_args()
    
    evaluate_test_set(args)
