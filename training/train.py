"""
Multi-Task Model Training Entry Point
=====================================
Usage:
    python training/train.py [--backbone resnet50] [--epochs_stage1 3] [--epochs_stage2 2]
"""

import os
import sys
import argparse

# Ensure repository root is on sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from train import train_baseline

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Multi-Task Baseline Model')
    parser.add_argument('--backbone', type=str, default='resnet50', choices=['resnet18', 'resnet50', 'efficientnet_b0'])
    parser.add_argument('--epochs_stage1', type=int, default=3)
    parser.add_argument('--epochs_stage2', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr_stage1', type=float, default=1e-3)
    parser.add_argument('--lr_stage2', type=float, default=1e-4)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--train_csv', type=str, default='train.csv')
    parser.add_argument('--val_csv', type=str, default='validation.csv')
    parser.add_argument('--max_train_samples', type=int, default=2000, help='Max train samples for CPU efficiency')
    parser.add_argument('--max_val_samples', type=int, default=500, help='Max val samples for CPU efficiency')
    args = parser.parse_args()
    
    train_baseline(args)
