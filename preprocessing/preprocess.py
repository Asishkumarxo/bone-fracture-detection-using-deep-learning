"""
Preprocessing Entry Point
=========================
Executes the patient-cluster, leak-free preprocessing pipeline.

Usage:
    python preprocessing/preprocess.py [--audit_csv dataset_audit.csv] [--output_dir processed]
"""

import os
import sys

# Ensure repository root is on sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from preprocess import main

if __name__ == '__main__':
    main()
