"""
Inference CLI Entry Point
=========================
Allows running inference directly via:
    python inference/predict.py --image path/to/xray.jpg

This forwards to the core BoneFractureInferencePipeline in predict.py.
"""

import os
import sys

# Ensure repository root is on sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from predict import (
    BoneFractureInferencePipeline,
    run_inference,
    print_terminal_card,
    main,
)

if __name__ == "__main__":
    main()
