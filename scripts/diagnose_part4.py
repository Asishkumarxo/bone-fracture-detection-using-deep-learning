import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
from inference.model_registry import ModelRegistry

def main():
    print("=== CLASSIFIER CHECKPOINT VERIFICATION ===")
    ckpt_path = ModelRegistry.CLASSIFIER_CHECKPOINT_PATH
    ckpt = torch.load(ckpt_path, map_location='cpu')

    print(f"Checkpoint Path:               {ckpt_path}")
    print(f"Model Architecture (backbone): {ckpt.get('backbone', 'resnet50')}")
    reg_mapping = ckpt.get('region_to_idx', {})
    print(f"Number of Region Classes:      {len(reg_mapping)}")
    print(f"Region Classes:                {list(reg_mapping.keys())}")
    print(f"Saved Epoch:                   {ckpt.get('epoch', 'N/A')}")
    print(f"Best Validation Loss:          {ckpt.get('val_loss', 'N/A')}")
    metrics = ckpt.get('metrics', {})
    print(f"Validation Macro F1 (Region):  {metrics.get('region_macro_f1', 'N/A')}")
    print(f"Validation Fracture ROC-AUC:   {metrics.get('fracture_roc_auc', 'N/A')}")
    print(f"Validation Composite Score:    {ckpt.get('composite_score', 'N/A')}")

    print("\n=== DETECTOR CHECKPOINT VERIFICATION ===")
    det_path = ModelRegistry.DETECTOR_CHECKPOINT_PATH
    det_ckpt = torch.load(det_path, map_location='cpu')
    print(f"Detector Checkpoint Path:      {det_path}")
    print(f"Detector Architecture:         {det_ckpt.get('architecture', ModelRegistry.DETECTOR_ARCHITECTURE)}")
    print(f"Number of Detector Classes:    {ModelRegistry.DETECTOR_NUM_CLASSES} (0=Background, 1=Fracture)")
    print(f"Target Input Size:             {det_ckpt.get('target_size', ModelRegistry.DETECTOR_TARGET_SIZE)}")
    print(f"Saved Epoch:                   {det_ckpt.get('epoch', 'N/A')}")
    print(f"Validation Loss:               {det_ckpt.get('val_loss', 'N/A')}")
    det_metrics = det_ckpt.get('metrics', {})
    print(f"Validation mAP @ 0.5:          {det_metrics.get('map_50', det_ckpt.get('val_map', 'N/A'))}")

if __name__ == '__main__':
    main()
