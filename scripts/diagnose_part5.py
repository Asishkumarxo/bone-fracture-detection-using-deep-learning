import os
import sys
import time
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.dataset import BoneFractDataset, get_transforms
from src.models import MultiTaskModel
from inference.model_registry import ModelRegistry

def main():
    device = ModelRegistry.get_device()
    print(f"Running Part 5 test set evaluation on device: {device}")
    
    test_csv = 'test.csv'
    test_df = pd.read_csv(test_csv)
    print(f"Total test set samples: {len(test_df)}")

    # Load checkpoint
    ckpt_path = ModelRegistry.CLASSIFIER_CHECKPOINT_PATH
    ckpt = torch.load(ckpt_path, map_location=device)
    backbone = ckpt.get('backbone', 'resnet50')
    region_to_idx = ckpt.get('region_to_idx', ModelRegistry.REGION_TO_IDX)
    idx_to_region = {idx: reg for reg, idx in region_to_idx.items()}

    model = MultiTaskModel(backbone=backbone, num_regions=len(region_to_idx), pretrained=False)
    model.load_state_dict(ckpt['state_dict'])
    model.to(device)
    model.eval()

    # Create dataset & loader
    test_dataset = BoneFractDataset(test_df, region_to_idx=region_to_idx, transform=get_transforms('test'))
    batch_size = 64 if torch.cuda.is_available() else 32
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    all_image_ids = []
    all_true_regions = []
    all_pred_regions = []
    all_region_confs = []
    all_region_probs = []
    all_true_fractures = []
    all_pred_fractures = []

    t0 = time.time()
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            images = batch['image'].to(device)
            region_targets = batch['region_target'].numpy()
            fracture_targets = batch['fracture_target'].numpy()

            reg_logits, frac_logits = model(images)
            reg_probs = torch.softmax(reg_logits, dim=-1).cpu().numpy()
            frac_probs = torch.sigmoid(frac_logits).cpu().numpy().flatten()

            pred_regions = np.argmax(reg_probs, axis=-1)
            reg_confs = np.max(reg_probs, axis=-1)
            pred_fractures = (frac_probs >= ModelRegistry.FRACTURE_THRESHOLD).astype(bool)

            for i in range(len(images)):
                idx_in_df = batch_idx * batch_size + i
                img_path = test_df.iloc[idx_in_df]['image_path']
                img_id = os.path.basename(img_path)

                all_image_ids.append(img_id)
                all_true_regions.append(idx_to_region[region_targets[i]])
                all_pred_regions.append(idx_to_region[pred_regions[i]])
                all_region_confs.append(float(round(reg_confs[i], 4)))
                all_region_probs.append(reg_probs[i])
                all_true_fractures.append(bool(fracture_targets[i] == 1))
                all_pred_fractures.append(bool(pred_fractures[i]))

    elapsed = time.time() - t0
    print(f"Inference completed in {elapsed:.2f}s ({len(test_df)/elapsed:.1f} img/s)")

    # Build exact requested columns:
    # image_id, true_region, predicted_region, region_confidence,
    # probability_arm, probability_wrist, probability_hand, probability_pelvis,
    # probability_thigh, probability_lower_leg, probability_foot,
    # true_fracture, predicted_fracture
    probs_arr = np.array(all_region_probs)
    
    # Map each column to corresponding index
    # 'Arm': 0, 'Foot': 1, 'Hand': 2, 'Lower leg': 3, 'Thigh': 4, 'pelvis': 5, 'wrist': 6
    out_df = pd.DataFrame({
        'image_id': all_image_ids,
        'true_region': all_true_regions,
        'predicted_region': all_pred_regions,
        'region_confidence': all_region_confs,
        'probability_arm': np.round(probs_arr[:, region_to_idx['Arm']], 4),
        'probability_wrist': np.round(probs_arr[:, region_to_idx['wrist']], 4),
        'probability_hand': np.round(probs_arr[:, region_to_idx['Hand']], 4),
        'probability_pelvis': np.round(probs_arr[:, region_to_idx['pelvis']], 4),
        'probability_thigh': np.round(probs_arr[:, region_to_idx['Thigh']], 4),
        'probability_lower_leg': np.round(probs_arr[:, region_to_idx['Lower leg']], 4),
        'probability_foot': np.round(probs_arr[:, region_to_idx['Foot']], 4),
        'true_fracture': all_true_fractures,
        'predicted_fracture': all_pred_fractures
    })

    os.makedirs('reports', exist_ok=True)
    out_path = 'reports/test_predictions.csv'
    out_df.to_csv(out_path, index=False)
    print(f"Saved {out_path} with {len(out_df)} rows and columns: {list(out_df.columns)}")

if __name__ == '__main__':
    main()
