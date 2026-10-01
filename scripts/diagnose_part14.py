"""
Part 14: Localization Ground Truth vs Prediction Comparison
===========================================================
Draws GREEN for ground truth and RED for model prediction on a known FracAtlas image.
Saves to reports/fracatlas_localization_gt_vs_pred.png
"""

import os
import sys
import pandas as pd
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from inference.model_registry import ModelRegistry
from inference.localization_predictor import LocalizationPredictor
from inference.preprocessing import preprocess_for_detector

def main():
    # Use known FracAtlas fractured image with verified ground truth
    clean_csv = 'reports/fracatlas_annotations_clean.csv'
    if not os.path.exists(clean_csv):
        print(f"Error: {clean_csv} not found.")
        return

    df = pd.read_csv(clean_csv)
    # Pick IMG0000092.jpg (verified leg fracture)
    target_img_id = 'IMG0000092.jpg'
    row = df[df['image_id'] == target_img_id].iloc[0]
    
    img_path = row['image_path']
    if not os.path.exists(img_path):
        print(f"Image not found at {img_path}")
        return

    orig_w, orig_h = int(row['width']), int(row['height'])
    gt_box = [round(float(row['x_min']), 1), round(float(row['y_min']), 1),
              round(float(row['x_max']), 1), round(float(row['y_max']), 1)]

    print(f"Testing FracAtlas Image: {target_img_id}")
    print(f"Dimensions: {orig_w} x {orig_h}")
    print(f"Ground Truth BBox (GREEN): {gt_box}")

    # Load predictor with responsive threshold (0.10) to capture model's detection
    device = ModelRegistry.get_device()
    predictor = LocalizationPredictor(score_thresh=0.10, device=device)

    tensor, _ = preprocess_for_detector(img_path)
    tensor = tensor.to(device)

    with torch.no_grad():
        preds = predictor.model(tensor)[0]
        boxes = preds['boxes'].cpu().numpy()
        scores = preds['scores'].cpu().numpy()
        labels = preds['labels'].cpu().numpy()

    # Filter
    scale_w = orig_w / float(predictor.target_size[0])
    scale_h = orig_h / float(predictor.target_size[1])

    valid_mask = (scores >= 0.10) & (labels == 1)
    f_boxes = boxes[valid_mask]
    f_scores = scores[valid_mask]

    pred_boxes = []
    for b, s in zip(f_boxes, f_scores):
        x1 = round(float(b[0] * scale_w), 1)
        y1 = round(float(b[1] * scale_h), 1)
        x2 = round(float(b[2] * scale_w), 1)
        y2 = round(float(b[3] * scale_h), 1)
        pred_boxes.append(([x1, y1, x2, y2], float(s)))

    print(f"Predicted BBoxes (RED): {pred_boxes}")

    # Draw image with PIL
    im = Image.open(img_path).convert('RGB')
    draw = ImageDraw.Draw(im)

    # Line thickness
    line_w = max(4, int(min(orig_w, orig_h) * 0.005))

    # 1. Draw Ground Truth in GREEN
    draw.rectangle(gt_box, outline='#00ff00', width=line_w)
    draw.text((gt_box[0] + 5, max(0, gt_box[1] - 30)), "GT Fracture (GREEN)", fill='#00ff00')

    # 2. Draw Predictions in RED
    for p_box, p_score in pred_boxes:
        draw.rectangle(p_box, outline='#ff0000', width=line_w)
        draw.text((p_box[0] + 5, min(orig_h - 25, p_box[3] + 5)), f"Pred ({p_score*100:.1f}%) [RED]", fill='#ff0000')

    # Save
    os.makedirs('reports', exist_ok=True)
    out_img_path = 'reports/fracatlas_localization_gt_vs_pred.png'
    im.save(out_img_path)
    print(f"Saved annotated image to {out_img_path}")

if __name__ == '__main__':
    main()
