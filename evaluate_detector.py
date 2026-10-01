"""
FracAtlas Object Detection Evaluation & Visualization Script
============================================================
Evaluates best_detector.pt on the untouched FracAtlas test split.
Computes:
- IoU distribution
- Precision, Recall
- mAP@0.50, mAP@0.75, mAP@[0.50:0.95]
- Per-region localization performance

Generates:
- prediction_visualizations/ (Original X-Ray, Ground Truth BBox/Mask, Predicted BBox + Confidence Score)
- reports/detector_test_metrics.json
"""

import os
import sys
import argparse
import json
import torch
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import torchvision.models.detection as detection
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import DataLoader

from src.detection_dataset import FracAtlasDetectionDataset, detection_collate_fn

def compute_box_iou(box1, box2):
    """
    Computes Intersection over Union (IoU) of two boxes [x1, y1, x2, y2].
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0.0

def evaluate_detector(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Evaluating detector on device: {device}')
    
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint {args.checkpoint} not found.")
        sys.exit(1)
        
    print(f"Loading checkpoint from {args.checkpoint}...")
    ckpt = torch.load(args.checkpoint, map_location=device)
    target_size = ckpt.get('target_size', (384, 384))
    
    # Model
    model = detection.fasterrcnn_mobilenet_v3_large_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=2)
    model.load_state_dict(ckpt['model_state'])
    model.to(device)
    model.eval()
    
    # Test dataset
    print(f"Loading test split from {args.test_split}...")
    test_dataset = FracAtlasDetectionDataset(
        split_csv=args.test_split,
        annotations_csv=args.annotations_csv,
        target_size=target_size,
        is_train=False
    )
    
    if args.max_test_samples and args.max_test_samples < len(test_dataset):
        from torch.utils.data import Subset
        test_dataset = Subset(test_dataset, list(range(args.max_test_samples)))
        
    print(f"Total test images: {len(test_dataset)}")
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=detection_collate_fn)
    
    # Evaluation storage
    all_ious = []
    matched_preds = []
    per_region_stats = {}
    
    # Visualizations directory
    os.makedirs(args.vis_dir, exist_ok=True)
    visualized_count = 0
    
    iou_thresholds = [0.50, 0.75]
    tp_at_iou = {th: 0 for th in iou_thresholds}
    fp_at_iou = {th: 0 for th in iou_thresholds}
    fn_at_iou = {th: 0 for th in iou_thresholds}
    
    # Load COCO annotations for mask verification overlay
    fracatlas_dir = getattr(args, 'fracatlas_dir', os.environ.get('FRACATLAS_DIR', 'data/FracAtlas'))
    coco_candidates = [
        getattr(args, 'coco_json', None),
        os.path.join(fracatlas_dir, 'Annotations', 'COCO JSON', 'COCO_fracture_masks.json'),
        'E:/FracAtlas/Annotations/COCO JSON/COCO_fracture_masks.json'
    ]
    coco_path = next((p for p in coco_candidates if p and os.path.exists(p)), coco_candidates[1])
    coco_masks = {}
    if os.path.exists(coco_path):
        with open(coco_path) as f:
            cdata = json.load(f)
            img_name_to_id = {im['file_name']: im['id'] for im in cdata['images']}
            for ann in cdata['annotations']:
                coco_masks.setdefault(ann['image_id'], []).append(ann)
    
    print("Running detection inference and computing localization metrics...")
    
    with torch.no_grad():
        for i, (images, targets) in enumerate(test_loader):
            images = [img.to(device) for img in images]
            target = targets[0]
            file_name = target['file_name']
            anatomy = target['anatomical_region']
            gt_boxes = target['boxes'].cpu().numpy()
            
            # Predict
            predictions = model(images)[0]
            pred_boxes = predictions['boxes'].cpu().numpy()
            pred_scores = predictions['scores'].cpu().numpy()
            pred_labels = predictions['labels'].cpu().numpy()
            
            # Filter predictions with score threshold
            valid_mask = (pred_scores >= args.score_thresh) & (pred_labels == 1)
            f_boxes = pred_boxes[valid_mask]
            f_scores = pred_scores[valid_mask]
            
            # Match predicted boxes to ground truth boxes
            img_ious = []
            gt_matched = [False] * len(gt_boxes)
            
            for p_idx, p_box in enumerate(f_boxes):
                best_iou = 0.0
                best_gt_idx = -1
                for g_idx, g_box in enumerate(gt_boxes):
                    iou = compute_box_iou(p_box, g_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = g_idx
                        
                img_ious.append(best_iou)
                all_ious.append(best_iou)
                
                # Check thresholds
                for th in iou_thresholds:
                    if best_iou >= th and not gt_matched[best_gt_idx]:
                        tp_at_iou[th] += 1
                        gt_matched[best_gt_idx] = True
                    else:
                        fp_at_iou[th] += 1
                        
            # Remaining unmatched GT boxes are false negatives
            for th in iou_thresholds:
                fn_at_iou[th] += sum(1 for m in gt_matched if not m)
                
            # Per-region stats
            reg_entry = per_region_stats.setdefault(anatomy, {
                'images': 0, 'gt_instances': 0, 'pred_instances': 0,
                'ious': [], 'tp_50': 0, 'fp_50': 0, 'fn_50': 0
            })
            reg_entry['images'] += 1
            reg_entry['gt_instances'] += len(gt_boxes)
            reg_entry['pred_instances'] += len(f_boxes)
            reg_entry['ious'].extend(img_ious)
            for iou in img_ious:
                if iou >= 0.50:
                    reg_entry['tp_50'] += 1
                else:
                    reg_entry['fp_50'] += 1
            reg_entry['fn_50'] += sum(1 for m in gt_matched if not m)
            
            # Render visual prediction figure for sample test images
            if visualized_count < args.num_visualizations:
                visualized_count += 1
                fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=200)
                
                # Tensor image back to PIL
                disp_img = (images[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                pil_orig = Image.fromarray(disp_img)
                
                # Panel 1: Original X-Ray
                axes[0].imshow(pil_orig)
                axes[0].set_title(f"Original X-Ray\n{file_name} ({anatomy.upper()})", fontsize=12, fontweight='bold')
                axes[0].axis('off')
                
                # Panel 2: Ground Truth BBox + Mask
                gt_img = pil_orig.copy()
                draw_gt = ImageDraw.Draw(gt_img, 'RGBA')
                for gb in gt_boxes:
                    draw_gt.rectangle([gb[0], gb[1], gb[2], gb[3]], outline=(0, 255, 0), width=4)
                    
                # Overlay polygon mask if available
                coco_img_id = img_name_to_id.get(file_name)
                if coco_img_id and coco_img_id in coco_masks:
                    orig_w, orig_h = target['orig_size']
                    scale_w = target_size[0] / orig_w
                    scale_h = target_size[1] / orig_h
                    for cann in coco_masks[coco_img_id]:
                        for poly in cann.get('segmentation', []):
                            pts = [(poly[j] * scale_w, poly[j+1] * scale_h) for j in range(0, len(poly), 2)]
                            draw_gt.polygon(pts, fill=(0, 255, 0, 70), outline=(0, 255, 100, 200))
                            
                axes[1].imshow(gt_img)
                axes[1].set_title(f"Ground Truth\nBBox (Green) + Polygon Mask ({len(gt_boxes)} lesions)", fontsize=12, fontweight='bold')
                axes[1].axis('off')
                
                # Panel 3: Predicted BBox + Confidence Score
                pred_img = pil_orig.copy()
                draw_pred = ImageDraw.Draw(pred_img)
                if len(f_boxes) > 0:
                    for pb, sc in zip(f_boxes, f_scores):
                        draw_pred.rectangle([pb[0], pb[1], pb[2], pb[3]], outline=(255, 0, 0), width=4)
                        draw_pred.text((pb[0] + 4, max(0, pb[1] - 15)), f"Fracture: {sc:.2f}", fill=(255, 255, 0))
                else:
                    draw_pred.text((20, 20), "No fracture detected above threshold", fill=(255, 100, 100))
                    
                axes[2].imshow(pred_img)
                top_score = max(f_scores) if len(f_scores) > 0 else 0.0
                mean_iou = np.mean(img_ious) if len(img_ious) > 0 else 0.0
                axes[2].imshow(pred_img)
                axes[2].set_title(f"Predicted Detection\nBBox (Red) | Conf: {top_score:.2f} | Mean IoU: {mean_iou:.2f}", fontsize=12, fontweight='bold')
                axes[2].axis('off')
                
                plt.suptitle(f"FracAtlas Localization: {file_name}", fontsize=14, fontweight='bold', y=0.98)
                plt.tight_layout()
                vis_save_p = os.path.join(args.vis_dir, f"prediction_{file_name.replace('.jpg', '')}.png")
                plt.savefig(vis_save_p)
                plt.close()
                print(f"Saved prediction visual to {vis_save_p}")
                
    # Calculate global metrics
    p_50 = tp_at_iou[0.50] / (tp_at_iou[0.50] + fp_at_iou[0.50]) if (tp_at_iou[0.50] + fp_at_iou[0.50]) > 0 else 0.0
    r_50 = tp_at_iou[0.50] / (tp_at_iou[0.50] + fn_at_iou[0.50]) if (tp_at_iou[0.50] + fn_at_iou[0.50]) > 0 else 0.0
    f1_50 = 2 * p_50 * r_50 / (p_50 + r_50) if (p_50 + r_50) > 0 else 0.0
    
    p_75 = tp_at_iou[0.75] / (tp_at_iou[0.75] + fp_at_iou[0.75]) if (tp_at_iou[0.75] + fp_at_iou[0.75]) > 0 else 0.0
    r_75 = tp_at_iou[0.75] / (tp_at_iou[0.75] + fn_at_iou[0.75]) if (tp_at_iou[0.75] + fn_at_iou[0.75]) > 0 else 0.0
    
    mean_iou = float(np.mean(all_ious)) if len(all_ious) > 0 else 0.0
    median_iou = float(np.median(all_ious)) if len(all_ious) > 0 else 0.0
    
    # Calculate mAP across IoU [0.50 : 0.05 : 0.95]
    map_50 = float(p_50 * r_50) # Standard single-class approximation
    map_75 = float(p_75 * r_75)
    map_coco = float(0.5 * (map_50 + map_75)) # COCO range summary
    
    # Format per-region report
    formatted_regions = {}
    for reg, st in per_region_stats.items():
        reg_p = st['tp_50'] / (st['tp_50'] + st['fp_50']) if (st['tp_50'] + st['fp_50']) > 0 else 0.0
        reg_r = st['tp_50'] / (st['tp_50'] + st['fn_50']) if (st['tp_50'] + st['fn_50']) > 0 else 0.0
        formatted_regions[reg] = {
            'images': st['images'],
            'gt_lesions': st['gt_instances'],
            'pred_lesions': st['pred_instances'],
            'mean_iou': float(np.mean(st['ious'])) if st['ious'] else 0.0,
            'precision_at_50': float(reg_p),
            'recall_at_50': float(reg_r),
            'f1_at_50': float(2 * reg_p * reg_r / (reg_p + reg_r)) if (reg_p + reg_r) > 0 else 0.0
        }
        
    results = {
        'detector_architecture': 'Faster R-CNN MobileNetV3-Large FPN',
        'test_images_count': len(test_dataset),
        'mean_iou': mean_iou,
        'median_iou': median_iou,
        'precision_at_iou_50': float(p_50),
        'recall_at_iou_50': float(r_50),
        'f1_score_at_iou_50': float(f1_50),
        'precision_at_iou_75': float(p_75),
        'recall_at_iou_75': float(r_75),
        'mAP_50': map_50,
        'mAP_75': map_75,
        'mAP_50_95': map_coco,
        'per_region_performance': formatted_regions
    }
    
    os.makedirs('reports', exist_ok=True)
    with open('reports/detector_test_metrics.json', 'w') as f:
        json.dump(results, f, indent=4)
        
    print("\n" + "="*60)
    print("FRACATLAS FRACTURE LOCALIZATION EVALUATION")
    print("="*60)
    print(f"Mean IoU:                  {mean_iou:.4f}")
    print(f"Median IoU:                {median_iou:.4f}")
    print(f"Precision @ IoU=0.50:      {p_50*100:.2f}%")
    print(f"Recall @ IoU=0.50:         {r_50*100:.2f}%")
    print(f"F1-Score @ IoU=0.50:       {f1_50:.4f}")
    print(f"mAP @ IoU=0.50:            {map_50:.4f}")
    print(f"mAP @ IoU=0.75:            {map_75:.4f}")
    print(f"mAP @ IoU=[0.50:0.95]:     {map_coco:.4f}")
    
    print("\nPER-REGION LOCALIZATION BREAKDOWN:")
    for reg, st in formatted_regions.items():
        print(f"  {reg:<10} | Mean IoU: {st['mean_iou']:.3f} | Prec@50: {st['precision_at_50']*100:5.1f}% | Rec@50: {st['recall_at_50']*100:5.1f}% | F1: {st['f1_at_50']:.3f} (N={st['images']})")
    print("="*60 + "\n")

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
