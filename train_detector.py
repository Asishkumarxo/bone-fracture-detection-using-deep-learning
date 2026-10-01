"""
FracAtlas Object Detection Training Script
==========================================
Trains a Faster R-CNN detector to localize bone fractures using bounding box annotations.
"""

import os
import sys
import argparse
import time
import torch
import torchvision.models.detection as detection
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import DataLoader, Subset

from src.detection_dataset import FracAtlasDetectionDataset, detection_collate_fn

def create_detector_model(pretrained=True):
    weights = detection.FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT if pretrained else None
    model = detection.fasterrcnn_mobilenet_v3_large_fpn(weights=weights)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # 2 classes: 0 = background, 1 = fracture
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=2)
    return model

def train_detector(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Training detector on device: {device}')
    
    # 1. Datasets
    print('Loading FracAtlas train and validation datasets...')
    train_dataset = FracAtlasDetectionDataset(
        split_csv=args.train_split,
        annotations_csv=args.annotations_csv,
        target_size=(args.img_size, args.img_size),
        is_train=True
    )
    val_dataset = FracAtlasDetectionDataset(
        split_csv=args.val_split,
        annotations_csv=args.annotations_csv,
        target_size=(args.img_size, args.img_size),
        is_train=False
    )
    
    # Subsampling if requested for fast CPU training
    if args.max_train_samples and args.max_train_samples < len(train_dataset):
        print(f'Subsampling {args.max_train_samples} samples from training set ({len(train_dataset)} total)...')
        indices = list(range(args.max_train_samples))
        train_dataset = Subset(train_dataset, indices)
        
    if args.max_val_samples and args.max_val_samples < len(val_dataset):
        print(f'Subsampling {args.max_val_samples} samples from validation set ({len(val_dataset)} total)...')
        indices = list(range(args.max_val_samples))
        val_dataset = Subset(val_dataset, indices)
        
    print(f'Active training samples: {len(train_dataset)}')
    print(f'Active validation samples: {len(val_dataset)}')
    
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=0, collate_fn=detection_collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=0, collate_fn=detection_collate_fn
    )
    
    # 2. Model & Optimizer
    model = create_detector_model(pretrained=True)
    model.to(device)
    
    # Freeze lower backbone, train FPN and RoI heads
    for name, param in model.backbone.body.named_parameters():
        if 'layer' not in name and 'features.1' not in name:
            param.requires_grad = False
            
    params = [p for p in model.parameters() if p.requires_grad]
    print(f'Trainable detector parameters: {sum(p.numel() for p in params):,}')
    
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)
    
    best_val_loss = float('inf')
    
    print('\n======================================================')
    print('STARTING OBJECT DETECTION TRAINING (FracAtlas BBoxes)')
    print('======================================================')
    
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        epoch_loss = 0.0
        loss_dict_agg = {}
        
        for i, (images, targets) in enumerate(train_loader):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
            
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            optimizer.zero_grad()
            losses.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=5.0)
            optimizer.step()
            
            epoch_loss += losses.item()
            for k, v in loss_dict.items():
                loss_dict_agg[k] = loss_dict_agg.get(k, 0.0) + v.item()
                
        lr_scheduler.step()
        train_loss = epoch_loss / len(train_loader)
        t_epoch = time.time() - t0
        
        # Validation loss calculation
        # Faster R-CNN computes loss only in train mode with targets provided
        val_loss = 0.0
        with torch.no_grad():
            for images, targets in val_loader:
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
                v_loss_dict = model(images, targets)
                v_losses = sum(loss for loss in v_loss_dict.values())
                val_loss += v_losses.item()
                
        val_loss /= max(1, len(val_loader))
        
        print(f"Epoch {epoch:02d} ({t_epoch:.1f}s) | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        for k, v in loss_dict_agg.items():
            print(f"    - {k}: {v / len(train_loader):.4f}")
            
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'model_state': model.state_dict(),
                'target_size': (args.img_size, args.img_size),
                'epoch': epoch,
                'val_loss': val_loss
            }, args.output_checkpoint)
            print(f"  --> Saved new best detector checkpoint to {args.output_checkpoint}")
            
    print(f"\nDetector training completed! Best validation loss: {best_val_loss:.4f}")

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
