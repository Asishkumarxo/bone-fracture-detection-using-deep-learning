"""
Multi-Task Baseline Model Training Script
=========================================
Trains a shared-backbone deep learning model for:
1. Anatomical-region classification (Multi-class)
2. Fracture presence detection (Binary)

Features:
- Two-stage transfer learning (Stage 1: frozen backbone, Stage 2: unfrozen upper layers)
- Class imbalance mitigation via pos_weight weighted BCE loss
- Learning rate scheduling, early stopping, and model checkpointing
- Full training history logging to training_history.csv
- Saves best_model.pt
"""

import os
import sys
import argparse
import time
import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR

from src.dataset import BoneFractDataset, get_transforms
from src.models import MultiTaskModel
from src.metrics import compute_all_metrics

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def evaluate_model(model, dataloader, criterion_region, criterion_fracture, device, idx_to_region):
    model.eval()
    total_loss = 0.0
    total_loss_r = 0.0
    total_loss_f = 0.0
    
    all_region_targets = []
    all_region_preds = []
    all_region_probs = []
    all_fracture_targets = []
    all_fracture_preds = []
    all_fracture_probs = []
    
    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(device)
            region_targets = batch['region_target'].to(device)
            fracture_targets = batch['fracture_target'].to(device)
            
            region_logits, fracture_logits = model(images)
            
            loss_r = criterion_region(region_logits, region_targets)
            loss_f = criterion_fracture(fracture_logits, fracture_targets)
            loss = loss_r + loss_f
            
            total_loss += loss.item() * images.size(0)
            total_loss_r += loss_r.item() * images.size(0)
            total_loss_f += loss_f.item() * images.size(0)
            
            # Region predictions
            probs_r = torch.softmax(region_logits, dim=-1).cpu().numpy()
            preds_r = np.argmax(probs_r, axis=-1)
            
            # Fracture predictions
            probs_f = torch.sigmoid(fracture_logits).cpu().numpy()
            preds_f = (probs_f >= 0.5).astype(int)
            
            all_region_targets.extend(region_targets.cpu().numpy())
            all_region_preds.extend(preds_r)
            all_region_probs.extend(probs_r)
            all_fracture_targets.extend(fracture_targets.cpu().numpy())
            all_fracture_preds.extend(preds_f)
            all_fracture_probs.extend(probs_f)
            
    n_samples = len(dataloader.dataset)
    avg_loss = total_loss / n_samples
    avg_loss_r = total_loss_r / n_samples
    avg_loss_f = total_loss_f / n_samples
    
    metrics = compute_all_metrics(
        all_region_targets, all_region_preds, all_region_probs,
        all_fracture_targets, all_fracture_preds, all_fracture_probs,
        idx_to_region
    )
    
    return avg_loss, avg_loss_r, avg_loss_f, metrics

def train_baseline(args):
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Training on execution device: {device}')
    
    # 1. Load Data
    print(f'Loading datasets: {args.train_csv}, {args.val_csv}')
    train_df = pd.read_csv(args.train_csv)
    val_df = pd.read_csv(args.val_csv)
    
    # Subsampling if requested (useful for CPU environments)
    from sklearn.model_selection import train_test_split
    if args.max_train_samples and args.max_train_samples < len(train_df):
        print(f'Stratified subsampling {args.max_train_samples} samples from training set ({len(train_df)} total)...')
        strat_t = train_df['anatomical_region'] + '_' + train_df['fracture_label']
        train_df, _ = train_test_split(train_df, train_size=args.max_train_samples, stratify=strat_t, random_state=args.seed)
        train_df = train_df.reset_index(drop=True)
        
    if args.max_val_samples and args.max_val_samples < len(val_df):
        print(f'Stratified subsampling {args.max_val_samples} samples from validation set ({len(val_df)} total)...')
        strat_v = val_df['anatomical_region'] + '_' + val_df['fracture_label']
        val_df, _ = train_test_split(val_df, train_size=args.max_val_samples, stratify=strat_v, random_state=args.seed)
        val_df = val_df.reset_index(drop=True)

    print(f'Active training samples: {len(train_df):,}')
    print(f'Active validation samples: {len(val_df):,}')
    
    # 2. Inspect Training Class Imbalance
    n_neg = (train_df['fracture_label'] == 'Negative').sum()
    n_pos = (train_df['fracture_label'] == 'Positive').sum()
    pos_weight_val = float(n_neg / n_pos) if n_pos > 0 else 1.0
    print(f'Class balance in training: Negative={n_neg:,}, Positive={n_pos:,} (Ratio = {pos_weight_val:.3f})')
    pos_weight = torch.tensor([pos_weight_val], dtype=torch.float32).to(device)
    
    # 3. Create Datasets & Loaders
    train_dataset = BoneFractDataset(train_df, transform=get_transforms('train'))
    val_dataset = BoneFractDataset(val_df, region_to_idx=train_dataset.region_to_idx, transform=get_transforms('validation'))
    
    num_regions = len(train_dataset.region_to_idx)
    idx_to_region = train_dataset.idx_to_region
    print(f'Anatomical regions detected ({num_regions}): {list(train_dataset.region_to_idx.keys())}')
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    # 4. Model & Loss Criteria
    model = MultiTaskModel(backbone=args.backbone, num_regions=num_regions, pretrained=True, dropout=args.dropout)
    model.to(device)
    
    criterion_region = nn.CrossEntropyLoss()
    criterion_fracture = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    # 5. Tracking structures
    history = []
    best_val_loss = float('inf')
    best_val_score = 0.0
    best_epoch = -1
    
    # --- STAGE 1: Train Task Heads with Frozen Backbone ---
    print('\n======================================================')
    print('STAGE 1: Training Task-Specific Heads (Frozen Backbone)')
    print('======================================================')
    model.freeze_backbone()
    trainable_params_s1 = [p for p in model.parameters() if p.requires_grad]
    print(f'Stage 1 Trainable parameters: {sum(p.numel() for p in trainable_params_s1):,}')
    
    optimizer_s1 = AdamW(trainable_params_s1, lr=args.lr_stage1, weight_decay=1e-4)
    scheduler_s1 = CosineAnnealingLR(optimizer_s1, T_max=args.epochs_stage1, eta_min=1e-5)
    
    current_epoch = 0
    for ep in range(1, args.epochs_stage1 + 1):
        current_epoch += 1
        t_ep0 = time.time()
        model.train()
        train_loss, train_loss_r, train_loss_f = 0.0, 0.0, 0.0
        
        for batch in train_loader:
            images = batch['image'].to(device)
            region_targets = batch['region_target'].to(device)
            fracture_targets = batch['fracture_target'].to(device)
            
            optimizer_s1.zero_grad()
            region_logits, fracture_logits = model(images)
            
            loss_r = criterion_region(region_logits, region_targets)
            loss_f = criterion_fracture(fracture_logits, fracture_targets)
            loss = loss_r + loss_f
            
            loss.backward()
            optimizer_s1.step()
            
            train_loss += loss.item() * images.size(0)
            train_loss_r += loss_r.item() * images.size(0)
            train_loss_f += loss_f.item() * images.size(0)
            
        scheduler_s1.step()
        
        # Epoch metrics
        train_loss /= len(train_dataset)
        train_loss_r /= len(train_dataset)
        train_loss_f /= len(train_dataset)
        
        val_loss, val_loss_r, val_loss_f, val_metrics = evaluate_model(
            model, val_loader, criterion_region, criterion_fracture, device, idx_to_region
        )
        
        t_ep = time.time() - t_ep0
        val_reg_acc = val_metrics['anatomical_classification']['accuracy']
        val_reg_f1 = val_metrics['anatomical_classification']['macro_f1']
        val_frac_acc = val_metrics['fracture_detection']['accuracy']
        val_frac_f1 = val_metrics['fracture_detection']['f1_score']
        val_frac_auc = val_metrics['fracture_detection']['roc_auc']
        combined_score = 0.5 * val_reg_f1 + 0.5 * val_frac_f1
        
        print(f"Epoch {current_epoch:02d} [Stage 1] ({t_ep:.1f}s) | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"Region Acc: {val_reg_acc:.3f}, F1: {val_reg_f1:.3f} | Frac Acc: {val_frac_acc:.3f}, F1: {val_frac_f1:.3f}, AUC: {val_frac_auc:.3f}")
              
        history_entry = {
            'epoch': current_epoch,
            'stage': 'Stage 1 (Heads Only)',
            'train_loss': train_loss,
            'train_loss_region': train_loss_r,
            'train_loss_fracture': train_loss_f,
            'val_loss': val_loss,
            'val_loss_region': val_loss_r,
            'val_loss_fracture': val_loss_f,
            'val_region_accuracy': val_reg_acc,
            'val_region_macro_f1': val_reg_f1,
            'val_fracture_accuracy': val_frac_acc,
            'val_fracture_f1': val_frac_f1,
            'val_fracture_auc': val_frac_auc,
            'val_fracture_recall': val_metrics['fracture_detection']['recall_sensitivity'],
            'val_fracture_specificity': val_metrics['fracture_detection']['specificity'],
            'combined_score': combined_score
        }
        history.append(history_entry)
        
        # Checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_score = combined_score
            best_epoch = current_epoch
            save_checkpoint(model, train_dataset.region_to_idx, args.backbone, pos_weight_val, 'best_model.pt')
            print(f"  --> Checkpoint saved: New lowest validation loss ({val_loss:.4f})")
            
    # --- STAGE 2: Fine-Tuning Upper Backbone ---
    if args.epochs_stage2 > 0:
        print('\n======================================================')
        print('STAGE 2: Fine-Tuning Upper Backbone (layer4 + Heads)')
        print('======================================================')
        model.unfreeze_upper_backbone()
        trainable_params_s2 = [p for p in model.parameters() if p.requires_grad]
        print(f'Stage 2 Trainable parameters: {sum(p.numel() for p in trainable_params_s2):,}')
        
        # Differential learning rates: smaller for backbone, moderate for heads
        backbone_params = [p for p in model.layer4.parameters() if p.requires_grad]
        head_params = list(model.region_head.parameters()) + list(model.fracture_head.parameters())
        
        optimizer_s2 = AdamW([
            {'params': backbone_params, 'lr': args.lr_stage2 * 0.5},
            {'params': head_params, 'lr': args.lr_stage2}
        ], weight_decay=1e-4)
        
        scheduler_s2 = ReduceLROnPlateau(optimizer_s2, mode='min', factor=0.5, patience=1)
        
        for ep in range(1, args.epochs_stage2 + 1):
            current_epoch += 1
            t_ep0 = time.time()
            model.train()
            train_loss, train_loss_r, train_loss_f = 0.0, 0.0, 0.0
            
            for batch in train_loader:
                images = batch['image'].to(device)
                region_targets = batch['region_target'].to(device)
                fracture_targets = batch['fracture_target'].to(device)
                
                optimizer_s2.zero_grad()
                region_logits, fracture_logits = model(images)
                
                loss_r = criterion_region(region_logits, region_targets)
                loss_f = criterion_fracture(fracture_logits, fracture_targets)
                loss = loss_r + loss_f
                
                loss.backward()
                optimizer_s2.step()
                
                train_loss += loss.item() * images.size(0)
                train_loss_r += loss_r.item() * images.size(0)
                train_loss_f += loss_f.item() * images.size(0)
                
            train_loss /= len(train_dataset)
            train_loss_r /= len(train_dataset)
            train_loss_f /= len(train_dataset)
            
            val_loss, val_loss_r, val_loss_f, val_metrics = evaluate_model(
                model, val_loader, criterion_region, criterion_fracture, device, idx_to_region
            )
            scheduler_s2.step(val_loss)
            
            t_ep = time.time() - t_ep0
            val_reg_acc = val_metrics['anatomical_classification']['accuracy']
            val_reg_f1 = val_metrics['anatomical_classification']['macro_f1']
            val_frac_acc = val_metrics['fracture_detection']['accuracy']
            val_frac_f1 = val_metrics['fracture_detection']['f1_score']
            val_frac_auc = val_metrics['fracture_detection']['roc_auc']
            combined_score = 0.5 * val_reg_f1 + 0.5 * val_frac_f1
            
            print(f"Epoch {current_epoch:02d} [Stage 2] ({t_ep:.1f}s) | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                  f"Region Acc: {val_reg_acc:.3f}, F1: {val_reg_f1:.3f} | Frac Acc: {val_frac_acc:.3f}, F1: {val_frac_f1:.3f}, AUC: {val_frac_auc:.3f}")
                  
            history_entry = {
                'epoch': current_epoch,
                'stage': 'Stage 2 (Upper Unfrozen)',
                'train_loss': train_loss,
                'train_loss_region': train_loss_r,
                'train_loss_fracture': train_loss_f,
                'val_loss': val_loss,
                'val_loss_region': val_loss_r,
                'val_loss_fracture': val_loss_f,
                'val_region_accuracy': val_reg_acc,
                'val_region_macro_f1': val_reg_f1,
                'val_fracture_accuracy': val_frac_acc,
                'val_fracture_f1': val_frac_f1,
                'val_fracture_auc': val_frac_auc,
                'val_fracture_recall': val_metrics['fracture_detection']['recall_sensitivity'],
                'val_fracture_specificity': val_metrics['fracture_detection']['specificity'],
                'combined_score': combined_score
            }
            history.append(history_entry)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_score = combined_score
                best_epoch = current_epoch
                save_checkpoint(model, train_dataset.region_to_idx, args.backbone, pos_weight_val, 'best_model.pt')
                print(f"  --> Checkpoint saved: New lowest validation loss ({val_loss:.4f})")

    # 6. Save Training History
    df_history = pd.DataFrame(history)
    df_history.to_csv('training_history.csv', index=False)
    print(f"\nTraining complete! History logged to training_history.csv. Best epoch: {best_epoch} with val_loss: {best_val_loss:.4f}")

def save_checkpoint(model, region_to_idx, backbone, pos_weight_val, path='best_model.pt'):
    checkpoint = {
        'state_dict': model.state_dict(),
        'region_to_idx': region_to_idx,
        'backbone': backbone,
        'pos_weight': pos_weight_val
    }
    tmp_path = path + '.tmp'
    torch.save(checkpoint, tmp_path)
    if os.path.exists(path):
        os.remove(path)
    os.rename(tmp_path, path)
    # Immediate integrity verification
    _ = torch.load(path, map_location='cpu')

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
    parser.add_argument('--max_train_samples', type=int, default=2000, help='Max train samples for CPU baseline efficiency')
    parser.add_argument('--max_val_samples', type=int, default=500, help='Max val samples for CPU baseline efficiency')
    args = parser.parse_args()
    
    train_baseline(args)
