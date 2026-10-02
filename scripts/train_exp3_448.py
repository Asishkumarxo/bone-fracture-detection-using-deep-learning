"""
Experiment 3: Controlled 448x448 High-Resolution Retraining
===========================================================
Trains the MultiTaskModel (ResNet-50 + 7-class Region Head + Binary Fracture Head)
using 448x448 input resolution with aspect-ratio-aware padding.

Constraints:
- Identical architecture (ResNet-50, multi-task)
- Identical loss functions (CrossEntropy + pos_weight BCEWithLogitsLoss)
- Identical augmentation policy (conservative medical jitter/rotation/scale)
- Identical 2-stage transfer learning setup (3 epochs frozen, 2 epochs unfrozen)
- Full leak-free project split: 38,366 train / 4,787 val / 4,778 test
- Target checkpoint: models/checkpoints/exp3_resnet50_448.pt
- baseline model models/checkpoints/best_model.pt remains strictly untouched
"""

import os
import sys
import time
import json
import random
import argparse
import pandas as pd
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
import torchvision.transforms as T
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.models import MultiTaskModel

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def resize_and_pad(img, target_size=(448, 448), fill_color=0):
    """
    Controlled aspect-ratio-preserving resize and zero padding.
    Prevents non-uniform anatomical distortion.
    """
    target_w, target_h = target_size
    orig_w, orig_h = img.size
    if orig_w == target_w and orig_h == target_h:
        return img
    ratio = min(target_w / orig_w, target_h / orig_h)
    new_w = max(1, int(round(orig_w * ratio)))
    new_h = max(1, int(round(orig_h * ratio)))
    resized_img = img.resize((new_w, new_h), Image.Resampling.BILINEAR)
    padded_img = Image.new('L', (target_w, target_h), color=fill_color)
    pad_x = (target_w - new_w) // 2
    pad_y = (target_h - new_h) // 2
    padded_img.paste(resized_img, (pad_x, pad_y))
    return padded_img

class BoneFract448Dataset(Dataset):
    def __init__(self, df, path_index, region_to_idx, target_size=(448, 448), split='train'):
        self.df = df.reset_index(drop=True)
        self.path_index = path_index
        self.region_to_idx = region_to_idx
        self.target_size = target_size
        self.split = split
        self.fracture_to_idx = {'Negative': 0, 'Positive': 1}

        if split == 'train':
            self.transform = T.Compose([
                T.RandomRotation(degrees=(-10, 10)),
                T.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
                T.ColorJitter(brightness=0.1, contrast=0.1),
                T.ToTensor(),
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        else:
            self.transform = T.Compose([
                T.ToTensor(),
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_id = row['image_id']
        path = self.path_index[img_id]
        with Image.open(path) as im:
            gray = im.convert('L')
            padded = resize_and_pad(gray, self.target_size)
            rgb = padded.convert('RGB')
        tensor = self.transform(rgb)
        return {
            'image': tensor,
            'region_target': torch.tensor(self.region_to_idx[row['anatomical_region']], dtype=torch.long),
            'fracture_target': torch.tensor(self.fracture_to_idx[row['fracture_label']], dtype=torch.float32),
            'image_id': img_id
        }

def evaluate_validation(model, dataloader, criterion_region, criterion_fracture, device):
    model.eval()
    total_loss, total_loss_r, total_loss_f = 0.0, 0.0, 0.0
    all_r_t, all_r_p = [], []
    all_f_t, all_f_p, all_f_prob = [], [], []

    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(device)
            region_targets = batch['region_target'].to(device)
            fracture_targets = batch['fracture_target'].to(device)

            region_logits, fracture_logits = model(images)
            loss_r = criterion_region(region_logits, region_targets)
            loss_f = criterion_fracture(fracture_logits, fracture_targets)
            loss = loss_r + loss_f

            bs = images.size(0)
            total_loss += loss.item() * bs
            total_loss_r += loss_r.item() * bs
            total_loss_f += loss_f.item() * bs

            probs_r = torch.softmax(region_logits, dim=-1).cpu().numpy()
            preds_r = np.argmax(probs_r, axis=-1)
            probs_f = torch.sigmoid(fracture_logits).cpu().numpy().flatten()
            preds_f = (probs_f >= 0.5).astype(int)

            all_r_t.extend(region_targets.cpu().numpy())
            all_r_p.extend(preds_r)
            all_f_t.extend(fracture_targets.cpu().numpy())
            all_f_p.extend(preds_f)
            all_f_prob.extend(probs_f)

    n_samples = len(dataloader.dataset)
    avg_loss = total_loss / n_samples
    avg_loss_r = total_loss_r / n_samples
    avg_loss_f = total_loss_f / n_samples

    y_f_true = np.array(all_f_t, dtype=int)
    y_f_pred = np.array(all_f_p, dtype=int)
    y_f_prob = np.array(all_f_prob, dtype=float)
    y_r_true = np.array(all_r_t, dtype=int)
    y_r_pred = np.array(all_r_p, dtype=int)

    f_acc = accuracy_score(y_f_true, y_f_pred)
    f_prec = precision_score(y_f_true, y_f_pred, zero_division=0)
    f_rec = recall_score(y_f_true, y_f_pred, zero_division=0)
    f_f1 = f1_score(y_f_true, y_f_pred, zero_division=0)
    try:
        f_auc = roc_auc_score(y_f_true, y_f_prob)
    except Exception:
        f_auc = 0.5

    tn = np.sum((y_f_true == 0) & (y_f_pred == 0))
    fp = np.sum((y_f_true == 0) & (y_f_pred == 1))
    f_spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    r_acc = accuracy_score(y_r_true, y_r_pred)
    r_f1 = f1_score(y_r_true, y_r_pred, average='macro', zero_division=0)

    metrics = {
        'val_loss': avg_loss,
        'val_loss_region': avg_loss_r,
        'val_loss_fracture': avg_loss_f,
        'fracture_acc': f_acc,
        'fracture_prec': f_prec,
        'fracture_recall': f_rec,
        'fracture_spec': f_spec,
        'fracture_f1': f_f1,
        'fracture_auc': f_auc,
        'region_acc': r_acc,
        'region_macro_f1': r_f1
    }
    return metrics

def train_exp3(args):
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("="*70, flush=True)
    print("EXPERIMENT 3: CONTROLLED 448x448 HIGH-RESOLUTION RETRAINING", flush=True)
    print("="*70, flush=True)
    print(f"Device: {device} | Random Seed: {args.seed}", flush=True)
    print(f"Target Input Resolution: 448x448", flush=True)
    print(f"Batch Size: {args.batch_size} (reduced from 32 to prevent CUDA OOM on 4GB GPU)", flush=True)

    with open('data/bonefract_path_index.json') as f:
        path_index = json.load(f)

    train_df = pd.read_csv('train.csv')
    val_df = pd.read_csv('validation.csv')
    print(f"Training dataset: {len(train_df):,} samples", flush=True)
    print(f"Validation dataset: {len(val_df):,} samples", flush=True)

    # Class balance and pos_weight calculation (identical to train.py)
    n_neg = (train_df['fracture_label'] == 'Negative').sum()
    n_pos = (train_df['fracture_label'] == 'Positive').sum()
    pos_weight_val = float(n_neg / n_pos) if n_pos > 0 else 1.0
    print(f"Training Class balance: Negative={n_neg:,}, Positive={n_pos:,} (pos_weight = {pos_weight_val:.4f})", flush=True)
    pos_weight = torch.tensor([pos_weight_val], dtype=torch.float32).to(device)

    # Region mapping strictly matching best_model.pt
    region_to_idx = {'Arm': 0, 'Foot': 1, 'Hand': 2, 'Lower leg': 3, 'Thigh': 4, 'pelvis': 5, 'wrist': 6}
    num_regions = len(region_to_idx)
    print(f"Anatomical regions ({num_regions}): {list(region_to_idx.keys())}", flush=True)

    train_dataset = BoneFract448Dataset(train_df, path_index, region_to_idx, target_size=(448, 448), split='train')
    val_dataset = BoneFract448Dataset(val_df, path_index, region_to_idx, target_size=(448, 448), split='val')

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # MultiTaskModel initialization (identical architecture)
    model = MultiTaskModel(backbone='resnet50', num_regions=num_regions, pretrained=True, dropout=args.dropout)
    model.to(device)

    criterion_region = nn.CrossEntropyLoss()
    criterion_fracture = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    checkpoint_dest = 'models/checkpoints/exp3_resnet50_448.pt'
    os.makedirs(os.path.dirname(checkpoint_dest), exist_ok=True)

    history = []
    best_val_loss = float('inf')
    best_epoch = -1
    current_epoch = 0

    # ========================================================
    # STAGE 1: Train Task Heads with Frozen Backbone (3 Epochs)
    # ========================================================
    print("\n" + "="*60, flush=True)
    print("STAGE 1: Training Task Heads (Frozen ResNet-50 Backbone)", flush=True)
    print("="*60, flush=True)
    model.freeze_backbone()
    trainable_s1 = [p for p in model.parameters() if p.requires_grad]
    print(f"Stage 1 Trainable parameters: {sum(p.numel() for p in trainable_s1):,}", flush=True)

    optimizer_s1 = AdamW(trainable_s1, lr=args.lr_stage1, weight_decay=1e-4)
    scheduler_s1 = CosineAnnealingLR(optimizer_s1, T_max=args.epochs_stage1, eta_min=1e-5)

    for ep in range(1, args.epochs_stage1 + 1):
        current_epoch += 1
        t_ep0 = time.time()
        model.train()
        train_loss, train_loss_r, train_loss_f = 0.0, 0.0, 0.0
        n_batches = len(train_loader)

        for b_idx, batch in enumerate(train_loader):
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

            bs = images.size(0)
            train_loss += loss.item() * bs
            train_loss_r += loss_r.item() * bs
            train_loss_f += loss_f.item() * bs

            if (b_idx + 1) % 400 == 0 or (b_idx + 1) == n_batches:
                elapsed_b = time.time() - t_ep0
                done_samples = (b_idx + 1) * args.batch_size
                print(f"  [Epoch {current_epoch:02d} - Batch {b_idx+1:04d}/{n_batches:04d}] "
                      f"Loss: {loss.item():.4f} (R: {loss_r.item():.4f}, F: {loss_f.item():.4f}) | "
                      f"Throughput: {done_samples / elapsed_b:.1f} img/s", flush=True)

        scheduler_s1.step()

        train_loss /= len(train_dataset)
        train_loss_r /= len(train_dataset)
        train_loss_f /= len(train_dataset)

        val_metrics = evaluate_validation(model, val_loader, criterion_region, criterion_fracture, device)
        t_ep = time.time() - t_ep0

        val_loss = val_metrics['val_loss']
        print(f"Epoch {current_epoch:02d} [Stage 1] ({t_ep:.1f}s) | "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"Region Acc: {val_metrics['region_acc']*100:.2f}%, F1: {val_metrics['region_macro_f1']:.3f} | "
              f"Frac Acc: {val_metrics['fracture_acc']*100:.2f}%, Rec: {val_metrics['fracture_recall']*100:.2f}%, "
              f"Spec: {val_metrics['fracture_spec']*100:.2f}%, AUC: {val_metrics['fracture_auc']:.3f}", flush=True)

        history.append({
            'epoch': current_epoch,
            'stage': 'Stage 1 (Heads Only)',
            'train_loss': train_loss,
            'train_loss_region': train_loss_r,
            'train_loss_fracture': train_loss_f,
            **val_metrics
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = current_epoch
            torch.save({
                'state_dict': model.state_dict(),
                'region_to_idx': region_to_idx,
                'backbone': 'resnet50',
                'target_size': (448, 448),
                'pos_weight': pos_weight_val,
                'epoch': best_epoch,
                'val_loss': best_val_loss
            }, checkpoint_dest)
            print(f"  --> Checkpoint saved: New lowest validation loss ({val_loss:.4f}) to {checkpoint_dest}", flush=True)

    # ========================================================
    # STAGE 2: Fine-Tuning Upper Backbone (layer4 + Heads) (2 Epochs)
    # ========================================================
    if args.epochs_stage2 > 0:
        print("\n" + "="*60, flush=True)
        print("STAGE 2: Fine-Tuning Upper Backbone (layer4 + Heads)", flush=True)
        print("="*60, flush=True)
        model.unfreeze_upper_backbone()
        trainable_s2 = [p for p in model.parameters() if p.requires_grad]
        print(f"Stage 2 Trainable parameters: {sum(p.numel() for p in trainable_s2):,}", flush=True)

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
            n_batches = len(train_loader)

            for b_idx, batch in enumerate(train_loader):
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

                bs = images.size(0)
                train_loss += loss.item() * bs
                train_loss_r += loss_r.item() * bs
                train_loss_f += loss_f.item() * bs

                if (b_idx + 1) % 400 == 0 or (b_idx + 1) == n_batches:
                    elapsed_b = time.time() - t_ep0
                    done_samples = (b_idx + 1) * args.batch_size
                    print(f"  [Epoch {current_epoch:02d} - Batch {b_idx+1:04d}/{n_batches:04d}] "
                          f"Loss: {loss.item():.4f} (R: {loss_r.item():.4f}, F: {loss_f.item():.4f}) | "
                          f"Throughput: {done_samples / elapsed_b:.1f} img/s", flush=True)

            train_loss /= len(train_dataset)
            train_loss_r /= len(train_dataset)
            train_loss_f /= len(train_dataset)

            val_metrics = evaluate_validation(model, val_loader, criterion_region, criterion_fracture, device)
            val_loss = val_metrics['val_loss']
            scheduler_s2.step(val_loss)
            t_ep = time.time() - t_ep0

            print(f"Epoch {current_epoch:02d} [Stage 2] ({t_ep:.1f}s) | "
                  f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                  f"Region Acc: {val_metrics['region_acc']*100:.2f}%, F1: {val_metrics['region_macro_f1']:.3f} | "
                  f"Frac Acc: {val_metrics['fracture_acc']*100:.2f}%, Rec: {val_metrics['fracture_recall']*100:.2f}%, "
                  f"Spec: {val_metrics['fracture_spec']*100:.2f}%, AUC: {val_metrics['fracture_auc']:.3f}", flush=True)

            history.append({
                'epoch': current_epoch,
                'stage': 'Stage 2 (Upper Unfrozen)',
                'train_loss': train_loss,
                'train_loss_region': train_loss_r,
                'train_loss_fracture': train_loss_f,
                **val_metrics
            })

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = current_epoch
                torch.save({
                    'state_dict': model.state_dict(),
                    'region_to_idx': region_to_idx,
                    'backbone': 'resnet50',
                    'target_size': (448, 448),
                    'pos_weight': pos_weight_val,
                    'epoch': best_epoch,
                    'val_loss': best_val_loss
                }, checkpoint_dest)
                print(f"  --> Checkpoint saved: New lowest validation loss ({val_loss:.4f}) to {checkpoint_dest}", flush=True)

    # Save training history
    os.makedirs('reports', exist_ok=True)
    df_history = pd.DataFrame(history)
    df_history.to_csv('reports/exp3_training_history.csv', index=False)
    print(f"\nTraining Complete! History logged to reports/exp3_training_history.csv.", flush=True)
    print(f"Best checkpoint saved at: {checkpoint_dest} (Epoch {best_epoch}, Val Loss: {best_val_loss:.4f})", flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Experiment 3: Controlled 448x448 Retraining')
    parser.add_argument('--epochs_stage1', type=int, default=3)
    parser.add_argument('--epochs_stage2', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr_stage1', type=float, default=1e-3)
    parser.add_argument('--lr_stage2', type=float, default=1e-4)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    train_exp3(args)
