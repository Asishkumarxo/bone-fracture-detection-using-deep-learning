"""
Experiment 4: Balanced Region-Aware ResNet-50 Fine-Tuning
=========================================================
Independent training experiment investigating whether balanced anatomical-region
training (2,000 unique images per region, 14,000 total) improves binary fracture
detection consistency across Arm, Foot, Hand, Lower leg, Thigh, Pelvis, and Wrist.

Key Constraints & Configuration:
- Initialized from verified checkpoint: models/checkpoints/exp3_resnet50_448.pt
- Checkpoints best_model.pt and exp3_resnet50_448.pt remain strictly untouched
- Target checkpoint: models/checkpoints/exp4_balanced_resnet50_448.pt
- Input resolution: 448x448 aspect-ratio padded preprocessing
- Two-stage fine-tuning:
    Stage 1: Frozen backbone, heads trained at lr=1e-3 (3 warm-up epochs)
    Stage 2: layer4 unfrozen at lr=2e-5, heads at lr=1e-4
- Early stopping: Validation PR-AUC, min_delta=0.001, patience=12, max 50 epochs
- Full metric tracking: Per-region fracture recall, PR-AUC, ROC-AUC, F1, Loss
- Logs output to reports/exp4_training_history.csv
"""

import os
import sys
import time
import json
import random
import hashlib
import argparse
import pandas as pd
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchvision.transforms as T
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, precision_recall_curve, auc, confusion_matrix
)

# Workspace root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.models import MultiTaskModel

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def compute_sha256(filepath):
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def resize_and_pad(img, target_size=(448, 448), fill_color=0):
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
            'image_id': img_id,
            'anatomical_region': row['anatomical_region'],
            'fracture_label': row['fracture_label']
        }

def compute_prauc(y_true, y_probs):
    prec, rec, _ = precision_recall_curve(y_true, y_probs)
    return auc(rec, prec)

def evaluate_model(model, dataloader, criterion_region, criterion_fracture, device, idx_to_region):
    model.eval()
    total_loss, total_loss_r, total_loss_f = 0.0, 0.0, 0.0
    all_reg_true, all_reg_pred = [], []
    all_frac_true, all_frac_prob = [], []
    all_regions = []

    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(device)
            reg_targets = batch['region_target'].to(device)
            frac_targets = batch['fracture_target'].to(device)

            reg_logits, frac_logits = model(images)
            loss_r = criterion_region(reg_logits, reg_targets)
            loss_f = criterion_fracture(frac_logits, frac_targets)
            loss = loss_r + 1.5 * loss_f

            bs = images.size(0)
            total_loss += loss.item() * bs
            total_loss_r += loss_r.item() * bs
            total_loss_f += loss_f.item() * bs

            reg_probs = torch.softmax(reg_logits, dim=-1).cpu().numpy()
            reg_preds = np.argmax(reg_probs, axis=-1)
            frac_probs = torch.sigmoid(frac_logits).cpu().numpy()

            all_reg_true.extend(reg_targets.cpu().numpy())
            all_reg_pred.extend(reg_preds)
            all_frac_true.extend(frac_targets.cpu().numpy())
            all_frac_prob.extend(frac_probs)
            all_regions.extend(batch['anatomical_region'])

    n = len(dataloader.dataset)
    avg_loss = total_loss / n
    avg_loss_r = total_loss_r / n
    avg_loss_f = total_loss_f / n

    y_true = np.array(all_frac_true)
    y_prob = np.array(all_frac_prob)
    y_pred = (y_prob >= 0.50).astype(int)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    try:
        roc_auc = roc_auc_score(y_true, y_prob)
    except Exception:
        roc_auc = 0.5
    pr_auc = compute_prauc(y_true, y_prob)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    reg_acc = accuracy_score(all_reg_true, all_reg_pred)

    # Per-region fracture recall
    region_recalls = {}
    regions_arr = np.array(all_regions)
    for reg_name in sorted(idx_to_region.values()):
        mask = (regions_arr == reg_name)
        reg_y_true = y_true[mask]
        reg_y_pred = y_pred[mask]
        pos_mask = (reg_y_true == 1)
        if pos_mask.sum() > 0:
            reg_rec = (reg_y_pred[pos_mask] == 1).mean()
        else:
            reg_rec = 0.0
        region_recalls[reg_name] = reg_rec

    return {
        'val_loss': avg_loss,
        'val_loss_r': avg_loss_r,
        'val_loss_f': avg_loss_f,
        'val_acc': acc,
        'val_precision': prec,
        'val_recall': rec,
        'val_f1': f1,
        'val_specificity': spec,
        'val_roc_auc': roc_auc,
        'val_pr_auc': pr_auc,
        'val_region_acc': reg_acc,
        'region_recalls': region_recalls
    }

def train_exp4(args):
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 80, flush=True)
    print("EXPERIMENT 4: BALANCED REGION-AWARE RESNET-50 FINE-TUNING", flush=True)
    print("=" * 80, flush=True)
    print(f"Device: {device} | Seed: {args.seed}", flush=True)
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)} | VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB", flush=True)
    print(f"Input Resolution: 448x448 | Batch Size: {args.batch_size} | Max Epochs: {args.max_epochs}", flush=True)

    # 1. Checkpoint Verification
    exp3_ckpt_path = "models/checkpoints/exp3_resnet50_448.pt"
    if not os.path.exists(exp3_ckpt_path):
        raise FileNotFoundError(f"Experiment 3 checkpoint not found at: {exp3_ckpt_path}")
    exp3_hash = compute_sha256(exp3_ckpt_path)
    print(f"Verified Experiment 3 Checkpoint: {exp3_ckpt_path} (SHA-256: {exp3_hash[:16]}...)", flush=True)

    # 2. Path Index & Datasets
    manifest_path = "reports/exp4_sample_manifest.csv"
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Sample manifest not found at: {manifest_path}. Please generate it first.")
    
    train_df = pd.read_csv(manifest_path)
    val_df = pd.read_csv("validation.csv")
    print(f"Active training set (Balanced Manifest): {len(train_df):,} samples (2,000 per region)", flush=True)
    print(f"Active validation set: {len(val_df):,} samples", flush=True)

    # Build direct path map to avoid slow relative resolutions
    path_index = {}
    for _, row in train_df.iterrows():
        path_index[row['image_id']] = row['resolved_path']
        
    # For validation, build from bonefract_path_index
    with open("data/bonefract_path_index.json", "r", encoding="utf-8") as f:
        old_idx = json.load(f)
    dataset_root = r"C:\Users\Donipudi Asish Kumar\Desktop\BoneFract A Bone Fracture Dataset"
    for img_id, old_p in old_idx.items():
        if img_id not in path_index:
            norm = old_p.replace("\\", "/")
            parts = norm.split("/")
            sub_idx = -1
            for i, p in enumerate(parts):
                if p in ("train", "valid", "test"):
                    sub_idx = i
                    break
            rel = os.path.join(*parts[sub_idx:])
            path_index[img_id] = os.path.join(dataset_root, rel)

    region_to_idx = {'Arm': 0, 'Foot': 1, 'Hand': 2, 'Lower leg': 3, 'Thigh': 4, 'pelvis': 5, 'wrist': 6}
    idx_to_region = {v: k for k, v in region_to_idx.items()}
    num_regions = len(region_to_idx)

    # Calculate class balance in manifest
    n_neg = (train_df['fracture_label'] == 'Negative').sum()
    n_pos = (train_df['fracture_label'] == 'Positive').sum()
    pos_weight_val = float(n_neg / n_pos) if n_pos > 0 else 1.0
    print(f"Manifest Binary Balance: Negative={n_neg:,}, Positive={n_pos:,} | pos_weight = {pos_weight_val:.4f}", flush=True)
    pos_weight = torch.tensor([pos_weight_val], dtype=torch.float32).to(device)

    train_dataset = BoneFract448Dataset(train_df, path_index, region_to_idx, target_size=(448, 448), split='train')
    val_dataset = BoneFract448Dataset(val_df, path_index, region_to_idx, target_size=(448, 448), split='val')

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=torch.cuda.is_available())

    # 3. Model Initialization from Experiment 3 Checkpoint
    print("\nInitializing model from Experiment 3 weights...", flush=True)
    model = MultiTaskModel(backbone='resnet50', num_regions=num_regions, pretrained=False, dropout=0.3)
    exp3_data = torch.load(exp3_ckpt_path, map_location='cpu')
    model.load_state_dict(exp3_data['state_dict'])
    model.to(device)
    print("Successfully loaded Experiment 3 weights into MultiTaskModel.", flush=True)

    criterion_region = nn.CrossEntropyLoss()
    criterion_fracture = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Mixed precision scaler
    use_amp = torch.cuda.is_available()
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    # 4. Target Checkpoint Path
    checkpoint_dest = "models/checkpoints/exp4_balanced_resnet50_448.pt"
    os.makedirs(os.path.dirname(checkpoint_dest), exist_ok=True)

    # 5. Tracking and Early Stopping
    history = []
    best_pr_auc = -1.0
    best_epoch = -1
    patience_counter = 0
    patience_limit = 12
    min_delta = 0.001
    stopping_reason = "Completed all maximum epochs"

    # ========================================================
    # STAGE 1: Head Adaptation (Frozen ResNet-50 Backbone)
    # ========================================================
    stage1_epochs = args.stage1_epochs
    print("\n" + "=" * 70, flush=True)
    print(f"STAGE 1: Training Classification Heads ({stage1_epochs} Warm-up Epochs)", flush=True)
    print("=" * 70, flush=True)
    model.freeze_backbone()
    trainable_s1 = [p for p in model.parameters() if p.requires_grad]
    print(f"Stage 1 Trainable Parameters: {sum(p.numel() for p in trainable_s1):,}", flush=True)

    optimizer_s1 = AdamW(trainable_s1, lr=args.lr_stage1, weight_decay=1e-4)

    current_epoch = 0
    for ep in range(1, stage1_epochs + 1):
        current_epoch += 1
        t_start = time.time()
        model.train()
        train_loss, train_loss_r, train_loss_f = 0.0, 0.0, 0.0

        for batch_idx, batch in enumerate(train_loader):
            images = batch['image'].to(device)
            reg_targets = batch['region_target'].to(device)
            frac_targets = batch['fracture_target'].to(device)

            optimizer_s1.zero_grad()
            with torch.amp.autocast('cuda', enabled=use_amp):
                reg_logits, frac_logits = model(images)
                loss_r = criterion_region(reg_logits, reg_targets)
                loss_f = criterion_fracture(frac_logits, frac_targets)
                loss = loss_r + 1.5 * loss_f

            scaler.scale(loss).backward()
            scaler.step(optimizer_s1)
            scaler.update()

            bs = images.size(0)
            train_loss += loss.item() * bs
            train_loss_r += loss_r.item() * bs
            train_loss_f += loss_f.item() * bs

        avg_train_loss = train_loss / len(train_dataset)
        avg_train_loss_r = train_loss_r / len(train_dataset)
        avg_train_loss_f = train_loss_f / len(train_dataset)

        # Validation
        val_metrics = evaluate_model(model, val_loader, criterion_region, criterion_fracture, device, idx_to_region)
        duration = time.time() - t_start

        # Check early stopping / best model
        val_pr_auc = val_metrics['val_pr_auc']
        improved = (val_pr_auc - best_pr_auc) >= min_delta
        if improved:
            best_pr_auc = val_pr_auc
            best_epoch = current_epoch
            patience_counter = 0
            # Save best checkpoint
            torch.save({
                'epoch': current_epoch,
                'state_dict': model.state_dict(),
                'region_to_idx': region_to_idx,
                'backbone': 'resnet50',
                'target_size': 448,
                'pos_weight': pos_weight_val,
                'best_pr_auc': best_pr_auc,
                'val_metrics': val_metrics,
                'stage': 1
            }, checkpoint_dest)
            status_tag = f"(*) BEST (PR-AUC: {best_pr_auc:.4f})"
        else:
            status_tag = ""

        print(
            f"Epoch {current_epoch:02d} [Stage 1] | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {val_metrics['val_loss']:.4f} | "
            f"Val PR-AUC: {val_pr_auc:.4f} | "
            f"Val Recall: {val_metrics['val_recall']:.4f} | "
            f"Val Acc: {val_metrics['val_acc']:.4f} | "
            f"Time: {duration:.1f}s {status_tag}",
            flush=True
        )

        rec_str = " | ".join([f"{k}: {v:.3f}" for k, v in val_metrics['region_recalls'].items()])
        print(f"  Per-Region Fracture Recall: {rec_str}", flush=True)

        log_entry = {
            'epoch': current_epoch,
            'stage': 1,
            'train_loss': avg_train_loss,
            'train_loss_region': avg_train_loss_r,
            'train_loss_fracture': avg_train_loss_f,
            'val_loss': val_metrics['val_loss'],
            'val_acc': val_metrics['val_acc'],
            'val_precision': val_metrics['val_precision'],
            'val_recall': val_metrics['val_recall'],
            'val_f1': val_metrics['val_f1'],
            'val_specificity': val_metrics['val_specificity'],
            'val_roc_auc': val_metrics['val_roc_auc'],
            'val_pr_auc': val_pr_auc,
            'val_region_acc': val_metrics['val_region_acc'],
            'patience_counter': patience_counter,
            'learning_rate': args.lr_stage1,
            'duration_sec': round(duration, 1)
        }
        for reg_k, reg_v in val_metrics['region_recalls'].items():
            log_entry[f"recall_{reg_k}"] = round(reg_v, 4)
        history.append(log_entry)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ========================================================
    # STAGE 2: Partial Backbone Fine-Tuning (layer4 Unfrozen)
    # ========================================================
    print("\n" + "=" * 70, flush=True)
    print("STAGE 2: Partial Backbone Fine-Tuning (ResNet-50 layer4 Unfrozen)", flush=True)
    print("=" * 70, flush=True)

    # Unfreeze layer4
    model.unfreeze_upper_backbone()

    trainable_s2 = [p for p in model.parameters() if p.requires_grad]
    print(f"Stage 2 Trainable Parameters: {sum(p.numel() for p in trainable_s2):,}", flush=True)

    optimizer_s2 = AdamW([
        {'params': model.layer4.parameters(), 'lr': args.lr_stage2_backbone},
        {'params': list(model.region_head.parameters()) + list(model.fracture_head.parameters()), 'lr': args.lr_stage2_heads}
    ], weight_decay=1e-4)

    scheduler_s2 = ReduceLROnPlateau(optimizer_s2, mode='max', factor=0.5, patience=4, min_lr=1e-6)

    stage2_epochs = args.max_epochs - stage1_epochs
    for ep in range(1, stage2_epochs + 1):
        current_epoch += 1
        t_start = time.time()
        model.train()
        train_loss, train_loss_r, train_loss_f = 0.0, 0.0, 0.0

        for batch_idx, batch in enumerate(train_loader):
            images = batch['image'].to(device)
            reg_targets = batch['region_target'].to(device)
            frac_targets = batch['fracture_target'].to(device)

            optimizer_s2.zero_grad()
            with torch.amp.autocast('cuda', enabled=use_amp):
                reg_logits, frac_logits = model(images)
                loss_r = criterion_region(reg_logits, reg_targets)
                loss_f = criterion_fracture(frac_logits, frac_targets)
                loss = loss_r + 1.5 * loss_f

            scaler.scale(loss).backward()
            scaler.step(optimizer_s2)
            scaler.update()

            bs = images.size(0)
            train_loss += loss.item() * bs
            train_loss_r += loss_r.item() * bs
            train_loss_f += loss_f.item() * bs

        avg_train_loss = train_loss / len(train_dataset)
        avg_train_loss_r = train_loss_r / len(train_dataset)
        avg_train_loss_f = train_loss_f / len(train_dataset)

        # Validation
        val_metrics = evaluate_model(model, val_loader, criterion_region, criterion_fracture, device, idx_to_region)
        duration = time.time() - t_start
        val_pr_auc = val_metrics['val_pr_auc']

        # Learning rate scheduler step
        scheduler_s2.step(val_pr_auc)

        # Adaptive Early Stopping check
        improved = (val_pr_auc - best_pr_auc) >= min_delta
        if improved:
            best_pr_auc = val_pr_auc
            best_epoch = current_epoch
            patience_counter = 0
            # Save best checkpoint
            torch.save({
                'epoch': current_epoch,
                'state_dict': model.state_dict(),
                'region_to_idx': region_to_idx,
                'backbone': 'resnet50',
                'target_size': 448,
                'pos_weight': pos_weight_val,
                'best_pr_auc': best_pr_auc,
                'val_metrics': val_metrics,
                'stage': 2
            }, checkpoint_dest)
            status_tag = f"(*) BEST (PR-AUC: {best_pr_auc:.4f})"
        else:
            patience_counter += 1
            status_tag = f"[Patience {patience_counter}/{patience_limit}]"

        curr_lr_heads = optimizer_s2.param_groups[1]['lr']
        print(
            f"Epoch {current_epoch:02d} [Stage 2] | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {val_metrics['val_loss']:.4f} | "
            f"Val PR-AUC: {val_pr_auc:.4f} | "
            f"Val Recall: {val_metrics['val_recall']:.4f} | "
            f"Val Acc: {val_metrics['val_acc']:.4f} | "
            f"Time: {duration:.1f}s {status_tag}",
            flush=True
        )

        rec_str = " | ".join([f"{k}: {v:.3f}" for k, v in val_metrics['region_recalls'].items()])
        print(f"  Per-Region Fracture Recall: {rec_str}", flush=True)

        log_entry = {
            'epoch': current_epoch,
            'stage': 2,
            'train_loss': avg_train_loss,
            'train_loss_region': avg_train_loss_r,
            'train_loss_fracture': avg_train_loss_f,
            'val_loss': val_metrics['val_loss'],
            'val_acc': val_metrics['val_acc'],
            'val_precision': val_metrics['val_precision'],
            'val_recall': val_metrics['val_recall'],
            'val_f1': val_metrics['val_f1'],
            'val_specificity': val_metrics['val_specificity'],
            'val_roc_auc': val_metrics['val_roc_auc'],
            'val_pr_auc': val_pr_auc,
            'val_region_acc': val_metrics['val_region_acc'],
            'patience_counter': patience_counter,
            'learning_rate': curr_lr_heads,
            'duration_sec': round(duration, 1)
        }
        for reg_k, reg_v in val_metrics['region_recalls'].items():
            log_entry[f"recall_{reg_k}"] = round(reg_v, 4)
        history.append(log_entry)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Check early stopping trigger
        if patience_counter >= patience_limit:
            stopping_reason = f"Early stopping triggered: 12 consecutive epochs without min_delta (0.001) improvement in Val PR-AUC"
            print(f"\n{stopping_reason}. Terminating at Epoch {current_epoch}.", flush=True)
            break

    # Save training history
    df_history = pd.DataFrame(history)
    os.makedirs("reports", exist_ok=True)
    df_history.to_csv("reports/exp4_training_history.csv", index=False)
    print("\nSaved training history to reports/exp4_training_history.csv", flush=True)

    print("=" * 80, flush=True)
    print("TRAINING RUN COMPLETE", flush=True)
    print(f"Total Epochs Completed: {current_epoch}", flush=True)
    print(f"Best Epoch: {best_epoch} with Val PR-AUC: {best_pr_auc:.4f}", flush=True)
    print(f"Stopping Reason: {stopping_reason}", flush=True)
    print(f"Best Checkpoint Saved to: {checkpoint_dest}", flush=True)
    print("=" * 80, flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Experiment 4: Balanced Region-Aware Fine-Tuning")
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size (default: 16 for RTX 3050 4GB)")
    parser.add_argument('--max_epochs', type=int, default=50, help="Maximum epochs total (default: 50)")
    parser.add_argument('--stage1_epochs', type=int, default=3, help="Warm-up epochs for Stage 1 (default: 3)")
    parser.add_argument('--lr_stage1', type=float, default=1e-3, help="Stage 1 Head Learning Rate")
    parser.add_argument('--lr_stage2_backbone', type=float, default=2e-5, help="Stage 2 layer4 Learning Rate")
    parser.add_argument('--lr_stage2_heads', type=float, default=1e-4, help="Stage 2 Head Learning Rate")
    parser.add_argument('--seed', type=int, default=42, help="Random seed")
    parser.add_argument('--num_workers', type=int, default=0, help="DataLoader worker processes")
    args = parser.parse_args()

    train_exp4(args)
