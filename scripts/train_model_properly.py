import os
import sys
import time
import glob
import random
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchvision.transforms as T

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models import MultiTaskModel
from inference.model_registry import ModelRegistry
from inference.preprocessing import resize_and_pad

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class CuratedXRayDataset(Dataset):
    def __init__(self, samples, transform=None, is_train=True):
        self.samples = samples # list of (path, region_idx, fracture_label)
        self.transform = transform
        self.is_train = is_train

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, region_idx, frac_val = self.samples[idx]
        with Image.open(path) as im:
            gray = im.convert('L')
            padded = resize_and_pad(gray, target_size=(224, 224))
            rgb = padded.convert('RGB')

        if self.transform:
            tensor = self.transform(rgb)
        else:
            tensor = T.ToTensor()(rgb)
            tensor = T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])(tensor)

        return {
            'image': tensor,
            'region_target': torch.tensor(region_idx, dtype=torch.long),
            'fracture_target': torch.tensor(frac_val, dtype=torch.float32)
        }

def get_train_transform():
    return T.Compose([
        T.RandomRotation(degrees=(-15, 15)),
        T.RandomAffine(degrees=0, translate=(0.08, 0.08), scale=(0.92, 1.08)),
        T.RandomHorizontalFlip(p=0.5),
        T.ColorJitter(brightness=0.15, contrast=0.15),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

def get_eval_transform():
    return T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

def main():
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")

    region_to_idx = ModelRegistry.REGION_TO_IDX
    idx_to_region = ModelRegistry.IDX_TO_REGION
    data_dir = 'data/training_curated'

    # Collect samples
    all_samples = []
    class_counts = {c: 0 for c in region_to_idx}

    for region_name, r_idx in region_to_idx.items():
        reg_dir = os.path.join(data_dir, region_name)
        if not os.path.exists(reg_dir):
            continue
        files = [f for f in glob.glob(f"{reg_dir}/*") if os.path.getsize(f) > 5000]
        for f in files:
            frac_val = 1.0 if 'Positive' in os.path.basename(f) else 0.0
            all_samples.append((f, r_idx, frac_val))
            class_counts[region_name] += 1

    print("Class sample counts:")
    for c, cnt in class_counts.items():
        print(f"  {c}: {cnt}")

    # Balance over-represented classes (e.g. pelvis has 79, cap to 20 per epoch to prevent pelvic bias)
    # And oversample minority classes (Arm, Foot, Hand) to ~20 per epoch
    target_per_class = 25
    balanced_samples = []
    for region_name, r_idx in region_to_idx.items():
        c_samples = [s for s in all_samples if s[1] == r_idx]
        if not c_samples:
            continue
        # Sample with replacement up to target_per_class
        multiplier = target_per_class // len(c_samples)
        remainder = target_per_class % len(c_samples)
        balanced_samples.extend(c_samples * multiplier + random.sample(c_samples, remainder))

    random.shuffle(balanced_samples)
    print(f"Total balanced training samples per epoch: {len(balanced_samples)}")

    train_ds = CuratedXRayDataset(balanced_samples, transform=get_train_transform(), is_train=True)
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=0)

    # Initialize model with ResNet-50 ImageNet backbone
    model = MultiTaskModel(backbone='resnet50', num_regions=len(region_to_idx), pretrained=True, dropout=0.3)
    model.to(device)

    # Freeze backbone initially (Stage 1: Train heads only)
    model.freeze_backbone()
    trainable_params_s1 = [p for p in model.parameters() if p.requires_grad]
    optimizer_s1 = AdamW(trainable_params_s1, lr=1e-3, weight_decay=1e-4)
    scheduler_s1 = CosineAnnealingLR(optimizer_s1, T_max=20, eta_min=1e-5)

    criterion_region = nn.CrossEntropyLoss()
    criterion_fracture = nn.BCEWithLogitsLoss()

    print("\n" + "="*60)
    print("STAGE 1: TRAINING TASK-SPECIFIC HEADS (FROZEN BACKBONE)")
    print("="*60)

    best_loss = float('inf')
    epochs_stage1 = 20

    for ep in range(1, epochs_stage1 + 1):
        t0 = time.time()
        model.train()
        total_loss, total_r_loss, total_f_loss = 0.0, 0.0, 0.0
        correct_reg, total_reg = 0, 0

        for batch in train_loader:
            images = batch['image'].to(device)
            r_targets = batch['region_target'].to(device)
            f_targets = batch['fracture_target'].to(device)

            optimizer_s1.zero_grad()
            r_logits, f_logits = model(images)

            l_r = criterion_region(r_logits, r_targets)
            l_f = criterion_fracture(f_logits, f_targets)
            loss = l_r + 0.5 * l_f

            loss.backward()
            optimizer_s1.step()

            total_loss += loss.item() * len(images)
            total_r_loss += l_r.item() * len(images)
            total_f_loss += l_f.item() * len(images)

            preds = torch.argmax(r_logits, dim=-1)
            correct_reg += (preds == r_targets).sum().item()
            total_reg += len(images)

        scheduler_s1.step()
        ep_loss = total_loss / total_reg
        ep_acc = correct_reg / total_reg * 100.0
        t_ep = time.time() - t0

        print(f"Epoch {ep:02d}/20 ({t_ep:.1f}s) | Loss: {ep_loss:.4f} (R: {total_r_loss/total_reg:.4f}, F: {total_f_loss/total_reg:.4f}) | Train Region Acc: {ep_acc:.1f}%")

    # STAGE 2: Fine-tune Upper Backbone (layer4) with small learning rate
    print("\n" + "="*60)
    print("STAGE 2: FINE-TUNING LAYER4 + HEADS")
    print("="*60)

    model.unfreeze_upper_backbone()
    backbone_params = [p for p in model.layer4.parameters() if p.requires_grad]
    head_params = list(model.region_head.parameters()) + list(model.fracture_head.parameters())

    optimizer_s2 = AdamW([
        {'params': backbone_params, 'lr': 5e-5},
        {'params': head_params, 'lr': 2e-4}
    ], weight_decay=1e-4)
    scheduler_s2 = CosineAnnealingLR(optimizer_s2, T_max=10, eta_min=1e-6)

    epochs_stage2 = 10
    for ep in range(1, epochs_stage2 + 1):
        t0 = time.time()
        model.train()
        total_loss, total_r_loss, total_f_loss = 0.0, 0.0, 0.0
        correct_reg, total_reg = 0, 0

        for batch in train_loader:
            images = batch['image'].to(device)
            r_targets = batch['region_target'].to(device)
            f_targets = batch['fracture_target'].to(device)

            optimizer_s2.zero_grad()
            r_logits, f_logits = model(images)

            l_r = criterion_region(r_logits, r_targets)
            l_f = criterion_fracture(f_logits, f_targets)
            loss = l_r + 0.5 * l_f

            loss.backward()
            optimizer_s2.step()

            total_loss += loss.item() * len(images)
            total_r_loss += l_r.item() * len(images)
            total_f_loss += l_f.item() * len(images)

            preds = torch.argmax(r_logits, dim=-1)
            correct_reg += (preds == r_targets).sum().item()
            total_reg += len(images)

        scheduler_s2.step()
        ep_loss = total_loss / total_reg
        ep_acc = correct_reg / total_reg * 100.0
        t_ep = time.time() - t0

        print(f"Epoch {ep:02d}/10 ({t_ep:.1f}s) | Loss: {ep_loss:.4f} (R: {total_r_loss/total_reg:.4f}, F: {total_f_loss/total_reg:.4f}) | Train Region Acc: {ep_acc:.1f}%")

    # Save trained checkpoint conforming to exact contract
    out_ckpt = 'models/checkpoints/best_model.pt'
    os.makedirs(os.path.dirname(out_ckpt), exist_ok=True)
    checkpoint_dict = {
        'state_dict': model.state_dict(),
        'region_to_idx': region_to_idx,
        'backbone': 'resnet50',
        'pos_weight': 1.0
    }
    torch.save(checkpoint_dict, out_ckpt)
    print(f"\nSuccessfully saved trained checkpoint to: {out_ckpt}")

if __name__ == '__main__':
    main()
