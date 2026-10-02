import os
import sys
import time
import random
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchvision.transforms as T
import torchvision.models as models
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from inference.preprocessing import resize_and_pad
from inference.model_registry import ModelRegistry

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class FractureDataset(Dataset):
    def __init__(self, df, transform=None, is_train=True):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.is_train = is_train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row['path']
        target = float(row['fracture_target'])

        with Image.open(img_path) as im:
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
            'target': torch.tensor(target, dtype=torch.float32),
            'filename': row['filename'],
            'region': row['region']
        }

def get_train_transforms():
    return T.Compose([
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(degrees=(-10, 10)),
        T.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        T.ColorJitter(brightness=0.15, contrast=0.20),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

def get_eval_transforms():
    return T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

class DedicatedFractureClassifier(nn.Module):
    """
    Dedicated ResNet-50 Binary Fracture Classifier.
    Specializes entirely on localized cortical fractures and bone disruptions
    without gradient competition from multi-task classification.
    """
    def __init__(self, pretrained=True, dropout=0.3):
        super(DedicatedFractureClassifier, self).__init__()
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        base = models.resnet50(weights=weights)
        
        self.backbone = nn.Sequential(
            base.conv1, base.bn1, base.relu, base.maxpool,
            base.layer1, base.layer2, base.layer3, base.layer4,
            base.avgpool, nn.Flatten()
        )
        self.layer4 = base.layer4
        
        self.classifier = nn.Sequential(
            nn.Linear(2048, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        features = self.backbone(x)
        logits = self.classifier(features).squeeze(-1)
        return logits

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for batch in loader:
            images = batch['image'].to(device)
            targets = batch['target'].to(device)
            logits = model(images)
            loss = criterion(logits, targets)

            total_loss += loss.item() * len(images)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs)
            all_targets.extend(targets.cpu().numpy())

    total_loss /= len(loader.dataset)
    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)
    preds = (all_probs >= 0.5).astype(int)

    acc = accuracy_score(all_targets, preds)
    prec = precision_score(all_targets, preds, zero_division=0)
    rec = recall_score(all_targets, preds, zero_division=0)
    f1 = f1_score(all_targets, preds, zero_division=0)
    try:
        auc = roc_auc_score(all_targets, all_probs)
    except:
        auc = 0.5
    try:
        pr_auc = average_precision_score(all_targets, all_probs)
    except:
        pr_auc = 0.0

    cm = confusion_matrix(all_targets, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        'loss': total_loss,
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'specificity': spec,
        'f1': f1,
        'auc': auc,
        'pr_auc': pr_auc,
        'confusion_matrix': cm.tolist(),
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
        'tp': int(tp),
        'probs': all_probs,
        'targets': all_targets
    }

def main():
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")

    train_df = pd.read_csv('data/fracture_splits/train_fracture.csv')
    val_df = pd.read_csv('data/fracture_splits/val_fracture.csv')
    print(f"Train samples: {len(train_df)} (Pos: {sum(train_df['fracture_target']==1)}, Neg: {sum(train_df['fracture_target']==0)})")
    print(f"Val samples:   {len(val_df)} (Pos: {sum(val_df['fracture_target']==1)}, Neg: {sum(val_df['fracture_target']==0)})")

    # Balanced training loader via WeightedRandomSampler: equal representation per batch
    train_dataset = FractureDataset(train_df, transform=get_train_transforms(), is_train=True)
    val_dataset = FractureDataset(val_df, transform=get_eval_transforms(), is_train=False)

    targets = train_df['fracture_target'].values.astype(int)
    class_counts = np.bincount(targets)
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[targets]
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(train_df) * 2,
        replacement=True
    )

    train_loader = DataLoader(train_dataset, batch_size=16, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)

    model = DedicatedFractureClassifier(pretrained=True, dropout=0.3).to(device)

    # Stage 1: Freeze lower backbone, train layer4 + head
    for param in model.backbone[:7].parameters(): # conv1 through layer3
        param.requires_grad = False
        
    optimizer = AdamW([
        {'params': model.layer4.parameters(), 'lr': 1e-4},
        {'params': model.classifier.parameters(), 'lr': 5e-4}
    ], weight_decay=1e-4)

    epochs = 25
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    criterion = nn.BCEWithLogitsLoss()

    best_val_f1 = 0.0
    best_val_loss = float('inf')
    best_checkpoint_path = 'models/checkpoints/fracture_model_improved.pth'
    os.makedirs('models/checkpoints', exist_ok=True)

    print("\n" + "="*70)
    print("TRAINING DEDICATED FRACTURE CLASSIFIER (AUDITED DATASET)")
    print("="*70)

    history = []
    for ep in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            images = batch['image'].to(device)
            targets = batch['target'].to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(images)

        scheduler.step()
        train_loss /= len(train_dataset)

        val_metrics = evaluate(model, val_loader, criterion, device)
        t_ep = time.time() - t0

        print(f"Epoch {ep:02d}/{epochs:02d} ({t_ep:.1f}s) | Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_metrics['loss']:.4f} | Acc: {val_metrics['accuracy']*100:.1f}% | "
              f"Rec: {val_metrics['recall']*100:.1f}% | Prec: {val_metrics['precision']*100:.1f}% | "
              f"Spec: {val_metrics['specificity']*100:.1f}% | F1: {val_metrics['f1']:.4f} | "
              f"AUC: {val_metrics['auc']:.4f} | FN: {val_metrics['fn']}")

        history.append({
            'epoch': ep,
            'train_loss': train_loss,
            'val_loss': val_metrics['loss'],
            'val_accuracy': val_metrics['accuracy'],
            'val_precision': val_metrics['precision'],
            'val_recall': val_metrics['recall'],
            'val_specificity': val_metrics['specificity'],
            'val_f1': val_metrics['f1'],
            'val_auc': val_metrics['auc'],
            'val_pr_auc': val_metrics['pr_auc'],
            'val_fn': val_metrics['fn'],
            'val_fp': val_metrics['fp'],
            'val_tp': val_metrics['tp'],
            'val_tn': val_metrics['tn']
        })

        if val_metrics['f1'] > best_val_f1 or (val_metrics['f1'] == best_val_f1 and val_metrics['loss'] < best_val_loss):
            best_val_f1 = val_metrics['f1']
            best_val_loss = val_metrics['loss']
            torch.save({
                'model_state_dict': model.state_dict(),
                'val_metrics': val_metrics,
                'epoch': ep,
                'backbone': 'resnet50'
            }, best_checkpoint_path)
            print(f"  --> Checkpoint saved: New best Val F1: {best_val_f1:.4f}")

    print("\nTraining completed.")
    print(f"Best model saved to: {best_checkpoint_path} with Val F1: {best_val_f1:.4f}")

    pd.DataFrame(history).to_csv('reports/fracture_training_history.csv', index=False)

if __name__ == '__main__':
    main()
