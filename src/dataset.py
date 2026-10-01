import os
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from PIL import Image
import pandas as pd

def get_transforms(split='train'):
    """
    Returns image transformation pipelines.
    Training uses conservative, medically sound augmentations.
    Validation and test use strictly deterministic transformations.
    """
    if split == 'train':
        return T.Compose([
            # Conservative augmentations: strictly small variations
            T.RandomRotation(degrees=(-10, 10)),
            T.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
            T.ColorJitter(brightness=0.1, contrast=0.1),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    else:
        return T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

class BoneFractDataset(Dataset):
    """
    PyTorch Dataset for multi-task bone X-ray classification.
    Predicts:
      1. Anatomical region (multi-class)
      2. Fracture presence (binary: 0=no fracture, 1=fracture)
    """
    def __init__(self, data_source, region_to_idx=None, transform=None, base_dir='.'):
        if isinstance(data_source, str):
            self.df = pd.read_csv(data_source)
        else:
            self.df = data_source.copy().reset_index(drop=True)
            
        self.base_dir = base_dir
        self.transform = transform
        
        # Build or assign region mapping
        if region_to_idx is None:
            unique_regions = sorted(self.df['anatomical_region'].unique())
            self.region_to_idx = {reg: idx for idx, reg in enumerate(unique_regions)}
        else:
            self.region_to_idx = region_to_idx
            
        self.idx_to_region = {idx: reg for reg, idx in self.region_to_idx.items()}
        
        # Binary fracture mapping: Negative=0, Positive=1
        self.fracture_to_idx = {'Negative': 0, 'Positive': 1}
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row['image_path']
        if not os.path.isabs(img_path) and self.base_dir:
            img_path = os.path.join(self.base_dir, img_path)
            
        with Image.open(img_path) as im:
            # Broadcast to 3-channel RGB for standard CNN/ViT backbones
            rgb_img = im.convert('RGB')
            
        if self.transform:
            tensor_img = self.transform(rgb_img)
        else:
            tensor_img = T.functional.to_tensor(rgb_img)
            
        region_label = self.region_to_idx[row['anatomical_region']]
        fracture_label = self.fracture_to_idx[row['fracture_label']]
        
        return {
            'image': tensor_img,
            'region_target': torch.tensor(region_label, dtype=torch.long),
            'fracture_target': torch.tensor(fracture_label, dtype=torch.float32),
            'image_id': row['image_id'],
            'patient_id': row['patient_id'],
            'anatomical_region': row['anatomical_region'],
            'fracture_label_str': row['fracture_label']
        }
