import os
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from PIL import Image
import pandas as pd
import numpy as np

class FracAtlasDetectionDataset(Dataset):
    """
    Object detection dataset for fracture localization on FracAtlas.
    Provides scaled bounding boxes, labels, and polygon segmentation references.
    """
    def __init__(self, split_csv, annotations_csv, target_size=(512, 512), is_train=False, fracatlas_dir=None):
        self.target_size = target_size # (W, H)
        self.is_train = is_train
        self.fracatlas_dir = fracatlas_dir or os.environ.get('FRACATLAS_DIR', 'data/FracAtlas')
        
        # Load split image IDs
        df_split = pd.read_csv(split_csv)
        split_ids = set(df_split['image_id'])
        
        # Load clean annotations
        df_ann = pd.read_csv(annotations_csv)
        self.df_ann = df_ann[df_ann['image_id'].isin(split_ids)].copy()
        
        # Unique images in this split that have annotations
        self.image_ids = sorted(list(self.df_ann['image_id'].unique()))
        
        # Group annotations by image_id
        self.ann_groups = self.df_ann.groupby('image_id')
        
    def __len__(self):
        return len(self.image_ids)
        
    def _resolve_image_path(self, raw_path: str, file_name: str) -> str:
        if os.path.exists(raw_path):
            return raw_path
        # Try finding in configured fracatlas_dir
        candidates = [
            os.path.join(self.fracatlas_dir, "images", "Fractured", file_name),
            os.path.join(self.fracatlas_dir, "images", "Non_fractured", file_name),
            os.path.join(self.fracatlas_dir, "images", file_name),
            os.path.join(self.fracatlas_dir, file_name),
            os.path.join("data", "FracAtlas", "images", "Fractured", file_name),
            os.path.join("data", "FracAtlas", "images", file_name)
        ]
        for c in candidates:
            if os.path.exists(c):
                return c
        return raw_path

    def __getitem__(self, idx):
        file_name = self.image_ids[idx]
        group = self.ann_groups.get_group(file_name)
        
        raw_path = group.iloc[0]['image_path']
        img_path = self._resolve_image_path(raw_path, file_name)
        orig_w = group.iloc[0]['width']
        orig_h = group.iloc[0]['height']
        anatomy = group.iloc[0]['anatomical_region']
        
        if not os.path.exists(img_path):
            raise FileNotFoundError(
                f"FracAtlas image '{file_name}' not found at '{raw_path}' or in '{self.fracatlas_dir}'. "
                f"Please ensure FracAtlas is located in data/FracAtlas or set the FRACATLAS_DIR environment variable."
            )
        
        with Image.open(img_path) as im:
            rgb_img = im.convert('RGB')
            
        target_w, target_h = self.target_size
        scale_x = target_w / orig_w
        scale_y = target_h / orig_h
        
        # Resize image
        resized_img = rgb_img.resize(self.target_size, Image.Resampling.BILINEAR)
        
        # Scale bounding boxes: [x_min, y_min, x_max, y_max]
        boxes = []
        labels = []
        areas = []
        
        for _, row in group.iterrows():
            x1 = row['x_min'] * scale_x
            y1 = row['y_min'] * scale_y
            x2 = row['x_max'] * scale_x
            y2 = row['y_max'] * scale_y
            
            # Constrain to target boundary
            x1 = max(0.0, min(float(x1), float(target_w - 1)))
            y1 = max(0.0, min(float(y1), float(target_h - 1)))
            x2 = max(x1 + 1.0, min(float(x2), float(target_w)))
            y2 = max(y1 + 1.0, min(float(y2), float(target_h)))
            
            boxes.append([x1, y1, x2, y2])
            labels.append(1) # Class 1: Fracture
            areas.append((x2 - x1) * (y2 - y1))
            
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        areas = torch.as_tensor(areas, dtype=torch.float32)
        iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)
        
        # Safe horizontal flip augmentation for training
        if self.is_train and np.random.rand() > 0.5:
            resized_img = TF.hflip(resized_img)
            # Flip box coordinates horizontally: x1_new = W - x2, x2_new = W - x1
            new_x1 = target_w - boxes[:, 2]
            new_x2 = target_w - boxes[:, 0]
            boxes[:, 0] = new_x1
            boxes[:, 2] = new_x2
            
        tensor_img = TF.to_tensor(resized_img)
        
        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([idx]),
            'area': areas,
            'iscrowd': iscrowd,
            'file_name': file_name,
            'anatomical_region': anatomy,
            'orig_size': (orig_w, orig_h)
        }
        
        return tensor_img, target

def detection_collate_fn(batch):
    return tuple(zip(*batch))
