"""
Multi-Task Inception-v3 Architecture for Experiment 5
=====================================================
Multi-Task Deep Learning Model for Bone Radiograph Analysis:
  - Inception-v3 Backbone pretrained on ImageNet
  - Multi-Scale Inception Feature Representations
  - Task 1: Anatomical Region Classification (7 classes)
  - Task 2: Binary Fracture Detection (1 output logit)
  - Auxiliary Classifier handling for stable training dynamics
"""

import torch
import torch.nn as nn
import torchvision.models as models

class MultiTaskInceptionV3(nn.Module):
    def __init__(self, num_regions=7, pretrained=True, dropout=0.3, use_aux=True):
        super(MultiTaskInceptionV3, self).__init__()
        self.num_regions = num_regions
        self.use_aux = use_aux
        
        # Load Inception-v3 with official pretrained weights
        if pretrained:
            weights = models.Inception_V3_Weights.DEFAULT
        else:
            weights = None
            
        base = models.inception_v3(weights=weights, aux_logits=True)
        self.inception = base
        
        # Replace original 1,000-class classifiers with Identity to extract pooled features
        self.inception.fc = nn.Identity()
        self.inception.AuxLogits.fc = nn.Identity()
        
        feat_dim = 2048
        aux_dim = 768
        
        # Primary Task Heads (Main Inception Features: 2,048-dim)
        self.region_head = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_regions)
        )
        
        self.fracture_head = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 1)
        )
        
        # Auxiliary Task Heads (Intermediate Inception Features: 768-dim)
        self.aux_region_head = nn.Sequential(
            nn.Linear(aux_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_regions)
        )
        
        self.aux_fracture_head = nn.Sequential(
            nn.Linear(aux_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        """
        Forward pass:
        - In training mode with use_aux=True:
            returns (region_logits, fracture_logits, aux_region_logits, aux_fracture_logits)
        - In eval mode (or use_aux=False):
            returns (region_logits, fracture_logits)
        """
        if self.training and self.use_aux:
            out = self.inception(x)
            main_feat = out.logits
            aux_feat = out.aux_logits
            
            reg_logits = self.region_head(main_feat)
            frac_logits = self.fracture_head(main_feat).squeeze(-1)
            
            aux_reg_logits = self.aux_region_head(aux_feat)
            aux_frac_logits = self.aux_fracture_head(aux_feat).squeeze(-1)
            
            return reg_logits, frac_logits, aux_reg_logits, aux_frac_logits
        else:
            out = self.inception(x)
            main_feat = out.logits if hasattr(out, 'logits') else out
            reg_logits = self.region_head(main_feat)
            frac_logits = self.fracture_head(main_feat).squeeze(-1)
            return reg_logits, frac_logits

    def freeze_backbone(self):
        """Stage 1: Freeze all Inception-v3 backbone parameters except custom prediction heads."""
        for name, param in self.inception.named_parameters():
            param.requires_grad = False
        # Ensure custom heads require grad
        for m in [self.region_head, self.fracture_head, self.aux_region_head, self.aux_fracture_head]:
            for param in m.parameters():
                param.requires_grad = True

    def unfreeze_upper_backbone(self):
        """
        Stage 2: Unfreeze upper Inception blocks (Mixed_7a, Mixed_7b, Mixed_7c) for fine-tuning.
        These blocks represent the highest-level semantic representations (~12.8M parameters).
        """
        for block in [self.inception.Mixed_7a, self.inception.Mixed_7b, self.inception.Mixed_7c]:
            for param in block.parameters():
                param.requires_grad = True

    def unfreeze_all(self):
        """Unfreeze all model parameters."""
        for param in self.parameters():
            param.requires_grad = True
