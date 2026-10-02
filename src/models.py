import torch
import torch.nn as nn
import torchvision.models as models

class MultiTaskModel(nn.Module):
    """
    Multi-Task Deep Learning Model for Bone X-Ray Analysis:
      - Shared Convolutional Backbone (ResNet-50 or EfficientNet)
      - Independent Anatomical-Region Classification Head (Multi-class)
      - Independent Fracture Classification Head (Binary)
    """
    def __init__(self, backbone='resnet50', num_regions=7, pretrained=True, dropout=0.3):
        super(MultiTaskModel, self).__init__()
        self.backbone_name = backbone.lower()
        self.num_regions = num_regions
        
        if self.backbone_name == 'resnet50':
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            base_model = models.resnet50(weights=weights)
            # Remove original linear classifier
            self.backbone = nn.Sequential(
                base_model.conv1,
                base_model.bn1,
                base_model.relu,
                base_model.maxpool,
                base_model.layer1,
                base_model.layer2,
                base_model.layer3,
                base_model.layer4,
                base_model.avgpool,
                nn.Flatten()
            )
            feat_dim = 2048
            self.layer4 = base_model.layer4
            
        elif self.backbone_name == 'resnet18':
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            base_model = models.resnet18(weights=weights)
            self.backbone = nn.Sequential(
                base_model.conv1,
                base_model.bn1,
                base_model.relu,
                base_model.maxpool,
                base_model.layer1,
                base_model.layer2,
                base_model.layer3,
                base_model.layer4,
                base_model.avgpool,
                nn.Flatten()
            )
            feat_dim = 512
            self.layer4 = base_model.layer4
            
        elif self.backbone_name == 'efficientnet_b0':
            weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            base_model = models.efficientnet_b0(weights=weights)
            self.backbone = nn.Sequential(
                base_model.features,
                base_model.avgpool,
                nn.Flatten()
            )
            feat_dim = 1280
            self.layer4 = base_model.features[-2:]
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
            
        # Task 1: Anatomical Region Classification Head
        self.region_head = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_regions)
        )
        
        # Task 2: Fracture Detection Classification Head
        self.fracture_head = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 1)
        )
        
    def forward(self, x):
        features = self.backbone(x)
        region_logits = self.region_head(features)
        fracture_logits = self.fracture_head(features).squeeze(-1)
        return region_logits, fracture_logits
        
    def get_features(self, x):
        return self.backbone(x)
        
    def freeze_backbone(self):
        """Stage 1: Freeze all backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = False
            
    def unfreeze_upper_backbone(self):
        """Stage 2: Unfreeze selected upper backbone layers for fine-tuning."""
        for param in self.layer4.parameters():
            param.requires_grad = True
            
    def unfreeze_all(self):
        """Unfreeze all model parameters."""
        for param in self.parameters():
            param.requires_grad = True

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
