"""
Part 8: Grad-CAM Explainability Diagnostic for Anatomical Classifier
===================================================================
Computes class activation maps for layer4 of ResNet50 to visualize where
the model is attending when it misclassifies hip/pelvis as Lower Leg.
"""

import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from inference.model_registry import ModelRegistry
from inference.preprocessing import preprocess_for_classifier
from src.models import MultiTaskModel

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self._save_activation)
        self.target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, target_class_idx):
        self.model.eval()
        self.model.zero_grad()
        
        reg_logits, _ = self.model(input_tensor)
        score = reg_logits[0, target_class_idx]
        score.backward()
        
        # Pool gradients across spatial dimensions (global average pooling)
        weights = torch.mean(self.gradients, dim=[2, 3], keepdim=True)
        # Weighted combination of forward activation maps
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        # ReLU to keep only features that have a positive influence
        cam = F.relu(cam)
        
        # Resize to input tensor spatial dimensions
        cam = F.interpolate(cam, size=input_tensor.shape[2:], mode='bilinear', align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        
        # Normalize between 0 and 1
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)
            
        return cam, reg_logits

def main():
    device = torch.device('cpu') # Run Grad-CAM on CPU for deterministic gradient propagation
    os.makedirs('error_analysis/gradcam', exist_ok=True)
    
    # Load model
    ckpt = torch.load(ModelRegistry.CLASSIFIER_CHECKPOINT_PATH, map_location=device)
    model = MultiTaskModel(backbone='resnet50', num_regions=7, pretrained=False)
    model.load_state_dict(ckpt['state_dict'])
    model.to(device)
    model.eval()
    
    # Target layer: last bottleneck block of layer4 in ResNet50
    target_layer = model.backbone[7][-1]
    grad_cam = GradCAM(model, target_layer)
    
    # Image to analyze: pelvis misclassified as Lower leg
    img_path = 'processed/test/pelvis_patient16064_Negative_001.png'
    print(f"Generating Grad-CAM for: {img_path}")
    
    input_tensor = preprocess_for_classifier(img_path).to(device)
    input_tensor.requires_grad = True
    
    # 1. Grad-CAM for PREDICTED class: Lower leg (idx 3)
    cam_pred, logits = grad_cam.generate(input_tensor, target_class_idx=ModelRegistry.REGION_TO_IDX['Lower leg'])
    
    # 2. Grad-CAM for TRUE class: pelvis (idx 5)
    # Re-run for pelvis
    input_tensor_true = preprocess_for_classifier(img_path).to(device)
    input_tensor_true.requires_grad = True
    cam_true, _ = grad_cam.generate(input_tensor_true, target_class_idx=ModelRegistry.REGION_TO_IDX['pelvis'])
    
    # Open original image
    raw_img = Image.open(img_path).convert('RGB')
    orig_np = np.array(raw_img)
    
    # Resize heatmaps to original image dimensions
    from PIL import Image as PILImage
    cam_pred_img = PILImage.fromarray((cam_pred * 255).astype(np.uint8)).resize((orig_np.shape[1], orig_np.shape[0]), PILImage.BILINEAR)
    cam_true_img = PILImage.fromarray((cam_true * 255).astype(np.uint8)).resize((orig_np.shape[1], orig_np.shape[0]), PILImage.BILINEAR)
    
    cam_pred_norm = np.array(cam_pred_img) / 255.0
    cam_true_norm = np.array(cam_true_img) / 255.0
    
    # Colormaps
    cmap = plt.get_cmap('jet')
    heatmap_pred = cmap(cam_pred_norm)[:, :, :3]
    heatmap_true = cmap(cam_true_norm)[:, :, :3]
    
    overlay_pred = (orig_np / 255.0) * 0.55 + heatmap_pred * 0.45
    overlay_true = (orig_np / 255.0) * 0.55 + heatmap_true * 0.45
    
    # Save requested individual assets:
    # 1. Original image
    raw_img.save('error_analysis/gradcam/pelvis_16064_original.png')
    # 2. Grad-CAM heatmap
    plt.imsave('error_analysis/gradcam/pelvis_16064_heatmap.png', heatmap_pred)
    # 3. Overlay
    plt.imsave('error_analysis/gradcam/pelvis_16064_overlay.png', overlay_pred)
    
    # Save comprehensive diagnostic panel
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    axes[0].imshow(orig_np)
    axes[0].set_title("1. Original Radiograph\n(True: Pelvis)", fontsize=11, fontweight='bold')
    axes[0].axis('off')
    
    axes[1].imshow(heatmap_pred)
    axes[1].set_title("2. Grad-CAM Heatmap\n(Evidence for 'Lower Leg')", fontsize=11, fontweight='bold')
    axes[1].axis('off')
    
    axes[2].imshow(overlay_pred)
    axes[2].set_title("3. Overlay on 'Lower Leg'\n(Salient cortical shaft edges)", fontsize=11, fontweight='bold')
    axes[2].axis('off')
    
    axes[3].imshow(overlay_true)
    axes[3].set_title("4. Overlay on 'Pelvis'\n(True class attention)", fontsize=11, fontweight='bold')
    axes[3].axis('off')
    
    plt.tight_layout()
    plt.savefig('error_analysis/gradcam/pelvis_16064_gradcam_panel.png', dpi=200, bbox_inches='tight')
    plt.close()
    
    print("Grad-CAM generation complete! Saved files to error_analysis/gradcam/")

if __name__ == '__main__':
    main()
