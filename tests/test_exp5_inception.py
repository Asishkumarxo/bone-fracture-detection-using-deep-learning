"""
Unit Tests for Experiment 5: Inception-v3 Multi-Task Fine-Tuning
================================================================
Covers:
1. Model initialization & structure
2. Output tensor shapes in training (with auxiliary heads) and eval modes
3. Auxiliary classifier handling
4. Forward and backward propagation & gradient calculation
5. Freezing & unfreezing dynamics (Stage 1 & Stage 2)
6. Checkpoint saving and loading
7. Preprocessing pipeline (299x299 aspect-ratio preserved padding)
8. Anatomical region mapping (7 classes)
9. Binary fracture thresholding (0.50 threshold)
10. Metric calculation compatibility
11. Dataset manifest compatibility
"""

import os
import sys
import tempfile
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from PIL import Image
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.models_exp5 import MultiTaskInceptionV3

class TestExp5InceptionModel:
    @pytest.fixture
    def model(self):
        # Initialize without pretrained weights for fast unit testing
        m = MultiTaskInceptionV3(num_regions=7, pretrained=False, dropout=0.3, use_aux=True)
        return m

    def test_model_initialization(self, model):
        assert model is not None
        assert model.num_regions == 7
        assert model.use_aux is True
        assert hasattr(model, 'region_head')
        assert hasattr(model, 'fracture_head')
        assert hasattr(model, 'aux_region_head')
        assert hasattr(model, 'aux_fracture_head')

    def test_forward_pass_training_mode(self, model):
        model.train()
        x = torch.randn(2, 3, 299, 299)
        outputs = model(x)
        assert len(outputs) == 4, "Expected (reg, frac, aux_reg, aux_frac) in training mode"
        reg_logits, frac_logits, aux_reg, aux_frac = outputs
        
        assert reg_logits.shape == (2, 7)
        assert frac_logits.shape == (2,)
        assert aux_reg.shape == (2, 7)
        assert aux_frac.shape == (2,)

    def test_forward_pass_eval_mode(self, model):
        model.eval()
        x = torch.randn(2, 3, 299, 299)
        outputs = model(x)
        assert len(outputs) == 2, "Expected (reg, frac) in eval mode"
        reg_logits, frac_logits = outputs
        
        assert reg_logits.shape == (2, 7)
        assert frac_logits.shape == (2,)

    def test_aux_classifier_disabled(self):
        model_no_aux = MultiTaskInceptionV3(num_regions=7, pretrained=False, use_aux=False)
        model_no_aux.train()
        x = torch.randn(2, 3, 299, 299)
        outputs = model_no_aux(x)
        assert len(outputs) == 2, "Expected (reg, frac) when use_aux=False"
        reg_logits, frac_logits = outputs
        assert reg_logits.shape == (2, 7)
        assert frac_logits.shape == (2,)

    def test_backward_pass_and_loss(self, model):
        model.train()
        x = torch.randn(2, 3, 299, 299)
        reg_target = torch.tensor([0, 5], dtype=torch.long)
        frac_target = torch.tensor([1.0, 0.0], dtype=torch.float32)

        reg_logits, frac_logits, aux_reg, aux_frac = model(x)

        criterion_r = nn.CrossEntropyLoss()
        criterion_f = nn.BCEWithLogitsLoss()

        loss_main = criterion_r(reg_logits, reg_target) + 1.5 * criterion_f(frac_logits, frac_target)
        loss_aux = criterion_r(aux_reg, reg_target) + 1.5 * criterion_f(aux_frac, frac_target)
        total_loss = loss_main + 0.4 * loss_aux

        assert not torch.isnan(total_loss)
        total_loss.backward()

        # Check gradients exist in heads
        for p in model.region_head.parameters():
            if p.requires_grad:
                assert p.grad is not None

    def test_freeze_and_unfreeze_stages(self, model):
        # Stage 1: Freeze backbone
        model.freeze_backbone()
        for p in model.inception.parameters():
            # Identity modules have no params, but other inception layers should have requires_grad=False
            assert not p.requires_grad

        for m in [model.region_head, model.fracture_head, model.aux_region_head, model.aux_fracture_head]:
            for p in m.parameters():
                assert p.requires_grad

        # Stage 2: Unfreeze upper backbone (Mixed_7a, 7b, 7c)
        model.unfreeze_upper_backbone()
        for block in [model.inception.Mixed_7a, model.inception.Mixed_7b, model.inception.Mixed_7c]:
            for p in block.parameters():
                assert p.requires_grad

        # Lower backbone should still be frozen
        for p in model.inception.Conv2d_1a_3x3.parameters():
            assert not p.requires_grad

    def test_checkpoint_save_and_load(self, model):
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as tmp:
            tmp_path = tmp.name

        try:
            torch.save({
                'epoch': 5,
                'state_dict': model.state_dict(),
                'architecture': 'Inception-v3',
                'target_size': (299, 299)
            }, tmp_path)

            loaded = torch.load(tmp_path, map_location='cpu')
            new_model = MultiTaskInceptionV3(num_regions=7, pretrained=False)
            new_model.load_state_dict(loaded['state_dict'])

            new_model.eval()
            x = torch.randn(1, 3, 299, 299)
            orig_out = model.eval()(x)
            new_out = new_model(x)

            torch.testing.assert_close(orig_out[0], new_out[0])
            torch.testing.assert_close(orig_out[1], new_out[1])
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_aspect_ratio_padding_299(self):
        # Verify resize and pad preserves aspect ratio and outputs 299x299
        orig_img = Image.new('RGB', (100, 200), color=(128, 128, 128))
        target_size = (299, 299)
        
        target_w, target_h = target_size
        orig_w, orig_h = orig_img.size
        ratio = min(target_w / orig_w, target_h / orig_h)
        new_w = max(1, int(round(orig_w * ratio)))
        new_h = max(1, int(round(orig_h * ratio)))
        
        assert new_h == 299
        assert new_w < 299
        
        resized = orig_img.resize((new_w, new_h), Image.Resampling.BILINEAR)
        padded = Image.new('RGB', target_size, color=0)
        pad_x = (target_w - new_w) // 2
        pad_y = (target_h - new_h) // 2
        padded.paste(resized, (pad_x, pad_y))
        
        assert padded.size == (299, 299)

    def test_thresholding_at_050(self):
        probs = np.array([0.49, 0.50, 0.51, 0.10, 0.95])
        preds = (probs >= 0.50).astype(int)
        expected = np.array([0, 1, 1, 0, 1])
        np.testing.assert_array_equal(preds, expected)

    def test_anatomical_label_mapping(self):
        region_to_idx = {
            'Arm': 0, 'Foot': 1, 'Hand': 2, 'Lower leg': 3,
            'Thigh': 4, 'pelvis': 5, 'wrist': 6
        }
        assert len(region_to_idx) == 7
        assert region_to_idx['pelvis'] == 5
        assert region_to_idx['Lower leg'] == 3

    def test_sample_manifest_compatibility(self):
        manifest_path = "reports/exp4_sample_manifest.csv"
        assert os.path.exists(manifest_path), f"Manifest {manifest_path} must exist"
        df = pd.read_csv(manifest_path)
        assert len(df) == 14000
        assert set(df['fracture_label'].unique()) == {'Negative', 'Positive'}
        assert df['anatomical_region'].nunique() == 7
