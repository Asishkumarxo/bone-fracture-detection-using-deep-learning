"""
Controlled Experiment 2: Aspect-Aware Cropping & Multi-Crop Evaluation
=====================================================================
Evaluates whether replacing 224x224 zero-padding with aspect-aware single/multi-crop
preserves fracture signal and resolves false negatives on wide/tall radiographs.

Compares:
1. Baseline: Current 224x224 aspect-preserving resize with black zero-padding
2. Single Aspect-Aware Crop: Central square crop at native resolution -> 224x224
3. Overlapping Multi-Crop: 3 to 5 overlapping square crops across long dimension
4. Aggregation Strategies: Mean, Max, and Top-2 Mean probability

Evaluated on:
- 5 external test radiographs (AdobeStock wrist, wrist crack, 2.jpg, istock, pelvis)
- 107 audited unique test images from data/fracture_splits/all_unique_images.csv
"""

import os
import sys
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import torch
import torchvision.transforms as T
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score, confusion_matrix
)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.models import MultiTaskModel
from inference.model_registry import ModelRegistry
from inference.preprocessing import resize_and_pad

def get_crops(img: Image.Image, min_ar_threshold=1.15):
    """
    Generates single crop and overlapping multi-crops for an image.
    Preserves original aspect ratio and real radiographic pixels (zero black padding).
    """
    w, h = img.size
    ar = max(w, h) / min(w, h)
    short_dim = min(w, h)

    # 1. Single Central Crop
    if w >= h:
        x_start = (w - h) // 2
        single_box = (x_start, 0, x_start + h, h)
    else:
        y_start = (h - w) // 2
        single_box = (0, y_start, w, y_start + w)

    # 2. Overlapping Multi-Crops
    if ar <= min_ar_threshold:
        # Near square: 1 crop (the full/center square)
        multi_boxes = [single_box]
    elif ar <= 1.6:
        # Moderately elongated: 3 overlapping crops
        k = 3
        if w > h:
            step = (w - h) / (k - 1)
            multi_boxes = [(int(round(i * step)), 0, int(round(i * step)) + h, h) for i in range(k)]
        else:
            step = (h - w) / (k - 1)
            multi_boxes = [(0, int(round(i * step)), w, int(round(i * step)) + w) for i in range(k)]
    else:
        # Highly elongated (e.g. panoramic 2.19 AR): 5 overlapping crops
        k = 5
        if w > h:
            step = (w - h) / (k - 1)
            multi_boxes = [(int(round(i * step)), 0, int(round(i * step)) + h, h) for i in range(k)]
        else:
            step = (h - w) / (k - 1)
            multi_boxes = [(0, int(round(i * step)), w, int(round(i * step)) + w) for i in range(k)]

    return single_box, multi_boxes

def evaluate_crop_tensor(crop_img: Image.Image, model, transform, device):
    """Evaluates a single cropped image through best_model.pt."""
    rgb = crop_img.convert('RGB')
    tensor = transform(rgb).unsqueeze(0).to(device)
    with torch.no_grad():
        reg_logits, frac_logits = model(tensor)
        probs_r = torch.softmax(reg_logits, dim=-1)[0].cpu().numpy()
        prob_f = float(torch.sigmoid(frac_logits).view(-1)[0].cpu().item())
        top_r = int(np.argmax(probs_r))
        conf_r = float(probs_r[top_r])
    return prob_f, top_r, conf_r

def generate_visual_contact_sheet(img_path, baseline_padded, multi_boxes, crop_probs, crop_regions, output_path):
    """
    Creates a visual debugging contact sheet showing:
    1. Original image with color-coded crop bounding boxes
    2. Baseline 224x224 padded representation
    3. Individual crop images with their fracture probabilities
    """
    with Image.open(img_path) as orig_raw:
        orig = orig_raw.convert('RGB')
    w, h = orig.size

    # Draw boxes on original
    annotated = orig.copy()
    draw = ImageDraw.Draw(annotated)
    colors = ['#FF3366', '#33CC66', '#3399FF', '#FF9900', '#9933FF']
    
    # Scale line thickness with image size
    thick = max(3, int(min(w, h) / 150))
    for i, box in enumerate(multi_boxes):
        c = colors[i % len(colors)]
        for t in range(thick):
            draw.rectangle([box[0]+t, box[1]+t, box[2]-t, box[3]-t], outline=c)

    # Prepare sheet canvas
    # Top row: Annotated Original (scaled to height 250) + Baseline Padded (250x250)
    scale_top = 250.0 / h
    ann_w = int(round(w * scale_top))
    ann_resized = annotated.resize((ann_w, 250), Image.Resampling.BILINEAR)
    base_resized = baseline_padded.convert('RGB').resize((250, 250), Image.Resampling.BILINEAR)

    k = len(multi_boxes)
    crop_thumb_size = 180
    sheet_w = max(ann_w + 250 + 40, k * (crop_thumb_size + 20) + 20)
    sheet_h = 250 + 60 + crop_thumb_size + 80

    sheet = Image.new('RGB', (sheet_w, sheet_h), color='#1A1F2C')
    sdraw = ImageDraw.Draw(sheet)

    # Paste top row
    sheet.paste(ann_resized, (20, 40))
    sheet.paste(base_resized, (ann_w + 40, 40))
    sdraw.text((20, 15), f"Original Radiograph ({w}x{h}, AR={w/h:.2f}) with {k} Crop Windows", fill='#FFFFFF')
    sdraw.text((ann_w + 40, 15), "Current Baseline (224x224 Padded)", fill='#FFFFFF')

    # Bottom row: Crops
    y_crop = 340
    sdraw.text((20, 315), "Individual Extracted Crops (High-Res 224x224):", fill='#00D2FF')
    for i, box in enumerate(multi_boxes):
        c_img = orig.crop(box).resize((crop_thumb_size, crop_thumb_size), Image.Resampling.BILINEAR)
        x_pos = 20 + i * (crop_thumb_size + 20)
        sheet.paste(c_img, (x_pos, y_crop))
        c_color = colors[i % len(colors)]
        # Outline thumb with matching box color
        sdraw.rectangle([x_pos, y_crop, x_pos + crop_thumb_size, y_crop + crop_thumb_size], outline=c_color, width=3)
        prob = crop_probs[i] * 100
        reg_name = ModelRegistry.IDX_TO_REGION.get(crop_regions[i], f"R{crop_regions[i]}")
        label_txt = f"Crop {i}: {prob:4.1f}%\nAnat: {reg_name}"
        sdraw.text((x_pos, y_crop + crop_thumb_size + 5), label_txt, fill=c_color)

    sheet.save(output_path, quality=92)

def run_experiment():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Executing Controlled Experiment 2 on execution device: {device}")

    # Load best_model.pt
    ckpt_path = ModelRegistry.get_classifier_path()
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = MultiTaskModel(backbone='resnet50', num_regions=7, pretrained=False).to(device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    # Preprocessing transforms
    crop_transform = T.Compose([
        T.Resize((224, 224), Image.Resampling.BILINEAR),
        T.ToTensor(),
        T.Normalize(mean=ModelRegistry.CLASSIFIER_NORMALIZATION_MEAN,
                    std=ModelRegistry.CLASSIFIER_NORMALIZATION_STD)
    ])
    baseline_norm = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=ModelRegistry.CLASSIFIER_NORMALIZATION_MEAN,
                    std=ModelRegistry.CLASSIFIER_NORMALIZATION_STD)
    ])

    # =========================================================================
    # PART 1: EVALUATE 5 EXTERNAL TEST RADIOGRAPHS
    # =========================================================================
    external_images = [
        (r'c:\Users\arbaz\Downloads\AdobeStock_594927383-1.jpeg', 'Positive', 'Wrist Fracture'),
        (r'c:\Users\arbaz\Downloads\wrist crack image.jpg', 'Positive', 'Wrist Fracture'),
        (r'c:\Users\arbaz\Downloads\2.jpg', 'Positive', 'Hand/Wrist Fracture'),
        (r'c:\Users\arbaz\Downloads\istockphoto-471457370-612x612.jpg', 'Positive', 'Hand/Finger Fracture'),
        (r'c:\Users\arbaz\Downloads\AdobeStock_200285274.webp', 'Negative', 'Normal Pelvis')
    ]

    os.makedirs('reports/controlled_exp_2', exist_ok=True)
    external_results = []

    print("\n" + "="*110)
    print("PART 1: EXTERNAL RADIOGRAPHS EVALUATION ACROSS CROPPING STRATEGIES")
    print("="*110)
    print(f"{'Image':<32} | {'AR':<5} | {'Baseline':<9} | {'SingleCrop':<11} | {'MultiMean':<10} | {'MultiMax':<9} | {'Top2Mean':<9} | {'GT':<8}")
    print("-" * 110)

    for p, gt, desc in external_images:
        if not os.path.exists(p):
            print(f"File not found: {p}")
            continue
        name = os.path.basename(p)
        with Image.open(p) as raw_im:
            w, h = raw_im.size
            ar = w / h
            gray = raw_im.convert('L')
            
            # Baseline: 224x224 padded
            padded = resize_and_pad(gray, target_size=(224, 224), fill_color=0)
            base_tensor = baseline_norm(padded.convert('RGB')).unsqueeze(0).to(device)
            with torch.no_grad():
                reg_logits, frac_logits = model(base_tensor)
                base_prob = float(torch.sigmoid(frac_logits).view(-1)[0].cpu().item())
                probs_r = torch.softmax(reg_logits, dim=-1)[0].cpu().numpy()
                base_anat_idx = int(np.argmax(probs_r))
                base_anat = ModelRegistry.IDX_TO_REGION[base_anat_idx]
                base_anat_conf = float(probs_r[base_anat_idx])

            # Cropping strategies
            single_box, multi_boxes = get_crops(raw_im)

            # Single Crop
            single_crop = gray.crop(single_box)
            single_prob, _, _ = evaluate_crop_tensor(single_crop, model, crop_transform, device)

            # Multi-Crops
            crop_probs = []
            crop_regions = []
            for box in multi_boxes:
                c_im = gray.crop(box)
                cp_f, cp_r, _ = evaluate_crop_tensor(c_im, model, crop_transform, device)
                crop_probs.append(cp_f)
                crop_regions.append(cp_r)

            # Aggregations
            multi_mean = float(np.mean(crop_probs))
            multi_max = float(np.max(crop_probs))
            if len(crop_probs) >= 2:
                top2_mean = float(np.mean(sorted(crop_probs, reverse=True)[:2]))
            else:
                top2_mean = multi_max

            record = {
                'Image': name,
                'AR': round(ar, 2),
                'Baseline': round(base_prob, 4),
                'Baseline_Anat': f"{base_anat} ({base_anat_conf*100:.1f}%)",
                'SingleCrop': round(single_prob, 4),
                'MultiMean': round(multi_mean, 4),
                'MultiMax': round(multi_max, 4),
                'Top2Mean': round(top2_mean, 4),
                'GroundTruth': gt,
                'NumCrops': len(multi_boxes),
                'CropProbs': [round(x, 4) for x in crop_probs]
            }
            external_results.append(record)

            print(f"{name:<32} | {ar:5.2f} | {base_prob*100:6.2f}%   | {single_prob*100:6.2f}%     | {multi_mean*100:6.2f}%    | {multi_max*100:6.2f}%   | {top2_mean*100:6.2f}%   | {gt:<8}")

            # Generate visual contact sheet
            vis_out = os.path.join('reports', 'controlled_exp_2', f"vis_{os.path.splitext(name)[0]}.png")
            generate_visual_contact_sheet(p, padded, multi_boxes, crop_probs, crop_regions, vis_out)

    # =========================================================================
    # PART 2: REPRESENTATIVE TEST DATASET EVALUATION (N=107 Audited Images)
    # =========================================================================
    test_csv = 'data/fracture_splits/all_unique_images.csv'
    df_test = pd.read_csv(test_csv)
    print("\n" + "="*110)
    print(f"PART 2: TEST-SET SUBSET EVALUATION (N={len(df_test)} Audited Radiographs on Disk)")
    print("="*110)

    gt_targets = df_test['fracture_target'].values.astype(int)
    all_base_probs = []
    all_single_probs = []
    all_multi_mean_probs = []
    all_multi_max_probs = []
    all_top2_probs = []
    all_regions = df_test['region'].tolist()

    for idx, row in df_test.iterrows():
        img_p = row['path']
        with Image.open(img_p) as im:
            gray = im.convert('L')
            
            # Baseline
            padded = resize_and_pad(gray, target_size=(224, 224), fill_color=0)
            base_t = baseline_norm(padded.convert('RGB')).unsqueeze(0).to(device)
            with torch.no_grad():
                _, f_logit = model(base_t)
                bp = float(torch.sigmoid(f_logit).view(-1)[0].cpu().item())
            all_base_probs.append(bp)

            # Single Crop
            sbox, mboxes = get_crops(im)
            sp, _, _ = evaluate_crop_tensor(gray.crop(sbox), model, crop_transform, device)
            all_single_probs.append(sp)

            # Multi Crop
            m_probs = []
            for b in mboxes:
                cp, _, _ = evaluate_crop_tensor(gray.crop(b), model, crop_transform, device)
                m_probs.append(cp)

            all_multi_mean_probs.append(float(np.mean(m_probs)))
            all_multi_max_probs.append(float(np.max(m_probs)))
            if len(m_probs) >= 2:
                all_top2_probs.append(float(np.mean(sorted(m_probs, reverse=True)[:2])))
            else:
                all_top2_probs.append(float(np.max(m_probs)))

    # Compute metrics for each strategy
    strategies = {
        'Baseline (224x224 Padded)': np.array(all_base_probs),
        'Single Aspect-Aware Crop': np.array(all_single_probs),
        'Multi-Crop (Mean)': np.array(all_multi_mean_probs),
        'Multi-Crop (Max)': np.array(all_multi_max_probs),
        'Multi-Crop (Top-2 Mean)': np.array(all_top2_probs)
    }

    metrics_summary = []
    print(f"{'Strategy':<26} | {'Accuracy':<9} | {'Precision':<9} | {'Recall':<8} | {'Spec':<8} | {'F1':<7} | {'ROC-AUC':<8} | {'FN / Pos':<10} | {'FP / Neg':<10}")
    print("-" * 110)

    for strat_name, probs in strategies.items():
        preds = (probs >= 0.50).astype(int)
        acc = accuracy_score(gt_targets, preds)
        prec = precision_score(gt_targets, preds, zero_division=0)
        rec = recall_score(gt_targets, preds, zero_division=0)
        cm = confusion_matrix(gt_targets, preds)
        tn, fp, fn, tp = cm.ravel()
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        f1 = f1_score(gt_targets, preds, zero_division=0)
        auc = roc_auc_score(gt_targets, probs)
        
        metrics_summary.append({
            'Strategy': strat_name,
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'Specificity': spec,
            'F1': f1,
            'ROC-AUC': auc,
            'FN': fn,
            'FP': fp,
            'TN': tn,
            'TP': tp
        })
        print(f"{strat_name:<26} | {acc*100:6.2f}%   | {prec*100:6.2f}%   | {rec*100:6.2f}% | {spec*100:6.2f}% | {f1:6.4f} | {auc:6.4f}   | {fn:2d} / {fn+tp:2d}    | {fp:2d} / {tn+fp:2d}")

    # =========================================================================
    # PART 3: PER-REGION RECALL (WRIST, HAND, ARM, ETC.)
    # =========================================================================
    print("\n" + "="*110)
    print("PART 3: FRACTURE RECALL BY ANATOMICAL REGION ACROSS STRATEGIES")
    print("="*110)
    print(f"{'Region':<12} | {'Positives':<10} | {'Baseline Recall':<16} | {'SingleCrop Recall':<18} | {'MultiMax Recall':<16} | {'Top2 Recall':<12}")
    print("-" * 110)

    df_res = pd.DataFrame({
        'region': all_regions,
        'target': gt_targets,
        'base': (strategies['Baseline (224x224 Padded)'] >= 0.50).astype(int),
        'single': (strategies['Single Aspect-Aware Crop'] >= 0.50).astype(int),
        'multimax': (strategies['Multi-Crop (Max)'] >= 0.50).astype(int),
        'top2': (strategies['Multi-Crop (Top-2 Mean)'] >= 0.50).astype(int)
    })

    for reg in sorted(df_res['region'].unique()):
        sub = df_res[df_res['region'] == reg]
        n_pos = sum(sub['target'] == 1)
        if n_pos == 0: continue
        r_base = sum((sub['target'] == 1) & (sub['base'] == 1)) / n_pos
        r_single = sum((sub['target'] == 1) & (sub['single'] == 1)) / n_pos
        r_max = sum((sub['target'] == 1) & (sub['multimax'] == 1)) / n_pos
        r_top2 = sum((sub['target'] == 1) & (sub['top2'] == 1)) / n_pos
        print(f"{reg:<12} | {n_pos:2d} pos     | {r_base*100:5.1f}%          | {r_single*100:5.1f}%            | {r_max*100:5.1f}%          | {r_top2*100:5.1f}%")

    # Save summary dataframe
    df_ext = pd.DataFrame(external_results)
    df_ext.to_csv('reports/controlled_exp_2/external_evaluation_results.csv', index=False)
    pd.DataFrame(metrics_summary).to_csv('reports/controlled_exp_2/test_subset_metrics.csv', index=False)
    print("\nResults and visualization sheets saved to reports/controlled_exp_2/")

if __name__ == '__main__':
    run_experiment()
