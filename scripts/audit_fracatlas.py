import os
import json
import glob
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

def audit_fracatlas(base_dir='E:/FracAtlas'):
    print(f'Starting FracAtlas Dataset Audit on: {base_dir}')
    
    # 1. Dataset.csv inspection
    df_meta = pd.read_csv(os.path.join(base_dir, 'dataset.csv'))
    total_images_meta = len(df_meta)
    fractured_meta = (df_meta['fractured'] == 1).sum()
    non_fractured_meta = (df_meta['fractured'] == 0).sum()
    total_fracture_instances_meta = df_meta['fracture_count'].sum()
    
    # Anatomical counts in dataset.csv
    anatomy_cols = ['hand', 'leg', 'hip', 'shoulder', 'mixed']
    anatomy_counts = {col: int(df_meta[col].sum()) for col in anatomy_cols}
    view_cols = ['frontal', 'lateral', 'oblique']
    view_counts = {col: int(df_meta[col].sum()) for col in view_cols}
    
    # 2. Check images on disk
    img_fractured_dir = os.path.join(base_dir, 'images', 'Fractured')
    img_non_fractured_dir = os.path.join(base_dir, 'images', 'Non_fractured')
    
    files_frac = os.listdir(img_fractured_dir) if os.path.exists(img_fractured_dir) else []
    files_non_frac = os.listdir(img_non_fractured_dir) if os.path.exists(img_non_fractured_dir) else []
    total_disk_images = len(files_frac) + len(files_non_frac)
    
    # Map file_name to disk path
    file_to_path = {}
    for f in files_frac:
        file_to_path[f] = os.path.join(img_fractured_dir, f).replace('\\', '/')
    for f in files_non_frac:
        file_to_path[f] = os.path.join(img_non_fractured_dir, f).replace('\\', '/')
        
    # 3. Check COCO JSON Annotations
    coco_path = os.path.join(base_dir, 'Annotations', 'COCO JSON', 'COCO_fracture_masks.json')
    with open(coco_path, 'r') as f:
        coco = json.load(f)
        
    coco_images = coco.get('images', [])
    coco_annotations = coco.get('annotations', [])
    coco_categories = coco.get('categories', [])
    
    image_id_to_file = {img['id']: img['file_name'] for img in coco_images}
    image_id_to_dims = {img['id']: (img['width'], img['height']) for img in coco_images}
    
    # 4. Annotation Integrity and Malformed BBox / Mask Detection
    malformed_count = 0
    malformed_reasons = []
    converted_records = []
    
    # Image dimensions stats
    widths = []
    heights = []
    aspect_ratios = []
    
    # Group annotations by image_id
    ann_by_img = {}
    for ann in coco_annotations:
        img_id = ann['image_id']
        ann_by_img.setdefault(img_id, []).append(ann)
        
    for img_obj in coco_images:
        img_id = img_obj['id']
        file_name = img_obj['file_name']
        w = img_obj['width']
        h = img_obj['height']
        widths.append(w)
        heights.append(h)
        aspect_ratios.append(w / h)
        
        disk_path = file_to_path.get(file_name, None)
        
        # Pull anatomical label from dataset.csv
        meta_row = df_meta[df_meta['image_id'] == file_name]
        primary_anatomy = 'unknown'
        if not meta_row.empty:
            for col in anatomy_cols:
                if meta_row[col].values[0] == 1:
                    primary_anatomy = col
                    break
                    
        anns = ann_by_img.get(img_id, [])
        for ann in anns:
            bbox = ann.get('bbox', []) # [x, y, width, height]
            seg = ann.get('segmentation', [])
            area = ann.get('area', 0)
            
            is_malformed = False
            reasons = []
            
            if len(bbox) != 4:
                is_malformed = True
                reasons.append('BBox does not have 4 coordinates')
            else:
                bx, by, bw, bh = bbox
                if bw <= 0 or bh <= 0:
                    is_malformed = True
                    reasons.append(f'Non-positive bbox dimension (w={bw}, h={bh})')
                if bx < 0 or by < 0:
                    is_malformed = True
                    reasons.append(f'Negative bbox origin (x={bx}, y={by})')
                if bx + bw > w + 1 or by + bh > h + 1:
                    is_malformed = True
                    reasons.append(f'BBox out of image bounds (x2={bx+bw} > {w} or y2={by+bh} > {h})')
                    
            if not seg or len(seg) == 0:
                is_malformed = True
                reasons.append('Empty segmentation polygon')
            else:
                for poly in seg:
                    if len(poly) < 6: # At least 3 points (x, y)
                        is_malformed = True
                        reasons.append(f'Degenerate polygon (< 3 vertices: {len(poly)} coords)')
                        
            if is_malformed:
                malformed_count += 1
                malformed_reasons.append(f"{file_name} (Ann {ann['id']}): " + '; '.join(reasons))
                
            converted_records.append({
                'image_id': file_name,
                'coco_image_id': img_id,
                'ann_id': ann['id'],
                'image_path': disk_path,
                'width': w,
                'height': h,
                'anatomical_region': primary_anatomy,
                'x_min': bbox[0],
                'y_min': bbox[1],
                'bbox_w': bbox[2],
                'bbox_h': bbox[3],
                'x_max': bbox[0] + bbox[2],
                'y_max': bbox[1] + bbox[3],
                'area': area,
                'segmentation_points': len(seg[0]) // 2 if seg and len(seg) > 0 else 0,
                'is_malformed': is_malformed,
                'malformed_reasons': '; '.join(reasons) if reasons else 'None'
            })
            
    df_clean_ann = pd.DataFrame(converted_records)
    os.makedirs('reports', exist_ok=True)
    df_clean_ann.to_csv('reports/fracatlas_annotations_clean.csv', index=False)
    print(f'Converted {len(df_clean_ann)} fracture annotations. Saved to reports/fracatlas_annotations_clean.csv')
    
    # 5. Visual Verification of Bounding Boxes and Masks
    print('Generating visual verifications of bounding boxes and masks on sample images...')
    os.makedirs('reports/plots', exist_ok=True)
    
    np.random.seed(42)
    sample_img_ids = np.random.choice(list(ann_by_img.keys()), 4, replace=False)
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10), dpi=200)
    
    for col_idx, img_id in enumerate(sample_img_ids):
        file_name = image_id_to_file[img_id]
        img_p = file_to_path[file_name]
        w, h = image_id_dims = image_id_to_dims[img_id]
        
        orig_img = Image.open(img_p).convert('RGB')
        
        # Row 1: Bounding boxes
        bbox_img = orig_img.copy()
        draw_b = ImageDraw.Draw(bbox_img)
        
        # Row 2: Segmentation masks
        mask_img = orig_img.copy()
        draw_m = ImageDraw.Draw(mask_img, 'RGBA')
        
        anns = ann_by_img[img_id]
        for ann in anns:
            bx, by, bw, bh = ann['bbox']
            # Draw bbox in bright red with outline
            draw_b.rectangle([bx, by, bx + bw, by + bh], outline=(255, 0, 0), width=6)
            
            # Draw polygon segmentation mask
            for poly in ann['segmentation']:
                pts = [(poly[i], poly[i+1]) for i in range(0, len(poly), 2)]
                draw_m.polygon(pts, fill=(255, 50, 50, 100), outline=(255, 255, 0, 220))
                
        # Plot BBox
        axes[0, col_idx].imshow(bbox_img)
        axes[0, col_idx].set_title(f'BBox: {file_name}\n({len(anns)} fracture instance{"s" if len(anns)>1 else ""})', fontsize=11, fontweight='bold')
        axes[0, col_idx].axis('off')
        
        # Plot Mask
        axes[1, col_idx].imshow(mask_img)
        axes[1, col_idx].set_title(f'Mask: {file_name}\nPolygon Segmentation', fontsize=11, fontweight='bold')
        axes[1, col_idx].axis('off')
        
    axes[0, 0].set_ylabel('Ground Truth BBox', fontsize=14, fontweight='bold', labelpad=10)
    axes[1, 0].set_ylabel('Ground Truth Mask', fontsize=14, fontweight='bold', labelpad=10)
    
    plt.suptitle('FracAtlas Dataset: Visual Ground Truth Verification (BBoxes & Segmentation Masks)', fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout()
    vis_path = 'reports/plots/fracatlas_visual_verification.png'
    plt.savefig(vis_path)
    plt.close()
    print(f'Saved visual verification figure to {vis_path}')
    
    # 6. Audit Metrics Summary
    audit_summary = {
        'total_images_in_dataset_csv': int(total_images_meta),
        'fractured_images_count': int(fractured_meta),
        'non_fractured_images_count': int(non_fractured_meta),
        'total_fracture_instances': int(total_fracture_instances_meta),
        'coco_annotated_images': int(len(coco_images)),
        'coco_annotation_instances': int(len(coco_annotations)),
        'anatomical_distribution': {k: int(v) for k, v in anatomy_counts.items()},
        'view_distribution': {k: int(v) for k, v in view_counts.items()},
        'malformed_annotations_count': int(malformed_count),
        'dimension_stats': {
            'min_width': int(min(widths)),
            'max_width': int(max(widths)),
            'mean_width': float(np.mean(widths)),
            'min_height': int(min(heights)),
            'max_height': int(max(heights)),
            'mean_height': float(np.mean(heights)),
            'mean_aspect_ratio': float(np.mean(aspect_ratios))
        }
    }
    
    with open('reports/fracatlas_audit_summary.json', 'w') as f:
        json.dump(audit_summary, f, indent=4)
    print('Saved audit summary to reports/fracatlas_audit_summary.json')
    print('FracAtlas audit complete!')

if __name__ == '__main__':
    audit_fracatlas()
