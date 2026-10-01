"""
Diagnostic Image Annotation & Visualization Component
======================================================
Draws non-destructive overlays depicting detected fracture bounding boxes,
anatomical region, confidence scores, and status banners.
Saves visualizations strictly into an isolated output directory.
"""

import os
from typing import Union, Dict, Any, Optional
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from .preprocessing import load_and_validate_image

def create_annotated_visualization(
    image_input: Union[str, Image.Image],
    prediction_result: Dict[str, Any],
    output_dir: str = "visualizations",
    filename: Optional[str] = None
) -> str:
    """
    Renders a high-resolution diagnostic visual overlay image.
    
    Parameters:
    -----------
    image_input : str or PIL.Image
        Input X-ray image.
    prediction_result : dict
        Structured output containing anatomical_region, anatomical_confidence,
        fracture, fracture_confidence, localization_available, localization, caption.
    output_dir : str
        Directory to save the visualization.
    filename : str, optional
        Custom filename. If None, derived from input or timestamp.
        
    Returns:
    --------
    output_path : str
        Absolute path to the saved visual image.
    """
    orig_img = load_and_validate_image(image_input).convert('RGB')
    orig_w, orig_h = orig_img.size
    
    os.makedirs(output_dir, exist_ok=True)
    
    if filename is None:
        if isinstance(image_input, str):
            base = os.path.splitext(os.path.basename(image_input))[0]
            filename = f"annotated_{base}.png"
        else:
            filename = "annotated_xray.png"
            
    output_path = os.path.join(output_dir, filename)
    
    fig, ax = plt.subplots(figsize=(8, 8), dpi=200)
    ax.imshow(orig_img)
    
    is_fracture = prediction_result.get('fracture', False)
    region = prediction_result.get('anatomical_region', 'Unknown').upper()
    reg_conf = prediction_result.get('anatomical_confidence', 0.0)
    frac_conf = prediction_result.get('fracture_confidence', 0.0)
    
    status_color = '#e60000' if is_fracture else '#009933'
    status_text = "FRACTURE DETECTED" if is_fracture else "NO FRACTURE"
    
    # Draw localized bounding boxes if available
    loc_available = prediction_result.get('localization_available', False)
    boxes = prediction_result.get('localization', None)
    
    if loc_available and boxes:
        for b_info in boxes:
            box = b_info.get('box', b_info.get('box_2d', []))
            conf = b_info.get('confidence', frac_conf)
            if len(box) == 4:
                x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
                w = max(1.0, x2 - x1)
                h = max(1.0, y2 - y1)
                
                rect = patches.Rectangle(
                    (x1, y1), w, h,
                    linewidth=3.0, edgecolor='#ff2222', facecolor='none'
                )
                ax.add_patch(rect)
                
                ax.text(
                    x1, max(0, y1 - 8),
                    f" Fracture: {conf*100:.1f}% ",
                    color='white', fontsize=9, fontweight='bold',
                    bbox=dict(facecolor='#ff2222', edgecolor='none', boxstyle='round,pad=0.25')
                )
                
    title_text = (
        f"Region: {region} ({reg_conf*100:.1f}%) | "
        f"Status: {status_text} ({frac_conf*100:.1f}%)"
    )
    ax.set_title(title_text, fontsize=12, fontweight='bold', color=status_color, pad=12)
    ax.axis('off')
    
    caption = prediction_result.get('caption', '')
    if caption:
        plt.figtext(
            0.5, 0.02, f"\"{caption}\"",
            wrap=True, horizontalalignment='center', fontsize=9,
            style='italic', bbox=dict(facecolor='#f0f4f8', edgecolor='#334466', boxstyle='round,pad=0.4')
        )
        
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.08)
    plt.savefig(output_path)
    plt.close()
    
    return os.path.abspath(output_path)
