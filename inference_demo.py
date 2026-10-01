"""
Interactive Multi-Task Inference & Visual Demonstration
=======================================================
Accepts a single bone X-ray image and visualizes:
1. Original X-Ray
2. Diagnostic overlay with predicted anatomical region & confidence
3. Predicted fracture probability & status
4. Spatial lesion localization bounding box with confidence score (when fracture detected)
5. Disciplined factual natural-language caption

Usage:
------
python inference_demo.py --image path/to/xray.png --output demo_prediction.png
"""

import os
import sys
import argparse
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from predict import run_inference

def run_demo(args):
    print(f"\nRunning end-to-end multi-task inference on: {args.image}")
    
    # 1. Run core inference pipeline
    result = run_inference(
        image_path=args.image,
        classifier_ckpt=args.classifier_ckpt,
        detector_ckpt=args.detector_ckpt,
        fracture_threshold=args.fracture_threshold,
        detector_threshold=args.detector_threshold
    )
    
    # 2. Load original image for visualization
    orig_img = Image.open(args.image).convert('RGB')
    orig_w, orig_h = orig_img.size
    
    # 3. Create high-resolution multi-panel visual figure
    fig, axes = plt.subplots(1, 2, figsize=(15, 7), dpi=200)
    
    # Panel 1: Original X-Ray
    axes[0].imshow(orig_img)
    axes[0].set_title(f"Input Radiograph\nResolution: {orig_w} × {orig_h}", fontsize=13, fontweight='bold', pad=10)
    axes[0].axis('off')
    
    # Panel 2: Diagnostic Overlay
    axes[1].imshow(orig_img)
    status_title = "FRACTURE DETECTED" if result['fracture'] else "NORMAL / NO FRACTURE"
    status_color = 'red' if result['fracture'] else 'green'
    
    # Draw bounding boxes if localization is available
    if result['localization_available']:
        for box_info in result['localization']:
            b = box_info['box_original']
            conf = box_info['confidence']
            x1, y1, x2, y2 = b[0], b[1], b[2], b[3]
            width = x2 - x1
            height = y2 - y1
            
            # Add rectangle patch
            rect = patches.Rectangle(
                (x1, y1), width, height,
                linewidth=3.5, edgecolor='#ff2222', facecolor='none'
            )
            axes[1].add_patch(rect)
            
            # Label banner
            axes[1].text(
                x1, max(0, y1 - 10),
                f" Fracture ({conf*100:.1f}%) ",
                color='white', fontsize=10, fontweight='bold',
                bbox=dict(facecolor='#ff2222', edgecolor='none', boxstyle='round,pad=0.3')
            )
            
    axes[1].set_title(
        f"Diagnostic Overlay: {status_title}\n"
        f"Region: {result['anatomical_region'].upper()} ({result['anatomical_confidence']*100:.1f}%) | "
        f"Fracture Prob: {result['fracture_confidence']*100:.1f}%",
        fontsize=13, fontweight='bold', color=status_color, pad=10
    )
    axes[1].axis('off')
    
    # Overall Title Banner with Generated Caption
    caption_wrapped = result['caption']
    plt.suptitle(
        f"BONE FRACTURE MULTI-TASK DIAGNOSTIC SYSTEM\n"
        f"Generated Factual Caption: \"{caption_wrapped}\"",
        fontsize=12, fontweight='bold', y=0.98,
        bbox=dict(facecolor='#f0f4f8', edgecolor='#334466', boxstyle='round,pad=0.5')
    )
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.86)
    
    output_path = args.output
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path)
    if getattr(args, 'show', False):
        try:
            plt.show()
        except Exception as e:
            print(f"Warning: Could not display window ({e}). Visualization saved to file.", file=sys.stderr)
    plt.close()
    
    # 4. Formatted Terminal Summary
    print("\n" + "="*65)
    print("         END-TO-END INFERENCE DEMO RESULTS")
    print("="*65)
    print(f"Input Image:             {args.image}")
    print(f"Predicted Region:        {result['anatomical_region'].upper()} ({result['anatomical_confidence']*100:.2f}%)")
    print(f"Fracture Status:         {'POSITIVE (Fracture)' if result['fracture'] else 'NEGATIVE (Normal)'}")
    print(f"Fracture Probability:    {result['fracture_confidence']*100:.2f}%")
    print(f"Localization Supported:  {result['localization_available']}")
    
    if result['localization_available']:
        print("Detected Fracture Lesion(s):")
        for idx, b_info in enumerate(result['localization']):
            coords = b_info.get('box_original', b_info.get('box', []))
            conf = b_info.get('confidence', 0.0)
            print(f"  [{idx+1}] BBox: {coords} | Conf: {conf*100:.2f}%")
            
    print(f"\nGenerated Clinical Caption:\n  \"{result['caption']}\"")
    print(f"\nVisualization saved to:  {output_path}")
    print("="*65 + "\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Interactive Bone Fracture Inference Demo')
    parser.add_argument('--image', type=str, required=True, help='Path to input X-ray image')
    parser.add_argument('--output', type=str, default='demo_prediction.png', help='Path to save visualization image')
    parser.add_argument('--show', action='store_true', help='Display interactive window with matplotlib')
    parser.add_argument('--classifier_ckpt', type=str, default='best_model.pt', help='Classifier model checkpoint')
    parser.add_argument('--detector_ckpt', type=str, default='best_detector.pt', help='Detector model checkpoint')
    parser.add_argument('--fracture_threshold', type=float, default=0.50, help='Decision threshold for fracture presence')
    parser.add_argument('--detector_threshold', type=float, default=0.20, help='Confidence threshold for localization')
    args = parser.parse_args()
    
    run_demo(args)
