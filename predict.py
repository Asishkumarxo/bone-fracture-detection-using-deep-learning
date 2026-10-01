"""
Production Bone Fracture Inference Application
==============================================
Main inference entry point for end-to-end bone radiograph analysis.

Usage:
------
python predict.py --image path/to/xray.jpg [--output-dir path/to/dir] [--json]

Returns machine-readable JSON matching the strict schema:
{
  "anatomical_region": "...",
  "anatomical_confidence": 0.0,
  "fracture": true,
  "fracture_confidence": 0.0,
  "localization_available": false,
  "localization": null,
  "caption": "..."
}
"""

import os
import sys
import json
import time
import argparse
from typing import Dict, Any, Optional
from PIL import Image
import torch

from src.models import MultiTaskModel
from inference.model_registry import ModelRegistry
from inference.region_predictor import RegionPredictor
from inference.fracture_predictor import FracturePredictor
from inference.localization_predictor import LocalizationPredictor
from inference.caption_generator import generate_caption
from inference.visualize import create_annotated_visualization

class BoneFractureInferencePipeline:
    """
    Unified end-to-end inference pipeline coordinating multi-task classification,
    fracture thresholding, lesion localization, factual captioning, and visualization.
    """
    def __init__(
        self,
        classifier_ckpt: Optional[str] = None,
        detector_ckpt: Optional[str] = None,
        fracture_threshold: Optional[float] = None,
        detector_threshold: Optional[float] = None,
        uncertainty_threshold: Optional[float] = None,
        device: Optional[torch.device] = None
    ):
        self.device = device or ModelRegistry.get_device()
        self.fracture_threshold = fracture_threshold or ModelRegistry.FRACTURE_THRESHOLD
        self.detector_threshold = detector_threshold or ModelRegistry.DETECTOR_CONFIDENCE_THRESHOLD
        self.uncertainty_threshold = uncertainty_threshold or ModelRegistry.REGION_UNCERTAINTY_THRESHOLD
        
        # 1. Load shared multi-task classifier
        ckpt_path = classifier_ckpt or ModelRegistry.get_classifier_path()
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Classifier checkpoint not found at: {ckpt_path}. Place checkpoint in models/checkpoints/ or set CLASSIFIER_CHECKPOINT.")
            
        checkpoint = torch.load(ckpt_path, map_location=self.device)
        backbone = checkpoint.get('backbone', ModelRegistry.CLASSIFIER_BACKBONE)
        num_regions = len(checkpoint.get('region_to_idx', ModelRegistry.REGION_TO_IDX))
        
        shared_model = MultiTaskModel(backbone=backbone, num_regions=num_regions, pretrained=False)
        shared_model.load_state_dict(checkpoint['state_dict'])
        shared_model.to(self.device)
        shared_model.eval()
        self.shared_model = shared_model
        
        # 2. Instantiate predictors sharing the model
        self.region_predictor = RegionPredictor(model=self.shared_model, device=self.device)
        self.fracture_predictor = FracturePredictor(model=self.shared_model, threshold=self.fracture_threshold, device=self.device)
        
        # 3. Instantiate localization predictor
        detector_path = detector_ckpt or ModelRegistry.get_detector_path()
        self.localization_predictor = LocalizationPredictor(
            checkpoint_path=detector_path,
            score_thresh=self.detector_threshold,
            device=self.device
        )

    def analyze(self, image_path: str, output_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Runs the complete end-to-end inference pipeline on one input radiograph.
        """
        t0 = time.time()
        
        # Validate existence & image integrity
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Input image not found: {image_path}")
            
        try:
            with Image.open(image_path) as raw_im:
                raw_im.verify()
        except Exception as e:
            raise ValueError(f"Corrupted or invalid image file: {e}")
            
        # 1. Predict anatomical region
        reg_res = self.region_predictor.predict(image_path)
        anatomical_region = reg_res['predicted_region']
        anatomical_confidence = reg_res['confidence']
        
        # Uncertainty handling: flag if below calibrated confidence threshold
        is_uncertain = bool(anatomical_confidence < self.uncertainty_threshold)
        
        # 2. Predict fracture status
        frac_res = self.fracture_predictor.predict(image_path)
        is_fracture = frac_res['fracture']
        fracture_confidence = frac_res['fracture_probability']
        
        # 3. Fracture localization (only if fracture detected and model exists)
        loc_res = self.localization_predictor.predict(image_path, fracture_detected=is_fracture)
        localization_available = loc_res['localization_available']
        localization = loc_res['localization'] # list or None
        
        # 4. Generate disciplined factual caption
        caption = generate_caption({
            'anatomical_region': anatomical_region,
            'fracture': is_fracture,
            'localization_available': localization_available,
            'localization': localization
        })
        
        # 5. Assemble exact requested JSON schema
        result = {
            "anatomical_region": anatomical_region,
            "anatomical_confidence": anatomical_confidence,
            "fracture": is_fracture,
            "fracture_confidence": fracture_confidence,
            "localization_available": localization_available,
            "localization": localization,
            "caption": caption
        }
        
        if is_uncertain:
            result["uncertainty_warning"] = f"Prediction uncertain: anatomical confidence ({anatomical_confidence*100:.1f}%) is below uncertainty threshold ({self.uncertainty_threshold*100:.1f}%)."
            
        # 6. Optional visualization export
        if output_dir:
            vis_path = create_annotated_visualization(
                image_input=image_path,
                prediction_result=result,
                output_dir=output_dir
            )
            result["visualization_path"] = vis_path
            
        result["_elapsed_seconds"] = round(time.time() - t0, 3)
        return result

def run_inference(
    image_path: str,
    classifier_ckpt: Optional[str] = None,
    detector_ckpt: Optional[str] = None,
    fracture_threshold: Optional[float] = None,
    detector_threshold: Optional[float] = None,
    uncertainty_threshold: Optional[float] = None,
    output_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Convenience functional interface for running the inference pipeline.
    """
    pipeline = BoneFractureInferencePipeline(
        classifier_ckpt=classifier_ckpt,
        detector_ckpt=detector_ckpt,
        fracture_threshold=fracture_threshold,
        detector_threshold=detector_threshold,
        uncertainty_threshold=uncertainty_threshold
    )
    return pipeline.analyze(image_path=image_path, output_dir=output_dir)

def print_terminal_card(result: Dict[str, Any], image_path: str):
    print("\n" + "="*65)
    print("         BONE FRACTURE STRUCTURED DIAGNOSTIC PREDICTION")
    print("="*65)
    print(f"Input File:              {image_path}")
    print(f"Anatomical Region:       {result['anatomical_region'].upper()} ({result['anatomical_confidence']*100:.2f}%)")
    status_str = "POSITIVE (Fracture Detected)" if result['fracture'] else "NEGATIVE (Normal / No Fracture)"
    print(f"Fracture Status:         {status_str}")
    print(f"Fracture Confidence:     {result['fracture_confidence']*100:.2f}%")
    print(f"Localization Supported:  {result['localization_available']}")
    
    if result['localization_available'] and result['localization']:
        print("Detected Lesion Locations:")
        for idx, box_info in enumerate(result['localization']):
            box = box_info.get('box', box_info.get('box_2d', []))
            conf = box_info.get('confidence', 0.0)
            print(f"  [{idx+1}] BBox: {box} | Conf: {conf*100:.2f}%")
    else:
        print("Lesion Localization:     N/A (Normal bone or no box above confidence threshold)")
        
    print("\nGenerated Factual Caption:")
    print(f"  \"{result['caption']}\"")
    if "visualization_path" in result:
        print(f"\nAnnotated Visualization: {result['visualization_path']}")
    print("="*65 + "\n")

def main():
    parser = argparse.ArgumentParser(description='Production Bone Fracture Inference Pipeline')
    parser.add_argument('--image', type=str, required=True, help='Path to input X-ray image')
    parser.add_argument('--output-dir', '--output_dir', dest='output_dir', type=str, default=None,
                        help='Directory to save annotated visualization image')
    parser.add_argument('--classifier-ckpt', type=str, default=None, help='Custom classifier checkpoint path')
    parser.add_argument('--detector-ckpt', type=str, default=None, help='Custom detector checkpoint path')
    parser.add_argument('--fracture-threshold', type=float, default=None, help='Decision threshold for fracture presence')
    parser.add_argument('--detector-threshold', type=float, default=None, help='Confidence threshold for localization')
    parser.add_argument('--uncertainty-threshold', type=float, default=None, help='Threshold to flag uncertain anatomical predictions')
    parser.add_argument('--json', action='store_true', help='Force strict JSON stdout output')
    args = parser.parse_args()
    
    try:
        pipeline = BoneFractureInferencePipeline(
            classifier_ckpt=args.classifier_ckpt,
            detector_ckpt=args.detector_ckpt,
            fracture_threshold=args.fracture_threshold,
            detector_threshold=args.detector_threshold,
            uncertainty_threshold=args.uncertainty_threshold
        )
        result = pipeline.analyze(image_path=args.image, output_dir=args.output_dir)
    except Exception as e:
        err_dict = {"error": str(e)}
        print(json.dumps(err_dict, indent=2), file=sys.stderr)
        sys.exit(1)
        
    # Output formatting:
    # If stdout is piped/captured OR --json is passed, output pure JSON to stdout.
    # Otherwise, print terminal diagnostic summary and the structured JSON.
    output_dict = {
        "anatomical_region": result["anatomical_region"],
        "anatomical_confidence": result["anatomical_confidence"],
        "fracture": result["fracture"],
        "fracture_confidence": result["fracture_confidence"],
        "localization_available": result["localization_available"],
        "localization": result["localization"],
        "caption": result["caption"]
    }
    if "uncertainty_warning" in result:
        output_dict["uncertainty_warning"] = result["uncertainty_warning"]
    
    if args.json or not sys.stdout.isatty():
        print(json.dumps(output_dict, indent=2))
    else:
        print_terminal_card(result, args.image)
        print("Structured Prediction (JSON):")
        print(json.dumps(output_dict, indent=2))

if __name__ == '__main__':
    main()
