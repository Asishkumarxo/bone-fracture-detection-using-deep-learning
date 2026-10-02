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
    Unified end-to-end resolution-aware inference pipeline coordinating
    resolution routing, multi-task classification, fracture thresholding,
    lesion localization, factual captioning, and visualization.
    """
    def __init__(
        self,
        classifier_ckpt: Optional[str] = None,
        exp3_classifier_ckpt: Optional[str] = None,
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
        
        # 1. Load baseline multi-task classifier (best_model.pt) for low-resolution branch
        base_ckpt_path = classifier_ckpt or ModelRegistry.get_baseline_classifier_path()
        if not os.path.exists(base_ckpt_path):
            raise FileNotFoundError(f"Baseline classifier checkpoint not found at: {base_ckpt_path}.")
            
        base_checkpoint = torch.load(base_ckpt_path, map_location=self.device)
        base_backbone = base_checkpoint.get('backbone', ModelRegistry.CLASSIFIER_BACKBONE)
        base_num_regions = len(base_checkpoint.get('region_to_idx', ModelRegistry.REGION_TO_IDX))
        
        baseline_model = MultiTaskModel(backbone=base_backbone, num_regions=base_num_regions, pretrained=False)
        baseline_model.load_state_dict(base_checkpoint['state_dict'])
        baseline_model.to(self.device)
        baseline_model.eval()
        self.baseline_model = baseline_model
        self.shared_model = baseline_model # Backward compatibility
        
        self.region_predictor = RegionPredictor(model=self.baseline_model, device=self.device)
        fracture_ckpt = ModelRegistry.get_fracture_path()
        if fracture_ckpt and os.path.exists(fracture_ckpt):
            self.fracture_predictor = FracturePredictor(checkpoint_path=fracture_ckpt, threshold=self.fracture_threshold, device=self.device)
        else:
            self.fracture_predictor = FracturePredictor(model=self.baseline_model, threshold=self.fracture_threshold, device=self.device)
            
        # 2. Load high-resolution Experiment 3 classifier (exp3_resnet50_448.pt) for high-resolution branch
        exp3_ckpt_path = exp3_classifier_ckpt or ModelRegistry.get_exp3_classifier_path()
        if os.path.exists(exp3_ckpt_path):
            exp3_checkpoint = torch.load(exp3_ckpt_path, map_location=self.device)
            exp3_backbone = exp3_checkpoint.get('backbone', ModelRegistry.CLASSIFIER_BACKBONE)
            exp3_num_regions = len(exp3_checkpoint.get('region_to_idx', ModelRegistry.REGION_TO_IDX))
            
            exp3_model = MultiTaskModel(backbone=exp3_backbone, num_regions=exp3_num_regions, pretrained=False)
            exp3_model.load_state_dict(exp3_checkpoint['state_dict'])
            exp3_model.to(self.device)
            exp3_model.eval()
            self.exp3_model = exp3_model
            self.exp3_region_predictor = RegionPredictor(model=self.exp3_model, device=self.device)
            self.exp3_fracture_predictor = FracturePredictor(model=self.exp3_model, threshold=self.fracture_threshold, device=self.device)
        else:
            self.exp3_model = None
            self.exp3_region_predictor = None
            self.exp3_fracture_predictor = None
        
        # 3. Instantiate localization predictor
        detector_path = detector_ckpt or ModelRegistry.get_detector_path()
        self.localization_predictor = LocalizationPredictor(
            checkpoint_path=detector_path,
            score_thresh=self.detector_threshold,
            device=self.device
        )

    def analyze(self, image_path: Any, output_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Runs the complete resolution-aware inference pipeline on one input radiograph.
        Inspects original image dimensions BEFORE preprocessing and routes accordingly:
        - min(width, height) >= 300 -> exp3_resnet50_448.pt (448x448)
        - min(width, height) < 300  -> best_model.pt (224x224)
        """
        t0 = time.time()
        
        from inference.preprocessing import load_and_validate_image
        img = load_and_validate_image(image_path)
        orig_w, orig_h = img.size
        
        # Resolution-aware routing decision before any resize operation
        routing_info = ModelRegistry.route_image(orig_w, orig_h)
        selected_model_name = routing_info["selected_model"]
        routing_res = routing_info["routing_resolution"]
        target_size = routing_info["target_size"]
        
        # Select appropriate model branch
        if selected_model_name == ModelRegistry.EXP3_CHECKPOINT_FILENAME and self.exp3_model is not None:
            reg_predictor = self.exp3_region_predictor
            frac_predictor = self.exp3_fracture_predictor
        else:
            reg_predictor = self.region_predictor
            frac_predictor = self.fracture_predictor
            
        # 1. Predict anatomical region
        reg_res = reg_predictor.predict(img, target_size=target_size)
        anatomical_region = reg_res['predicted_region']
        anatomical_confidence = reg_res['confidence']
        
        # Uncertainty handling: flag if below calibrated confidence threshold
        is_uncertain = bool(anatomical_confidence < self.uncertainty_threshold)
        
        # 2. Predict fracture status (decision threshold = 0.50 for BOTH branches)
        frac_res = frac_predictor.predict(img, target_size=target_size)
        is_fracture = frac_res['fracture']
        fracture_confidence = frac_res['fracture_probability']
        
        # 3. Fracture localization (only if fracture detected and model exists)
        loc_res = self.localization_predictor.predict(img, fracture_detected=is_fracture)
        localization_available = loc_res['localization_available']
        localization = loc_res['localization'] # list or None
        
        # 4. Generate disciplined factual caption
        caption = generate_caption({
            'anatomical_region': anatomical_region,
            'fracture': is_fracture,
            'localization_available': localization_available,
            'localization': localization
        })
        
        # Compute Top-3 predictions for uncertainty inspection
        dist = reg_res.get('probability_distribution', {})
        top_3 = sorted(dist.items(), key=lambda x: x[1], reverse=True)[:3]
        top_predictions = [{"region": r, "confidence": round(p, 4)} for r, p in top_3]
        
        # 5. Assemble exact requested JSON schema with routing metadata
        result = {
            "anatomical_region": anatomical_region,
            "anatomical_confidence": anatomical_confidence,
            "fracture": is_fracture,
            "fracture_confidence": fracture_confidence,
            "localization_available": localization_available,
            "localization": localization,
            "caption": caption,
            "top_predictions": top_predictions,
            "input_width": orig_w,
            "input_height": orig_h,
            "routing_resolution": routing_res,
            "selected_model": selected_model_name
        }
        
        if is_uncertain:
            result["uncertainty_warning"] = f"Prediction uncertain: anatomical confidence ({anatomical_confidence*100:.1f}%) is below uncertainty threshold ({self.uncertainty_threshold*100:.1f}%)."
            
        # 6. Optional visualization export
        if output_dir:
            vis_path = create_annotated_visualization(
                image_input=img,
                prediction_result=result,
                output_dir=output_dir
            )
            result["visualization_path"] = vis_path
            
        result["_elapsed_seconds"] = round(time.time() - t0, 3)
        return result

def run_inference(
    image_path: str,
    classifier_ckpt: Optional[str] = None,
    exp3_classifier_ckpt: Optional[str] = None,
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
        exp3_classifier_ckpt=exp3_classifier_ckpt,
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
    if "input_width" in result and "input_height" in result:
        print(f"Input Dimensions:        {result['input_width']}x{result['input_height']} px")
    if "routing_resolution" in result and "selected_model" in result:
        print(f"Resolution Routing:      {result['routing_resolution']}x{result['routing_resolution']} -> {result['selected_model']}")
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
    parser.add_argument('--classifier-ckpt', type=str, default=None, help='Custom baseline classifier checkpoint path')
    parser.add_argument('--exp3-classifier-ckpt', type=str, default=None, help='Custom high-resolution classifier checkpoint path')
    parser.add_argument('--detector-ckpt', type=str, default=None, help='Custom detector checkpoint path')
    parser.add_argument('--fracture-threshold', type=float, default=None, help='Decision threshold for fracture presence')
    parser.add_argument('--detector-threshold', type=float, default=None, help='Confidence threshold for localization')
    parser.add_argument('--uncertainty-threshold', type=float, default=None, help='Threshold to flag uncertain anatomical predictions')
    parser.add_argument('--json', action='store_true', help='Force strict JSON stdout output')
    args = parser.parse_args()
    
    try:
        pipeline = BoneFractureInferencePipeline(
            classifier_ckpt=args.classifier_ckpt,
            exp3_classifier_ckpt=args.exp3_classifier_ckpt,
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
        "caption": result["caption"],
        "input_width": result.get("input_width"),
        "input_height": result.get("input_height"),
        "routing_resolution": result.get("routing_resolution"),
        "selected_model": result.get("selected_model")
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
