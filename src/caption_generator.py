"""
Factual Medical Description Generator
======================================
Generates concise, factual natural-language descriptions from structured
computer-vision model predictions.

Adheres strictly to clinical validation boundaries:
1. Uses only information explicitly present in structured model predictions.
2. Never infers unsupported clinical findings.
3. Never invents fracture subtype (e.g. transverse, comminuted, hairline).
4. Never invents displacement (e.g. displaced, angulated).
5. Never invents severity (e.g. mild, severe, acute).
6. Never invents patient age or sex.
7. Never invents trauma mechanism (e.g. FOOSH, collision, fall).
8. Never provides treatment recommendations (e.g. cast, surgery).
9. Never claims to provide a definitive medical diagnosis ("detected by the model").
10. Keeps description concise and deterministic.
"""

from typing import Dict, Any, Optional, Union

def _normalize_region(region: Optional[str]) -> Optional[str]:
    if not region:
        return None
    r = str(region).strip()
    if r.lower().startswith("the "):
        r = r[4:].strip()
    if not r:
        return None
    # Preserve acronyms like 'C-spine', otherwise convert standard anatomical names to lowercase
    if r.isupper() and len(r) > 4:
        return r.lower()
    elif not r.isupper():
        return r.lower()
    return r

def _normalize_location(location: Optional[str]) -> Optional[str]:
    if not location:
        return None
    loc = str(location).strip()
    if loc.lower() in ("none", "n/a", "null", "unavailable", "unknown", ""):
        return None
    if loc.lower().startswith("the "):
        loc = loc[4:].strip()
    if not loc:
        return None
    return loc

def _parse_fracture_status(fracture_val: Any, confidence_val: Optional[float] = None) -> bool:
    if isinstance(fracture_val, bool):
        return fracture_val
    if isinstance(fracture_val, (int, float)):
        return bool(fracture_val >= 0.5)
    if isinstance(fracture_val, str):
        val_str = fracture_val.strip().lower()
        if val_str in ("true", "positive", "1", "yes", "fractured", "fracture"):
            return True
        if val_str in ("false", "negative", "0", "no", "normal"):
            return False
    if confidence_val is not None and isinstance(confidence_val, (int, float)):
        return bool(confidence_val >= 0.5)
    return bool(fracture_val)

def generate_description(prediction: Dict[str, Any]) -> str:
    """
    Converts structured model prediction into a concise, factual clinical description.
    
    Parameters:
    -----------
    prediction : dict
        Structured model output dictionary containing:
        - anatomical_region / region (str, optional)
        - fracture (bool / int / str)
        - fracture_location / location (str, optional)
        - fracture_confidence / confidence (float, optional)
        
    Returns:
    --------
    description : str
        Factual, hallucination-free description.
        
    Examples:
    ---------
    >>> generate_description({"anatomical_region": "wrist", "fracture": False})
    'X-ray of the wrist with no fracture detected by the model.'
    
    >>> generate_description({"anatomical_region": "wrist", "fracture": True})
    'X-ray of the wrist with a fracture detected by the model.'
    
    >>> generate_description({"anatomical_region": "wrist", "fracture": True, "fracture_location": "distal radius"})
    'X-ray of the wrist showing a fracture involving the distal radius.'
    """
    if not isinstance(prediction, dict):
        raise TypeError(f"Expected prediction to be a dict, got {type(prediction).__name__}")
        
    # Extract anatomical region
    raw_region = prediction.get("anatomical_region", prediction.get("region"))
    region = _normalize_region(raw_region)
    
    # Extract fracture confidence
    conf = prediction.get("fracture_confidence", prediction.get("confidence"))
    
    # Extract fracture status
    raw_fracture = prediction.get("fracture", prediction.get("fracture_status"))
    is_fracture = _parse_fracture_status(raw_fracture, conf)
    
    # Extract localization / location
    raw_location = prediction.get("fracture_location", prediction.get("location"))
    location = _normalize_location(raw_location)
    
    # Build deterministic caption strictly following requirements
    # 1. No fracture detected
    if not is_fracture:
        if region:
            return f"X-ray of the {region} with no fracture detected by the model."
        return "X-ray with no fracture detected by the model."
        
    # 2. Fracture detected with validated anatomical location
    if location:
        if region:
            return f"X-ray of the {region} showing a fracture involving the {location}."
        return f"X-ray showing a fracture involving the {location}."
        
    # 3. Fracture detected without specific sub-location
    if region:
        return f"X-ray of the {region} with a fracture detected by the model."
    return "X-ray with a fracture detected by the model."

def generate_factual_caption(*args, **kwargs) -> str:
    """
    Backwards-compatible wrapper that accepts dictionary or positional/keyword arguments.
    """
    if len(args) == 1 and isinstance(args[0], dict):
        return generate_description(args[0])
        
    # Handle legacy call signature:
    # (anatomical_region, region_confidence, fracture_detected, fracture_confidence, localization_boxes)
    pred_dict: Dict[str, Any] = {}
    if len(args) >= 1:
        pred_dict["anatomical_region"] = args[0]
    if len(args) >= 3:
        pred_dict["fracture"] = args[2]
    if len(args) >= 4:
        pred_dict["fracture_confidence"] = args[3]
    if len(args) >= 6:
        pred_dict["fracture_location"] = args[5]
        
    # Keyword overrides
    if "anatomical_region" in kwargs:
        pred_dict["anatomical_region"] = kwargs["anatomical_region"]
    elif "region" in kwargs:
        pred_dict["anatomical_region"] = kwargs["region"]
        
    if "fracture_detected" in kwargs:
        pred_dict["fracture"] = kwargs["fracture_detected"]
    elif "fracture" in kwargs:
        pred_dict["fracture"] = kwargs["fracture"]
        
    if "fracture_confidence" in kwargs:
        pred_dict["fracture_confidence"] = kwargs["fracture_confidence"]
    elif "confidence" in kwargs:
        pred_dict["fracture_confidence"] = kwargs["confidence"]
        
    if "fracture_location" in kwargs:
        pred_dict["fracture_location"] = kwargs["fracture_location"]
    elif "location" in kwargs:
        pred_dict["fracture_location"] = kwargs["location"]
        
    return generate_description(pred_dict)
