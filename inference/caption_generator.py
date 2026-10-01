"""
Production Factual Caption Generator
====================================
Synthesizes factual, clinically disciplined natural-language captions
strictly bounded by validated model outputs.
"""

from typing import Dict, Any, Optional

def _normalize_string(val: Optional[str]) -> Optional[str]:
    if not val:
        return None
    s = str(val).strip()
    if s.lower() in ("none", "n/a", "null", "unavailable", "unknown", ""):
        return None
    if s.lower().startswith("the "):
        s = s[4:].strip()
    return s

def generate_caption(prediction: Dict[str, Any]) -> str:
    """
    Generates a concise, strictly factual natural-language description.
    
    Parameters:
    -----------
    prediction : dict
        Structured output containing:
        - anatomical_region (str)
        - fracture (bool)
        - fracture_location / location (str, optional)
        
    Returns:
    --------
    caption : str
        Factual caption adhering to validation constraints.
        
    Examples:
    ---------
    >>> generate_caption({"anatomical_region": "wrist", "fracture": True})
    'X-ray of the wrist showing a fracture.'
    
    >>> generate_caption({"anatomical_region": "wrist", "fracture": True, "fracture_location": "distal radius"})
    'X-ray of the wrist showing a fracture involving the distal radius.'
    
    >>> generate_caption({"anatomical_region": "wrist", "fracture": False})
    'X-ray of the wrist with no fracture detected by the model.'
    """
    raw_region = prediction.get("anatomical_region", prediction.get("region"))
    region = _normalize_string(raw_region)
    if region and not region.isupper():
        region = region.lower()
        
    is_fracture = bool(prediction.get("fracture", False))
    raw_loc = prediction.get("fracture_location", prediction.get("location"))
    location = _normalize_string(raw_loc)
    
    if not is_fracture:
        if region:
            return f"X-ray of the {region} with no fracture detected by the model."
        return "X-ray with no fracture detected by the model."
        
    if location:
        if region:
            return f"X-ray of the {region} showing a fracture involving the {location}."
        return f"X-ray showing a fracture involving the {location}."
        
    if region:
        return f"X-ray of the {region} showing a fracture."
    return "X-ray showing a fracture."
