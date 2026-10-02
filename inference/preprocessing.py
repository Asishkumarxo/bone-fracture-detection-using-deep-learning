"""
Deterministic Preprocessing Pipeline for Inference
===================================================
Implements the exact aspect-ratio preserving padding, intensity standardization,
and normalization transformations used during model training.
"""

import os
from typing import Union, Tuple, Optional
from PIL import Image
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF

from .model_registry import ModelRegistry

def resize_and_pad(img: Image.Image, target_size: Tuple[int, int] = (224, 224), fill_color: int = 0) -> Image.Image:
    """
    Applies the exact aspect-ratio preserving resize and padding used in training.
    """
    target_w, target_h = target_size
    orig_w, orig_h = img.size
    
    if orig_w == target_w and orig_h == target_h:
        return img
        
    ratio = min(target_w / orig_w, target_h / orig_h)
    new_w = max(1, int(round(orig_w * ratio)))
    new_h = max(1, int(round(orig_h * ratio)))
    
    resized_img = img.resize((new_w, new_h), Image.Resampling.BILINEAR)
    
    if new_w == target_w and new_h == target_h:
        return resized_img
        
    padded_img = Image.new('L', (target_w, target_h), color=fill_color)
    pad_x = (target_w - new_w) // 2
    pad_y = (target_h - new_h) // 2
    padded_img.paste(resized_img, (pad_x, pad_y))
    return padded_img

def load_and_validate_image(image_input: Union[str, Image.Image]) -> Image.Image:
    """
    Loads and validates image from path or PIL Image.
    """
    if isinstance(image_input, str):
        if not os.path.exists(image_input):
            raise FileNotFoundError(f"Input image file not found: {image_input}")
        try:
            with Image.open(image_input) as raw_im:
                raw_im.verify()
            img = Image.open(image_input)
            img.load()
            return img
        except Exception as e:
            raise ValueError(f"Corrupted or invalid image file: {e}")
    elif isinstance(image_input, Image.Image):
        return image_input
    else:
        raise TypeError(f"Expected file path str or PIL Image, got {type(image_input).__name__}")

def get_original_dimensions(image_input: Union[str, Image.Image]) -> Tuple[int, int]:
    """
    Returns the original unresized (width, height) dimensions of an image input.
    """
    img = load_and_validate_image(image_input)
    return img.size # (width, height)

def preprocess_for_classifier(
    image_input: Union[str, Image.Image],
    target_size: Optional[Tuple[int, int]] = None
) -> torch.Tensor:
    """
    Preprocesses an input image for the multi-task classifier using exact training convention:
    Grayscale ('L') -> Aspect-ratio preserving pad to target_size -> Broadcast to 3 channels -> Normalize.
    
    If target_size is None, defaults to ModelRegistry.CLASSIFIER_IMAGE_SIZE (224, 224).
    
    Returns:
    --------
    torch.Tensor: Shape (1, 3, target_h, target_w)
    """
    img = load_and_validate_image(image_input)
    
    if target_size is None:
        target_size = ModelRegistry.CLASSIFIER_IMAGE_SIZE
        
    # 1. Grayscale conversion
    gray = img.convert('L')
    
    # 2. Aspect-ratio preserving padding to target_size
    padded = resize_and_pad(gray, target_size=target_size)
    
    # 3. 3-channel broadcast for ImageNet backbone
    rgb = padded.convert('RGB')
    
    # 4. Standard deterministic normalization
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=ModelRegistry.CLASSIFIER_NORMALIZATION_MEAN,
                    std=ModelRegistry.CLASSIFIER_NORMALIZATION_STD)
    ])
    tensor = transform(rgb).unsqueeze(0)
    return tensor

def preprocess_for_detector(image_input: Union[str, Image.Image]) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """
    Preprocesses an input image for the fracture detector using exact detector convention:
    RGB -> Bilinear resize to target size (384, 384) -> ToTensor.
    
    Returns:
    --------
    tensor: Shape (1, 3, H, W)
    orig_size: (width, height) of the original image
    """
    img = load_and_validate_image(image_input)
    orig_size = img.size # (W, H)
    
    rgb = img.convert('RGB')
    resized = rgb.resize(ModelRegistry.DETECTOR_TARGET_SIZE, Image.Resampling.BILINEAR)
    tensor = TF.to_tensor(resized).unsqueeze(0)
    return tensor, orig_size
