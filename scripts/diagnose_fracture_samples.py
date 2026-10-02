import os
from PIL import Image
import numpy as np

files = [
    'c:/Users/arbaz/Downloads/2.jpg',
    'c:/Users/arbaz/Downloads/istockphoto-471457370-612x612.jpg',
    'c:/Users/arbaz/Downloads/AdobeStock_594927383-1.jpeg',
    'c:/Users/arbaz/Downloads/wrist crack image.jpg',
    'c:/Users/arbaz/Downloads/AdobeStock_200285274.webp'
]

for f in files:
    if os.path.exists(f):
        with Image.open(f) as im:
            arr = np.array(im.convert('L'))
            print(f"File: {os.path.basename(f)}")
            print(f"  Format: {im.format}, Mode: {im.mode}, Size: {im.size}")
            print(f"  Min: {arr.min()}, Max: {arr.max()}, Mean: {arr.mean():.1f}, Std: {arr.std():.1f}")
            print(f"  Corner pixels (top-left 5x5 mean): {arr[:5, :5].mean():.1f}")
