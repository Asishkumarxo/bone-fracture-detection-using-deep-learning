import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import torch.nn.functional as F
from predict import BoneFractureInferencePipeline
from inference.preprocessing import preprocess_for_classifier

p = BoneFractureInferencePipeline()
model = p.shared_model

test_files = [
    ('wrist_crack', 'c:/Users/arbaz/Downloads/wrist crack image.jpg'),
    ('2_jpg', 'c:/Users/arbaz/Downloads/2.jpg'),
    ('istockphoto', 'c:/Users/arbaz/Downloads/istockphoto-471457370-612x612.jpg'),
    ('adobestock_wide', 'c:/Users/arbaz/Downloads/AdobeStock_594927383-1.jpeg'),
    ('pelvis_norm', 'c:/Users/arbaz/Downloads/AdobeStock_200285274.webp'),
    ('fn_hand_pos', 'error_analysis/false_negatives/fn_Hand_patient08635_Positive_001.png.png'),
    ('hand_neg', 'data/training_curated/Hand/Hand_Negative_002.jpg')
]

features = {}
probs = {}
for name, path in test_files:
    if os.path.exists(path):
        t = preprocess_for_classifier(path).to(p.device)
        with torch.no_grad():
            feat = model.get_features(t) # 2048-dim
            r_log, f_log = model(t)
            prob = torch.sigmoid(f_log).item()
            features[name] = F.normalize(feat, p=2, dim=1)
            probs[name] = prob
            print(f"{name:<16}: Frac Logit = {f_log.item():6.3f} | Prob = {prob*100:5.2f}%")

print("\nCosine Similarities between 2048-dim feature representations:")
names = list(features.keys())
header = f"{'':<16}" + "".join([f"{n[:10]:>12}" for n in names])
print(header)
for n1 in names:
    row = f"{n1:<16}"
    for n2 in names:
        sim = (features[n1] * features[n2]).sum().item()
        row += f"{sim:12.3f}"
    print(row)
