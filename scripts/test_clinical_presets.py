import requests, os

presets = [
    ('sample_pelvis.webp', 'Hip / Pelvis X-ray'),
    ('sample_wrist.jpg', 'Wrist X-ray (AP)'),
    ('sample_forearm.jpg', 'Forearm X-ray'),
    ('sample_hand.jpg', 'Hand X-ray (PA)')
]

print('=== TESTING ALL 4 CLINICAL PRESETS VIA API ===')
for fname, desc in presets:
    p = os.path.join('frontend/assets', fname)
    if os.path.exists(p):
        with open(p, 'rb') as f:
            mime = 'image/webp' if fname.endswith('.webp') else 'image/jpeg'
            res = requests.post('http://127.0.0.1:8000/predict', files={'image': (fname, f, mime)}).json()
        reg = res.get('anatomical_region', 'Unknown')
        reg_c = res.get('anatomical_confidence', 0.0)
        frac = res.get('fracture', False)
        frac_c = res.get('fracture_confidence', 0.0)
        caption = res.get('caption', '')
        status_str = "FRACTURE DETECTED" if frac else "NO FRACTURE DETECTED"
        print(f'{desc:25s} ({fname:18s}): Region={reg.upper():7s} ({reg_c*100:5.1f}%) | Status={status_str:20s} ({frac_c*100:5.1f}%) | Caption: "{caption}"')
