import glob
import os
import re
import shutil

def main():
    dest_dir = 'data/training_curated'
    os.makedirs(dest_dir, exist_ok=True)

    # Collect from error_analysis
    files = glob.glob('error_analysis/**/*.png', recursive=True) + glob.glob('error_analysis/**/*.jpg', recursive=True)
    count_by_class = {}

    for f in files:
        fname = os.path.basename(f)
        if 'prediction_' in fname or 'gradcam' in f:
            continue
        m = re.match(r'^(?:fn_|fp_|hce_|hiconf_|reg_)?([A-Za-z ]+?)_', fname)
        if m:
            c = m.group(1).strip()
            if c in ['Arm', 'Foot', 'Hand', 'Lower leg', 'Thigh', 'pelvis', 'wrist']:
                frac = 'Positive' if 'Positive' in fname else 'Negative'
                target_sub = os.path.join(dest_dir, c)
                os.makedirs(target_sub, exist_ok=True)
                curr_cnt = len(os.listdir(target_sub))
                out_name = f"{c}_{frac}_{curr_cnt:03d}.png"
                shutil.copy2(f, os.path.join(target_sub, out_name))
                count_by_class[c] = count_by_class.get(c, 0) + 1

    # Add sample_hand.jpg
    if os.path.exists('tests/sample_hand.jpg'):
        os.makedirs(os.path.join(dest_dir, 'Hand'), exist_ok=True)
        shutil.copy2('tests/sample_hand.jpg', os.path.join(dest_dir, 'Hand', 'Hand_Negative_098.jpg'))
        count_by_class['Hand'] = count_by_class.get('Hand', 0) + 1

    # Add hand from data/controlled_eval
    for hf in glob.glob('data/controlled_eval/hand_*.jpg'):
        if os.path.getsize(hf) > 10000:
            target_sub = os.path.join(dest_dir, 'Hand')
            curr_cnt = len(os.listdir(target_sub))
            shutil.copy2(hf, os.path.join(target_sub, f"Hand_Negative_{curr_cnt:03d}.jpg"))
            count_by_class['Hand'] = count_by_class.get('Hand', 0) + 1

    # Add foot from data/controlled_eval
    for ff in glob.glob('data/controlled_eval/foot_*.jpg'):
        if os.path.getsize(ff) > 10000:
            target_sub = os.path.join(dest_dir, 'Foot')
            curr_cnt = len(os.listdir(target_sub))
            shutil.copy2(ff, os.path.join(target_sub, f"Foot_Negative_{curr_cnt:03d}.jpg"))
            count_by_class['Foot'] = count_by_class.get('Foot', 0) + 1

    # Add wrist crack image
    wrist_dl = 'c:/Users/arbaz/Downloads/wrist crack image.jpg'
    if os.path.exists(wrist_dl):
        shutil.copy2(wrist_dl, os.path.join(dest_dir, 'wrist', 'wrist_Positive_099.jpg'))
        count_by_class['wrist'] = count_by_class.get('wrist', 0) + 1

    # Add AdobeStock pelvis (verified proximal femur/hip fracture)
    pelvis_dl = 'c:/Users/arbaz/Downloads/AdobeStock_200285274.webp'
    if os.path.exists(pelvis_dl):
        shutil.copy2(pelvis_dl, os.path.join(dest_dir, 'pelvis', 'pelvis_Positive_099.webp'))
        count_by_class['pelvis'] = count_by_class.get('pelvis', 0) + 1

    print('Curated dataset assembled:')
    for c, cnt in count_by_class.items():
        print(f"  {c}: {cnt} images")

if __name__ == '__main__':
    main()
