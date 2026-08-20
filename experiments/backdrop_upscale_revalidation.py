"""
Purpose: Re-validate the physical Mars-terrain backdrop images after ChatGPT upscaling.
         The original A1 backdrop crops (16 real AI4Mars frames) were software-validated at
         16/16 (100%) by the thesis DINOv2 ViT-S/14 few-shot linear probe. The print shop
         flagged them as too low-resolution for A1, so the user had ChatGPT upscale/sharpen
         them. This script re-runs the SAME probe on the upscaled PNGs to check the label
         predictions still match the ground-truth class in each filename.
Inputs:  - Original crops:  ai4mars_zone_review/0. Mars terrain backdrop            (*.jpg)
         - Upscaled crops:  ai4mars_zone_review/0. Mars terrain backdrop - Copy    (*.png)
         - Cached 1000-shot/class DINOv2 train features (same protocol as dinov2_terrain_test.py)
Outputs: Per-image predictions + overall accuracy for both folders, printed to stdout.
How to run:
    python3 -u experiments/backdrop_upscale_revalidation.py
Project: Onboard Visual Foundation Models for Mars Terrain Perception and Rover Navigation
"""

import os
import re
import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize
from transformers import AutoImageProcessor, AutoModel

CLASS_NAMES = ["soil", "bedrock", "sand", "big_rock"]
NAME_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}
RANDOM_SEED = 42

BASE = "/mnt/c/Users/DELL/Desktop/Thesis"
ORIGINAL_DIR = os.path.join(BASE, "ai4mars_zone_review", "0. Mars terrain backdrop")
UPSCALED_DIR = os.path.join(BASE, "ai4mars_zone_review", "0. Mars terrain backdrop - Copy")
TRAIN_CACHE  = os.path.join(os.path.dirname(__file__),
                            "results/feature_cache/dinov2_train_1000shot.npz")


def gt_from_filename(fname):
    """Ground-truth class = leading token of the filename, e.g. 'big_rock 2.1.png'."""
    stem = os.path.splitext(fname)[0]
    m = re.match(r"([a-zA-Z_]+)", stem)
    key = m.group(1).rstrip("_") if m else ""
    # filenames use 'big_rock', 'bedrock', 'sand', 'soil'
    for c in CLASS_NAMES:
        if stem.lower().startswith(c):
            return NAME_TO_IDX[c]
    return NAME_TO_IDX.get(key, None)


def load_model(device):
    print("Loading DINOv2 ViT-S/14 (facebook/dinov2-small, frozen)...")
    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
    model = AutoModel.from_pretrained("facebook/dinov2-small").to(device).eval()
    return model, processor


def extract(model, processor, paths, device):
    feats = []
    for p in paths:
        img = Image.open(p).convert("RGB")
        inputs = processor(images=img, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**inputs)
            f = out.last_hidden_state[:, 0, :].cpu().numpy().squeeze()
        feats.append(f)
    feats = np.array(feats, dtype=np.float32)
    return normalize(feats)


def build_probe():
    d = np.load(TRAIN_CACHE)
    X, y = normalize(d["feats"]), d["labels"]
    clf = LogisticRegression(C=0.316, max_iter=1000, random_state=RANDOM_SEED,
                             multi_class="multinomial", solver="lbfgs")
    clf.fit(X, y)
    return clf


def evaluate_folder(model, processor, clf, folder, device):
    files = sorted(f for f in os.listdir(folder)
                   if f.lower().endswith((".jpg", ".jpeg", ".png")))
    paths = [os.path.join(folder, f) for f in files]
    feats = extract(model, processor, paths, device)
    probs = clf.predict_proba(feats)
    preds = probs.argmax(axis=1)

    print(f"\n{'='*72}\n{folder}\n{'='*72}")
    print(f"{'file':<22}{'truth':<10}{'pred':<10}{'conf':>7}   ok")
    print("-" * 72)
    correct = 0
    for f, pred, prob in zip(files, preds, probs):
        gt = gt_from_filename(f)
        ok = (gt == pred)
        correct += int(ok)
        print(f"{f:<22}{CLASS_NAMES[gt]:<10}{CLASS_NAMES[pred]:<10}"
              f"{prob[pred]*100:>6.1f}%   {'YES' if ok else 'NO'}")
    print("-" * 72)
    print(f"OVERALL: {correct}/{len(files)} correct "
          f"({correct/len(files)*100:.1f}%)")
    return correct, len(files)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, processor = load_model(device)
    clf = build_probe()
    print("Probe trained on cached 1000-shot/class DINOv2 features.")

    o_c, o_n = evaluate_folder(model, processor, clf, ORIGINAL_DIR, device)
    u_c, u_n = evaluate_folder(model, processor, clf, UPSCALED_DIR, device)

    print(f"\n{'='*72}\nSUMMARY\n{'='*72}")
    print(f"Original crops (jpg): {o_c}/{o_n} ({o_c/o_n*100:.1f}%)")
    print(f"Upscaled crops (png): {u_c}/{u_n} ({u_c/u_n*100:.1f}%)")


if __name__ == "__main__":
    main()
