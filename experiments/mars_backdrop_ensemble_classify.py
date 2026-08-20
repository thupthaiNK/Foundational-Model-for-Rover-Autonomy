#!/usr/bin/env python3
"""
Purpose: Classify the printed Mars-terrain backdrop photos in
         ai4mars_zone_review using the single highest-accuracy model from the
         22-model leaderboard: Ensemble B (DINOv2 ViT-L/14 + DINOv2 ViT-B/14,
         1792-d concatenated features, 94.43% on the 287-image AI4Mars gold
         test set). This is NOT the model deployed on the Pi/rover (that is
         the much smaller DINOv2+reg ViT-S, 90.24%/86.41% real-hardware) --
         it answers "what does the best model available see", not "what will
         the rover see".
Inputs:  "ai4mars_zone_review/0. Mars terrain backdrop" and
         "ai4mars_zone_review/0. Mars terrain backdrop - Copy"
         Feature caches: experiments/results/feature_cache/dinov2_vitl_train_1000_*.npy
                          experiments/results/feature_cache/dinov2_vitb_train_1000_*.npy
How to run:
    source /opt/ros/humble/setup.bash && source ros2_ws/install/setup.bash
    python3 experiments/mars_backdrop_ensemble_classify.py
Project: Onboard Visual Foundation Models for Mars Terrain Perception and Rover Navigation
"""

import os

import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize

CACHE_DIR = os.path.join(os.path.dirname(__file__), "results", "feature_cache")
FOLDERS = [
    "/mnt/c/Users/DELL/Desktop/Thesis/ai4mars_zone_review/0. Mars terrain backdrop",
    "/mnt/c/Users/DELL/Desktop/Thesis/ai4mars_zone_review/0. Mars terrain backdrop - Copy",
]
IMG_EXT = (".jpg", ".jpeg", ".png")
CLASS_NAMES = ["soil", "bedrock", "sand", "big_rock"]  # label ints 0..3

VITL_ID = "facebook/dinov2-large"
VITB_ID = "facebook/dinov2-base"


def expected_class_from_name(fname: str):
    stem = fname.lower()
    if stem.startswith("bedrock"):
        return "bedrock"
    if stem.startswith("big_rock") or stem.startswith("big rock"):
        return "big_rock"
    if stem.startswith("sand"):
        return "sand"
    if stem.startswith("soil"):
        return "soil"
    return None


def load_encoder(model_id: str, device: str):
    from transformers import AutoImageProcessor, AutoModel
    print(f"Loading {model_id} (frozen, FP32 torch)...")
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id).to(device).eval()
    return processor, model


def extract_feature(processor, model, device: str, pil_img: Image.Image) -> np.ndarray:
    import torch
    inputs = processor(images=pil_img, return_tensors="pt").to(device)
    with torch.no_grad():
        feat = model(**inputs).last_hidden_state[:, 0, :]  # CLS token
    return feat.cpu().numpy()  # [1, dim], normalised jointly later


def fit_ensemble_probe():
    """1000-shot LogReg on concatenated ViT-L + ViT-B AI4Mars train features,
    same protocol as ensemble_terrain_test.py's Ensemble B (94.43%)."""
    vitl_feats = np.load(os.path.join(CACHE_DIR, "dinov2_vitl_train_1000_feats.npy"))
    vitl_labels = np.load(os.path.join(CACHE_DIR, "dinov2_vitl_train_1000_labels.npy"))
    vitb_feats = np.load(os.path.join(CACHE_DIR, "dinov2_vitb_train_1000_feats.npy"))

    train_f = normalize(np.concatenate([vitl_feats, vitb_feats], axis=1), norm="l2")
    clf = LogisticRegression(
        C=0.316, max_iter=1000, multi_class="multinomial", solver="lbfgs",
        random_state=42,
    )
    clf.fit(train_f, vitl_labels)
    return clf


def main():
    device = "cpu"
    print("Fitting Ensemble B (DINOv2 ViT-L + ViT-B, 1792-d, 94.43% on AI4Mars gold-287)...")
    clf = fit_ensemble_probe()

    proc_l, model_l = load_encoder(VITL_ID, device)
    proc_b, model_b = load_encoder(VITB_ID, device)

    n_wrong = 0
    n_scored = 0
    for folder in FOLDERS:
        if not os.path.isdir(folder):
            print(f"MISSING FOLDER: {folder}")
            continue
        print(f"\n=== {folder} ===")
        for fname in sorted(os.listdir(folder)):
            if not fname.lower().endswith(IMG_EXT):
                continue
            path = os.path.join(folder, fname)
            expected = expected_class_from_name(fname)
            pil_img = Image.open(path).convert("RGB")

            feat_l = extract_feature(proc_l, model_l, device, pil_img)
            feat_b = extract_feature(proc_b, model_b, device, pil_img)
            feat = normalize(np.concatenate([feat_l, feat_b], axis=1), norm="l2")

            probs = clf.predict_proba(feat)[0]
            pred_idx = int(np.argmax(probs))
            label = CLASS_NAMES[pred_idx]
            confidence = float(probs[pred_idx])

            correct = (label == expected) if expected else None
            if expected:
                n_scored += 1
                if not correct:
                    n_wrong += 1
            flag = "" if correct or correct is None else "  <-- MISCLASSIFIED"
            probs_str = " ".join(f"{c}:{p:.2f}" for c, p in zip(CLASS_NAMES, probs))
            print(f"{fname:24s} expected={str(expected):10s} pred={label:10s} "
                  f"conf={confidence:.3f} [{probs_str}]{flag}")

    print(f"\n{n_wrong}/{n_scored} misclassified across both folders (Ensemble B, 94.43%).")


if __name__ == "__main__":
    main()
