#!/usr/bin/env python3
"""
Purpose: Run the same frozen DINOv2 ViT-S/14 + LogReg terrain classifier used
         by dinov2_terrain_node.py against the printed Mars-terrain backdrop
         photos in ai4mars_zone_review, to find which panel(s) DINOv2
         misclassifies before/after the "- Copy" (upscaled) revalidation.
         Expected class is read from each filename's prefix
         (bedrock/big_rock/sand/soil).
Inputs:  "ai4mars_zone_review/0. Mars terrain backdrop" and
         "ai4mars_zone_review/0. Mars terrain backdrop - Copy"
         Feature cache: experiments/results/feature_cache/dinov2_reg_small_train_1000shot.npz
Outputs: Printed per-image table (folder, file, expected, predicted, confidence, correct)
How to run:
    source /opt/ros/humble/setup.bash && source ros2_ws/install/setup.bash
    python3 experiments/mars_backdrop_photo_classify.py
Project: Onboard Visual Foundation Models for Mars Terrain Perception and Rover Navigation
"""

import os
import sys

import numpy as np
from PIL import Image
from sklearn.preprocessing import normalize

sys.path.insert(
    0, os.path.join(
        os.path.dirname(__file__), "..", "ros2_ws", "src", "fm_perception"
    )
)
from fm_perception.dinov2_terrain_node import (  # noqa: E402
    CLASS_NAMES, CONFIDENCE_THRESHOLD, DINOV2_MODEL_ID, T_STAR,
    fit_or_load_logreg,
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
DEFAULT_CACHE = os.path.join(
    RESULTS_DIR, "feature_cache", "dinov2_reg_small_train_1000shot.npz"
)
FOLDERS = [
    "/mnt/c/Users/DELL/Desktop/Thesis/ai4mars_zone_review/0. Mars terrain backdrop",
    "/mnt/c/Users/DELL/Desktop/Thesis/ai4mars_zone_review/0. Mars terrain backdrop - Copy",
]
IMG_EXT = (".jpg", ".jpeg", ".png")


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


def load_encoder(device: str):
    from transformers import AutoImageProcessor, AutoModel
    print(f"Loading {DINOV2_MODEL_ID} (frozen, FP32 torch)...")
    processor = AutoImageProcessor.from_pretrained(DINOV2_MODEL_ID)
    model = AutoModel.from_pretrained(DINOV2_MODEL_ID).to(device).eval()
    return processor, model


def extract_features(processor, model, device: str, pil_img: Image.Image) -> np.ndarray:
    import torch
    inputs = processor(images=pil_img, return_tensors="pt").to(device)
    with torch.no_grad():
        feat = model(**inputs).last_hidden_state[:, 0, :]
    return normalize(feat.cpu().numpy(), norm="l2")


def classify(clf, feat_np: np.ndarray):
    logits = clf.decision_function(feat_np)[0]
    scaled = logits / T_STAR
    scaled -= scaled.max()
    exp_s = np.exp(scaled)
    probs = exp_s / exp_s.sum()

    full_probs = np.zeros(len(CLASS_NAMES), dtype=np.float32)
    for i, cls in enumerate(clf.classes_):
        full_probs[cls] = probs[i]

    pred_idx = int(full_probs.argmax())
    confidence = float(full_probs[pred_idx])
    label = CLASS_NAMES[pred_idx] if confidence >= CONFIDENCE_THRESHOLD else "uncertain"
    return label, confidence, full_probs


def main():
    device = "cpu"
    processor, model = load_encoder(device)
    clf, _ = fit_or_load_logreg(DEFAULT_CACHE, n_shot=1000, class_weight_balanced=False)

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
            feat = extract_features(processor, model, device, pil_img)
            label, confidence, probs = classify(clf, feat)

            correct = (label == expected) if expected else None
            if expected:
                n_scored += 1
                if not correct:
                    n_wrong += 1
            flag = "" if correct or correct is None else "  <-- MISCLASSIFIED"
            probs_str = " ".join(f"{c}:{p:.2f}" for c, p in zip(CLASS_NAMES, probs))
            print(f"{fname:24s} expected={str(expected):10s} pred={label:10s} "
                  f"conf={confidence:.3f} [{probs_str}]{flag}")

    print(f"\n{n_wrong}/{n_scored} misclassified across both folders.")


if __name__ == "__main__":
    main()
