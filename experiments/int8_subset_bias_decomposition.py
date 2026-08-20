#!/usr/bin/env python3
"""
Purpose: Explain, and correct, the disagreement between two INT8 quantization
         accuracy figures in this thesis.

         Experiment B1 (experiments/int8_quantization_rpi.py) reports FP32 94.0%
         -> INT8 93.5% on AI4Mars, a 0.50 pp drop, and Chapter 4 §4.8.19 cites it
         as evidence that INT8 quantization is "near-lossless" and therefore safe
         to make the real-hardware default. Re-measuring the same INT8 encoder
         through the pipeline the rover actually deploys
         (experiments/dinov2_preprocess_revalidation.py) instead gives 90.24% ->
         86.41%, a 3.83 pp drop. Both cannot be quoted.

         Two candidate causes were identified by reading B1's code, and this
         script crosses them 2x2 to attribute the gap rather than guess:
           subset:        B1 takes the first 200 test images by filename and
                          `break`s, so it never sees 87 of the 287 gold images and
                          the sample is ordered, not random.
           preprocessing: B1 squashes each image to 224x224, discarding aspect
                          ratio, whereas the 90.24% headline result and the
                          deployed dinov2_terrain_node both resize the shortest
                          edge to 256 and take a 224 centre crop.
         The encoders (B1's own FP32 and INT8 ONNX exports) and the 1000-shot
         LogReg probe are held identical across all four cells, so the only things
         varying are the two candidates.

Inputs:  AI4Mars gold test set (masked-gold-min3-100agree) + MSL EDR images
         experiments/results/dinov2_reg_small_encoder.onnx      (FP32)
         experiments/results/dinov2_reg_small_encoder_int8.onnx (INT8)
         experiments/results/feature_cache/dinov2_reg_small_train_1000shot.npz
Outputs: Console 2x2 table + experiments/results/int8_subset_bias_decomposition.csv
How to run:
    python3 -u experiments/int8_subset_bias_decomposition.py
Project: Onboard Visual Foundation Models for Mars Terrain Perception and Rover Navigation
"""

import csv
import os
import sys

import numpy as np
import onnxruntime as ort
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.normpath(os.path.join(_THIS_DIR, ".."))
sys.path.insert(0, os.path.join(_REPO_ROOT, "ros2_ws", "src", "fm_perception"))

from fm_perception.dinov2_preprocess import preprocess_dinov2  # noqa: E402

AI4MARS_BASE = "/mnt/c/Users/DELL/Desktop/Thesis/github source/ai4mars-dataset-merged-0.1"
IMAGES_DIR = os.path.join(AI4MARS_BASE, "msl/images/edr")
TEST_LABELS = os.path.join(AI4MARS_BASE, "msl/labels/test/masked-gold-min3-100agree")

RESULTS_DIR = os.path.join(_THIS_DIR, "results")
FP32_ONNX = os.path.join(RESULTS_DIR, "dinov2_reg_small_encoder.onnx")
INT8_ONNX = os.path.join(RESULTS_DIR, "dinov2_reg_small_encoder_int8.onnx")
TRAIN_CACHE = os.path.join(RESULTS_DIR, "feature_cache", "dinov2_reg_small_train_1000shot.npz")
OUT_CSV = os.path.join(RESULTS_DIR, "int8_subset_bias_decomposition.csv")

IMAGE_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGE_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
CLASS_NAMES = ["soil", "bedrock", "sand", "big_rock"]
B1_N_TEST_MAX = 200
LOGR_C = 0.316
N_SHOT = 1000


def b1_preprocess(image_path):
    """B1's preprocessing, copied verbatim from int8_quantization_rpi.py._preprocess."""
    img = Image.open(image_path).convert("RGB").resize((224, 224))
    x = np.array(img, dtype=np.float32) / 255.0
    x = (x - IMAGE_MEAN) / IMAGE_STD
    return x.transpose(2, 0, 1)[np.newaxis, :]


def label_from_mask(mask_path):
    mask = np.array(Image.open(mask_path).convert("L"))
    valid = mask[mask < 4]
    return int(np.bincount(valid).argmax()) if len(valid) else -1


def load_gold_pairs():
    """All gold test pairs, in the same filename order B1 truncates."""
    pairs = []
    for fname in sorted(os.listdir(TEST_LABELS)):
        if not fname.endswith("_merged.png"):
            continue
        stem = fname.replace("_merged.png", "")
        image_path = os.path.join(IMAGES_DIR, stem + ".JPG")
        if not os.path.exists(image_path):
            continue
        gt = label_from_mask(os.path.join(TEST_LABELS, fname))
        if gt >= 0:
            pairs.append((stem, image_path, gt))
    return pairs


def train_probe():
    data = np.load(TRAIN_CACHE)
    feats, labels = data["feats"], data["labels"]
    idx = []
    for c in np.unique(labels):
        c_idx = np.where(labels == c)[0]
        idx.extend(c_idx[:N_SHOT].tolist())
    idx = np.array(idx)
    clf = LogisticRegression(C=LOGR_C, max_iter=1000, solver="lbfgs", random_state=42)
    clf.fit(normalize(feats[idx], norm="l2"), labels[idx])
    return clf


def main():
    for path in (FP32_ONNX, INT8_ONNX, TRAIN_CACHE):
        if not os.path.exists(path):
            raise FileNotFoundError(path)

    clf = train_probe()
    sess = {
        "fp32": ort.InferenceSession(FP32_ONNX, providers=["CPUExecutionProvider"]),
        "int8": ort.InferenceSession(INT8_ONNX, providers=["CPUExecutionProvider"]),
    }

    pairs = load_gold_pairs()
    print(f"Gold test pairs: {len(pairs)}  (B1 truncates to the first {B1_N_TEST_MAX})\n")

    # Predict once per (preprocessing, encoder) over every image, then slice the
    # subsets. Slicing cached predictions guarantees the two subset cells differ
    # only by which images they include.
    records = []
    for i, (stem, image_path, gt) in enumerate(pairs, 1):
        pil = Image.open(image_path)
        pixels = {
            "squash": b1_preprocess(image_path),
            "cropped": preprocess_dinov2(pil).astype(np.float32),
        }
        rec = {"image": stem, "gt_idx": gt, "ground_truth": CLASS_NAMES[gt]}
        for pname, x in pixels.items():
            for ename, s in sess.items():
                feat = normalize(s.run(None, {"pixel_values": x})[0], norm="l2")
                rec[f"{pname}_{ename}"] = int(clf.predict(feat)[0])
        records.append(rec)
        if i % 50 == 0 or i == len(pairs):
            print(f"  {i}/{len(pairs)}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        w.writeheader()
        w.writerows(records)

    gt = np.array([r["gt_idx"] for r in records])
    subsets = (
        (f"first-{B1_N_TEST_MAX} (B1)", slice(0, B1_N_TEST_MAX)),
        (f"all-{len(records)}", slice(None)),
    )
    preprocs = (
        ("squash (B1)", "squash"),
        ("crop (deployed)", "cropped"),
    )

    print("\n" + "=" * 68)
    print("INT8 accuracy drop, decomposed by test subset and preprocessing")
    print("=" * 68)
    print(f"{'subset':<18}{'preprocessing':<18}{'FP32':>10}{'INT8':>10}{'drop':>10}")
    table = {}
    for sname, sl in subsets:
        for plabel, pkey in preprocs:
            a = 100.0 * (np.array([r[f"{pkey}_fp32"] for r in records])[sl] == gt[sl]).mean()
            b = 100.0 * (np.array([r[f"{pkey}_int8"] for r in records])[sl] == gt[sl]).mean()
            table[(sname, pkey)] = (a, b)
            print(f"{sname:<18}{plabel:<18}{a:>9.2f}%{b:>9.2f}%{a - b:>+9.2f}")

    b1 = table[(f"first-{B1_N_TEST_MAX} (B1)", "squash")]
    deployed = table[(f"all-{len(records)}", "cropped")]
    subset_only = table[(f"all-{len(records)}", "squash")]

    print("\nAttribution of the drop (0.50 pp reported -> 3.83 pp measured):")
    print(f"  B1 as published            : {b1[0]:.2f} -> {b1[1]:.2f}   drop {b1[0] - b1[1]:.2f} pp")
    print(f"  + use all {len(records)} gold images : {subset_only[0]:.2f} -> {subset_only[1]:.2f}   "
          f"drop {subset_only[0] - subset_only[1]:.2f} pp")
    print(f"  + deployed preprocessing   : {deployed[0]:.2f} -> {deployed[1]:.2f}   "
          f"drop {deployed[0] - deployed[1]:.2f} pp")
    print(f"\n  subset truncation accounts for "
          f"{(subset_only[0] - subset_only[1]) - (b1[0] - b1[1]):.2f} pp of the difference")
    print(f"  preprocessing accounts for     "
          f"{(deployed[0] - deployed[1]) - (subset_only[0] - subset_only[1]):.2f} pp")
    print(f"\nPer-image CSV: {OUT_CSV}")


if __name__ == "__main__":
    main()
