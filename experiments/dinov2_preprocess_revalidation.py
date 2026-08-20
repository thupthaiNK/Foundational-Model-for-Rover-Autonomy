#!/usr/bin/env python3
"""
Purpose: Close the open risk left by the 2026-07-27 torch-removal change. The Pi
         container cannot import torch or transformers, so dinov2_terrain_node's
         INT8 ONNX path was switched from HuggingFace AutoImageProcessor to the
         pure-numpy fm_perception.dinov2_preprocess.preprocess_dinov2. That swap
         was only validated on six synthetic frame shapes, where it produced a
         feature cosine of 0.984-0.992 against the HuggingFace processor, not the
         >0.999 that would make it obviously equivalent. A cosine that low leaves
         open whether the deployed rover classifies terrain as accurately as the
         thesis reports, so no Pi accuracy figure can be quoted until it is
         resolved on the real benchmark.

         This script resolves it by holding everything else fixed (same INT8 ONNX
         encoder, same LogReg probe, same 287-image AI4Mars gold test set) and
         varying only the preprocessing, so any accuracy difference is attributable
         to the swap alone:
             Arm A (reference): AutoImageProcessor -> INT8 ONNX -> LogReg
             Arm B (deployed):  preprocess_dinov2  -> INT8 ONNX -> LogReg
             Arm C (upper bound, --with-fp32): AutoImageProcessor -> FP32 torch -> LogReg
         The LogReg probe is rebuilt exactly as dinov2_terrain_node does it
         (1000-shot from the cached training features, C=0.316, no class weighting)
         so the comparison reflects the deployed classifier, not a re-tuned one.

         Arm C exists because of what a first run of A vs B revealed: the two
         preprocessors agree to 3.6e-07 (pure float32 rounding, not a real
         algorithmic difference), yet their INT8 features still differ by a cosine
         of ~0.96-0.99, while re-running the same input through the same session
         reproduces bit-for-bit. The encoder is therefore deterministic but
         extremely sensitive -- a perturbation far below any perceptual threshold
         crosses quantization bin boundaries and moves the feature vector. That
         reframes the original worry: the low cosine was never evidence against
         the numpy preprocessor, it is a property of INT8 quantization. The
         question that actually gates quoting a Pi accuracy figure is whether that
         sensitivity costs end-to-end accuracy, which only the FP32 arm can bound.

Inputs:  AI4Mars gold test set (masked-gold-min3-100agree) + MSL EDR images
         experiments/results/dinov2_reg_small_encoder_int8.onnx
         experiments/results/feature_cache/dinov2_reg_small_train_1000shot.npz
Outputs: Console report + experiments/results/dinov2_preprocess_revalidation.csv
         (per-image: ground truth, both predictions, both confidences, feature
         cosine, max absolute pixel difference)
How to run:
    python3 -u experiments/dinov2_preprocess_revalidation.py
    python3 -u experiments/dinov2_preprocess_revalidation.py --limit 20   # smoke test
Project: Onboard Visual Foundation Models for Mars Terrain Perception and Rover Navigation
"""

import argparse
import csv
import os
import sys
import time

import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.normpath(os.path.join(_THIS_DIR, ".."))
sys.path.insert(0, os.path.join(_REPO_ROOT, "ros2_ws", "src", "fm_perception"))

from fm_perception.dinov2_preprocess import preprocess_dinov2  # noqa: E402

# ── Paths and constants, mirrored from dinov2_terrain_test.py and
#    dinov2_terrain_node.py so this script cannot silently drift from either ──
AI4MARS_BASE = "/mnt/c/Users/DELL/Desktop/Thesis/github source/ai4mars-dataset-merged-0.1"
IMAGES_DIR = os.path.join(AI4MARS_BASE, "msl/images/edr")
TEST_LABELS = os.path.join(AI4MARS_BASE, "msl/labels/test/masked-gold-min3-100agree")

RESULTS_DIR = os.path.join(_THIS_DIR, "results")
ONNX_ENCODER = os.path.join(RESULTS_DIR, "dinov2_reg_small_encoder_int8.onnx")
TRAIN_CACHE = os.path.join(RESULTS_DIR, "feature_cache", "dinov2_reg_small_train_1000shot.npz")
OUT_CSV = os.path.join(RESULTS_DIR, "dinov2_preprocess_revalidation.csv")

DINOV2_MODEL_ID = "facebook/dinov2-with-registers-small"
CLASS_NAMES = ["soil", "bedrock", "sand", "big_rock"]
IGNORE_PIXEL = 255
T_STAR = 0.461
LOGR_C = 0.316
N_SHOT = 1000


# ── Ground truth (identical to dinov2_terrain_test.py) ────────────────────────

def dominant_class(label_path):
    label = np.array(Image.open(label_path))
    valid = label[label != IGNORE_PIXEL]
    if len(valid) == 0:
        return None
    return int(np.argmax(np.bincount(valid, minlength=4)))


def load_test_split():
    pairs = []
    for fname in sorted(os.listdir(TEST_LABELS)):
        if not fname.endswith(".png"):
            continue
        stem = fname.replace("_merged.png", "").replace(".png", "")
        image_path = os.path.join(IMAGES_DIR, stem + ".JPG")
        if not os.path.exists(image_path):
            continue
        gt = dominant_class(os.path.join(TEST_LABELS, fname))
        if gt is not None:
            pairs.append((stem, image_path, gt))
    return pairs


# ── Probe, rebuilt exactly as the node does ───────────────────────────────────

def train_probe():
    data = np.load(TRAIN_CACHE)
    feats, labels = data["feats"], data["labels"]
    idx = []
    for c in np.unique(labels):
        c_idx = np.where(labels == c)[0]
        idx.extend((c_idx[:N_SHOT] if len(c_idx) >= N_SHOT else c_idx).tolist())
    idx = np.array(idx)
    clf = LogisticRegression(
        C=LOGR_C, max_iter=1000, solver="lbfgs", random_state=42, class_weight=None
    )
    clf.fit(normalize(feats[idx], norm="l2"), labels[idx])
    return clf


def classify(clf, feat):
    """Node's _classify: temperature-scaled softmax over decision_function."""
    logits = clf.decision_function(feat)[0]
    scaled = logits / T_STAR
    scaled -= scaled.max()
    exp_s = np.exp(scaled)
    probs = exp_s / exp_s.sum()
    full = np.zeros(len(CLASS_NAMES), dtype=np.float32)
    for i, cls in enumerate(clf.classes_):
        full[cls] = probs[i]
    idx = int(full.argmax())
    return idx, float(full[idx])


# ── Report helpers ────────────────────────────────────────────────────────────

def per_class_accuracy(gt, pred):
    out = {}
    for c, name in enumerate(CLASS_NAMES):
        mask = gt == c
        out[name] = (100.0 * (pred[mask] == c).mean(), int(mask.sum())) if mask.any() else (float("nan"), 0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="only the first N images (smoke test)")
    ap.add_argument("--no-fp32", action="store_true",
                    help="skip Arm C (FP32 torch encoder); halves runtime")
    args = ap.parse_args()
    with_fp32 = not args.no_fp32

    for path, what in ((ONNX_ENCODER, "INT8 ONNX encoder"), (TRAIN_CACHE, "training feature cache")):
        if not os.path.exists(path):
            raise FileNotFoundError(f"{what} not found: {path}")

    print("Loading HuggingFace processor (reference arm)...")
    from transformers import AutoImageProcessor
    processor = AutoImageProcessor.from_pretrained(DINOV2_MODEL_ID)

    print(f"Loading INT8 ONNX encoder: {os.path.basename(ONNX_ENCODER)}")
    import onnxruntime as ort
    session = ort.InferenceSession(ONNX_ENCODER, providers=["CPUExecutionProvider"])

    encoder_fp32 = None
    if with_fp32:
        print(f"Loading FP32 torch encoder: {DINOV2_MODEL_ID} (Arm C)")
        import torch
        from transformers import AutoModel
        encoder_fp32 = AutoModel.from_pretrained(DINOV2_MODEL_ID).eval()

    print("Rebuilding LogReg probe from the 1000-shot cache...")
    clf = train_probe()

    pairs = load_test_split()
    if args.limit:
        pairs = pairs[: args.limit]
    print(f"Test images: {len(pairs)}\n")

    rows = []
    t0 = time.perf_counter()
    for i, (stem, image_path, gt) in enumerate(pairs, 1):
        img = Image.open(image_path)

        px_a = processor(images=img, return_tensors="np")["pixel_values"].astype(np.float32)
        px_b = preprocess_dinov2(img).astype(np.float32)

        feat_a = normalize(session.run(None, {"pixel_values": px_a})[0], norm="l2")
        feat_b = normalize(session.run(None, {"pixel_values": px_b})[0], norm="l2")

        pred_a, conf_a = classify(clf, feat_a)
        pred_b, conf_b = classify(clf, feat_b)

        row = {
            "image": stem,
            "ground_truth": CLASS_NAMES[gt],
            "gt_idx": gt,
            "pred_hf": CLASS_NAMES[pred_a],
            "pred_hf_idx": pred_a,
            "conf_hf": round(conf_a, 6),
            "pred_numpy": CLASS_NAMES[pred_b],
            "pred_numpy_idx": pred_b,
            "conf_numpy": round(conf_b, 6),
            "feature_cosine": round(float(np.dot(feat_a[0], feat_b[0])), 8),
            # NOT rounded to 6dp: the real difference is ~3.6e-07 and rounding
            # displayed it as an exact zero, which hid the whole finding.
            "max_abs_pixel_diff": f"{float(np.abs(px_a - px_b).max()):.3e}",
        }

        if encoder_fp32 is not None:
            import torch
            with torch.no_grad():
                out = encoder_fp32(pixel_values=torch.from_numpy(px_a))
            feat_c = normalize(out.last_hidden_state[:, 0, :].numpy(), norm="l2")
            pred_c, conf_c = classify(clf, feat_c)
            row.update(
                {
                    "pred_fp32": CLASS_NAMES[pred_c],
                    "pred_fp32_idx": pred_c,
                    "conf_fp32": round(conf_c, 6),
                    "cosine_fp32_vs_int8hf": round(float(np.dot(feat_c[0], feat_a[0])), 8),
                    "cosine_fp32_vs_int8numpy": round(float(np.dot(feat_c[0], feat_b[0])), 8),
                }
            )

        rows.append(row)
        if i % 25 == 0 or i == len(pairs):
            print(f"  {i}/{len(pairs)}  ({time.perf_counter() - t0:.0f}s)")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    gt = np.array([r["gt_idx"] for r in rows])
    pa = np.array([r["pred_hf_idx"] for r in rows])
    pb = np.array([r["pred_numpy_idx"] for r in rows])
    cos = np.array([r["feature_cosine"] for r in rows])
    pxd = np.array([float(r["max_abs_pixel_diff"]) for r in rows])

    acc_a = 100.0 * (pa == gt).mean()
    acc_b = 100.0 * (pb == gt).mean()
    disagree = int((pa != pb).sum())
    has_fp32 = "pred_fp32_idx" in rows[0]
    if has_fp32:
        pc = np.array([r["pred_fp32_idx"] for r in rows])
        acc_c = 100.0 * (pc == gt).mean()
        cos_ca = np.array([r["cosine_fp32_vs_int8hf"] for r in rows])

    print("\n" + "=" * 66)
    print("DINOv2 preprocessing re-validation — AI4Mars gold test set")
    print("=" * 66)
    print(f"Images                         : {len(rows)}")
    print(f"Encoder                        : {os.path.basename(ONNX_ENCODER)} (identical both arms)")
    print(f"Probe                          : LogReg 1000-shot, C={LOGR_C}, T*={T_STAR}")
    print()
    print(f"Arm A  INT8 + AutoImageProcessor : {acc_a:.2f}%")
    print(f"Arm B  INT8 + preprocess_dinov2  : {acc_b:.2f}%   (deployed path)")
    if has_fp32:
        print(f"Arm C  FP32 + AutoImageProcessor : {acc_c:.2f}%   (quantization-free upper bound)")
    print(f"Difference (B - A)               : {acc_b - acc_a:+.2f} pp")
    if has_fp32:
        print(f"Cost of INT8 quantization (B - C): {acc_b - acc_c:+.2f} pp")
    print(f"A/B prediction disagreements     : {disagree}/{len(rows)} "
          f"({100.0 * disagree / len(rows):.2f}%)")
    print()
    print("Feature cosine A vs B            : "
          f"min {cos.min():.6f}  mean {cos.mean():.6f}  max {cos.max():.6f}")
    if has_fp32:
        print("Feature cosine FP32 vs INT8      : "
              f"min {cos_ca.min():.6f}  mean {cos_ca.mean():.6f}  max {cos_ca.max():.6f}")
    print("Max abs pixel difference A vs B  : "
          f"min {pxd.min():.3e}  mean {pxd.mean():.3e}  max {pxd.max():.3e}")
    print()
    header = f"{'class':<10}{'n':>6}{'Arm A':>10}{'Arm B':>10}{'B-A':>9}"
    if has_fp32:
        header += f"{'Arm C':>10}{'B-C':>9}"
    print(header)
    ca, cb = per_class_accuracy(gt, pa), per_class_accuracy(gt, pb)
    cc = per_class_accuracy(gt, pc) if has_fp32 else None
    for name in CLASS_NAMES:
        a, n = ca[name]
        b, _ = cb[name]
        if n == 0:
            line = f"{name:<10}{n:>6}{'-':>10}{'-':>10}{'-':>9}"
            if has_fp32:
                line += f"{'-':>10}{'-':>9}"
        else:
            line = f"{name:<10}{n:>6}{a:>9.2f}%{b:>9.2f}%{b - a:>+9.2f}"
            if has_fp32:
                c = cc[name][0]
                line += f"{c:>9.2f}%{b - c:>+9.2f}"
        print(line)

    print(f"\nPer-image CSV: {OUT_CSV}")

    print()
    if disagree == 0:
        print("VERDICT (A vs B): identical predictions on every image. The preprocessing "
              "swap does not change what the deployed rover classifies.")
    else:
        worse = int(((pa == gt) & (pb != gt)).sum())
        better = int(((pa != gt) & (pb == gt)).sum())
        print(f"VERDICT (A vs B): {disagree} images differ ({worse} correct->wrong, "
              f"{better} wrong->correct).")
        print("  Note the two preprocessors agree to ~1e-07, so these flips are the INT8 "
              "encoder amplifying float rounding, not a preprocessing defect. Neither arm "
              "is 'more correct' than the other; the spread between them is the noise floor "
              "of the quantized model, and any Pi accuracy figure must be quoted with it.")
    if has_fp32:
        print(f"VERDICT (INT8 cost): {acc_b - acc_c:+.2f} pp against the FP32 encoder "
              f"on the same {len(rows)} images.")


if __name__ == "__main__":
    main()
