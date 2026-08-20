#!/usr/bin/env python3
"""
Purpose: Run the real deployed DINOv2 ViT-S/14 + LogReg terrain classifier
         (same encoder, same probe-fitting protocol as dinov2_terrain_node.py)
         against a set of real-world photos the user collected around campus
         and in the lab sandpit, to measure the sim-to-real / Mars-to-Earth
         perception gap directly rather than assume it.
         Folders with a known intended class (Bedrock/Big rock/Sand/Soil) are
         scored as accuracy against that label, split by source (the user's
         own campus/lab photos vs the AI4Mars images mixed into the same
         folders) so a real domain gap is not hidden inside one aggregate
         number. Folders with no AI4Mars-equivalent class (Other, Wall) are
         NOT scored -- there is no ground truth to score against -- they are
         reported as raw predicted class + confidence only, for exploratory
         failure-mode inspection (e.g. does a wall read as a rock).
Inputs:  Photo folders under "Terrain Picture" (path below), FP32 torch
         DINOv2 encoder (facebook/dinov2-with-registers-small) + the same
         1000-shot LogReg feature cache dinov2_terrain_node.py trains from.
Outputs: experiments/results/terrain_photo_gap_analysis.csv (per-image rows)
         experiments/results/terrain_photo_gap_analysis_summary.csv (per-folder accuracy)
         Printed summary table to stdout.
How to run:
    source /opt/ros/humble/setup.bash && source ros2_ws/install/setup.bash
    python3 experiments/terrain_photo_gap_analysis.py
    python3 experiments/terrain_photo_gap_analysis.py --photo-root "/mnt/c/Users/DELL/Downloads/Terrain Picture"
Project: Onboard Visual Foundation Models for Mars Terrain Perception and Rover Navigation
"""

import argparse
import csv
import os
import sys
import time

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
    fit_or_load_logreg, traversability_score,
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
DEFAULT_CACHE = os.path.join(
    RESULTS_DIR, "feature_cache", "dinov2_reg_small_train_1000shot.npz"
)
DEFAULT_PHOTO_ROOT = "/mnt/c/Users/DELL/Downloads/Terrain Picture"
IMG_EXT = (".jpg", ".jpeg", ".png")

# (relative dir under photo root, expected class or None, source tag)
# expected=None folders are exploratory only (no scored accuracy).
FOLDER_SPEC = [
    ("1. AI4MAR Bedrock",          "bedrock",  "AI4Mars"),
    ("2. Big rock",                "big_rock", "lab_sandpit"),
    ("3. Sand",                    "sand",     "lab_sandpit"),      # top-level files only
    ("3. Sand/AI4MAR Sand",        "sand",     "AI4Mars"),
    ("4. Soil",                    "soil",     "campus"),           # top-level files only
    ("4. Soil/AI4MAR Soil",        "soil",     "AI4Mars"),
    ("Other",                      None,       "campus_exploratory"),
    ("Wall",                       None,       "campus_wall"),
]


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


def list_images(root: str, rel_dir: str, recurse_into_ai4mars_subdir: bool = False):
    """Files directly inside root/rel_dir. Does NOT recurse into subfolders,
    so e.g. '3. Sand' does not silently pull in '3. Sand/AI4MAR Sand'."""
    d = os.path.join(root, rel_dir)
    if not os.path.isdir(d):
        return []
    out = []
    for name in sorted(os.listdir(d)):
        p = os.path.join(d, name)
        if os.path.isfile(p) and name.lower().endswith(IMG_EXT):
            out.append(p)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--photo-root", default=DEFAULT_PHOTO_ROOT)
    ap.add_argument("--cache-path", default=DEFAULT_CACHE)
    ap.add_argument("--n-shot", type=int, default=1000)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    print(f"Fitting/loading LogReg probe from {args.cache_path} (n_shot={args.n_shot})...")
    clf, fitted = fit_or_load_logreg(args.cache_path, args.n_shot, class_weight_balanced=False)
    print(f"Probe {'fitted' if fitted else 'loaded from cache'}.")

    processor, model = load_encoder(args.device)

    rows = []
    for rel_dir, expected, source in FOLDER_SPEC:
        paths = list_images(args.photo_root, rel_dir)
        if not paths:
            print(f"WARNING: no images found in '{rel_dir}' -- skipping")
            continue
        print(f"\n{rel_dir}  ({len(paths)} images, expected={expected}, source={source})")
        for p in paths:
            img = Image.open(p).convert("RGB")
            t0 = time.perf_counter()
            feat = extract_features(processor, model, args.device, img)
            label, confidence, probs = classify(clf, feat)
            ms = (time.perf_counter() - t0) * 1000
            score = traversability_score(probs, confidence)
            correct = (label == expected) if expected else None
            rows.append({
                "folder": rel_dir,
                "source": source,
                "filename": os.path.basename(p),
                "expected_class": expected or "",
                "predicted_class": label,
                "confidence": round(confidence, 4),
                "traversability_score": round(score, 4),
                "correct": "" if correct is None else int(correct),
                "p_soil": round(float(probs[0]), 4),
                "p_bedrock": round(float(probs[1]), 4),
                "p_sand": round(float(probs[2]), 4),
                "p_big_rock": round(float(probs[3]), 4),
                "inference_ms": round(ms, 1),
            })
            tag = "" if correct is None else ("OK" if correct else "WRONG")
            print(f"  {os.path.basename(p):45s} -> {label:9s} ({confidence:.3f}) {tag}")

    csv_path = os.path.join(RESULTS_DIR, "terrain_photo_gap_analysis.csv")
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nPer-image results written to {csv_path}")

    # ── Per-folder summary (accuracy only where expected_class is known) ──
    summary_path = os.path.join(RESULTS_DIR, "terrain_photo_gap_analysis_summary.csv")
    summary_rows = []
    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    seen = {}
    for r in rows:
        seen.setdefault((r["folder"], r["source"]), []).append(r)
    for rel_dir, expected, source in FOLDER_SPEC:
        group = seen.get((rel_dir, source), [])
        if not group:
            continue
        n = len(group)
        if expected:
            n_correct = sum(int(r["correct"]) for r in group)
            acc = 100.0 * n_correct / n
            print(f"{rel_dir:28s} [{source:18s}] n={n:3d}  accuracy={acc:5.1f}%  "
                  f"(expected={expected})")
            summary_rows.append({
                "folder": rel_dir, "source": source, "expected_class": expected,
                "n": n, "n_correct": n_correct, "accuracy_pct": round(acc, 1),
            })
        else:
            from collections import Counter
            dist = Counter(r["predicted_class"] for r in group)
            dist_str = ", ".join(f"{k}:{v}" for k, v in dist.most_common())
            print(f"{rel_dir:28s} [{source:18s}] n={n:3d}  predicted dist: {dist_str}")
            summary_rows.append({
                "folder": rel_dir, "source": source, "expected_class": "",
                "n": n, "n_correct": "", "accuracy_pct": "",
                "predicted_distribution": dist_str,
            })

    fieldnames = sorted({k for r in summary_rows for k in r.keys()})
    with open(summary_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"\nPer-folder summary written to {summary_path}")


if __name__ == "__main__":
    main()
