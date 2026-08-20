"""
Purpose: Experiment 10 — DINOv2+reg ViT-S trained and evaluated on Mars-Bench
         surface_classification (36 classes, MastCam/MAHILI RGB imagery).
         Unlike Exp §4.7.32 (zero-shot with AI4Mars probe), this script trains
         a fresh 36-class LogReg probe directly on Mars-Bench training data,
         giving a fair in-domain baseline for the leaderboard.
         Produces predictions CSV compatible with the Mars-Bench leaderboard
         at https://huggingface.co/spaces/Mirali33/Mars-Bench

Inputs:
  - HuggingFace Mirali33/mb-surface_cls  (train + test splits)
  - facebook/dinov2-with-registers-small  (frozen encoder, same as thesis)
Outputs:
  - experiments/results/marsbench_dinov2_exp10_results.csv
  - experiments/results/marsbench_dinov2_exp10_predictions.csv
  - experiments/results/feature_cache/marsbench_train_feats.npy  (cached)
  - experiments/results/feature_cache/marsbench_train_labels.npy
  - experiments/results/feature_cache/marsbench_test_feats.npy
  - experiments/results/feature_cache/marsbench_test_labels.npy

How to run:
  python3 experiments/marsbench_dinov2_exp10.py

  # To reuse cached features (skip re-extraction):
  python3 experiments/marsbench_dinov2_exp10.py --use-cache

  # Leaderboard submission (after running):
  # 1. Upload marsbench_dinov2_exp10_predictions.csv to:
  #    https://huggingface.co/spaces/Mirali33/Mars-Bench
  # 2. Follow the Space UI to submit predictions on held-out test set.

Project: Onboard Visual Foundation Models for Mars Terrain Perception and Rover Navigation
"""

import argparse
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import normalize
from transformers import AutoImageProcessor, AutoModel

# ── Paths ──────────────────────────────────────────────────────────────────────
REPO_ROOT   = Path(__file__).parent.parent
CACHE_DIR   = REPO_ROOT / "experiments/results/feature_cache"
OUTPUT_DIR  = REPO_ROOT / "experiments/results"
RESULTS_CSV = OUTPUT_DIR / "marsbench_dinov2_exp10_results.csv"
PREDS_CSV   = OUTPUT_DIR / "marsbench_dinov2_exp10_predictions.csv"

CACHE_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "facebook/dinov2-with-registers-small"
HF_DATASET = "Mirali33/mb-surface_cls"
BATCH_SIZE = 32
LOGR_C     = 0.316   # same as all other thesis experiments


# ── Feature extraction ─────────────────────────────────────────────────────────

def extract_features(images: list, model, processor, device: str) -> np.ndarray:
    all_feats = []
    n = len(images)
    for i in range(0, n, BATCH_SIZE):
        batch = [img.convert("RGB") for img in images[i : i + BATCH_SIZE]]
        inputs = processor(images=batch, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**inputs)
        feats = out.last_hidden_state[:, 0, :].cpu().numpy()  # CLS token
        all_feats.append(feats)
        done = min(i + BATCH_SIZE, n)
        print(f"  {done}/{n}", end="\r", flush=True)
    print()
    return np.vstack(all_feats)


def load_or_extract(split: str, model, processor, device: str, use_cache: bool):
    feat_path  = CACHE_DIR / f"marsbench_{split}_feats.npy"
    label_path = CACHE_DIR / f"marsbench_{split}_labels.npy"
    ids_path   = CACHE_DIR / f"marsbench_{split}_ids.npy"

    if use_cache and feat_path.exists() and label_path.exists():
        print(f"  Loading cached {split} features ...")
        feats  = np.load(feat_path)
        labels = np.load(label_path)
        # allow_pickle=True: safe — file is our own cached string array written by this script
        ids    = np.load(ids_path, allow_pickle=True) if ids_path.exists() else np.arange(len(labels))
        print(f"  {len(labels)} samples (from cache)")
        return feats, labels, ids

    print(f"\nLoading {HF_DATASET} ({split} split) ...")
    from datasets import load_dataset
    try:
        ds = load_dataset(HF_DATASET, split=split)
    except ValueError:
        # Dataset may not have a named train split — load all and split manually
        print(f"  '{split}' split not found. Loading full dataset and splitting 80/20 ...")
        full = load_dataset(HF_DATASET, split="test")
        n_total = len(full)
        n_train = int(n_total * 0.8)
        if split == "train":
            ds = full.select(range(n_train))
        else:
            ds = full.select(range(n_train, n_total))

    class_names = ds.features["label"].names
    images = [item["image"] for item in ds]
    labels = np.array([item["label"] for item in ds])
    ids    = np.array([str(i) for i in range(len(images))], dtype=object)

    print(f"  {len(images)} images, {len(class_names)} classes")
    print(f"  Extracting features ...")
    t0 = time.time()
    feats = extract_features(images, model, processor, device)
    print(f"  Done in {time.time()-t0:.1f}s")

    np.save(feat_path,  feats)
    np.save(label_path, labels)
    np.save(ids_path,   ids)
    print(f"  Features cached → {feat_path}")
    return feats, labels, ids


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-cache", action="store_true",
                        help="Reuse cached features instead of re-extracting")
    args = parser.parse_args()

    print("=" * 65)
    print("Experiment 10 — DINOv2 on Mars-Bench surface_classification")
    print("=" * 65)

    device = "cpu"
    print(f"\nDevice: {device}")
    print(f"Loading {MODEL_NAME} ...")
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    model     = AutoModel.from_pretrained(MODEL_NAME).to(device).eval()
    n_params  = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model ready — {n_params:.1f}M params")

    # ── Extract or load features ───────────────────────────────────────────────
    print("\n[TRAIN SPLIT]")
    train_feats, train_labels, _ = load_or_extract(
        "train", model, processor, device, args.use_cache
    )

    print("\n[TEST SPLIT]")
    test_feats, test_labels, test_ids = load_or_extract(
        "test", model, processor, device, args.use_cache
    )

    # ── Class names from test split ────────────────────────────────────────────
    from datasets import load_dataset
    ds_meta   = load_dataset(HF_DATASET, split="test")
    class_names = ds_meta.features["label"].names   # 36 class codes

    n_classes = len(class_names)
    print(f"\n{n_classes} classes: {', '.join(class_names[:8])} ...")
    print(f"Train: {len(train_labels)} samples  |  Test: {len(test_labels)} samples")

    # ── Train LogReg probe (36 classes) ───────────────────────────────────────
    print("\nTraining 36-class LogReg probe ...")
    X_train = normalize(train_feats, norm="l2")
    X_test  = normalize(test_feats,  norm="l2")

    t0 = time.time()
    clf = LogisticRegression(C=LOGR_C, max_iter=2000, solver="lbfgs",
                             random_state=42, multi_class="multinomial")
    clf.fit(X_train, train_labels)
    print(f"Training done in {time.time()-t0:.1f}s")

    # ── Evaluate ───────────────────────────────────────────────────────────────
    y_pred      = clf.predict(X_test)
    y_prob      = clf.predict_proba(X_test)
    y_prob_max  = y_prob.max(axis=1)

    correct  = (y_pred == test_labels).sum()
    total    = len(test_labels)
    accuracy = correct / total * 100

    print("\n" + "=" * 65)
    print("RESULTS")
    print("=" * 65)
    print(f"Overall accuracy:            {accuracy:.2f}%  ({correct}/{total})")
    print(f"Zero-shot baseline (§4.7.32): 24.94%  (AI4Mars probe, 10-class map)")
    print(f"Improvement:                 +{accuracy - 24.94:.2f} pp")

    # Per-class accuracy
    print(f"\nPer-class accuracy (36 classes):")
    print(f"  {'Class':>5}  {'Name':<6}  {'Acc':>7}  {'N':>5}")
    print(f"  {'─'*5}  {'─'*6}  {'─'*7}  {'─'*5}")
    per_class_rows = []
    for idx in sorted(np.unique(test_labels)):
        name   = class_names[idx]
        mask   = test_labels == idx
        n_cls  = mask.sum()
        if n_cls == 0:
            acc_cls = float("nan")
        else:
            acc_cls = (y_pred[mask] == test_labels[mask]).mean() * 100
        print(f"  {idx:>5}  {name:<6}  {acc_cls:>6.1f}%  {n_cls:>5}")
        per_class_rows.append({
            "class_idx":  idx,
            "class_name": name,
            "n_test":     int(n_cls),
            "accuracy":   round(acc_cls, 2) if not np.isnan(acc_cls) else None,
        })

    # ── Save results CSV ───────────────────────────────────────────────────────
    results_df = pd.DataFrame(per_class_rows)
    results_df.to_csv(RESULTS_CSV, index=False)
    print(f"\nPer-class results → {RESULTS_CSV}")

    # ── Save predictions CSV (for leaderboard submission) ─────────────────────
    # Format: image_index, predicted_class_idx, predicted_class_name, confidence
    preds_df = pd.DataFrame({
        "image_index":          list(range(total)),
        "true_label_idx":       test_labels.tolist(),
        "true_label_name":      [class_names[i] for i in test_labels],
        "pred_label_idx":       y_pred.tolist(),
        "pred_label_name":      [class_names[i] for i in y_pred],
        "confidence":           np.round(y_prob_max, 4).tolist(),
        "correct":              (y_pred == test_labels).tolist(),
    })
    preds_df.to_csv(PREDS_CSV, index=False)
    print(f"Predictions (leaderboard) → {PREDS_CSV}")

    # ── Summary ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("COMPARISON TABLE (Thesis §4.7.32 / §4.X Exp 10)")
    print("=" * 65)
    print(f"  {'Protocol':<40} {'Acc':>8}")
    print(f"  {'─'*40} {'─'*8}")
    print(f"  {'Zero-shot (AI4Mars probe → MB, 10 classes)':<40} {'24.94%':>8}")
    print(f"  {'In-domain (DINOv2 probe trained on MB train)':<40} {accuracy:>7.2f}%")
    print(f"  {'─'*40} {'─'*8}")
    print(f"  {'Domain gap resolved by in-domain training:':<40} {accuracy - 24.94:>+7.2f}pp")
    print("=" * 65)
    print("\nLeaderboard submission:")
    print("  1. Go to https://huggingface.co/spaces/Mirali33/Mars-Bench")
    print(f"  2. Upload: {PREDS_CSV}")
    print("  3. Select task: surface_classification")
    print("  4. Enter model name: DINOv2+reg ViT-S/14 (frozen + LogReg probe)")
    print("=" * 65)

    return accuracy


if __name__ == "__main__":
    main()
