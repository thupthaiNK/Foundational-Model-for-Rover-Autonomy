"""
Purpose: Zero-shot cross-dataset transfer test.
         Applies the AI4Mars-trained DINOv2+reg ViT-S/14 linear probe to
         Mars-Bench mb-surface_cls (MastCam RGB) without any retraining,
         to measure the NAVCAM-to-MastCam domain gap.
Inputs:
  - experiments/results/dinov2_logreg_probe.pkl  (4-class LogReg, AI4Mars 1000-shot)
  - HuggingFace Mirali33/mb-surface_cls          (test split, ~1594 images, 36 classes)
Outputs:
  - experiments/results/marsbench_zero_shot_transfer.csv
How to run:
  pip install datasets
  python3 experiments/marsbench_zero_shot_transfer.py
Project: Onboard Visual Foundation Models for Mars Terrain Perception and Rover Navigation
"""

import os
import pickle
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

# ── Paths ──────────────────────────────────────────────────────────────────────
REPO_ROOT  = Path(__file__).parent.parent
PROBE_PATH = REPO_ROOT / "experiments/results/dinov2_logreg_probe.pkl"
OUTPUT_CSV = REPO_ROOT / "experiments/results/marsbench_zero_shot_transfer.csv"

MODEL_NAME = "facebook/dinov2-with-registers-small"
HF_DATASET = "Mirali33/mb-surface_cls"
BATCH_SIZE = 32

# AI4Mars in-domain reference accuracy (from Exp §4.7.X)
AI4MARS_ACCURACY = 90.24

# ── Class mapping: mb-surface_cls idx → AI4Mars class index ───────────────────
# Conservative: only classes with clear, unambiguous traversability semantics.
# 26 of 36 mb classes are excluded (wheel, sun, hardware, arm, horizon, etc.)
CLASS_NAMES = ["soil", "bedrock", "sand", "big_rock"]  # matches probe.classes_

MB_TO_AI4MARS = {
    13: 0,   # gro  → soil     (ground/regolith)
    29: 2,   # san  → sand     (sand)
    8:  2,   # dri  → sand     (drift — aeolian deposit)
    9:  2,   # drh  → sand     (drift high-albedo)
    10: 2,   # drp  → sand     (drift ramp)
    11: 2,   # drt  → sand     (drift top)
    28: 1,   # rrd  → bedrock  (rough rocky debris)
    30: 1,   # sco  → bedrock  (soil with cobbles)
    12: 3,   # flr  → big_rock (float rock)
    16: 3,   # lar  → big_rock (large area rock)
}


def load_probe(path: Path):
    # Safe: this pkl is our own sklearn LogReg written by dinov2_terrain_test.py
    # in this same repo. It is not loaded from an untrusted external source.
    with open(path, "rb") as f:
        probe = pickle.load(f)
    print(f"Probe loaded — classes: {probe.classes_}, coef shape: {probe.coef_.shape}")
    return probe


def extract_features(images, model, processor, device):
    """Batch-extract DINOv2 CLS-token features from PIL RGB images."""
    all_feats = []
    n = len(images)
    for i in range(0, n, BATCH_SIZE):
        batch = images[i : i + BATCH_SIZE]
        inputs = processor(images=batch, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**inputs)
        # CLS token is index 0
        feats = out.last_hidden_state[:, 0, :].cpu().numpy()
        all_feats.append(feats)
        done = min(i + BATCH_SIZE, n)
        print(f"  features: {done}/{n}", end="\r", flush=True)
    print()
    return np.vstack(all_feats)


def main():
    print("=" * 60)
    print("Mars-Bench Zero-Shot Cross-Dataset Transfer")
    print("=" * 60)

    # ── Load probe ──────────────────────────────────────────────────────────
    probe = load_probe(PROBE_PATH)

    # ── Load DINOv2 encoder (same frozen model as AI4Mars experiments) ──────
    # Force CPU: MX330 VRAM (2 GB) cannot run DINOv2 conv ops on this machine
    device = "cpu"
    print(f"\nDevice: {device}")
    print(f"Loading {MODEL_NAME} ...")
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    model     = AutoModel.from_pretrained(MODEL_NAME).to(device).eval()
    print("Model ready.")

    # ── Download mb-surface_cls test split ──────────────────────────────────
    print(f"\nLoading {HF_DATASET} (test split) ...")
    from datasets import load_dataset
    ds = load_dataset(HF_DATASET, split="test")
    mb_class_names = ds.features["label"].names   # list of 36 short codes
    print(f"Test images total: {len(ds)}")

    # ── Filter to mapped classes ────────────────────────────────────────────
    images, y_true, mb_indices, mb_codes = [], [], [], []
    for item in ds:
        mb_idx = item["label"]
        if mb_idx not in MB_TO_AI4MARS:
            continue
        images.append(item["image"].convert("RGB"))
        y_true.append(MB_TO_AI4MARS[mb_idx])
        mb_indices.append(mb_idx)
        mb_codes.append(mb_class_names[mb_idx])

    y_true = np.array(y_true)
    print(f"After mapping filter: {len(images)} images\n")

    print("Mapped class distribution:")
    dist = Counter(mb_codes)
    for code in sorted(dist):
        mb_idx  = mb_class_names.index(code)
        ai4mars = CLASS_NAMES[MB_TO_AI4MARS[mb_idx]]
        print(f"  {code:5s} (mb {mb_idx:2d}) → {ai4mars:8s}: {dist[code]:4d} images")

    # ── Extract features ────────────────────────────────────────────────────
    print("\nExtracting DINOv2 features ...")
    t0 = time.time()
    features = extract_features(images, model, processor, device)
    elapsed  = time.time() - t0
    print(f"Done in {elapsed:.1f}s  ({elapsed/len(images)*1000:.1f} ms/img)")

    # ── Zero-shot prediction ────────────────────────────────────────────────
    print("\nRunning zero-shot prediction ...")
    y_pred     = probe.predict(features)
    y_prob_max = probe.predict_proba(features).max(axis=1)

    # ── Results ─────────────────────────────────────────────────────────────
    correct  = (y_pred == y_true).sum()
    total    = len(y_true)
    accuracy = correct / total * 100
    domain_gap = AI4MARS_ACCURACY - accuracy

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"{'Overall accuracy:':<30} {accuracy:.2f}%  ({correct}/{total})")
    print(f"{'AI4Mars in-domain (ref):':<30} {AI4MARS_ACCURACY:.2f}%")
    print(f"{'Domain gap (NAVCAM→MastCam):':<30} {domain_gap:+.2f} pp")
    print()

    per_class_acc = {}
    for idx, name in enumerate(CLASS_NAMES):
        mask = y_true == idx
        if mask.sum() == 0:
            per_class_acc[name] = None
            continue
        cls_acc = (y_pred[mask] == y_true[mask]).mean() * 100
        per_class_acc[name] = cls_acc
        n_cls = mask.sum()
        print(f"  {name:10s}: {cls_acc:5.2f}%  (n={n_cls})")

    # ── Save CSV ─────────────────────────────────────────────────────────────
    rows = []
    for i in range(total):
        rows.append({
            "mb_class_code":  mb_codes[i],
            "mb_class_idx":   mb_indices[i],
            "ai4mars_gt":     CLASS_NAMES[y_true[i]],
            "ai4mars_pred":   CLASS_NAMES[y_pred[i]],
            "confidence":     round(float(y_prob_max[i]), 4),
            "correct":        bool(y_pred[i] == y_true[i]),
        })

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved: {OUTPUT_CSV}  ({len(df)} rows)")

    # ── Thesis summary ────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("THESIS SUMMARY")
    print("=" * 60)
    print(f"Dataset:     Mars-Bench mb-surface_cls (MastCam RGB, NeurIPS 2025)")
    print(f"Model:       DINOv2+reg ViT-S/14, frozen encoder (no retraining)")
    print(f"Probe:       LogReg trained on AI4Mars NAVCAM grayscale (1000-shot)")
    print(f"Test subset: {total} images from {len(set(mb_codes))} mb classes → 4 AI4Mars classes")
    print(f"Accuracy:    {accuracy:.2f}%")
    print(f"Ref (AI4Mars in-domain): {AI4MARS_ACCURACY:.2f}%")
    print(f"Domain gap:  {domain_gap:+.2f} pp  (NAVCAM grayscale → MastCam RGB)")
    print()
    print("Per-class:")
    for name, acc in per_class_acc.items():
        if acc is None:
            print(f"  {name:10s}: N/A (no images)")
        else:
            print(f"  {name:10s}: {acc:.2f}%")


if __name__ == "__main__":
    main()
