"""
Purpose: Test-time augmentation (TTA) on the DINOv2+reg ViT-S/14 linear
         probe -- backlog item 25, scoped 2026-07-20. Re-extracts CLS
         features for the 287-image AI4Mars gold-standard test set under
         two views (original + horizontal flip, a safe augmentation for
         ground terrain texture where left/right orientation carries no
         class-relevant meaning), averages the two feature vectors per
         image, and compares accuracy against the existing single-view
         cached baseline. Backlog's own expectation going in: "+0.5-1pp,
         doesn't change conclusions" -- reported honestly either way.
Inputs:  experiments/results/feature_cache/dinov2_reg_small_train_1000_{feats,labels}.npy
         AI4Mars test images + gold-standard labels (raw, for re-extraction).
Outputs: experiments/results/tta_linear_probe.csv
         Printed baseline-vs-TTA accuracy comparison for Ch4.
How to run:
    python3 experiments/tta_linear_probe.py
Project: Onboard Visual Foundation Models for Mars Terrain Perception and Rover Navigation
"""
import csv
import os

import numpy as np
import torch
from PIL import Image, ImageOps
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize
from transformers import AutoImageProcessor, AutoModel

AI4MARS_BASE = "/mnt/c/Users/DELL/Desktop/Thesis/github source/ai4mars-dataset-merged-0.1"
IMAGES_DIR = os.path.join(AI4MARS_BASE, "msl/images/edr")
TEST_LABELS = os.path.join(AI4MARS_BASE, "msl/labels/test/masked-gold-min3-100agree")
CACHE_DIR = os.path.join(os.path.dirname(__file__), "results", "feature_cache")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

MODEL_ID = "facebook/dinov2-with-registers-small"  # thesis deployment model (90.24%)
LOGR_C = 0.316
IGNORE_PX = 255


def dominant_class(label_path):
    lbl = np.array(Image.open(label_path))
    valid = lbl[lbl != IGNORE_PX]
    if len(valid) == 0:
        return None
    return int(np.argmax(np.bincount(valid, minlength=4)))


def load_test_pairs():
    pairs = []
    for fname in sorted(os.listdir(TEST_LABELS)):
        if not fname.endswith(".png"):
            continue
        stem = fname.replace("_merged.png", "").replace(".png", "")
        img_path = os.path.join(IMAGES_DIR, stem + ".JPG")
        lbl_path = os.path.join(TEST_LABELS, fname)
        if not os.path.exists(img_path):
            continue
        gt = dominant_class(lbl_path)
        if gt is not None:
            pairs.append((img_path, gt))
    return pairs


def extract_cls(model, proc, device, image: Image.Image) -> np.ndarray:
    inputs = proc(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**inputs)
    return out.last_hidden_state[:, 0, :].cpu().numpy().squeeze()


def main():
    # CPU-only, matching every other feature-extraction script in this thesis
    # (this machine's CUDA/cuDNN stack is not usable for conv ops here --
    # "GET was unable to find an engine to execute this computation" -- and
    # per this thesis's standing dev-machine-constraint practice, environment
    # debugging is out of scope; CPU inference for a single ViT-S/14 forward
    # pass per image is fast enough for 287 images x 2 views).
    device = "cpu"
    print(f"Device: {device}")

    pairs = load_test_pairs()
    print(f"Test images: {len(pairs)}")

    tr_feats = np.load(os.path.join(CACHE_DIR, "dinov2_reg_small_train_1000_feats.npy"))
    tr_labels = np.load(os.path.join(CACHE_DIR, "dinov2_reg_small_train_1000_labels.npy"))
    baseline_test_feats = np.load(os.path.join(CACHE_DIR, "dinov2_reg_small_test_287_feats.npy"))
    baseline_test_labels = np.load(os.path.join(CACHE_DIR, "dinov2_reg_small_test_287_labels.npy"))

    proc = AutoImageProcessor.from_pretrained(MODEL_ID)
    model = AutoModel.from_pretrained(MODEL_ID).to(device).eval()

    original_feats, flipped_feats, labels = [], [], []
    for i, (img_path, gt) in enumerate(pairs):
        img = Image.open(img_path).convert("RGB")
        original_feats.append(extract_cls(model, proc, device, img))
        flipped_feats.append(extract_cls(model, proc, device, ImageOps.mirror(img)))
        labels.append(gt)
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(pairs)}")

    original_feats = np.array(original_feats)
    flipped_feats = np.array(flipped_feats)
    labels = np.array(labels)
    assert np.array_equal(labels, baseline_test_labels), \
        "re-extracted label order does not match the cached baseline order"

    tta_feats = (original_feats + flipped_feats) / 2.0

    clf = LogisticRegression(C=LOGR_C, max_iter=1000, random_state=42,
                              multi_class="multinomial", solver="lbfgs")
    clf.fit(normalize(tr_feats, norm="l2"), tr_labels)

    baseline_pred = clf.predict(normalize(baseline_test_feats, norm="l2"))
    baseline_acc = (baseline_pred == baseline_test_labels).mean()

    reextracted_pred = clf.predict(normalize(original_feats, norm="l2"))
    reextracted_acc = (reextracted_pred == labels).mean()

    tta_pred = clf.predict(normalize(tta_feats, norm="l2"))
    tta_acc = (tta_pred == labels).mean()

    out_path = os.path.join(RESULTS_DIR, "tta_linear_probe.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_index", "true_label", "baseline_pred", "tta_pred"])
        for i in range(len(labels)):
            writer.writerow([i, int(labels[i]), int(baseline_pred[i]), int(tta_pred[i])])

    print(f"\nCached baseline accuracy:            {100*baseline_acc:.2f}%")
    print(f"Re-extracted single-view accuracy:   {100*reextracted_acc:.2f}%  "
          f"(sanity check vs cached baseline, should match closely)")
    print(f"TTA (original + h-flip averaged):    {100*tta_acc:.2f}%  "
          f"({100*(tta_acc - reextracted_acc):+.2f}pp vs re-extracted single-view)")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
