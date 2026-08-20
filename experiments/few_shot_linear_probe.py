"""
Purpose: Few-shot linear probe on frozen CLIP ViT-B/32 features using AI4Mars train labels.
         Evaluates how much accuracy improves when N labelled images per class are available.
         Compares zero-shot (34.8%), prompt engineering v9 (54.4%), and this approach vs
         supervised baseline (96.67%).
Inputs:  AI4Mars train labels (16,064 images) + gold-standard test set (287 images)
Outputs: Accuracy table by number of shots, logged to wandb, saved to CSV
How to run:
    python3 few_shot_linear_probe.py --cpu
    python3 few_shot_linear_probe.py --cpu --log      # + log all results to wandb
    python3 few_shot_linear_probe.py --cpu --shots 10 50  # specific shot counts only
Project: Onboard Visual Foundation Models for Mars Terrain Perception and Rover Navigation
"""

import argparse
import csv
import os
import random
import time

import clip
import numpy as np
import torch
import wandb
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize

# ── Dataset paths ─────────────────────────────────────────────────────────────
AI4MARS_BASE    = "/mnt/c/Users/DELL/Desktop/Thesis/github source/ai4mars-dataset-merged-0.1"
IMAGES_DIR      = os.path.join(AI4MARS_BASE, "msl/images/edr")
TRAIN_LABELS    = os.path.join(AI4MARS_BASE, "msl/labels/train")
TEST_LABELS     = os.path.join(AI4MARS_BASE, "msl/labels/test/masked-gold-min3-100agree")

CLASS_NAMES         = ["soil", "bedrock", "sand", "big_rock"]
IGNORE_PIXEL        = 255
SUPERVISED_BASELINE = 96.67   # DeepLabv3+ ResNet-101, Swan et al. 2021
ZERO_SHOT_BASELINE  = 34.8    # CLIP v1 zero-shot (Experiment 1)
PROMPT_ENG_BEST     = 54.4    # CLIP v9 prompt engineering (Experiment 2)

SHOT_COUNTS = [10, 50, 100, 500, 1000]   # N images per class
RANDOM_SEED = 42


# ── Device ────────────────────────────────────────────────────────────────────

def get_device(force_cpu: bool) -> str:
    if force_cpu:
        return "cpu"
    if torch.cuda.is_available():
        try:
            torch.zeros(1).cuda()
            return "cuda"
        except Exception:
            print("CUDA available but failed — falling back to CPU")
    return "cpu"


# ── Ground truth ──────────────────────────────────────────────────────────────

def dominant_class(label_path: str):
    """Return the most frequent non-ignored class in a label PNG, or None."""
    label = np.array(Image.open(label_path))
    valid = label[label != IGNORE_PIXEL]
    if len(valid) == 0:
        return None
    return int(np.argmax(np.bincount(valid, minlength=4)))


def load_split(label_dir: str):
    """Return list of (image_path, class_index) for all valid images in label_dir."""
    pairs = []
    for fname in sorted(os.listdir(label_dir)):
        if not fname.endswith(".png"):
            continue
        # Train labels: NLA_xxx.png (no suffix)
        # Test labels:  NLA_xxx_merged.png
        stem = fname.replace("_merged.png", "").replace(".png", "")
        image_path = os.path.join(IMAGES_DIR, stem + ".JPG")
        label_path = os.path.join(label_dir, fname)
        if not os.path.exists(image_path):
            continue
        gt = dominant_class(label_path)
        if gt is not None:
            pairs.append((image_path, gt))
    return pairs


# ── Feature extraction ────────────────────────────────────────────────────────

def extract_features(model, preprocess, pairs, device, desc=""):
    """Extract normalised CLIP image features for a list of (image_path, label) pairs."""
    features, labels = [], []
    n = len(pairs)
    for i, (image_path, gt) in enumerate(pairs):
        if (i + 1) % 200 == 0 or i == n - 1:
            print(f"  {desc} {i+1}/{n}")
        image = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = model.encode_image(image).cpu().numpy().squeeze()
        features.append(feat)
        labels.append(gt)
    features = np.array(features, dtype=np.float32)
    features = normalize(features)   # L2 normalise — matches CLIP zero-shot protocol
    return features, np.array(labels)


# ── Few-shot sampling ─────────────────────────────────────────────────────────

def sample_n_per_class(pairs, n_per_class: int, seed: int = RANDOM_SEED):
    """
    Sample exactly n_per_class images per class.
    If a class has fewer than n_per_class images, use all available.
    """
    rng = random.Random(seed)
    by_class = {c: [] for c in range(len(CLASS_NAMES))}
    for pair in pairs:
        by_class[pair[1]].append(pair)
    sampled = []
    for c in range(len(CLASS_NAMES)):
        pool = by_class[c]
        k = min(n_per_class, len(pool))
        sampled.extend(rng.sample(pool, k))
    rng.shuffle(sampled)
    return sampled


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_probe(train_feats, train_labels, test_feats, test_labels):
    """Train logistic regression on train features, evaluate on test features."""
    clf = LogisticRegression(
        max_iter=1000,
        C=0.316,           # standard CLIP linear probe regularisation
        random_state=RANDOM_SEED,
    )
    clf.fit(train_feats, train_labels)
    preds = clf.predict(test_feats)

    n_classes = len(CLASS_NAMES)
    correct = {c: 0 for c in range(n_classes)}
    total   = {c: 0 for c in range(n_classes)}
    for pred, gt in zip(preds, test_labels):
        correct[gt] += int(pred == gt)
        total[gt]   += 1

    per_class = {
        name: (correct[i] / total[i] * 100 if total[i] > 0 else None)
        for i, name in enumerate(CLASS_NAMES)
    }
    n = len(test_labels)
    overall = sum(correct.values()) / n * 100 if n > 0 else 0
    return {"overall": overall, "per_class": per_class, "n_test": n}


# ── Reporting ─────────────────────────────────────────────────────────────────

def print_results(results: dict):
    shots = sorted(results.keys())
    col_w = 10
    width = 22 + col_w * len(shots)

    print("\n" + "=" * width)
    print("FEW-SHOT LINEAR PROBE — CLIP ViT-B/32 on AI4Mars")
    print("=" * width)

    header = f"{'':22}" + "".join(f"  {f'{s}-shot':>{col_w-2}}" for s in shots)
    print(header)
    print("-" * width)

    for name in CLASS_NAMES:
        row = f"{name:<22}"
        for s in shots:
            val = results[s]["per_class"].get(name)
            row += f"  {val:>{col_w-2}.1f}%" if val is not None else f"  {'N/A':>{col_w-2}}"
        print(row)

    print("-" * width)
    print(f"{'OVERALL':<22}" + "".join(f"  {results[s]['overall']:>{col_w-2}.1f}%" for s in shots))
    print(f"{'gap vs 96.67%':<22}" +
          "".join(f"  {results[s]['overall']-SUPERVISED_BASELINE:>+{col_w-2}.1f}%" for s in shots))
    print("=" * width)

    print(f"\nReference points:")
    print(f"  Zero-shot v1 (no labels):           {ZERO_SHOT_BASELINE:.1f}%")
    print(f"  Prompt engineering v9 (no labels):  {PROMPT_ENG_BEST:.1f}%")
    best_s = max(shots, key=lambda s: results[s]["overall"])
    print(f"  Best few-shot ({best_s}-shot):              {results[best_s]['overall']:.1f}%")
    print(f"  Supervised baseline:                {SUPERVISED_BASELINE:.1f}%")


def save_csv(results: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    shots = sorted(results.keys())
    fieldnames = ["shots", "overall", "gap"] + CLASS_NAMES
    rows = []
    for s in shots:
        r = results[s]
        row = {
            "shots":   s,
            "overall": f"{r['overall']:.2f}",
            "gap":     f"{r['overall'] - SUPERVISED_BASELINE:.2f}",
        }
        for name in CLASS_NAMES:
            val = r["per_class"].get(name)
            row[name] = f"{val:.2f}" if val is not None else "N/A"
        rows.append(row)
    # Add reference rows
    for label, val in [("zero_shot_v1", ZERO_SHOT_BASELINE),
                       ("prompt_eng_v9", PROMPT_ENG_BEST),
                       ("supervised", SUPERVISED_BASELINE)]:
        rows.append({"shots": label, "overall": f"{val:.2f}",
                     "gap": f"{val - SUPERVISED_BASELINE:.2f}",
                     **{name: "N/A" for name in CLASS_NAMES}})
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nCSV saved → {path}")


def log_to_wandb(results: dict):
    for shots, res in results.items():
        run = wandb.init(
            project="rover-autonomy-thesis",
            name=f"clip-vit-b32-linear-probe-{shots}shot",
            config={
                "model":       "CLIP ViT-B/32 (frozen)",
                "classifier":  "LogisticRegression C=0.316",
                "shots":       shots,
                "experiment":  "few_shot_linear_probe",
                "dataset":     "AI4Mars gold-standard test (287 images)",
            },
        )
        pc = res["per_class"]
        wandb.log({
            "accuracy/overall":             res["overall"],
            "accuracy/soil":                pc.get("soil")     or 0,
            "accuracy/bedrock":             pc.get("bedrock")  or 0,
            "accuracy/sand":                pc.get("sand")     or 0,
            "accuracy/big_rock":            pc.get("big_rock") or 0,
            "accuracy/supervised_baseline": SUPERVISED_BASELINE,
            "accuracy/zero_shot_baseline":  ZERO_SHOT_BASELINE,
            "accuracy/prompt_eng_v9":       PROMPT_ENG_BEST,
            "accuracy/gap_vs_baseline":     res["overall"] - SUPERVISED_BASELINE,
            "shots":                        shots,
        })
        wandb.finish()
    print("All runs logged to wandb → rover-autonomy-thesis")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Few-shot linear probe on CLIP features")
    parser.add_argument("--shots",  nargs="+", type=int, default=SHOT_COUNTS,
                        help=f"Shot counts to evaluate (default: {SHOT_COUNTS})")
    parser.add_argument("--log",    action="store_true", help="Log all results to wandb")
    parser.add_argument("--csv",    action="store_true", help="Save results to CSV")
    parser.add_argument("--cpu",    action="store_true", help="Force CPU")
    parser.add_argument("--model",  default="ViT-B/32",
                        choices=["ViT-B/32", "ViT-L/14"],
                        help="CLIP model variant (default: ViT-B/32)")
    args = parser.parse_args()

    device = get_device(args.cpu)
    print(f"Device: {device} | Model: {args.model}")

    print(f"Loading CLIP {args.model}...")
    model, preprocess = clip.load(args.model, device=device)
    model.eval()

    # ── Load test set ──────────────────────────────────────────────────────────
    print("Loading test set...")
    test_pairs = load_split(TEST_LABELS)
    print(f"Test set: {len(test_pairs)} images")

    print("Extracting test features...")
    t0 = time.perf_counter()
    test_feats, test_labels = extract_features(model, preprocess, test_pairs, device, "test")
    print(f"Test features done in {time.perf_counter()-t0:.1f}s")

    # ── Load train set (once, sample per shot count) ──────────────────────────
    print("\nLoading train set index...")
    train_pairs = load_split(TRAIN_LABELS)
    print(f"Train set: {len(train_pairs)} images available")

    # Class distribution in train set
    counts = [0] * len(CLASS_NAMES)
    for _, c in train_pairs:
        counts[c] += 1
    print("Train class distribution:", {CLASS_NAMES[i]: counts[i] for i in range(4)})

    # ── Extract train features once at max shots ───────────────────────────────
    max_shots = max(args.shots)
    print(f"\nSampling {max_shots} images/class for feature extraction...")
    max_sample = sample_n_per_class(train_pairs, max_shots)
    print(f"Extracting features for {len(max_sample)} train images...")
    t0 = time.perf_counter()
    all_train_feats, all_train_labels = extract_features(
        model, preprocess, max_sample, device, "train")
    print(f"Train features done in {time.perf_counter()-t0:.1f}s")

    # ── Evaluate each shot count ───────────────────────────────────────────────
    results = {}
    print()
    for shots in sorted(args.shots):
        # Sub-sample to exactly `shots` per class from the already-extracted features
        idx = []
        for c in range(len(CLASS_NAMES)):
            class_idx = [i for i, l in enumerate(all_train_labels) if l == c]
            k = min(shots, len(class_idx))
            idx.extend(class_idx[:k])

        train_f = all_train_feats[idx]
        train_l = all_train_labels[idx]
        n_actual = {CLASS_NAMES[c]: sum(1 for l in train_l if l == c)
                    for c in range(len(CLASS_NAMES))}

        print(f"── {shots}-shot  (actual per class: {n_actual})")
        res = evaluate_probe(train_f, train_l, test_feats, test_labels)
        results[shots] = res
        pc = res["per_class"]
        print(f"   Overall {res['overall']:.1f}%  |  "
              f"Soil {pc['soil']:.1f}%  "
              f"Bedrock {pc['bedrock']:.1f}%  "
              f"Sand {pc['sand']:.1f}%")

    print_results(results)

    csv_path = os.path.join(os.path.dirname(__file__), "results",
                            f"few_shot_linear_probe_{args.model.replace('/', '-')}.csv")
    save_csv(results, csv_path)

    if args.log:
        log_to_wandb(results)


if __name__ == "__main__":
    main()
