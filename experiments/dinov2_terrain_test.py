"""
Purpose: Few-shot linear probe on frozen DINOv2 ViT-S/14 features using AI4Mars dataset.
         Tests whether self-supervised features (DINOv2) fix bedrock misclassification
         compared to contrastive features (CLIP). This is the main thesis contribution:
         self-supervised vs contrastive on the same Mars terrain benchmark.
Inputs:  AI4Mars train labels (for few-shot sampling) + gold-standard test set (287 images)
Outputs: Per-class accuracy by shot count, CSV saved to results/, comparison vs CLIP baselines
How to run:
    python3 -u experiments/dinov2_terrain_test.py              # full run (10/50/100/500/1000-shot)
    python3 -u experiments/dinov2_terrain_test.py --shots 100 1000   # specific shots only
    python3 -u experiments/dinov2_terrain_test.py --shots 100        # quick test (~12 min)
Project: Onboard Visual Foundation Models for Mars Terrain Perception and Rover Navigation
"""

import argparse
import csv
import os
import random
import time

import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize
from transformers import AutoImageProcessor, AutoModel

# ── Paths ─────────────────────────────────────────────────────────────────────
AI4MARS_BASE = "/mnt/c/Users/DELL/Desktop/Thesis/github source/ai4mars-dataset-merged-0.1"
IMAGES_DIR   = os.path.join(AI4MARS_BASE, "msl/images/edr")
TRAIN_LABELS = os.path.join(AI4MARS_BASE, "msl/labels/train")
TEST_LABELS  = os.path.join(AI4MARS_BASE, "msl/labels/test/masked-gold-min3-100agree")
RESULTS_DIR  = os.path.join(os.path.dirname(__file__), "results")

CLASS_NAMES         = ["soil", "bedrock", "sand", "big_rock"]
IGNORE_PIXEL        = 255
SUPERVISED_BASELINE = 96.67

# ── CLIP baselines for direct comparison (from existing experiments) ──────────
CLIP_BASELINES = {
    10:   {"overall": 57.5, "soil": 82.1, "bedrock": 15.2, "sand": 69.4},
    50:   {"overall": 72.1, "soil": 88.6, "bedrock": 54.3, "sand": 66.7},
    100:  {"overall": 76.0, "soil": 87.8, "bedrock": 62.0, "sand": 73.6},
    500:  {"overall": 86.1, "soil": 94.3, "bedrock": 76.1, "sand": 84.7},
    1000: {"overall": 87.8, "soil": 95.9, "bedrock": 79.3, "sand": 84.7},
}

DEFAULT_SHOTS = [10, 50, 100, 500, 1000]
RANDOM_SEED   = 42


# ── Model ─────────────────────────────────────────────────────────────────────

def load_model(device: str):
    print("Loading DINOv2 ViT-S/14 (frozen)...")
    t0 = time.perf_counter()
    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
    model     = AutoModel.from_pretrained("facebook/dinov2-small").to(device).eval()
    load_s    = time.perf_counter() - t0
    n_params  = sum(p.numel() for p in model.parameters()) / 1e6
    feat_dim  = model.config.hidden_size
    print(f"  Loaded in {load_s:.1f}s  |  Params: {n_params:.1f}M  "
          f"|  Feature dim: {feat_dim}  |  Device: {device}")
    return model, processor


# ── Feature extraction ────────────────────────────────────────────────────────

def extract_features(model, processor, image_paths, labels, device: str, desc: str = ""):
    """Extract L2-normalised DINOv2 CLS-token features for a list of images."""
    features = []
    n = len(image_paths)
    t0 = time.perf_counter()

    for i, path in enumerate(image_paths):
        if (i + 1) % 100 == 0 or i == n - 1:
            elapsed = time.perf_counter() - t0
            eta = elapsed / (i + 1) * (n - i - 1)
            print(f"  {desc} {i+1}/{n}  ({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)")

        image  = Image.open(path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            feat = outputs.last_hidden_state[:, 0, :]   # CLS token → [1, 384]
            feat = feat.cpu().numpy().squeeze()          # [384]

        features.append(feat)

    features = np.array(features, dtype=np.float32)
    features = normalize(features)   # L2 normalise (same protocol as CLIP few-shot)
    return features, np.array(labels)


# ── Ground truth ──────────────────────────────────────────────────────────────

def dominant_class(label_path: str):
    label = np.array(Image.open(label_path))
    valid = label[label != IGNORE_PIXEL]
    if len(valid) == 0:
        return None
    return int(np.argmax(np.bincount(valid, minlength=4)))


def load_split(label_dir: str):
    pairs = []
    for fname in sorted(os.listdir(label_dir)):
        if not fname.endswith(".png"):
            continue
        stem       = fname.replace("_merged.png", "").replace(".png", "")
        image_path = os.path.join(IMAGES_DIR, stem + ".JPG")
        label_path = os.path.join(label_dir, fname)
        if not os.path.exists(image_path):
            continue
        gt = dominant_class(label_path)
        if gt is not None:
            pairs.append((image_path, gt))
    return pairs


# ── Few-shot sampling ─────────────────────────────────────────────────────────

def sample_n_per_class(pairs, n_per_class: int, seed: int = RANDOM_SEED):
    rng = random.Random(seed)
    by_class = {c: [] for c in range(len(CLASS_NAMES))}
    for pair in pairs:
        by_class[pair[1]].append(pair)
    sampled = []
    for c in range(len(CLASS_NAMES)):
        pool = by_class[c]
        rng.shuffle(pool)
        sampled.extend(pool[:n_per_class])
    return sampled


# ── Evaluation ────────────────────────────────────────────────────────────────

def run_few_shot(train_feats, train_labels, test_feats, test_labels, n_shots: int):
    """Train LogReg on n_shots×4 training features, evaluate on test set."""
    # subsample from pre-extracted training features
    sampled_idx = []
    rng = np.random.RandomState(RANDOM_SEED)
    for c in range(len(CLASS_NAMES)):
        idx = np.where(train_labels == c)[0]
        if len(idx) == 0:
            continue
        chosen = rng.choice(idx, size=min(n_shots, len(idx)), replace=False)
        sampled_idx.extend(chosen.tolist())

    X_train = train_feats[sampled_idx]
    y_train = train_labels[sampled_idx]

    clf = LogisticRegression(
        C=0.316, max_iter=1000, random_state=RANDOM_SEED,
        multi_class="multinomial", solver="lbfgs"
    )
    clf.fit(X_train, y_train)
    preds = clf.predict(test_feats)

    correct = {i: 0 for i in range(len(CLASS_NAMES))}
    total   = {i: 0 for i in range(len(CLASS_NAMES))}
    for pred, gt in zip(preds, test_labels):
        total[gt]   += 1
        correct[gt] += int(pred == gt)

    per_class = {
        CLASS_NAMES[i]: (correct[i] / total[i] * 100 if total[i] > 0 else None)
        for i in range(len(CLASS_NAMES))
    }
    n_total  = sum(total.values())
    overall  = sum(correct.values()) / n_total * 100 if n_total > 0 else 0
    return overall, per_class


# ── Reporting ─────────────────────────────────────────────────────────────────

def print_results(results: dict):
    shots = sorted(results.keys())
    col_w = 10

    print("\n" + "=" * 85)
    print("DINOv2 ViT-S/14 Few-Shot Linear Probe — AI4Mars (287 test images)")
    print("=" * 85)

    # header
    header = f"{'Shots':<8}"
    for s in shots:
        header += f"{'DINOv2':>{col_w}} {'CLIP':>{col_w}}"
    print(f"{'':8}" + "".join(f"{'DINOv2':>{col_w}} {'vs CLIP':>{col_w}}" for _ in shots))
    print(f"{'Shots':<8}" + "".join(f"{s:>{col_w}}{'':>{col_w}}" for s in shots))
    print("-" * 85)

    for metric in ["overall"] + CLASS_NAMES:
        row = f"{metric:<8}"
        for s in shots:
            dino_val = results[s]["overall"] if metric == "overall" else results[s]["per_class"].get(metric)
            clip_val = (CLIP_BASELINES.get(s, {}).get("overall")
                        if metric == "overall"
                        else CLIP_BASELINES.get(s, {}).get(metric))
            if dino_val is not None:
                row += f"  {dino_val:>7.1f}%"
            else:
                row += f"  {'N/A':>7}"
            if dino_val is not None and clip_val is not None:
                diff = dino_val - clip_val
                sign = "+" if diff >= 0 else ""
                row += f"  {sign}{diff:>+6.1f}%"
            else:
                row += f"  {'':>8}"
        print(row)

    print("=" * 85)

    # highlight bedrock — the key metric for this experiment
    print("\nKey metric — Bedrock (main thesis contribution):")
    for s in shots:
        dino_b = results[s]["per_class"].get("bedrock")
        clip_b = CLIP_BASELINES.get(s, {}).get("bedrock")
        if dino_b is not None and clip_b is not None:
            diff = dino_b - clip_b
            sign = "+" if diff >= 0 else ""
            print(f"  {s:>5}-shot:  DINOv2 {dino_b:.1f}%  vs  CLIP {clip_b:.1f}%  "
                  f"({sign}{diff:.1f}%)")


def save_csv(results: dict, avg_ms_extract: float):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, "dinov2_terrain_few_shot.csv")
    shots = sorted(results.keys())

    fieldnames = ["shots", "overall", "gap_vs_supervised"] + CLASS_NAMES + \
                 ["clip_overall"] + [f"clip_{n}" for n in CLASS_NAMES] + \
                 ["bedrock_delta", "avg_ms_extract"]

    rows = []
    for s in shots:
        r = results[s]
        clip = CLIP_BASELINES.get(s, {})
        row = {
            "shots":              s,
            "overall":            f"{r['overall']:.2f}",
            "gap_vs_supervised":  f"{r['overall'] - SUPERVISED_BASELINE:.2f}",
            "avg_ms_extract":     f"{avg_ms_extract:.1f}",
        }
        for name in CLASS_NAMES:
            val = r["per_class"].get(name)
            row[name] = f"{val:.2f}" if val is not None else "N/A"
        row["clip_overall"] = f"{clip.get('overall', 0):.2f}"
        for name in CLASS_NAMES:
            row[f"clip_{name}"] = f"{clip.get(name, 0):.2f}"
        bedrock_delta = (r["per_class"].get("bedrock") or 0) - clip.get("bedrock", 0)
        row["bedrock_delta"] = f"{bedrock_delta:+.2f}"
        rows.append(row)

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nCSV saved → {path}")
    return path


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="DINOv2 ViT-S/14 few-shot terrain probe")
    parser.add_argument("--shots", type=int, nargs="+", default=DEFAULT_SHOTS,
                        help="Shot counts to evaluate (default: 10 50 100 500 1000)")
    parser.add_argument("--cpu",   action="store_true", help="Force CPU")
    args = parser.parse_args()

    # MX330 GPU does not reliably support all transformer ops — use CPU
    device = "cpu"

    model, processor = load_model(device)

    # ── Load & extract test features (287 images, done once) ──────────────────
    print("\nLoading test set...")
    test_pairs = load_split(TEST_LABELS)
    print(f"Test images: {len(test_pairs)}")

    print("\nExtracting test features (287 images)...")
    t_test = time.perf_counter()
    test_paths  = [p for p, _ in test_pairs]
    test_labels = [l for _, l in test_pairs]
    test_feats, test_labels_arr = extract_features(
        model, processor, test_paths, test_labels, device, desc="test"
    )
    ms_per_img = (time.perf_counter() - t_test) / len(test_pairs) * 1000
    print(f"  Done — avg {ms_per_img:.0f}ms/img")

    # ── Load training set & sample max_shots per class ────────────────────────
    max_shots = max(args.shots)
    print(f"\nLoading train set (sampling up to {max_shots}/class)...")
    train_all = load_split(TRAIN_LABELS)
    train_sampled = sample_n_per_class(train_all, max_shots)

    by_class = {i: 0 for i in range(4)}
    for _, c in train_sampled:
        by_class[c] += 1
    print(f"  Train sample: " + ", ".join(f"{CLASS_NAMES[i]} {by_class[i]}" for i in range(4)))

    print(f"\nExtracting train features ({len(train_sampled)} images)...")
    train_paths  = [p for p, _ in train_sampled]
    train_labels = [l for _, l in train_sampled]
    train_feats, train_labels_arr = extract_features(
        model, processor, train_paths, train_labels, device, desc="train"
    )

    # ── Run few-shot probe for each shot count ─────────────────────────────────
    print("\nRunning few-shot linear probes...")
    results = {}
    for n_shots in sorted(args.shots):
        print(f"  {n_shots}-shot...", end=" ", flush=True)
        t0 = time.perf_counter()
        overall, per_class = run_few_shot(
            train_feats, train_labels_arr,
            test_feats,  test_labels_arr,
            n_shots
        )
        elapsed = time.perf_counter() - t0
        results[n_shots] = {"overall": overall, "per_class": per_class}
        clip_overall = CLIP_BASELINES.get(n_shots, {}).get("overall", 0)
        print(f"overall {overall:.1f}%  (CLIP {clip_overall:.1f}%)  [{elapsed:.1f}s]")

    print_results(results)
    save_csv(results, ms_per_img)

    print("\nDone. Key finding for thesis:")
    if 100 in results and 1000 in results:
        b100  = results[100]["per_class"].get("bedrock", 0)
        b1000 = results[1000]["per_class"].get("bedrock", 0)
        cb100  = CLIP_BASELINES[100]["bedrock"]
        cb1000 = CLIP_BASELINES[1000]["bedrock"]
        print(f"  Bedrock 100-shot:  DINOv2 {b100:.1f}%  vs  CLIP {cb100:.1f}%  "
              f"({'improved' if b100 > cb100 else 'not improved'})")
        print(f"  Bedrock 1000-shot: DINOv2 {b1000:.1f}%  vs  CLIP {cb1000:.1f}%  "
              f"({'improved' if b1000 > cb1000 else 'not improved'})")
    o100  = results.get(100, {}).get("overall", 0)
    o1000 = results.get(1000, {}).get("overall", 0)
    print(f"  Overall 100-shot:  {o100:.1f}%  |  1000-shot: {o1000:.1f}%")


if __name__ == "__main__":
    main()
