"""
Purpose: Few-shot linear probe on frozen DINOv3 ViT-S/16 features using AI4Mars dataset.
         Compares DINOv3 (2025, self-supervised distilled) vs DINOv2 (2023) and CLIP on the
         same Mars terrain benchmark. Tests whether the newer self-supervised model further
         improves bedrock classification accuracy.
Inputs:  AI4Mars train labels (few-shot sampling) + gold-standard test set (287 images)
Outputs: Per-class accuracy by shot count, CSV saved to results/, comparison vs DINOv2 + CLIP
How to run:
    python3 -u experiments/dinov3_terrain_test.py              # full run (10/50/100/500/1000-shot)
    python3 -u experiments/dinov3_terrain_test.py --shots 100 1000   # specific shots only
    python3 -u experiments/dinov3_terrain_test.py --shots 10         # quick smoke test (~3 min)
Note: Requires HuggingFace token (gated model). Run `hf auth login` first or set HF_TOKEN.
Reference: Siméoni et al. (2025) DINOv3. Meta AI. arXiv:2508.10104
           https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m
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

MODEL_ID            = "facebook/dinov3-vits16-pretrain-lvd1689m"
CLASS_NAMES         = ["soil", "bedrock", "sand", "big_rock"]
IGNORE_PIXEL        = 255
SUPERVISED_BASELINE = 96.67

# ── Baselines for comparison ──────────────────────────────────────────────────
CLIP_BASELINES = {
    10:   {"overall": 57.5,  "soil": 82.1,  "bedrock": 15.2, "sand": 69.4},
    50:   {"overall": 72.1,  "soil": 88.6,  "bedrock": 54.3, "sand": 66.7},
    100:  {"overall": 76.0,  "soil": 87.8,  "bedrock": 62.0, "sand": 73.6},
    500:  {"overall": 86.1,  "soil": 94.3,  "bedrock": 76.1, "sand": 84.7},
    1000: {"overall": 87.8,  "soil": 95.9,  "bedrock": 79.3, "sand": 84.7},
}

DINOV2_BASELINES = {
    10:   {"overall": 56.79, "soil": 48.78, "bedrock": 56.52, "sand": 70.83},
    50:   {"overall": 76.31, "soil": None,  "bedrock": 50.00, "sand": 79.17},
    100:  {"overall": 81.18, "soil": 96.75, "bedrock": 55.43, "sand": 87.50},
    500:  {"overall": 88.50, "soil": None,  "bedrock": 81.52, "sand": 81.94},
    1000: {"overall": 89.90, "soil": 97.56, "bedrock": 84.78, "sand": 83.33},
}

DEFAULT_SHOTS = [10, 50, 100, 500, 1000]
RANDOM_SEED   = 42


# ── Model ─────────────────────────────────────────────────────────────────────

def load_model(device: str):
    print(f"Loading DINOv3 ViT-S/16 (frozen)  [{MODEL_ID}]...")
    t0 = time.perf_counter()
    processor = AutoImageProcessor.from_pretrained(MODEL_ID, token=True)
    model     = AutoModel.from_pretrained(MODEL_ID, token=True).to(device).eval()
    load_s    = time.perf_counter() - t0
    n_params  = sum(p.numel() for p in model.parameters()) / 1e6
    feat_dim  = model.config.hidden_size
    print(f"  Loaded in {load_s:.1f}s  |  Params: {n_params:.1f}M  "
          f"|  Feature dim: {feat_dim}  |  Device: {device}")
    return model, processor


# ── Feature extraction ────────────────────────────────────────────────────────

def extract_features(model, processor, image_paths, labels, device: str, desc: str = ""):
    """Extract L2-normalised DINOv3 CLS-token features for a list of images."""
    features = []
    n  = len(image_paths)
    t0 = time.perf_counter()

    for i, path in enumerate(image_paths):
        if (i + 1) % 100 == 0 or i == n - 1:
            elapsed = time.perf_counter() - t0
            eta     = elapsed / (i + 1) * (n - i - 1)
            print(f"  {desc} {i+1}/{n}  ({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)")

        image  = Image.open(path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            feat    = outputs.last_hidden_state[:, 0, :]  # CLS token → [1, 384]
            feat    = feat.cpu().numpy().squeeze()         # [384]

        features.append(feat)

    features = np.array(features, dtype=np.float32)
    features = normalize(features)
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
    rng      = random.Random(seed)
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
    rng = np.random.RandomState(RANDOM_SEED)
    sampled_idx = []
    for c in range(len(CLASS_NAMES)):
        idx    = np.where(train_labels == c)[0]
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
    n_total = sum(total.values())
    overall = sum(correct.values()) / n_total * 100 if n_total > 0 else 0
    return overall, per_class


# ── Reporting ─────────────────────────────────────────────────────────────────

def print_results(results: dict):
    shots = sorted(results.keys())
    print("\n" + "=" * 100)
    print("DINOv3 ViT-S/16 Few-Shot Linear Probe — AI4Mars (287 test images)")
    print("Comparison: DINOv3 vs DINOv2 vs CLIP (same protocol, same test set)")
    print("=" * 100)

    for metric in ["overall"] + CLASS_NAMES:
        row = f"{metric:<10}"
        for s in shots:
            v3_val   = results[s]["overall"] if metric == "overall" else results[s]["per_class"].get(metric)
            v2_val   = (DINOV2_BASELINES.get(s, {}).get("overall")
                        if metric == "overall"
                        else DINOV2_BASELINES.get(s, {}).get(metric))
            clip_val = (CLIP_BASELINES.get(s, {}).get("overall")
                        if metric == "overall"
                        else CLIP_BASELINES.get(s, {}).get(metric))

            v3_str   = f"{v3_val:>6.1f}%" if v3_val is not None else "   N/A"
            v2_str   = f"{v2_val:>6.1f}%" if v2_val is not None else "   N/A"
            delta_v2 = f"{(v3_val - v2_val):>+6.1f}%" if (v3_val is not None and v2_val is not None) else "    N/A"
            row += f"  [{s:>4}shot] DINOv3:{v3_str} DINOv2:{v2_str} Δ:{delta_v2}"
        print(row)

    print("=" * 100)
    print("\nKey metric — Bedrock (main thesis contribution):")
    for s in shots:
        v3_b   = results[s]["per_class"].get("bedrock")
        v2_b   = DINOV2_BASELINES.get(s, {}).get("bedrock")
        clip_b = CLIP_BASELINES.get(s, {}).get("bedrock")
        if v3_b is not None:
            d_v2   = f"{(v3_b - v2_b):>+.1f}%" if v2_b is not None else "N/A"
            d_clip = f"{(v3_b - clip_b):>+.1f}%" if clip_b is not None else "N/A"
            print(f"  {s:>5}-shot:  DINOv3 {v3_b:.1f}%  |  DINOv2 {v2_b:.1f}%  (Δ {d_v2})  "
                  f"|  CLIP {clip_b:.1f}%  (Δ {d_clip})")


def save_csv(results: dict, avg_ms_extract: float):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, "dinov3_terrain_few_shot.csv")
    shots = sorted(results.keys())

    fieldnames = (["shots", "overall", "gap_vs_supervised"] + CLASS_NAMES +
                  ["dinov2_overall", "dinov2_bedrock", "dinov2_delta_overall", "dinov2_delta_bedrock",
                   "clip_overall",   "clip_bedrock",   "avg_ms_extract"])

    rows = []
    for s in shots:
        r    = results[s]
        dv2  = DINOV2_BASELINES.get(s, {})
        clip = CLIP_BASELINES.get(s, {})
        row  = {
            "shots":             s,
            "overall":           f"{r['overall']:.2f}",
            "gap_vs_supervised": f"{r['overall'] - SUPERVISED_BASELINE:.2f}",
            "avg_ms_extract":    f"{avg_ms_extract:.1f}",
        }
        for name in CLASS_NAMES:
            val      = r["per_class"].get(name)
            row[name] = f"{val:.2f}" if val is not None else "N/A"

        row["dinov2_overall"]       = f"{dv2.get('overall', 0):.2f}"
        row["dinov2_bedrock"]       = f"{dv2.get('bedrock', 0):.2f}"
        row["dinov2_delta_overall"] = (f"{r['overall'] - dv2['overall']:+.2f}"
                                       if dv2.get("overall") is not None else "N/A")
        row["dinov2_delta_bedrock"] = (f"{(r['per_class'].get('bedrock') or 0) - dv2['bedrock']:+.2f}"
                                       if dv2.get("bedrock") is not None else "N/A")
        row["clip_overall"]         = f"{clip.get('overall', 0):.2f}"
        row["clip_bedrock"]         = f"{clip.get('bedrock', 0):.2f}"
        rows.append(row)

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nCSV saved → {path}")
    return path


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="DINOv3 ViT-S/16 few-shot terrain probe")
    parser.add_argument("--shots", type=int, nargs="+", default=DEFAULT_SHOTS,
                        help="Shot counts to evaluate (default: 10 50 100 500 1000)")
    args = parser.parse_args()

    device = "cpu"  # MX330 does not reliably support all transformer ops

    model, processor = load_model(device)

    print("\nLoading test set...")
    test_pairs  = load_split(TEST_LABELS)
    print(f"Test images: {len(test_pairs)}")

    print("\nExtracting test features (287 images)...")
    t_test       = time.perf_counter()
    test_paths   = [p for p, _ in test_pairs]
    test_labels  = [l for _, l in test_pairs]
    test_feats, test_labels_arr = extract_features(
        model, processor, test_paths, test_labels, device, desc="test"
    )
    ms_per_img = (time.perf_counter() - t_test) / len(test_pairs) * 1000
    print(f"  Done — avg {ms_per_img:.0f}ms/img")

    max_shots = max(args.shots)
    print(f"\nLoading train set (up to {max_shots}/class)...")
    train_all     = load_split(TRAIN_LABELS)
    train_sampled = sample_n_per_class(train_all, max_shots)

    by_class = {i: 0 for i in range(4)}
    for _, c in train_sampled:
        by_class[c] += 1
    print("  Train sample: " + ", ".join(f"{CLASS_NAMES[i]} {by_class[i]}" for i in range(4)))

    print(f"\nExtracting train features ({len(train_sampled)} images)...")
    train_paths  = [p for p, _ in train_sampled]
    train_labels = [l for _, l in train_sampled]
    train_feats, train_labels_arr = extract_features(
        model, processor, train_paths, train_labels, device, desc="train"
    )

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
        elapsed     = time.perf_counter() - t0
        results[n_shots] = {"overall": overall, "per_class": per_class}
        dv2_overall = DINOV2_BASELINES.get(n_shots, {}).get("overall", 0)
        delta       = overall - dv2_overall
        print(f"overall {overall:.1f}%  (DINOv2 {dv2_overall:.1f}%  Δ{delta:+.1f}%)  [{elapsed:.1f}s]")

    print_results(results)
    save_csv(results, ms_per_img)

    print("\nDone. Key finding for thesis:")
    for s in [100, 1000]:
        if s not in results:
            continue
        v3_b  = results[s]["per_class"].get("bedrock", 0)
        v2_b  = DINOV2_BASELINES.get(s, {}).get("bedrock", 0)
        v3_ov = results[s]["overall"]
        v2_ov = DINOV2_BASELINES.get(s, {}).get("overall", 0)
        print(f"  {s}-shot  Overall: DINOv3 {v3_ov:.1f}% vs DINOv2 {v2_ov:.1f}% "
              f"({'improved' if v3_ov > v2_ov else 'not improved'})")
        print(f"  {s}-shot  Bedrock: DINOv3 {v3_b:.1f}% vs DINOv2 {v2_b:.1f}% "
              f"({'improved' if v3_b > v2_b else 'not improved'})")


if __name__ == "__main__":
    main()
