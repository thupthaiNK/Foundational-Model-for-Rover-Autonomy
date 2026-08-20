"""
Purpose: Few-shot linear probe on frozen DINOv3 ViT-L/16 features using AI4Mars dataset.
         Tests whether scaling the DINOv3 backbone from ViT-S (21.6M, 384-d) to
         ViT-L (~307M, 1024-d) closes the gap to supervised (96.67%) further than
         the ViT-S baseline of 90.2% at 1000-shot.
Inputs:  AI4Mars train labels (few-shot sampling) + gold-standard test set (287 images)
Outputs: Per-class accuracy by shot count, CSV saved to results/
         Feature cache saved to results/feature_cache/dinov3_vitl_*
How to run:
    python3 -u experiments/dinov3_vitl_terrain_test.py | tee /tmp/dinov3_vitl_log.txt
    python3 -u experiments/dinov3_vitl_terrain_test.py --shots 1000  # 1000-shot only
Note: Requires HuggingFace token (gated model). Access approved with ViT-S — same gating.
      OVERNIGHT RUN: ~12–17 hours on CPU (ViT-L ~2–4s/img for 3108 train + 287 test images)
Reference: Siméoni et al. (2025) DINOv3. Meta AI. arXiv:2508.10104
           https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m
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
CACHE_DIR    = os.path.join(RESULTS_DIR, "feature_cache")

MODEL_ID            = "facebook/dinov3-vitl16-pretrain-lvd1689m"
CLASS_NAMES         = ["soil", "bedrock", "sand", "big_rock"]
IGNORE_PIXEL        = 255
SUPERVISED_BASELINE = 96.67

# ── Baselines for comparison ──────────────────────────────────────────────────
DINOV3_VITS_BASELINES = {
    10:   {"overall": 58.5,  "soil": 51.2,  "bedrock": 65.2, "sand": 62.5},
    50:   {"overall": 75.3,  "soil": None,  "bedrock": 51.1, "sand": None},
    100:  {"overall": 80.5,  "soil": None,  "bedrock": 60.9, "sand": None},
    500:  {"overall": 89.5,  "soil": None,  "bedrock": 80.4, "sand": None},
    1000: {"overall": 90.2,  "soil": 97.6,  "bedrock": 83.7, "sand": 86.1},
}

DINOV2_BASELINES = {
    10:   {"overall": 56.79, "bedrock": 56.52},
    100:  {"overall": 81.18, "bedrock": 55.43},
    1000: {"overall": 89.90, "bedrock": 84.78},
}

DEFAULT_SHOTS = [10, 50, 100, 500, 1000]
RANDOM_SEED   = 42


# ── Model ─────────────────────────────────────────────────────────────────────

def load_model(device: str):
    print(f"Loading DINOv3 ViT-L/16 (frozen)  [{MODEL_ID}]")
    print(f"  Expected: ~307M params, 1024-d features — OVERNIGHT RUN")
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
    """Extract L2-normalised DINOv3 ViT-L CLS-token features."""
    features = []
    n  = len(image_paths)
    t0 = time.perf_counter()

    for i, path in enumerate(image_paths):
        if (i + 1) % 50 == 0 or i == n - 1:
            elapsed = time.perf_counter() - t0
            eta     = elapsed / (i + 1) * (n - i - 1)
            eta_h   = eta / 3600
            print(f"  {desc} {i+1}/{n}  "
                  f"({elapsed/60:.1f}min elapsed, ~{eta_h:.1f}h remaining)", flush=True)

        image  = Image.open(path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            feat    = outputs.last_hidden_state[:, 0, :]  # CLS token → [1, 1024]
            feat    = feat.cpu().numpy().squeeze()         # [1024]

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
    print("DINOv3 ViT-L/16 Few-Shot Linear Probe — AI4Mars (287 test images)")
    print("Comparison: ViT-L (1024-d) vs ViT-S baseline (384-d, 90.2%)")
    print("=" * 100)

    for metric in ["overall"] + CLASS_NAMES:
        row = f"{metric:<10}"
        for s in shots:
            vl_val = results[s]["overall"] if metric == "overall" else results[s]["per_class"].get(metric)
            vs_val = (DINOV3_VITS_BASELINES.get(s, {}).get("overall")
                      if metric == "overall"
                      else DINOV3_VITS_BASELINES.get(s, {}).get(metric))

            vl_str = f"{vl_val:>6.1f}%" if vl_val is not None else "   N/A"
            vs_str = f"{vs_val:>6.1f}%" if vs_val is not None else "   N/A"
            delta  = f"{(vl_val - vs_val):>+6.1f}%" if (vl_val is not None and vs_val is not None) else "    N/A"
            row   += f"  [{s:>4}shot] ViT-L:{vl_str} ViT-S:{vs_str} Δ:{delta}"
        print(row)

    print("=" * 100)
    print("\nKey metric — Bedrock (main thesis contribution):")
    for s in shots:
        vl_b = results[s]["per_class"].get("bedrock")
        vs_b = DINOV3_VITS_BASELINES.get(s, {}).get("bedrock")
        if vl_b is not None and vs_b is not None:
            print(f"  {s:>5}-shot:  ViT-L {vl_b:.1f}%  |  ViT-S {vs_b:.1f}%  "
                  f"(Δ {vl_b - vs_b:+.1f}%)")

    if 1000 in results:
        vl_ov = results[1000]["overall"]
        vs_ov = DINOV3_VITS_BASELINES[1000]["overall"]
        print(f"\n>>> 1000-shot Overall: ViT-L {vl_ov:.2f}%  |  ViT-S {vs_ov:.2f}%  "
              f"(Δ {vl_ov - vs_ov:+.2f}%)  |  Gap vs supervised: {vl_ov - SUPERVISED_BASELINE:+.2f}%")


def save_csv(results: dict, avg_ms_extract: float):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, "dinov3_vitl_terrain_few_shot.csv")
    shots = sorted(results.keys())

    fieldnames = (["shots", "overall", "gap_vs_supervised"] + CLASS_NAMES +
                  ["vits_overall", "vits_bedrock", "delta_vs_vits_overall",
                   "delta_vs_vits_bedrock", "avg_ms_extract"])

    rows = []
    for s in shots:
        r  = results[s]
        vs = DINOV3_VITS_BASELINES.get(s, {})
        row = {
            "shots":             s,
            "overall":           f"{r['overall']:.2f}",
            "gap_vs_supervised": f"{r['overall'] - SUPERVISED_BASELINE:.2f}",
            "avg_ms_extract":    f"{avg_ms_extract:.1f}",
        }
        for name in CLASS_NAMES:
            val      = r["per_class"].get(name)
            row[name] = f"{val:.2f}" if val is not None else "N/A"

        vs_ov = vs.get("overall")
        vs_br = vs.get("bedrock")
        vl_br = r["per_class"].get("bedrock")
        row["vits_overall"]          = f"{vs_ov:.2f}" if vs_ov is not None else "N/A"
        row["vits_bedrock"]          = f"{vs_br:.2f}" if vs_br is not None else "N/A"
        row["delta_vs_vits_overall"] = (f"{r['overall'] - vs_ov:+.2f}"
                                        if vs_ov is not None else "N/A")
        row["delta_vs_vits_bedrock"] = (f"{(vl_br or 0) - vs_br:+.2f}"
                                        if (vl_br is not None and vs_br is not None) else "N/A")
        rows.append(row)

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nCSV saved → {path}")
    return path


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="DINOv3 ViT-L/16 few-shot terrain probe (overnight)")
    parser.add_argument("--shots", type=int, nargs="+", default=DEFAULT_SHOTS)
    parser.add_argument("--no-cache", action="store_true",
                        help="Re-extract features even if cache exists")
    args = parser.parse_args()

    device = "cpu"

    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_test_feats  = os.path.join(CACHE_DIR, "dinov3_vitl_test_287_feats.npy")
    cache_test_labels = os.path.join(CACHE_DIR, "dinov3_vitl_test_287_labels.npy")
    cache_train_feats = os.path.join(CACHE_DIR, "dinov3_vitl_train_1000_feats.npy")
    cache_train_labels= os.path.join(CACHE_DIR, "dinov3_vitl_train_1000_labels.npy")

    max_shots_needed = max(args.shots)
    train_cache_ok = (
        os.path.exists(cache_train_feats) and
        np.load(cache_train_feats, mmap_mode="r").shape[0] >= max_shots_needed * len(CLASS_NAMES)
    )
    use_cache = (not args.no_cache and
                 os.path.exists(cache_test_feats) and
                 os.path.exists(cache_test_labels) and
                 train_cache_ok and
                 os.path.exists(cache_train_labels))

    if use_cache:
        print("Loading features from cache...")
        test_feats_arr   = np.load(cache_test_feats)
        test_labels_arr  = np.load(cache_test_labels)
        train_feats_arr  = np.load(cache_train_feats)
        train_labels_arr = np.load(cache_train_labels)
        print(f"  test: {test_feats_arr.shape}   train: {train_feats_arr.shape}")
        ms_per_img = 0.0
    else:
        model, processor = load_model(device)

        print("\nLoading test set...")
        test_pairs = load_split(TEST_LABELS)
        print(f"  Test images: {len(test_pairs)}")

        print(f"\nExtracting test features ({len(test_pairs)} images)...")
        t_test = time.perf_counter()
        test_feats_arr, test_labels_arr = extract_features(
            model, processor,
            [p for p, _ in test_pairs], [l for _, l in test_pairs],
            device, desc="test"
        )
        ms_per_img = (time.perf_counter() - t_test) / len(test_pairs) * 1000
        print(f"  Done — avg {ms_per_img:.0f}ms/img  feat shape: {test_feats_arr.shape}")

        max_shots = max(args.shots)
        print(f"\nLoading train set (up to {max_shots}/class)...")
        train_all     = load_split(TRAIN_LABELS)
        train_sampled = sample_n_per_class(train_all, max_shots)

        by_class = {i: 0 for i in range(4)}
        for _, c in train_sampled:
            by_class[c] += 1
        print("  Train sample: " + ", ".join(f"{CLASS_NAMES[i]} {by_class[i]}" for i in range(4)))

        print(f"\nExtracting train features ({len(train_sampled)} images)...")
        print(f"  Estimated time: {len(train_sampled) * ms_per_img / 1000 / 3600:.1f}h at {ms_per_img:.0f}ms/img")
        train_feats_arr, train_labels_arr = extract_features(
            model, processor,
            [p for p, _ in train_sampled], [l for _, l in train_sampled],
            device, desc="train"
        )

        np.save(cache_test_feats,   test_feats_arr)
        np.save(cache_test_labels,  test_labels_arr)
        np.save(cache_train_feats,  train_feats_arr)
        np.save(cache_train_labels, train_labels_arr)
        print(f"  Features cached → {CACHE_DIR}/dinov3_vitl_*")

    print("\nRunning few-shot linear probes...")
    results = {}
    for n_shots in sorted(args.shots):
        print(f"  {n_shots}-shot...", end=" ", flush=True)
        t0 = time.perf_counter()
        overall, per_class = run_few_shot(
            train_feats_arr, train_labels_arr,
            test_feats_arr,  test_labels_arr,
            n_shots
        )
        elapsed  = time.perf_counter() - t0
        results[n_shots] = {"overall": overall, "per_class": per_class}
        vs_overall = DINOV3_VITS_BASELINES.get(n_shots, {}).get("overall", 0)
        delta      = overall - vs_overall
        print(f"overall {overall:.1f}%  (ViT-S {vs_overall:.1f}%  Δ{delta:+.1f}%)  [{elapsed:.1f}s]")

    print_results(results)
    save_csv(results, ms_per_img)

    print("\n=== FINAL RESULT — D4 DINOv3 ViT-L/16 ===")
    if 1000 in results:
        vl_ov = results[1000]["overall"]
        vs_ov = DINOV3_VITS_BASELINES[1000]["overall"]
        vl_br = results[1000]["per_class"].get("bedrock", 0)
        vs_br = DINOV3_VITS_BASELINES[1000].get("bedrock", 0)
        status = "✅ IMPROVED" if vl_ov > vs_ov else "❌ NO IMPROVEMENT"
        print(f"  1000-shot Overall: {vl_ov:.2f}%  (ViT-S: {vs_ov:.2f}%  Δ{vl_ov-vs_ov:+.2f}%)  → {status}")
        print(f"  1000-shot Bedrock: {vl_br:.2f}%  (ViT-S: {vs_br:.2f}%  Δ{vl_br-vs_br:+.2f}%)")
        print(f"  Gap vs supervised (96.67%): {vl_ov - SUPERVISED_BASELINE:+.2f}%")


if __name__ == "__main__":
    main()
