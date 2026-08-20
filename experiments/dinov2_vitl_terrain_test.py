"""
Purpose: Few-shot linear probe on frozen DINOv2 ViT-L/14 features using AI4Mars dataset.
         Tests whether scaling DINOv2 to ViT-L (307M, 1024-d) exceeds DINOv3 ViT-L (92.3%)
         which is the current best non-supervised result.
         Context: DINOv2 ViT-B/14 (91.3%) > DINOv3 ViT-B (90.6%) at same scale — geological
         paper (arXiv:2407.18100) attributes this to DINOv2 data curation superiority.
Inputs:  AI4Mars train labels (few-shot sampling) + gold-standard test set (287 images)
Outputs: Per-class accuracy by shot count, CSV saved to results/
         Feature cache saved to results/feature_cache/dinov2_vitl_*
How to run:
    python3 -u experiments/dinov2_vitl_terrain_test.py | tee /tmp/dinov2_vitl_log.txt
    python3 -u experiments/dinov2_vitl_terrain_test.py --shots 1000  # 1000-shot only
    python3 -u experiments/dinov2_vitl_terrain_test.py --shots 10    # smoke test
Note: NOT gated — no HuggingFace token required.
      Expected: ~307M params, 1024-d features, ~1100ms/img (similar to DINOv3 ViT-L)
      Expected runtime: ~90 min on laptop CPU
Reference: Oquab et al. (2024) DINOv2. TMLR. arXiv:2304.07193
           Burgat et al. (2024) DINOv2 Rocks Geological Image Analysis. arXiv:2407.18100
           https://huggingface.co/facebook/dinov2-large
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

MODEL_ID            = "facebook/dinov2-large"
CLASS_NAMES         = ["soil", "bedrock", "sand", "big_rock"]
IGNORE_PIXEL        = 255
SUPERVISED_BASELINE = 96.67

# ── Baselines ─────────────────────────────────────────────────────────────────
DINOV2_VITB_BASELINES = {
    10:   {"overall": 63.1,  "bedrock": 75.0},
    50:   {"overall": 78.0,  "bedrock": 54.3},
    100:  {"overall": 82.9,  "bedrock": 60.9},
    500:  {"overall": 89.5,  "bedrock": 85.9},
    1000: {"overall": 91.3,  "bedrock": 87.0},
}
DINOV3_VITL_BASELINES = {
    10:   {"overall": 70.0,  "bedrock": None},
    50:   {"overall": 83.3,  "bedrock": None},
    100:  {"overall": 84.0,  "bedrock": None},
    500:  {"overall": 92.7,  "bedrock": None},
    1000: {"overall": 92.3,  "bedrock": 89.1},
}

DEFAULT_SHOTS = [10, 50, 100, 500, 1000]
RANDOM_SEED   = 42


# ── Model ─────────────────────────────────────────────────────────────────────

def load_model(device: str):
    print(f"Loading DINOv2 ViT-L/14 (frozen)  [{MODEL_ID}]")
    print(f"  Pretraining: DINO self-distillation (same objective as ViT-S/B)")
    print(f"  Expected: ~307M params, 1024-d features — NOT gated, no token required")
    t0 = time.perf_counter()
    processor = AutoImageProcessor.from_pretrained(MODEL_ID)
    model     = AutoModel.from_pretrained(MODEL_ID).to(device).eval()
    load_s    = time.perf_counter() - t0
    n_params  = sum(p.numel() for p in model.parameters()) / 1e6
    feat_dim  = model.config.hidden_size
    print(f"  Loaded in {load_s:.1f}s  |  Params: {n_params:.1f}M  "
          f"|  Feature dim: {feat_dim}  |  Device: {device}")
    return model, processor


# ── Feature extraction ────────────────────────────────────────────────────────

def extract_features(model, processor, image_paths, labels, device: str, desc: str = ""):
    features = []
    n  = len(image_paths)
    t0 = time.perf_counter()

    for i, path in enumerate(image_paths):
        if (i + 1) % 50 == 0 or i == n - 1:
            elapsed = time.perf_counter() - t0
            eta     = elapsed / (i + 1) * (n - i - 1)
            print(f"  {desc} {i+1}/{n}  "
                  f"({elapsed/60:.1f}min elapsed, ~{eta/60:.1f}min remaining)", flush=True)

        image  = Image.open(path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            feat    = outputs.last_hidden_state[:, 0, :]  # CLS token → [1, 1024]
            feat    = feat.cpu().numpy().squeeze()         # [1024]

        features.append(feat)

    features = np.array(features, dtype=np.float32)
    features = normalize(features)
    elapsed_total = time.perf_counter() - t0
    ms_per = elapsed_total / n * 1000
    print(f"  Done — avg {ms_per:.0f}ms/img  feat shape: {features.shape}")
    return features, np.array(labels), ms_per


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
    print("\n" + "=" * 120)
    print("DINOv2 ViT-L/14 Few-Shot Linear Probe — AI4Mars (287 test images)")
    print("Comparison: DINOv2 ViT-L (307M) vs DINOv2 ViT-B (91.3%) vs DINOv3 ViT-L (92.3%)")
    print("=" * 120)

    for metric in ["overall"] + CLASS_NAMES:
        row = f"{metric:<10}"
        for s in shots:
            vl_val  = results[s]["overall"] if metric == "overall" else results[s]["per_class"].get(metric)
            vb_val  = (DINOV2_VITB_BASELINES.get(s, {}).get("overall")
                       if metric == "overall"
                       else DINOV2_VITB_BASELINES.get(s, {}).get(metric))
            v3l_val = (DINOV3_VITL_BASELINES.get(s, {}).get("overall")
                       if metric == "overall"
                       else DINOV3_VITL_BASELINES.get(s, {}).get(metric))

            vl_str  = f"{vl_val:>6.1f}%"  if vl_val  is not None else "   N/A"
            vb_str  = f"{vb_val:>6.1f}%"  if vb_val  is not None else "   N/A"
            v3l_str = f"{v3l_val:>6.1f}%" if v3l_val is not None else "   N/A"
            d_vb    = f"{(vl_val - vb_val):>+6.1f}%" if (vl_val is not None and vb_val is not None) else "    N/A"
            row    += f"  [{s:>4}shot] v2L:{vl_str} v2B:{vb_str} Δ:{d_vb} v3L:{v3l_str}"
        print(row)

    print("=" * 120)

    if 1000 in results:
        vl_ov  = results[1000]["overall"]
        vb_ov  = DINOV2_VITB_BASELINES[1000]["overall"]
        v3l_ov = DINOV3_VITL_BASELINES[1000]["overall"]
        vl_br  = results[1000]["per_class"].get("bedrock", 0)
        vb_br  = DINOV2_VITB_BASELINES[1000]["bedrock"]
        print(f"\n>>> 1000-shot: v2B {vb_ov:.2f}% → v2L {vl_ov:.2f}%  "
              f"(Δ_v2B {vl_ov - vb_ov:+.2f}%)  |  v3L {v3l_ov:.2f}%  "
              f"(Δ_v3L {vl_ov - v3l_ov:+.2f}%)  |  Gap vs supervised: {vl_ov - SUPERVISED_BASELINE:+.2f}%")
        print(f">>> Bedrock 1000-shot: v2B {vb_br:.1f}% → v2L {vl_br:.1f}%  "
              f"(Δ {vl_br - vb_br:+.1f}pp)")


def save_csv(results: dict, avg_ms: float):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, "dinov2_vitl_terrain_few_shot.csv")
    shots = sorted(results.keys())

    fieldnames = (["shots", "overall", "gap_vs_supervised"] + CLASS_NAMES +
                  ["dinov2_vitb_overall", "dinov2_vitb_bedrock",
                   "dinov3_vitl_overall", "dinov3_vitl_bedrock",
                   "delta_vs_v2b_overall", "delta_vs_v3l_overall", "avg_ms_extract"])
    rows = []
    for s in shots:
        r   = results[s]
        v2b = DINOV2_VITB_BASELINES.get(s, {})
        v3l = DINOV3_VITL_BASELINES.get(s, {})
        vl_br = r["per_class"].get("bedrock")
        row = {
            "shots":             s,
            "overall":           f"{r['overall']:.2f}",
            "gap_vs_supervised": f"{r['overall'] - SUPERVISED_BASELINE:.2f}",
            "avg_ms_extract":    f"{avg_ms:.1f}",
        }
        for name in CLASS_NAMES:
            val = r["per_class"].get(name)
            row[name] = f"{val:.2f}" if val is not None else "N/A"
        row["dinov2_vitb_overall"]  = f"{v2b.get('overall', 0):.2f}" if v2b.get("overall") else "N/A"
        row["dinov2_vitb_bedrock"]  = f"{v2b.get('bedrock', 0):.2f}" if v2b.get("bedrock") else "N/A"
        row["dinov3_vitl_overall"]  = f"{v3l.get('overall', 0):.2f}" if v3l.get("overall") else "N/A"
        row["dinov3_vitl_bedrock"]  = f"{v3l.get('bedrock', 0):.2f}" if v3l.get("bedrock") else "N/A"
        row["delta_vs_v2b_overall"] = (f"{r['overall'] - v2b['overall']:+.2f}"
                                       if v2b.get("overall") is not None else "N/A")
        row["delta_vs_v3l_overall"] = (f"{r['overall'] - v3l['overall']:+.2f}"
                                       if v3l.get("overall") is not None else "N/A")
        rows.append(row)

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nCSV saved → {path}")
    return path


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="DINOv2 ViT-L/14 few-shot terrain probe")
    parser.add_argument("--shots", type=int, nargs="+", default=DEFAULT_SHOTS)
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    device = "cpu"
    os.makedirs(CACHE_DIR, exist_ok=True)

    cache_test_feats   = os.path.join(CACHE_DIR, "dinov2_vitl_test_287_feats.npy")
    cache_test_labels  = os.path.join(CACHE_DIR, "dinov2_vitl_test_287_labels.npy")
    cache_train_feats  = os.path.join(CACHE_DIR, "dinov2_vitl_train_1000_feats.npy")
    cache_train_labels = os.path.join(CACHE_DIR, "dinov2_vitl_train_1000_labels.npy")

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

    avg_ms = 0.0
    if use_cache:
        print("Loading features from cache...")
        test_feats_arr   = np.load(cache_test_feats)
        test_labels_arr  = np.load(cache_test_labels)
        train_feats_arr  = np.load(cache_train_feats)
        train_labels_arr = np.load(cache_train_labels)
        print(f"  test: {test_feats_arr.shape}   train: {train_feats_arr.shape}")
    else:
        model, processor = load_model(device)

        print("\nLoading test set...")
        test_pairs = load_split(TEST_LABELS)
        print(f"  Test images: {len(test_pairs)}")

        print(f"\nExtracting test features ({len(test_pairs)} images)...")
        test_feats_arr, test_labels_arr, avg_ms = extract_features(
            model, processor,
            [p for p, _ in test_pairs], [l for _, l in test_pairs],
            device, desc="test"
        )

        max_shots = max(args.shots)
        print(f"\nLoading train set (up to {max_shots}/class)...")
        train_all     = load_split(TRAIN_LABELS)
        train_sampled = sample_n_per_class(train_all, max_shots)
        by_class = {i: 0 for i in range(4)}
        for _, c in train_sampled:
            by_class[c] += 1
        print("  Train sample: " + ", ".join(f"{CLASS_NAMES[i]} {by_class[i]}" for i in range(4)))

        print(f"\nExtracting train features ({len(train_sampled)} images)...")
        est_min = len(train_sampled) * avg_ms / 1000 / 60
        print(f"  Estimated time: {est_min:.0f}min at {avg_ms:.0f}ms/img")
        train_feats_arr, train_labels_arr, _ = extract_features(
            model, processor,
            [p for p, _ in train_sampled], [l for _, l in train_sampled],
            device, desc="train"
        )

        np.save(cache_test_feats,   test_feats_arr)
        np.save(cache_test_labels,  test_labels_arr)
        np.save(cache_train_feats,  train_feats_arr)
        np.save(cache_train_labels, train_labels_arr)
        print(f"  Features cached → {CACHE_DIR}/dinov2_vitl_*")

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
        elapsed = time.perf_counter() - t0
        results[n_shots] = {"overall": overall, "per_class": per_class}
        v2b_ov = DINOV2_VITB_BASELINES.get(n_shots, {}).get("overall", 0)
        v3l_ov = DINOV3_VITL_BASELINES.get(n_shots, {}).get("overall", 0)
        print(f"overall {overall:.1f}%  (v2B {v2b_ov:.1f}% Δ{overall-v2b_ov:+.1f}%  "
              f"v3L {v3l_ov:.1f}% Δ{overall-v3l_ov:+.1f}%)  [{elapsed:.1f}s]")

    print_results(results)
    save_csv(results, avg_ms)

    print("\n=== FINAL RESULT — DINOv2 ViT-L/14 ===")
    if 1000 in results:
        vl_ov  = results[1000]["overall"]
        vb_ov  = DINOV2_VITB_BASELINES[1000]["overall"]
        v3l_ov = DINOV3_VITL_BASELINES[1000]["overall"]
        vl_br  = results[1000]["per_class"].get("bedrock", 0)
        if vl_ov > v3l_ov:
            print(f"  ✅ NEW BEST: DINOv2 ViT-L {vl_ov:.2f}% > DINOv3 ViT-L {v3l_ov:.2f}%")
        elif vl_ov > vb_ov:
            print(f"  ✅ IMPROVED vs ViT-B: {vb_ov:.2f}% → {vl_ov:.2f}%  "
                  f"(below DINOv3 ViT-L {v3l_ov:.2f}%)")
        else:
            print(f"  ❌ No improvement vs ViT-B: {vb_ov:.2f}% → {vl_ov:.2f}%")
        print(f"  Bedrock 1000-shot: {vl_br:.1f}%")
        print(f"  Gap vs supervised (96.67%): {vl_ov - SUPERVISED_BASELINE:+.2f}%")


if __name__ == "__main__":
    main()
