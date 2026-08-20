"""
Purpose: Evaluate Depth Anything V2 Small for Mars terrain geometric understanding.
         Three experiments run in sequence:
         (1) Per-class depth statistics — does monocular depth discriminate terrain class?
         (2) Depth-only few-shot classification on 40-dimensional depth feature vectors.
         (3) DINOv3 + depth fusion — does adding depth improve the best visual baseline?
Inputs:  AI4Mars dataset (gold-standard test split n=287; train split for few-shot).
         Feature cache in results/feature_cache/ avoids re-extraction on re-runs.
Outputs: depth_anything_terrain_test.csv + cached .npy feature arrays.
How to run:
    python3 -u experiments/depth_anything_terrain_test.py              # full run (10/100/1000)
    python3 -u experiments/depth_anything_terrain_test.py --shots 10   # quick smoke test
    python3 -u experiments/depth_anything_terrain_test.py --no-fusion  # skip DINOv3 fusion
    python3 -u experiments/depth_anything_terrain_test.py --no-cache   # force re-extraction
Runtime estimate: ~20-30 min first run (CPU); subsequent runs use cache (~1 min).
Reference: Yang et al. (2024) Depth Anything V2. NeurIPS 2024. arXiv:2406.09414
           https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf
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
from transformers import AutoImageProcessor, AutoModel, AutoModelForDepthEstimation

# ── Paths ─────────────────────────────────────────────────────────────────────
AI4MARS_BASE = "/mnt/c/Users/DELL/Desktop/Thesis/github source/ai4mars-dataset-merged-0.1"
IMAGES_DIR   = os.path.join(AI4MARS_BASE, "msl/images/edr")
TRAIN_LABELS = os.path.join(AI4MARS_BASE, "msl/labels/train")
TEST_LABELS  = os.path.join(AI4MARS_BASE, "msl/labels/test/masked-gold-min3-100agree")
RESULTS_DIR  = os.path.join(os.path.dirname(__file__), "results")
CACHE_DIR    = os.path.join(RESULTS_DIR, "feature_cache")

DEPTH_MODEL_ID  = "depth-anything/Depth-Anything-V2-Small-hf"
DINOV3_MODEL_ID = "facebook/dinov3-vits16-pretrain-lvd1689m"

CLASS_NAMES     = ["soil", "bedrock", "sand", "big_rock"]
IGNORE_PIXEL    = 255
SUPERVISED_BASE = 96.67
DEFAULT_SHOTS   = [10, 100, 1000]
RANDOM_SEED     = 42

# ── Baselines (from previous experiments) ─────────────────────────────────────
DINOV3_BASELINES = {
    10:   {"overall": 58.5,  "soil": 51.2,  "bedrock": 65.2,  "sand": 62.5},
    100:  {"overall": 80.5,  "soil": 95.9,  "bedrock": 60.9,  "sand": 79.2},
    1000: {"overall": 90.2,  "soil": 97.6,  "bedrock": 83.7,  "sand": 86.1},
}
DINOV2_BASELINES = {
    10:   {"overall": 56.79, "bedrock": 56.52},
    100:  {"overall": 81.18, "bedrock": 55.43},
    1000: {"overall": 89.90, "bedrock": 84.78},
}
CLIP_BASELINES = {
    10:   {"overall": 57.5,  "bedrock": 15.2},
    100:  {"overall": 76.0,  "bedrock": 62.0},
    1000: {"overall": 87.8,  "bedrock": 79.3},
}


# ── Ground truth loading ──────────────────────────────────────────────────────

def dominant_class(label_path):
    label = np.array(Image.open(label_path))
    valid = label[label != IGNORE_PIXEL]
    if len(valid) == 0:
        return None
    return int(np.argmax(np.bincount(valid, minlength=4)))


def load_split(label_dir):
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


def sample_n_per_class(pairs, n_per_class, seed=RANDOM_SEED):
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


# ── Depth feature extraction ──────────────────────────────────────────────────

def _f_statistic(groups):
    """One-way ANOVA F-statistic (pure numpy, no scipy)."""
    groups      = [g for g in groups if len(g) > 0]
    if len(groups) < 2:
        return float("nan")
    all_data    = np.concatenate(groups)
    grand_mean  = all_data.mean()
    n_total     = len(all_data)
    n_groups    = len(groups)
    ss_between  = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
    ss_within   = sum(((g - g.mean()) ** 2).sum() for g in groups)
    df_b, df_w  = n_groups - 1, n_total - n_groups
    if df_w == 0 or ss_within == 0:
        return float("nan")
    return (ss_between / df_b) / (ss_within / df_w)


DEPTH_FEAT_DIM = 40  # 8 global + 16 grid-means + 16 grid-stds


def depth_feature_vector(depth_map):
    """Extract 40-d feature vector from a raw depth map (H×W float32).

    Features: 8 global statistics + 32 spatial-grid statistics (4×4 grid, mean+std each cell).
    raw_std is computed before per-image normalisation so it captures absolute depth spread.
    All other features use depth normalised to [0,1].
    """
    d_raw = depth_map.astype(np.float32)
    raw_std = float(d_raw.std())  # pre-norm: absolute depth variation proxy

    d = (d_raw - d_raw.min()) / (d_raw.max() - d_raw.min() + 1e-8)  # normalise to [0,1]

    # Gradient (local roughness / slope)
    gy, gx = np.gradient(d)
    grad   = np.sqrt(gx ** 2 + gy ** 2)

    # Global features (8)
    feats = [
        float(d.mean()),
        float(d.std()),
        raw_std,                                              # replaces buggy post-norm range
        float(np.percentile(d, 10)),
        float(np.percentile(d, 90)),
        float(np.percentile(d, 75) - np.percentile(d, 25)),  # IQR
        float(grad.mean()),   # mean roughness
        float(grad.std()),    # heterogeneity of roughness
    ]

    # Spatial 4×4 grid: mean + std per cell (32)
    H, W   = d.shape
    n      = 4
    h_step = H // n
    w_step = W // n
    for i in range(n):
        for j in range(n):
            patch = d[i * h_step:(i + 1) * h_step, j * w_step:(j + 1) * w_step]
            feats.append(float(patch.mean()))
            feats.append(float(patch.std()))

    return np.array(feats, dtype=np.float32)


def _extract_depth_raw(depth_model, depth_proc, image_paths, labels, desc):
    features = []
    n        = len(image_paths)
    t0       = time.perf_counter()
    for i, path in enumerate(image_paths):
        if (i + 1) % 100 == 0 or i == n - 1:
            elapsed = time.perf_counter() - t0
            eta     = elapsed / (i + 1) * (n - i - 1)
            print(f"  {desc} {i+1}/{n}  ({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)")
        image  = Image.open(path).convert("RGB")
        inputs = depth_proc(images=image, return_tensors="pt")
        with torch.no_grad():
            out   = depth_model(**inputs)
            depth = out.predicted_depth.squeeze().numpy()  # (H, W)
        features.append(depth_feature_vector(depth))
    return np.array(features, dtype=np.float32), np.array(labels)


def load_or_extract_depth(depth_model, depth_proc, image_paths, labels,
                          cache_key, desc, use_cache=True):
    os.makedirs(CACHE_DIR, exist_ok=True)
    feat_path  = os.path.join(CACHE_DIR, f"depth_{cache_key}_feats.npy")
    label_path = os.path.join(CACHE_DIR, f"depth_{cache_key}_labels.npy")

    if use_cache and os.path.exists(feat_path) and os.path.exists(label_path):
        print(f"  Cache hit — loading {desc} depth features from {feat_path}")
        return np.load(feat_path), np.load(label_path)

    feats, labs = _extract_depth_raw(depth_model, depth_proc, image_paths, labels, desc)
    np.save(feat_path, feats)
    np.save(label_path, labs)
    print(f"  Cached → {feat_path}")
    return feats, labs


# ── DINOv3 feature extraction (for fusion) ────────────────────────────────────

def _extract_dinov3_raw(image_paths, labels, desc):
    print(f"  Loading DINOv3 ViT-S/16 for {desc} feature extraction...")
    proc  = AutoImageProcessor.from_pretrained(DINOV3_MODEL_ID, token=True)
    model = AutoModel.from_pretrained(DINOV3_MODEL_ID, token=True).eval()

    features = []
    n        = len(image_paths)
    t0       = time.perf_counter()
    for i, path in enumerate(image_paths):
        if (i + 1) % 100 == 0 or i == n - 1:
            elapsed = time.perf_counter() - t0
            eta     = elapsed / (i + 1) * (n - i - 1)
            print(f"  {desc} {i+1}/{n}  ({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)")
        image  = Image.open(path).convert("RGB")
        inputs = proc(images=image, return_tensors="pt")
        with torch.no_grad():
            out  = model(**inputs)
            feat = out.last_hidden_state[:, 0, :].cpu().numpy().squeeze()
        features.append(feat)

    del model  # free ~0.3 GB RAM
    return np.array(features, dtype=np.float32), np.array(labels)


def load_or_extract_dinov3(image_paths, labels, cache_key, desc, use_cache=True):
    os.makedirs(CACHE_DIR, exist_ok=True)
    feat_path  = os.path.join(CACHE_DIR, f"dinov3_{cache_key}_feats.npy")
    label_path = os.path.join(CACHE_DIR, f"dinov3_{cache_key}_labels.npy")

    if use_cache and os.path.exists(feat_path) and os.path.exists(label_path):
        print(f"  Cache hit — loading {desc} DINOv3 features from {feat_path}")
        return np.load(feat_path), np.load(label_path)

    feats, labs = _extract_dinov3_raw(image_paths, labels, desc)
    np.save(feat_path, feats)
    np.save(label_path, labs)
    print(f"  Cached → {feat_path}")
    return feats, labs


# ── Per-class statistics ──────────────────────────────────────────────────────

GLOBAL_FEAT_NAMES = [
    "depth_mean", "depth_std", "raw_depth_std",
    "depth_p10",  "depth_p90", "depth_iqr",
    "grad_mean",  "grad_std",
]
KEY_FEATS = {"depth_std", "grad_mean", "depth_iqr"}  # primary roughness proxies


def print_class_statistics(depth_feats, labels):
    print("\n" + "=" * 84)
    print("Per-class depth statistics — AI4Mars test set (n=287)")
    print("Hypothesis: bedrock > soil/sand for roughness features (depth_std, grad_mean, iqr)")
    print("=" * 84)
    header = f"  {'Feature':<18}  {'soil':>8}  {'bedrock':>8}  {'sand':>8}  {'big_rock':>8}  {'F-score':>8}"
    print(header)
    print("-" * 84)

    for fi, name in enumerate(GLOBAL_FEAT_NAMES):
        vals = [depth_feats[labels == c, fi] for c in range(4)]
        means = [v.mean() if len(v) > 0 else float("nan") for v in vals]
        f = _f_statistic(vals)
        flag = " ←" if name in KEY_FEATS else ""
        print(f"  {name:<18}  {means[0]:>8.4f}  {means[1]:>8.4f}  "
              f"{means[2]:>8.4f}  {means[3]:>8.4f}  {f:>8.2f}{flag}")

    print("=" * 84)
    print("(← = primary roughness/geometry features; F-score = class separability)")
    print()

    # Hypothesis check: bedrock roughness vs soil/sand
    soil_std    = depth_feats[labels == 0, 1].mean()   # depth_std index
    bedrock_std = depth_feats[labels == 1, 1].mean()
    sand_std    = depth_feats[labels == 2, 1].mean()
    soil_grad   = depth_feats[labels == 0, 6].mean()   # grad_mean index
    bedrock_grad= depth_feats[labels == 1, 6].mean()
    sand_grad   = depth_feats[labels == 2, 6].mean()

    print("Hypothesis check (bedrock > soil/sand):")
    print(f"  depth_std  — soil {soil_std:.4f}  |  bedrock {bedrock_std:.4f}  |  sand {sand_std:.4f}")
    print(f"               bedrock > soil: {'✓' if bedrock_std > soil_std else '✗'}  |  "
          f"bedrock > sand: {'✓' if bedrock_std > sand_std else '✗'}")
    print(f"  grad_mean  — soil {soil_grad:.4f}  |  bedrock {bedrock_grad:.4f}  |  sand {sand_grad:.4f}")
    print(f"               bedrock > soil: {'✓' if bedrock_grad > soil_grad else '✗'}  |  "
          f"bedrock > sand: {'✓' if bedrock_grad > sand_grad else '✗'}")


# ── Few-shot classification ───────────────────────────────────────────────────

def run_few_shot(train_feats, train_labels, test_feats, test_labels, n_shots):
    rng         = np.random.RandomState(RANDOM_SEED)
    sampled_idx = []
    for c in range(len(CLASS_NAMES)):
        idx    = np.where(train_labels == c)[0]
        chosen = rng.choice(idx, size=min(n_shots, len(idx)), replace=False)
        sampled_idx.extend(chosen.tolist())

    clf = LogisticRegression(
        C=0.316, max_iter=2000, random_state=RANDOM_SEED,
        multi_class="multinomial", solver="lbfgs"
    )
    clf.fit(train_feats[sampled_idx], train_labels[sampled_idx])
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
    overall = sum(correct.values()) / n_total * 100 if n_total > 0 else 0.0
    return overall, per_class


# ── CSV output ────────────────────────────────────────────────────────────────

def save_csv(depth_results, fusion_results, shots, ms_per_img_depth):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, "depth_anything_terrain_test.csv")

    fieldnames = (
        ["shots",
         "depth_only_overall"] + [f"depth_only_{c}" for c in CLASS_NAMES] +
        ["fusion_overall"]     + [f"fusion_{c}"      for c in CLASS_NAMES] +
        ["dinov3_only_overall", "dinov3_only_bedrock",
         "delta_overall_vs_dinov3_depth_only",
         "delta_overall_vs_dinov3_fusion",
         "delta_bedrock_vs_dinov3_fusion",
         "ms_per_img_depth"]
    )

    rows = []
    for s in shots:
        do  = depth_results.get(s, {})
        fu  = fusion_results.get(s, {})
        d3  = DINOV3_BASELINES.get(s, {})
        row = {
            "shots":                             s,
            "depth_only_overall":                f"{do.get('overall', float('nan')):.2f}",
            "fusion_overall":                    f"{fu.get('overall', float('nan')):.2f}",
            "dinov3_only_overall":               f"{d3.get('overall', float('nan')):.2f}",
            "dinov3_only_bedrock":               f"{d3.get('bedrock', float('nan')):.2f}",
            "ms_per_img_depth":                  f"{ms_per_img_depth:.1f}",
        }
        do_ov = do.get("overall", float("nan"))
        fu_ov = fu.get("overall", float("nan"))
        d3_ov = d3.get("overall", float("nan"))
        fu_br = (fu.get("per_class") or {}).get("bedrock") or float("nan")
        d3_br = d3.get("bedrock", float("nan"))
        row["delta_overall_vs_dinov3_depth_only"] = (
            f"{do_ov - d3_ov:+.2f}" if not np.isnan(do_ov) else "N/A")
        row["delta_overall_vs_dinov3_fusion"] = (
            f"{fu_ov - d3_ov:+.2f}" if not (np.isnan(fu_ov) or np.isnan(d3_ov)) else "N/A")
        row["delta_bedrock_vs_dinov3_fusion"] = (
            f"{fu_br - d3_br:+.2f}" if not (np.isnan(fu_br) or np.isnan(d3_br)) else "N/A")

        for c in CLASS_NAMES:
            v = (do.get("per_class") or {}).get(c)
            row[f"depth_only_{c}"] = f"{v:.2f}" if v is not None else "N/A"
            v = (fu.get("per_class") or {}).get(c)
            row[f"fusion_{c}"] = f"{v:.2f}" if v is not None else "N/A"
        rows.append(row)

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"CSV saved → {path}")
    return path


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Depth Anything V2 terrain analysis")
    parser.add_argument("--shots", type=int, nargs="+", default=DEFAULT_SHOTS,
                        help="Shot counts (default: 10 100 1000)")
    parser.add_argument("--no-fusion", action="store_true",
                        help="Skip DINOv3 + depth fusion (saves ~10 min)")
    parser.add_argument("--no-cache", action="store_true",
                        help="Force re-extraction even if cache exists")
    args = parser.parse_args()
    use_cache = not args.no_cache

    # ── Load Depth Anything V2 ────────────────────────────────────────────────
    print(f"\nLoading Depth Anything V2 Small  [{DEPTH_MODEL_ID}]...")
    t0          = time.perf_counter()
    depth_proc  = AutoImageProcessor.from_pretrained(DEPTH_MODEL_ID)
    depth_model = AutoModelForDepthEstimation.from_pretrained(DEPTH_MODEL_ID).eval()
    n_params    = sum(p.numel() for p in depth_model.parameters()) / 1e6
    print(f"  Loaded in {time.perf_counter()-t0:.1f}s  |  Params: {n_params:.1f}M  |  "
          f"Depth feature dim: {DEPTH_FEAT_DIM}")

    # ── Dataset ───────────────────────────────────────────────────────────────
    print("\nLoading test set...")
    test_pairs  = load_split(TEST_LABELS)
    test_paths  = [p for p, _ in test_pairs]
    test_labs   = [l for _, l in test_pairs]
    print(f"  Test images: {len(test_pairs)}")

    max_shots = max(args.shots)
    print(f"\nLoading train set (up to {max_shots}/class)...")
    train_all     = load_split(TRAIN_LABELS)
    train_sampled = sample_n_per_class(train_all, max_shots)
    train_paths   = [p for p, _ in train_sampled]
    train_labs    = [l for _, l in train_sampled]
    by_class = {i: 0 for i in range(4)}
    for _, c in train_sampled:
        by_class[c] += 1
    print("  Train: " + ", ".join(f"{CLASS_NAMES[i]} {by_class[i]}" for i in range(4)))

    # ── Extract depth features ────────────────────────────────────────────────
    print(f"\nExtracting test depth features ({len(test_pairs)} images)...")
    t_test = time.perf_counter()
    test_depth, test_labels_arr = load_or_extract_depth(
        depth_model, depth_proc, test_paths, test_labs,
        f"test_{len(test_pairs)}", "test", use_cache
    )
    ms_per_img = (time.perf_counter() - t_test) / len(test_pairs) * 1000
    print(f"  avg {ms_per_img:.0f} ms/img  |  feature shape: {test_depth.shape}")

    print(f"\nExtracting train depth features ({len(train_sampled)} images)...")
    train_depth, train_labels_arr = load_or_extract_depth(
        depth_model, depth_proc, train_paths, train_labs,
        f"train_{max_shots}", "train", use_cache
    )
    del depth_model  # free RAM after extraction

    # ── Per-class statistics (Experiment 1) ───────────────────────────────────
    print_class_statistics(test_depth, test_labels_arr)

    # ── Depth-only few-shot (Experiment 2) ────────────────────────────────────
    print("=" * 84)
    print("Experiment 2: Depth-only few-shot classification")
    print("=" * 84)
    depth_results = {}
    test_depth_n  = normalize(test_depth)
    train_depth_n = normalize(train_depth)
    for s in sorted(args.shots):
        print(f"  {s}-shot...", end=" ", flush=True)
        t0      = time.perf_counter()
        ov, pc  = run_few_shot(train_depth_n, train_labels_arr,
                               test_depth_n,  test_labels_arr, s)
        elapsed = time.perf_counter() - t0
        depth_results[s] = {"overall": ov, "per_class": pc}
        br = pc.get("bedrock", 0) or 0
        d3_ov = DINOV3_BASELINES.get(s, {}).get("overall", 0)
        print(f"overall {ov:.1f}%  bedrock {br:.1f}%  "
              f"(vs DINOv3 {d3_ov:.1f}%  Δ{ov-d3_ov:+.1f}%)  [{elapsed:.1f}s]")

    # ── DINOv3 + depth fusion (Experiment 3) ──────────────────────────────────
    fusion_results = {}
    if not args.no_fusion:
        print(f"\n{'='*84}")
        print("Experiment 3: DINOv3 + depth fusion")
        print(f"{'='*84}")
        print(f"  Loading DINOv3 features  [{DINOV3_MODEL_ID}]...")

        test_d3, _  = load_or_extract_dinov3(
            test_paths, test_labs, f"test_{len(test_pairs)}", "test", use_cache)
        train_d3, _ = load_or_extract_dinov3(
            train_paths, train_labs, f"train_{max_shots}", "train", use_cache)

        # Concat: [L2-norm DINOv3 (384-d) | L2-norm depth (40-d)] → L2-norm again
        test_fuse  = normalize(np.concatenate([normalize(test_d3),  normalize(test_depth)],  axis=1))
        train_fuse = normalize(np.concatenate([normalize(train_d3), normalize(train_depth)], axis=1))
        print(f"  Fusion dim: {test_fuse.shape[1]} "
              f"(DINOv3 {test_d3.shape[1]}-d + depth {test_depth.shape[1]}-d, L2-norm)")

        for s in sorted(args.shots):
            print(f"  {s}-shot...", end=" ", flush=True)
            t0     = time.perf_counter()
            ov, pc = run_few_shot(train_fuse, train_labels_arr,
                                  test_fuse,  test_labels_arr, s)
            elapsed = time.perf_counter() - t0
            fusion_results[s] = {"overall": ov, "per_class": pc}
            br    = pc.get("bedrock", 0) or 0
            d3_ov = DINOV3_BASELINES.get(s, {}).get("overall", 0)
            print(f"overall {ov:.1f}%  bedrock {br:.1f}%  "
                  f"(DINOv3 alone {d3_ov:.1f}%  Δ{ov-d3_ov:+.1f}%)  [{elapsed:.1f}s]")

    # ── Final summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 84)
    print("SUMMARY — Depth Anything V2 Small  (AI4Mars n=287)")
    print(f"  Depth feature dim: {DEPTH_FEAT_DIM}  |  Depth extract: {ms_per_img:.0f} ms/img")
    print("=" * 84)
    print(f"  {'Shots':<8} {'Depth-only':>12} {'DINOv3':>10} {'Fusion':>10} "
          f"{'Δ fuse-v-d3':>12}  Bedrock: {'Depth':>8} {'DINOv3':>8} {'Fusion':>8}")
    for s in sorted(args.shots):
        do_ov = depth_results.get(s, {}).get("overall", float("nan"))
        fu_ov = fusion_results.get(s, {}).get("overall", float("nan")) if fusion_results else float("nan")
        d3_ov = DINOV3_BASELINES.get(s, {}).get("overall", float("nan"))
        do_br = (depth_results.get(s, {}).get("per_class") or {}).get("bedrock", float("nan")) or float("nan")
        fu_br = (fusion_results.get(s, {}).get("per_class") or {}).get("bedrock", float("nan")) if fusion_results else float("nan")
        d3_br = DINOV3_BASELINES.get(s, {}).get("bedrock", float("nan"))
        delta = fu_ov - d3_ov if not (np.isnan(fu_ov) or np.isnan(d3_ov)) else float("nan")
        fu_str    = f"{fu_ov:>8.1f}%" if not np.isnan(fu_ov) else "       N/A"
        delta_str = f"{delta:>+.1f}%"  if not np.isnan(delta) else "       N/A"
        fu_br_str = f"{fu_br:>6.1f}%"  if not np.isnan(fu_br) else "     N/A"
        print(f"  {s:<8} {do_ov:>10.1f}%  {d3_ov:>8.1f}%  {fu_str}  "
              f"{delta_str:>11}   {do_br:>7.1f}%  {d3_br:>6.1f}%  {fu_br_str}")
    print("=" * 84)

    save_csv(depth_results, fusion_results, sorted(args.shots), ms_per_img)

    print("\nKey thesis findings:")
    for s in sorted(args.shots):
        if s not in fusion_results:
            continue
        fu_ov = fusion_results[s]["overall"]
        d3_ov = DINOV3_BASELINES.get(s, {}).get("overall", 0)
        improved = fu_ov > d3_ov
        print(f"  {s}-shot fusion: {fu_ov:.1f}% vs DINOv3 {d3_ov:.1f}% "
              f"({'depth helps ✓' if improved else 'depth does not help ✗'})")


if __name__ == "__main__":
    main()
