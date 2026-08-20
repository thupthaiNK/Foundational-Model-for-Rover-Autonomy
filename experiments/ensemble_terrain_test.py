"""
Purpose: Ensemble few-shot linear probe using concatenated feature caches from multiple frozen encoders.
         No re-running any model — all features are loaded from existing .npy cache files.
         Tests three combinations: (A) DINOv2 ViT-L + DINOv3 ViT-L, (B) DINOv2 ViT-L + DINOv2 ViT-B,
         (C) All three. Baseline comparison: best single model DINOv2 ViT-L (93.73%).
Inputs: experiments/results/feature_cache/dinov2_vitl_*.npy (1024-d)
        experiments/results/feature_cache/dinov3_vitl_*.npy (1024-d)
        experiments/results/feature_cache/dinov2_vitb_*.npy (768-d)   [if available]
        experiments/results/feature_cache/dinov3_vitb_*.npy (768-d)   [if available]
Outputs: experiments/results/ensemble_terrain_few_shot.csv
How to run: python experiments/ensemble_terrain_test.py
Project: Onboard Visual Foundation Models for Mars Terrain Perception and Rover Navigation
"""

import os
import csv
import time
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize

# ── paths ──────────────────────────────────────────────────────────────────────
CACHE_DIR = os.path.join(os.path.dirname(__file__), "results", "feature_cache")
OUT_CSV   = os.path.join(os.path.dirname(__file__), "results", "ensemble_terrain_few_shot.csv")

CACHE_FILES = {
    "dinov2_vitl": {
        "train": os.path.join(CACHE_DIR, "dinov2_vitl_train_1000_feats.npy"),
        "train_labels": os.path.join(CACHE_DIR, "dinov2_vitl_train_1000_labels.npy"),
        "test":  os.path.join(CACHE_DIR, "dinov2_vitl_test_287_feats.npy"),
        "test_labels":  os.path.join(CACHE_DIR, "dinov2_vitl_test_287_labels.npy"),
        "dim": 1024,
    },
    "dinov3_vitl": {
        "train": os.path.join(CACHE_DIR, "dinov3_vitl_train_1000_feats.npy"),
        "train_labels": os.path.join(CACHE_DIR, "dinov3_vitl_train_1000_labels.npy"),
        "test":  os.path.join(CACHE_DIR, "dinov3_vitl_test_287_feats.npy"),
        "test_labels":  os.path.join(CACHE_DIR, "dinov3_vitl_test_287_labels.npy"),
        "dim": 1024,
    },
    "dinov2_vitb": {
        "train": os.path.join(CACHE_DIR, "dinov2_vitb_train_1000_feats.npy"),
        "train_labels": os.path.join(CACHE_DIR, "dinov2_vitb_train_1000_labels.npy"),
        "test":  os.path.join(CACHE_DIR, "dinov2_vitb_test_287_feats.npy"),
        "test_labels":  os.path.join(CACHE_DIR, "dinov2_vitb_test_287_labels.npy"),
        "dim": 768,
    },
    "dinov3_vitb": {
        "train": os.path.join(CACHE_DIR, "dinov3_vitb_train_1000_feats.npy"),
        "train_labels": os.path.join(CACHE_DIR, "dinov3_vitb_train_1000_labels.npy"),
        "test":  os.path.join(CACHE_DIR, "dinov3_vitb_test_287_feats.npy"),
        "test_labels":  os.path.join(CACHE_DIR, "dinov3_vitb_test_287_labels.npy"),
        "dim": 768,
    },
}

SHOT_COUNTS = [10, 50, 100, 500, 1000]
SUPERVISED_BASELINE = 96.67

CLASS_NAMES = ["soil", "bedrock", "sand", "big_rock"]
# Label mapping (from dinov2_vitl cache): 0=soil, 1=bedrock, 2=sand
LABEL_MAP = {0: "soil", 1: "bedrock", 2: "sand"}


def load_cache(key):
    paths = CACHE_FILES[key]
    if not os.path.exists(paths["train"]):
        print(f"  ⚠️  Cache not found for {key}: {paths['train']}")
        return None
    train_feats  = np.load(paths["train"])
    train_labels = np.load(paths["train_labels"])
    test_feats   = np.load(paths["test"])
    test_labels  = np.load(paths["test_labels"])
    print(f"  Loaded {key}: train {train_feats.shape}, test {test_feats.shape}")
    return {
        "train_feats":  train_feats,
        "train_labels": train_labels,
        "test_feats":   test_feats,
        "test_labels":  test_labels,
    }


def concat_and_normalize(feat_arrays):
    """Concatenate feature arrays along dim=1, then L2-normalise each row."""
    combined = np.concatenate(feat_arrays, axis=1)
    return normalize(combined, norm="l2")


def run_few_shot(train_f, train_l, test_f, test_l, shots, combo_name):
    classes = np.unique(train_l)
    # sample shots per class
    idx = []
    for c in classes:
        c_idx = np.where(train_l == c)[0]
        chosen = c_idx[:shots] if len(c_idx) >= shots else c_idx
        idx.extend(chosen.tolist())
    idx = np.array(idx)
    X_train, y_train = train_f[idx], train_l[idx]

    clf = LogisticRegression(
        C=0.316, max_iter=1000, multi_class="multinomial", solver="lbfgs",
        random_state=42
    )
    t0 = time.time()
    clf.fit(X_train, y_train)
    preds = clf.predict(test_f)
    elapsed = time.time() - t0

    overall = (preds == test_l).mean() * 100
    per_class = {}
    for label_int, cn in LABEL_MAP.items():
        mask = test_l == label_int
        if mask.sum() == 0:
            per_class[cn] = float("nan")
        else:
            per_class[cn] = (preds[mask] == test_l[mask]).mean() * 100
    for cn in CLASS_NAMES:
        if cn not in per_class:
            per_class[cn] = float("nan")

    return overall, per_class, elapsed


def run_ensemble(name, keys, caches, label):
    """Run few-shot evaluation for a named ensemble combination."""
    print(f"\n{'='*70}")
    print(f"  Ensemble: {name}")
    dims = sum(CACHE_FILES[k]["dim"] for k in keys)
    print(f"  Keys: {keys}  →  concat dim: {dims}-d")
    print(f"{'='*70}")

    train_feats_list = [caches[k]["train_feats"]  for k in keys]
    test_feats_list  = [caches[k]["test_feats"]   for k in keys]
    # Labels are the same for all caches (same dataset ordering)
    train_labels = caches[keys[0]]["train_labels"]
    test_labels  = caches[keys[0]]["test_labels"]

    train_f = concat_and_normalize(train_feats_list)
    test_f  = concat_and_normalize(test_feats_list)

    rows = []
    for shots in SHOT_COUNTS:
        overall, per_class, elapsed = run_few_shot(
            train_f, train_labels, test_f, test_labels, shots, name
        )
        gap = overall - SUPERVISED_BASELINE
        soil_str    = f"{per_class['soil']:.1f}%" if not np.isnan(per_class["soil"]) else "N/A"
        bedrock_str = f"{per_class['bedrock']:.1f}%" if not np.isnan(per_class["bedrock"]) else "N/A"
        sand_str    = f"{per_class['sand']:.1f}%" if not np.isnan(per_class["sand"]) else "N/A"
        print(f"  {shots:4d}-shot  overall {overall:.1f}%  "
              f"soil {soil_str}  bedrock {bedrock_str}  sand {sand_str}  "
              f"gap {gap:+.2f}%  [{elapsed:.1f}s]")
        rows.append({
            "ensemble":  name,
            "keys":      "+".join(keys),
            "dim":       dims,
            "shots":     shots,
            "overall":   round(overall, 2),
            "soil":      round(per_class["soil"], 2) if not np.isnan(per_class["soil"]) else None,
            "bedrock":   round(per_class["bedrock"], 2) if not np.isnan(per_class["bedrock"]) else None,
            "sand":      round(per_class["sand"], 2) if not np.isnan(per_class["sand"]) else None,
            "gap_vs_supervised": round(gap, 2),
        })

    return rows


def main():
    print("Ensemble Few-Shot Terrain Classification — AI4Mars")
    print(f"Baselines: DINOv2 ViT-L single 93.73% | DINOv3 ViT-L single 92.3%")
    print(f"Supervised: {SUPERVISED_BASELINE}% (DeepLabv3+)\n")

    # Load all available caches
    caches = {}
    for key in CACHE_FILES:
        c = load_cache(key)
        if c is not None:
            caches[key] = c

    available = list(caches.keys())
    print(f"\nAvailable caches: {available}\n")

    all_rows = []

    # Define ensembles to test
    combos = []

    if "dinov2_vitl" in caches and "dinov3_vitl" in caches:
        combos.append(("A_v2L+v3L", ["dinov2_vitl", "dinov3_vitl"], "2048-d"))

    if "dinov2_vitl" in caches and "dinov2_vitb" in caches:
        combos.append(("B_v2L+v2B", ["dinov2_vitl", "dinov2_vitb"], "1792-d"))

    if "dinov2_vitl" in caches and "dinov3_vitb" in caches:
        combos.append(("C_v2L+v3B", ["dinov2_vitl", "dinov3_vitb"], "1792-d"))

    if "dinov2_vitl" in caches and "dinov3_vitl" in caches and "dinov2_vitb" in caches:
        combos.append(("D_v2L+v3L+v2B", ["dinov2_vitl", "dinov3_vitl", "dinov2_vitb"], "2816-d"))

    if not combos:
        print("No valid ensemble combinations available. Check cache files.")
        return

    for name, keys, label in combos:
        rows = run_ensemble(name, keys, caches, label)
        all_rows.extend(rows)

    # Write CSV manually
    if all_rows:
        fieldnames = list(all_rows[0].keys())
        with open(OUT_CSV, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)
    print(f"\n\nCSV saved → {OUT_CSV}")

    print("\n" + "="*70)
    print("SUMMARY — 1000-shot overall accuracy")
    print("="*70)
    rows_1k = [r for r in all_rows if r["shots"] == 1000]
    rows_1k.sort(key=lambda r: r["overall"], reverse=True)
    for row in rows_1k:
        marker = " ← NEW BEST" if row["overall"] > 93.73 else ""
        bedrock = f"{row['bedrock']:.1f}%" if row["bedrock"] is not None else "N/A"
        print(f"  {row['ensemble']:30s} {row['overall']:.2f}%  gap {row['gap_vs_supervised']:+.2f}%"
              f"  Bedrock {bedrock}{marker}")
    print(f"\n  Baseline (DINOv2 ViT-L single):  93.73%  gap -2.94%  Bedrock 92.4%")
    print(f"  Supervised (DeepLabv3+):          {SUPERVISED_BASELINE}%")


if __name__ == "__main__":
    main()
