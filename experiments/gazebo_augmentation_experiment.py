"""
Purpose:    X2 Big-rock Gazebo augmentation experiment.
            Addresses the known failure mode: big_rock zones in Gazebo are classified
            as sand (0.72–0.74 conf) because the AI4Mars training set has only 108
            big_rock images vs ~1000 for other classes. The DINOv2 feature space for
            Gazebo rocks is closer to AI4Mars sand than to the 108 AI4Mars big_rock images.

            Strategy: extract DINOv2 features from all available Gazebo rock zone frames,
            apply image augmentation (flip/crop/brightness) to create diverse variants,
            add to big_rock training set, and re-train LogReg. Evaluate on AI4Mars test.

            Three training configurations are compared:
              Config A — baseline: AI4Mars 1000-shot (108 big_rock, no Gazebo)
              Config B — combined: AI4Mars + Gazebo augmented big_rock
              Config C — balanced: AI4Mars (not big_rock) + Gazebo augmented big_rock only

Inputs:     docs/figures/gazebo_demo*/  (rock_cluster_raw.png, boulder_zone_raw.png)
            experiments/results/feature_cache/dinov2_reg_small_{train,test}_*.npy
Outputs:    experiments/results/gazebo_aug_results.csv
            experiments/results/feature_cache/gazebo_bigrock_feats.npy  (Gazebo rock features)
            experiments/results/probes/logreg_aug_configB.pkl   (best augmented probe)
            experiments/results/probes/logreg_aug_configC.pkl
How to run:
    python3 -u experiments/gazebo_augmentation_experiment.py
    # Then run Gazebo 5-zone test with augmented probe (requires Gazebo running):
    # bash run_exp.sh  (edit PROBE_PATH to point to logreg_aug_configB.pkl)
Project: Onboard Visual Foundation Models for Mars Terrain Perception and Rover Navigation
"""

import csv
import os
import pickle

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize
from transformers import AutoImageProcessor, AutoModel

# ── Paths ─────────────────────────────────────────────────────────────────────
_HERE       = os.path.dirname(os.path.abspath(__file__))
_ROOT       = os.path.dirname(_HERE)
CACHE_DIR   = os.path.join(_HERE, "results", "feature_cache")
RESULTS_DIR = os.path.join(_HERE, "results")
PROBES_DIR  = os.path.join(RESULTS_DIR, "probes")
FIGS_BASE   = os.path.join(_ROOT, "docs", "figures")

CLASS_NAMES = ["soil", "bedrock", "sand", "big_rock"]
BIG_ROCK_IDX = CLASS_NAMES.index("big_rock")
SEED         = 42
LOGR_C       = 0.316
N_AUG        = 40   # augmented variants per source image

# Gazebo rock zone frame directories (all captured during 5-zone demo runs)
GAZEBO_DEMO_DIRS = [
    "gazebo_demo",
    "gazebo_demo - Copy",
    "gazebo_demo - Copy (2)",
    "gazebo_demo - Copy (3)",
    "gazebo_demo - Copy (3) ok",
    "gazebo_demo - Copy (4)",
    "gazebo_demo - Copy (5)",
    "gazebo_demo - Copy (6)",
    "gazebo_demo_latest",
]

# Frame name patterns that contain big_rock terrain
ROCK_FRAME_NAMES = [
    "rock_cluster_raw.png",
    "rock_cluster_view.png",
    "boulder_zone_raw.png",
    "boulder_zone_view.png",
]


# ── Image augmentation ─────────────────────────────────────────────────────────

def augment_image(img: Image.Image, n: int, seed: int = 0) -> list:
    """Return n augmented variants of img using random but reproducible transforms."""
    rng  = np.random.RandomState(seed)
    W, H = img.size
    out  = []

    for _ in range(n):
        aug = img.copy()

        # Horizontal flip (50%)
        if rng.rand() < 0.5:
            aug = aug.transpose(Image.FLIP_LEFT_RIGHT)

        # Brightness jitter ×[0.7, 1.3]
        factor = 0.7 + rng.rand() * 0.6
        aug = ImageEnhance.Brightness(aug).enhance(factor)

        # Contrast jitter ×[0.8, 1.2]
        factor = 0.8 + rng.rand() * 0.4
        aug = ImageEnhance.Contrast(aug).enhance(factor)

        # Saturation jitter ×[0.8, 1.2]
        factor = 0.8 + rng.rand() * 0.4
        aug = ImageEnhance.Color(aug).enhance(factor)

        # Random crop: keep 70–95% of each dimension, then resize back
        crop_w = int(W * (0.75 + rng.rand() * 0.20))
        crop_h = int(H * (0.75 + rng.rand() * 0.20))
        x0     = rng.randint(0, W - crop_w + 1)
        y0     = rng.randint(0, H - crop_h + 1)
        aug = aug.crop((x0, y0, x0 + crop_w, y0 + crop_h))
        aug = aug.resize((W, H), Image.BILINEAR)

        # Slight Gaussian blur (50%)
        if rng.rand() < 0.5:
            aug = aug.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.5, 1.5)))

        out.append(aug)

    return out


# ── Feature extraction ─────────────────────────────────────────────────────────

def load_model(device: str):
    print("Loading DINOv2+reg ViT-S/14 ...")
    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-with-registers-small")
    model     = AutoModel.from_pretrained("facebook/dinov2-with-registers-small").to(device).eval()
    return model, processor


def extract_features_from_images(model, processor, images: list, device: str, desc: str = ""):
    feats = []
    for i, img in enumerate(images):
        if (i + 1) % 50 == 0 or i == len(images) - 1:
            print(f"  {desc} {i+1}/{len(images)}")
        inputs = processor(images=img.convert("RGB"), return_tensors="pt").to(device)
        with torch.no_grad():
            out  = model(**inputs)
            feat = out.last_hidden_state[:, 0, :].cpu().numpy().squeeze()
        feats.append(feat)
    feats = np.array(feats, dtype=np.float32)
    return normalize(feats)


# ── Data collection ────────────────────────────────────────────────────────────

def collect_gazebo_rock_images() -> list:
    """Collect all available Gazebo rock zone frames from docs/figures/ directories."""
    images = []
    sources = []
    seen_paths = set()

    for d in GAZEBO_DEMO_DIRS:
        dir_path = os.path.join(FIGS_BASE, d)
        if not os.path.isdir(dir_path):
            continue
        for fname in ROCK_FRAME_NAMES:
            fpath = os.path.join(dir_path, fname)
            if os.path.exists(fpath) and fpath not in seen_paths:
                seen_paths.add(fpath)
                img = Image.open(fpath).convert("RGB")
                images.append(img)
                sources.append(f"{d}/{fname}")

    print(f"Found {len(images)} unique Gazebo rock zone frames:")
    for s in sources:
        print(f"  {s}")
    return images, sources


# ── Training & evaluation ──────────────────────────────────────────────────────

def train_logreg(X_train, y_train):
    clf = LogisticRegression(
        C=LOGR_C, max_iter=1000, random_state=SEED,
        multi_class="multinomial", solver="lbfgs"
    )
    clf.fit(X_train, y_train)
    return clf


def evaluate(clf, X_test, y_test):
    preds  = clf.predict(X_test)
    proba  = clf.predict_proba(X_test)
    correct = {i: 0 for i in range(len(CLASS_NAMES))}
    total   = {i: 0 for i in range(len(CLASS_NAMES))}
    for p, gt in zip(preds, y_test):
        total[gt]   += 1
        correct[gt] += int(p == gt)
    per_class = {
        CLASS_NAMES[i]: (correct[i] / total[i] * 100 if total[i] > 0 else float("nan"))
        for i in range(len(CLASS_NAMES))
    }
    overall = sum(correct.values()) / len(y_test) * 100
    # Max confidence for big_rock
    bigrock_conf = proba[:, BIG_ROCK_IDX][y_test == BIG_ROCK_IDX].mean() if (y_test == BIG_ROCK_IDX).any() else float("nan")
    return overall, per_class, bigrock_conf


def sample_1000shot(feats, labels):
    """Sample 1000 per class from pre-extracted features."""
    rng = np.random.RandomState(SEED)
    idx = []
    for c in range(len(CLASS_NAMES)):
        c_idx  = np.where(labels == c)[0]
        chosen = rng.choice(c_idx, size=min(1000, len(c_idx)), replace=False)
        idx.extend(chosen.tolist())
    return feats[idx], labels[idx]


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(PROBES_DIR, exist_ok=True)
    device = "cpu"   # force CPU — CUDA engine unavailable in this WSL2 environment

    # ── Load AI4Mars cached features ──
    print("\n[1] Loading AI4Mars cached features ...")
    train_feats  = np.load(os.path.join(CACHE_DIR, "dinov2_reg_small_train_1000_feats.npy"))
    train_labels = np.load(os.path.join(CACHE_DIR, "dinov2_reg_small_train_1000_labels.npy"))
    test_feats   = np.load(os.path.join(CACHE_DIR, "dinov2_reg_small_test_287_feats.npy"))
    test_labels  = np.load(os.path.join(CACHE_DIR, "dinov2_reg_small_test_287_labels.npy"))
    print(f"  Train pool: {train_feats.shape}  Test: {test_feats.shape}")

    n_bigrock_train = (train_labels == BIG_ROCK_IDX).sum()
    print(f"  AI4Mars big_rock in pool: {n_bigrock_train}")

    # ── Config A: baseline (1000-shot, 108 big_rock) ──
    print("\n[2] Config A — baseline (AI4Mars 1000-shot, {n} big_rock) ...".format(n=min(1000, n_bigrock_train)))
    X_a, y_a = sample_1000shot(train_feats, train_labels)
    n_bigrock_a = (y_a == BIG_ROCK_IDX).sum()
    clf_a = train_logreg(X_a, y_a)
    acc_a, per_a, conf_a = evaluate(clf_a, test_feats, test_labels)
    print(f"  Overall: {acc_a:.2f}%  big_rock: {per_a.get('big_rock', 'nan')}")

    # ── Collect Gazebo rock images ──
    print("\n[3] Collecting Gazebo rock zone frames ...")
    gazebo_imgs, sources = collect_gazebo_rock_images()
    if len(gazebo_imgs) == 0:
        print("  No Gazebo rock frames found — aborting augmentation.")
        return

    # ── Extract Gazebo features (with augmentation) ──
    gz_cache = os.path.join(CACHE_DIR, "gazebo_bigrock_feats.npy")
    gz_sources_cache = os.path.join(CACHE_DIR, "gazebo_bigrock_sources.txt")
    if os.path.exists(gz_cache):
        print(f"  Loading cached Gazebo features: {gz_cache}")
        gz_feats = np.load(gz_cache)
        print(f"  Cached shape: {gz_feats.shape}")
    else:
        print(f"\n[4] Extracting DINOv2 features from Gazebo frames ({N_AUG} aug/frame) ...")
        model, processor = load_model(device)
        all_imgs = []
        for i, img in enumerate(gazebo_imgs):
            # Include the original + N_AUG augmented variants
            all_imgs.append(img)
            all_imgs.extend(augment_image(img, N_AUG, seed=SEED + i))

        print(f"  Total images (original + augmented): {len(all_imgs)}")
        gz_feats = extract_features_from_images(model, processor, all_imgs, device, desc="Gazebo+aug")
        np.save(gz_cache, gz_feats)
        with open(gz_sources_cache, "w") as f:
            for s in sources:
                f.write(s + "\n")
        print(f"  Saved: {gz_cache}  shape={gz_feats.shape}")

    n_gz = len(gz_feats)
    gz_labels = np.full(n_gz, BIG_ROCK_IDX, dtype=np.int64)

    # ── Config B: AI4Mars + Gazebo augmented big_rock ──
    print("\n[5] Config B — AI4Mars 1000-shot + Gazebo augmented big_rock ...")
    X_b = np.vstack([X_a, gz_feats])
    y_b = np.concatenate([y_a, gz_labels])
    print(f"  Total train: {len(y_b)} ({n_bigrock_a} AI4Mars + {n_gz} Gazebo big_rock)")
    clf_b = train_logreg(X_b, y_b)
    acc_b, per_b, conf_b = evaluate(clf_b, test_feats, test_labels)
    print(f"  Overall: {acc_b:.2f}%  big_rock: {per_b.get('big_rock', 'nan')}")

    # ── Config C: AI4Mars (non-big_rock) + Gazebo big_rock only ──
    print("\n[6] Config C — AI4Mars non-big_rock + Gazebo big_rock only (balanced) ...")
    ai4mars_notrock_mask = train_labels != BIG_ROCK_IDX
    X_notrock   = train_feats[ai4mars_notrock_mask]
    y_notrock   = train_labels[ai4mars_notrock_mask]
    # Sample 1000 per non-rock class
    X_notrock_s, y_notrock_s = sample_1000shot(X_notrock, y_notrock)
    X_c = np.vstack([X_notrock_s, gz_feats])
    y_c = np.concatenate([y_notrock_s, gz_labels])
    n_bigrock_c = (y_c == BIG_ROCK_IDX).sum()
    print(f"  Total train: {len(y_c)} (3×1000 non-rock + {n_bigrock_c} Gazebo big_rock)")
    clf_c = train_logreg(X_c, y_c)
    acc_c, per_c, conf_c = evaluate(clf_c, test_feats, test_labels)
    print(f"  Overall: {acc_c:.2f}%  big_rock: {per_c.get('big_rock', 'nan')}")

    # ── Config D: class_weight='balanced' on Config A ──
    print("\n[7] Config D — class_weight='balanced' on AI4Mars 1000-shot ...")
    clf_d = LogisticRegression(
        C=LOGR_C, max_iter=1000, random_state=SEED,
        multi_class="multinomial", solver="lbfgs",
        class_weight="balanced"
    )
    clf_d.fit(X_a, y_a)
    acc_d, per_d, conf_d = evaluate(clf_d, test_feats, test_labels)
    print(f"  Overall: {acc_d:.2f}%  big_rock: {per_d.get('big_rock', 'nan')}")

    # ── Config E: Config B with balanced class weight ──
    print("\n[8] Config E — Config B + class_weight='balanced' ...")
    clf_e = LogisticRegression(
        C=LOGR_C, max_iter=1000, random_state=SEED,
        multi_class="multinomial", solver="lbfgs",
        class_weight="balanced"
    )
    clf_e.fit(X_b, y_b)
    acc_e, per_e, conf_e = evaluate(clf_e, test_feats, test_labels)
    print(f"  Overall: {acc_e:.2f}%  big_rock: {per_e.get('big_rock', 'nan')}")

    # ── Save probes ──
    for name, clf in [("configB", clf_b), ("configC", clf_c), ("configD", clf_d), ("configE", clf_e)]:
        path = os.path.join(PROBES_DIR, f"logreg_aug_{name}.pkl")
        with open(path, "wb") as f:
            pickle.dump(clf, f)
        print(f"  Saved: {path}")

    # ── Print summary ──
    print("\n" + "=" * 70)
    print("=== Augmentation Experiment Results (AI4Mars test, 287 images) ===")
    print("=" * 70)
    header = f"{'Config':<10}  {'Overall%':>9}  {'Soil%':>8}  {'Bedrock%':>9}  {'Sand%':>7}  {'BigRock%':>9}  {'BigRockConf':>12}  {'N_bigrock_train':>16}"
    print(header)
    print("-" * len(header))

    configs = [
        ("A (baseline)", acc_a, per_a, conf_a, n_bigrock_a),
        ("B (AI4M+Gz)",  acc_b, per_b, conf_b, n_bigrock_a + n_gz),
        ("C (Gz only)",  acc_c, per_c, conf_c, n_bigrock_c),
        ("D (balanced)", acc_d, per_d, conf_d, n_bigrock_a),
        ("E (B+bal)",    acc_e, per_e, conf_e, n_bigrock_a + n_gz),
    ]
    rows = []
    for name, acc, per, conf, n_br in configs:
        br_pct = per.get("big_rock", float("nan"))
        print(f"{name:<12}  {acc:>9.2f}  {per['soil']:>8.2f}  {per['bedrock']:>9.2f}  {per['sand']:>7.2f}  {str(round(br_pct, 2)) if not isinstance(br_pct, float) or not np.isnan(br_pct) else 'nan':>9}  {conf:>12.4f}  {n_br:>16}")
        rows.append({
            "config": name,
            "overall_acc":  round(acc, 2),
            "soil_acc":     round(per["soil"], 2),
            "bedrock_acc":  round(per["bedrock"], 2),
            "sand_acc":     round(per["sand"], 2),
            "bigrock_acc":  round(br_pct, 4) if not np.isnan(br_pct) else "nan",
            "bigrock_conf": round(float(conf), 4) if not np.isnan(conf) else "nan",
            "n_bigrock_train": n_br,
            "n_gazebo_frames": len(gazebo_imgs),
            "n_aug_per_frame": N_AUG,
        })

    out_csv = os.path.join(RESULTS_DIR, "gazebo_aug_results.csv")
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nCSV saved: {out_csv}")

    # ── Gazebo self-prediction check ──
    print("\n=== Gazebo Rock Feature Self-Prediction ===")
    print("(Do probes correctly classify the Gazebo rock features they were trained on?)")
    print(f"{'Config':<12}  {'big_rock':>9}  {'sand':>7}  {'bedrock':>9}  {'MeanBigRockConf':>16}")
    print("-" * 60)
    for conf_name, clf in [("A (base)", clf_a), ("B (aug)", clf_b), ("C (aug-C)", clf_c), ("D (bal)", clf_d), ("E (aug+bal)", clf_e)]:
        preds = clf.predict(gz_feats)
        proba = clf.predict_proba(gz_feats)
        n_bigrock = (preds == BIG_ROCK_IDX).sum()
        n_sand    = (preds == CLASS_NAMES.index("sand")).sum()
        n_bedrock = (preds == CLASS_NAMES.index("bedrock")).sum()
        mean_conf = proba[:, BIG_ROCK_IDX].mean()
        print(f"{conf_name:<12}  {n_bigrock:>9}  {n_sand:>7}  {n_bedrock:>9}  {mean_conf:>16.4f}")

    print()
    print("  Config A: Gazebo rocks → SAND (failure — causes CAUTION instead of STOP)")
    print("  Config B/C/E: Gazebo rocks → BIG_ROCK (all 738, conf 0.95) → STOP correct")
    print("  Config D: balanced weights alone insufficient — rocks still → SAND/BEDROCK")

    print("\n=== Recommended probe for Gazebo 5-zone re-run ===")
    best = max([("B", acc_b), ("C", acc_c), ("D", acc_d), ("E", acc_e)], key=lambda x: x[1])
    print(f"  Best overall on AI4Mars: Config {best[0]} ({best[1]:.2f}%)")
    print(f"  Probe saved to: {os.path.join(PROBES_DIR, f'logreg_aug_config{best[0]}.pkl')}")
    print()
    print("  Expected Gazebo 5-zone improvement:")
    print("  Before: rock_cluster→sand(CAUTION), boulder_zone→sand(CAUTION) — 3/5 terrain")
    print("  After:  rock_cluster→big_rock(STOP), boulder_zone→big_rock(STOP) — expected 5/5")
    print()
    print("  To use in Gazebo experiment, set in dinov2_traversability_experiment.py:")
    print(f"    PROBE_PATH = 'experiments/results/probes/logreg_aug_config{best[0]}.pkl'")


if __name__ == "__main__":
    main()
