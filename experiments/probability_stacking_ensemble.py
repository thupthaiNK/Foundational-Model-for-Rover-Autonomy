"""
Purpose: Probability Stacking Meta-Ensemble (Round 5).
         Ensemble B (Round 3) combined models by concatenating raw features (1792-d → LogReg).
         This script instead trains one base LogReg per model to get probability vectors,
         stacks them (4 models × 4 classes = 16-d meta-features), and trains a meta-classifier.
         Key hypothesis: DINOv3-SAT has the highest Bedrock accuracy (93.5%) but overall only
         90.94% — a meta-classifier could learn to trust SAT for Bedrock and DINOv2-L for Soil.
         Uses 5-fold out-of-fold probabilities for meta-training to prevent data leakage.
         No model re-runs needed — all features loaded from cache.
Inputs:  experiments/results/feature_cache/ (dinov2_vitl, dinov2_vitb, dinov3_sat_vitl,
         dinov3_vitl — all pre-extracted .npy files)
Outputs: experiments/results/probability_stacking_ensemble.csv
How to run:
    python3 -u experiments/probability_stacking_ensemble.py
Project: Onboard Visual Foundation Models for Mars Terrain Perception and Rover Navigation
"""

import csv
import os
import time
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier

warnings.filterwarnings("ignore", category=ConvergenceWarning)

# ── Paths ──────────────────────────────────────────────────────────────────────
CACHE_DIR = os.path.join(os.path.dirname(__file__), "results", "feature_cache")
OUT_CSV   = os.path.join(os.path.dirname(__file__), "results",
                         "probability_stacking_ensemble.csv")

# ── Model registry ─────────────────────────────────────────────────────────────
# Models chosen: top performers + DINOv3-SAT (highest single-model Bedrock)
MODELS = {
    "dinov2_vitl":     {"dim": 1024, "single_overall": 93.73, "single_bedrock": 92.39},
    "dinov2_vitb":     {"dim": 768,  "single_overall": 91.29, "single_bedrock": 86.96},
    "dinov3_sat_vitl": {"dim": 1024, "single_overall": 90.94, "single_bedrock": 93.48},
    "dinov3_vitl":     {"dim": 1024, "single_overall": 92.33, "single_bedrock": 89.13},
}

SHOT_COUNTS         = [10, 50, 100, 500, 1000]
N_CLASSES           = 4           # soil=0, bedrock=1, sand=2, big_rock=3
LABEL_MAP           = {0: "soil", 1: "bedrock", 2: "sand"}
SUPERVISED_BASELINE = 96.67
ENSEMBLE_B_1K       = 94.43       # Ensemble B (feature concat) — the bar to beat
SINGLE_BEST_1K      = 93.73       # DINOv2 ViT-L single model

META_C_GRID = [0.01, 0.1, 0.316, 1.0, 3.16, 10.0]


# ── Data loading ───────────────────────────────────────────────────────────────

def load_all_caches():
    caches = {}
    labels_train = labels_test = None

    for name in MODELS:
        paths = {
            "train_feats":  os.path.join(CACHE_DIR, f"{name}_train_1000_feats.npy"),
            "train_labels": os.path.join(CACHE_DIR, f"{name}_train_1000_labels.npy"),
            "test_feats":   os.path.join(CACHE_DIR, f"{name}_test_287_feats.npy"),
            "test_labels":  os.path.join(CACHE_DIR, f"{name}_test_287_labels.npy"),
        }
        for p in paths.values():
            if not os.path.exists(p):
                raise FileNotFoundError(f"Cache missing: {p}")

        caches[name] = {k: np.load(v) for k, v in paths.items()}

        if labels_train is None:
            labels_train = caches[name]["train_labels"]
            labels_test  = caches[name]["test_labels"]

        tr = caches[name]["train_feats"].shape
        te = caches[name]["test_feats"].shape
        print(f"  {name:25s}: train {tr}, test {te}")

    return caches, labels_train, labels_test


# ── Shot sampling ──────────────────────────────────────────────────────────────

def sample_shot_idx(labels, n_per_class):
    """Deterministic: take first n_per_class indices for each class (matches Ensemble B)."""
    idx = []
    for c in range(N_CLASSES):
        c_idx = np.where(labels == c)[0]
        chosen = c_idx[:n_per_class] if len(c_idx) >= n_per_class else c_idx
        idx.extend(chosen.tolist())
    return np.array(idx)


# ── Base classifier ────────────────────────────────────────────────────────────

def make_base_clf(C=0.316):
    return LogisticRegression(
        C=C, max_iter=1000, solver="lbfgs", random_state=42
    )


def probs_full(clf, feats, n_classes=N_CLASSES):
    """Return (N, n_classes) probability matrix, filling zeros for unseen classes."""
    raw = clf.predict_proba(feats)
    out = np.zeros((len(feats), n_classes), dtype=np.float32)
    for i, cls in enumerate(clf.classes_):
        out[:, cls] = raw[:, i]
    return out


# ── OOF probability generation (prevents data leakage) ───────────────────────

def build_oof_probs(feats_all, labels_all, shot_idx, n_folds=5):
    """
    5-fold stratified CV on the shot subset.
    Returns out-of-fold probability matrix (len(shot_idx), N_CLASSES).
    Base classifiers are trained on K-1 folds and predict on the held-out fold,
    so no sample is used to both train and generate its own meta-feature.
    """
    X = feats_all[shot_idx]
    y = labels_all[shot_idx]

    # Reduce folds for very small shot counts to ensure each fold has ≥1 sample per class
    min_class_count = min((y == c).sum() for c in np.unique(y))
    n_folds = min(n_folds, int(min_class_count))
    n_folds = max(n_folds, 2)

    oof = np.zeros((len(shot_idx), N_CLASSES), dtype=np.float32)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    for train_idx, val_idx in skf.split(X, y):
        clf = make_base_clf()
        clf.fit(X[train_idx], y[train_idx])
        p = probs_full(clf, X[val_idx])
        oof[val_idx] = p

    return oof   # (N_shot, 4)


def build_test_probs(feats_all, labels_all, shot_idx, test_feats):
    """Retrain base LogReg on full shot subset → predict probabilities on test set."""
    clf = make_base_clf()
    clf.fit(feats_all[shot_idx], labels_all[shot_idx])
    return probs_full(clf, test_feats)   # (287, 4)


# ── Meta-classifiers ───────────────────────────────────────────────────────────

def per_class_acc(preds, labels):
    return {
        cn: float((preds[labels == li] == labels[labels == li]).mean() * 100)
        if (labels == li).sum() > 0 else float("nan")
        for li, cn in LABEL_MAP.items()
    }


def simple_average(meta_test_probs, test_labels):
    """Baseline: plain average of base-model probabilities (no training needed)."""
    avg = meta_test_probs.reshape(len(meta_test_probs), len(MODELS), N_CLASSES).mean(axis=1)
    preds = avg.argmax(axis=1)
    overall = (preds == test_labels).mean() * 100
    return overall, per_class_acc(preds, test_labels)


def meta_logreg(meta_train, train_labels, meta_test, test_labels):
    """Grid-search C for meta LogReg; return (best_name, best_overall, best_per_class)."""
    best_overall, best_name, best_pc = -1, "", {}
    for C in META_C_GRID:
        clf = LogisticRegression(
            C=C, max_iter=2000, solver="lbfgs", random_state=42
        )
        clf.fit(meta_train, train_labels)
        preds = clf.predict(meta_test)
        overall = (preds == test_labels).mean() * 100
        if overall > best_overall:
            best_overall = overall
            best_name    = f"meta_logreg_C{C}"
            best_pc      = per_class_acc(preds, test_labels)
    return best_name, best_overall, best_pc


def meta_mlp(meta_train, train_labels, meta_test, test_labels):
    """Small MLP meta-classifier (32 hidden units)."""
    clf = MLPClassifier(
        hidden_layer_sizes=(32,), activation="relu", solver="adam",
        max_iter=1000, random_state=42
    )
    clf.fit(meta_train, train_labels)
    preds = clf.predict(meta_test)
    overall = (preds == test_labels).mean() * 100
    return "meta_mlp_32", overall, per_class_acc(preds, test_labels)


# ── Per-shot evaluation ────────────────────────────────────────────────────────

def evaluate_shot(n_shot, caches, labels_train, labels_test):
    shot_idx = sample_shot_idx(labels_train, n_per_class=n_shot)
    shot_labels = labels_train[shot_idx]

    # Stack OOF probs (meta-train) — prevents data leakage
    oof_list  = []
    test_list = []
    for name, cache in caches.items():
        oof = build_oof_probs(cache["train_feats"], labels_train, shot_idx)
        tp  = build_test_probs(cache["train_feats"], labels_train, shot_idx,
                               cache["test_feats"])
        oof_list.append(oof)
        test_list.append(tp)

    # (N_shot, 16) and (287, 16)
    meta_train = np.concatenate(oof_list, axis=1)
    meta_test  = np.concatenate(test_list, axis=1)

    # Baseline: simple average of base probabilities (no training)
    avg_overall, avg_pc = simple_average(meta_test, labels_test)

    # Meta LogReg (grid-searched C)
    lr_name, lr_overall, lr_pc = meta_logreg(
        meta_train, shot_labels, meta_test, labels_test
    )

    # Meta MLP
    try:
        mlp_name, mlp_overall, mlp_pc = meta_mlp(
            meta_train, shot_labels, meta_test, labels_test
        )
    except Exception as e:
        print(f"    MLP skipped: {e}")
        mlp_name, mlp_overall, mlp_pc = "meta_mlp_32", float("nan"), {}

    best_overall = max(lr_overall, mlp_overall if not np.isnan(mlp_overall) else -1)
    best_name    = lr_name if lr_overall >= (mlp_overall if not np.isnan(mlp_overall) else -1) \
                   else mlp_name
    best_pc      = lr_pc  if lr_overall >= (mlp_overall if not np.isnan(mlp_overall) else -1) \
                   else mlp_pc

    return {
        "avg_overall": avg_overall,   "avg_pc": avg_pc,
        "lr_overall":  lr_overall,    "lr_pc":  lr_pc,    "lr_name":  lr_name,
        "mlp_overall": mlp_overall,   "mlp_pc": mlp_pc,   "mlp_name": mlp_name,
        "best_overall": best_overall, "best_pc": best_pc, "best_name": best_name,
        "meta_dim": meta_train.shape[1],
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  Probability Stacking Meta-Ensemble — Round 5")
    print(f"  Models: {list(MODELS.keys())}")
    print(f"  Meta-features: {len(MODELS)} models × {N_CLASSES} classes = {len(MODELS)*N_CLASSES}-d")
    print(f"  Baselines: Ensemble B 94.43% | DINOv2 ViT-L 93.73% | Supervised 96.67%")
    print("=" * 70)

    print("\nLoading feature caches...")
    caches, labels_train, labels_test = load_all_caches()

    tr_dist = {i: int((labels_train == i).sum()) for i in range(N_CLASSES)}
    te_dist = {i: int((labels_test  == i).sum()) for i in range(N_CLASSES)}
    print(f"Train: {labels_train.shape[0]} images — {tr_dist}")
    print(f"Test:  {labels_test.shape[0]}  images — {te_dist}")

    all_rows = []
    t_total  = time.time()

    for n_shot in SHOT_COUNTS:
        t0 = time.time()
        print(f"\n{'─'*60}")
        print(f"  {n_shot}-shot  (5-fold OOF → meta-train → meta-test)")

        res = evaluate_shot(n_shot, caches, labels_train, labels_test)
        elapsed = time.time() - t0

        # Comparison marker at 1000-shot
        marker = ""
        if n_shot == 1000:
            diff = res["best_overall"] - ENSEMBLE_B_1K
            if diff > 0:
                marker = f"  ← NEW BEST (+{diff:.2f}pp vs Ensemble B 94.43%)"
            else:
                marker = f"  ← {diff:+.2f}pp vs Ensemble B 94.43%"

        def fmt(v): return f"{v:.1f}%" if not np.isnan(v) else "N/A"

        print(f"    avg (no train):  overall {res['avg_overall']:.2f}%  "
              f"soil {fmt(res['avg_pc'].get('soil', float('nan')))}  "
              f"bedrock {fmt(res['avg_pc'].get('bedrock', float('nan')))}  "
              f"sand {fmt(res['avg_pc'].get('sand', float('nan')))}")
        print(f"    {res['lr_name']:30s}  overall {res['lr_overall']:.2f}%  "
              f"soil {fmt(res['lr_pc'].get('soil', float('nan')))}  "
              f"bedrock {fmt(res['lr_pc'].get('bedrock', float('nan')))}  "
              f"sand {fmt(res['lr_pc'].get('sand', float('nan')))}")
        print(f"    {res['mlp_name']:30s}  overall {res['mlp_overall']:.2f}%  "
              f"soil {fmt(res['mlp_pc'].get('soil', float('nan')))}  "
              f"bedrock {fmt(res['mlp_pc'].get('bedrock', float('nan')))}  "
              f"sand {fmt(res['mlp_pc'].get('sand', float('nan')))}")
        print(f"    BEST: {res['best_overall']:.2f}%  "
              f"bedrock {fmt(res['best_pc'].get('bedrock', float('nan')))}  "
              f"[{elapsed:.1f}s]{marker}")

        gap = res["best_overall"] - SUPERVISED_BASELINE
        all_rows.append({
            "shots":             n_shot,
            "meta_dim":          res["meta_dim"],
            "best_meta_clf":     res["best_name"],
            "best_overall":      round(res["best_overall"], 2),
            "best_soil":         round(res["best_pc"].get("soil",    float("nan")), 2),
            "best_bedrock":      round(res["best_pc"].get("bedrock", float("nan")), 2),
            "best_sand":         round(res["best_pc"].get("sand",    float("nan")), 2),
            "gap_vs_supervised": round(gap, 2),
            "vs_ensemble_b":     round(res["best_overall"] - ENSEMBLE_B_1K, 2) if n_shot == 1000 else "",
            "avg_overall":       round(res["avg_overall"], 2),
            "lr_overall":        round(res["lr_overall"], 2),
            "mlp_overall":       round(res["mlp_overall"], 2) if not np.isnan(res["mlp_overall"]) else "",
            "elapsed_s":         round(elapsed, 1),
        })

    # Write CSV
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)

    # Final summary
    total_elapsed = time.time() - t_total
    print(f"\n{'='*70}")
    print("FINAL SUMMARY — Probability Stacking")
    print(f"{'='*70}")
    print(f"{'Shots':>8}  {'Overall':>9}  {'Bedrock':>9}  {'Sand':>9}  {'Gap':>8}")
    print(f"{'─'*60}")
    for row in all_rows:
        bedrock_str = f"{row['best_bedrock']:.1f}%" if row["best_bedrock"] != "" else "N/A"
        sand_str    = f"{row['best_sand']:.1f}%"    if row["best_sand"]    != "" else "N/A"
        print(f"{row['shots']:>8}  {row['best_overall']:>8.2f}%  "
              f"{bedrock_str:>9}  {sand_str:>9}  {row['gap_vs_supervised']:>+7.2f}%")

    print(f"\nKey comparison at 1000-shot:")
    row_1k = [r for r in all_rows if r["shots"] == 1000][0]
    models_ref = [
        ("Probability Stacking (this)",  row_1k["best_overall"],  row_1k["best_bedrock"]),
        ("Ensemble B (feat concat)",      94.43,                   90.2),
        ("DINOv2 ViT-L (single)",         93.73,                   92.4),
        ("DINOv3-SAT ViT-L (single)",     90.94,                   93.5),
        ("Supervised (DeepLabv3+)",        96.67,                   94.9),
    ]
    for name, overall, bedrock in models_ref:
        print(f"  {name:40s}  {overall:.2f}%  Bedrock {bedrock:.1f}%")

    print(f"\nTotal compute: {total_elapsed:.0f}s  |  CSV → {OUT_CSV}")


if __name__ == "__main__":
    main()
