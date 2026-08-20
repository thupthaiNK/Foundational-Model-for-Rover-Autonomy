"""
Purpose:    McNemar's test comparing Ensemble B (DINOv2 ViT-L + ViT-B concat, 94.43%)
            vs DINOv2 ViT-L single model (93.73%) on the AI4Mars test set (n=287).
            Tests whether the accuracy difference is statistically significant.
Inputs:     experiments/results/feature_cache/dinov2_vitl_{train,test}_*.npy
            experiments/results/feature_cache/dinov2_vitb_{train,test}_*.npy
Outputs:    Console report + experiments/results/mcnemar_results.csv
How to run:
    python3 experiments/mcnemar_test.py
Project: Onboard Visual Foundation Models for Mars Terrain Perception and Rover Navigation
"""

import csv
import os

import numpy as np
from scipy.stats import chi2
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize

# ── Config ─────────────────────────────────────────────────────────────────────
LOGR_C   = 0.316
N_SHOT   = 1000
CLASS_NAMES = ["soil", "bedrock", "sand", "big_rock"]

_HERE      = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR  = os.path.join(_HERE, "results", "feature_cache")
RESULTS_DIR = os.path.join(_HERE, "results")


# ── Load feature caches ────────────────────────────────────────────────────────

def load_cache(prefix: str, split: str, n: int) -> tuple:
    """Load .npy feature + label files.  split = 'train' | 'test'."""
    if split == "train":
        feats  = np.load(os.path.join(CACHE_DIR, f"{prefix}_train_{n}_feats.npy"))
        labels = np.load(os.path.join(CACHE_DIR, f"{prefix}_train_{n}_labels.npy"))
    else:
        feats  = np.load(os.path.join(CACHE_DIR, f"{prefix}_test_287_feats.npy"))
        labels = np.load(os.path.join(CACHE_DIR, f"{prefix}_test_287_labels.npy"))
    return feats, labels


def sample_nshot(feats: np.ndarray, labels: np.ndarray, n_shot: int) -> tuple:
    """Sample n_shot per class — identical protocol to all other experiments."""
    idx = []
    for c in np.unique(labels):
        c_idx  = np.where(labels == c)[0]
        chosen = c_idx[:n_shot] if len(c_idx) >= n_shot else c_idx
        idx.extend(chosen.tolist())
    idx = np.array(idx)
    return feats[idx], labels[idx]


# ── Train LogReg + predict ─────────────────────────────────────────────────────

def train_and_predict(X_train, y_train, X_test) -> np.ndarray:
    X_tr = normalize(X_train, norm="l2")
    X_te = normalize(X_test,  norm="l2")
    clf  = LogisticRegression(C=LOGR_C, max_iter=1000, solver="lbfgs", random_state=42)
    clf.fit(X_tr, y_tr := y_train)
    return clf.predict(X_te)


# ── McNemar's test ─────────────────────────────────────────────────────────────

def mcnemar_test(correct_a: np.ndarray, correct_b: np.ndarray) -> dict:
    """
    McNemar's test with continuity correction (Edwards, 1948).
    correct_a, correct_b: boolean arrays (True = correct prediction).
    Returns dict with contingency table, chi2, p-value, interpretation.
    """
    n11 = int(np.sum( correct_a &  correct_b))   # both correct
    n10 = int(np.sum( correct_a & ~correct_b))   # A correct, B wrong
    n01 = int(np.sum(~correct_a &  correct_b))   # A wrong,  B correct
    n00 = int(np.sum(~correct_a & ~correct_b))   # both wrong

    # Chi-squared with continuity correction (recommended for small n01+n10)
    b, c = n01, n10
    denom = b + c
    if denom == 0:
        return {"n11": n11, "n10": n10, "n01": n01, "n00": n00,
                "chi2": 0.0, "p_value": 1.0, "significant": False,
                "note": "b+c=0: models make identical errors"}

    chi2_stat = (abs(b - c) - 1) ** 2 / denom
    p_value   = 1 - chi2.cdf(chi2_stat, df=1)

    return {
        "n11": n11, "n10": n10, "n01": n01, "n00": n00,
        "chi2": round(chi2_stat, 4),
        "p_value": round(p_value, 4),
        "significant": p_value < 0.05,
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("McNemar's Test: Ensemble B vs DINOv2 ViT-L single")
    print(f"Test set: n=287 | LogReg C={LOGR_C} | {N_SHOT}-shot")
    print("=" * 60)

    # Load ViT-L features
    print("\nLoading feature caches …")
    vitl_train_feats, vitl_train_labels = load_cache("dinov2_vitl", "train", N_SHOT)
    vitl_test_feats,  vitl_test_labels  = load_cache("dinov2_vitl", "test",  N_SHOT)

    # Load ViT-B features
    vitb_train_feats, vitb_train_labels = load_cache("dinov2_vitb", "train", N_SHOT)
    vitb_test_feats,  _                 = load_cache("dinov2_vitb", "test",  N_SHOT)

    y_test = vitl_test_labels   # ground truth (same for both models)
    n      = len(y_test)
    print(f"  ViT-L train: {vitl_train_feats.shape}  test: {vitl_test_feats.shape}")
    print(f"  ViT-B train: {vitb_train_feats.shape}  test: {vitb_test_feats.shape}")

    # ── Model A: DINOv2 ViT-L single (1024-d) ─────────────────────────────────
    print("\n[Model A] DINOv2 ViT-L single (1024-d) …")
    X_vitl_tr, y_vitl_tr = sample_nshot(vitl_train_feats, vitl_train_labels, N_SHOT)
    pred_vitl = train_and_predict(X_vitl_tr, y_vitl_tr, vitl_test_feats)
    acc_vitl  = float(np.mean(pred_vitl == y_test)) * 100
    print(f"  Accuracy: {acc_vitl:.2f}%  (reported: 93.73%)")

    # ── Model B: Ensemble B — ViT-L + ViT-B concat (1792-d) ──────────────────
    print("\n[Model B] Ensemble B — ViT-L + ViT-B concat (1792-d) …")
    X_vitb_tr, y_vitb_tr = sample_nshot(vitb_train_feats, vitb_train_labels, N_SHOT)

    # Original protocol (ensemble_terrain_test.py): concat raw features THEN L2-normalise
    X_ens_tr = normalize(np.concatenate([X_vitl_tr, X_vitb_tr], axis=1), norm="l2")
    X_ens_te = normalize(np.concatenate([vitl_test_feats, vitb_test_feats], axis=1), norm="l2")

    clf_ens = LogisticRegression(C=LOGR_C, max_iter=1000, solver="lbfgs", random_state=42)
    clf_ens.fit(X_ens_tr, y_vitl_tr)   # labels same for both
    pred_ens = clf_ens.predict(X_ens_te)
    acc_ens  = float(np.mean(pred_ens == y_test)) * 100
    print(f"  Accuracy: {acc_ens:.2f}%  (reported: 94.43%)")

    # ── McNemar's test ─────────────────────────────────────────────────────────
    correct_vitl = pred_vitl == y_test
    correct_ens  = pred_ens  == y_test

    result = mcnemar_test(correct_vitl, correct_ens)

    print("\n" + "=" * 60)
    print("McNemar Contingency Table")
    print("  Rows = DINOv2 ViT-L  |  Cols = Ensemble B")
    print(f"  {'':20s}  {'Ens correct':>12}  {'Ens wrong':>10}")
    print(f"  {'ViT-L correct':20s}  {result['n11']:>12}  {result['n10']:>10}")
    print(f"  {'ViT-L wrong':20s}  {result['n01']:>12}  {result['n00']:>10}")

    print(f"\n  b (ViT-L wrong, Ens correct): {result['n01']}")
    print(f"  c (ViT-L correct, Ens wrong): {result['n10']}")
    print(f"  b + c (discordant pairs):     {result['n01'] + result['n10']}")
    print(f"\n  χ² (with continuity correction): {result['chi2']}")
    print(f"  p-value:                         {result['p_value']}")
    print(f"  Significant (α=0.05):            {result['significant']}")

    print("\n" + "=" * 60)
    print("INTERPRETATION")
    diff = acc_ens - acc_vitl
    if result["significant"]:
        print(f"  Ensemble B ({acc_ens:.2f}%) is SIGNIFICANTLY better than")
        print(f"  DINOv2 ViT-L ({acc_vitl:.2f}%) — difference {diff:+.2f}pp")
        print(f"  (p={result['p_value']:.4f} < 0.05, McNemar χ²={result['chi2']:.4f})")
        print(f"  → The feature-concat ensemble provides a statistically")
        print(f"    real improvement over the single best model.")
    else:
        print(f"  Ensemble B ({acc_ens:.2f}%) is NOT significantly better than")
        print(f"  DINOv2 ViT-L ({acc_vitl:.2f}%) — difference {diff:+.2f}pp")
        print(f"  (p={result['p_value']:.4f} ≥ 0.05, McNemar χ²={result['chi2']:.4f})")
        print(f"  → The accuracy gap ({diff:+.2f}pp) is within chance variation")
        print(f"    on n={n} images. DINOv2 ViT-L single is the recommended")
        print(f"    deployment model (simpler, ~same accuracy, 6× faster).")

    # ── Save CSV ───────────────────────────────────────────────────────────────
    os.makedirs(RESULTS_DIR, exist_ok=True)
    csv_path = os.path.join(RESULTS_DIR, "mcnemar_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "model_a", "model_b", "n_test",
            "acc_a", "acc_b", "diff_pp",
            "n11", "n10", "n01", "n00",
            "chi2", "p_value", "significant"
        ])
        writer.writeheader()
        writer.writerow({
            "model_a": "DINOv2_ViT-L_single",
            "model_b": "Ensemble_B_ViT-L+ViT-B",
            "n_test":  n,
            "acc_a":   round(acc_vitl, 4),
            "acc_b":   round(acc_ens,  4),
            "diff_pp": round(acc_ens - acc_vitl, 4),
            **{k: result[k] for k in ["n11", "n10", "n01", "n00", "chi2", "p_value", "significant"]},
        })
    print(f"\nCSV saved → {csv_path}")


if __name__ == "__main__":
    main()
