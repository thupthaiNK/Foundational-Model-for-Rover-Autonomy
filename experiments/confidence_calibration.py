"""
Purpose:    Confidence calibration (reliability diagram) for DINOv2 ViT-S, ViT-L, and Ensemble B.
            Plots predicted confidence vs actual accuracy across 10 equal-width bins.
            Answers: "When the model says 90% confidence, is it actually right 90% of the time?"
            Also computes ECE (Expected Calibration Error) — a single calibration quality metric.
Inputs:     Feature caches in experiments/results/feature_cache/
            (same caches used by seed_stability_test.py — no model re-run needed)
Outputs:    experiments/results/figures/confidence_calibration.png
            Console table with per-bin accuracy and ECE per model
How to run:
    python3 -u experiments/confidence_calibration.py
Project: Onboard Visual Foundation Models for Mars Terrain Perception and Rover Navigation
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize

# ── Paths ─────────────────────────────────────────────────────────────────────
_HERE       = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR   = os.path.join(_HERE, "results", "feature_cache")
FIGURES_DIR = os.path.join(_HERE, "results", "figures")

CLASS_NAMES = ["soil", "bedrock", "sand", "big_rock"]
LOGR_C      = 0.316
SEED        = 42
N_BINS      = 10


# ── Cache loaders (same as seed_stability_test.py) ────────────────────────────

def load_dinov2_vits():
    tr = np.load(os.path.join(CACHE_DIR, "dinov2_train_1000shot.npz"))
    te = np.load(os.path.join(CACHE_DIR, "dinov2_test_1000shot.npz"))
    return tr["feats"], tr["labels"], te["feats"], te["labels"]


def load_dinov2_vitl():
    return (
        np.load(os.path.join(CACHE_DIR, "dinov2_vitl_train_1000_feats.npy")),
        np.load(os.path.join(CACHE_DIR, "dinov2_vitl_train_1000_labels.npy")),
        np.load(os.path.join(CACHE_DIR, "dinov2_vitl_test_287_feats.npy")),
        np.load(os.path.join(CACHE_DIR, "dinov2_vitl_test_287_labels.npy")),
    )


def load_ensemble_b():
    vitl_tr_f = np.load(os.path.join(CACHE_DIR, "dinov2_vitl_train_1000_feats.npy"))
    vitl_tr_l = np.load(os.path.join(CACHE_DIR, "dinov2_vitl_train_1000_labels.npy"))
    vitl_te_f = np.load(os.path.join(CACHE_DIR, "dinov2_vitl_test_287_feats.npy"))
    vitl_te_l = np.load(os.path.join(CACHE_DIR, "dinov2_vitl_test_287_labels.npy"))
    vitb_tr_f = np.load(os.path.join(CACHE_DIR, "dinov2_vitb_train_1000_feats.npy"))
    vitb_te_f = np.load(os.path.join(CACHE_DIR, "dinov2_vitb_test_287_feats.npy"))
    return (np.hstack([vitl_tr_f, vitb_tr_f]), vitl_tr_l,
            np.hstack([vitl_te_f, vitb_te_f]), vitl_te_l)


MODELS = [
    ("DINOv2 ViT-S/14",  load_dinov2_vits,  "#1f77b4"),
    ("DINOv2 ViT-L/14",  load_dinov2_vitl,  "#ff7f0e"),
    ("Ensemble B",        load_ensemble_b,   "#2ca02c"),
]


# ── Calibration helpers ───────────────────────────────────────────────────────

def get_confidences_and_correctness(train_feats, train_labels, test_feats, test_labels):
    """Train 1000-shot LogReg, return (max_prob per sample, correct bool per sample)."""
    rng = np.random.default_rng(SEED)
    idx = []
    for c in np.unique(train_labels):
        c_idx  = np.where(train_labels == c)[0]
        chosen = rng.choice(c_idx, size=min(1000, len(c_idx)), replace=False)
        idx.extend(chosen.tolist())

    X_tr = normalize(train_feats[np.array(idx)], norm="l2")
    y_tr = train_labels[np.array(idx)]
    X_te = normalize(test_feats, norm="l2")
    y_te = test_labels

    clf = LogisticRegression(C=LOGR_C, max_iter=1000, solver="lbfgs", random_state=SEED)
    clf.fit(X_tr, y_tr)

    probs   = clf.predict_proba(X_te)          # (287, 4)
    conf    = probs.max(axis=1)                # max predicted probability
    preds   = clf.predict(X_te)
    correct = (preds == y_te).astype(float)
    return conf, correct


def calibration_bins(conf, correct, n_bins=N_BINS):
    """Compute per-bin mean confidence, accuracy, and count."""
    bins   = np.linspace(0, 1, n_bins + 1)
    bin_conf, bin_acc, bin_n = [], [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (conf >= lo) & (conf < hi)
        if lo == bins[-2]:               # include right edge in last bin
            mask = (conf >= lo) & (conf <= hi)
        n = mask.sum()
        bin_n.append(n)
        if n > 0:
            bin_conf.append(conf[mask].mean())
            bin_acc.append(correct[mask].mean())
        else:
            bin_conf.append((lo + hi) / 2)
            bin_acc.append(float("nan"))
    return np.array(bin_conf), np.array(bin_acc), np.array(bin_n)


def ece(conf, correct, n_bins=N_BINS):
    """Expected Calibration Error (weighted average |acc - conf| per bin)."""
    n_total = len(conf)
    bins    = np.linspace(0, 1, n_bins + 1)
    ece_val = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (conf >= lo) & (conf < hi)
        if lo == bins[-2]:
            mask = (conf >= lo) & (conf <= hi)
        n = mask.sum()
        if n > 0:
            ece_val += (n / n_total) * abs(correct[mask].mean() - conf[mask].mean())
    return ece_val


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    fig.suptitle("Confidence Calibration — DINOv2 Linear Probe (1000-shot, n=287)",
                 fontsize=13, fontweight="bold")

    print(f"\n{'Model':<22}  {'ECE':>6}  {'Overall Acc':>12}  {'Mean Conf':>10}")
    print("─" * 58)

    for ax, (name, loader_fn, colour) in zip(axes, MODELS):
        train_f, train_l, test_f, test_l = loader_fn()
        conf, correct = get_confidences_and_correctness(train_f, train_l, test_f, test_l)

        bc, ba, bn = calibration_bins(conf, correct)
        ece_val    = ece(conf, correct)

        # Bar chart
        bar_w = 1.0 / N_BINS
        bars = ax.bar(bc, ba, width=bar_w * 0.85, color=colour, alpha=0.7,
                      label="Accuracy", align="center")

        # Perfect calibration diagonal
        ax.plot([0, 1], [0, 1], "k--", linewidth=1.2, label="Perfect calibration")

        # Gap shading (over/under confidence)
        for bconf, bacc, bcount in zip(bc, ba, bn):
            if np.isnan(bacc):
                continue
            if bacc < bconf:
                ax.fill_between([bconf - bar_w * 0.425, bconf + bar_w * 0.425],
                                [bacc, bacc], [bconf, bconf],
                                color="red", alpha=0.25)
            else:
                ax.fill_between([bconf - bar_w * 0.425, bconf + bar_w * 0.425],
                                [bconf, bconf], [bacc, bacc],
                                color="green", alpha=0.25)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("Confidence (max predicted prob)", fontsize=10)
        if ax == axes[0]:
            ax.set_ylabel("Accuracy", fontsize=10)
        ax.set_title(f"{name}\nECE = {ece_val:.4f}  |  Acc = {correct.mean()*100:.2f}%",
                     fontsize=10)
        ax.legend(fontsize=8, loc="upper left")
        ax.set_aspect("equal")

        # Count labels on bars
        for bconf, bacc, bcount in zip(bc, ba, bn):
            if bcount > 0 and not np.isnan(bacc):
                ax.text(bconf, min(bacc + 0.03, 0.98), f"n={bcount}",
                        ha="center", va="bottom", fontsize=6, color="black")

        print(f"{name:<22}  {ece_val:.4f}  {correct.mean()*100:>11.2f}%  "
              f"{conf.mean():>10.4f}")

    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "confidence_calibration.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nFigure saved → {out_path}")
    print("\nInterpretation guide:")
    print("  ECE < 0.05 = well-calibrated  |  ECE 0.05–0.10 = moderate  |  ECE > 0.10 = poor")
    print("  Red shading = over-confident  |  Green shading = under-confident")


if __name__ == "__main__":
    main()
