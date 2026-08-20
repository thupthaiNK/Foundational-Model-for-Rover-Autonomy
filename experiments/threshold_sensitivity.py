"""
Purpose:    Confidence threshold sensitivity study for DINOv2+reg ViT-S/14 traversability.
            Sweeps threshold T from 0.10 to 0.90 in 0.05 steps and reports, at each T:
              - Terrain accuracy on confident predictions (max_prob >= T)
              - Overall accuracy (uncertain→wrong, so abstentions count as errors)
              - Abstention rate (% images below threshold → STOP)
              - Safety fail rate (% big_rock/bedrock images sent at non-STOP speed)
            Generates a 4-panel figure and CSV of the sweep results.
            Validates T*=0.40 (chosen from temperature scaling ECE analysis).
Inputs:     experiments/results/feature_cache/dinov2_reg_small_{train,test}_*.npy
            (1000-shot train features, 287 test features — same as Exp 1–4)
Outputs:    experiments/results/threshold_sensitivity.csv
            experiments/results/figures/threshold_sensitivity.png
How to run:
    python3 -u experiments/threshold_sensitivity.py
Project: Onboard Visual Foundation Models for Mars Terrain Perception and Rover Navigation
"""

import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from scipy.special import softmax as scipy_softmax

# ── Paths ─────────────────────────────────────────────────────────────────────
_HERE       = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR   = os.path.join(_HERE, "results", "feature_cache")
RESULTS_DIR = os.path.join(_HERE, "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")

CLASS_NAMES = ["soil", "bedrock", "sand", "big_rock"]
SEED        = 42
LOGR_C      = 0.316
T_STAR      = 0.461   # temperature from Exp 4.7.27 (ECE 0.1695 → 0.0325)
T_CHOSEN    = 0.40    # controller threshold validated in thesis

# Traversability policy: speed in m/s
# Traversability policy: speed in m/s
POLICY = {"soil": 0.10, "bedrock": 0.03, "sand": 0.05, "big_rock": 0.00, "uncertain": 0.00}

# NOTE: the AI4Mars gold test set (masked-gold-min3-100agree) contains 0 big_rock images
# (rare class, hard to annotate with 100% agreement) — 3-class benchmark only.
# Safety is therefore measured as: commanded_speed <= gt_speed (no overspeed),
# i.e. we never drive faster than the ground truth terrain demands.


def load_features():
    train_feats  = np.load(os.path.join(CACHE_DIR, "dinov2_reg_small_train_1000_feats.npy"))
    train_labels = np.load(os.path.join(CACHE_DIR, "dinov2_reg_small_train_1000_labels.npy"))
    test_feats   = np.load(os.path.join(CACHE_DIR, "dinov2_reg_small_test_287_feats.npy"))
    test_labels  = np.load(os.path.join(CACHE_DIR, "dinov2_reg_small_test_287_labels.npy"))
    return train_feats, train_labels, test_feats, test_labels


def train_probe(train_feats, train_labels):
    rng = np.random.RandomState(SEED)
    idx = []
    for c in range(len(CLASS_NAMES)):
        c_idx  = np.where(train_labels == c)[0]
        chosen = rng.choice(c_idx, size=min(1000, len(c_idx)), replace=False)
        idx.extend(chosen.tolist())
    clf = LogisticRegression(
        C=LOGR_C, max_iter=1000, random_state=SEED,
        multi_class="multinomial", solver="lbfgs"
    )
    clf.fit(train_feats[idx], train_labels[idx])
    return clf


def calibrated_proba(clf, test_feats, temperature: float):
    """Return calibrated softmax probabilities using temperature scaling."""
    logits = clf.decision_function(test_feats)          # [N, 4]
    return scipy_softmax(logits / temperature, axis=1)  # [N, 4]


def evaluate_threshold(proba, test_labels, threshold: float):
    """
    Returns dict of metrics for a given confidence threshold.
    Images below threshold are treated as 'uncertain' → STOP (0.00 m/s).
    """
    max_probs = proba.max(axis=1)          # [N]
    pred_idx  = proba.argmax(axis=1)       # [N]

    n_total    = len(test_labels)
    confident  = max_probs >= threshold    # [N] boolean mask
    abstained  = ~confident

    # Terrain accuracy on confident subset only
    correct_confident = 0
    n_confident = confident.sum()
    for i in np.where(confident)[0]:
        if pred_idx[i] == test_labels[i]:
            correct_confident += 1
    acc_confident = (correct_confident / n_confident * 100) if n_confident > 0 else float("nan")

    # Overall accuracy: abstentions count as wrong
    correct_overall = correct_confident
    acc_overall     = correct_overall / n_total * 100

    # Abstention rate
    abstention_rate = abstained.sum() / n_total * 100

    # Safety rate: % images where commanded speed <= ground truth speed.
    # Abstentions → STOP (0.00 m/s) — always conservative (never overspeed).
    # A safety fail = we commanded a HIGHER speed than ground truth terrain requires.
    safety_fails = 0
    for i in range(n_total):
        gt_class   = CLASS_NAMES[test_labels[i]]
        gt_speed   = POLICY[gt_class]
        if abstained[i]:
            pred_speed = 0.00   # uncertain → STOP
        else:
            pred_speed = POLICY[CLASS_NAMES[pred_idx[i]]]
        if pred_speed > gt_speed:
            safety_fails += 1

    safety_fail_rate = safety_fails / n_total * 100
    safety_rate      = 100.0 - safety_fail_rate

    # Safety fails disaggregated by source class
    bedrock_idx     = np.where(test_labels == CLASS_NAMES.index("bedrock"))[0]
    bedrock_fails   = 0
    for i in bedrock_idx:
        gt_speed = POLICY["bedrock"]
        if abstained[i]:
            pred_speed = 0.00
        else:
            pred_speed = POLICY[CLASS_NAMES[pred_idx[i]]]
        if pred_speed > gt_speed:
            bedrock_fails += 1
    bedrock_safe_rate = (1.0 - bedrock_fails / len(bedrock_idx)) * 100 if len(bedrock_idx) > 0 else float("nan")

    # Big-rock: 0 images in gold test set — track for completeness
    bigrock_idx    = np.where(test_labels == CLASS_NAMES.index("big_rock"))[0]
    bigrock_safety = float("nan") if len(bigrock_idx) == 0 else (
        sum(1 for i in bigrock_idx if abstained[i] or CLASS_NAMES[pred_idx[i]] == "big_rock")
        / len(bigrock_idx) * 100
    )

    return {
        "threshold":         threshold,
        "n_total":           n_total,
        "n_confident":       int(n_confident),
        "n_abstained":       int(abstained.sum()),
        "abstention_rate":   round(abstention_rate, 2),
        "acc_confident":     round(acc_confident, 2),
        "acc_overall":       round(acc_overall, 2),
        "safety_rate":       round(safety_rate, 2),
        "safety_fail_rate":  round(safety_fail_rate, 2),
        "bedrock_safe_rate": round(bedrock_safe_rate, 2),
        "bigrock_safety":    round(bigrock_safety, 2) if not np.isnan(bigrock_safety) else float("nan"),
    }


def plot_results(rows, out_path: str):
    thresholds       = [r["threshold"] for r in rows]
    acc_confident    = [r["acc_confident"] for r in rows]
    acc_overall      = [r["acc_overall"] for r in rows]
    abstention       = [r["abstention_rate"] for r in rows]
    safety           = [r["safety_rate"] for r in rows]
    bedrock_safe     = [r["bedrock_safe_rate"] for r in rows]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle(
        "Confidence Threshold Sensitivity — DINOv2+reg ViT-S/14\n"
        "AI4Mars 287-image test set (soil/bedrock/sand), 1000-shot LogReg, T=0.461 temp scaling",
        fontsize=11
    )

    def vline(ax):
        ax.axvline(T_CHOSEN, color="red", linestyle="--", linewidth=1.5,
                   label=f"T*={T_CHOSEN} (deployed)")
        ax.legend(fontsize=9)
        ax.set_xlabel("Confidence threshold")
        ax.grid(True, alpha=0.3)

    # Panel 1: accuracy
    ax = axes[0, 0]
    ax.plot(thresholds, acc_confident, "b-o", markersize=4, label="Accuracy (confident only)")
    ax.plot(thresholds, acc_overall,   "b--s", markersize=4, label="Accuracy (all 287)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Terrain Classification Accuracy")
    ax.set_ylim(50, 102)
    vline(ax)

    # Panel 2: abstention rate
    ax = axes[0, 1]
    ax.plot(thresholds, abstention, "m-o", markersize=4)
    ax.set_ylabel("Abstention rate (%)")
    ax.set_title("Abstention Rate (uncertain → STOP)")
    ax.set_ylim(0, 50)
    vline(ax)

    # Panel 3: overall safety rate
    ax = axes[1, 0]
    ax.plot(thresholds, safety, "g-o", markersize=4)
    ax.set_ylabel("Safety rate (%)")
    ax.set_title("Overall Safety Rate\n(% images: commanded speed ≤ GT speed)")
    ax.set_ylim(50, 102)
    vline(ax)

    # Panel 4: bedrock-specific safety (hardest class)
    ax = axes[1, 1]
    ax.plot(thresholds, bedrock_safe, "r-o", markersize=4)
    ax.set_ylabel("Bedrock safety rate (%)")
    ax.set_title("Bedrock Safety Rate\n(hardest class, 92/287 test images, 15.2% error)")
    ax.set_ylim(50, 102)
    vline(ax)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved: {out_path}")


def print_table(rows):
    hdr = f"{'T':>6}  {'Acc(conf)':>10}  {'Acc(all)':>9}  {'Abstain%':>9}  {'Safety%':>8}  {'BedSafe%':>9}  {'N_conf':>7}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        mark = " <-- deployed" if abs(r["threshold"] - T_CHOSEN) < 1e-9 else ""
        print(
            f"{r['threshold']:>6.2f}  "
            f"{r['acc_confident']:>10.2f}  "
            f"{r['acc_overall']:>9.2f}  "
            f"{r['abstention_rate']:>9.2f}  "
            f"{r['safety_rate']:>8.2f}  "
            f"{r['bedrock_safe_rate']:>9.2f}  "
            f"{r['n_confident']:>7}"
            f"{mark}"
        )


def save_csv(rows, out_path: str):
    keys = list(rows[0].keys())
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    print(f"CSV saved: {out_path}")


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)

    print("Loading pre-extracted features ...")
    train_feats, train_labels, test_feats, test_labels = load_features()
    print(f"  Train: {train_feats.shape}  Test: {test_feats.shape}")

    print("Training 1000-shot LogReg probe ...")
    clf = train_probe(train_feats, train_labels)
    raw_acc = (clf.predict(test_feats) == test_labels).mean() * 100
    print(f"  Raw (uncalibrated) accuracy: {raw_acc:.2f}%")

    print(f"Applying temperature scaling T={T_STAR} ...")
    proba = calibrated_proba(clf, test_feats, T_STAR)
    cal_acc = (proba.argmax(axis=1) == test_labels).mean() * 100
    print(f"  Calibrated accuracy: {cal_acc:.2f}%  (accuracy invariant to temperature)")

    thresholds = np.round(np.arange(0.10, 0.91, 0.05), 2)
    rows = []
    for t in thresholds:
        rows.append(evaluate_threshold(proba, test_labels, float(t)))

    print("\n=== Threshold Sensitivity Results ===")
    print_table(rows)

    save_csv(rows, os.path.join(RESULTS_DIR, "threshold_sensitivity.csv"))
    plot_results(rows, os.path.join(FIGURES_DIR, "threshold_sensitivity.png"))

    # Print chosen-threshold summary
    chosen = next(r for r in rows if abs(r["threshold"] - T_CHOSEN) < 1e-9)
    print(f"\n=== Deployed threshold T*={T_CHOSEN} ===")
    print(f"  Terrain accuracy (confident subset):  {chosen['acc_confident']:.2f}%")
    print(f"  Terrain accuracy (all 287):            {chosen['acc_overall']:.2f}%")
    print(f"  Abstention rate (uncertain → STOP):   {chosen['abstention_rate']:.2f}%")
    print(f"  Overall safety rate (no overspeed):   {chosen['safety_rate']:.2f}%")
    print(f"  Bedrock safety rate (hardest class):  {chosen['bedrock_safe_rate']:.2f}%")
    print()
    print("  NOTE: AI4Mars gold test set has 0 big_rock images (100%-agreement mask).")
    print("  Big-rock safety in Gazebo is reported separately (Exp 4.7.X, 5-zone).")


if __name__ == "__main__":
    main()
