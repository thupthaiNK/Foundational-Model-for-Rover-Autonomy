"""
Purpose: E1 joint-domain probe experiment.
         Trains three linear probes on frozen DINOv2+reg ViT-S features:
           Probe A  — AI4Mars-only (NAVCAM, 4 classes)
           Probe B  — Mars-Bench-only (MastCam, 10 MB classes mapped to 4)
           Probe C  — Joint (A + B combined)
         Evaluates each probe on both test sets to measure how much
         cross-domain training reduces the 65.30 pp NAVCAM→MastCam gap.
Inputs:
  experiments/results/feature_cache/dinov2_reg_small_train_1000_feats.npy
  experiments/results/feature_cache/dinov2_reg_small_train_1000_labels.npy
  experiments/results/feature_cache/dinov2_reg_small_test_287_feats.npy
  experiments/results/feature_cache/dinov2_reg_small_test_287_labels.npy
  experiments/results/feature_cache/marsbench_train_feats.npy
  experiments/results/feature_cache/marsbench_train_labels.npy
  experiments/results/feature_cache/marsbench_test_feats.npy
  experiments/results/feature_cache/marsbench_test_labels.npy
Outputs:
  experiments/results/e1_joint_domain_probe_results.csv
  experiments/results/figures/e1_joint_domain_bar.png
How to run:
  python3 experiments/joint_domain_probe.py
Project: Onboard Visual Foundation Models for Mars Terrain Perception and Rover Navigation
"""

import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import normalize

# ── Paths ──────────────────────────────────────────────────────────────────────
REPO_ROOT   = Path(__file__).parent.parent
CACHE_DIR   = REPO_ROOT / "experiments/results/feature_cache"
OUTPUT_DIR  = REPO_ROOT / "experiments/results"
FIGURES_DIR = OUTPUT_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_CSV   = OUTPUT_DIR / "e1_joint_domain_probe_results.csv"
OUTPUT_FIG   = FIGURES_DIR / "e1_joint_domain_bar.png"

CLASS_NAMES = ["soil", "bedrock", "sand", "big_rock"]

# Matches Exp 10 and all other thesis experiments (C=0.316 ≈ 10^-0.5)
LOGR_C = 0.316

# MB class index → AI4Mars 4-class index
# Source: marsbench_zero_shot_transfer.py (same mapping used in §4.7.32)
MB_TO_AI4MARS = {
    13: 0,   # gro  → soil
    29: 2,   # san  → sand
    8:  2,   # dri  → sand (drift)
    9:  2,   # drh  → sand (drift high-albedo)
    10: 2,   # drp  → sand (drift ramp)
    11: 2,   # drt  → sand (drift top)
    28: 1,   # rrd  → bedrock
    30: 1,   # sco  → bedrock (soil with cobbles)
    12: 3,   # flr  → big_rock (float rock)
    16: 3,   # lar  → big_rock (large area rock)
}

# Reference values from thesis
ZERO_SHOT_REF   = 24.94   # AI4Mars probe on MB test (Exp §4.7.32)
AI4MARS_REF     = 90.24   # AI4Mars probe on AI4Mars test (Exp §4.7.X)
DOMAIN_GAP_REF  = 65.30   # pp gap established in thesis


# ── Data loading ───────────────────────────────────────────────────────────────

def load_ai4mars():
    """Load AI4Mars cached features (DINOv2+reg ViT-S, 1000-shot training)."""
    train_feats  = np.load(CACHE_DIR / "dinov2_reg_small_train_1000_feats.npy")
    train_labels = np.load(CACHE_DIR / "dinov2_reg_small_train_1000_labels.npy")
    test_feats   = np.load(CACHE_DIR / "dinov2_reg_small_test_287_feats.npy")
    test_labels  = np.load(CACHE_DIR / "dinov2_reg_small_test_287_labels.npy")
    return train_feats, train_labels, test_feats, test_labels


def load_marsbench_mapped():
    """
    Load Mars-Bench cached features and filter to the 10 classes that have
    unambiguous counterparts in the AI4Mars 4-class scheme.
    Returns features and 4-class remapped labels.
    """
    train_feats  = np.load(CACHE_DIR / "marsbench_train_feats.npy")
    # allow_pickle=True: safe — files are written by marsbench_dinov2_exp10.py in this repo
    # and contain a simple int64 array stored as an object array (numpy saved with str IDs nearby)
    train_labels = np.load(CACHE_DIR / "marsbench_train_labels.npy", allow_pickle=True)
    test_feats   = np.load(CACHE_DIR / "marsbench_test_feats.npy")
    test_labels  = np.load(CACHE_DIR / "marsbench_test_labels.npy", allow_pickle=True)

    mapped_keys = np.array(list(MB_TO_AI4MARS.keys()))

    train_mask   = np.isin(train_labels, mapped_keys)
    test_mask    = np.isin(test_labels,  mapped_keys)

    tr_feats  = train_feats[train_mask]
    tr_labels = np.array([MB_TO_AI4MARS[int(l)] for l in train_labels[train_mask]])
    te_feats  = test_feats[test_mask]
    te_labels = np.array([MB_TO_AI4MARS[int(l)] for l in test_labels[test_mask]])

    return tr_feats, tr_labels, te_feats, te_labels


# ── Probe training + evaluation ────────────────────────────────────────────────

def train_probe(feats: np.ndarray, labels: np.ndarray) -> LogisticRegression:
    X = normalize(feats, norm="l2")
    # Parameters match dinov2_terrain_test.py exactly (reproduces 90.24% baseline)
    clf = LogisticRegression(
        C=LOGR_C,
        max_iter=1000,
        random_state=42,
        multi_class="multinomial",
        solver="lbfgs",
    )
    t0 = time.time()
    clf.fit(X, labels)
    print(f"    trained in {time.time() - t0:.1f}s  |  n={len(labels)}")
    return clf


def eval_probe(clf: LogisticRegression, feats: np.ndarray, labels: np.ndarray,
               label_str: str) -> float:
    X    = normalize(feats, norm="l2")
    preds = clf.predict(X)
    acc   = accuracy_score(labels, preds) * 100
    present = sorted(np.unique(labels))
    target_names = [CLASS_NAMES[i] for i in present]
    report = classification_report(labels, preds, labels=present,
                                   target_names=target_names, zero_division=0)
    print(f"\n  [{label_str}]  accuracy = {acc:.2f}%")
    print(report)
    return acc


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("=" * 62)
    print("E1 Joint-Domain Probe — DINOv2+reg ViT-S")
    print("=" * 62)

    # ── Load data ──────────────────────────────────────────────────────────────
    print("\nLoading AI4Mars features ...")
    a4m_tr_f, a4m_tr_l, a4m_te_f, a4m_te_l = load_ai4mars()
    print(f"  train {a4m_tr_f.shape}  test {a4m_te_f.shape}")

    print("\nLoading Mars-Bench features (mapped to 4-class) ...")
    mb_tr_f, mb_tr_l, mb_te_f, mb_te_l = load_marsbench_mapped()
    print(f"  train {mb_tr_f.shape}  test {mb_te_f.shape}")

    # Joint = stack both training sets
    joint_tr_f = np.vstack([a4m_tr_f, mb_tr_f])
    joint_tr_l = np.concatenate([a4m_tr_l, mb_tr_l])
    print(f"\nJoint train {joint_tr_f.shape}")

    # ── Train three probes ─────────────────────────────────────────────────────
    print("\n── Probe A: AI4Mars-only ─────────────────────────────────────────")
    probe_a = train_probe(a4m_tr_f, a4m_tr_l)

    print("\n── Probe B: Mars-Bench-only ──────────────────────────────────────")
    probe_b = train_probe(mb_tr_f, mb_tr_l)

    print("\n── Probe C: Joint ────────────────────────────────────────────────")
    probe_c = train_probe(joint_tr_f, joint_tr_l)

    # ── Evaluate all probes on both test sets ──────────────────────────────────
    print("\n\n========== RESULTS ==========\n")

    # Load full MB test (all 1594 samples) for cross-check with thesis ref 24.94%
    mb_te_f_full  = np.load(CACHE_DIR / "marsbench_test_feats.npy")
    mb_te_l_full  = np.load(CACHE_DIR / "marsbench_test_labels.npy", allow_pickle=True)  # safe: own cache

    rows = []
    for name, probe in [("A4M-only", probe_a),
                         ("MB-only",  probe_b),
                         ("Joint",    probe_c)]:
        print(f"\n{'─'*40}")
        print(f"Probe: {name}")
        acc_a4m = eval_probe(probe, a4m_te_f, a4m_te_l, f"{name} → AI4Mars test")
        acc_mb  = eval_probe(probe, mb_te_f,  mb_te_l,  f"{name} → MB mapped test (810)")

        # "Full-dataset" style: predict all 1594 MB test samples; unmapped classes always wrong
        X_full = normalize(mb_te_f_full, norm="l2")
        preds_full = probe.predict(X_full)
        n_correct = 0
        for pred, lbl in zip(preds_full, mb_te_l_full):
            mapped_gt = MB_TO_AI4MARS.get(int(lbl), -1)
            if mapped_gt != -1 and pred == mapped_gt:
                n_correct += 1
        acc_mb_full = n_correct / len(mb_te_l_full) * 100

        gap     = acc_a4m - acc_mb
        rows.append({
            "probe":         name,
            "n_train":       (len(a4m_tr_l) if name == "A4M-only"
                              else len(mb_tr_l) if name == "MB-only"
                              else len(joint_tr_l)),
            "a4m_acc":       round(acc_a4m,    2),
            "mb_mapped_acc": round(acc_mb,     2),
            "mb_full_acc":   round(acc_mb_full, 2),
            "gap_mapped_pp": round(gap,         2),
        })

    # ── Summary table ──────────────────────────────────────────────────────────
    df = pd.DataFrame(rows)
    print("\n\n========== SUMMARY TABLE ==========\n")
    print(df.to_string(index=False))

    gap_a  = df.loc[df.probe == "A4M-only", "gap_mapped_pp"].values[0]
    gap_c  = df.loc[df.probe == "Joint",    "gap_mapped_pp"].values[0]
    reduction = gap_a - gap_c
    pct_reduction = reduction / gap_a * 100 if gap_a > 0 else 0

    ref_full_a4m = df.loc[df.probe == "A4M-only", "mb_full_acc"].values[0]
    print(f"\nNOTE: 'mb_full_acc' = accuracy on all 1594 MB test samples (unmapped → wrong)")
    print(f"      A4M-only mb_full_acc = {ref_full_a4m:.2f}%  (cf. thesis ref {ZERO_SHOT_REF:.2f}%)")
    print(f"      The 65.30 pp gap was computed with this full-dataset metric.")
    print(f"\nMapped-subset gap (810 of 1594 MB test samples):")
    print(f"  A4M-only gap:  {gap_a:.2f} pp")
    print(f"  Joint gap:     {gap_c:.2f} pp")
    print(f"  Reduction:     {reduction:.2f} pp  ({pct_reduction:.1f}% of A4M-only gap)")

    # ── Save CSV ───────────────────────────────────────────────────────────────
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved: {OUTPUT_CSV}")

    # ── Figure ─────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))

    probes = df["probe"].tolist()
    x = np.arange(len(probes))
    width = 0.32

    bars_a4m = ax.bar(x - width / 2, df["a4m_acc"],       width,
                      label="AI4Mars test (NAVCAM)", color="#4C72B0")
    bars_mb  = ax.bar(x + width / 2, df["mb_mapped_acc"], width,
                      label="Mars-Bench test — mapped subset (MastCam)", color="#DD8452")

    # Reference lines
    ax.axhline(AI4MARS_REF,  color="#4C72B0", linestyle="--",
               linewidth=1.2, alpha=0.6, label=f"A4M ref {AI4MARS_REF:.2f}%")
    ax.axhline(ZERO_SHOT_REF, color="#DD8452", linestyle="--",
               linewidth=1.2, alpha=0.6, label=f"Zero-shot ref {ZERO_SHOT_REF:.2f}%")

    # Value labels on bars
    for bar in list(bars_a4m) + list(bars_mb):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.8,
                f"{h:.1f}%", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(["Probe A\n(AI4Mars-only)", "Probe B\n(Mars-Bench-only)",
                         "Probe C\n(Joint)"], fontsize=11)
    ax.set_ylabel("Top-1 Accuracy (%)", fontsize=11)
    ax.set_title("E1: Joint-Domain Probe — DINOv2+reg ViT-S", fontsize=12)
    ax.set_ylim(0, 105)
    ax.legend(fontsize=9)
    sns.despine(ax=ax)

    plt.tight_layout()
    plt.savefig(OUTPUT_FIG, dpi=150, bbox_inches="tight")
    print(f"Saved: {OUTPUT_FIG}")
    plt.close()

    print("\nDone.")


if __name__ == "__main__":
    main()
