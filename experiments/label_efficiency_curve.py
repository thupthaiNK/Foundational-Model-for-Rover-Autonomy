"""
Purpose:    Label efficiency curve for DINOv2+reg ViT-S/14 on AI4Mars terrain.
            Sweeps N training labels per class from 1 to 108 (big_rock limit),
            training a frozen linear probe at each N. Quantifies how many labelled
            Mars terrain images are needed before performance saturates, motivating
            the 1000-shot (effectively full-data) protocol used throughout the thesis.
Inputs:
  experiments/results/feature_cache/dinov2_reg_small_train_1000_feats.npy
  experiments/results/feature_cache/dinov2_reg_small_train_1000_labels.npy
  experiments/results/feature_cache/dinov2_reg_small_test_287_feats.npy
  experiments/results/feature_cache/dinov2_reg_small_test_287_labels.npy
Outputs:
  experiments/results/label_efficiency_curve.csv
  experiments/results/figures/label_efficiency_curve.png
How to run:
  python3 -u experiments/label_efficiency_curve.py
Project: Onboard Visual Foundation Models for Mars Terrain Perception and Rover Navigation
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import normalize

# ── Paths ──────────────────────────────────────────────────────────────────────
REPO_ROOT   = Path(__file__).parent.parent
CACHE_DIR   = REPO_ROOT / "experiments/results/feature_cache"
OUTPUT_DIR  = REPO_ROOT / "experiments/results"
FIGURES_DIR = OUTPUT_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_CSV = OUTPUT_DIR / "label_efficiency_curve.csv"
OUTPUT_FIG = FIGURES_DIR / "label_efficiency_curve.png"

CLASS_NAMES   = ["soil", "bedrock", "sand", "big_rock"]
N_CLASS       = 4
RANDOM_CHANCE = 100.0 / N_CLASS       # 25%
DEPLOYED_ACC  = 89.90                  # authoritative 1000-shot result (§4.7.20)
LOGR_C        = 0.316
N_SEEDS       = 5

# Balanced sweep: N per class (big_rock always capped at 108)
N_SWEEP_BALANCED = sorted(set([1, 2, 3, 5, 8, 10, 15, 20, 30, 50, 75, 100, 108]))

# Practical sweep: vary soil/bedrock/sand (big_rock fixed at all 108 samples)
# Shows how accuracy grows toward the deployed 89.90% as annotation increases
N_SWEEP_PRACTICAL = sorted(set([1, 2, 5, 10, 20, 50, 100, 108, 200, 300, 500, 750, 1000]))


def train_eval(tr_f, tr_l, te_f, te_l, n_per_class: int, seed: int,
               n_bigrock: int = None):
    """Sample n_per_class per class, train probe, return accuracy + per-class acc.
    If n_bigrock is set, big_rock uses that many samples instead of n_per_class.
    """
    rng = np.random.default_rng(seed)
    idx = []
    for c in range(N_CLASS):
        c_idx = np.where(tr_l == c)[0]
        n_avail = len(c_idx)
        n_draw = min(n_bigrock if (c == 3 and n_bigrock is not None) else n_per_class,
                     n_avail)
        chosen = rng.choice(c_idx, size=n_draw, replace=False)
        idx.extend(chosen.tolist())
    idx = np.array(idx)

    X_tr = normalize(tr_f[idx], norm="l2")
    y_tr = tr_l[idx]
    X_te = normalize(te_f, norm="l2")

    clf = LogisticRegression(
        C=LOGR_C, max_iter=1000, random_state=seed,
        multi_class="multinomial", solver="lbfgs",
    )
    clf.fit(X_tr, y_tr)
    preds = clf.predict(X_te)

    overall = accuracy_score(te_l, preds) * 100
    per_cls = {}
    for c, name in enumerate(CLASS_NAMES):
        mask = te_l == c
        if mask.sum() > 0:
            per_cls[name] = (preds[mask] == te_l[mask]).mean() * 100
        else:
            per_cls[name] = float("nan")

    return overall, per_cls


def main():
    print("=" * 62)
    print("A3: Label Efficiency Curve — DINOv2+reg ViT-S/14")
    print("=" * 62)

    tr_f = np.load(CACHE_DIR / "dinov2_reg_small_train_1000_feats.npy")
    tr_l = np.load(CACHE_DIR / "dinov2_reg_small_train_1000_labels.npy")
    te_f = np.load(CACHE_DIR / "dinov2_reg_small_test_287_feats.npy")
    te_l = np.load(CACHE_DIR / "dinov2_reg_small_test_287_labels.npy")

    print(f"\nTrain cache: {tr_f.shape}  Test: {te_f.shape}")
    for c, name in enumerate(CLASS_NAMES):
        print(f"  train {name}: {(tr_l==c).sum()}  test {name}: {(te_l==c).sum()}")

    def run_sweep(n_values, fixed_bigrock=None, label="balanced"):
        rows = []
        for n in n_values:
            accs = []
            cls_accs = {name: [] for name in CLASS_NAMES}
            for seed in range(N_SEEDS):
                overall, per_cls = train_eval(
                    tr_f, tr_l, te_f, te_l, n, seed, n_bigrock=fixed_bigrock)
                accs.append(overall)
                for name in CLASS_NAMES:
                    if not np.isnan(per_cls[name]):
                        cls_accs[name].append(per_cls[name])
            mean_acc = np.mean(accs)
            std_acc  = np.std(accs)
            print(f"  [{label}] N={n:>4}  acc={mean_acc:.2f}% ±{std_acc:.2f}")
            row = {"sweep": label, "n_per_class": n,
                   "mean_acc": round(mean_acc, 4), "std_acc": round(std_acc, 4)}
            for name in CLASS_NAMES:
                vals = cls_accs[name]
                row[f"mean_{name}"] = round(np.mean(vals), 4) if vals else float("nan")
                row[f"std_{name}"]  = round(np.std(vals),  4) if vals else float("nan")
            rows.append(row)
        return pd.DataFrame(rows)

    print("\n── Balanced sweep (N per class, big_rock capped at 108) ─────")
    df_bal = run_sweep(N_SWEEP_BALANCED, fixed_bigrock=None, label="balanced")

    print("\n── Practical sweep (soil/bedrock/sand vary, big_rock=108) ───")
    df_prac = run_sweep(N_SWEEP_PRACTICAL, fixed_bigrock=108, label="practical")

    df = pd.concat([df_bal, df_prac], ignore_index=True)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved: {OUTPUT_CSV}")

    # ── Figure ─────────────────────────────────────────────────────────────────
    fig, (ax_main, ax_cls) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        "Label Efficiency — DINOv2+reg ViT-S/14 on AI4Mars\n"
        "Frozen linear probe | 287-image gold test | 5 seeds per N",
        fontsize=12, fontweight="bold",
    )

    df_b  = df_bal
    # Practical sweep: only N >= 108 (avoids class-imbalance artifact at low N)
    df_p  = df_prac[df_prac["n_per_class"] >= 108].copy()

    ns_b  = df_b["n_per_class"].values
    acc_b = df_b["mean_acc"].values
    std_b = df_b["std_acc"].values
    ns_p  = df_p["n_per_class"].values
    acc_p = df_p["mean_acc"].values
    std_p = df_p["std_acc"].values

    # Combined curve: balanced (N≤108) + practical (N≥108), join at N=108
    ns_combined  = np.concatenate([ns_b, ns_p[ns_p > 108]])
    acc_combined = np.concatenate([acc_b, acc_p[ns_p > 108]])
    std_combined = np.concatenate([std_b, std_p[ns_p > 108]])

    # ── Panel 1: overall accuracy — combined curve ─────────────────────────────
    ax_main.fill_between(ns_combined, acc_combined - std_combined,
                         acc_combined + std_combined,
                         alpha=0.18, color="#4472C4", label="_ci")
    ax_main.plot(ns_combined, acc_combined, "o-", color="#4472C4", linewidth=2.2,
                 markersize=5, label="DINOv2+reg ViT-S/14 (mean ± std, 5 seeds)")

    # Transition marker at N=108 (big_rock data ceiling)
    ax_main.axvline(108, color="#70AD47", linewidth=1.5, linestyle="--",
                    alpha=0.85, label="big_rock data ceiling (n=108)")
    ax_main.text(112, 28, "← balanced\n    regime",
                 fontsize=7.5, color="#70AD47", ha="left")
    ax_main.text(112, 18, "  practical\n  regime →",
                 fontsize=7.5, color="#70AD47", ha="left")

    ax_main.axhline(DEPLOYED_ACC, color="#C00000", linewidth=1.5, linestyle="--",
                    label=f"Deployed 1000-shot: {DEPLOYED_ACC:.2f}% (§4.7.20)")
    ax_main.axhline(RANDOM_CHANCE, color="#888888", linewidth=1.0, linestyle=":",
                    label=f"Random baseline: {RANDOM_CHANCE:.1f}%")

    # Annotate key points
    acc_10  = float(df_b[df_b["n_per_class"] == 10]["mean_acc"].iloc[0])
    acc_108 = float(df_b[df_b["n_per_class"] == 108]["mean_acc"].iloc[0])
    acc_1000 = float(df_prac[df_prac["n_per_class"] == 1000]["mean_acc"].iloc[0])
    ax_main.annotate(f"{acc_10:.1f}%\n@N=10",
                     xy=(10, acc_10), xytext=(6, acc_10 + 10),
                     fontsize=7.5, color="#4472C4",
                     arrowprops=dict(arrowstyle="->", color="#4472C4", lw=0.9))
    ax_main.scatter([1000], [acc_1000], s=120, color="#C00000", zorder=5, marker="*")
    ax_main.annotate(f"{acc_1000:.2f}%\n@N=1000",
                     xy=(1000, acc_1000), xytext=(500, acc_1000 - 12),
                     fontsize=7.5, color="#C00000",
                     arrowprops=dict(arrowstyle="->", color="#C00000", lw=0.9))

    ax_main.set_xscale("log")
    ax_main.set_xlim(0.8, 1300)
    ax_main.set_ylim(0, 105)
    ax_main.set_xlabel("Training labels per class (log scale)", fontsize=11)
    ax_main.set_ylabel("Overall accuracy on AI4Mars test (%)", fontsize=11)
    ax_main.set_title("Overall Label Efficiency\n(N≤108: all 4 classes balanced; N>108: soil/bedrock/sand only)",
                      fontsize=10, pad=8)
    ax_main.legend(fontsize=8, loc="lower right")
    ax_main.grid(alpha=0.3)
    ax_main.set_xticks([1, 5, 10, 50, 108, 500, 1000])
    ax_main.set_xticklabels(["1", "5", "10", "50", "108", "500", "1000"])

    # ── Panel 2: per-class accuracy (balanced sweep, N=1 to 108) ─────────────
    cls_colours = {"soil": "#2E75B6", "bedrock": "#C55A11",
                   "sand": "#70AD47", "big_rock": "#7030A0"}
    cls_markers = {"soil": "o", "bedrock": "s", "sand": "^", "big_rock": "D"}

    for name in CLASS_NAMES:
        col = f"mean_{name}"
        if col not in df_b.columns:
            continue
        y_cls = df_b[col].values
        s_cls = df_b[f"std_{name}"].values
        valid = ~np.isnan(y_cls)
        if valid.sum() == 0:
            label_txt = f"{name.replace('_', ' ').title()} (n=0 in test, N/A)"
            ax_cls.plot([], [], marker=cls_markers[name], linewidth=1.8,
                        color=cls_colours[name], linestyle="--", label=label_txt)
            continue
        ax_cls.fill_between(ns_b[valid], y_cls[valid] - s_cls[valid],
                            y_cls[valid] + s_cls[valid],
                            alpha=0.12, color=cls_colours[name])
        ax_cls.plot(ns_b[valid], y_cls[valid],
                    marker=cls_markers[name], linewidth=1.8, markersize=5,
                    color=cls_colours[name],
                    label=name.replace("_", " ").title())

    ax_cls.axhline(RANDOM_CHANCE, color="#888888", linewidth=1.0, linestyle=":",
                   label="Random baseline")
    ax_cls.axvline(108, color="#70AD47", linewidth=1.5, linestyle="--", alpha=0.85,
                   label="big_rock data ceiling")
    ax_cls.set_xscale("log")
    ax_cls.set_xlim(0.8, 130)
    ax_cls.set_ylim(0, 105)
    ax_cls.set_xlabel("Training labels per class (log scale)", fontsize=11)
    ax_cls.set_ylabel("Per-class accuracy on AI4Mars test (%)", fontsize=11)
    ax_cls.set_title("Per-Class Efficiency (balanced N/class)", fontsize=11, pad=8)
    ax_cls.legend(fontsize=8.5, loc="lower right")
    ax_cls.grid(alpha=0.3)
    ax_cls.set_xticks([1, 2, 5, 10, 20, 50, 100, 108])
    ax_cls.set_xticklabels(["1", "2", "5", "10", "20", "50", "100", "108"])

    plt.tight_layout()
    plt.savefig(OUTPUT_FIG, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUTPUT_FIG}")

    # ── Key findings printout ──────────────────────────────────────────────────
    def get_acc(dframe, n):
        rows = dframe[dframe["n_per_class"] == n]
        return float(rows["mean_acc"].iloc[0]) if len(rows) else float("nan")

    print("\n" + "=" * 62)
    print("KEY FINDINGS — Balanced sweep (N per class, big_rock ≤ 108):")
    for n in [1, 10, 50, 100, 108]:
        print(f"  N={n:>4}/class:  acc = {get_acc(df_bal, n):.2f}%")
    print(f"\nPractical sweep (soil/bedrock/sand vary, big_rock=108):")
    for n in [10, 50, 100, 200, 500, 1000]:
        a = get_acc(df_prac, n)
        print(f"  N={n:>4}/class:  acc = {a:.2f}%")
    print(f"\n  Deployed 1000-shot (authoritative): {DEPLOYED_ACC}%")
    print(f"  Practical N=1000 reproduces: {get_acc(df_prac, 1000):.2f}%")
    print("=" * 62)


if __name__ == "__main__":
    main()
