"""
Purpose:    Per-class F1 comparison — Probe A (AI4Mars-only) vs Probe C (Joint).
            Measures whether cross-domain training specifically improves big_rock
            detection in the Mars-Bench (MastCam) visual domain — the class most
            critical for Gazebo traversability safety.
            Both probes use frozen DINOv2+reg ViT-S/14 384-d CLS features.
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
  experiments/results/per_class_f1_probe.csv
  experiments/results/figures/per_class_f1_probe.png
How to run:
  python3 -u experiments/per_class_f1_probe.py
Project: Onboard Visual Foundation Models for Mars Terrain Perception and Rover Navigation
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import normalize

# ── Paths ──────────────────────────────────────────────────────────────────────
REPO_ROOT   = Path(__file__).parent.parent
CACHE_DIR   = REPO_ROOT / "experiments/results/feature_cache"
OUTPUT_DIR  = REPO_ROOT / "experiments/results"
FIGURES_DIR = OUTPUT_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_CSV = OUTPUT_DIR / "per_class_f1_probe.csv"
OUTPUT_FIG = FIGURES_DIR / "per_class_f1_probe.png"

CLASS_NAMES = ["soil", "bedrock", "sand", "big_rock"]

LOGR_C = 0.316  # matches all thesis experiments (10^-0.5)

MB_TO_AI4MARS = {
    13: 0, 29: 2, 8: 2, 9: 2, 10: 2, 11: 2, 28: 1, 30: 1, 12: 3, 16: 3,
}


# ── Data loading ───────────────────────────────────────────────────────────────

def load_ai4mars():
    tr_f = np.load(CACHE_DIR / "dinov2_reg_small_train_1000_feats.npy")
    tr_l = np.load(CACHE_DIR / "dinov2_reg_small_train_1000_labels.npy")
    te_f = np.load(CACHE_DIR / "dinov2_reg_small_test_287_feats.npy")
    te_l = np.load(CACHE_DIR / "dinov2_reg_small_test_287_labels.npy")
    return tr_f, tr_l, te_f, te_l


def load_marsbench_mapped():
    tr_f = np.load(CACHE_DIR / "marsbench_train_feats.npy")
    tr_l = np.load(CACHE_DIR / "marsbench_train_labels.npy", allow_pickle=True)  # safe: own cache
    te_f = np.load(CACHE_DIR / "marsbench_test_feats.npy")
    te_l = np.load(CACHE_DIR / "marsbench_test_labels.npy", allow_pickle=True)   # safe: own cache

    mapped_keys = np.array(list(MB_TO_AI4MARS.keys()))
    tr_mask = np.isin(tr_l, mapped_keys)
    te_mask = np.isin(te_l, mapped_keys)

    tr_f4 = tr_f[tr_mask]
    tr_l4 = np.array([MB_TO_AI4MARS[int(l)] for l in tr_l[tr_mask]])
    te_f4 = te_f[te_mask]
    te_l4 = np.array([MB_TO_AI4MARS[int(l)] for l in te_l[te_mask]])

    return tr_f4, tr_l4, te_f4, te_l4


# ── Probe ──────────────────────────────────────────────────────────────────────

def train_probe(feats, labels):
    clf = LogisticRegression(
        C=LOGR_C, max_iter=1000, random_state=42,
        multi_class="multinomial", solver="lbfgs",
    )
    clf.fit(normalize(feats, norm="l2"), labels)
    return clf


def eval_per_class(clf, feats, labels):
    """Return per-class precision/recall/F1 dict and accuracy."""
    preds = clf.predict(normalize(feats, norm="l2"))
    acc = accuracy_score(labels, preds)
    present = sorted(np.unique(labels))
    target_names = [CLASS_NAMES[i] for i in present]
    report = classification_report(
        labels, preds, labels=present, target_names=target_names,
        zero_division=0, output_dict=True,
    )
    # Inject accuracy under a consistent key (classification_report may omit it
    # when labels= restricts the class set while predictions span all 4 classes)
    report["accuracy"] = acc
    return report, preds


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("=" * 62)
    print("A2: Per-class F1 — Probe A vs Probe C (AI4Mars / Joint)")
    print("=" * 62)

    # Load features
    a4m_tr_f, a4m_tr_l, a4m_te_f, a4m_te_l = load_ai4mars()
    mb_tr_f, mb_tr_l, mb_te_f, mb_te_l = load_marsbench_mapped()

    print(f"\nAI4Mars  train {a4m_tr_f.shape}  test {a4m_te_f.shape}")
    print(f"MB mapped train {mb_tr_f.shape}  test {mb_te_f.shape}")
    for c, name in enumerate(CLASS_NAMES):
        print(f"  MB train {name}: {(mb_tr_l==c).sum()}  |  MB test {name}: {(mb_te_l==c).sum()}")

    joint_tr_f = np.vstack([a4m_tr_f, mb_tr_f])
    joint_tr_l = np.concatenate([a4m_tr_l, mb_tr_l])

    # Train probes
    print("\nTraining Probe A (AI4Mars-only) ...")
    probe_a = train_probe(a4m_tr_f, a4m_tr_l)

    print("Training Probe C (Joint) ...")
    probe_c = train_probe(joint_tr_f, joint_tr_l)

    # Evaluate on AI4Mars test
    print("\n── AI4Mars test set (287 samples) ───────────────────────────")
    a4m_report_a, _ = eval_per_class(probe_a, a4m_te_f, a4m_te_l)
    a4m_report_c, _ = eval_per_class(probe_c, a4m_te_f, a4m_te_l)
    print(f"  Probe A accuracy: {a4m_report_a['accuracy']*100:.2f}%")
    print(f"  Probe C accuracy: {a4m_report_c['accuracy']*100:.2f}%")
    for cls in CLASS_NAMES:
        if cls in a4m_report_a:
            f1_a = a4m_report_a[cls]['f1-score']
            f1_c = a4m_report_c[cls]['f1-score']
            sup  = int(a4m_report_a[cls]['support'])
            print(f"  {cls:<10} A={f1_a:.3f}  C={f1_c:.3f}  (n={sup})")

    # Evaluate on MB mapped test
    print("\n── Mars-Bench mapped test set (810 samples) ─────────────────")
    mb_report_a, preds_a = eval_per_class(probe_a, mb_te_f, mb_te_l)
    mb_report_c, preds_c = eval_per_class(probe_c, mb_te_f, mb_te_l)
    print(f"  Probe A accuracy: {mb_report_a['accuracy']*100:.2f}%")
    print(f"  Probe C accuracy: {mb_report_c['accuracy']*100:.2f}%")
    for cls in CLASS_NAMES:
        if cls in mb_report_a:
            f1_a = mb_report_a[cls]['f1-score']
            f1_c = mb_report_c[cls]['f1-score']
            prec_a = mb_report_a[cls]['precision']
            prec_c = mb_report_c[cls]['precision']
            rec_a  = mb_report_a[cls]['recall']
            rec_c  = mb_report_c[cls]['recall']
            sup    = int(mb_report_a[cls]['support'])
            delta  = f1_c - f1_a
            print(f"  {cls:<10} A-F1={f1_a:.3f} C-F1={f1_c:.3f}  Δ={delta:+.3f}  (n={sup})")

    # ── Build CSV ──────────────────────────────────────────────────────────────
    rows = []
    for probe_name, mb_rep, a4m_rep in [
        ("Probe A (AI4Mars-only)", mb_report_a, a4m_report_a),
        ("Probe C (Joint)",        mb_report_c, a4m_report_c),
    ]:
        for cls in CLASS_NAMES:
            # MB test
            if cls in mb_rep:
                rows.append({
                    "probe": probe_name,
                    "test_set": "MB_mapped",
                    "class": cls,
                    "precision": round(mb_rep[cls]["precision"], 4),
                    "recall":    round(mb_rep[cls]["recall"],    4),
                    "f1":        round(mb_rep[cls]["f1-score"],  4),
                    "support":   int(mb_rep[cls]["support"]),
                })
            else:
                rows.append({
                    "probe": probe_name, "test_set": "MB_mapped",
                    "class": cls, "precision": 0.0, "recall": 0.0,
                    "f1": 0.0, "support": 0,
                })
            # AI4Mars test
            if cls in a4m_rep:
                rows.append({
                    "probe": probe_name,
                    "test_set": "AI4Mars",
                    "class": cls,
                    "precision": round(a4m_rep[cls]["precision"], 4),
                    "recall":    round(a4m_rep[cls]["recall"],    4),
                    "f1":        round(a4m_rep[cls]["f1-score"],  4),
                    "support":   int(a4m_rep[cls]["support"]),
                })

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved: {OUTPUT_CSV}")

    # ── Figure ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        "Per-Class F1 — Probe A (AI4Mars-only) vs Probe C (Joint Training)\n"
        "DINOv2+reg ViT-S/14 | Linear probe | 384-d CLS features",
        fontsize=12, fontweight="bold", y=1.01,
    )

    COLOURS = {"Probe A (AI4Mars-only)": "#4472C4", "Probe C (Joint)": "#ED7D31"}

    for ax, test_set, title in [
        (axes[0], "MB_mapped",
         "Mars-Bench Mapped Test (810 samples)\n"
         "[Unseen domain for Probe A — MastCam RGB, 36-class]"),
        (axes[1], "AI4Mars",
         "AI4Mars Gold Test (287 samples)\n"
         "[Same domain as Probe A — NAVCAM grayscale]"),
    ]:
        sub = df[df["test_set"] == test_set]
        n_cls = len(CLASS_NAMES)
        x = np.arange(n_cls)
        w = 0.35

        for i, (probe_name, colour) in enumerate(COLOURS.items()):
            f1_vals = []
            for cls in CLASS_NAMES:
                row = sub[(sub["probe"] == probe_name) & (sub["class"] == cls)]
                f1_vals.append(float(row["f1"].iloc[0]) if len(row) else 0.0)
            bars = ax.bar(x + (i - 0.5) * w, f1_vals, w,
                          label=probe_name, color=colour, alpha=0.88, zorder=3)
            for bar, val in zip(bars, f1_vals):
                if val > 0.02:
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 0.015,
                            f"{val:.3f}", ha="center", va="bottom",
                            fontsize=7.5, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(["Soil", "Bedrock", "Sand", "Big Rock"], fontsize=10)
        ax.set_ylabel("F1 Score", fontsize=10)
        ax.set_ylim(0, 1.12)
        ax.set_title(title, fontsize=10, pad=8)
        ax.legend(fontsize=8.5, loc="upper right")
        ax.grid(axis="y", alpha=0.3, zorder=0)
        ax.axhline(0, color="black", linewidth=0.5)

        # Highlight big_rock bars with a shaded box
        ax.axvspan(n_cls - 1 - 0.55, n_cls - 1 + 0.55,
                   alpha=0.10, color="red", zorder=0, label="_nolegend_")
        ax.text(n_cls - 1, 1.08, "⚠ safety-critical", ha="center",
                fontsize=7.5, color="darkred", style="italic")

        # Support annotation
        for c_idx, cls in enumerate(CLASS_NAMES):
            sup_row = sub[(sub["probe"] == "Probe A (AI4Mars-only)") & (sub["class"] == cls)]
            sup_val = int(sup_row["support"].iloc[0]) if len(sup_row) else 0
            ax.text(c_idx, -0.07, f"n={sup_val}", ha="center",
                    fontsize=6.5, color="#555555")

    # Big-rock improvement annotation on MB panel
    mb_sub = df[df["test_set"] == "MB_mapped"]
    br_a = float(mb_sub[(mb_sub["probe"] == "Probe A (AI4Mars-only)") &
                        (mb_sub["class"] == "big_rock")]["f1"].iloc[0])
    br_c = float(mb_sub[(mb_sub["probe"] == "Probe C (Joint)") &
                        (mb_sub["class"] == "big_rock")]["f1"].iloc[0])
    delta = br_c - br_a
    axes[0].annotate(
        f"Δ big_rock = {delta:+.3f}\n({br_a:.3f} → {br_c:.3f})",
        xy=(3 + 0.175, br_c + 0.03), fontsize=8, color="darkred",
        ha="left", va="bottom",
        bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="darkred", alpha=0.85),
    )

    plt.tight_layout()
    plt.savefig(OUTPUT_FIG, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUTPUT_FIG}")

    # ── Summary ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 62)
    print("KEY RESULT (MB mapped test — unseen domain):")
    print(f"  big_rock  Probe A F1 = {br_a:.3f}")
    print(f"  big_rock  Probe C F1 = {br_c:.3f}  (Δ = {delta:+.3f})")
    overall_a = mb_report_a["accuracy"] * 100
    overall_c = mb_report_c["accuracy"] * 100
    print(f"\n  Overall   Probe A = {overall_a:.2f}%  Probe C = {overall_c:.2f}%")
    print(f"  Δ overall = {overall_c - overall_a:+.2f} pp")
    print("=" * 62)


if __name__ == "__main__":
    main()
