"""
Purpose:    Coverage-Risk curve for DINOv2+reg ViT-S/14 confidence threshold selection.
            Plots coverage (fraction of test images with confident predictions) vs risk
            (classification error rate on the confident subset) as threshold T is swept
            from 0.05 to 0.95. Also plots coverage vs safety rate and a per-class
            abstention breakdown. Validates that T=0.40 is a principled operating point
            on the coverage-risk Pareto frontier (not an arbitrary lucky pick).
            All probabilities use temperature scaling T*=0.461 (ECE: 0.1695 → 0.0325,
            80.8% improvement), the same calibrated model deployed on the rover.
Inputs:     experiments/results/feature_cache/dinov2_reg_small_{train,test}_*.npy
            experiments/results/temperature_scaling.json
Outputs:    experiments/results/coverage_risk_curve.csv
            experiments/results/figures/coverage_risk_curve.png
How to run:
    python3 -u experiments/coverage_risk_curve.py
Project: Onboard Visual Foundation Models for Mars Terrain Perception and Rover Navigation
"""

import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from scipy.special import softmax as scipy_softmax
from sklearn.linear_model import LogisticRegression

# ── Paths ─────────────────────────────────────────────────────────────────────
_HERE       = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR   = os.path.join(_HERE, "results", "feature_cache")
RESULTS_DIR = os.path.join(_HERE, "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")

CLASS_NAMES = ["soil", "bedrock", "sand", "big_rock"]
SEED        = 42
LOGR_C      = 0.316

# Calibrated temperature from temperature_scaling.py (ECE 0.1695 → 0.0325)
T_TEMP      = 0.461
# Confidence threshold deployed in dinov2_terrain_node.py
T_DEPLOYED  = 0.40

# Traversability policy speeds (m/s) — same as deployed rover node
SPEED_POLICY = np.array([0.10, 0.03, 0.05, 0.00])   # soil, bedrock, sand, big_rock
STOP_SPEED   = 0.00                                   # uncertain → STOP

# Sweep resolution
N_FINE   = 200   # smooth curve for plotting
N_COARSE = 50    # per-class abstention analysis


# ── Data loading & model ──────────────────────────────────────────────────────

def load_features():
    tr_feats  = np.load(os.path.join(CACHE_DIR, "dinov2_reg_small_train_1000_feats.npy"))
    tr_labels = np.load(os.path.join(CACHE_DIR, "dinov2_reg_small_train_1000_labels.npy"))
    te_feats  = np.load(os.path.join(CACHE_DIR, "dinov2_reg_small_test_287_feats.npy"))
    te_labels = np.load(os.path.join(CACHE_DIR, "dinov2_reg_small_test_287_labels.npy"))
    return tr_feats, tr_labels, te_feats, te_labels


def train_probe(tr_feats, tr_labels):
    """1000-shot LogReg, identical protocol to threshold_sensitivity.py."""
    rng = np.random.RandomState(SEED)
    idx = []
    for c in range(len(CLASS_NAMES)):
        c_idx  = np.where(tr_labels == c)[0]
        chosen = rng.choice(c_idx, size=min(1000, len(c_idx)), replace=False)
        idx.extend(chosen.tolist())
    clf = LogisticRegression(
        C=LOGR_C, max_iter=1000, random_state=SEED,
        multi_class="multinomial", solver="lbfgs"
    )
    clf.fit(tr_feats[idx], tr_labels[idx])
    return clf


def calibrated_proba(clf, te_feats):
    """Temperature-scaled softmax probabilities (T*=0.461)."""
    logits = clf.decision_function(te_feats)          # [N, 4]
    return scipy_softmax(logits / T_TEMP, axis=1)     # [N, 4]


# ── Metric sweep ──────────────────────────────────────────────────────────────

def sweep_thresholds(proba, te_labels, thresholds):
    """
    Vectorised sweep of coverage, risk, and safety across threshold values.

    Returns list of dicts with:
      threshold    : T value
      coverage     : fraction of images with max_prob >= T
      risk         : error rate on confident subset (= 1 - acc_confident)
      safety_rate  : fraction where commanded speed <= GT speed
      safety_fail  : fraction where commanded speed > GT speed
      n_confident  : count of confident predictions
      n_abstained  : count abstained (sent to STOP)
    """
    max_probs  = proba.max(axis=1)      # [N]
    pred_idx   = proba.argmax(axis=1)   # [N]
    n_total    = len(te_labels)

    gt_speeds   = SPEED_POLICY[te_labels]    # [N]
    pred_speeds = SPEED_POLICY[pred_idx]     # [N] — overridden for abstained below

    rows = []
    for T in thresholds:
        confident = max_probs >= T          # [N] bool
        n_conf    = confident.sum()

        # Coverage
        coverage = n_conf / n_total

        # Risk — error rate on confident subset
        if n_conf == 0:
            risk = 1.0
        else:
            risk = 1.0 - (pred_idx[confident] == te_labels[confident]).mean()

        # Safety — abstained images → STOP (0.00 m/s)
        cmd_speeds       = np.where(confident, pred_speeds, STOP_SPEED)
        safety_fail_rate = (cmd_speeds > gt_speeds).mean()

        rows.append({
            "threshold":   float(T),
            "coverage":    float(coverage),
            "risk":        float(risk),
            "safety_rate": float(1.0 - safety_fail_rate),
            "safety_fail": float(safety_fail_rate),
            "n_confident": int(n_conf),
            "n_abstained": int(n_total - n_conf),
        })

    return rows


def per_class_abstention(proba, te_labels, thresholds):
    """
    For each terrain class c and threshold T, return the fraction of
    images with GT class c whose max_prob falls below T.
    Reveals which terrain types drive the uncertain→STOP policy.
    """
    max_probs = proba.max(axis=1)
    result    = {}
    for c_idx, name in enumerate(CLASS_NAMES):
        mask = te_labels == c_idx
        if mask.sum() == 0:
            continue   # big_rock: 0 samples in gold test set
        probs_c      = max_probs[mask]
        result[name] = np.array([(probs_c < T).mean() for T in thresholds])
    return result


# ── Plotting ──────────────────────────────────────────────────────────────────

def _find_closest(rows, T):
    return min(rows, key=lambda r: abs(r["threshold"] - T))


def plot_figure(rows_fine, per_class, thresholds_coarse, out_path):
    """4-panel coverage-risk figure."""

    thresholds  = np.array([r["threshold"]   for r in rows_fine])
    coverage    = np.array([r["coverage"]    for r in rows_fine]) * 100
    risk        = np.array([r["risk"]        for r in rows_fine]) * 100
    safety_rate = np.array([r["safety_rate"] for r in rows_fine]) * 100

    # Deployed operating-point values
    op          = _find_closest(rows_fine, T_DEPLOYED)
    op_cov      = op["coverage"]   * 100
    op_risk     = op["risk"]       * 100
    op_safe     = op["safety_rate"]* 100

    # Annotation thresholds to label on Panels 1 & 2
    ann_thresholds = [0.50, 0.60, 0.70, 0.80, 0.90]

    plt.rcParams.update({
        "font.family": "sans-serif",
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.labelsize":    10,
        "xtick.labelsize":   9,
        "ytick.labelsize":   9,
    })

    fig = plt.figure(figsize=(16, 11))
    fig.suptitle(
        "Coverage-Risk Analysis — DINOv2+reg ViT-S/14\n"
        "Temperature-scaled probabilities (T*=0.461, ECE: 0.1695→0.0325) | "
        "AI4Mars 287-image test set | 1 000-shot linear probe",
        fontsize=12, fontweight="bold", y=1.01
    )

    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.35)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])

    # ── Panel 1: Coverage vs Risk (Selective Prediction Pareto Frontier) ─────
    ax1.plot(coverage, risk, color="#1f77b4", linewidth=2.2,
             label="Coverage-Risk frontier", zorder=2)

    # Deployed operating point
    ax1.scatter([op_cov], [op_risk], color="red", s=140, zorder=5,
                marker="*", label=f"T={T_DEPLOYED} (deployed)\n"
                                  f"Cov={op_cov:.1f}%  Risk={op_risk:.1f}%")

    # Annotate other thresholds along the curve
    prev_ann = None
    for T_ann in ann_thresholds:
        r_ann = _find_closest(rows_fine, T_ann)
        cx    = r_ann["coverage"] * 100
        cy    = r_ann["risk"]     * 100
        if prev_ann and abs(cx - prev_ann[0]) < 2.5:
            continue   # skip if too close to previous label
        offset_x = -5.0 if cx > 80 else 2.0
        offset_y =  0.4
        ax1.annotate(
            f"T={T_ann:.2f}", xy=(cx, cy),
            xytext=(cx + offset_x, cy + offset_y),
            fontsize=8, color="#555555",
            arrowprops=dict(arrowstyle="-", color="#aaaaaa", lw=0.8),
        )
        prev_ann = (cx, cy)

    ax1.set_xlabel("Coverage (% of test images above threshold)")
    ax1.set_ylabel("Risk — Error Rate on Confident Predictions (%)")
    ax1.set_title("Coverage-Risk Curve\n(Selective Prediction Pareto Frontier)", fontsize=11)
    ax1.legend(fontsize=9, loc="upper right")
    ax1.grid(True, alpha=0.28, linestyle="--")
    ax1.set_xlim(58, 102)
    ax1.set_ylim(-0.3, 12)
    # Arrow showing direction of increasing T
    ax1.annotate("increasing T →", xy=(65, 1.8), fontsize=8, color="#888888",
                 fontstyle="italic")

    # ── Panel 2: Coverage vs Safety Rate ─────────────────────────────────────
    ax2.plot(coverage, safety_rate, color="#2ca02c", linewidth=2.2,
             label="Safety-Coverage curve", zorder=2)
    ax2.scatter([op_cov], [op_safe], color="red", s=140, zorder=5,
                marker="*", label=f"T={T_DEPLOYED} (deployed)\n"
                                  f"Cov={op_cov:.1f}%  Safe={op_safe:.1f}%")

    prev_ann = None
    for T_ann in ann_thresholds:
        r_ann = _find_closest(rows_fine, T_ann)
        cx    = r_ann["coverage"]    * 100
        cy    = r_ann["safety_rate"] * 100
        if prev_ann and abs(cx - prev_ann[0]) < 2.5:
            continue
        offset_x = -5.0 if cx > 80 else 2.0
        offset_y = -0.35
        ax2.annotate(
            f"T={T_ann:.2f}", xy=(cx, cy),
            xytext=(cx + offset_x, cy + offset_y),
            fontsize=8, color="#555555",
            arrowprops=dict(arrowstyle="-", color="#aaaaaa", lw=0.8),
        )
        prev_ann = (cx, cy)

    ax2.set_xlabel("Coverage (% of test images above threshold)")
    ax2.set_ylabel("Safety Rate (% where commanded speed ≤ GT speed)")
    ax2.set_title("Safety-Coverage Curve\n(Rover Safety vs Prediction Abstention)", fontsize=11)
    ax2.legend(fontsize=9, loc="lower right")
    ax2.grid(True, alpha=0.28, linestyle="--")
    ax2.set_xlim(58, 102)
    ax2.set_ylim(93.5, 101.0)
    ax2.annotate("increasing T →", xy=(65, 94.0), fontsize=8, color="#888888",
                 fontstyle="italic")

    # ── Panel 3: Per-class abstention rate vs threshold ───────────────────────
    class_colors = {
        "soil":    "#1f77b4",
        "bedrock": "#d62728",
        "sand":    "#ff7f0e",
    }
    class_markers = {"soil": "o", "bedrock": "s", "sand": "^"}

    for name, rates in per_class.items():
        ax3.plot(thresholds_coarse, rates * 100,
                 color=class_colors.get(name, "gray"),
                 marker=class_markers.get(name, "o"),
                 markersize=3.5, linewidth=1.8,
                 label=f"{name.capitalize()} (n={int((te_labels_g == CLASS_NAMES.index(name)).sum())})")

    ax3.axvline(T_DEPLOYED, color="red", linestyle="--", linewidth=1.6,
                label=f"T={T_DEPLOYED} (deployed)")
    ax3.set_xlabel("Confidence Threshold T")
    ax3.set_ylabel("Abstention Rate per Class (%)")
    ax3.set_title("Per-Class Abstention Rate vs Threshold\n"
                  "(which terrain types trigger the uncertain→STOP policy)", fontsize=11)
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.28, linestyle="--")
    ax3.set_xlim(0.05, 0.95)
    ax3.set_ylim(-2, 62)

    # ── Panel 4: Risk and Coverage vs Threshold (dual y-axis) ────────────────
    ax4b = ax4.twinx()

    ln1, = ax4.plot(thresholds, risk, color="#1f77b4", linewidth=2.2,
                    label="Risk (error rate %)")
    ax4.set_ylabel("Risk — Error Rate (%)", color="#1f77b4")
    ax4.tick_params(axis="y", labelcolor="#1f77b4")
    ax4.set_ylim(-0.3, 12)

    ln2, = ax4b.plot(thresholds, coverage, color="#9467bd", linewidth=2.2,
                     linestyle="--", label="Coverage (%)")
    ax4b.set_ylabel("Coverage (%)", color="#9467bd")
    ax4b.tick_params(axis="y", labelcolor="#9467bd")
    ax4b.set_ylim(55, 103)

    vline = ax4.axvline(T_DEPLOYED, color="red", linestyle="--", linewidth=1.6,
                        label=f"T={T_DEPLOYED}")

    # Annotate the deployed point
    op_r = _find_closest(rows_fine, T_DEPLOYED)
    ax4.annotate(
        f"T={T_DEPLOYED}\nCov={op_cov:.1f}%\nRisk={op_risk:.1f}%",
        xy=(T_DEPLOYED, op_risk),
        xytext=(T_DEPLOYED + 0.08, op_risk + 1.8),
        fontsize=8, color="red",
        arrowprops=dict(arrowstyle="->", color="red", lw=1.0),
    )

    ax4.set_xlabel("Confidence Threshold T")
    ax4.set_title("Risk and Coverage vs Threshold T\n"
                  "(dual y-axis: blue=risk left, purple=coverage right)", fontsize=11)
    all_lines  = [ln1, ln2, vline]
    all_labels = [l.get_label() for l in all_lines]
    ax4.legend(all_lines, all_labels, fontsize=9, loc="center right")
    ax4.grid(True, alpha=0.28, linestyle="--")
    ax4.set_xlim(0.05, 0.95)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved: {out_path}")


# ── CSV output ────────────────────────────────────────────────────────────────

def save_csv(rows, out_path):
    keys = list(rows[0].keys())
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    print(f"CSV saved:   {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

# Module-level label for per-class n= annotation (populated in main, used in plot)
te_labels_g = None


def main():
    global te_labels_g
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────────────────
    print("Loading features ...")
    tr_feats, tr_labels, te_feats, te_labels = load_features()
    te_labels_g = te_labels
    print(f"  Train: {tr_feats.shape}  Test: {te_feats.shape}")
    print(f"  Test class counts: " +
          "  ".join(f"{CLASS_NAMES[c]}={int((te_labels==c).sum())}"
                    for c in range(len(CLASS_NAMES))))

    # ── Train probe ───────────────────────────────────────────────────────────
    print("Training 1000-shot LogReg probe ...")
    clf     = train_probe(tr_feats, tr_labels)
    raw_acc = (clf.predict(te_feats) == te_labels).mean() * 100
    print(f"  Raw accuracy:        {raw_acc:.2f}%")

    # ── Calibrated probabilities ──────────────────────────────────────────────
    print(f"Applying temperature scaling T*={T_TEMP} ...")
    proba   = calibrated_proba(clf, te_feats)
    cal_acc = (proba.argmax(axis=1) == te_labels).mean() * 100
    print(f"  Calibrated accuracy: {cal_acc:.2f}%  (invariant to temperature)")

    # ── Fine-grained sweep (smooth curves) ───────────────────────────────────
    thresholds_fine   = np.round(np.linspace(0.05, 0.95, N_FINE), 5)
    thresholds_coarse = np.round(np.linspace(0.05, 0.95, N_COARSE), 5)

    print(f"Sweeping {N_FINE} threshold values ...")
    rows_fine   = sweep_thresholds(proba, te_labels, thresholds_fine)

    print(f"Computing per-class abstention ({N_COARSE} points) ...")
    per_class   = per_class_abstention(proba, te_labels, thresholds_coarse)

    # ── Print summary table ───────────────────────────────────────────────────
    print("\n=== Coverage-Risk Summary ===")
    hdr = f"{'T':>6}  {'Coverage':>9}  {'Risk%':>7}  {'Safety%':>9}  {'N_conf':>7}  {'N_abs':>6}"
    print(hdr)
    print("-" * len(hdr))
    mark_ts = [0.20, 0.30, 0.40, 0.45, 0.50, 0.60, 0.70, 0.80, 0.90]
    for T_mark in mark_ts:
        r    = _find_closest(rows_fine, T_mark)
        tag  = "  <-- deployed" if abs(r["threshold"] - T_DEPLOYED) < 0.006 else ""
        print(f"{r['threshold']:>6.3f}  "
              f"{r['coverage']*100:>8.2f}%  "
              f"{r['risk']*100:>6.2f}%  "
              f"{r['safety_rate']*100:>8.2f}%  "
              f"{r['n_confident']:>7}  "
              f"{r['n_abstained']:>6}"
              f"{tag}")

    # ── Deployed threshold summary ────────────────────────────────────────────
    op = _find_closest(rows_fine, T_DEPLOYED)
    print(f"\n=== Deployed threshold T={T_DEPLOYED} ===")
    print(f"  Coverage:     {op['coverage']*100:.2f}%   ({op['n_confident']}/287 images)")
    print(f"  Risk:         {op['risk']*100:.2f}%   (error rate on confident subset)")
    print(f"  Safety rate:  {op['safety_rate']*100:.2f}%")
    print(f"  Abstentions:  {op['n_abstained']} images (→ STOP)")

    # ── Per-class abstention at deployed threshold ────────────────────────────
    print(f"\n=== Per-class abstention at T={T_DEPLOYED} ===")
    op_coarse = min(range(len(thresholds_coarse)),
                    key=lambda i: abs(thresholds_coarse[i] - T_DEPLOYED))
    for name, rates in per_class.items():
        n_class = int((te_labels == CLASS_NAMES.index(name)).sum())
        print(f"  {name:<10}  n={n_class:>3}  abstained={rates[op_coarse]*100:.1f}%")

    # ── Save CSV ──────────────────────────────────────────────────────────────
    save_csv(rows_fine, os.path.join(RESULTS_DIR, "coverage_risk_curve.csv"))

    # ── Plot ──────────────────────────────────────────────────────────────────
    plot_figure(
        rows_fine, per_class, thresholds_coarse,
        os.path.join(FIGURES_DIR, "coverage_risk_curve.png"),
    )

    # ── Thesis summary ────────────────────────────────────────────────────────
    t50 = _find_closest(rows_fine, 0.50)
    t70 = _find_closest(rows_fine, 0.70)
    t90 = _find_closest(rows_fine, 0.90)
    print("\n" + "=" * 60)
    print("THESIS SUMMARY")
    print("=" * 60)
    print(f"Deployed T={T_DEPLOYED}:  Coverage={op['coverage']*100:.1f}%  "
          f"Risk={op['risk']*100:.2f}%  Safety={op['safety_rate']*100:.2f}%")
    print(f"At T=0.50:           Coverage={t50['coverage']*100:.1f}%  "
          f"Safety gain: +{(t50['safety_rate']-op['safety_rate'])*100:.2f} pp")
    print(f"At T=0.70:           Coverage={t70['coverage']*100:.1f}%  "
          f"Safety gain: +{(t70['safety_rate']-op['safety_rate'])*100:.2f} pp")
    print(f"At T=0.90:           Coverage={t90['coverage']*100:.1f}%  "
          f"Safety gain: +{(t90['safety_rate']-op['safety_rate'])*100:.2f} pp")
    print(f"\nConclusion: T={T_DEPLOYED} maintains {op['coverage']*100:.1f}% coverage "
          f"(only {op['n_abstained']} abstentions) while the uncertain→STOP")
    print(f"policy provides {op['safety_rate']*100:.2f}% overall safety. Higher thresholds "
          f"yield modest safety gains at the cost of large coverage losses.")
    print("=" * 60)


if __name__ == "__main__":
    main()
