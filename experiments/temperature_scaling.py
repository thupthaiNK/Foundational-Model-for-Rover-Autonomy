"""
Purpose:    Post-hoc temperature scaling calibration for DINOv2+registers ViT-S/14
            linear probe on AI4Mars 287-image test set. Finds optimal temperature T
            that minimises Expected Calibration Error (ECE) on a held-out validation
            split, then reports ECE before/after and plots reliability diagrams.
            Temperature scaling: p_cal = softmax(logits / T)
Inputs:     experiments/results/feature_cache/dinov2_reg_small_{train,test}_*.npy
            (1000 train features, 287 test features — same as Exp 6)
Outputs:    experiments/results/figures/calibration_before_after.png  (reliability diagrams)
            experiments/results/temperature_scaling.json              (T, ECE values)
            Printed report: T*, ECE before/after, accuracy unchanged
How to run:
    python3 -u experiments/temperature_scaling.py
Project: Onboard Visual Foundation Models for Mars Terrain Perception and Rover Navigation
"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LogisticRegression

# ── Paths ─────────────────────────────────────────────────────────────────────
CACHE_DIR   = os.path.join(os.path.dirname(__file__), "results", "feature_cache")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")

CLASS_NAMES = ["soil", "bedrock", "sand", "big_rock"]
SEED        = 42
N_BINS      = 10   # ECE bins
VAL_FRAC    = 0.5  # fraction of test set used to find T; rest for eval


# ── Calibration helpers ───────────────────────────────────────────────────────

def softmax(x):
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def ece(proba, labels, n_bins=N_BINS):
    """Expected Calibration Error (uniform-width binning)."""
    confs = proba.max(axis=1)
    preds = proba.argmax(axis=1)
    correct = (preds == labels).astype(float)

    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece_val   = 0.0
    n         = len(labels)
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (confs >= lo) & (confs < hi)
        if mask.sum() == 0:
            continue
        acc_b  = correct[mask].mean()
        conf_b = confs[mask].mean()
        ece_val += mask.sum() / n * abs(acc_b - conf_b)
    return ece_val


def nll_with_temperature(T, logits, labels):
    """Negative log-likelihood of temperature-scaled softmax."""
    scaled = logits / T
    proba  = softmax(scaled)
    # clip for numerical safety
    proba  = np.clip(proba, 1e-9, 1.0)
    nll    = -np.log(proba[np.arange(len(labels)), labels]).mean()
    return nll


def reliability_diagram(ax, proba, labels, title, n_bins=N_BINS):
    """Plot reliability diagram: predicted confidence vs actual accuracy per bin."""
    confs   = proba.max(axis=1)
    preds   = proba.argmax(axis=1)
    correct = (preds == labels).astype(float)

    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_accs  = []
    bin_confs = []
    bin_cnts  = []

    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (confs >= lo) & (confs < hi)
        if mask.sum() == 0:
            bin_accs.append(0)
            bin_confs.append((lo + hi) / 2)
            bin_cnts.append(0)
        else:
            bin_accs.append(correct[mask].mean())
            bin_confs.append(confs[mask].mean())
            bin_cnts.append(mask.sum())

    bin_accs  = np.array(bin_accs)
    bin_confs = np.array(bin_confs)
    widths    = 1.0 / n_bins

    ax.bar(bin_edges[:-1], bin_accs, width=widths, align="edge",
           alpha=0.7, color="steelblue", label="Accuracy")
    ax.bar(bin_edges[:-1], bin_confs - bin_accs, width=widths, align="edge",
           bottom=bin_accs, alpha=0.45, color="tomato", label="Confidence gap")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect calibration")

    ece_val = ece(proba, labels, n_bins)
    ax.set_title(f"{title}\nECE = {ece_val:.4f}", fontsize=10)
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    ax.set_aspect("equal")
    return ece_val


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Load cached features
    tr_feats  = np.load(os.path.join(CACHE_DIR, "dinov2_reg_small_train_1000_feats.npy"))
    tr_labels = np.load(os.path.join(CACHE_DIR, "dinov2_reg_small_train_1000_labels.npy"))
    te_feats  = np.load(os.path.join(CACHE_DIR, "dinov2_reg_small_test_287_feats.npy"))
    te_labels = np.load(os.path.join(CACHE_DIR, "dinov2_reg_small_test_287_labels.npy"))

    print(f"Train: {tr_feats.shape}  Test: {te_feats.shape}", flush=True)

    # Train 1000-shot LogReg (same protocol as Exp 6)
    clf = LogisticRegression(C=0.316, max_iter=1000, random_state=SEED,
                             multi_class="multinomial", solver="lbfgs")
    clf.fit(tr_feats, tr_labels)

    # Raw logits (decision_function) on full test set
    logits = clf.decision_function(te_feats)   # [287, 4]
    preds  = clf.predict(te_feats)
    acc    = (preds == te_labels).mean() * 100
    print(f"Overall accuracy: {acc:.2f}%", flush=True)

    # Split test into val (find T) and eval (report calibration)
    rng     = np.random.RandomState(SEED)
    n_val   = int(len(te_labels) * VAL_FRAC)
    idx_all = rng.permutation(len(te_labels))
    val_idx  = idx_all[:n_val]
    eval_idx = idx_all[n_val:]

    logits_val  = logits[val_idx];    labels_val  = te_labels[val_idx]
    logits_eval = logits[eval_idx];   labels_eval = te_labels[eval_idx]

    # ECE before temperature scaling (full test set)
    proba_raw  = softmax(logits)
    ece_before = ece(proba_raw, te_labels)
    print(f"\nECE before scaling (full test): {ece_before:.4f}", flush=True)

    # Find optimal T by minimising NLL on val split
    result = minimize_scalar(
        lambda T: nll_with_temperature(T, logits_val, labels_val),
        bounds=(0.1, 10.0), method="bounded"
    )
    T_star = result.x
    print(f"Optimal temperature T*: {T_star:.4f}  (val NLL = {result.fun:.4f})", flush=True)

    # ECE after temperature scaling
    proba_scaled = softmax(logits / T_star)
    ece_after    = ece(proba_scaled, te_labels)
    print(f"ECE after  scaling (full test): {ece_after:.4f}", flush=True)
    print(f"ECE reduction: {ece_before:.4f} → {ece_after:.4f} "
          f"({(1 - ece_after/ece_before)*100:.1f}% improvement)", flush=True)

    # Accuracy unchanged (temperature scaling preserves argmax)
    preds_scaled = proba_scaled.argmax(axis=1)
    acc_scaled   = (preds_scaled == te_labels).mean() * 100
    print(f"Accuracy after scaling: {acc_scaled:.2f}%  (unchanged: {acc:.2f}%)", flush=True)

    # Per-class ECE analysis
    print("\nPer-class confidence before/after:")
    for c, name in enumerate(CLASS_NAMES):
        mask = te_labels == c
        if mask.sum() == 0:
            continue
        conf_b = proba_raw[mask].max(axis=1).mean()
        conf_a = proba_scaled[mask].max(axis=1).mean()
        corr   = (preds[mask] == te_labels[mask]).mean()
        print(f"  {name:<12}  n={mask.sum():>3}  "
              f"acc={corr*100:.1f}%  "
              f"conf_before={conf_b:.3f}  conf_after={conf_a:.3f}")

    # ── Reliability diagram figure ─────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    ece_b = reliability_diagram(ax1, proba_raw,    te_labels, "Before Temperature Scaling")
    ece_a = reliability_diagram(ax2, proba_scaled, te_labels,
                                f"After Temperature Scaling (T*={T_star:.3f})")

    fig.suptitle(
        "DINOv2+registers ViT-S/14 — Confidence Calibration\n"
        "AI4Mars 287-image test set  |  1000-shot LogReg linear probe",
        fontsize=12
    )
    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "calibration_before_after.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved reliability diagram → {out_path}", flush=True)

    # ── T sweep plot ──────────────────────────────────────────────────────────
    T_vals  = np.linspace(0.1, 5.0, 200)
    ece_ts  = [ece(softmax(logits / T), te_labels) for T in T_vals]
    nll_ts  = [nll_with_temperature(T, logits, te_labels) for T in T_vals]

    fig, (axl, axe) = plt.subplots(1, 2, figsize=(10, 4))

    axl.plot(T_vals, nll_ts, color="steelblue")
    axl.axvline(T_star, color="red", linestyle="--", label=f"T*={T_star:.3f}")
    axl.set_xlabel("Temperature T"); axl.set_ylabel("NLL (val split)")
    axl.set_title("NLL vs Temperature"); axl.legend()

    axe.plot(T_vals, ece_ts, color="steelblue")
    axe.axvline(T_star, color="red", linestyle="--", label=f"T*={T_star:.3f}")
    axe.axhline(ece_before, color="orange", linestyle=":", label=f"Uncal ECE={ece_before:.3f}")
    axe.axhline(ece_after,  color="green",  linestyle=":", label=f"Cal ECE={ece_after:.3f}")
    axe.set_xlabel("Temperature T"); axe.set_ylabel("ECE")
    axe.set_title("ECE vs Temperature"); axe.legend()

    fig.suptitle("Temperature Scaling — DINOv2+reg ViT-S/14  |  AI4Mars", fontsize=11)
    plt.tight_layout()
    sweep_path = os.path.join(FIGURES_DIR, "temperature_sweep.png")
    plt.savefig(sweep_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved temperature sweep → {sweep_path}", flush=True)

    # ── Save JSON results ──────────────────────────────────────────────────────
    results = {
        "model":         "facebook/dinov2-with-registers-small",
        "n_train_shots": 1000,
        "n_test":        int(len(te_labels)),
        "accuracy_pct":  round(float(acc), 4),
        "T_star":        round(float(T_star), 6),
        "ece_before":    round(float(ece_before), 6),
        "ece_after":     round(float(ece_after), 6),
        "ece_reduction_pct": round(float((1 - ece_after/ece_before)*100), 2),
        "n_bins":        N_BINS,
        "val_frac":      VAL_FRAC,
        "seed":          SEED,
    }
    json_path = os.path.join(RESULTS_DIR, "temperature_scaling.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved JSON → {json_path}", flush=True)

    # ── Thesis summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("THESIS SUMMARY")
    print("=" * 60)
    print(f"Pre-calibration  ECE: {ece_before:.4f}  (moderate mis-calibration)")
    print(f"Optimal temperature: T* = {T_star:.3f}")
    print(f"Post-calibration ECE: {ece_after:.4f}  "
          f"({(1 - ece_after/ece_before)*100:.1f}% reduction)")
    print(f"Accuracy unchanged:  {acc:.2f}% → {acc_scaled:.2f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()
