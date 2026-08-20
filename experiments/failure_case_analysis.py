"""
Purpose:    Systematic analysis of DINOv2+registers ViT-S/14 failure cases on the AI4Mars
            287-image gold-standard test set. Identifies which images are misclassified,
            visualises them in a grid, and reports confusion patterns for the thesis.
            Uses cached features (no GPU inference needed) + 1000-shot LogReg probe.
Inputs:     experiments/results/feature_cache/dinov2_reg_small_{train,test}_*.npy
            AI4Mars test images (paths reconstructed in same sorted order as cache)
Outputs:    experiments/results/figures/failure_cases_grid.png  (misclassified images grid)
            experiments/results/failure_case_analysis.csv       (per-image prediction table)
            Printed summary with confusion pattern breakdown
How to run:
    python3 -u experiments/failure_case_analysis.py
Project: Onboard Visual Foundation Models for Mars Terrain Perception and Rover Navigation
"""

import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression

# ── Paths ─────────────────────────────────────────────────────────────────────
AI4MARS_BASE = "/mnt/c/Users/DELL/Desktop/Thesis/github source/ai4mars-dataset-merged-0.1"
IMAGES_DIR   = os.path.join(AI4MARS_BASE, "msl/images/edr")
TEST_LABELS  = os.path.join(AI4MARS_BASE, "msl/labels/test/masked-gold-min3-100agree")
CACHE_DIR    = os.path.join(os.path.dirname(__file__), "results", "feature_cache")
RESULTS_DIR  = os.path.join(os.path.dirname(__file__), "results")
FIGURES_DIR  = os.path.join(RESULTS_DIR, "figures")

CLASS_NAMES = ["soil", "bedrock", "sand", "big_rock"]
IGNORE_PX   = 255
SEED        = 42
MAX_GRID    = 20   # max misclassified examples to show in grid


# ── Helpers ───────────────────────────────────────────────────────────────────

def dominant_class(label_path):
    lbl = np.array(Image.open(label_path))
    valid = lbl[lbl != IGNORE_PX]
    if len(valid) == 0:
        return None
    return int(np.argmax(np.bincount(valid, minlength=4)))


def load_test_pairs():
    """Reconstruct test image list in same order as cached features (sorted alphabetical)."""
    pairs = []
    for fname in sorted(os.listdir(TEST_LABELS)):
        if not fname.endswith(".png"):
            continue
        stem     = fname.replace("_merged.png", "").replace(".png", "")
        img_path = os.path.join(IMAGES_DIR, stem + ".JPG")
        lbl_path = os.path.join(TEST_LABELS, fname)
        if not os.path.exists(img_path):
            continue
        gt = dominant_class(lbl_path)
        if gt is not None:
            pairs.append((img_path, gt))
    return pairs


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

    # 1000-shot LogReg (same protocol as Exp 6)
    clf = LogisticRegression(C=0.316, max_iter=1000, random_state=SEED,
                             multi_class="multinomial", solver="lbfgs")
    clf.fit(tr_feats, tr_labels)

    preds      = clf.predict(te_feats)
    proba      = clf.predict_proba(te_feats)
    confidence = proba.max(axis=1)

    overall_acc = (preds == te_labels).mean() * 100
    print(f"\nOverall accuracy: {overall_acc:.2f}%  "
          f"({(preds == te_labels).sum()}/{len(te_labels)} correct)", flush=True)

    # Reconstruct image paths
    pairs = load_test_pairs()
    assert len(pairs) == len(te_labels), \
        f"Mismatch: {len(pairs)} pairs vs {len(te_labels)} labels in cache"

    # ── Confusion breakdown ────────────────────────────────────────────────────
    print("\nConfusion matrix (row=GT, col=Pred):", flush=True)
    n_classes = len(CLASS_NAMES)
    conf_mat  = np.zeros((n_classes, n_classes), dtype=int)
    for gt, pred in zip(te_labels, preds):
        conf_mat[gt, pred] += 1

    gt_pred = "GT\\Pred"
    header = f"{gt_pred:<12}" + "".join(f"{c:>12}" for c in CLASS_NAMES)
    print(header)
    for i, row in enumerate(conf_mat):
        line = f"{CLASS_NAMES[i]:<12}" + "".join(f"{v:>12}" for v in row)
        print(line)

    # ── Failure cases ─────────────────────────────────────────────────────────
    wrong_idx = [i for i, (p, g) in enumerate(zip(preds, te_labels)) if p != g]
    print(f"\nTotal failures: {len(wrong_idx)} / {len(te_labels)}"
          f"  ({len(wrong_idx)/len(te_labels)*100:.1f}%)", flush=True)

    # Sort by confidence (most confidently wrong first — most interesting failures)
    wrong_idx.sort(key=lambda i: -confidence[i])

    # Pattern counts
    patterns = {}
    for i in wrong_idx:
        key = f"{CLASS_NAMES[te_labels[i]]}→{CLASS_NAMES[preds[i]]}"
        patterns[key] = patterns.get(key, 0) + 1

    print("\nFailure patterns (GT→Pred):")
    for k, v in sorted(patterns.items(), key=lambda x: -x[1]):
        print(f"  {k:<25} {v:>3}  ({v/len(wrong_idx)*100:.1f}%)")

    # Per-class analysis
    print("\nPer-class failure rates:")
    for c in range(n_classes):
        c_idx  = [i for i, g in enumerate(te_labels) if g == c]
        c_fail = [i for i in c_idx if preds[i] != te_labels[i]]
        if len(c_idx) > 0:
            print(f"  {CLASS_NAMES[c]:<12} {len(c_fail):>3}/{len(c_idx):>3} "
                  f"  ({len(c_fail)/len(c_idx)*100:.1f}% error rate)")

    # Confidence of failures vs correct
    correct_conf = confidence[preds == te_labels]
    wrong_conf   = confidence[preds != te_labels]
    print(f"\nMean confidence — correct: {correct_conf.mean():.3f}  "
          f"wrong: {wrong_conf.mean():.3f}")
    print(f"  Failures with confidence >0.9: "
          f"{(wrong_conf > 0.9).sum()}  "
          f"({(wrong_conf > 0.9).mean()*100:.0f}% of failures)")

    # ── Save CSV ───────────────────────────────────────────────────────────────
    csv_path = os.path.join(RESULTS_DIR, "failure_case_analysis.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "image_name", "gt_class", "pred_class",
                          "confidence", "correct",
                          "p_soil", "p_bedrock", "p_sand", "p_big_rock"])
        for i, (img_path, gt) in enumerate(pairs):
            name = os.path.basename(img_path)
            row  = [i, name, CLASS_NAMES[gt], CLASS_NAMES[preds[i]],
                    f"{confidence[i]:.4f}", int(preds[i] == gt)]
            row += [f"{proba[i, c]:.4f}" for c in range(n_classes)]
            writer.writerow(row)
    print(f"\nSaved CSV → {csv_path}", flush=True)

    # ── Grid figure: most-confidently-wrong images ─────────────────────────────
    n_show  = min(MAX_GRID, len(wrong_idx))
    if n_show == 0:
        print("No failures to visualise.", flush=True)
        return

    ncols = 5
    nrows = (n_show + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3.4 * nrows))
    axes = axes.flatten() if nrows > 1 else [axes] if ncols == 1 else axes.flatten()

    for k, idx in enumerate(wrong_idx[:n_show]):
        ax       = axes[k]
        img_path = pairs[idx][0]
        gt_name  = CLASS_NAMES[te_labels[idx]]
        pd_name  = CLASS_NAMES[preds[idx]]
        conf_val = confidence[idx]

        try:
            img = Image.open(img_path).convert("RGB").resize((224, 224), Image.LANCZOS)
            ax.imshow(img)
        except Exception:
            ax.text(0.5, 0.5, "load error", ha="center", va="center",
                    transform=ax.transAxes)

        ax.set_title(
            f"GT: {gt_name}\nPred: {pd_name}  ({conf_val:.2f})",
            fontsize=8, color="red"
        )
        ax.axis("off")

    # Hide unused panels
    for k in range(n_show, len(axes)):
        axes[k].axis("off")

    fig.suptitle(
        f"DINOv2+reg ViT-S — Top-{n_show} Most-Confidently-Wrong Failures\n"
        f"AI4Mars 287-image test set  |  1000-shot LogReg  |  Sorted by confidence ↓",
        fontsize=11, y=1.01
    )

    out_path = os.path.join(FIGURES_DIR, "failure_cases_grid.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved grid → {out_path}", flush=True)

    # ── Pattern summary for thesis ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("THESIS SUMMARY")
    print("=" * 60)
    print(f"DINOv2+reg ViT-S achieves {overall_acc:.2f}% on 287-image test set.")
    print(f"Total failures: {len(wrong_idx)} images.")
    top_pattern = max(patterns, key=patterns.get)
    print(f"Dominant failure pattern: {top_pattern} "
          f"({patterns[top_pattern]}/{len(wrong_idx)} = "
          f"{patterns[top_pattern]/len(wrong_idx)*100:.0f}% of errors).")
    print(f"High-confidence failures (conf>0.9): {(wrong_conf > 0.9).sum()} images.")
    print("=" * 60)


if __name__ == "__main__":
    main()
