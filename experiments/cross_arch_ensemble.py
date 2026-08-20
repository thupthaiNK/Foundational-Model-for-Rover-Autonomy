"""
Purpose:    Cross-architecture feature ensemble: concatenate DINOv3 ViT-L/16 (1024-d) and
            DINOv2 ViT-B/14 (768-d) features into a 1792-d combined representation, then
            train a LogReg linear probe on the full 1000-shot training cache.
            Hypothesis: DINOv3 ViT-L provides stronger multi-scale texture features
            while DINOv2 ViT-B adds fine-grained 14px patch-level geological detail.
            All feature caches already exist — no GPU inference required.
Inputs:     experiments/results/feature_cache/
              dinov3_vitl_{train,test}_*.npy       (1024-d per image)
              dinov2_vitb_{train,test}_*.npy       (768-d per image)
Outputs:    experiments/results/cross_arch_ensemble_results.csv
            Printed per-class accuracy vs Ensemble B (DINOv2 ViT-L + ViT-B, 94.43%)
How to run:
    python3 -u experiments/cross_arch_ensemble.py
Project: Onboard Visual Foundation Models for Mars Terrain Perception and Rover Navigation
"""

import csv
import os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize

CACHE_DIR   = os.path.join(os.path.dirname(__file__), "results", "feature_cache")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
CLASS_NAMES = ["soil", "bedrock", "sand", "big_rock"]
SEED        = 42

# Ensemble B (existing best result) for comparison
ENSEMBLE_B  = {"overall": 94.43, "soil": None, "bedrock": 91.32, "sand": 94.44}


def load_concat(prefixes, split, n):
    """Load and L2-normalise each prefix, then concatenate."""
    parts  = []
    labels = None
    for p in prefixes:
        feats = np.load(os.path.join(CACHE_DIR, f"{p}_{split}_{n}_feats.npy"))
        lbl   = np.load(os.path.join(CACHE_DIR, f"{p}_{split}_{n}_labels.npy"))
        parts.append(normalize(feats))  # re-normalise each modality independently
        if labels is None:
            labels = lbl
    combined = np.concatenate(parts, axis=1)
    return combined, labels


def per_class_acc(preds, labels):
    result = {}
    for i, name in enumerate(CLASS_NAMES):
        mask = labels == i
        if mask.sum() == 0:
            result[name] = None
        else:
            result[name] = (preds[mask] == labels[mask]).mean() * 100
    return result


def run_ensemble(name, prefixes, train_tag, test_tag):
    tr_feats, tr_labels = load_concat(prefixes, "train", train_tag)
    te_feats, te_labels = load_concat(prefixes, "test",  test_tag)

    clf = LogisticRegression(C=0.316, max_iter=1000, random_state=SEED,
                             multi_class="multinomial", solver="lbfgs")
    clf.fit(tr_feats, tr_labels)
    preds    = clf.predict(te_feats)
    overall  = (preds == te_labels).mean() * 100
    per_cls  = per_class_acc(preds, te_labels)

    print(f"\n{'='*60}")
    print(f"Ensemble: {name}  ({tr_feats.shape[1]}-d)")
    print(f"{'='*60}")
    print(f"Overall:  {overall:.2f}%")
    for c in CLASS_NAMES:
        v = per_cls.get(c)
        if v is not None:
            print(f"  {c:<12} {v:.2f}%")
    print(f"\nvs Ensemble B (DINOv2 ViT-L + ViT-B, 94.43%):")
    print(f"  Overall:  {overall - ENSEMBLE_B['overall']:+.2f}%")
    return overall, per_cls, tr_feats.shape[1]


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    results = []

    # Experiment A: DINOv3 ViT-L + DINOv2 ViT-B  (1024 + 768 = 1792-d)
    oa, pc, dim = run_ensemble(
        "DINOv3 ViT-L + DINOv2 ViT-B",
        ["dinov3_vitl", "dinov2_vitb"],
        "1000", "287"
    )
    results.append(("DINOv3_ViT-L + DINOv2_ViT-B", dim, oa, pc))

    # Experiment B: DINOv3 ViT-L + DINOv2+reg ViT-B  (1024 + 768 = 1792-d)
    oa2, pc2, dim2 = run_ensemble(
        "DINOv3 ViT-L + DINOv2+reg ViT-B",
        ["dinov3_vitl", "dinov2_reg_base"],
        "1000", "287"
    )
    results.append(("DINOv3_ViT-L + DINOv2_reg_ViT-B", dim2, oa2, pc2))

    # Experiment C: DINOv3 ViT-L + DINOv2 ViT-L  (1024 + 1024 = 2048-d)
    oa3, pc3, dim3 = run_ensemble(
        "DINOv3 ViT-L + DINOv2 ViT-L",
        ["dinov3_vitl", "dinov2_vitl"],
        "1000", "287"
    )
    results.append(("DINOv3_ViT-L + DINOv2_ViT-L", dim3, oa3, pc3))

    # Experiment D: all-three: DINOv3 ViT-L + DINOv2 ViT-L + DINOv2 ViT-B
    oa4, pc4, dim4 = run_ensemble(
        "DINOv3 ViT-L + DINOv2 ViT-L + DINOv2 ViT-B",
        ["dinov3_vitl", "dinov2_vitl", "dinov2_vitb"],
        "1000", "287"
    )
    results.append(("DINOv3_ViT-L + DINOv2_ViT-L + DINOv2_ViT-B", dim4, oa4, pc4))

    # Summary table
    print("\n" + "=" * 70)
    print("CROSS-ARCHITECTURE ENSEMBLE SUMMARY")
    print("=" * 70)
    print(f"{'Name':<45} {'Dim':>6} {'Overall':>9} {'vs EnsBl':>10}")
    print("-" * 70)
    for name, dim, oa, _ in results:
        delta = oa - ENSEMBLE_B["overall"]
        print(f"{name:<45} {dim:>6} {oa:>8.2f}%  {delta:>+9.2f}%")
    print(f"{'Ensemble B (DINOv2 ViT-L + ViT-B)  [baseline]':<45} {'1792':>6} {'94.43%':>9}")
    print("=" * 70)

    # Save CSV
    csv_path = os.path.join(RESULTS_DIR, "cross_arch_ensemble_results.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ensemble", "dim", "overall_pct"] + CLASS_NAMES)
        for name, dim, oa, pc in results:
            row = [name, dim, round(oa, 4)] + [
                round(pc.get(c), 4) if pc.get(c) is not None else "" for c in CLASS_NAMES
            ]
            w.writerow(row)
        # Also write Ensemble B for reference
        w.writerow(["Ensemble_B_DINOv2_ViT-L+ViT-B", 1792, 94.43, "", 91.32, 94.44, ""])
    print(f"\nSaved CSV → {csv_path}")


if __name__ == "__main__":
    main()
