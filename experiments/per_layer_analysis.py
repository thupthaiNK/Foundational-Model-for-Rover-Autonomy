"""
Purpose:    Per-layer optimal layer search for DINOv2 ViT-L/14 multilayer features.
            The multilayer cache contains 4 layers × 1024-d = 4096-d concatenated features.
            This script tests each layer independently to identify which ViT-L layer
            provides the best terrain classification accuracy, and specifically which layer
            is optimal for Bedrock (the safety-critical class).
            Approach from Perception Encoder (Bolya et al., 2025, arXiv:2504.13181):
            task-specific optimal features exist at intermediate layers, not final output.
Inputs:     experiments/results/feature_cache/dinov2_vitl_multilayer_{train,test}_*.npy
            (4096-d = 4 concatenated 1024-d layers from DINOv2 ViT-L/14)
Outputs:    experiments/results/per_layer_analysis.csv
            Printed accuracy per layer (overall + per-class) + comparison vs single final layer
How to run:
    python3 -u experiments/per_layer_analysis.py
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
LAYER_DIM   = 1024   # DINOv2 ViT-L hidden size
N_LAYERS    = 4      # concat of 4 layers in the multilayer cache

# Single-layer baseline (final layer only, from standard dinov2_vitl cache)
SINGLE_LAYER_BASELINE = {"overall": 93.73, "bedrock": 90.94}


def per_class_acc(preds, labels):
    result = {}
    for i, name in enumerate(CLASS_NAMES):
        mask = labels == i
        result[name] = (preds[mask] == labels[mask]).mean() * 100 if mask.sum() > 0 else None
    return result


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Load multilayer caches
    tr_ml = np.load(os.path.join(CACHE_DIR, "dinov2_vitl_multilayer_train_1000_feats.npy"))
    tr_lb = np.load(os.path.join(CACHE_DIR, "dinov2_vitl_multilayer_train_1000_labels.npy"))
    te_ml = np.load(os.path.join(CACHE_DIR, "dinov2_vitl_multilayer_test_287_feats.npy"))
    te_lb = np.load(os.path.join(CACHE_DIR, "dinov2_vitl_multilayer_test_287_labels.npy"))

    print(f"Multilayer cache: train {tr_ml.shape}  test {te_ml.shape}", flush=True)
    print(f"Expected: 4 layers × {LAYER_DIM}-d = {N_LAYERS * LAYER_DIM}-d total\n", flush=True)

    results = []
    best_overall  = 0
    best_bedrock  = 0
    best_overall_layer  = -1
    best_bedrock_layer  = -1

    # Test each layer independently
    for i in range(N_LAYERS):
        lo = i * LAYER_DIM
        hi = lo + LAYER_DIM
        tr_feats = normalize(tr_ml[:, lo:hi])
        te_feats = normalize(te_ml[:, lo:hi])

        clf = LogisticRegression(C=0.316, max_iter=1000, random_state=SEED,
                                 multi_class="multinomial", solver="lbfgs")
        clf.fit(tr_feats, tr_lb)
        preds   = clf.predict(te_feats)
        overall = (preds == te_lb).mean() * 100
        pc      = per_class_acc(preds, te_lb)

        layer_name = f"Layer {i+1} (block {[-1, -3, -6, -9][i]} approx)"
        print(f"Layer {i+1}/{N_LAYERS}:  Overall={overall:.2f}%  "
              f"Bedrock={pc['bedrock']:.2f}%  "
              f"Soil={pc['soil']:.2f}%  Sand={pc['sand']:.2f}%", flush=True)

        if overall > best_overall:
            best_overall = overall
            best_overall_layer = i + 1
        if pc['bedrock'] is not None and pc['bedrock'] > best_bedrock:
            best_bedrock = pc['bedrock']
            best_bedrock_layer = i + 1

        results.append({
            "layer": i + 1,
            "overall": round(overall, 4),
            "bedrock": round(pc['bedrock'], 4) if pc['bedrock'] else None,
            "soil":    round(pc['soil'], 4)    if pc['soil']    else None,
            "sand":    round(pc['sand'], 4)    if pc['sand']    else None,
        })

    # Test full concat (all 4 layers — should match §4.7.18 result of ~93.03%)
    tr_all = normalize(tr_ml)
    te_all = normalize(te_ml)
    clf_all = LogisticRegression(C=0.316, max_iter=1000, random_state=SEED,
                                 multi_class="multinomial", solver="lbfgs")
    clf_all.fit(tr_all, tr_lb)
    preds_all = clf_all.predict(te_all)
    oa_all  = (preds_all == te_lb).mean() * 100
    pc_all  = per_class_acc(preds_all, te_lb)

    print(f"\nFull 4-layer concat:  Overall={oa_all:.2f}%  Bedrock={pc_all['bedrock']:.2f}%", flush=True)
    results.append({
        "layer": "all4_concat",
        "overall": round(oa_all, 4),
        "bedrock": round(pc_all['bedrock'], 4),
        "soil":    round(pc_all['soil'], 4),
        "sand":    round(pc_all['sand'], 4),
    })

    # Summary
    print("\n" + "=" * 65)
    print("PER-LAYER ANALYSIS — DINOv2 ViT-L/14")
    print("=" * 65)
    print(f"{'Layer':<18} {'Overall':>9} {'Bedrock':>9} {'Soil':>9} {'Sand':>9}")
    print("-" * 65)
    for r in results:
        lbl = f"Layer {r['layer']}" if isinstance(r['layer'], int) else r['layer']
        print(f"{lbl:<18} {r['overall']:>8.2f}%  "
              f"{r['bedrock']:>8.2f}%  "
              f"{r['soil']:>8.2f}%  "
              f"{r['sand']:>8.2f}%")
    print("-" * 65)
    print(f"{'Single-layer (§4.7)':<18} {SINGLE_LAYER_BASELINE['overall']:>8.2f}%  "
          f"{SINGLE_LAYER_BASELINE['bedrock']:>8.2f}%  "
          f"{'—':>9}  {'—':>9}  ← final layer only")
    print("=" * 65)
    print(f"\nBest overall: Layer {best_overall_layer} ({best_overall:.2f}%)")
    print(f"Best bedrock: Layer {best_bedrock_layer} ({best_bedrock:.2f}%)")

    if best_overall > SINGLE_LAYER_BASELINE['overall']:
        delta = best_overall - SINGLE_LAYER_BASELINE['overall']
        print(f"→ Intermediate layer BEATS final layer by +{delta:.2f}% overall")
    else:
        delta = SINGLE_LAYER_BASELINE['overall'] - best_overall
        print(f"→ Final layer is optimal (intermediate layers −{delta:.2f}% at best)")

    # Save CSV
    csv_path = os.path.join(RESULTS_DIR, "per_layer_analysis.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["layer", "overall", "bedrock", "soil", "sand"])
        w.writeheader()
        w.writerows(results)
        w.writerow({"layer": "single_layer_baseline_final",
                    "overall": SINGLE_LAYER_BASELINE["overall"],
                    "bedrock": SINGLE_LAYER_BASELINE["bedrock"],
                    "soil": None, "sand": None})
    print(f"\nSaved → {csv_path}")


if __name__ == "__main__":
    main()
