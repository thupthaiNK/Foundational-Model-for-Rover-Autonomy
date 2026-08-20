"""
Purpose: Empirical check for backlog item 16 (Depth Anything V2 geometry
         layer, L3) -- before building a live near-field obstacle/proximity
         ramp fed into traversability_score_fusion_node.py, test directly
         whether DAv2's own depth statistics separate big_rock frames from
         other terrain in the data this thesis already has cached, using
         the existing 40-d depth feature vectors (depth_anything_terrain_test.py)
         on the 1000-shot AI4Mars training split (the only split with
         big_rock samples -- the 287-image gold-standard test split has
         zero, per item 10's earlier finding, Ch4 line 542).
Inputs:  experiments/results/feature_cache/depth_train_1000_{feats,labels}.npy
Outputs: Printed AUC summary for Ch4/Ch6 (extends the existing DAv2
         classification-negative finding, §4.8/Ch5 §5.2.3B-3, to a
         proximity-signal framing rather than a classification framing).
How to run:
    python3 experiments/depth_proximity_signal_analysis.py
Project: Onboard Visual Foundation Models for Mars Terrain Perception and Rover Navigation
"""
import os

import numpy as np
from sklearn.metrics import roc_auc_score

CACHE_DIR = os.path.join(os.path.dirname(__file__), "results", "feature_cache")

# Feature indices from depth_anything_terrain_test.py's depth_feature_vector():
# 0=mean 1=std 2=raw_std 3=p10 4=p90 5=iqr 6=grad_mean 7=grad_std
CANDIDATE_FEATURES = {
    "p10_normalised_depth": 3,   # nearest 10% of pixels -- most direct "closest point" proxy
    "raw_std_pre_norm": 2,       # absolute depth spread, pre-normalisation
    "std_normalised": 1,
    "grad_mean_roughness": 6,    # the feature already found weak (F=0.12) in the classification framing
}


def main():
    feats = np.load(os.path.join(CACHE_DIR, "depth_train_1000_feats.npy"))
    labels = np.load(os.path.join(CACHE_DIR, "depth_train_1000_labels.npy"))
    y_big_rock = (labels == 3).astype(int)

    print(f"n={len(labels)}  big_rock={y_big_rock.sum()}  other={len(labels) - y_big_rock.sum()}")
    print("\nCandidate near-field/proximity statistic vs big_rock (AUC, 0.5=random):")
    best_name, best_dev = None, 0.0
    for name, idx in CANDIDATE_FEATURES.items():
        auc = roc_auc_score(y_big_rock, feats[:, idx])
        dev = abs(auc - 0.5)
        print(f"  {name:25s} AUC={auc:.3f}  |deviation from random|={dev:.3f}")
        if dev > best_dev:
            best_name, best_dev = name, dev

    print(f"\nBest candidate: {best_name} (deviation {best_dev:.3f} from random)")
    if best_dev < 0.15:
        print("HONEST NEGATIVE: no candidate depth statistic reliably separates big_rock frames "
              "from other terrain (all AUCs near 0.5). This extends the existing DAv2 "
              "classification-negative finding (depth-only accuracy 36.6%, grad_mean F=0.12, "
              "Ch4/Ch5 S5.2.3B-3) to a proximity-signal framing: even reframed as 'closest point "
              "in frame' rather than 'roughness class', DAv2's depth estimate does not carry a "
              "usable rock-proximity signal for this dataset. A live near-field ramp built on top "
              "of this statistic would not be expected to add real safety value -- not built as a "
              "live feature for that reason, consistent with this thesis's practice of not building "
              "features whose offline signal has already tested negative (cf. item 10's per-class "
              "calibration, skipped before any code for the same kind of reason).")
    else:
        print("A usable signal may exist -- would be worth scoping the live ramp further.")


if __name__ == "__main__":
    main()
