"""
Purpose: Test whether DINOv2/CLIP prediction disagreement is a useful live
         out-of-distribution/uncertainty flag -- backlog item 22, scoped
         2026-07-20. Trains a LogReg probe for each frozen encoder on the
         same 1000-shot AI4Mars training split (matching every other probe
         in this thesis, joint_domain_probe.py's protocol/hyperparameters),
         predicts both on the shared 287-image gold-standard test set, and
         measures whether cases where the two models disagree are enriched
         for DINOv2 classification errors -- i.e. whether disagreement is a
         usable, free (no extra inference cost beyond running both models)
         OOD signal, or just noise.
Inputs:  experiments/results/feature_cache/dinov2_reg_small_{train,test}_*.npy
         experiments/results/feature_cache/clip_{train,test}_1000shot.npz
Outputs: experiments/results/cross_model_disagreement_ood_flag.csv
         Printed precision/recall summary for Ch4.
How to run:
    python3 experiments/cross_model_disagreement_ood_flag.py
Project: Onboard Visual Foundation Models for Mars Terrain Perception and Rover Navigation
"""
import csv
import os

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
FEATURE_CACHE = os.path.join(RESULTS_DIR, "feature_cache")
LOGR_C = 0.316  # matches joint_domain_probe.py / all other thesis probes


def train_probe(feats, labels):
    X = normalize(feats, norm="l2")
    clf = LogisticRegression(C=LOGR_C, max_iter=1000, random_state=42,
                              multi_class="multinomial", solver="lbfgs")
    clf.fit(X, labels)
    return clf


def predict(clf, feats):
    X = normalize(feats, norm="l2")
    return clf.predict(X)


def main():
    dinov2_train_feats = np.load(os.path.join(FEATURE_CACHE, "dinov2_reg_small_train_1000_feats.npy"))
    dinov2_train_labels = np.load(os.path.join(FEATURE_CACHE, "dinov2_reg_small_train_1000_labels.npy"))
    dinov2_test_feats = np.load(os.path.join(FEATURE_CACHE, "dinov2_reg_small_test_287_feats.npy"))
    dinov2_test_labels = np.load(os.path.join(FEATURE_CACHE, "dinov2_reg_small_test_287_labels.npy"))

    clip_train = np.load(os.path.join(FEATURE_CACHE, "clip_train_1000shot.npz"))
    clip_test = np.load(os.path.join(FEATURE_CACHE, "clip_test_1000shot.npz"))

    assert np.array_equal(dinov2_test_labels, clip_test["labels"]), \
        "DINOv2 and CLIP test caches are not the same 287-image set in the same order"

    print("Training DINOv2 probe...")
    dinov2_clf = train_probe(dinov2_train_feats, dinov2_train_labels)
    print("Training CLIP probe...")
    clip_clf = train_probe(clip_train["feats"], clip_train["labels"])

    dinov2_pred = predict(dinov2_clf, dinov2_test_feats)
    clip_pred = predict(clip_clf, clip_test["feats"])
    true_labels = dinov2_test_labels

    disagree = dinov2_pred != clip_pred
    dinov2_wrong = dinov2_pred != true_labels

    n = len(true_labels)
    n_disagree = int(disagree.sum())
    n_wrong = int(dinov2_wrong.sum())
    n_disagree_and_wrong = int((disagree & dinov2_wrong).sum())

    # Disagreement flag as an error predictor
    precision = n_disagree_and_wrong / n_disagree if n_disagree else float("nan")
    recall = n_disagree_and_wrong / n_wrong if n_wrong else float("nan")
    base_rate = n_wrong / n  # error rate with no flag at all -- precision must beat this to be useful

    out_path = os.path.join(RESULTS_DIR, "cross_model_disagreement_ood_flag.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_index", "true_label", "dinov2_pred", "clip_pred", "disagree", "dinov2_wrong"])
        for i in range(n):
            writer.writerow([i, int(true_labels[i]), int(dinov2_pred[i]), int(clip_pred[i]),
                              bool(disagree[i]), bool(dinov2_wrong[i])])

    print(f"\nn={n}  DINOv2 errors={n_wrong} ({100*n_wrong/n:.1f}%)  "
          f"disagreements={n_disagree} ({100*n_disagree/n:.1f}%)")
    print(f"Disagreement-as-error-flag: precision={precision:.3f} recall={recall:.3f} "
          f"(base error rate={base_rate:.3f})")
    if not np.isnan(precision):
        lift = precision / base_rate if base_rate else float("nan")
        print(f"Precision lift over base rate: {lift:.2f}x")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
