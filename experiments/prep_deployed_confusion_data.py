"""
Purpose: Compute the confusion matrix for the DEPLOYED model (DINOv2+reg
         ViT-S/14, 1000-shot, the one that actually runs on the Raspberry
         Pi), from the already-cached frozen features, and export as a
         plain CSV. Same protocol as prep_confusion_tsne_data.py, which
         does this for the comparison model (DINOv2 ViT-L) shown in the
         report's Chapter 4 -- this script exists because the deep-dive
         deck's confusion-matrix slide originally reused the ViT-L file for
         lack of an equivalent for the deployed model, and the deployed
         model is the more relevant one to show on that slide.
Inputs:  experiments/results/feature_cache/dinov2_reg_small_{train_1000,test_287}_{feats,labels}.npy
Outputs: experiments/results/confusion_matrix_dinov2_reg_small.csv
How to run:
    python3 experiments/prep_deployed_confusion_data.py
Project: Onboard Visual Foundation Models for Mars Terrain Perception and Rover Navigation
"""
import os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize
from sklearn.metrics import confusion_matrix

CACHE_DIR = os.path.join(os.path.dirname(__file__), "results", "feature_cache")
OUT_DIR = os.path.join(os.path.dirname(__file__), "results")
CLASS_NAMES = ["Soil", "Bedrock", "Sand"]


def load(name):
    return np.load(os.path.join(CACHE_DIR, name))


train_X = normalize(load("dinov2_reg_small_train_1000_feats.npy"))
train_y = load("dinov2_reg_small_train_1000_labels.npy")
test_X = normalize(load("dinov2_reg_small_test_287_feats.npy"))
test_y = load("dinov2_reg_small_test_287_labels.npy")

# Same protocol as the rest of this thesis, and the same one
# prep_confusion_tsne_data.py uses for ViT-L: LogReg, C=0.316, seed=42,
# fit on whatever the cached training labels contain (which is 4 classes,
# soil/bedrock/sand at 1000-shot plus the 108 available big_rock images),
# scored against the 287-image gold test set, which has zero big_rock
# images, so the confusion matrix is reported over the three classes the
# test set actually contains.
clf = LogisticRegression(C=0.316, max_iter=1000, random_state=42)
clf.fit(train_X, train_y)
pred_y = clf.predict(test_X)
acc = (pred_y == test_y).mean() * 100
cm = confusion_matrix(test_y, pred_y, labels=[0, 1, 2])

cm_path = os.path.join(OUT_DIR, "confusion_matrix_dinov2_reg_small.csv")
with open(cm_path, "w") as f:
    f.write("true_class," + ",".join(CLASS_NAMES) + "\n")
    for i, name in enumerate(CLASS_NAMES):
        f.write(name + "," + ",".join(str(x) for x in cm[i]) + "\n")
print(f"Confusion matrix written to {cm_path}. Overall accuracy on this run: {acc:.2f}%")
print(f"Correct: {(pred_y == test_y).sum()} / {len(test_y)}")
