"""
Purpose: Compute the confusion matrix and t-SNE 2D embedding for the model
         actually deployed on the Raspberry Pi (DINOv2+reg ViT-S/14, 1000-shot),
         from the already-cached frozen features, and export as plain CSVs.
         This mirrors prep_confusion_tsne_data.py (which does the same for the
         best-accuracy comparison model, DINOv2 ViT-L/14) so Figure 5.1 can be
         swapped from the comparison model to the deployed model without
         changing the plotting pipeline. Data preparation only -- the actual
         figure is drawn in MATLAB (experiments/make_thesis_figures_2.m) per
         this thesis's tooling convention.
Inputs:  experiments/results/feature_cache/dinov2_reg_small_{train_1000,test_287}_{feats,labels}.npy
Outputs: experiments/results/confusion_matrix_dinov2_reg_small.csv
         experiments/results/tsne_dinov2_reg_small.csv
How to run:
    python3 experiments/prep_confusion_tsne_data_deployed.py
Project: Onboard Visual Foundation Models for Mars Terrain Perception and Rover Navigation
"""
import os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
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

# --- Confusion matrix (same protocol as the rest of this thesis: LogReg, C=0.316, seed=42) ---
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

# --- t-SNE 2D embedding of the test-set features (PCA to 50-d first, matching thesis protocol) ---
n_components = min(50, test_X.shape[0] - 1, test_X.shape[1])
pca = PCA(n_components=n_components, random_state=42)
test_X_pca = pca.fit_transform(test_X)
tsne = TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=42, init="pca")
emb = tsne.fit_transform(test_X_pca)

tsne_path = os.path.join(OUT_DIR, "tsne_dinov2_reg_small.csv")
with open(tsne_path, "w") as f:
    # pred_class / correct let the figure ring the misclassified points, so
    # the claim that errors sit on the boundaries between neighbouring classes
    # is visible in the plot rather than only asserted in the text.
    f.write("x,y,label,class_name,pred,pred_class,correct\n")
    for (x, y), lbl, pr in zip(emb, test_y, pred_y):
        f.write(f"{x:.6f},{y:.6f},{int(lbl)},{CLASS_NAMES[int(lbl)]},"
                f"{int(pr)},{CLASS_NAMES[int(pr)]},{int(lbl == pr)}\n")
print(f"t-SNE embedding written to {tsne_path}. "
      f"Misclassified points in this run: {(pred_y != test_y).sum()} of {len(test_y)}.")
