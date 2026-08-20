"""
Purpose: Supplementary experiment. The deployed probe already has a big-rock output,
         trained on the 108 images AI4Mars provides against roughly 1,000 per other
         class. It simply never selects it: zero predictions on the 287-image gold
         test set, and zero on the 36 real rock photographs of Section 4.x. So the
         reported 0% is measured, not definitional. This script asks what the main
         results cannot: is that because the frozen features cannot separate big
         rock, or only because the class is outnumbered 10 to 1?

         Three probes are trained on the same features and scored on the same held-out
         images, so the only thing that changes is the label set and its balance:
           A. 3-class (soil, bedrock, sand)  -- the class removed altogether
           B. 4-class, imbalanced            -- the deployed configuration
           C. 4-class, balanced              -- equal shots per class
         Comparing B against C separates a data-volume effect from a feature-space one.

         The AI4Mars gold-standard test set cannot be used: none of its 322 images has
         big rock as the dominant class (only 5 contain any big-rock pixels at all).
         The test split is therefore held out from the crowd-sourced training labels,
         so accuracies here are NOT comparable to the 90.24% headline figure.
Inputs:  AI4Mars train labels + EDR images (paths below)
Outputs: experiments/results/big_rock_fourclass_probe.csv
         experiments/results/big_rock_fourclass_confusion.csv
         a selection of big-rock images copied out for visual inspection
How to run:
    python3 -u experiments/big_rock_fourclass_probe.py
Project: Onboard Visual Foundation Models for Mars Terrain Perception and Rover Navigation
"""

import csv
import os
import random
import shutil

import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import normalize
from transformers import AutoImageProcessor, AutoModel

AI4MARS_BASE = "/mnt/c/Users/DELL/Desktop/Thesis/github source/ai4mars-dataset-merged-0.1"
IMAGES_DIR = os.path.join(AI4MARS_BASE, "msl/images/edr")
TRAIN_LABELS = os.path.join(AI4MARS_BASE, "msl/labels/train")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
# Where the user inspects the images this experiment actually used.
INSPECT_DIR = "/mnt/c/Users/DELL/Downloads/Terrain Picture/2. Big rock/2. AI4MAR Big rock"

CLASS_NAMES = ["soil", "bedrock", "sand", "big rock"]
BIG_ROCK = 3
IGNORE_PIXEL = 255
SEED = 42
# Same regularisation as every other probe in this thesis.
LOGREG_C = 0.316
# Held out per class. 30 keeps a usable big-rock test set out of only 108 images
# while leaving 78 to train on.
N_TEST_PER_CLASS = 30
# Cap on the non-big-rock pool, to keep feature extraction to about a thousand
# images. Sampling is seeded, so the choice is reproducible.
N_POOL_OTHER = 300
N_INSPECT_COPIES = 20

os.makedirs(RESULTS_DIR, exist_ok=True)


def dominant_class(label_path):
    label = np.asarray(Image.open(label_path))
    valid = label[label != IGNORE_PIXEL]
    if valid.size == 0:
        return None, 0
    counts = np.bincount(valid.ravel(), minlength=4)[:4]
    return int(counts.argmax()), int(counts[BIG_ROCK])


def build_index():
    """Every training image with its dominant class and its big-rock pixel count."""
    rows = []
    names = sorted(f for f in os.listdir(TRAIN_LABELS) if f.endswith(".png"))
    for i, fname in enumerate(names):
        if (i + 1) % 2000 == 0:
            print(f"  indexed {i+1}/{len(names)}")
        stem = fname.replace("_merged.png", "").replace(".png", "")
        image_path = os.path.join(IMAGES_DIR, stem + ".JPG")
        if not os.path.exists(image_path):
            continue
        gt, big_px = dominant_class(os.path.join(TRAIN_LABELS, fname))
        if gt is None:
            continue
        rows.append({"stem": stem, "image": image_path, "gt": gt, "big_px": big_px})
    return rows


def extract(model, processor, paths, device):
    feats = []
    for i, p in enumerate(paths):
        if (i + 1) % 100 == 0 or i == len(paths) - 1:
            print(f"  features {i+1}/{len(paths)}")
        image = Image.open(p).convert("RGB")
        inputs = processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**inputs)
        feats.append(out.last_hidden_state[:, 0, :].cpu().numpy().squeeze())
    return normalize(np.array(feats, dtype=np.float32))


def per_class_accuracy(y_true, y_pred, classes):
    out = {}
    for c in classes:
        m = y_true == c
        out[CLASS_NAMES[c]] = float((y_pred[m] == c).mean() * 100) if m.any() else float("nan")
    return out


def main():
    # CPU only, matching every other experiment in this thesis. The development
    # machine's MX330 reports as available but has no cuDNN support, so a CUDA
    # forward pass dies with "unable to find an engine to execute this computation".
    device = "cpu"
    rng = random.Random(SEED)

    print("Indexing AI4Mars training labels...")
    index = build_index()
    by_class = {c: [r for r in index if r["gt"] == c] for c in range(4)}
    print("  dominant-class counts:",
          {CLASS_NAMES[c]: len(by_class[c]) for c in range(4)})

    # Big rock: keep every image. Others: seeded sample, so the run is reproducible.
    pools = {}
    for c in range(4):
        pool = list(by_class[c])
        rng.shuffle(pool)
        pools[c] = pool if c == BIG_ROCK else pool[:N_POOL_OTHER]

    test_rows, train_rows = [], []
    for c in range(4):
        test_rows += pools[c][:N_TEST_PER_CLASS]
        train_rows += pools[c][N_TEST_PER_CLASS:]
    print(f"  test {len(test_rows)} images, train pool {len(train_rows)} images")

    print(f"Loading DINOv2+reg ViT-S/14 (the deployed encoder) on {device}...")
    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-with-registers-small")
    model = AutoModel.from_pretrained("facebook/dinov2-with-registers-small").to(device).eval()

    print("Extracting test features...")
    Xte = extract(model, processor, [r["image"] for r in test_rows], device)
    yte = np.array([r["gt"] for r in test_rows])
    print("Extracting train features...")
    Xtr = extract(model, processor, [r["image"] for r in train_rows], device)
    ytr = np.array([r["gt"] for r in train_rows])

    n_big_train = int((ytr == BIG_ROCK).sum())
    results = []

    def fit_eval(name, classes, balanced):
        keep = np.isin(ytr, classes)
        idx = np.where(keep)[0]
        if balanced:
            r = np.random.RandomState(SEED)
            picked = []
            for c in classes:
                ci = np.where(ytr == c)[0]
                picked += r.choice(ci, size=min(n_big_train, len(ci)),
                                   replace=False).tolist()
            idx = np.array(sorted(picked))
        clf = LogisticRegression(C=LOGREG_C, max_iter=2000, random_state=SEED)
        clf.fit(Xtr[idx], ytr[idx])
        tmask = np.isin(yte, classes)
        pred = clf.predict(Xte[tmask])
        true = yte[tmask]
        acc = float((pred == true).mean() * 100)
        pc = per_class_accuracy(true, pred, classes)
        n_per = {CLASS_NAMES[c]: int((ytr[idx] == c).sum()) for c in classes}
        print(f"\n{name}\n  train per class: {n_per}\n  overall {acc:.2f}%  per class {pc}")
        results.append({"probe": name, "n_train": len(idx), "overall": round(acc, 2),
                        **{f"acc_{k}": (round(v, 2) if v == v else "") for k, v in pc.items()},
                        **{f"ntrain_{k}": v for k, v in n_per.items()}})
        return clf, true, pred

    fit_eval("A. 3-class, balanced (deployed label set)", [0, 1, 2], True)
    fit_eval("B. 4-class, balanced", [0, 1, 2, 3], True)
    _, true4, pred4 = fit_eval("C. 4-class, all available (imbalanced)", [0, 1, 2, 3], False)

    out = os.path.join(RESULTS_DIR, "big_rock_fourclass_probe.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[-1].keys()))
        w.writeheader()
        for r in results:
            w.writerow({k: r.get(k, "") for k in results[-1].keys()})
    print(f"\nWrote {out}")

    cm = confusion_matrix(true4, pred4, labels=[0, 1, 2, 3])
    cmp_path = os.path.join(RESULTS_DIR, "big_rock_fourclass_confusion.csv")
    with open(cmp_path, "w", newline="") as f:
        f.write("true," + ",".join(CLASS_NAMES) + "\n")
        for i, n in enumerate(CLASS_NAMES):
            f.write(n + "," + ",".join(str(v) for v in cm[i]) + "\n")
    print(f"Wrote {cmp_path}")

    # Copy out the big-rock images this experiment scored, so the selection can be
    # checked by eye rather than taken on trust.
    big_test = sorted([r for r in test_rows if r["gt"] == BIG_ROCK],
                      key=lambda r: -r["big_px"])[:N_INSPECT_COPIES]
    if os.path.isdir(os.path.dirname(INSPECT_DIR)):
        os.makedirs(INSPECT_DIR, exist_ok=True)
        with open(os.path.join(INSPECT_DIR, "_selection.csv"), "w", newline="") as f:
            f.write("rank,filename,big_rock_pixels,split\n")
            for i, r in enumerate(big_test, 1):
                shutil.copy2(r["image"], os.path.join(INSPECT_DIR, r["stem"] + ".JPG"))
                f.write(f"{i},{r['stem']}.JPG,{r['big_px']},test\n")
        print(f"Copied {len(big_test)} big-rock images to {INSPECT_DIR}")


if __name__ == "__main__":
    main()
