"""
Purpose: Unified Earth+Mars cross-planet generalization probe. Extracts symmetric
         clean single-class terrain crops from AI4Mars (Mars) and RUGD (Earth),
         embeds them with a FROZEN DINOv2+reg ViT-S encoder (the thesis deployment
         model, 90.24% AI4Mars), and trains 3 linear probes (Mars-only, Earth-only,
         Joint) to measure cross-planet transfer of frozen foundation-model features.
         Design + interpretation criteria are FIXED in
         docs/earth_mars_probe_preregistration.md (registered before this was run).
Inputs:  AI4Mars msl images+labels; RUGD images+annotations+colormap.
Outputs: experiments/results/earth_mars_probe_results.csv + printed criteria verdict.
How to run:
    python3 -u experiments/earth_mars_probe.py --cpu
Project: Onboard Visual Foundation Models for Mars Terrain Perception and Rover Navigation
"""

import argparse
import csv
import os
import random

import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize
from transformers import AutoImageProcessor, AutoModel

from earth_mars_crop_extractor import (
    AI4MARS_TO_SHARED, SHARED_CLASSES, crop_purity, parse_rugd_colormap,
    remap_labels, rugd_color_to_shared, sample_clean_crops,
)

# ── Paths ─────────────────────────────────────────────────────────────────────
THESIS_ROOT = "/mnt/c/Users/DELL/Desktop/Thesis"
AI4MARS = os.path.join(THESIS_ROOT, "github source/ai4mars-dataset-merged-0.1")
AI4_IMAGES = os.path.join(AI4MARS, "msl/images/edr")
AI4_TRAIN_LABELS = os.path.join(AI4MARS, "msl/labels/train")
AI4_TEST_LABELS = os.path.join(AI4MARS, "msl/labels/test/masked-gold-min3-100agree")
RUGD_IMAGES_ROOT = os.path.join(THESIS_ROOT, "github source/RUGD/RUGD_frames-with-annotations")
RUGD_ANNOT_ROOT = os.path.join(THESIS_ROOT, "github source/RUGD/RUGD_annotations")
RUGD_COLORMAP = os.path.join(THESIS_ROOT,
                             "github source/RUGD/sample/RUGD_sample-data/RUGD_annotation-colormap.txt")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
CACHE = os.path.join(RESULTS_DIR, "earth_mars_feature_cache.npz")

MODEL_ID = "facebook/dinov2-with-registers-small"  # thesis deployment model (90.24%)
SEED = 42
CROP_FRAC = 0.40          # crop side = 40% of image min-dim (comparable scene scale)
PURITY = 0.90
N_TRAIN = 1000            # target crops / class / planet (cap + report if fewer)
N_TEST = 250
MAX_PER_IMAGE = 6         # diversity cap
CHANCE = 100.0 / len(SHARED_CLASSES)


# ── Crop harvesting (streaming: labels are loaded ONE image at a time and
#    discarded immediately after sampling, never held in memory as a full list —
#    AI4Mars alone has 16,064 x 1024x1024 label masks, ~16.8GB if materialised
#    upfront, which OOM-killed this machine's 7.5GB RAM twice before this fix) ──

def harvest(path_label_loader_pairs, n_target, rng, desc):
    """path_label_loader_pairs: list of (rgb_path, label_loader) where label_loader()
    lazily returns the shared-class array for that one image. Returns list of
    (rgb_path, box, cls)."""
    got = {c: [] for c in range(len(SHARED_CLASSES))}
    order = path_label_loader_pairs[:]
    rng.shuffle(order)
    for rgb_path, label_loader in order:
        if all(len(got[c]) >= n_target for c in got):
            break
        shared = label_loader()
        h, w = shared.shape
        crop_size = int(CROP_FRAC * min(h, w))
        for c in range(len(SHARED_CLASSES)):
            if len(got[c]) >= n_target:
                continue
            boxes = sample_clean_crops(shared, c, crop_size, PURITY,
                                       MAX_PER_IMAGE, rng, max_attempts=60)
            for (top, left) in boxes:
                got[c].append((rgb_path, (top, left, crop_size), c))
        del shared
    out = []
    for c in got:
        out.extend(got[c])
        print(f"  [{desc}] {SHARED_CLASSES[c]}: {len(got[c])} crops")
    return out


def _load_ai4mars_label(label_path):
    raw = np.array(Image.open(label_path))
    return remap_labels(raw, AI4MARS_TO_SHARED)


def ai4mars_records(label_dir):
    """List of (rgb_path, lazy_label_loader) — no label pixels loaded yet."""
    recs = []
    for fname in sorted(os.listdir(label_dir)):
        if not fname.endswith(".png"):
            continue
        stem = fname.replace("_merged.png", "").replace(".png", "")
        img = os.path.join(AI4_IMAGES, stem + ".JPG")
        label_path = os.path.join(label_dir, fname)
        if not os.path.exists(img):
            continue
        recs.append((img, lambda p=label_path: _load_ai4mars_label(p)))
    return recs


def _load_rugd_label(ann_path, name_to_color):
    rgb = np.array(Image.open(ann_path).convert("RGB"))
    return rugd_color_to_shared(rgb, name_to_color)


def rugd_records(scenes, name_to_color):
    # Real RUGD layout: images in RUGD_frames-with-annotations/<scene>/<scene>_NNNNN.png,
    # colour label masks in RUGD_annotations/<scene>/<scene>_NNNNN.png (same filenames).
    recs = []
    ann_index = {}
    for root, _dirs, files in os.walk(RUGD_ANNOT_ROOT):
        for f in files:
            if f.endswith(".png"):
                ann_index[f] = os.path.join(root, f)
    img_index = {}
    for root, _dirs, files in os.walk(RUGD_IMAGES_ROOT):
        for f in files:
            if f.endswith(".png"):
                img_index[f] = os.path.join(root, f)
    for fname, ann_path in sorted(ann_index.items()):
        scene = fname.rsplit("_", 1)[0]
        if scene not in scenes:
            continue
        if fname not in img_index:
            continue
        recs.append((img_index[fname], lambda p=ann_path: _load_rugd_label(p, name_to_color)))
    return recs


def rugd_scene_split():
    scenes = set()
    for root, _dirs, files in os.walk(RUGD_ANNOT_ROOT):
        for f in files:
            if f.endswith(".png"):
                scenes.add(f.rsplit("_", 1)[0])
    scenes = sorted(scenes)
    test = set(scenes[::4])          # every 4th scene held out (deterministic)
    train = [s for s in scenes if s not in test]
    return train, sorted(test)


# ── Feature extraction ────────────────────────────────────────────────────────

def load_model(device):
    proc = AutoImageProcessor.from_pretrained(MODEL_ID)
    model = AutoModel.from_pretrained(MODEL_ID).to(device).eval()
    return proc, model


def embed(crops, proc, model, device, desc):
    feats, labels = [], []
    n = len(crops)
    cache_img = {}
    for i, (rgb_path, (top, left, size), cls) in enumerate(crops):
        if (i + 1) % 200 == 0 or i == n - 1:
            print(f"  embed {desc} {i+1}/{n}")
        if rgb_path not in cache_img:
            cache_img.clear()
            cache_img[rgb_path] = Image.open(rgb_path).convert("RGB")
        tile = cache_img[rgb_path].crop((left, top, left + size, top + size))
        inp = proc(images=tile, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**inp)
            feat = out.last_hidden_state[:, 0, :].cpu().numpy().squeeze()
        feats.append(feat)
        labels.append(cls)
    return normalize(np.array(feats, dtype=np.float32)), np.array(labels)


# ── Probe + evaluation ────────────────────────────────────────────────────────

def acc(clf, X, y):
    return 100.0 * (clf.predict(X) == y).mean()


def fit(X, y):
    return LogisticRegression(max_iter=1000, C=0.316, random_state=SEED).fit(X, y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args()
    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    rng = random.Random(SEED)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    if os.path.exists(CACHE) and not args.rebuild:
        d = np.load(CACHE)
        Xtr_m, ytr_m, Xte_m, yte_m = d["Xtr_m"], d["ytr_m"], d["Xte_m"], d["yte_m"]
        Xtr_e, ytr_e, Xte_e, yte_e = d["Xtr_e"], d["ytr_e"], d["Xte_e"], d["yte_e"]
    else:
        proc, model = load_model(device)
        print("Harvesting Mars (AI4Mars) crops...")
        mars_tr = harvest(ai4mars_records(AI4_TRAIN_LABELS), N_TRAIN, rng, "mars-train")
        mars_te = harvest(ai4mars_records(AI4_TEST_LABELS), N_TEST, rng, "mars-test")
        name_to_color = parse_rugd_colormap(open(RUGD_COLORMAP).read())
        tr_scenes, te_scenes = rugd_scene_split()
        print(f"RUGD train scenes={len(tr_scenes)} test scenes={len(te_scenes)}")
        print("Harvesting Earth (RUGD) crops...")
        earth_tr = harvest(rugd_records(tr_scenes, name_to_color), N_TRAIN, rng, "earth-train")
        earth_te = harvest(rugd_records(te_scenes, name_to_color), N_TEST, rng, "earth-test")
        Xtr_m, ytr_m = embed(mars_tr, proc, model, device, "mars-train")
        Xte_m, yte_m = embed(mars_te, proc, model, device, "mars-test")
        Xtr_e, ytr_e = embed(earth_tr, proc, model, device, "earth-train")
        Xte_e, yte_e = embed(earth_te, proc, model, device, "earth-test")
        np.savez(CACHE, Xtr_m=Xtr_m, ytr_m=ytr_m, Xte_m=Xte_m, yte_m=yte_m,
                 Xtr_e=Xtr_e, ytr_e=ytr_e, Xte_e=Xte_e, yte_e=yte_e)

    mars_only = fit(Xtr_m, ytr_m)
    earth_only = fit(Xtr_e, ytr_e)
    joint = fit(np.vstack([Xtr_m, Xtr_e]), np.concatenate([ytr_m, ytr_e]))

    r = {
        "mars_only@mars": acc(mars_only, Xte_m, yte_m),
        "earth_only@earth": acc(earth_only, Xte_e, yte_e),
        "earth_only@mars": acc(earth_only, Xte_m, yte_m),   # transfer Earth->Mars
        "mars_only@earth": acc(mars_only, Xte_e, yte_e),    # transfer Mars->Earth
        "joint@mars": acc(joint, Xte_m, yte_m),
        "joint@earth": acc(joint, Xte_e, yte_e),
    }
    r["joint_cost_mars"] = r["mars_only@mars"] - r["joint@mars"]
    r["joint_cost_earth"] = r["earth_only@earth"] - r["joint@earth"]

    print("\n=== Results (test acc %%, chance=%.1f) ===" % CHANCE)
    for k, v in r.items():
        print(f"  {k:20s} {v:6.2f}")
    print(f"  N train mars/earth = {len(ytr_m)}/{len(ytr_e)}  "
          f"test mars/earth = {len(yte_m)}/{len(yte_e)}")

    # ── Pre-registered verdict ───────────────────────────────────────────────
    transfer = max(r["earth_only@mars"], r["mars_only@earth"])
    joint_ok = r["joint_cost_mars"] <= 5 and r["joint_cost_earth"] <= 5
    if r["mars_only@mars"] < 85:
        verdict = "SANITY-FAIL (mars_only@mars < 85%) — no transfer claim"
    elif transfer >= 50 and joint_ok:
        verdict = "POSITIVE — frozen features generalise across planets"
    elif transfer <= 40:
        verdict = "HONEST-NEGATIVE — frozen features are planet-specific"
    else:
        verdict = "MIXED — asymmetric generalisation"
    print(f"\nPRE-REGISTERED VERDICT: {verdict}")

    with open(os.path.join(RESULTS_DIR, "earth_mars_probe_results.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        for k, v in r.items():
            w.writerow([k, f"{v:.2f}"])
        w.writerow(["chance", f"{CHANCE:.2f}"])
        w.writerow(["n_train_mars", len(ytr_m)])
        w.writerow(["n_train_earth", len(ytr_e)])
        w.writerow(["n_test_mars", len(yte_m)])
        w.writerow(["n_test_earth", len(yte_e)])
        w.writerow(["verdict", verdict])
    print("Saved earth_mars_probe_results.csv")


if __name__ == "__main__":
    main()
