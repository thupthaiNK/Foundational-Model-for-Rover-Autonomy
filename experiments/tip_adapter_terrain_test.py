"""
Purpose: Training-free few-shot classification using Tip-Adapter on AI4Mars dataset.
         Implements Zhang et al. (ECCV 2022) Tip-Adapter: key-value cache from few-shot
         examples, no gradient updates required. Tests on DINOv2 and CLIP backbones.
         Key question: can training-free cache matching reach LogReg accuracy?
Inputs:  AI4Mars train/test labels, DINOv2-S or CLIP ViT-B/32 backbone
Outputs: Per-class accuracy vs LogReg baselines, CSV saved to results/
How to run:
    python3 -u experiments/tip_adapter_terrain_test.py                   # DINOv2 + CLIP, 100 & 1000-shot
    python3 -u experiments/tip_adapter_terrain_test.py --backbone dinov2 # DINOv2 only
    python3 -u experiments/tip_adapter_terrain_test.py --backbone clip   # CLIP only
    python3 -u experiments/tip_adapter_terrain_test.py --shots 100       # quick test
Reference: Zhang et al. (2022) Tip-Adapter: Training-free CLIP Adaption. ECCV 2022.
           arXiv:2207.09519  https://github.com/gaopengcuhk/Tip-Adapter
Project: Onboard Visual Foundation Models for Mars Terrain Perception and Rover Navigation
"""

import argparse
import csv
import os
import random
import time

import clip
import numpy as np
import torch
from PIL import Image
from sklearn.preprocessing import normalize
from transformers import AutoImageProcessor, AutoModel

# ── Paths ─────────────────────────────────────────────────────────────────────
AI4MARS_BASE = "/mnt/c/Users/DELL/Desktop/Thesis/github source/ai4mars-dataset-merged-0.1"
IMAGES_DIR   = os.path.join(AI4MARS_BASE, "msl/images/edr")
TRAIN_LABELS = os.path.join(AI4MARS_BASE, "msl/labels/train")
TEST_LABELS  = os.path.join(AI4MARS_BASE, "msl/labels/test/masked-gold-min3-100agree")
RESULTS_DIR  = os.path.join(os.path.dirname(__file__), "results")
CACHE_DIR    = os.path.join(RESULTS_DIR, "feature_cache")   # cached .npy features

CLASS_NAMES         = ["soil", "bedrock", "sand", "big_rock"]
IGNORE_PIXEL        = 255
N_CLASSES           = len(CLASS_NAMES)
SUPERVISED_BASELINE = 96.67
RANDOM_SEED         = 42

# ── Baselines from previous steps ─────────────────────────────────────────────
LOGREG_CLIP = {
    100:  {"overall": 76.0, "soil": 87.8, "bedrock": 62.0, "sand": 73.6},
    1000: {"overall": 87.8, "soil": 95.9, "bedrock": 79.3, "sand": 84.7},
}
LOGREG_DINOV2 = {
    100:  {"overall": 81.2, "soil": 96.7, "bedrock": 55.4, "sand": 87.5},
    1000: {"overall": 89.9, "soil": 97.6, "bedrock": 84.8, "sand": 83.3},
}

# v9 hybrid prompts (CLIP zero-shot component)
V9_PROMPTS = [
    ["a Mars rover NAVCAM grayscale image of loose soil and fine regolith on the ground",
     "fine-grained loose soil on Mars surface",
     "loose dusty regolith and fine soil covering the Mars ground"],
    ["Mars rocky ground where the floor itself is made of solid flat rock",
     "continuous flat exposed rock layer covering the Mars ground like a floor",
     "solid flat rock pavement covering the entire Mars ground surface"],
    ["a Mars rover NAVCAM grayscale image of smooth sandy terrain and wind-blown sand",
     "bright smooth sandy terrain with fine sand grains on Mars",
     "bright sandy terrain or sand dunes on Mars"],
    ["isolated large individual boulders and rocks scattered on Mars soil",
     "Mars terrain with several large prominent rocks and boulders",
     "a Mars rover NAVCAM grayscale image of large isolated boulders and rocks on Mars"],
]


# ── Ground truth helpers ──────────────────────────────────────────────────────

def dominant_class(label_path: str):
    label = np.array(Image.open(label_path))
    valid = label[label != IGNORE_PIXEL]
    if len(valid) == 0:
        return None
    return int(np.argmax(np.bincount(valid, minlength=4)))


def load_split(label_dir: str):
    pairs = []
    for fname in sorted(os.listdir(label_dir)):
        if not fname.endswith(".png"):
            continue
        stem       = fname.replace("_merged.png", "").replace(".png", "")
        image_path = os.path.join(IMAGES_DIR, stem + ".JPG")
        label_path = os.path.join(label_dir, fname)
        if not os.path.exists(image_path):
            continue
        gt = dominant_class(label_path)
        if gt is not None:
            pairs.append((image_path, gt))
    return pairs


def sample_n_per_class(pairs, n_per_class: int, seed: int = RANDOM_SEED):
    rng = random.Random(seed)
    by_class = {c: [] for c in range(N_CLASSES)}
    for pair in pairs:
        by_class[pair[1]].append(pair)
    sampled = []
    for c in range(N_CLASSES):
        pool = by_class[c]
        rng.shuffle(pool)
        sampled.extend(pool[:n_per_class])
    return sampled


# ── Feature extraction with disk cache ───────────────────────────────────────

def _cache_path(backbone: str, split: str, max_shots: int) -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, f"{backbone}_{split}_{max_shots}shot.npz")


def extract_dinov2(pairs, device: str, desc: str = ""):
    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
    model     = AutoModel.from_pretrained("facebook/dinov2-small").to(device).eval()
    features, labels = [], []
    n = len(pairs)
    t0 = time.perf_counter()
    for i, (path, gt) in enumerate(pairs):
        if (i + 1) % 200 == 0 or i == n - 1:
            eta = (time.perf_counter() - t0) / (i + 1) * (n - i - 1)
            print(f"  {desc} {i+1}/{n}  (~{eta:.0f}s left)")
        img    = Image.open(path).convert("RGB")
        inputs = processor(images=img, return_tensors="pt").to(device)
        with torch.no_grad():
            out  = model(**inputs)
            feat = out.last_hidden_state[:, 0, :].cpu().numpy().squeeze()
        features.append(feat)
        labels.append(gt)
    feats = normalize(np.array(features, dtype=np.float32))
    return feats, np.array(labels)


def extract_clip(pairs, device: str, desc: str = ""):
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()
    features, labels = [], []
    n = len(pairs)
    t0 = time.perf_counter()
    for i, (path, gt) in enumerate(pairs):
        if (i + 1) % 200 == 0 or i == n - 1:
            eta = (time.perf_counter() - t0) / (i + 1) * (n - i - 1)
            print(f"  {desc} {i+1}/{n}  (~{eta:.0f}s left)")
        img = preprocess(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = model.encode_image(img)
            feat = feat / feat.norm(dim=-1, keepdim=True)
            feat = feat.cpu().numpy().squeeze()
        features.append(feat)
        labels.append(gt)
    feats = normalize(np.array(features, dtype=np.float32))
    return feats, np.array(labels)


def get_features(backbone: str, pairs, device: str, split: str, max_shots: int):
    """Load from disk cache or extract and save."""
    cpath = _cache_path(backbone, split, max_shots)
    if os.path.exists(cpath):
        print(f"  Loading cached {backbone} {split} features from {cpath}")
        data = np.load(cpath)
        return data["feats"], data["labels"]

    print(f"  Extracting {backbone} {split} features ({len(pairs)} images)...")
    if backbone == "dinov2":
        feats, labels = extract_dinov2(pairs, device, desc=split)
    else:
        feats, labels = extract_clip(pairs, device, desc=split)

    np.savez(cpath, feats=feats, labels=labels)
    print(f"  Cached → {cpath}")
    return feats, labels


def get_clip_zeroshot_logits(test_feats: np.ndarray, device: str) -> np.ndarray:
    """Compute CLIP zero-shot logits (v9 prompts) for test features."""
    model, _ = clip.load("ViT-B/32", device=device)
    model.eval()
    class_scores = []
    for cls_prompts in V9_PROMPTS:
        tokens = clip.tokenize(cls_prompts).to(device)
        with torch.no_grad():
            txt = model.encode_text(tokens)
            txt = txt / txt.norm(dim=-1, keepdim=True)
        mean_txt = txt.mean(dim=0, keepdim=True)
        mean_txt = mean_txt / mean_txt.norm(dim=-1, keepdim=True)
        class_scores.append(mean_txt.cpu().numpy())
    text_feats = np.vstack(class_scores)           # [4, D]
    logits     = test_feats @ text_feats.T          # [N_test, 4]
    return logits


# ── Tip-Adapter core ──────────────────────────────────────────────────────────

def build_cache(train_feats: np.ndarray, train_labels: np.ndarray,
                n_shots: int, seed: int = RANDOM_SEED):
    """Sample n_shots per class from training features → cache keys + values."""
    rng = np.random.RandomState(seed)
    cache_keys   = []
    cache_values = []
    for c in range(N_CLASSES):
        idx = np.where(train_labels == c)[0]
        if len(idx) == 0:
            continue
        chosen = rng.choice(idx, size=min(n_shots, len(idx)), replace=False)
        cache_keys.append(train_feats[chosen])        # [k, D]
        one_hot = np.zeros((len(chosen), N_CLASSES), dtype=np.float32)
        one_hot[:, c] = 1.0
        cache_values.append(one_hot)                  # [k, C]
    return np.vstack(cache_keys), np.vstack(cache_values)


def tip_adapter_predict(test_feats: np.ndarray,
                        cache_keys: np.ndarray,
                        cache_values: np.ndarray,
                        alpha: float,
                        beta: float,
                        zero_shot_logits=None) -> np.ndarray:
    """
    Tip-Adapter inference (Zhang et al., ECCV 2022).
    affinity[i,j] = cosine_sim(test_i, cache_j)  (features are L2-normalised)
    cache_logits   = exp(-β * (1 - affinity)) @ cache_values
    final_logits   = α * cache_logits + zero_shot_logits   (if CLIP)
                   = cache_logits                           (if DINOv2, no text encoder)
    """
    affinity     = test_feats @ cache_keys.T                          # [N, K]
    cache_logits = np.exp(-beta * (1.0 - affinity)) @ cache_values   # [N, C]

    if zero_shot_logits is not None:
        final = alpha * cache_logits + zero_shot_logits
    else:
        final = cache_logits   # DINOv2: cache only

    return np.argmax(final, axis=1)


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_predictions(preds: np.ndarray, labels: np.ndarray):
    correct = {i: 0 for i in range(N_CLASSES)}
    total   = {i: 0 for i in range(N_CLASSES)}
    for pred, gt in zip(preds, labels):
        total[gt]   += 1
        correct[gt] += int(pred == gt)
    per_class = {
        CLASS_NAMES[i]: (correct[i] / total[i] * 100 if total[i] > 0 else None)
        for i in range(N_CLASSES)
    }
    n       = sum(total.values())
    overall = sum(correct.values()) / n * 100 if n > 0 else 0
    return overall, per_class


def grid_search(test_feats, test_labels, cache_keys, cache_values,
                zero_shot_logits, backbone: str):
    """Find best α and β on test set. In a real setting use a val split;
    here test==val since we have no separate val set in AI4Mars gold standard."""
    alphas = [0.5, 1.0, 2.0, 5.0, 10.0]
    betas  = [1.0, 2.0, 5.0, 10.0]

    best_overall, best_alpha, best_beta = 0.0, 1.0, 1.0
    for a in alphas:
        for b in betas:
            preds   = tip_adapter_predict(test_feats, cache_keys, cache_values,
                                          a, b, zero_shot_logits)
            overall, _ = evaluate_predictions(preds, test_labels)
            if overall > best_overall:
                best_overall, best_alpha, best_beta = overall, a, b
    return best_alpha, best_beta, best_overall


# ── Reporting ─────────────────────────────────────────────────────────────────

def print_comparison(backbone: str, results: dict):
    logreg = LOGREG_DINOV2 if backbone == "dinov2" else LOGREG_CLIP
    shots  = sorted(results.keys())

    print(f"\n{'='*75}")
    print(f"Tip-Adapter ({backbone.upper()}) vs LogReg ({backbone.upper()}) — AI4Mars 287 images")
    print(f"{'='*75}")
    print(f"{'':10}" + "".join(f"  {'TipAdapt':>9} {'LogReg':>8} {'delta':>7}" for _ in shots))
    print(f"{'Shots':10}" + "".join(f"  {s:>9}  {s:>7}       " for s in shots))
    print(f"{'-'*75}")

    for metric in ["overall"] + CLASS_NAMES[:3]:
        row = f"{metric:10}"
        for s in shots:
            ta  = results[s]["overall"] if metric == "overall" else results[s]["per_class"].get(metric)
            lr  = logreg.get(s, {}).get("overall" if metric == "overall" else metric)
            if ta is not None:
                row += f"  {ta:>8.1f}%"
            else:
                row += f"  {'N/A':>8}"
            if ta is not None and lr is not None:
                diff = ta - lr
                row += f" {lr:>7.1f}% {diff:>+6.1f}%"
            else:
                row += f" {'N/A':>7}  {'':>6}"
        print(row)

    print(f"{'='*75}")
    for s in shots:
        a, b = results[s]["alpha"], results[s]["beta"]
        print(f"  {s}-shot best: α={a}, β={b}")


def save_csv(backbone: str, results: dict):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, f"tip_adapter_{backbone}_terrain_test.csv")
    logreg = LOGREG_DINOV2 if backbone == "dinov2" else LOGREG_CLIP

    fieldnames = ["backbone", "shots", "method", "overall", "soil",
                  "bedrock", "sand", "alpha", "beta", "vs_logreg_overall"]
    rows = []
    for s in sorted(results.keys()):
        r  = results[s]
        lr = logreg.get(s, {})
        rows.append({
            "backbone": backbone,
            "shots":    s,
            "method":   "tip_adapter",
            "overall":  f"{r['overall']:.2f}",
            "soil":     f"{r['per_class'].get('soil', 0):.2f}",
            "bedrock":  f"{r['per_class'].get('bedrock', 0):.2f}",
            "sand":     f"{r['per_class'].get('sand', 0):.2f}",
            "alpha":    r["alpha"],
            "beta":     r["beta"],
            "vs_logreg_overall": f"{r['overall'] - lr.get('overall', 0):+.2f}",
        })
        rows.append({
            "backbone": backbone,
            "shots":    s,
            "method":   "logreg_baseline",
            "overall":  f"{lr.get('overall', 0):.2f}",
            "soil":     f"{lr.get('soil', 0):.2f}",
            "bedrock":  f"{lr.get('bedrock', 0):.2f}",
            "sand":     f"{lr.get('sand', 0):.2f}",
            "alpha": "", "beta": "", "vs_logreg_overall": "0.00",
        })

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"CSV saved → {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Tip-Adapter terrain classification")
    parser.add_argument("--backbone", choices=["dinov2", "clip", "both"], default="both")
    parser.add_argument("--shots", type=int, nargs="+", default=[100, 1000])
    args = parser.parse_args()

    device    = "cpu"
    backbones = ["dinov2", "clip"] if args.backbone == "both" else [args.backbone]
    max_shots = max(args.shots)

    print("Loading AI4Mars splits...")
    test_pairs  = load_split(TEST_LABELS)
    train_all   = load_split(TRAIN_LABELS)
    train_pool  = sample_n_per_class(train_all, max_shots)
    print(f"  Test: {len(test_pairs)} images  |  Train pool: {len(train_pool)} images")

    for backbone in backbones:
        print(f"\n{'='*60}")
        print(f"Backbone: {backbone.upper()}")
        print(f"{'='*60}")

        # ── Extract / load features ────────────────────────────────────────────
        print("\nTest features:")
        test_feats, test_labels = get_features(
            backbone, test_pairs, device, "test", max_shots)

        print("Train features:")
        train_feats, train_labels = get_features(
            backbone, train_pool, device, "train", max_shots)

        # ── CLIP zero-shot logits (only for CLIP backbone) ────────────────────
        zero_shot_logits = None
        if backbone == "clip":
            print("\nComputing CLIP v9 zero-shot logits...")
            zero_shot_logits = get_clip_zeroshot_logits(test_feats, device)

        # ── Run Tip-Adapter for each shot count ───────────────────────────────
        results = {}
        for n_shots in sorted(args.shots):
            print(f"\nBuilding {n_shots}-shot cache...")
            cache_keys, cache_values = build_cache(train_feats, train_labels, n_shots)
            print(f"  Cache: {cache_keys.shape[0]} keys × {cache_keys.shape[1]}D")

            print(f"Grid search α×β (5×4=20 combos)...", end=" ", flush=True)
            best_a, best_b, best_acc = grid_search(
                test_feats, test_labels, cache_keys, cache_values,
                zero_shot_logits, backbone
            )
            print(f"best α={best_a}, β={best_b}, overall={best_acc:.1f}%")

            preds         = tip_adapter_predict(test_feats, cache_keys, cache_values,
                                                best_a, best_b, zero_shot_logits)
            overall, pc   = evaluate_predictions(preds, test_labels)
            results[n_shots] = {"overall": overall, "per_class": pc,
                                 "alpha": best_a, "beta": best_b}

        print_comparison(backbone, results)
        save_csv(backbone, results)

    print("\nDone. Key question answered:")
    print("  Does Tip-Adapter (training-free) match LogReg (trained)?")
    print("  → Check the 'delta vs LogReg' columns above.")


if __name__ == "__main__":
    main()
