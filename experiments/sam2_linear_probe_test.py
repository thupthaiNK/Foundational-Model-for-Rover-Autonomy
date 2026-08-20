"""
Purpose: Few-shot linear probe on frozen SAM2 Hiera-tiny image encoder features using AI4Mars.
         SAM2 (Ravi et al., 2024) is trained with a segmentation objective (mask prediction),
         not classification. This tests whether segmentation pretraining produces features
         useful for terrain texture classification — expected to fail similarly to Depth Anything V2
         (task-specific pretraining ≠ general texture features).
         Uses SAM2 Hiera-tiny image encoder (27.2M params), features extracted at 224x224
         (non-native; SAM2 trained at 1024x1024) via global average pooling -> 256-d.
Inputs:  AI4Mars train labels (few-shot sampling) + gold-standard test set (287 images)
Outputs: Per-class accuracy by shot count, CSV saved to results/
         Feature cache saved to results/feature_cache/sam2lp_*
How to run:
    python3 -u experiments/sam2_linear_probe_test.py | tee /tmp/sam2lp_log.txt
    python3 -u experiments/sam2_linear_probe_test.py --shots 1000
Note: Requires sam2 package and SAM2 Hiera-tiny checkpoint (cached at HuggingFace).
      Expected runtime: ~10 min total on CPU (172ms/img).
Reference: Ravi et al. (2024) SAM 2: Segment Anything in Images and Videos. arXiv:2408.00714.
Project: Onboard Visual Foundation Models for Mars Terrain Perception and Rover Navigation
"""

import argparse
import csv
import os
import random
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize

# -- Paths --------------------------------------------------------------------
AI4MARS_BASE = "/mnt/c/Users/DELL/Desktop/Thesis/github source/ai4mars-dataset-merged-0.1"
IMAGES_DIR   = os.path.join(AI4MARS_BASE, "msl/images/edr")
TRAIN_LABELS = os.path.join(AI4MARS_BASE, "msl/labels/train")
TEST_LABELS  = os.path.join(AI4MARS_BASE, "msl/labels/test/masked-gold-min3-100agree")
RESULTS_DIR  = os.path.join(os.path.dirname(__file__), "results")
CACHE_DIR    = os.path.join(RESULTS_DIR, "feature_cache")

SAM2_CKPT = ("/home/thupthai/.cache/huggingface/hub/"
             "models--facebook--sam2-hiera-tiny/snapshots/"
             "7c218beaf0bb87874785f32b582f640134fc1c09/sam2_hiera_tiny.pt")
SAM2_CFG  = "sam2_hiera_t.yaml"

CLASS_NAMES         = ["soil", "bedrock", "sand", "big_rock"]
IGNORE_PIXEL        = 255
SUPERVISED_BASELINE = 96.67

DINOV2_VITB_BASELINES = {
    10:   {"overall": 63.1,  "bedrock": 75.0},
    50:   {"overall": 78.0,  "bedrock": 54.3},
    100:  {"overall": 82.9,  "bedrock": 60.9},
    500:  {"overall": 89.5,  "bedrock": 85.9},
    1000: {"overall": 91.3,  "bedrock": 87.0},
}
DINOV3_VITL_BASELINES = {
    10:   {"overall": 70.0,  "bedrock": None},
    50:   {"overall": 83.3,  "bedrock": None},
    100:  {"overall": 84.0,  "bedrock": None},
    500:  {"overall": 92.7,  "bedrock": None},
    1000: {"overall": 92.3,  "bedrock": 89.1},
}

DEFAULT_SHOTS = [10, 50, 100, 500, 1000]
RANDOM_SEED   = 42

TRANSFORM = T.Compose([
    T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def load_model(device):
    from sam2.build_sam import build_sam2
    print("Loading SAM2 Hiera-tiny image encoder  [facebook/sam2-hiera-tiny]")
    print("  Pretraining: Segmentation mask prediction (SAM2 objective)")
    print("  Features: vision_features global avg pool -> 256-d (at 224x224 input)")
    t0 = time.perf_counter()
    sam2_model    = build_sam2(SAM2_CFG, SAM2_CKPT, device=device)
    image_encoder = sam2_model.image_encoder.eval()
    n_params = sum(p.numel() for p in image_encoder.parameters()) / 1e6
    print(f"  Loaded in {time.perf_counter()-t0:.1f}s  |  Params: {n_params:.1f}M  |  Device: {device}")
    return image_encoder


def extract_features(image_encoder, image_paths, labels, device, desc=""):
    features = []
    n  = len(image_paths)
    t0 = time.perf_counter()
    for i, path in enumerate(image_paths):
        if (i + 1) % 50 == 0 or i == n - 1:
            elapsed = time.perf_counter() - t0
            eta     = elapsed / (i + 1) * (n - i - 1)
            print(f"  {desc} {i+1}/{n}  ({elapsed/60:.1f}min elapsed, ~{eta/60:.1f}min remaining)",
                  flush=True)
        image = Image.open(path).convert("RGB")
        with torch.no_grad():
            x    = TRANSFORM(image).unsqueeze(0).to(device)
            out  = image_encoder(x)
            feat = out["vision_features"].mean(dim=[2, 3]).cpu().numpy().squeeze()
        features.append(feat)
    features = np.array(features, dtype=np.float32)
    return normalize(features), np.array(labels)


def dominant_class(label_path):
    label = np.array(Image.open(label_path))
    valid = label[label != IGNORE_PIXEL]
    if len(valid) == 0:
        return None
    return int(np.argmax(np.bincount(valid, minlength=4)))


def load_split(label_dir):
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


def sample_n_per_class(pairs, n_per_class, seed=RANDOM_SEED):
    rng      = random.Random(seed)
    by_class = {c: [] for c in range(len(CLASS_NAMES))}
    for pair in pairs:
        by_class[pair[1]].append(pair)
    sampled = []
    for c in range(len(CLASS_NAMES)):
        pool = by_class[c]
        rng.shuffle(pool)
        sampled.extend(pool[:n_per_class])
    return sampled


def run_few_shot(train_feats, train_labels, test_feats, test_labels, n_shots):
    rng = np.random.RandomState(RANDOM_SEED)
    sampled_idx = []
    for c in range(len(CLASS_NAMES)):
        idx    = np.where(train_labels == c)[0]
        chosen = rng.choice(idx, size=min(n_shots, len(idx)), replace=False)
        sampled_idx.extend(chosen.tolist())
    clf = LogisticRegression(C=0.316, max_iter=1000, random_state=RANDOM_SEED,
                             multi_class="multinomial", solver="lbfgs")
    clf.fit(train_feats[sampled_idx], train_labels[sampled_idx])
    preds = clf.predict(test_feats)
    correct = {i: 0 for i in range(len(CLASS_NAMES))}
    total   = {i: 0 for i in range(len(CLASS_NAMES))}
    for pred, gt in zip(preds, test_labels):
        total[gt]   += 1
        correct[gt] += int(pred == gt)
    per_class = {CLASS_NAMES[i]: (correct[i]/total[i]*100 if total[i] > 0 else None)
                 for i in range(len(CLASS_NAMES))}
    overall = sum(correct.values()) / sum(total.values()) * 100
    return overall, per_class


def print_results(results):
    shots = sorted(results.keys())
    print("\n" + "="*110)
    print("SAM2 Hiera-tiny Few-Shot Linear Probe -- AI4Mars (287 test images)")
    print("Segmentation pretraining vs DINO: DINOv2-B (91.3%), DINOv3-L (92.3%)")
    print("="*110)
    for metric in ["overall"] + CLASS_NAMES:
        row = f"{metric:<10}"
        for s in shots:
            sv  = results[s]["overall"] if metric == "overall" else results[s]["per_class"].get(metric)
            v2b = (DINOV2_VITB_BASELINES.get(s,{}).get("overall") if metric == "overall"
                   else DINOV2_VITB_BASELINES.get(s,{}).get(metric))
            v3l = (DINOV3_VITL_BASELINES.get(s,{}).get("overall") if metric == "overall"
                   else DINOV3_VITL_BASELINES.get(s,{}).get(metric))
            sv_s  = f"{sv:>6.1f}%"  if sv  is not None else "   N/A"
            v2b_s = f"{v2b:>6.1f}%" if v2b is not None else "   N/A"
            v3l_s = f"{v3l:>6.1f}%" if v3l is not None else "   N/A"
            dv2b  = f"{sv-v2b:>+6.1f}%" if (sv is not None and v2b) else "    N/A"
            row  += f"  [{s:>4}shot] SAM2:{sv_s} v2B:{v2b_s} D:{dv2b} v3L:{v3l_s}"
        print(row)
    print("="*110)
    print("\nBedrock -- SAM2 segmentation vs DINOv2-B:")
    for s in shots:
        sb   = results[s]["per_class"].get("bedrock")
        v2bb = DINOV2_VITB_BASELINES.get(s,{}).get("bedrock")
        if sb is not None and v2bb is not None:
            print(f"  {s:>5}-shot:  v2B {v2bb:.1f}%  ->  SAM2 {sb:.1f}%  (D {sb-v2bb:+.1f}%)")
    if 1000 in results:
        so = results[1000]["overall"]
        print(f"\n>>> 1000-shot: v2B {DINOV2_VITB_BASELINES[1000]['overall']:.2f}%"
              f"  SAM2 {so:.2f}%  v3L {DINOV3_VITL_BASELINES[1000]['overall']:.2f}%"
              f"  |  Gap vs supervised: {so-SUPERVISED_BASELINE:+.2f}%")


def save_csv(results, avg_ms):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, "sam2_linear_probe_few_shot.csv")
    shots = sorted(results.keys())
    fieldnames = (["shots", "overall", "gap_vs_supervised"] + CLASS_NAMES +
                  ["dinov2_vitb_overall", "dinov2_vitb_bedrock",
                   "dinov3_vitl_overall", "dinov3_vitl_bedrock",
                   "delta_vs_dinov2_vitb_overall", "avg_ms_extract"])
    rows = []
    for s in shots:
        r   = results[s]
        v2b = DINOV2_VITB_BASELINES.get(s, {})
        v3l = DINOV3_VITL_BASELINES.get(s, {})
        row = {"shots": s,
               "overall": f"{r['overall']:.2f}",
               "gap_vs_supervised": f"{r['overall']-SUPERVISED_BASELINE:.2f}",
               "avg_ms_extract": f"{avg_ms:.1f}"}
        for name in CLASS_NAMES:
            val = r["per_class"].get(name)
            row[name] = f"{val:.2f}" if val is not None else "N/A"
        row["dinov2_vitb_overall"]          = f"{v2b['overall']:.2f}" if v2b.get("overall") else "N/A"
        row["dinov2_vitb_bedrock"]          = f"{v2b['bedrock']:.2f}" if v2b.get("bedrock") else "N/A"
        row["dinov3_vitl_overall"]          = f"{v3l['overall']:.2f}" if v3l.get("overall") else "N/A"
        row["dinov3_vitl_bedrock"]          = f"{v3l['bedrock']:.2f}" if v3l.get("bedrock") else "N/A"
        row["delta_vs_dinov2_vitb_overall"] = (f"{r['overall']-v2b['overall']:+.2f}"
                                               if v2b.get("overall") else "N/A")
        rows.append(row)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nCSV saved -> {path}")
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shots", type=int, nargs="+", default=DEFAULT_SHOTS)
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    device = "cpu"
    os.makedirs(CACHE_DIR, exist_ok=True)

    cache_tf = os.path.join(CACHE_DIR, "sam2lp_test_287_feats.npy")
    cache_tl = os.path.join(CACHE_DIR, "sam2lp_test_287_labels.npy")
    cache_rf = os.path.join(CACHE_DIR, "sam2lp_train_1000_feats.npy")
    cache_rl = os.path.join(CACHE_DIR, "sam2lp_train_1000_labels.npy")

    max_shots = max(args.shots)
    train_ok = (os.path.exists(cache_rf) and
                np.load(cache_rf, mmap_mode="r").shape[0] >= max_shots * len(CLASS_NAMES))
    use_cache = (not args.no_cache and
                 os.path.exists(cache_tf) and os.path.exists(cache_tl) and
                 train_ok and os.path.exists(cache_rl))

    ms_per_img = 0.0
    if use_cache:
        print("Loading features from cache...")
        test_feats_arr   = np.load(cache_tf)
        test_labels_arr  = np.load(cache_tl)
        train_feats_arr  = np.load(cache_rf)
        train_labels_arr = np.load(cache_rl)
        print(f"  test: {test_feats_arr.shape}   train: {train_feats_arr.shape}")
    else:
        image_encoder = load_model(device)

        print("\nLoading test set...")
        test_pairs = load_split(TEST_LABELS)
        print(f"  Test images: {len(test_pairs)}")

        print(f"\nExtracting test features ({len(test_pairs)} images)...")
        t0 = time.perf_counter()
        test_feats_arr, test_labels_arr = extract_features(
            image_encoder, [p for p,_ in test_pairs], [l for _,l in test_pairs],
            device, desc="test")
        ms_per_img = (time.perf_counter()-t0) / len(test_pairs) * 1000
        print(f"  Done -- avg {ms_per_img:.0f}ms/img  feat shape: {test_feats_arr.shape}")

        print(f"\nLoading train set (up to {max_shots}/class)...")
        train_all     = load_split(TRAIN_LABELS)
        train_sampled = sample_n_per_class(train_all, max_shots)
        by_class = {i: 0 for i in range(4)}
        for _, c in train_sampled:
            by_class[c] += 1
        print("  Train sample: " + ", ".join(f"{CLASS_NAMES[i]} {by_class[i]}" for i in range(4)))

        print(f"\nExtracting train features ({len(train_sampled)} images)...")
        print(f"  Estimated time: {len(train_sampled)*ms_per_img/1000/60:.0f}min at {ms_per_img:.0f}ms/img")
        train_feats_arr, train_labels_arr = extract_features(
            image_encoder, [p for p,_ in train_sampled], [l for _,l in train_sampled],
            device, desc="train")

        np.save(cache_tf, test_feats_arr)
        np.save(cache_tl, test_labels_arr)
        np.save(cache_rf, train_feats_arr)
        np.save(cache_rl, train_labels_arr)
        print(f"  Features cached -> {CACHE_DIR}/sam2lp_*")

    print("\nRunning few-shot linear probes...")
    results = {}
    for n_shots in sorted(args.shots):
        print(f"  {n_shots}-shot...", end=" ", flush=True)
        t0 = time.perf_counter()
        overall, per_class = run_few_shot(train_feats_arr, train_labels_arr,
                                          test_feats_arr, test_labels_arr, n_shots)
        elapsed = time.perf_counter() - t0
        results[n_shots] = {"overall": overall, "per_class": per_class}
        v2b = DINOV2_VITB_BASELINES.get(n_shots, {}).get("overall", 0)
        print(f"overall {overall:.1f}%  (DINOv2-B {v2b:.1f}%  D{overall-v2b:+.1f}%)  [{elapsed:.1f}s]")

    print_results(results)
    save_csv(results, ms_per_img)

    print("\n=== FINAL RESULT -- SAM2 Hiera-tiny (segmentation pretraining linear probe) ===")
    if 1000 in results:
        so   = results[1000]["overall"]
        v2bo = DINOV2_VITB_BASELINES[1000]["overall"]
        v3lo = DINOV3_VITL_BASELINES[1000]["overall"]
        sb   = results[1000]["per_class"].get("bedrock") or 0
        v2bb = DINOV2_VITB_BASELINES[1000]["bedrock"]
        status = "BEATS DINOv2-B" if so > v2bo else "BELOW DINOv2-B"
        print(f"  1000-shot Overall: DINOv2-B {v2bo:.2f}% vs SAM2 {so:.2f}%  (D {so-v2bo:+.2f}%)  -> {status}")
        print(f"  1000-shot Bedrock: DINOv2-B {v2bb:.2f}% vs SAM2 {sb:.2f}%  (D {sb-v2bb:+.2f}%)")
        print(f"  vs Depth Anything V2 (task-specific, 1000-shot): SAM2 {so:.2f}% vs DepthV2 36.6%")
        print(f"  vs DINOv3 ViT-L (best): {so:.2f}% vs {v3lo:.2f}%  (D {so-v3lo:+.2f}%)")
        print(f"  Gap vs supervised (96.67%): {so-SUPERVISED_BASELINE:+.2f}%")


if __name__ == "__main__":
    main()
