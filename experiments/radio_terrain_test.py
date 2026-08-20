"""
Purpose: Few-shot linear probe on frozen NVIDIA RADIOv2.5-B features using AI4Mars dataset.
         RADIO is a multi-teacher distillation model (DINOv2 + CLIP + SAM → single encoder),
         testing whether combining multiple pretraining paradigms improves over single-teacher
         models (DINOv2 ViT-B 91.3%, DINOv3 ViT-L 92.3%) for Mars terrain classification.
Inputs:  AI4Mars train labels (few-shot sampling) + gold-standard test set (287 images)
Outputs: Per-class accuracy by shot count, CSV saved to results/
         Feature cache saved to results/feature_cache/radio_*
How to run:
    python3 -u experiments/radio_terrain_test.py | tee /tmp/radio_log.txt
    python3 -u experiments/radio_terrain_test.py --shots 1000  # 1000-shot only
    python3 -u experiments/radio_terrain_test.py --shots 10    # smoke test
Note: Loads via torch.hub from NVlabs/RADIO (downloads ~400MB on first run, then cached).
      Expected runtime: ~25–35 min on CPU (~500–700ms/img estimate).
Reference: Ranzinger et al. (2024) RADIO: Radio Active Distillation Is Over-parameterised.
           RADIOv2.5: arXiv:2412.07679; C-RADIOv4: arXiv:2601.17237
           https://huggingface.co/nvidia/RADIO
           https://github.com/NVlabs/RADIO
Project: Onboard Visual Foundation Models for Mars Terrain Perception and Rover Navigation
"""

import argparse
import csv
import os
import random
import time

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize

# ── Paths ─────────────────────────────────────────────────────────────────────
AI4MARS_BASE = "/mnt/c/Users/DELL/Desktop/Thesis/github source/ai4mars-dataset-merged-0.1"
IMAGES_DIR   = os.path.join(AI4MARS_BASE, "msl/images/edr")
TRAIN_LABELS = os.path.join(AI4MARS_BASE, "msl/labels/train")
TEST_LABELS  = os.path.join(AI4MARS_BASE, "msl/labels/test/masked-gold-min3-100agree")
RESULTS_DIR  = os.path.join(os.path.dirname(__file__), "results")
CACHE_DIR    = os.path.join(RESULTS_DIR, "feature_cache")

RADIO_VERSION       = "radio_v2.5-b"
CLASS_NAMES         = ["soil", "bedrock", "sand", "big_rock"]
IGNORE_PIXEL        = 255
SUPERVISED_BASELINE = 96.67

# ── Baselines ─────────────────────────────────────────────────────────────────
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


# ── Model ─────────────────────────────────────────────────────────────────────

def load_model(device: str):
    print(f"Loading NVIDIA RADIOv2.5-B via torch.hub  [version: {RADIO_VERSION}]")
    print(f"  Multi-teacher distillation: DINOv2 + CLIP + SAM → single encoder")
    print(f"  Expected: ~98M params, 768-d summary — cached after first download")
    t0 = time.perf_counter()
    model = torch.hub.load(
        "NVlabs/RADIO",
        "radio_model",
        version=RADIO_VERSION,
        progress=True,
        skip_validation=True,
        trust_repo=True,
    )
    model = model.to(device).eval()
    load_s   = time.perf_counter() - t0
    n_params = sum(p.numel() for p in model.parameters()) / 1e6

    # Standard ImageNet preprocessing (RADIO accepts 224×224 or 432×432)
    preprocessor = T.Compose([
        T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=(0.481, 0.458, 0.408), std=(0.269, 0.261, 0.276)),
    ])

    print(f"  Loaded in {load_s:.1f}s  |  Params: {n_params:.1f}M  |  Device: {device}")
    print(f"  Input: 224×224, RADIO ImageNet normalisation")
    return model, preprocessor


# ── Feature extraction ────────────────────────────────────────────────────────

def extract_features(model, preprocessor, image_paths, labels, device: str, desc: str = ""):
    """Extract L2-normalised RADIO summary features (analogous to CLS token)."""
    features = []
    n  = len(image_paths)
    t0 = time.perf_counter()

    for i, path in enumerate(image_paths):
        if (i + 1) % 50 == 0 or i == n - 1:
            elapsed = time.perf_counter() - t0
            eta     = elapsed / (i + 1) * (n - i - 1)
            print(f"  {desc} {i+1}/{n}  "
                  f"({elapsed/60:.1f}min elapsed, ~{eta/60:.1f}min remaining)", flush=True)

        image = Image.open(path).convert("RGB")

        with torch.no_grad():
            x = preprocessor(image)
            if x.dim() == 3:
                x = x.unsqueeze(0)
            x = x.to(device)
            summary, _ = model(x)          # summary: [1, D], spatial: [1, N, D]
            feat = summary.cpu().numpy().squeeze()  # [D]

        features.append(feat)

    features = np.array(features, dtype=np.float32)
    features = normalize(features)
    return features, np.array(labels)


# ── Ground truth ──────────────────────────────────────────────────────────────

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


# ── Few-shot sampling ─────────────────────────────────────────────────────────

def sample_n_per_class(pairs, n_per_class: int, seed: int = RANDOM_SEED):
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


# ── Evaluation ────────────────────────────────────────────────────────────────

def run_few_shot(train_feats, train_labels, test_feats, test_labels, n_shots: int):
    rng = np.random.RandomState(RANDOM_SEED)
    sampled_idx = []
    for c in range(len(CLASS_NAMES)):
        idx    = np.where(train_labels == c)[0]
        chosen = rng.choice(idx, size=min(n_shots, len(idx)), replace=False)
        sampled_idx.extend(chosen.tolist())

    clf = LogisticRegression(
        C=0.316, max_iter=1000, random_state=RANDOM_SEED,
        multi_class="multinomial", solver="lbfgs"
    )
    clf.fit(train_feats[sampled_idx], train_labels[sampled_idx])
    preds = clf.predict(test_feats)

    correct = {i: 0 for i in range(len(CLASS_NAMES))}
    total   = {i: 0 for i in range(len(CLASS_NAMES))}
    for pred, gt in zip(preds, test_labels):
        total[gt]   += 1
        correct[gt] += int(pred == gt)

    per_class = {
        CLASS_NAMES[i]: (correct[i] / total[i] * 100 if total[i] > 0 else None)
        for i in range(len(CLASS_NAMES))
    }
    overall = sum(correct.values()) / sum(total.values()) * 100
    return overall, per_class


# ── Reporting ─────────────────────────────────────────────────────────────────

def print_results(results: dict):
    shots = sorted(results.keys())
    print("\n" + "=" * 110)
    print("NVIDIA RADIOv2.5-B Few-Shot Linear Probe — AI4Mars (287 test images)")
    print("Multi-teacher (DINOv2+CLIP+SAM) vs single-teacher: DINOv2-B (91.3%), DINOv3-L (92.3%)")
    print("=" * 110)

    for metric in ["overall"] + CLASS_NAMES:
        row = f"{metric:<10}"
        for s in shots:
            r_val  = results[s]["overall"] if metric == "overall" else results[s]["per_class"].get(metric)
            v2b    = (DINOV2_VITB_BASELINES.get(s, {}).get("overall")
                      if metric == "overall" else DINOV2_VITB_BASELINES.get(s, {}).get(metric))
            v3l    = (DINOV3_VITL_BASELINES.get(s, {}).get("overall")
                      if metric == "overall" else DINOV3_VITL_BASELINES.get(s, {}).get(metric))
            r_str  = f"{r_val:>6.1f}%"  if r_val  is not None else "   N/A"
            v2b_str= f"{v2b:>6.1f}%"   if v2b    is not None else "   N/A"
            v3l_str= f"{v3l:>6.1f}%"   if v3l    is not None else "   N/A"
            d_v2b  = f"{(r_val-v2b):>+6.1f}%" if (r_val and v2b) else "    N/A"
            row   += f"  [{s:>4}shot] RADIO:{r_str} v2B:{v2b_str} Δ:{d_v2b} v3L:{v3l_str}"
        print(row)

    print("=" * 110)
    print("\nBedrock — multi-teacher vs best single-teacher:")
    for s in shots:
        r_b  = results[s]["per_class"].get("bedrock")
        v2b_b= DINOV2_VITB_BASELINES.get(s, {}).get("bedrock")
        if r_b is not None and v2b_b is not None:
            print(f"  {s:>5}-shot:  v2B {v2b_b:.1f}%  →  RADIO {r_b:.1f}%  "
                  f"(Δ {r_b-v2b_b:+.1f}%)")

    if 1000 in results:
        r_ov  = results[1000]["overall"]
        v2b_ov= DINOV2_VITB_BASELINES[1000]["overall"]
        v3l_ov= DINOV3_VITL_BASELINES[1000]["overall"]
        print(f"\n>>> 1000-shot: v2B {v2b_ov:.2f}%  RADIO {r_ov:.2f}%  v3L {v3l_ov:.2f}%  "
              f"|  Gap vs supervised: {r_ov - SUPERVISED_BASELINE:+.2f}%")


def save_csv(results: dict, avg_ms_extract: float):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path  = os.path.join(RESULTS_DIR, "radio_terrain_few_shot.csv")
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
        row = {
            "shots":             s,
            "overall":           f"{r['overall']:.2f}",
            "gap_vs_supervised": f"{r['overall'] - SUPERVISED_BASELINE:.2f}",
            "avg_ms_extract":    f"{avg_ms_extract:.1f}",
        }
        for name in CLASS_NAMES:
            val       = r["per_class"].get(name)
            row[name] = f"{val:.2f}" if val is not None else "N/A"
        row["dinov2_vitb_overall"]         = f"{v2b.get('overall',0):.2f}" if v2b.get("overall") else "N/A"
        row["dinov2_vitb_bedrock"]         = f"{v2b.get('bedrock',0):.2f}" if v2b.get("bedrock") else "N/A"
        row["dinov3_vitl_overall"]         = f"{v3l.get('overall',0):.2f}" if v3l.get("overall") else "N/A"
        row["dinov3_vitl_bedrock"]         = f"{v3l.get('bedrock',0):.2f}" if v3l.get("bedrock") else "N/A"
        row["delta_vs_dinov2_vitb_overall"]= (f"{r['overall'] - v2b['overall']:+.2f}"
                                              if v2b.get("overall") else "N/A")
        rows.append(row)

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nCSV saved → {path}")
    return path


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="RADIO RADIOv2.5-B few-shot terrain probe")
    parser.add_argument("--shots", type=int, nargs="+", default=DEFAULT_SHOTS)
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    device = "cpu"
    os.makedirs(CACHE_DIR, exist_ok=True)

    cache_test_feats   = os.path.join(CACHE_DIR, "radio_test_287_feats.npy")
    cache_test_labels  = os.path.join(CACHE_DIR, "radio_test_287_labels.npy")
    cache_train_feats  = os.path.join(CACHE_DIR, "radio_train_1000_feats.npy")
    cache_train_labels = os.path.join(CACHE_DIR, "radio_train_1000_labels.npy")

    max_shots_needed = max(args.shots)
    train_cache_ok = (
        os.path.exists(cache_train_feats) and
        np.load(cache_train_feats, mmap_mode="r").shape[0] >= max_shots_needed * len(CLASS_NAMES)
    )
    use_cache = (not args.no_cache and
                 os.path.exists(cache_test_feats) and os.path.exists(cache_test_labels) and
                 train_cache_ok and os.path.exists(cache_train_labels))

    if use_cache:
        print("Loading features from cache...")
        test_feats_arr   = np.load(cache_test_feats)
        test_labels_arr  = np.load(cache_test_labels)
        train_feats_arr  = np.load(cache_train_feats)
        train_labels_arr = np.load(cache_train_labels)
        print(f"  test: {test_feats_arr.shape}   train: {train_feats_arr.shape}")
        ms_per_img = 0.0
    else:
        model, preprocessor = load_model(device)

        print("\nLoading test set...")
        test_pairs = load_split(TEST_LABELS)
        print(f"  Test images: {len(test_pairs)}")

        print(f"\nExtracting test features ({len(test_pairs)} images)...")
        t_test = time.perf_counter()
        test_feats_arr, test_labels_arr = extract_features(
            model, preprocessor,
            [p for p, _ in test_pairs], [l for _, l in test_pairs],
            device, desc="test"
        )
        ms_per_img = (time.perf_counter() - t_test) / len(test_pairs) * 1000
        print(f"  Done — avg {ms_per_img:.0f}ms/img  feat shape: {test_feats_arr.shape}")

        max_shots = max(args.shots)
        print(f"\nLoading train set (up to {max_shots}/class)...")
        train_all     = load_split(TRAIN_LABELS)
        train_sampled = sample_n_per_class(train_all, max_shots)
        by_class = {i: 0 for i in range(4)}
        for _, c in train_sampled:
            by_class[c] += 1
        print("  Train sample: " + ", ".join(f"{CLASS_NAMES[i]} {by_class[i]}" for i in range(4)))

        print(f"\nExtracting train features ({len(train_sampled)} images)...")
        print(f"  Estimated time: {len(train_sampled) * ms_per_img / 1000 / 60:.0f}min at {ms_per_img:.0f}ms/img")
        train_feats_arr, train_labels_arr = extract_features(
            model, preprocessor,
            [p for p, _ in train_sampled], [l for _, l in train_sampled],
            device, desc="train"
        )

        np.save(cache_test_feats,   test_feats_arr)
        np.save(cache_test_labels,  test_labels_arr)
        np.save(cache_train_feats,  train_feats_arr)
        np.save(cache_train_labels, train_labels_arr)
        print(f"  Features cached → {CACHE_DIR}/radio_*")

    print("\nRunning few-shot linear probes...")
    results = {}
    for n_shots in sorted(args.shots):
        print(f"  {n_shots}-shot...", end=" ", flush=True)
        t0 = time.perf_counter()
        overall, per_class = run_few_shot(
            train_feats_arr, train_labels_arr,
            test_feats_arr,  test_labels_arr,
            n_shots
        )
        elapsed = time.perf_counter() - t0
        results[n_shots] = {"overall": overall, "per_class": per_class}
        v2b_ov = DINOV2_VITB_BASELINES.get(n_shots, {}).get("overall", 0)
        print(f"overall {overall:.1f}%  (DINOv2-B {v2b_ov:.1f}%  Δ{overall-v2b_ov:+.1f}%)  [{elapsed:.1f}s]")

    print_results(results)
    save_csv(results, ms_per_img)

    print("\n=== FINAL RESULT — D5 NVIDIA RADIOv2.5-B ===")
    if 1000 in results:
        r_ov  = results[1000]["overall"]
        v2b_ov= DINOV2_VITB_BASELINES[1000]["overall"]
        v3l_ov= DINOV3_VITL_BASELINES[1000]["overall"]
        r_br  = results[1000]["per_class"].get("bedrock", 0)
        v2b_br= DINOV2_VITB_BASELINES[1000].get("bedrock", 0)
        status= "✅ BEATS DINOv2-B" if r_ov > v2b_ov else "❌ BELOW DINOv2-B"
        print(f"  1000-shot Overall: DINOv2-B {v2b_ov:.2f}% → RADIO {r_ov:.2f}%  "
              f"(Δ {r_ov-v2b_ov:+.2f}%)  → {status}")
        print(f"  1000-shot Bedrock: DINOv2-B {v2b_br:.2f}% → RADIO {r_br:.2f}%  "
              f"(Δ {r_br-v2b_br:+.2f}%)")
        print(f"  vs DINOv3 ViT-L (best): {r_ov:.2f}% vs {v3l_ov:.2f}%  (Δ {r_ov-v3l_ov:+.2f}%)")
        print(f"  Gap vs supervised (96.67%): {r_ov - SUPERVISED_BASELINE:+.2f}%")
        paradigm = "multi-teacher RADIO" if r_ov > v2b_ov else "single-teacher DINOv2"
        print(f"  Multi-teacher verdict: {paradigm} wins for Mars terrain")


if __name__ == "__main__":
    main()
