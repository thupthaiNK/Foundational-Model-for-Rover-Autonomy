"""
Purpose: Few-shot linear probe on frozen DINOv3 ViT-S/16 using multi-layer feature
         concatenation. Concatenates CLS tokens from the last 4 transformer layers
         (4×384 = 1536-d) instead of only the final CLS token (384-d). Tests whether
         richer multi-layer representations improve Mars terrain classification over the
         single-layer baseline (90.2% at 1000-shot).
Inputs:  AI4Mars train labels (few-shot sampling) + gold-standard test set (287 images)
Outputs: Per-class accuracy by shot count, CSV saved to results/
         Feature cache saved to results/feature_cache/dinov3_multilayer_*
How to run:
    python3 -u experiments/dinov3_multilayer_terrain_test.py              # full run
    python3 -u experiments/dinov3_multilayer_terrain_test.py --shots 1000
    python3 -u experiments/dinov3_multilayer_terrain_test.py --shots 10   # quick smoke test
Note: Requires HuggingFace token (gated model). Run `hf auth login` first or set HF_TOKEN.
Reference: Siméoni et al. (2025) DINOv3. Meta AI. arXiv:2508.10104
           Hao et al. (2024) Rock Type Classification from DINOv2 Features. arXiv:2407.18100
           (multi-layer concat → 100% accuracy on geological rock type)
Project: Onboard Visual Foundation Models for Mars Terrain Perception and Rover Navigation
"""

import argparse
import csv
import os
import random
import time

import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize
from transformers import AutoImageProcessor, AutoModel

# ── Paths ─────────────────────────────────────────────────────────────────────
AI4MARS_BASE = "/mnt/c/Users/DELL/Desktop/Thesis/github source/ai4mars-dataset-merged-0.1"
IMAGES_DIR   = os.path.join(AI4MARS_BASE, "msl/images/edr")
TRAIN_LABELS = os.path.join(AI4MARS_BASE, "msl/labels/train")
TEST_LABELS  = os.path.join(AI4MARS_BASE, "msl/labels/test/masked-gold-min3-100agree")
RESULTS_DIR  = os.path.join(os.path.dirname(__file__), "results")
CACHE_DIR    = os.path.join(RESULTS_DIR, "feature_cache")

MODEL_ID            = "facebook/dinov3-vits16-pretrain-lvd1689m"
CLASS_NAMES         = ["soil", "bedrock", "sand", "big_rock"]
IGNORE_PIXEL        = 255
SUPERVISED_BASELINE = 96.67
N_LAYERS_CONCAT     = 4   # concat CLS from last N layers → 4×384 = 1536-d

# ── Baselines for comparison ──────────────────────────────────────────────────
DINOV3_SINGLE_BASELINES = {
    10:   {"overall": 58.5,  "soil": 51.2,  "bedrock": 65.2, "sand": 62.5},
    50:   {"overall": 75.3,  "soil": None,  "bedrock": 51.1, "sand": None},
    100:  {"overall": 80.5,  "soil": None,  "bedrock": 60.9, "sand": None},
    500:  {"overall": 89.5,  "soil": None,  "bedrock": 80.4, "sand": None},
    1000: {"overall": 90.2,  "soil": 97.6,  "bedrock": 83.7, "sand": 86.1},
}

DINOV2_BASELINES = {
    10:   {"overall": 56.79, "bedrock": 56.52},
    100:  {"overall": 81.18, "bedrock": 55.43},
    1000: {"overall": 89.90, "bedrock": 84.78},
}

DEFAULT_SHOTS = [10, 50, 100, 500, 1000]
RANDOM_SEED   = 42


# ── Model ─────────────────────────────────────────────────────────────────────

def load_model(device: str):
    feat_dim = 384 * N_LAYERS_CONCAT
    print(f"Loading DINOv3 ViT-S/16 (frozen)  [{MODEL_ID}]")
    print(f"  Mode: multi-layer concat (last {N_LAYERS_CONCAT} layers × 384-d = {feat_dim}-d)")
    t0 = time.perf_counter()
    processor = AutoImageProcessor.from_pretrained(MODEL_ID, token=True)
    model     = AutoModel.from_pretrained(MODEL_ID, token=True).to(device).eval()
    load_s    = time.perf_counter() - t0
    n_params  = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Loaded in {load_s:.1f}s  |  Params: {n_params:.1f}M  "
          f"|  Feature dim: {feat_dim}  |  Device: {device}")
    return model, processor


# ── Feature extraction ────────────────────────────────────────────────────────

def extract_features(model, processor, image_paths, labels, device: str, desc: str = ""):
    """Extract L2-normalised multi-layer DINOv3 CLS features.
    Concatenates CLS tokens from last N_LAYERS_CONCAT transformer layers → 1536-d.
    """
    features = []
    n  = len(image_paths)
    t0 = time.perf_counter()

    for i, path in enumerate(image_paths):
        if (i + 1) % 100 == 0 or i == n - 1:
            elapsed = time.perf_counter() - t0
            eta     = elapsed / (i + 1) * (n - i - 1)
            print(f"  {desc} {i+1}/{n}  ({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)")

        image  = Image.open(path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            # hidden_states: tuple of (n_layers+1) tensors, each [1, seq_len, 384]
            # Take CLS token (index 0) from last N_LAYERS_CONCAT layers
            cls_tokens = [
                outputs.hidden_states[-(N_LAYERS_CONCAT - k)][:, 0, :]
                for k in range(N_LAYERS_CONCAT - 1, -1, -1)
            ]  # ordered from deepest-N+1 → deepest
            feat = torch.cat(cls_tokens, dim=-1)  # [1, 1536]
            feat = feat.cpu().numpy().squeeze()    # [1536]

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

    X_train = train_feats[sampled_idx]
    y_train = train_labels[sampled_idx]

    clf = LogisticRegression(
        C=0.316, max_iter=1000, random_state=RANDOM_SEED,
        multi_class="multinomial", solver="lbfgs"
    )
    clf.fit(X_train, y_train)
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
    n_total = sum(total.values())
    overall = sum(correct.values()) / n_total * 100 if n_total > 0 else 0
    return overall, per_class


# ── Reporting ─────────────────────────────────────────────────────────────────

def print_results(results: dict):
    shots = sorted(results.keys())
    feat_dim = 384 * N_LAYERS_CONCAT
    print("\n" + "=" * 100)
    print(f"DINOv3 ViT-S/16 Multi-Layer Concat ({feat_dim}-d) — AI4Mars (287 test images)")
    print(f"Comparison: {feat_dim}-d multi-layer vs 384-d single-layer DINOv3")
    print("=" * 100)

    for metric in ["overall"] + CLASS_NAMES:
        row = f"{metric:<10}"
        for s in shots:
            ml_val  = results[s]["overall"] if metric == "overall" else results[s]["per_class"].get(metric)
            sl_val  = (DINOV3_SINGLE_BASELINES.get(s, {}).get("overall")
                       if metric == "overall"
                       else DINOV3_SINGLE_BASELINES.get(s, {}).get(metric))

            ml_str  = f"{ml_val:>6.1f}%" if ml_val is not None else "   N/A"
            sl_str  = f"{sl_val:>6.1f}%" if sl_val is not None else "   N/A"
            delta   = f"{(ml_val - sl_val):>+6.1f}%" if (ml_val is not None and sl_val is not None) else "    N/A"
            row    += f"  [{s:>4}shot] Multi:{ml_str} Single:{sl_str} Δ:{delta}"
        print(row)

    print("=" * 100)
    print(f"\nKey metric — Bedrock (main thesis contribution):")
    for s in shots:
        ml_b = results[s]["per_class"].get("bedrock")
        sl_b = DINOV3_SINGLE_BASELINES.get(s, {}).get("bedrock")
        if ml_b is not None and sl_b is not None:
            print(f"  {s:>5}-shot:  Multi-layer {ml_b:.1f}%  |  Single-layer {sl_b:.1f}%  "
                  f"(Δ {ml_b - sl_b:+.1f}%)")

    print(f"\nKey metric — Overall 1000-shot:")
    if 1000 in results:
        ml_ov = results[1000]["overall"]
        sl_ov = DINOV3_SINGLE_BASELINES[1000]["overall"]
        print(f"  Multi-layer  {ml_ov:.2f}%  |  Single-layer {sl_ov:.2f}%  "
              f"(Δ {ml_ov - sl_ov:+.2f}%)  |  Gap vs supervised: {ml_ov - SUPERVISED_BASELINE:+.2f}%")


def save_csv(results: dict, avg_ms_extract: float):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, "dinov3_multilayer_terrain_few_shot.csv")
    shots = sorted(results.keys())

    fieldnames = (["shots", "overall", "gap_vs_supervised", "feat_dim"] + CLASS_NAMES +
                  ["dinov3_single_overall", "dinov3_single_bedrock",
                   "delta_vs_single_overall", "delta_vs_single_bedrock",
                   "avg_ms_extract"])

    rows = []
    for s in shots:
        r  = results[s]
        sl = DINOV3_SINGLE_BASELINES.get(s, {})
        row = {
            "shots":             s,
            "overall":           f"{r['overall']:.2f}",
            "gap_vs_supervised": f"{r['overall'] - SUPERVISED_BASELINE:.2f}",
            "feat_dim":          384 * N_LAYERS_CONCAT,
            "avg_ms_extract":    f"{avg_ms_extract:.1f}",
        }
        for name in CLASS_NAMES:
            val      = r["per_class"].get(name)
            row[name] = f"{val:.2f}" if val is not None else "N/A"

        sl_ov = sl.get("overall")
        sl_br = sl.get("bedrock")
        ml_br = r["per_class"].get("bedrock")
        row["dinov3_single_overall"]   = f"{sl_ov:.2f}" if sl_ov is not None else "N/A"
        row["dinov3_single_bedrock"]   = f"{sl_br:.2f}" if sl_br is not None else "N/A"
        row["delta_vs_single_overall"] = (f"{r['overall'] - sl_ov:+.2f}"
                                          if sl_ov is not None else "N/A")
        row["delta_vs_single_bedrock"] = (f"{(ml_br or 0) - sl_br:+.2f}"
                                          if (ml_br is not None and sl_br is not None) else "N/A")
        rows.append(row)

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nCSV saved → {path}")
    return path


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=f"DINOv3 ViT-S/16 multi-layer concat ({N_LAYERS_CONCAT}×384={N_LAYERS_CONCAT*384}-d) terrain probe"
    )
    parser.add_argument("--shots", type=int, nargs="+", default=DEFAULT_SHOTS,
                        help="Shot counts to evaluate (default: 10 50 100 500 1000)")
    parser.add_argument("--no-cache", action="store_true",
                        help="Re-extract features even if cache exists")
    args = parser.parse_args()

    device = "cpu"

    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_test_feats  = os.path.join(CACHE_DIR, "dinov3_multilayer_test_287_feats.npy")
    cache_test_labels = os.path.join(CACHE_DIR, "dinov3_multilayer_test_287_labels.npy")
    cache_train_feats = os.path.join(CACHE_DIR, "dinov3_multilayer_train_1000_feats.npy")
    cache_train_labels= os.path.join(CACHE_DIR, "dinov3_multilayer_train_1000_labels.npy")

    max_shots_needed = max(args.shots)
    train_cache_ok = (
        os.path.exists(cache_train_feats) and
        np.load(cache_train_feats, mmap_mode="r").shape[0] >= max_shots_needed * len(CLASS_NAMES)
    )
    use_cache = (not args.no_cache and
                 os.path.exists(cache_test_feats) and
                 os.path.exists(cache_test_labels) and
                 train_cache_ok and
                 os.path.exists(cache_train_labels))

    if use_cache:
        print("Loading features from cache...")
        test_feats_arr   = np.load(cache_test_feats)
        test_labels_arr  = np.load(cache_test_labels)
        train_feats_arr  = np.load(cache_train_feats)
        train_labels_arr = np.load(cache_train_labels)
        print(f"  test:  {test_feats_arr.shape}   train: {train_feats_arr.shape}")
        ms_per_img = 0.0
    else:
        model, processor = load_model(device)

        print("\nLoading test set...")
        test_pairs = load_split(TEST_LABELS)
        print(f"  Test images: {len(test_pairs)}")

        print(f"\nExtracting test features ({len(test_pairs)} images)...")
        t_test = time.perf_counter()
        test_feats_arr, test_labels_arr = extract_features(
            model, processor,
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
        train_feats_arr, train_labels_arr = extract_features(
            model, processor,
            [p for p, _ in train_sampled], [l for _, l in train_sampled],
            device, desc="train"
        )

        np.save(cache_test_feats,   test_feats_arr)
        np.save(cache_test_labels,  test_labels_arr)
        np.save(cache_train_feats,  train_feats_arr)
        np.save(cache_train_labels, train_labels_arr)
        print(f"  Features cached → {CACHE_DIR}/dinov3_multilayer_*")

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
        elapsed  = time.perf_counter() - t0
        results[n_shots] = {"overall": overall, "per_class": per_class}
        sl_overall = DINOV3_SINGLE_BASELINES.get(n_shots, {}).get("overall", 0)
        delta      = overall - sl_overall
        print(f"overall {overall:.1f}%  (single-layer {sl_overall:.1f}%  Δ{delta:+.1f}%)  [{elapsed:.1f}s]")

    print_results(results)
    save_csv(results, ms_per_img)

    print("\nDone. D1 multi-layer concat result for thesis:")
    if 1000 in results:
        ml_ov = results[1000]["overall"]
        sl_ov = DINOV3_SINGLE_BASELINES[1000]["overall"]
        ml_br = results[1000]["per_class"].get("bedrock", 0)
        sl_br = DINOV3_SINGLE_BASELINES[1000].get("bedrock", 0)
        improved = "✅ IMPROVED" if ml_ov > sl_ov else "❌ NO IMPROVEMENT"
        print(f"  1000-shot Overall: {ml_ov:.2f}% vs {sl_ov:.2f}% single-layer  →  {improved}  "
              f"(Δ{ml_ov - sl_ov:+.2f}%)")
        print(f"  1000-shot Bedrock: {ml_br:.2f}% vs {sl_br:.2f}% single-layer  "
              f"(Δ{ml_br - sl_br:+.2f}%)")
        print(f"  Gap vs supervised (96.67%): {ml_ov - SUPERVISED_BASELINE:+.2f}%")


if __name__ == "__main__":
    main()
