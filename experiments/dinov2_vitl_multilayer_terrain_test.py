"""
Purpose: E3 — Few-shot linear probe on DINOv2 ViT-L/14 using multi-layer CLS concatenation.
         Concatenates CLS tokens from the last 4 transformer layers (1024×4 = 4096-d).
         Tests whether multi-layer features improve over single-layer DINOv2 ViT-L (93.73%)
         and Ensemble B (94.43%), following the pattern from arXiv:2407.18100.
         Note: DINOv3 ViT-S multi-layer (1536-d) was a negative result (−0.3pp). This tests
         whether the same technique works better on a richer ViT-L backbone.
Inputs:  AI4Mars dataset (287 test / up to 1000/class train)
Outputs: experiments/results/dinov2_vitl_multilayer_terrain_few_shot.csv
         experiments/results/feature_cache/dinov2_vitl_multilayer_test_287_feats.npy
         experiments/results/feature_cache/dinov2_vitl_multilayer_train_1000_feats.npy
How to run:
    python3 -u experiments/dinov2_vitl_multilayer_terrain_test.py | tee /tmp/dinov2_vitl_multi_log.txt
Reference: Burgat et al. (2024) DINOv2 Rocks Geological Image Analysis. arXiv:2407.18100
           Oquab et al. (2024) DINOv2. TMLR. arXiv:2304.07193
Project: Onboard Visual Foundation Models for Mars Terrain Perception and Rover Navigation
"""

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

MODEL_ID            = "facebook/dinov2-large"
N_LAYERS            = 4          # last N layers to concat
FEAT_DIM_SINGLE     = 1024
FEAT_DIM_MULTI      = FEAT_DIM_SINGLE * N_LAYERS   # 4096
CLASS_NAMES         = ["soil", "bedrock", "sand", "big_rock"]
IGNORE_PIXEL        = 255
SUPERVISED_BASELINE = 96.67

# Baselines for comparison printout
SINGLE_VITL_BASELINES = {
    10:   {"overall": 65.2,  "bedrock": 73.9},
    50:   {"overall": 80.5,  "bedrock": 63.0},
    100:  {"overall": 84.0,  "bedrock": 68.5},
    500:  {"overall": 93.7,  "bedrock": 91.3},
    1000: {"overall": 93.73, "bedrock": 92.4},
}
ENSEMBLE_B_BASELINES = {
    1000: {"overall": 94.43, "bedrock": 90.2},
}

DEFAULT_SHOTS = [10, 50, 100, 500, 1000]
RANDOM_SEED   = 42


# ── Model ─────────────────────────────────────────────────────────────────────

def load_model(device: str):
    print(f"Loading DINOv2 ViT-L/14 for multi-layer extraction  [{MODEL_ID}]")
    print(f"  Mode: last {N_LAYERS} layers CLS concat → {FEAT_DIM_MULTI}-d")
    print(f"  Baselines: single-layer ViT-L 93.73% | Ensemble B 94.43%")
    t0 = time.perf_counter()
    processor = AutoImageProcessor.from_pretrained(MODEL_ID)
    model     = AutoModel.from_pretrained(MODEL_ID).to(device).eval()
    load_s    = time.perf_counter() - t0
    n_params  = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Loaded in {load_s:.1f}s  |  Params: {n_params:.1f}M  "
          f"|  Single-layer dim: {FEAT_DIM_SINGLE}  |  Multi-layer dim: {FEAT_DIM_MULTI}  |  Device: {device}")
    return model, processor


# ── Feature extraction ────────────────────────────────────────────────────────

def extract_features_multilayer(model, processor, image_paths, device: str, desc: str = ""):
    features = []
    n  = len(image_paths)
    t0 = time.perf_counter()

    for i, path in enumerate(image_paths):
        if (i + 1) % 50 == 0 or i == n - 1:
            elapsed = time.perf_counter() - t0
            eta     = elapsed / (i + 1) * (n - i - 1)
            print(f"  {desc} {i+1}/{n}  "
                  f"({elapsed/60:.1f}min elapsed, ~{eta/60:.1f}min remaining)", flush=True)

        image  = Image.open(path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            # hidden_states: tuple of (n_layers+1) tensors, each [1, seq_len, 1024]
            hidden  = outputs.hidden_states
            # Take CLS ([:,0,:]) from last N_LAYERS layers
            cls_list = [hidden[i][:, 0, :] for i in range(-N_LAYERS, 0)]
            feat = torch.cat(cls_list, dim=-1).cpu().numpy().squeeze()  # [4096]

        features.append(feat)

    features = np.array(features, dtype=np.float32)
    features = normalize(features)
    elapsed_total = time.perf_counter() - t0
    ms_per = elapsed_total / n * 1000
    print(f"  Done — avg {ms_per:.0f}ms/img  feat shape: {features.shape}")
    return features, ms_per


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


# ── Few-shot evaluation ───────────────────────────────────────────────────────

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


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(CACHE_DIR, exist_ok=True)
    device = "cpu"  # previous experiments all ran CPU; CUDA unavailable on this setup
    model, processor = load_model(device)

    # ── Test features ──────────────────────────────────────────────────────────
    cache_test_f = os.path.join(CACHE_DIR, "dinov2_vitl_multilayer_test_287_feats.npy")
    cache_test_l = os.path.join(CACHE_DIR, "dinov2_vitl_multilayer_test_287_labels.npy")

    if os.path.exists(cache_test_f):
        print("\nLoading test features from cache...")
        test_feats  = np.load(cache_test_f)
        test_labels = np.load(cache_test_l)
        avg_ms = 0.0
        print(f"  test {test_feats.shape}")
    else:
        print("\nLoading test set...")
        test_pairs  = load_split(TEST_LABELS)
        test_paths  = [p for p, _ in test_pairs]
        test_labels = np.array([l for _, l in test_pairs])
        print(f"  Test images: {len(test_paths)}")
        test_feats, avg_ms = extract_features_multilayer(
            model, processor, test_paths, device, "test"
        )
        np.save(cache_test_f, test_feats)
        np.save(cache_test_l, test_labels)
        print(f"  Features cached → {cache_test_f}")

    # ── Train features ─────────────────────────────────────────────────────────
    cache_train_f = os.path.join(CACHE_DIR, "dinov2_vitl_multilayer_train_1000_feats.npy")
    cache_train_l = os.path.join(CACHE_DIR, "dinov2_vitl_multilayer_train_1000_labels.npy")

    if os.path.exists(cache_train_f):
        print("Loading train features from cache...")
        train_feats  = np.load(cache_train_f)
        train_labels = np.load(cache_train_l)
        print(f"  train {train_feats.shape}")
    else:
        print("\nLoading train set (up to 1000/class)...")
        train_pairs = load_split(TRAIN_LABELS)
        by_class = {c: [] for c in range(len(CLASS_NAMES))}
        for p, l in train_pairs:
            by_class[l].append((p, l))
        rng = random.Random(RANDOM_SEED)
        selected = []
        for c in range(len(CLASS_NAMES)):
            pool = by_class[c]
            rng.shuffle(pool)
            selected.extend(pool[:1000])
        train_paths  = [p for p, _ in selected]
        train_labels_raw = [l for _, l in selected]
        counts = {CLASS_NAMES[c]: sum(1 for l in train_labels_raw if l == c) for c in range(len(CLASS_NAMES))}
        print(f"  Train sample: {', '.join(f'{k} {v}' for k, v in counts.items())}")
        print(f"  Estimated time: ~{len(train_paths) * 1.5 / 60:.0f} min at ~1500ms/img")
        train_feats, avg_ms = extract_features_multilayer(
            model, processor, train_paths, device, "train"
        )
        train_labels = np.array(train_labels_raw)
        np.save(cache_train_f, train_feats)
        np.save(cache_train_l, train_labels)
        print(f"  Features cached → {cache_train_f}")

    # ── Few-shot probes ────────────────────────────────────────────────────────
    print("\nRunning few-shot linear probes...")
    results = {}
    for shots in DEFAULT_SHOTS:
        t0 = time.perf_counter()
        print(f"  {shots}-shot...", end=" ", flush=True)
        overall, per_class = run_few_shot(
            train_feats, train_labels, test_feats, test_labels, shots
        )
        elapsed = time.perf_counter() - t0
        single_base = SINGLE_VITL_BASELINES.get(shots, {}).get("overall")
        delta = f"{overall - single_base:+.1f}%" if single_base else "N/A"
        print(f"overall {overall:.1f}%  (single-ViT-L {single_base}% Δ{delta})  [{elapsed:.1f}s]")
        results[shots] = {"overall": overall, "per_class": per_class}

    # ── Print table ────────────────────────────────────────────────────────────
    shots_sorted = sorted(results.keys())
    print("\n" + "=" * 100)
    print(f"DINOv2 ViT-L/14 Multi-Layer ({N_LAYERS}×CLS, {FEAT_DIM_MULTI}-d) — AI4Mars 287 test images")
    print(f"Comparison vs: single-layer ViT-L (1024-d, 93.73%) | Ensemble B (1792-d, 94.43%)")
    print("=" * 100)
    for metric in ["overall"] + CLASS_NAMES:
        row = f"{metric:<10}"
        for s in shots_sorted:
            multi_val  = results[s]["overall"] if metric == "overall" else results[s]["per_class"].get(metric)
            single_val = SINGLE_VITL_BASELINES.get(s, {}).get("overall" if metric == "overall" else metric)
            multi_str  = f"{multi_val:>6.1f}%" if multi_val is not None else "   N/A"
            single_str = f"{single_val:>6.1f}%" if single_val is not None else "   N/A"
            delta      = (f"{multi_val - single_val:>+6.1f}%"
                          if (multi_val is not None and single_val is not None) else "    N/A")
            row += f"  [{s:>4}shot] multi:{multi_str} single:{single_str} Δ:{delta}"
        print(row)
    print("=" * 100)

    if 1000 in results:
        ov1000 = results[1000]["overall"]
        s1000  = SINGLE_VITL_BASELINES[1000]["overall"]
        e1000  = ENSEMBLE_B_BASELINES[1000]["overall"]
        br1000 = results[1000]["per_class"].get("bedrock", 0)
        br_s   = SINGLE_VITL_BASELINES[1000]["bedrock"]
        print(f"\n>>> 1000-shot: single {s1000:.2f}% → multi {ov1000:.2f}%  (Δ {ov1000 - s1000:+.2f}pp)"
              f"  |  Ensemble B {e1000:.2f}%  (Δ vs ensemble {ov1000 - e1000:+.2f}pp)"
              f"  |  Gap vs supervised: {ov1000 - SUPERVISED_BASELINE:+.2f}%")
        print(f">>> Bedrock 1000-shot: single {br_s:.1f}% → multi {br1000:.1f}%  "
              f"(Δ {br1000 - br_s:+.1f}pp)")
        if ov1000 > e1000:
            print(f"\n=== NEW BEST: Multi-layer ViT-L {ov1000:.2f}% > Ensemble B {e1000:.2f}% ===")
        elif ov1000 > s1000:
            print(f"\n✅ Multi-layer improves over single ViT-L (+{ov1000 - s1000:.2f}pp) but below Ensemble B")
        else:
            print(f"\n❌ Multi-layer ({ov1000:.2f}%) does NOT improve over single ViT-L ({s1000:.2f}%)")

    # ── Save CSV ───────────────────────────────────────────────────────────────
    os.makedirs(RESULTS_DIR, exist_ok=True)
    csv_path = os.path.join(RESULTS_DIR, "dinov2_vitl_multilayer_terrain_few_shot.csv")
    fieldnames = ["shots", "overall", "gap_vs_supervised"] + CLASS_NAMES + [
        "single_vitl_overall", "ensemble_b_overall",
        "delta_vs_single", "delta_vs_ensemble", "feat_dim"
    ]
    rows = []
    for s in shots_sorted:
        r = results[s]
        sv = SINGLE_VITL_BASELINES.get(s, {}).get("overall")
        ev = ENSEMBLE_B_BASELINES.get(s, {}).get("overall")
        row = {
            "shots":              s,
            "overall":            f"{r['overall']:.2f}",
            "gap_vs_supervised":  f"{r['overall'] - SUPERVISED_BASELINE:.2f}",
            "single_vitl_overall": f"{sv:.2f}" if sv else "N/A",
            "ensemble_b_overall":  f"{ev:.2f}" if ev else "N/A",
            "delta_vs_single":     f"{r['overall'] - sv:+.2f}" if sv else "N/A",
            "delta_vs_ensemble":   f"{r['overall'] - ev:+.2f}" if ev else "N/A",
            "feat_dim":            FEAT_DIM_MULTI,
        }
        for name in CLASS_NAMES:
            val = r["per_class"].get(name)
            row[name] = f"{val:.2f}" if val is not None else "N/A"
        rows.append(row)

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nCSV saved → {csv_path}")

    # Final summary
    if 1000 in results:
        ov = results[1000]["overall"]
        print(f"\n=== FINAL RESULT — DINOv2 ViT-L/14 Multi-Layer ({FEAT_DIM_MULTI}-d) ===")
        print(f"  1000-shot overall: {ov:.2f}%")
        print(f"  Bedrock: {results[1000]['per_class'].get('bedrock', 0):.1f}%")
        print(f"  Gap vs supervised: {ov - SUPERVISED_BASELINE:.2f}%")
        print(f"  vs single-layer ViT-L (93.73%): {ov - 93.73:+.2f}pp")
        print(f"  vs Ensemble B (94.43%): {ov - 94.43:+.2f}pp")


if __name__ == "__main__":
    main()
