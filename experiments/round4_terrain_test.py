"""
Purpose: Round 4 confirmatory few-shot linear probe on the final untested DINO-family
         encoders for AI4Mars terrain classification. Verifies the model-landscape survey
         conclusion (docs/model_landscape_survey.md) that no frozen encoder clearly beats
         the DINOv2/DINOv3 baseline, and tests two specific hypotheses:
           - DINOv3 ConvNeXt-Tiny: better RPi deployment (CNN quantises better on ARM)
           - DINOv2+registers (S/B): cleaner feature maps may add accuracy
           - DINOv3-SAT ViT-L (satellite-pretrained): domain match may break 94.43% ceiling
         Same frozen-encoder + L2-norm + LogReg(C=0.316) pipeline as all prior experiments.
Inputs:  AI4Mars train labels (few-shot sampling) + gold-standard test set (287 images)
Outputs: Per-model CSV in results/, combined results/round4_terrain_few_shot.csv,
         feature caches in results/feature_cache/<prefix>_*
How to run:
    python3 -u experiments/round4_terrain_test.py | tee /tmp/round4_log.txt
    python3 -u experiments/round4_terrain_test.py --only convnext_tiny
Note: ViT-L (SAT) is the slow one (~1-1.5h CPU); ConvNeXt-Tiny + registers are fast (~10-30min).
      DINOv3 checkpoints are gated (manual) — token access already verified.
Reference: DINOv3 (arXiv:2508.10104), DINOv2 w/ registers (Darcet 2024, arXiv:2309.16588)
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

CLASS_NAMES         = ["soil", "bedrock", "sand", "big_rock"]
IGNORE_PIXEL        = 255
SUPERVISED_BASELINE = 96.67
ENSEMBLE_B_BEST     = 94.43   # current best (frozen)
DEFAULT_SHOTS       = [10, 50, 100, 500, 1000]
RANDOM_SEED         = 42

# ── Model registry ────────────────────────────────────────────────────────────
# arch: "vit" → CLS token (last_hidden_state[:,0]); "convnext" → pooler/mean-pool
MODELS = {
    "convnext_tiny": {
        "hf_id":  "facebook/dinov3-convnext-tiny-pretrain-lvd1689m",
        "arch":   "convnext",
        "prefix": "dinov3_convnext_tiny",
        "label":  "DINOv3 ConvNeXt-Tiny (29M)",
    },
    "reg_small": {
        "hf_id":  "facebook/dinov2-with-registers-small",
        "arch":   "vit",
        "prefix": "dinov2_reg_small",
        "label":  "DINOv2+registers ViT-S (21M)",
    },
    "reg_base": {
        "hf_id":  "facebook/dinov2-with-registers-base",
        "arch":   "vit",
        "prefix": "dinov2_reg_base",
        "label":  "DINOv2+registers ViT-B (86M)",
    },
    "sat_vitl": {
        "hf_id":  "facebook/dinov3-vitl16-pretrain-sat493m",
        "arch":   "vit",
        "prefix": "dinov3_sat_vitl",
        "label":  "DINOv3-SAT ViT-L (300M, satellite)",
    },
}
RUN_ORDER = ["convnext_tiny", "reg_small", "reg_base", "sat_vitl"]


# ── Model + feature extraction ────────────────────────────────────────────────

def load_model(hf_id, device):
    print(f"  Loading {hf_id} (frozen)...", flush=True)
    t0 = time.perf_counter()
    processor = AutoImageProcessor.from_pretrained(hf_id, token=True)
    model     = AutoModel.from_pretrained(hf_id, token=True).to(device).eval()
    n_params  = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"    Loaded in {time.perf_counter()-t0:.1f}s | {n_params:.1f}M params | device {device}",
          flush=True)
    return model, processor


def pool_features(outputs, arch):
    """Return a single feature vector per image for the given architecture."""
    if arch == "vit":
        return outputs.last_hidden_state[:, 0, :]          # CLS token
    # convnext: prefer pooler_output, else global-average-pool the spatial map
    if getattr(outputs, "pooler_output", None) is not None:
        return outputs.pooler_output
    lhs = outputs.last_hidden_state
    if lhs.dim() == 4:        # [B, C, H, W]
        return lhs.mean(dim=(2, 3))
    return lhs.mean(dim=1)    # [B, N, C] fallback


def extract_features(model, processor, image_paths, labels, arch, device, desc=""):
    features = []
    n  = len(image_paths)
    t0 = time.perf_counter()
    for i, path in enumerate(image_paths):
        if (i + 1) % 50 == 0 or i == n - 1:
            elapsed = time.perf_counter() - t0
            eta_h   = elapsed / (i + 1) * (n - i - 1) / 3600
            print(f"    {desc} {i+1}/{n} ({elapsed/60:.1f}min, ~{eta_h:.2f}h left)", flush=True)
        image  = Image.open(path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            feat    = pool_features(outputs, arch).cpu().numpy().squeeze()
        features.append(feat)
    features = normalize(np.array(features, dtype=np.float32))
    return features, np.array(labels)


# ── Ground truth + sampling (shared with prior experiments) ───────────────────

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
        if not os.path.exists(image_path):
            continue
        gt = dominant_class(os.path.join(label_dir, fname))
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
    idx = []
    for c in range(len(CLASS_NAMES)):
        ci     = np.where(train_labels == c)[0]
        chosen = rng.choice(ci, size=min(n_shots, len(ci)), replace=False)
        idx.extend(chosen.tolist())
    clf = LogisticRegression(C=0.316, max_iter=1000, random_state=RANDOM_SEED,
                             multi_class="multinomial", solver="lbfgs")
    clf.fit(train_feats[idx], train_labels[idx])
    preds = clf.predict(test_feats)

    correct = {i: 0 for i in range(len(CLASS_NAMES))}
    total   = {i: 0 for i in range(len(CLASS_NAMES))}
    for pred, gt in zip(preds, test_labels):
        total[gt]   += 1
        correct[gt] += int(pred == gt)
    per_class = {CLASS_NAMES[i]: (correct[i] / total[i] * 100 if total[i] else None)
                 for i in range(len(CLASS_NAMES))}
    n_total = sum(total.values())
    overall = sum(correct.values()) / n_total * 100 if n_total else 0
    return overall, per_class


# ── Per-model driver ──────────────────────────────────────────────────────────

def get_or_extract(cfg, shots, device, no_cache):
    prefix = cfg["prefix"]
    c_tef  = os.path.join(CACHE_DIR, f"{prefix}_test_287_feats.npy")
    c_tel  = os.path.join(CACHE_DIR, f"{prefix}_test_287_labels.npy")
    c_trf  = os.path.join(CACHE_DIR, f"{prefix}_train_1000_feats.npy")
    c_trl  = os.path.join(CACHE_DIR, f"{prefix}_train_1000_labels.npy")
    max_shots = max(shots)

    cache_ok = (not no_cache and all(os.path.exists(p) for p in [c_tef, c_tel, c_trf, c_trl])
                and np.load(c_trf, mmap_mode="r").shape[0] >= max_shots)  # ≥ enough samples
    if cache_ok:
        print("  Using cached features", flush=True)
        return (np.load(c_trf), np.load(c_trl), np.load(c_tef), np.load(c_tel), 0.0)

    model, processor = load_model(cfg["hf_id"], device)

    test_pairs = load_split(TEST_LABELS)
    print(f"  Test images: {len(test_pairs)} — extracting...", flush=True)
    t_test = time.perf_counter()
    te_f, te_l = extract_features(model, processor,
                                  [p for p, _ in test_pairs], [l for _, l in test_pairs],
                                  cfg["arch"], device, desc="test")
    ms_img = (time.perf_counter() - t_test) / len(test_pairs) * 1000
    print(f"  Test done — {ms_img:.0f}ms/img, feat dim {te_f.shape[1]}", flush=True)

    train_all     = load_split(TRAIN_LABELS)
    train_sampled = sample_n_per_class(train_all, max_shots)
    print(f"  Train images: {len(train_sampled)} — ~{len(train_sampled)*ms_img/1000/3600:.2f}h",
          flush=True)
    tr_f, tr_l = extract_features(model, processor,
                                  [p for p, _ in train_sampled], [l for _, l in train_sampled],
                                  cfg["arch"], device, desc="train")
    np.save(c_tef, te_f); np.save(c_tel, te_l); np.save(c_trf, tr_f); np.save(c_trl, tr_l)
    print(f"  Cached → {prefix}_*", flush=True)
    return tr_f, tr_l, te_f, te_l, ms_img


def save_model_csv(cfg, results, ms_img):
    path = os.path.join(RESULTS_DIR, f"{cfg['prefix']}_terrain_few_shot.csv")
    fields = ["shots", "overall", "gap_vs_supervised", "gap_vs_ensembleB"] + CLASS_NAMES + ["ms_per_img"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for s in sorted(results):
            r = results[s]
            row = {"shots": s, "overall": f"{r['overall']:.2f}",
                   "gap_vs_supervised": f"{r['overall']-SUPERVISED_BASELINE:.2f}",
                   "gap_vs_ensembleB":  f"{r['overall']-ENSEMBLE_B_BEST:.2f}",
                   "ms_per_img": f"{ms_img:.1f}"}
            for name in CLASS_NAMES:
                v = r["per_class"].get(name)
                row[name] = f"{v:.2f}" if v is not None else "N/A"
            w.writerow(row)
    print(f"  CSV → {path}", flush=True)
    return path


def main():
    ap = argparse.ArgumentParser(description="Round 4 confirmatory DINO-family probes")
    ap.add_argument("--only", choices=list(MODELS), help="run a single model")
    ap.add_argument("--shots", type=int, nargs="+", default=DEFAULT_SHOTS)
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()

    os.makedirs(CACHE_DIR, exist_ok=True)
    device = "cpu"
    keys   = [args.only] if args.only else RUN_ORDER

    summary = []
    for key in keys:
        cfg = MODELS[key]
        print("\n" + "=" * 90)
        print(f"### {cfg['label']}  [{cfg['hf_id']}]")
        print("=" * 90, flush=True)
        tr_f, tr_l, te_f, te_l, ms_img = get_or_extract(cfg, args.shots, device, args.no_cache)

        results = {}
        for s in sorted(args.shots):
            ov, pc = run_few_shot(tr_f, tr_l, te_f, te_l, s)
            results[s] = {"overall": ov, "per_class": pc}
            print(f"  {s:>4}-shot: overall {ov:5.2f}%  "
                  f"bedrock {pc.get('bedrock') or 0:5.2f}%  sand {pc.get('sand') or 0:5.2f}%",
                  flush=True)
        save_model_csv(cfg, results, ms_img)

        best = results[max(args.shots)]
        ov   = best["overall"]
        verdict = ("✅ BEATS Ensemble B" if ov > ENSEMBLE_B_BEST
                   else "≈ parity" if ov >= 89.0 else "❌ below baseline")
        print(f"  → 1000-shot {ov:.2f}% | vs Ensemble B {ov-ENSEMBLE_B_BEST:+.2f}% "
              f"| vs supervised {ov-SUPERVISED_BASELINE:+.2f}% → {verdict}", flush=True)
        summary.append((cfg["label"], ov, best["per_class"], ms_img, verdict))

    # combined summary CSV
    comb = os.path.join(RESULTS_DIR, "round4_terrain_few_shot.csv")
    with open(comb, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "overall_1000shot", "soil", "bedrock", "sand",
                    "ms_per_img", "vs_ensembleB", "verdict"])
        for label, ov, pc, ms, verdict in summary:
            w.writerow([label, f"{ov:.2f}",
                        *(f"{pc.get(n):.2f}" if pc.get(n) is not None else "N/A"
                          for n in ["soil", "bedrock", "sand"]),
                        f"{ms:.1f}", f"{ov-ENSEMBLE_B_BEST:+.2f}", verdict])

    print("\n" + "=" * 90)
    print("ROUND 4 SUMMARY (1000-shot)")
    print("=" * 90)
    for label, ov, pc, ms, verdict in summary:
        print(f"  {label:<42} {ov:6.2f}%  ({ov-ENSEMBLE_B_BEST:+.2f} vs best)  {verdict}")
    print(f"\nCombined CSV → {comb}")


if __name__ == "__main__":
    main()
