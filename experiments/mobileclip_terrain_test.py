"""
Purpose: Zero-shot terrain classification using MobileCLIP2-S0 on AI4Mars dataset.
         Evaluates on 287 gold-standard test images using v9 hybrid prompts.
         Compares accuracy and latency against CLIP ViT-B/32 baseline (54.4%, 195ms).
Inputs:  AI4Mars dataset (local path defined below)
Outputs: Per-class and overall accuracy, inference time, CSV saved to results/
How to run:
    python3 -u experiments/mobileclip_terrain_test.py          # full 287-image eval
    python3 -u experiments/mobileclip_terrain_test.py --n 20   # quick 20-image smoke test
    python3 -u experiments/mobileclip_terrain_test.py --cpu    # force CPU (RPi simulation)
Project: Onboard Visual Foundation Models for Mars Terrain Perception and Rover Navigation
"""

import argparse
import csv
import os
import time

import numpy as np
import open_clip
import torch
from PIL import Image

# ── Paths ─────────────────────────────────────────────────────────────────────
AI4MARS_BASE   = "/mnt/c/Users/DELL/Desktop/Thesis/github source/ai4mars-dataset-merged-0.1"
AI4MARS_IMAGES = os.path.join(AI4MARS_BASE, "msl/images/edr")
AI4MARS_LABELS = os.path.join(AI4MARS_BASE, "msl/labels/test/masked-gold-min3-100agree")
RESULTS_DIR    = os.path.join(os.path.dirname(__file__), "results")

# ── AI4Mars class definitions ─────────────────────────────────────────────────
# Pixel values: 0=Soil 1=Bedrock 2=Sand 3=BigRock 255=Unlabeled
CLASS_NAMES   = ["soil", "bedrock", "sand", "big_rock"]
IGNORE_PIXEL  = 255
SUPERVISED_BASELINE = 96.67   # DeepLabv3+ ResNet-101

# ── CLIP baselines for comparison ─────────────────────────────────────────────
CLIP_BASELINE = {
    "overall": 54.4, "soil": 95.9, "bedrock": 18.5, "sand": 29.2, "avg_ms": 195.0
}

# ── v9 hybrid ensemble prompts (best CLIP result — 54.4%) ────────────────────
# Reuse same prompts for fair comparison: if MobileCLIP2 does better, the gain
# is from the model, not from prompt tuning.
V9_PROMPTS = [
    [   # soil — NAVCAM-prefix style (gave 95.9% with CLIP)
        "a Mars rover NAVCAM grayscale image of loose soil and fine regolith on the ground",
        "fine-grained loose soil on Mars surface",
        "loose dusty regolith and fine soil covering the Mars ground",
    ],
    [   # bedrock — texture+scale contrast (gave 91.3% in CLIP v4)
        "Mars rocky ground where the floor itself is made of solid flat rock",
        "continuous flat exposed rock layer covering the Mars ground like a floor",
        "solid flat rock pavement covering the entire Mars ground surface",
    ],
    [   # sand — NAVCAM + brightness cues
        "a Mars rover NAVCAM grayscale image of smooth sandy terrain and wind-blown sand",
        "bright smooth sandy terrain with fine sand grains on Mars",
        "bright sandy terrain or sand dunes on Mars",
    ],
    [   # big_rock — isolation emphasis
        "isolated large individual boulders and rocks scattered on Mars soil",
        "Mars terrain with several large prominent rocks and boulders",
        "a Mars rover NAVCAM grayscale image of large isolated boulders and rocks on Mars",
    ],
]


# ── Model ─────────────────────────────────────────────────────────────────────

def load_model(device: str):
    print("Loading MobileCLIP2-S0 (pretrained=dfndr2b)...")
    print("  First run: auto-downloading weights (~80MB) from HuggingFace...")
    t0 = time.perf_counter()
    model, _, preprocess = open_clip.create_model_and_transforms(
        "MobileCLIP2-S0", pretrained="dfndr2b"
    )
    tokenizer = open_clip.get_tokenizer("MobileCLIP2-S0")
    model = model.to(device).eval()
    load_s = time.perf_counter() - t0
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Loaded in {load_s:.1f}s  |  Params: {n_params:.1f}M  |  Device: {device}")
    return model, preprocess, tokenizer


# ── Text feature encoding ─────────────────────────────────────────────────────

def encode_text_ensemble(model, tokenizer, prompts_per_class: list, device: str):
    """Pre-encode text features for all classes. Returns list of tensors [k, D]."""
    class_feats = []
    for cls_prompts in prompts_per_class:
        tokens = tokenizer(cls_prompts).to(device)
        with torch.no_grad():
            feats = model.encode_text(tokens)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        class_feats.append(feats)  # [n_prompts, D]
    return class_feats


# ── Ground truth ──────────────────────────────────────────────────────────────

def dominant_class(label_path: str):
    """Return the most common non-255 pixel class in a label PNG."""
    label = np.array(Image.open(label_path))
    valid = label[label != IGNORE_PIXEL]
    if len(valid) == 0:
        return None
    return int(np.argmax(np.bincount(valid, minlength=4)))


def load_test_pairs(max_n: int = None):
    label_files = sorted(f for f in os.listdir(AI4MARS_LABELS) if f.endswith("_merged.png"))
    pairs = []
    for lf in label_files:
        stem       = lf.replace("_merged.png", "")
        image_path = os.path.join(AI4MARS_IMAGES, stem + ".JPG")
        label_path = os.path.join(AI4MARS_LABELS, lf)
        if not os.path.exists(image_path):
            continue
        gt = dominant_class(label_path)
        if gt is not None:
            pairs.append((image_path, gt, stem))
        if max_n and len(pairs) >= max_n:
            break
    return pairs


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(model, preprocess, class_feats, test_pairs, device: str):
    correct   = {i: 0 for i in range(4)}
    total     = {i: 0 for i in range(4)}
    times     = []
    rows      = []

    print(f"\n{'Image':<45} {'GT':<10} {'Pred':<10} {'Conf':>6}  {'ms':>6}")
    print("=" * 85)

    for image_path, gt, stem in test_pairs:
        image = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)

        t0 = time.perf_counter()
        with torch.no_grad():
            img_feat = model.encode_image(image)
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)   # [1, D]

            # mean similarity across prompts per class, then softmax
            scores = []
            for feats in class_feats:                                    # feats: [k, D]
                sims = (100.0 * img_feat @ feats.T).squeeze(0)          # [k]
                scores.append(sims.mean())

            probs = torch.stack(scores).softmax(dim=0).cpu().numpy()
        elapsed_ms = (time.perf_counter() - t0) * 1000
        times.append(elapsed_ms)

        pred = int(np.argmax(probs))
        conf = probs[pred] * 100
        hit  = (pred == gt)

        correct[gt] += int(hit)
        total[gt]   += 1

        gt_name   = CLASS_NAMES[gt]
        pred_name = CLASS_NAMES[pred]
        marker    = "✓" if hit else "✗"
        print(f"{stem[:44]:<45} {gt_name:<10} {pred_name:<10} {conf:>5.1f}%  {elapsed_ms:>5.1f} {marker}")

        rows.append({
            "image": stem, "gt": gt_name, "pred": pred_name,
            "conf": f"{conf:.1f}", "ms": f"{elapsed_ms:.1f}", "correct": hit,
        })

    return correct, total, times, rows


# ── Results ───────────────────────────────────────────────────────────────────

def print_results(correct, total, times):
    total_correct = sum(correct.values())
    total_images  = sum(total.values())
    overall       = total_correct / total_images * 100 if total_images > 0 else 0
    avg_ms        = np.mean(times)

    print("\n" + "=" * 65)
    print(f"{'Class':<12} {'Correct':>8} {'Total':>7} {'Accuracy':>10}  {'vs CLIP v9':>12}")
    print("-" * 65)

    per_class = {}
    for i, name in enumerate(CLASS_NAMES):
        acc = correct[i] / total[i] * 100 if total[i] > 0 else 0
        per_class[name] = acc
        diff = acc - CLIP_BASELINE.get(name, 0)
        sign = "+" if diff >= 0 else ""
        print(f"{name:<12} {correct[i]:>8} {total[i]:>7} {acc:>9.1f}%  {sign}{diff:>+10.1f}%")

    print("-" * 65)
    overall_diff = overall - CLIP_BASELINE["overall"]
    sign = "+" if overall_diff >= 0 else ""
    print(f"{'OVERALL':<12} {total_correct:>8} {total_images:>7} {overall:>9.1f}%  {sign}{overall_diff:>+10.1f}%")

    print(f"\n{'Metric':<25} {'MobileCLIP2-S0':>16} {'CLIP ViT-B/32':>16}")
    print("-" * 60)
    print(f"{'Overall accuracy':<25} {overall:>15.1f}%  {CLIP_BASELINE['overall']:>15.1f}%")
    print(f"{'avg ms / img':<25} {avg_ms:>15.1f}   {CLIP_BASELINE['avg_ms']:>15.1f}")
    print(f"{'FPS':<25} {1000/avg_ms:>15.2f}   {1000/CLIP_BASELINE['avg_ms']:>15.2f}")
    print(f"{'vs supervised (96.67%)':<25} {overall-SUPERVISED_BASELINE:>+15.2f}%  "
          f"{CLIP_BASELINE['overall']-SUPERVISED_BASELINE:>+15.2f}%")
    print("=" * 65)

    return overall, per_class, avg_ms


def save_csv(rows, overall, per_class, avg_ms):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, "mobileclip2_s0_terrain_test.csv")

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["image", "gt", "pred", "conf", "ms", "correct"]
        )
        writer.writeheader()
        writer.writerows(rows)

        # Summary rows at the bottom
        writer.writerow({})
        writer.writerow({"image": "SUMMARY", "gt": "overall", "pred": f"{overall:.2f}%",
                         "conf": f"avg_ms={avg_ms:.1f}", "ms": "", "correct": ""})
        for name in CLASS_NAMES:
            writer.writerow({"image": "", "gt": name,
                             "pred": f"{per_class.get(name, 0):.2f}%",
                             "conf": "", "ms": "", "correct": ""})

    print(f"\nCSV saved → {path}")
    return path


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="MobileCLIP2-S0 terrain classification")
    parser.add_argument("--n",   type=int, default=None,
                        help="Limit number of test images (default: all ~287)")
    parser.add_argument("--cpu", action="store_true", help="Force CPU (default: auto)")
    args = parser.parse_args()

    # MX330 GPU does not support FastViT conv ops used by MobileCLIP2 — force CPU
    device = "cpu"

    model, preprocess, tokenizer = load_model(device)

    print("\nPre-encoding v9 text prompts...")
    class_feats = encode_text_ensemble(model, tokenizer, V9_PROMPTS, device)

    # Warm-up
    dummy = preprocess(Image.new("RGB", (224, 224))).unsqueeze(0).to(device)
    with torch.no_grad():
        model.encode_image(dummy)
    print("Warm-up done.\n")

    print("Loading AI4Mars test set...")
    test_pairs = load_test_pairs(max_n=args.n)
    print(f"Images to evaluate: {len(test_pairs)}")

    correct, total, times, rows = evaluate(model, preprocess, class_feats, test_pairs, device)
    overall, per_class, avg_ms  = print_results(correct, total, times)
    save_csv(rows, overall, per_class, avg_ms)

    print("\nDone. Key findings to note for thesis:")
    print(f"  MobileCLIP2-S0 overall: {overall:.1f}%  vs  CLIP v9: {CLIP_BASELINE['overall']}%")
    print(f"  Speed: {avg_ms:.0f}ms vs CLIP: {CLIP_BASELINE['avg_ms']:.0f}ms  "
          f"({'faster' if avg_ms < CLIP_BASELINE['avg_ms'] else 'slower'})")


if __name__ == "__main__":
    main()
