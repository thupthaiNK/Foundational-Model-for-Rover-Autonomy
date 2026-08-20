"""
Purpose: Test SAM2 (facebook/sam2-hiera-tiny) terrain segmentation on AI4Mars images.
         Segments terrain regions, then classifies each region with CLIP.
         This is the segmentation stage of the cascade pipeline: CLIP + SAM2 + BLIP-2.
Inputs:  AI4Mars gold-standard test set (322 images, local)
Outputs: Segmentation masks, per-region CLIP classification, inference time — saved to CSV
How to run:
    pip install sam2                              # install SAM2 first
    python3 sam2_terrain_test.py --cpu --n 10    # quick test: 10 images
    python3 sam2_terrain_test.py --cpu --n 50    # medium test
    python3 sam2_terrain_test.py --cpu           # full test (slow on CPU)
Project: Onboard Visual Foundation Models for Mars Terrain Perception and Rover Navigation
"""

import argparse
import csv
import os
import time

import clip
import numpy as np
import torch
from PIL import Image

# ── Dataset ───────────────────────────────────────────────────────────────────
AI4MARS_BASE = "/mnt/c/Users/DELL/Desktop/Thesis/github source/ai4mars-dataset-merged-0.1"
IMAGES_DIR   = os.path.join(AI4MARS_BASE, "msl/images/edr")
TEST_LABELS  = os.path.join(AI4MARS_BASE, "msl/labels/test/masked-gold-min3-100agree")

CLASS_NAMES         = ["soil", "bedrock", "sand", "big_rock"]
IGNORE_PIXEL        = 255
SUPERVISED_BASELINE = 96.67

# Best prompts from Experiment 2 (v9 hybrid ensemble)
CLIP_PROMPTS = [
    "a Mars rover NAVCAM grayscale image of loose soil and fine regolith on the ground",
    "Mars rocky ground where the floor itself is made of solid flat rock",
    "a Mars rover NAVCAM grayscale image of smooth sandy terrain and wind-blown sand",
    "isolated large individual boulders and rocks scattered on Mars soil",
]

# SAM2 sampling points: centre + 4 corners of image
def get_sample_points(w: int, h: int):
    return np.array([
        [w // 2, h // 2],          # centre
        [w // 4, h // 4],          # top-left quad
        [3 * w // 4, h // 4],      # top-right quad
        [w // 4, 3 * h // 4],      # bottom-left quad
        [3 * w // 4, 3 * h // 4],  # bottom-right quad
    ], dtype=np.float32)


# ── Ground truth ──────────────────────────────────────────────────────────────

def dominant_class(label_path: str):
    label = np.array(Image.open(label_path))
    valid = label[label != IGNORE_PIXEL]
    if len(valid) == 0:
        return None
    return int(np.argmax(np.bincount(valid, minlength=4)))


def load_test_set(max_n=None):
    pairs = []
    for fname in sorted(os.listdir(TEST_LABELS)):
        if not fname.endswith("_merged.png"):
            continue
        stem       = fname.replace("_merged.png", "")
        image_path = os.path.join(IMAGES_DIR, stem + ".JPG")
        label_path = os.path.join(TEST_LABELS, fname)
        if not os.path.exists(image_path):
            continue
        gt = dominant_class(label_path)
        if gt is not None:
            pairs.append((image_path, gt))
        if max_n and len(pairs) >= max_n:
            break
    return pairs


# ── CLIP classification ───────────────────────────────────────────────────────

def load_clip(device: str):
    print(f"Loading CLIP ViT-B/32 on {device}...")
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()
    text_tokens = clip.tokenize(CLIP_PROMPTS).to(device)
    with torch.no_grad():
        txt_feat = model.encode_text(text_tokens)
        txt_feat /= txt_feat.norm(dim=-1, keepdim=True)
    return model, preprocess, txt_feat


def classify_region(model, preprocess, txt_feat, region_image: Image.Image, device: str):
    """Classify a single image region using CLIP."""
    img_tensor = preprocess(region_image.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        img_feat = model.encode_image(img_tensor)
        img_feat /= img_feat.norm(dim=-1, keepdim=True)
        probs = (100.0 * img_feat @ txt_feat.T).softmax(dim=-1)
    pred  = int(probs[0].argmax())
    conf  = float(probs[0][pred])
    return pred, conf


# ── SAM2 segmentation ─────────────────────────────────────────────────────────

def try_import_sam2():
    """Try to import SAM2 — returns (predictor_class, build_fn) or None if not installed."""
    try:
        from sam2.build_sam import build_sam2_hf
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        return build_sam2_hf, SAM2ImagePredictor
    except ImportError:
        return None, None


def segment_image(predictor, image_np: np.ndarray, n_points: int = 5):
    """
    Run SAM2 with automatic grid points across the image.
    Returns list of (mask, score) tuples.
    """
    h, w = image_np.shape[:2]
    points = get_sample_points(w, h)[:n_points]
    labels = np.ones(len(points), dtype=np.int32)

    predictor.set_image(image_np)
    masks, scores, _ = predictor.predict(
        point_coords=points,
        point_labels=labels,
        multimask_output=True,
    )
    return list(zip(masks, scores))


# ── Pipeline: SAM2 → CLIP per region ─────────────────────────────────────────

def pipeline_classify(clip_model, preprocess, txt_feat, sam2_predictor,
                      image_path: str, device: str, use_sam2: bool):
    """
    Full pipeline: segment image → classify each region → majority vote.
    Falls back to whole-image CLIP if SAM2 not available.
    """
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)
    t0 = time.perf_counter()

    if use_sam2 and sam2_predictor is not None:
        segments = segment_image(sam2_predictor, image_np)
        region_preds = []
        for mask, score in segments:
            # Crop to bounding box of each mask
            ys, xs = np.where(mask)
            if len(ys) == 0:
                continue
            y1, y2, x1, x2 = ys.min(), ys.max(), xs.min(), xs.max()
            region_img = image.crop((x1, y1, x2, y2))
            pred, conf = classify_region(clip_model, preprocess, txt_feat, region_img, device)
            region_preds.append((pred, conf * float(score)))

        if region_preds:
            # Weighted majority vote
            votes = np.zeros(len(CLASS_NAMES))
            for pred, weight in region_preds:
                votes[pred] += weight
            final_pred = int(np.argmax(votes))
        else:
            # Fallback to whole image
            final_pred, _ = classify_region(
                clip_model, preprocess, txt_feat, image, device)
    else:
        # SAM2 not available — whole-image CLIP only (for comparison)
        final_pred, _ = classify_region(
            clip_model, preprocess, txt_feat, image, device)

    elapsed_ms = (time.perf_counter() - t0) * 1000
    return final_pred, elapsed_ms


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(clip_model, preprocess, txt_feat, sam2_predictor,
             test_pairs: list, device: str, use_sam2: bool):
    n_classes = len(CLASS_NAMES)
    correct   = {c: 0 for c in range(n_classes)}
    total     = {c: 0 for c in range(n_classes)}
    times_ms  = []

    mode = "SAM2+CLIP" if use_sam2 else "CLIP-only (no SAM2)"
    print(f"  Mode: {mode} | {len(test_pairs)} images")

    for i, (image_path, gt) in enumerate(test_pairs):
        pred, ms = pipeline_classify(
            clip_model, preprocess, txt_feat, sam2_predictor,
            image_path, device, use_sam2)
        correct[gt] += int(pred == gt)
        total[gt]   += 1
        times_ms.append(ms)
        if (i + 1) % 20 == 0:
            print(f"    {i+1}/{len(test_pairs)}  avg {np.mean(times_ms):.0f}ms/img")

    per_class = {
        name: (correct[i] / total[i] * 100 if total[i] > 0 else None)
        for i, name in enumerate(CLASS_NAMES)
    }
    n = sum(total.values())
    return {
        "overall":   sum(correct.values()) / n * 100 if n > 0 else 0,
        "per_class": per_class,
        "avg_ms":    float(np.mean(times_ms)),
        "fps":       float(1000 / np.mean(times_ms)),
        "mode":      mode,
    }


# ── Reporting ─────────────────────────────────────────────────────────────────

def print_results(results: dict):
    print("\n" + "=" * 60)
    print("SAM2 + CLIP CASCADE — AI4Mars terrain classification")
    print("=" * 60)
    for mode, res in results.items():
        pc = res["per_class"]
        print(f"\n{mode}: {res['mode']}")
        print(f"  Overall:  {res['overall']:.1f}%")
        for name in CLASS_NAMES:
            val = pc.get(name)
            print(f"  {name:<10}: {val:.1f}%" if val is not None else f"  {name:<10}: N/A")
        print(f"  avg ms/img: {res['avg_ms']:.0f}   FPS: {res['fps']:.3f}")
    print(f"\nSupervisd baseline: {SUPERVISED_BASELINE:.1f}%")


def save_csv(results: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = ["mode", "overall", "gap"] + CLASS_NAMES + ["avg_ms", "fps"]
    rows = []
    for key, res in results.items():
        row = {
            "mode":    key,
            "overall": f"{res['overall']:.2f}",
            "gap":     f"{res['overall'] - SUPERVISED_BASELINE:.2f}",
            "avg_ms":  f"{res['avg_ms']:.0f}",
            "fps":     f"{res['fps']:.4f}",
        }
        for name in CLASS_NAMES:
            val = res["per_class"].get(name)
            row[name] = f"{val:.2f}" if val is not None else "N/A"
        rows.append(row)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"CSV saved → {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SAM2+CLIP cascade on AI4Mars")
    parser.add_argument("--n",   type=int, default=None,
                        help="Max images to evaluate (default: all ~287)")
    parser.add_argument("--cpu", action="store_true", help="Force CPU")
    args = parser.parse_args()

    device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    clip_model, preprocess, txt_feat = load_clip(device)

    # Try to load SAM2
    build_sam2, SAM2ImagePredictor = try_import_sam2()
    sam2_predictor = None

    if build_sam2 is not None:
        print("SAM2 found — loading facebook/sam2-hiera-tiny from HuggingFace...")
        try:
            sam2_model = build_sam2("facebook/sam2-hiera-tiny", device=device)
            sam2_predictor = SAM2ImagePredictor(sam2_model)
            print("SAM2 loaded successfully")
        except Exception as e:
            print(f"SAM2 load failed: {e}")
            print("Proceeding with CLIP-only mode")
    else:
        print("SAM2 not installed — install with: pip install sam2")
        print("Proceeding with CLIP-only mode for comparison baseline")

    print("\nLoading test set...")
    test_pairs = load_test_set(args.n)
    print(f"Test set: {len(test_pairs)} images\n")

    results = {}

    # Always run CLIP-only baseline
    print("── CLIP-only (no SAM2) ──")
    results["clip_only"] = evaluate(
        clip_model, preprocess, txt_feat, None,
        test_pairs, device, use_sam2=False)

    # Run SAM2+CLIP if available
    if sam2_predictor is not None:
        print("\n── SAM2 + CLIP cascade ──")
        results["sam2_clip"] = evaluate(
            clip_model, preprocess, txt_feat, sam2_predictor,
            test_pairs, device, use_sam2=True)

    print_results(results)

    csv_path = os.path.join(os.path.dirname(__file__), "results", "sam2_terrain_test.csv")
    save_csv(results, csv_path)


if __name__ == "__main__":
    main()
