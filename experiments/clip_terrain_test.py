"""
Purpose: Zero-shot terrain classification using CLIP on AI4Mars dataset.
         Evaluates on 322 gold-standard test images, logs results to wandb.
Inputs:  AI4Mars dataset (local), or single image/folder for quick demo
Outputs: Accuracy per class, overall accuracy, inference time — printed + logged to wandb
How to run:
    python3 clip_terrain_test.py --ai4mars          # full evaluation on 322 test images
    python3 clip_terrain_test.py --image path/img   # single image demo
    python3 clip_terrain_test.py --folder path/     # folder of images
    python3 clip_terrain_test.py                    # synthetic test (no dataset needed)
Project: Onboard Visual Foundation Models for Mars Terrain Perception and Rover Navigation
"""

import argparse
import os
import time

import clip
import numpy as np
import torch
import wandb
from PIL import Image, ImageDraw

# ── Dataset paths ─────────────────────────────────────────────────────────────
AI4MARS_BASE   = "/mnt/c/Users/DELL/Desktop/Thesis/github source/ai4mars-dataset-merged-0.1"
AI4MARS_IMAGES = os.path.join(AI4MARS_BASE, "msl/images/edr")
AI4MARS_LABELS = os.path.join(AI4MARS_BASE, "msl/labels/test/masked-gold-min3-100agree")

# ── AI4Mars class definitions ─────────────────────────────────────────────────
# Pixel values in label PNGs: 0=Soil, 1=Bedrock, 2=Sand, 3=BigRock, 255=Unlabeled
AI4MARS_CLASSES = {0: "soil", 1: "bedrock", 2: "sand", 3: "big_rock"}
IGNORE_PIXEL    = 255
CLASS_NAMES     = [AI4MARS_CLASSES[i] for i in range(4)]

# ── CLIP text prompts (one per AI4Mars class) ─────────────────────────────────
# Designed for Mars NAVCAM grayscale-converted imagery
CLIP_PROMPTS = [
    "fine-grained loose soil on Mars surface",        # 0 = soil
    "flat smooth bedrock rock surface on Mars",       # 1 = bedrock
    "bright sandy terrain or sand dunes on Mars",     # 2 = sand
    "large boulders or big rocks on Mars terrain",    # 3 = big_rock
]


# ── Model ─────────────────────────────────────────────────────────────────────

def load_model(device):
    print(f"Loading CLIP ViT-B/32 on {device}...")
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()
    return model, preprocess


# ── Inference ─────────────────────────────────────────────────────────────────

def classify_image(model, preprocess, image_path, text_tokens, device):
    image = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
    t0 = time.perf_counter()
    with torch.no_grad():
        img_feat = model.encode_image(image)
        txt_feat = model.encode_text(text_tokens)
        img_feat /= img_feat.norm(dim=-1, keepdim=True)
        txt_feat /= txt_feat.norm(dim=-1, keepdim=True)
        probs = (100.0 * img_feat @ txt_feat.T).softmax(dim=-1)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    probs = probs[0].cpu().numpy()
    return int(np.argmax(probs)), probs, elapsed_ms


# ── Ground truth ──────────────────────────────────────────────────────────────

def dominant_class(label_path):
    """Return the most common non-255 pixel class in a label PNG."""
    label = np.array(Image.open(label_path))
    valid = label[label != IGNORE_PIXEL]
    if len(valid) == 0:
        return None
    counts = np.bincount(valid, minlength=4)
    return int(np.argmax(counts))


# ── AI4Mars evaluation ────────────────────────────────────────────────────────

def run_ai4mars_eval(model, preprocess, text_tokens, device):
    label_files = sorted(f for f in os.listdir(AI4MARS_LABELS) if f.endswith("_merged.png"))
    print(f"\nFound {len(label_files)} label files in test set")

    correct = {c: 0 for c in range(4)}
    total   = {c: 0 for c in range(4)}
    times   = []
    rows    = []  # for wandb Table

    print(f"\n{'Image':<45} {'GT':<10} {'Pred':<10} {'Conf':>6}  {'ms':>6}")
    print("=" * 85)

    for lf in label_files:
        stem       = lf.replace("_merged.png", "")
        image_path = os.path.join(AI4MARS_IMAGES, stem + ".JPG")
        label_path = os.path.join(AI4MARS_LABELS, lf)

        if not os.path.exists(image_path):
            continue

        gt = dominant_class(label_path)
        if gt is None:
            continue

        pred_idx, probs, ms = classify_image(model, preprocess, image_path, text_tokens, device)
        times.append(ms)

        gt_name   = CLASS_NAMES[gt]
        pred_name = CLASS_NAMES[pred_idx]
        conf      = probs[pred_idx] * 100
        hit       = (pred_idx == gt)

        correct[gt] += int(hit)
        total[gt]   += 1
        rows.append([stem[:40], gt_name, pred_name, f"{conf:.1f}", f"{ms:.1f}", hit])

        marker = "✓" if hit else "✗"
        print(f"{stem[:44]:<45} {gt_name:<10} {pred_name:<10} {conf:>5.1f}%  {ms:>5.1f} {marker}")

    # ── Summary ───────────────────────────────────────────────────────────────
    total_correct = sum(correct.values())
    total_images  = sum(total.values())
    overall_acc   = total_correct / total_images * 100 if total_images > 0 else 0

    print("\n" + "=" * 55)
    print(f"{'Class':<12} {'Correct':>8} {'Total':>7} {'Accuracy':>10}")
    print("-" * 55)
    per_class_acc = {}
    for i, name in enumerate(CLASS_NAMES):
        acc = correct[i] / total[i] * 100 if total[i] > 0 else 0
        per_class_acc[name] = acc
        print(f"{name:<12} {correct[i]:>8} {total[i]:>7} {acc:>9.1f}%")
    print("-" * 55)
    print(f"{'OVERALL':<12} {total_correct:>8} {total_images:>7} {overall_acc:>9.1f}%")
    print(f"\nSupervised baseline (DeepLabv3+): 96.67%")
    print(f"CLIP zero-shot gap: {96.67 - overall_acc:+.2f}%")
    print(f"\nAvg inference: {np.mean(times):.1f} ms  |  FPS: {1000/np.mean(times):.1f}")
    print(f"Device: {device}  |  Images evaluated: {total_images}")

    # ── wandb logging ─────────────────────────────────────────────────────────
    wandb.init(
        project="rover-autonomy-thesis",
        name="clip-vit-b32-ai4mars-zeroshot",
        config={
            "model":       "CLIP ViT-B/32",
            "dataset":     "AI4Mars gold-standard (masked-gold-min3-100agree)",
            "n_images":    total_images,
            "device":      str(device),
            "prompts":     CLIP_PROMPTS,
            "eval_method": "dominant class per image",
        },
    )

    wandb.log({
        "accuracy/overall":             overall_acc,
        "accuracy/soil":                per_class_acc.get("soil", 0),
        "accuracy/bedrock":             per_class_acc.get("bedrock", 0),
        "accuracy/sand":                per_class_acc.get("sand", 0),
        "accuracy/big_rock":            per_class_acc.get("big_rock", 0),
        "accuracy/supervised_baseline": 96.67,
        "accuracy/gap_vs_baseline":     overall_acc - 96.67,
        "inference/avg_ms":             np.mean(times),
        "inference/fps":                1000 / np.mean(times),
    })

    columns = ["image", "ground_truth", "prediction", "confidence_%", "time_ms", "correct"]
    wandb.log({"predictions": wandb.Table(columns=columns, data=rows)})
    wandb.finish()

    print("\nResults logged to wandb → project: rover-autonomy-thesis")
    return overall_acc, per_class_acc


# ── Quick demo (no dataset needed) ───────────────────────────────────────────

def make_synthetic_image(path, label="rocky"):
    img = Image.new("RGB", (224, 224), color=(180, 140, 100))
    draw = ImageDraw.Draw(img)
    if label == "rocky":
        for i in range(0, 224, 30):
            draw.ellipse([i, i % 100, i + 25, i % 100 + 20], fill=(120, 100, 80))
    elif label == "sandy":
        for i in range(0, 224, 10):
            draw.line([(0, i), (224, i + 5)], fill=(200, 170, 130), width=2)
    img.save(path)
    return path


def run_demo(image_paths, model, preprocess, text_tokens, device):
    print(f"\n{'='*60}")
    print(f"{'Image':<30} {'Prediction':<12} {'Conf':>6}  {'ms':>6}")
    print(f"{'='*60}")
    times = []
    for path in image_paths:
        pred_idx, probs, ms = classify_image(model, preprocess, path, text_tokens, device)
        conf = probs[pred_idx] * 100
        name = os.path.basename(path)[:29]
        print(f"{name:<30} {CLASS_NAMES[pred_idx]:<12} {conf:>5.1f}%  {ms:>5.1f}")
        times.append(ms)
    print(f"{'='*60}")
    print(f"Avg: {np.mean(times):.1f} ms  |  FPS: {1000/np.mean(times):.1f}  |  Device: {device}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="CLIP zero-shot terrain classifier")
    parser.add_argument("--ai4mars", action="store_true",
                        help="Run full evaluation on AI4Mars 322 test images")
    parser.add_argument("--image",  help="Single image path (demo mode)")
    parser.add_argument("--folder", help="Folder of images (demo mode)")
    parser.add_argument("--cpu",    action="store_true", help="Force CPU (simulate RPi)")
    args = parser.parse_args()

    device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  CUDA: {torch.cuda.is_available()}")

    model, preprocess = load_model(device)
    text_tokens = clip.tokenize(CLIP_PROMPTS).to(device)

    # Warm-up pass
    with torch.no_grad():
        model.encode_image(torch.zeros(1, 3, 224, 224).to(device))

    if args.ai4mars:
        run_ai4mars_eval(model, preprocess, text_tokens, device)

    elif args.folder:
        exts = (".jpg", ".jpeg", ".png", ".bmp")
        paths = [os.path.join(args.folder, f)
                 for f in os.listdir(args.folder) if f.lower().endswith(exts)]
        run_demo(paths, model, preprocess, text_tokens, device)

    elif args.image:
        run_demo([args.image], model, preprocess, text_tokens, device)

    else:
        print("\nNo input — generating synthetic test images...")
        os.makedirs("/tmp/terrain_test", exist_ok=True)
        paths = []
        for lbl in ["rocky", "sandy", "rocky", "sandy"]:
            p = f"/tmp/terrain_test/{lbl}_{len(paths)}.jpg"
            make_synthetic_image(p, lbl)
            paths.append(p)
        run_demo(paths, model, preprocess, text_tokens, device)


if __name__ == "__main__":
    main()
