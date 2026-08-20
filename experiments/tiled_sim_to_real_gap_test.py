"""
Purpose: Diagnostic test -- does splitting each real ExoMy test photo into a
         grid of tiles (instead of classifying the whole frame at once)
         improve big_rock detection? sim_to_real_gap_test.py showed 0/10
         big_rock accuracy; visual inspection of the source photos found the
         rock(s) typically occupy only part of the frame, with the majority
         of pixels being plain sand background -- a whole-frame classifier
         has no way to "see" a hazard that isn't the dominant texture.
         This script reuses the same ONNX encoder + LogReg probe (no
         retraining) and only changes how the image is fed to it.
Inputs:  --images-dir: same class-subfolder structure as sim_to_real_gap_test.py
         --grid: tiles per side (default 3, i.e. 3x3 = 9 tiles per image)
         --grayscale: apply the same grayscale ablation before tiling
Outputs: Console per-image tile breakdown + two aggregation strategies:
           - majority vote across tiles (fair comparison to whole-frame accuracy)
           - "any tile flagged big_rock" (safety-oriented: a deployed system
             should treat a hazard tile anywhere in frame as reason to stop)
         CSV: experiments/results/tiled_sim_to_real_gap_results.csv
How to run:
    python3 experiments/tiled_sim_to_real_gap_test.py \
        --images-dir "/mnt/c/Users/DELL/Desktop/Thesis/Picture" --grayscale --verbose
Project: Onboard Visual Foundation Models for Mars Terrain Perception and Rover Navigation
"""

import argparse
import csv
import os
import sys
from collections import Counter
from datetime import datetime

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sim_to_real_gap_test import (
    CLASS_NAMES, ONNX_PATH, PROBE_PATH, RESULTS_DIR,
    _MEAN, _STD, to_grayscale_3ch, discover_images, load_model,
)


def tile_image(img: Image.Image, grid: int) -> list[Image.Image]:
    """Split a PIL image into grid x grid non-overlapping tiles, row-major order."""
    w, h = img.size
    tile_w, tile_h = w // grid, h // grid
    tiles = []
    for row in range(grid):
        for col in range(grid):
            box = (col * tile_w, row * tile_h,
                   (col + 1) * tile_w if col < grid - 1 else w,
                   (row + 1) * tile_h if row < grid - 1 else h)
            tiles.append(img.crop(box))
    return tiles


def preprocess_tile(tile: Image.Image, grayscale: bool) -> np.ndarray:
    """Same DINOv2 ImageNet preprocessing as sim_to_real_gap_test.preprocess(),
    but starting from an in-memory tile crop instead of a file path."""
    tile = tile.convert("RGB").resize((224, 224), Image.BILINEAR)
    arr = np.array(tile, dtype=np.uint8)
    if grayscale:
        arr = to_grayscale_3ch(arr)
    x = arr.astype(np.float32) / 255.0
    x = (x - _MEAN) / _STD
    x = x.transpose(2, 0, 1)[np.newaxis, :]
    return x


def predict_array(session, probe, x: np.ndarray) -> tuple[str, float]:
    feats = session.run(None, {"pixel_values": x})[0]
    logits = feats @ probe["coef"].T + probe["intercept"]
    e = np.exp(logits[0] - np.max(logits[0]))
    proba = e / e.sum()
    idx = int(np.argmax(proba))
    cls = probe["classes"][idx]
    pred = CLASS_NAMES[int(cls)] if isinstance(cls, (int, np.integer)) else str(cls)
    return pred, float(proba[idx])


def evaluate_tiled(session, probe, dataset: dict, grid: int, grayscale: bool,
                    verbose: bool) -> dict:
    per_class = {}
    all_correct_majority = 0
    all_correct_anyhazard = 0
    all_total = 0
    rows = []

    for true_label, paths in sorted(dataset.items()):
        correct_majority = 0
        correct_anyhazard = 0
        total = len(paths)
        print(f"\n  [{true_label.upper()}]  {total} images, {grid}x{grid} tiles each")

        for img_path in paths:
            img = Image.open(img_path)
            tiles = tile_image(img, grid)
            tile_preds = []
            for t in tiles:
                x = preprocess_tile(t, grayscale)
                pred, conf = predict_array(session, probe, x)
                tile_preds.append((pred, conf))

            labels = [p for p, _ in tile_preds]
            majority_label, majority_count = Counter(labels).most_common(1)[0]
            any_big_rock = any(p == "big_rock" for p, _ in tile_preds)

            hit_majority = (majority_label == true_label)
            # For the safety-oriented metric, only big_rock ground truth cares
            # whether a hazard tile was ever flagged; for non-hazard classes,
            # "correct" just falls back to the majority vote.
            hit_anyhazard = any_big_rock if true_label == "big_rock" else hit_majority

            if hit_majority:
                correct_majority += 1
            if hit_anyhazard:
                correct_anyhazard += 1

            if verbose:
                tile_str = " ".join(f"{p}:{c:.2f}" for p, c in tile_preds)
                print(f"    {os.path.basename(img_path):20s}  "
                      f"majority={majority_label} ({majority_count}/{len(tiles)})  "
                      f"any_big_rock={any_big_rock}  "
                      f"[{tile_str}]")

            rows.append({
                "image": os.path.basename(img_path), "true_label": true_label,
                "majority_pred": majority_label, "any_big_rock_tile": any_big_rock,
                "hit_majority": hit_majority, "hit_anyhazard": hit_anyhazard,
            })

        acc_majority = 100.0 * correct_majority / total if total else 0.0
        acc_anyhazard = 100.0 * correct_anyhazard / total if total else 0.0
        per_class[true_label] = {
            "total": total, "correct_majority": correct_majority,
            "acc_majority": acc_majority, "correct_anyhazard": correct_anyhazard,
            "acc_anyhazard": acc_anyhazard,
        }
        all_correct_majority += correct_majority
        all_correct_anyhazard += correct_anyhazard
        all_total += total
        print(f"    Majority-vote accuracy:   {correct_majority}/{total} = {acc_majority:.1f}%")
        print(f"    Any-hazard-tile accuracy: {correct_anyhazard}/{total} = {acc_anyhazard:.1f}%")

    overall_majority = 100.0 * all_correct_majority / all_total if all_total else 0.0
    overall_anyhazard = 100.0 * all_correct_anyhazard / all_total if all_total else 0.0
    return {
        "per_class": per_class, "overall_majority": overall_majority,
        "overall_anyhazard": overall_anyhazard, "n_total": all_total, "rows": rows,
    }


def save_csv(results: dict, images_dir: str, grid: int) -> str:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, "tiled_sim_to_real_gap_results.csv")
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    fieldnames = ["timestamp", "images_dir", "grid", "image", "true_label",
                  "majority_pred", "any_big_rock_tile", "hit_majority", "hit_anyhazard"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results["rows"]:
            writer.writerow({"timestamp": ts, "images_dir": images_dir, "grid": grid, **row})
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--onnx-path", default=ONNX_PATH)
    parser.add_argument("--probe-path", default=PROBE_PATH)
    parser.add_argument("--grid", type=int, default=3, help="tiles per side (default 3 = 3x3)")
    parser.add_argument("--grayscale", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    images_dir = os.path.abspath(args.images_dir)
    dataset = discover_images(images_dir)
    if not dataset:
        sys.exit(f"[ERROR] No images found under {images_dir}")

    print(f"\n=== Tiled Sim-to-Real Test ({args.grid}x{args.grid} grid, "
          f"grayscale={args.grayscale}) ===")
    session, probe = load_model(args.onnx_path, args.probe_path)

    results = evaluate_tiled(session, probe, dataset, args.grid, args.grayscale, args.verbose)

    print("\n" + "=" * 65)
    print("  TILED AGGREGATION -- SUMMARY")
    print("=" * 65)
    for cls, stats in results["per_class"].items():
        print(f"  {cls:<10}  majority={stats['acc_majority']:.1f}%  "
              f"any-hazard={stats['acc_anyhazard']:.1f}%")
    print(f"  {'OVERALL':<10}  majority={results['overall_majority']:.1f}%  "
          f"any-hazard={results['overall_anyhazard']:.1f}%")
    print("=" * 65)

    csv_path = save_csv(results, images_dir, args.grid)
    print(f"\n  Results saved -> {csv_path}")


if __name__ == "__main__":
    main()
