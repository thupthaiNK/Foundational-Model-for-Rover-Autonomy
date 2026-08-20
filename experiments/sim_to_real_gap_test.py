"""
Purpose: Sim-to-real domain gap evaluation for DINOv2+reg ViT-S/14 ONNX model.
         Runs inference on real ExoMy camera images and compares accuracy to the
         AI4Mars gold-standard test set result (90.24% overall, 84.78% Bedrock).
         Quantifies the domain gap between NASA NAVCAM imagery and ExoMy RPi camera.
Inputs:  --images-dir: directory of manually captured images, one subfolder per class:
           <images-dir>/
             soil/        ← real photos of soil / regolith-like terrain
             bedrock/     ← real photos of rocky / paved surfaces
             sand/        ← real photos of sandy surfaces (optional)
             big_rock/    ← real photos of large rocks / obstacles (optional)
         ONNX model:  experiments/results/dinov2_small_encoder.onnx  (exported via export_dinov2_onnx.py)
         Probe:       experiments/results/dinov2_logreg_probe.pkl
Outputs: Console comparison table (vs AI4Mars baseline) + CSV:
           experiments/results/sim_to_real_gap_results.csv
How to run:
    # Step 1 — capture images with ExoMy camera (run on RPi or PC with USB cam):
    #   mkdir -p real_images/{soil,bedrock,sand,big_rock}
    #   python3 experiments/save_camera_frame.py --output real_images/soil/
    #   (repeat for each class)
    #
    # Step 2 — run evaluation (on PC, pointing at the captured image folder):
    #   python3 experiments/sim_to_real_gap_test.py --images-dir /path/to/real_images/
    #
    # Step 3 — verbose mode shows per-image predictions:
    #   python3 experiments/sim_to_real_gap_test.py --images-dir /path/to/real_images/ --verbose
Project: Onboard Visual Foundation Models for Mars Terrain Perception and Rover Navigation
"""

import argparse
import csv
import os
import pickle
import sys
from collections import defaultdict
from datetime import datetime

import numpy as np
from PIL import Image

# ── Paths ──────────────────────────────────────────────────────────────────────
_HERE       = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(_HERE, "results")
ONNX_PATH   = os.path.join(RESULTS_DIR, "dinov2_reg_small_encoder.onnx")
PROBE_PATH  = os.path.join(RESULTS_DIR, "dinov2_reg_small_probe.npz")

CLASS_NAMES = ["soil", "bedrock", "sand", "big_rock"]

# AI4Mars gold-standard baseline (DINOv2+reg ViT-S/14, 1000-shot, n=287)
AI4MARS_BASELINE = {
    "overall":  90.24,
    "soil":     None,   # per-class not stored in memory
    "bedrock":  84.78,
    "sand":     None,
    "big_rock": None,
}

# DINOv2 ImageNet normalisation constants
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


# ── Image loading ──────────────────────────────────────────────────────────────

def discover_images(images_dir: str) -> dict[str, list[str]]:
    """Return {class_name: [image_path, ...]} from subdirectory structure."""
    dataset: dict[str, list[str]] = {}
    for cls in CLASS_NAMES:
        cls_dir = os.path.join(images_dir, cls)
        if not os.path.isdir(cls_dir):
            continue
        paths = [
            os.path.join(cls_dir, f)
            for f in sorted(os.listdir(cls_dir))
            if os.path.splitext(f.lower())[1] in IMG_EXTENSIONS
        ]
        if paths:
            dataset[cls] = paths
    return dataset


def to_grayscale_3ch(rgb_arr: np.ndarray) -> np.ndarray:
    """Convert an HxWx3 RGB array to grayscale, replicated across 3 channels.

    Same ITU-R BT.601 luma conversion as dinov2_terrain_node.py's
    to_grayscale_3ch() (kept standalone here rather than imported, since this
    script runs offline against an ONNX export, not the live ROS2 node).
    """
    gray = (
        rgb_arr[..., 0].astype(np.float32) * 0.299
        + rgb_arr[..., 1].astype(np.float32) * 0.587
        + rgb_arr[..., 2].astype(np.float32) * 0.114
    ).astype(rgb_arr.dtype)
    return np.stack([gray, gray, gray], axis=-1)


def preprocess(img_path: str, grayscale: bool = False) -> np.ndarray:
    """Load image and apply DINOv2 ImageNet preprocessing. Returns (1, 3, 224, 224) float32."""
    img = Image.open(img_path).convert("RGB").resize((224, 224), Image.BILINEAR)
    arr = np.array(img, dtype=np.uint8)
    if grayscale:
        arr = to_grayscale_3ch(arr)
    x   = arr.astype(np.float32) / 255.0               # (224, 224, 3)
    x   = (x - _MEAN) / _STD                           # normalise
    x   = x.transpose(2, 0, 1)[np.newaxis, :]          # (1, 3, 224, 224)
    return x


# ── Inference ──────────────────────────────────────────────────────────────────

def load_model(onnx_path: str, probe_path: str):
    """Load ONNX session and LogReg probe (.pkl or .npz). Returns (session, probe_dict)."""
    try:
        import onnxruntime as ort
    except ImportError:
        sys.exit("[ERROR] onnxruntime not installed. Run: pip install onnxruntime")

    if not os.path.exists(onnx_path):
        sys.exit(f"[ERROR] ONNX model not found: {onnx_path}")
    if not os.path.exists(probe_path):
        sys.exit(f"[ERROR] LogReg probe not found: {probe_path}")

    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

    if probe_path.endswith(".npz"):
        data  = np.load(probe_path)
        probe = {"coef": data["coef"], "intercept": data["intercept"],
                 "classes": data["classes"]}
    else:
        with open(probe_path, "rb") as f:
            probe = pickle.load(f)
    return session, probe


def predict(session, probe, img_path: str, grayscale: bool = False) -> tuple[str, float]:
    """Run inference on one image. Returns (predicted_label, confidence)."""
    x     = preprocess(img_path, grayscale=grayscale)
    feats = session.run(None, {"pixel_values": x})[0]          # (1, 384)

    if isinstance(probe, dict):
        logits = feats @ probe["coef"].T + probe["intercept"]  # (1, n_classes)
        e      = np.exp(logits[0] - np.max(logits[0]))
        proba  = e / e.sum()
        idx    = int(np.argmax(proba))
        cls    = probe["classes"][idx]
        # classes stored as numeric indices → map to name
        pred   = CLASS_NAMES[int(cls)] if isinstance(cls, (int, np.integer)) else str(cls)
        conf   = float(proba[idx])
    else:
        proba = probe.predict_proba(feats)[0]
        pred  = str(probe.classes_[int(np.argmax(proba))])
        conf  = float(np.max(proba))
    return pred, conf


# ── Evaluation ─────────────────────────────────────────────────────────────────

def evaluate(session, probe, dataset: dict[str, list[str]], verbose: bool,
             grayscale: bool = False) -> dict:
    """Run inference on all images. Returns per-class and overall accuracy stats."""
    per_class: dict[str, dict] = {}
    all_correct = 0
    all_total   = 0

    for true_label, paths in sorted(dataset.items()):
        correct     = 0
        total       = len(paths)
        conf_sum    = 0.0
        errors      = []

        print(f"\n  [{true_label.upper()}]  {total} images")
        for img_path in paths:
            pred, conf = predict(session, probe, img_path, grayscale=grayscale)
            hit = (pred == true_label)
            if hit:
                correct += 1
            else:
                errors.append((os.path.basename(img_path), pred, conf))
            conf_sum += conf
            if verbose:
                status = "OK" if hit else f"WRONG→{pred}"
                print(f"    {os.path.basename(img_path):30s}  {status}  conf={conf:.2f}")

        acc = 100.0 * correct / total if total else 0.0
        per_class[true_label] = {
            "correct":   correct,
            "total":     total,
            "accuracy":  acc,
            "avg_conf":  conf_sum / total if total else 0.0,
            "errors":    errors,
        }
        all_correct += correct
        all_total   += total
        print(f"    Accuracy: {correct}/{total} = {acc:.1f}%  avg_conf={per_class[true_label]['avg_conf']:.2f}")
        if errors and not verbose:
            for fname, pred, conf in errors:
                print(f"    WRONG: {fname} → {pred} (conf={conf:.2f})")

    overall = 100.0 * all_correct / all_total if all_total else 0.0
    return {"per_class": per_class, "overall": overall,
            "n_correct": all_correct, "n_total": all_total}


# ── Reporting ──────────────────────────────────────────────────────────────────

def print_comparison(results: dict) -> None:
    """Print side-by-side comparison with AI4Mars baseline."""
    print("\n" + "=" * 65)
    print("  SIM-TO-REAL GAP — DINOv2+reg ViT-S/14 ONNX")
    print("=" * 65)
    print(f"  {'Class':<12}  {'Real ExoMy':>12}  {'AI4Mars':>12}  {'Gap':>10}")
    print(f"  {'-'*12}  {'-'*12}  {'-'*12}  {'-'*10}")

    for cls in CLASS_NAMES:
        stats = results["per_class"].get(cls)
        if stats is None:
            continue
        real_acc = stats["accuracy"]
        baseline = AI4MARS_BASELINE.get(cls)
        if baseline is not None:
            gap = real_acc - baseline
            gap_str = f"{gap:+.1f}pp"
        else:
            gap_str = "N/A"
            baseline = float("nan")
        print(f"  {cls:<12}  {real_acc:>11.1f}%  "
              f"{'N/A' if np.isnan(baseline) else f'{baseline:.2f}%':>12}  "
              f"{gap_str:>10}")

    overall_gap = results["overall"] - AI4MARS_BASELINE["overall"]
    print(f"  {'-'*12}  {'-'*12}  {'-'*12}  {'-'*10}")
    print(f"  {'OVERALL':<12}  {results['overall']:>11.1f}%  "
          f"{AI4MARS_BASELINE['overall']:>11.2f}%  "
          f"{overall_gap:>+9.1f}pp")
    print("=" * 65)
    print(f"\n  Total images evaluated: {results['n_total']}")
    print(f"  Correct: {results['n_correct']} / {results['n_total']}")

    domain_gap = abs(overall_gap)
    if domain_gap > 20:
        verdict = f"LARGE domain gap ({domain_gap:.1f}pp) — expected from NAVCAM→RPi camera shift"
    elif domain_gap > 10:
        verdict = f"MODERATE domain gap ({domain_gap:.1f}pp) — ExoMy adapts reasonably"
    else:
        verdict = f"SMALL domain gap ({domain_gap:.1f}pp) — strong generalisation"
    print(f"  Verdict: {verdict}\n")


def save_csv(results: dict, images_dir: str, suffix: str = "") -> str:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    # A non-empty suffix (e.g. "_grayscale") writes to a separate file so a
    # re-run under a different preprocessing ablation never overwrites the
    # original RGB baseline (sim_to_real_gap_results.csv, 20.0% overall,
    # already cited in Ch4 SS4.8.12 / Ch5 SS5.5.4c).
    path = os.path.join(RESULTS_DIR, f"sim_to_real_gap_results{suffix}.csv")
    fieldnames = ["timestamp", "images_dir", "class", "n_images",
                  "correct", "accuracy_pct", "avg_conf",
                  "ai4mars_baseline_pct", "gap_pp"]
    rows = []
    ts   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    for cls, stats in results["per_class"].items():
        baseline = AI4MARS_BASELINE.get(cls)
        gap      = (stats["accuracy"] - baseline) if baseline is not None else None
        rows.append({
            "timestamp":          ts,
            "images_dir":         images_dir,
            "class":              cls,
            "n_images":           stats["total"],
            "correct":            stats["correct"],
            "accuracy_pct":       round(stats["accuracy"], 2),
            "avg_conf":           round(stats["avg_conf"], 4),
            "ai4mars_baseline_pct": baseline if baseline is not None else "",
            "gap_pp":             round(gap, 2) if gap is not None else "",
        })
    # Overall row
    overall_gap = results["overall"] - AI4MARS_BASELINE["overall"]
    rows.append({
        "timestamp":          ts,
        "images_dir":         images_dir,
        "class":              "OVERALL",
        "n_images":           results["n_total"],
        "correct":            results["n_correct"],
        "accuracy_pct":       round(results["overall"], 2),
        "avg_conf":           "",
        "ai4mars_baseline_pct": AI4MARS_BASELINE["overall"],
        "gap_pp":             round(overall_gap, 2),
    })

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Sim-to-real domain gap test for DINOv2+reg ViT-S/14 ONNX.\n"
            "Evaluates model on real ExoMy camera images and compares to AI4Mars baseline.\n"
            "Images must be organised into subdirectories by terrain class."
        )
    )
    parser.add_argument(
        "--images-dir", required=True,
        help="Root directory containing soil/, bedrock/, sand/, big_rock/ subdirs."
    )
    parser.add_argument(
        "--onnx-path", default=ONNX_PATH,
        help=f"Path to DINOv2 ONNX encoder (default: {ONNX_PATH})"
    )
    parser.add_argument(
        "--probe-path", default=PROBE_PATH,
        help=f"Path to LogReg probe .pkl (default: {PROBE_PATH})"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print per-image prediction results."
    )
    parser.add_argument(
        "--grayscale", action="store_true",
        help=(
            "Convert images to grayscale (3-channel replicate, ITU-R BT.601) before "
            "inference -- ablation targeting the NAVCAM-grayscale-training vs "
            "RPi-RGB-inference luminance mismatch identified as Exp 5b's error mechanism. "
            "Writes to sim_to_real_gap_results_grayscale.csv, not the RGB baseline file."
        )
    )
    args = parser.parse_args()

    images_dir = os.path.abspath(args.images_dir)
    if not os.path.isdir(images_dir):
        sys.exit(f"[ERROR] Images directory not found: {images_dir}")

    print(f"\n=== Sim-to-Real Domain Gap Test ===")
    print(f"  Images dir:  {images_dir}")
    print(f"  ONNX model:  {args.onnx_path}")
    print(f"  Probe:       {args.probe_path}")
    print(f"  Baseline:    AI4Mars DINOv2+reg ViT-S/14 @ 1000-shot = {AI4MARS_BASELINE['overall']}%")

    dataset = discover_images(images_dir)
    if not dataset:
        sys.exit(
            f"[ERROR] No images found under {images_dir}\n"
            "  Expected subdirs: soil/, bedrock/, sand/, big_rock/\n"
            "  Run: python3 experiments/save_camera_frame.py --output <images-dir>/soil/"
        )

    total_imgs = sum(len(v) for v in dataset.values())
    print(f"\n  Found {total_imgs} images across {len(dataset)} classes: "
          f"{list(dataset.keys())}")
    print("\n  Loading model...")
    session, probe = load_model(args.onnx_path, args.probe_path)
    print("  Model loaded. Running inference...\n")

    if args.grayscale:
        print("  Preprocessing: GRAYSCALE ablation (3-channel replicate, ITU-R BT.601)")
    results = evaluate(session, probe, dataset, args.verbose, grayscale=args.grayscale)
    print_comparison(results)

    csv_path = save_csv(results, images_dir, suffix="_grayscale" if args.grayscale else "")
    print(f"  Results saved → {csv_path}")


if __name__ == "__main__":
    main()
