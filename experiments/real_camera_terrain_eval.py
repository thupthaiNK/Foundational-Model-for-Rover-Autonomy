#!/usr/bin/env python3
"""
Purpose: Evaluate DINOv2 terrain classification on real frames captured from the
         rover's own camera, and settle whether the grayscale preprocessing
         ablation helps on real hardware.

         Two gaps this closes. First, the only live camera test so far
         (2026-07-28) pointed at one static indoor scene and returned
         soil:0.99 for 1090 consecutive frames. That shows the pipeline runs;
         it says nothing about whether the model can tell surfaces apart,
         which is the actual claim the thesis makes. Second, and more
         awkwardly, that run used grayscale_preprocessing=False while
         real_hardware_deployment.launch.py sets it True -- so the
         configuration that will actually fly has never been tested on the
         real camera at all, even though it is the one an earlier offline
         ablation preferred (Exp 5b 20.0% -> 25.0%).

         Running offline on captured frames rather than live on the Pi is
         deliberate: the INT8 ONNX graph is deterministic, so the same pixels
         give the same features anywhere, and the live path is confounded by
         thermal throttling (83.7C, throttled=0xe0008 observed) and by the
         camera node competing for CPU. Colour and grayscale are compared on
         byte-identical images, which live A/B testing cannot do.

Inputs:  Frames captured by ros2_ws/docker/capture_terrain_frames.py, copied
         off the Pi. Directory layout: <root>/<surface>/frame_NNN.png, with the
         directory name being the surface the camera was pointed at.
         Also experiments/results/dinov2_reg_small_encoder_int8.onnx and the
         1000-shot feature cache.
Outputs: Console per-surface breakdown for both preprocessing modes, plus
         experiments/results/real_camera_terrain_eval.csv (per frame: surface,
         file, colour prediction + confidence, grayscale prediction +
         confidence, mean/min/max pixel value for an exposure sanity check).
How to run:
    python3 -u experiments/real_camera_terrain_eval.py --root data/camera_captures
Project: Onboard Visual Foundation Models for Mars Terrain Perception and Rover Navigation
"""

import argparse
import csv
import os
import sys

import numpy as np
import onnxruntime as ort
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.normpath(os.path.join(_THIS_DIR, ".."))
sys.path.insert(0, os.path.join(_REPO_ROOT, "ros2_ws", "src", "fm_perception"))

from fm_perception.dinov2_preprocess import preprocess_dinov2  # noqa: E402
from fm_perception.dinov2_terrain_node import to_grayscale_3ch  # noqa: E402

RESULTS_DIR = os.path.join(_THIS_DIR, "results")
ONNX_ENCODER = os.path.join(RESULTS_DIR, "dinov2_reg_small_encoder_int8.onnx")
TRAIN_CACHE = os.path.join(RESULTS_DIR, "feature_cache", "dinov2_reg_small_train_1000shot.npz")
OUT_CSV = os.path.join(RESULTS_DIR, "real_camera_terrain_eval.csv")

CLASS_NAMES = ["soil", "bedrock", "sand", "big_rock"]
T_STAR = 0.461
LOGR_C = 0.316
N_SHOT = 1000
CONFIDENCE_THRESHOLD = 0.40   # dinov2_terrain_node's default; below this -> uncertain


def train_probe():
    data = np.load(TRAIN_CACHE)
    feats, labels = data["feats"], data["labels"]
    idx = []
    for c in np.unique(labels):
        idx.extend(np.where(labels == c)[0][:N_SHOT].tolist())
    idx = np.array(idx)
    clf = LogisticRegression(C=LOGR_C, max_iter=1000, solver="lbfgs", random_state=42)
    clf.fit(normalize(feats[idx], norm="l2"), labels[idx])
    return clf


def classify(session, clf, pil_img):
    """Exactly dinov2_terrain_node's INT8 path: preprocess, ONNX, L2, temperature-scaled softmax."""
    feat = normalize(session.run(None, {"pixel_values": preprocess_dinov2(pil_img)})[0], norm="l2")
    logits = clf.decision_function(feat)[0]
    scaled = logits / T_STAR
    scaled -= scaled.max()
    exp_s = np.exp(scaled)
    probs = exp_s / exp_s.sum()
    full = np.zeros(len(CLASS_NAMES), dtype=np.float32)
    for i, cls in enumerate(clf.classes_):
        full[cls] = probs[i]
    idx = int(full.argmax())
    conf = float(full[idx])
    label = CLASS_NAMES[idx] if conf >= CONFIDENCE_THRESHOLD else "uncertain"
    return label, conf


def find_frames(root):
    """[(surface, path)] from <root>/<surface>/*.png."""
    out = []
    for surface in sorted(os.listdir(root)):
        d = os.path.join(root, surface)
        if not os.path.isdir(d):
            continue
        for name in sorted(os.listdir(d)):
            if name.lower().endswith((".png", ".jpg", ".jpeg")):
                out.append((surface, os.path.join(d, name)))
    return out


def summarise(rows, pred_key, conf_key, title):
    print(f"\n{title}")
    print(f"  {'surface':<14}{'n':>4}  {'predictions (share)':<44}{'mean conf':>10}")
    surfaces = sorted({r["surface"] for r in rows})
    for s in surfaces:
        sub = [r for r in rows if r["surface"] == s]
        counts = {}
        for r in sub:
            counts[r[pred_key]] = counts.get(r[pred_key], 0) + 1
        share = "  ".join(
            f"{k} {100.0 * v / len(sub):.0f}%"
            for k, v in sorted(counts.items(), key=lambda kv: -kv[1])
        )
        mean_conf = float(np.mean([r[conf_key] for r in sub]))
        print(f"  {s:<14}{len(sub):>4}  {share:<44}{mean_conf:>10.3f}")

    # The question the static-scene run could not answer: does the output vary
    # with the surface at all, or is it constant regardless of what it sees?
    distinct = {r[pred_key] for r in rows}
    print(f"  distinct predictions across all surfaces: {len(distinct)} {sorted(distinct)}")
    if len(distinct) == 1 and len(surfaces) > 1:
        print("  WARNING: one label for every surface. The model is not discriminating "
              "here; a high confidence means confidently uniform, not correct.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True,
                    help="directory of <surface>/ subdirectories of captured frames")
    args = ap.parse_args()

    for p in (ONNX_ENCODER, TRAIN_CACHE):
        if not os.path.exists(p):
            raise FileNotFoundError(p)
    if not os.path.isdir(args.root):
        raise NotADirectoryError(args.root)

    frames = find_frames(args.root)
    if not frames:
        raise SystemExit(f"no images found under {args.root}/<surface>/")

    session = ort.InferenceSession(ONNX_ENCODER, providers=["CPUExecutionProvider"])
    clf = train_probe()
    print(f"Frames: {len(frames)} across "
          f"{len({s for s, _ in frames})} surfaces\n")

    rows = []
    for i, (surface, path) in enumerate(frames, 1):
        img = Image.open(path).convert("RGB")
        arr = np.asarray(img)

        label_c, conf_c = classify(session, clf, img)
        gray = Image.fromarray(to_grayscale_3ch(arr), "RGB")
        label_g, conf_g = classify(session, clf, gray)

        rows.append({
            "surface": surface,
            "file": os.path.relpath(path, args.root),
            "pred_colour": label_c,
            "conf_colour": round(conf_c, 4),
            "pred_grayscale": label_g,
            "conf_grayscale": round(conf_g, 4),
            # Exposure sanity check: a mean near 255 or near 0 means the frame
            # is blown out or black, and no classification of it means much.
            "pixel_mean": round(float(arr.mean()), 1),
            "pixel_min": int(arr.min()),
            "pixel_max": int(arr.max()),
            "pct_saturated": round(100.0 * float((arr >= 250).mean()), 2),
        })
        if i % 10 == 0 or i == len(frames):
            print(f"  {i}/{len(frames)}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print("\n" + "=" * 74)
    print("DINOv2 on real rover camera frames — colour vs grayscale, same pixels")
    print("=" * 74)
    summarise(rows, "pred_colour", "conf_colour",
              "Colour (grayscale_preprocessing=False)")
    summarise(rows, "pred_grayscale", "conf_grayscale",
              "Grayscale (grayscale_preprocessing=True — what the launch file sets)")

    changed = sum(1 for r in rows if r["pred_colour"] != r["pred_grayscale"])
    print(f"\nFrames where grayscale changed the prediction: {changed}/{len(rows)} "
          f"({100.0 * changed / len(rows):.1f}%)")

    print("\nExposure check (a blown-out frame classifies as noise, whatever it says)")
    for s in sorted({r["surface"] for r in rows}):
        sub = [r for r in rows if r["surface"] == s]
        print(f"  {s:<14} mean {np.mean([r['pixel_mean'] for r in sub]):>6.1f}  "
              f"saturated pixels {np.mean([r['pct_saturated'] for r in sub]):>5.2f}%")

    print(f"\nPer-frame CSV: {OUT_CSV}")
    print("No ground truth is assumed here: these are field frames, not a labelled "
          "benchmark.\nWhat this answers is whether the output varies with the surface "
          "and whether\ngrayscale changes it, not an accuracy figure.")


if __name__ == "__main__":
    main()
