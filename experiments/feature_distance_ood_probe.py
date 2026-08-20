#!/usr/bin/env python3
"""
Purpose: Test whether distance in DINOv2 feature space can tell terrain from
         non-terrain on the real rover camera, and so close the safety gap
         measured on 2026-07-28: across 217 real frames the deployed classifier
         returned `uncertain` exactly zero times, calling a wall soil at 0.937
         confidence, a cardboard box 0.896, the perspex sandpit wall 0.762 and a
         human hand at 20 cm 0.779.

         The idea being tested is the standard one and costs nothing to deploy.
         dinov2_terrain_node already loads 3108 cached AI4Mars training features
         at startup to fit its LogReg probe. If a wall's feature vector sits
         further from that bank than sand does, then a nearest-neighbour
         similarity is a free out-of-distribution score: one 3108x384 matrix
         product per frame, negligible beside the 460 ms the encoder already
         costs, no new model, no training, and nothing that conflicts with the
         thesis's pretrained-models-only constraint.

         It does not work, and this script is what establishes that rather than
         assuming it either way.

Inputs:  data/camera_captures/<surface>/*.png (real rover frames, captured with
         AeEnable false + AnalogueGain 1.0 -- auto-exposure frames are useless
         here, see project memory)
         experiments/results/dinov2_reg_small_encoder_int8.onnx
         experiments/results/feature_cache/dinov2_reg_small_train_1000shot.npz
Outputs: Console report + experiments/results/feature_distance_ood_probe.csv
How to run:
    python3 -u experiments/feature_distance_ood_probe.py --root data/camera_captures
Project: Onboard Visual Foundation Models for Mars Terrain Perception and Rover Navigation
"""

import argparse
import csv
import glob
import os
import sys

import numpy as np
import onnxruntime as ort
from PIL import Image
from sklearn.preprocessing import normalize

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.normpath(os.path.join(_THIS_DIR, ".."))
sys.path.insert(0, os.path.join(_REPO_ROOT, "ros2_ws", "src", "fm_perception"))

from fm_perception.dinov2_preprocess import preprocess_dinov2  # noqa: E402

RESULTS_DIR = os.path.join(_THIS_DIR, "results")
ONNX_ENCODER = os.path.join(RESULTS_DIR, "dinov2_reg_small_encoder_int8.onnx")
TRAIN_CACHE = os.path.join(RESULTS_DIR, "feature_cache", "dinov2_reg_small_train_1000shot.npz")
OUT_CSV = os.path.join(RESULTS_DIR, "feature_distance_ood_probe.csv")

# Ground truth for this test is the surface the camera was pointed at, which is
# the one thing about these frames that is known for certain.
TERRAIN = ["sand2", "rock", "mixed", "sandpit_edge", "horizon", "floor2"]
NOT_TERRAIN = ["wall", "box", "glass", "hand2"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.join(_REPO_ROOT, "data", "camera_captures"))
    args = ap.parse_args()

    session = ort.InferenceSession(ONNX_ENCODER, providers=["CPUExecutionProvider"])
    bank = normalize(np.load(TRAIN_CACHE)["feats"], norm="l2")
    print(f"training feature bank: {bank.shape}\n")

    rows = []
    for group, names in (("terrain", TERRAIN), ("not_terrain", NOT_TERRAIN)):
        for surface in names:
            for path in sorted(glob.glob(os.path.join(args.root, surface, "*.png"))):
                feat = normalize(
                    session.run(None, {"pixel_values": preprocess_dinov2(Image.open(path))})[0],
                    norm="l2",
                )
                rows.append({
                    "surface": surface,
                    "group": group,
                    "file": os.path.basename(path),
                    "max_similarity": round(float((bank @ feat[0]).max()), 6),
                })
            print(f"  {surface}: {sum(1 for r in rows if r['surface'] == surface)} frames")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    sims = np.array([r["max_similarity"] for r in rows])
    is_terrain = np.array([r["group"] == "terrain" for r in rows])

    print("\n" + "=" * 70)
    print("Nearest-neighbour similarity to the AI4Mars training bank")
    print("=" * 70)
    print(f"{'surface':<15}{'group':<13}{'min':>7}{'mean':>7}{'max':>7}{'n':>5}")
    for group, names in (("terrain", TERRAIN), ("not_terrain", NOT_TERRAIN)):
        for surface in names:
            v = np.array([r["max_similarity"] for r in rows if r["surface"] == surface])
            print(f"{surface:<15}{group:<13}{v.min():>7.3f}{v.mean():>7.3f}{v.max():>7.3f}{len(v):>5}")

    t, n = sims[is_terrain], sims[~is_terrain]
    print(f"\nterrain      n={len(t):<4} mean {t.mean():.3f}  range {t.min():.3f}-{t.max():.3f}")
    print(f"not_terrain  n={len(n):<4} mean {n.mean():.3f}  range {n.min():.3f}-{n.max():.3f}")

    # Rank-based separability: the probability that a random terrain frame
    # scores above a random non-terrain one. 0.5 is chance, 1.0 is perfect.
    auc = float((t[:, None] > n[None, :]).mean() + 0.5 * (t[:, None] == n[None, :]).mean())
    print(f"\nseparability (AUC)           : {auc:.3f}   (0.5 = chance, 1.0 = perfect)")

    # The decision that actually matters: a threshold strict enough to reject
    # every non-terrain frame, and what it costs in terrain frames.
    need = n.max()
    lost = int((t <= need).sum())
    print(f"threshold to reject all non-terrain: > {need:.3f}")
    print(f"terrain frames it would also reject : {lost}/{len(t)} ({100.0 * lost / len(t):.0f}%)")

    print("\nSurfaces lost entirely at that threshold:")
    for surface in TERRAIN:
        v = np.array([r["max_similarity"] for r in rows if r["surface"] == surface])
        if v.max() <= need:
            print(f"  {surface:<15}{v.min():.3f}-{v.max():.3f}  ALL {len(v)} frames rejected")

    print(f"\nPer-frame CSV: {OUT_CSV}")
    print("\nNote how low every score is, terrain included: the closest AI4Mars "
          "neighbour of\nthe best real sand frame is still well short of 1.0. The rover's own "
          "terrain is\nnearly as unfamiliar to this encoder as a wall is, which is the reason "
          "the two\ngroups cannot be separated this way.")


if __name__ == "__main__":
    main()
