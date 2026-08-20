"""
Purpose: Systematic prompt engineering for CLIP zero-shot terrain classification.
         Tests single-prompt and multi-prompt ensemble strategies on AI4Mars.
Inputs:  AI4Mars gold-standard test set (322 images, local)
Outputs: Per-class accuracy table, confusion matrix, optional wandb log and CSV
How to run:
    python3 clip_prompt_engineering.py --cpu              # all single-prompt sets (v1-v6)
    python3 clip_prompt_engineering.py --cpu --ensemble   # ensemble sets only (v7, v8)
    python3 clip_prompt_engineering.py --cpu --all        # all sets + ensemble
    python3 clip_prompt_engineering.py --cpu --all --log  # + log best to wandb
    python3 clip_prompt_engineering.py --cpu --all --csv  # + save CSV for thesis table
    python3 clip_prompt_engineering.py --cpu --model ViT-L/14  # larger CLIP model
Project: Onboard Visual Foundation Models for Mars Terrain Perception and Rover Navigation
"""

import argparse
import csv
import os
import time

import clip
import numpy as np
import torch
import wandb
from PIL import Image

# ── Dataset ───────────────────────────────────────────────────────────────────
AI4MARS_BASE   = "/mnt/c/Users/DELL/Desktop/Thesis/github source/ai4mars-dataset-merged-0.1"
AI4MARS_IMAGES = os.path.join(AI4MARS_BASE, "msl/images/edr")
AI4MARS_LABELS = os.path.join(AI4MARS_BASE, "msl/labels/test/masked-gold-min3-100agree")

CLASS_NAMES         = ["soil", "bedrock", "sand", "big_rock"]
IGNORE_PIXEL        = 255
SUPERVISED_BASELINE = 96.67  # DeepLabv3+ ResNet-101, Swan et al. 2021

# ── Single-prompt sets ────────────────────────────────────────────────────────
# Each entry: list of 4 strings, one per class [soil, bedrock, sand, big_rock]
PROMPT_SETS = {
    "v1": {
        "description": "Baseline (Experiment 1 — 34.8%)",
        "prompts": [
            "fine-grained loose soil on Mars surface",
            "flat smooth bedrock rock surface on Mars",
            "bright sandy terrain or sand dunes on Mars",
            "large boulders or big rocks on Mars terrain",
        ],
    },
    "v2": {
        "description": "Continuity vs isolation",
        "prompts": [
            "fine-grained loose soil and dust covering Mars ground",
            "continuous flat exposed rock layer covering the Mars ground like a floor",
            "bright smooth sandy terrain with fine sand grains on Mars",
            "isolated large individual boulders and rocks scattered on Mars soil",
        ],
    },
    "v3": {
        "description": "Geological + shadow cue",
        "prompts": [
            "powdery fine regolith and loose soil covering Mars surface",
            "flat rocky outcrop pavement on Mars with no individual boulders",
            "wind-blown sand ripples and dunes on bright Mars terrain",
            "large Mars rocks and boulders with shadows on the ground",
        ],
    },
    "v4": {
        "description": "Texture + scale contrast (best overall so far — 52.6%)",
        "prompts": [
            "Mars ground covered in fine-grained soil, dust, and small pebbles",
            "Mars rocky ground where the floor itself is made of solid flat rock",
            "Mars desert with smooth sandy surface and light-coloured sand",
            "Mars terrain with several large prominent rocks and boulders",
        ],
    },
    "v5": {
        "description": "Best-per-class mix (soil←v1, bedrock←v4, sand←v2)",
        # Per-class winners: soil v1=64.2%, bedrock v4=91.3%, sand v2=30.6%
        "prompts": [
            "fine-grained loose soil on Mars surface",
            "Mars rocky ground where the floor itself is made of solid flat rock",
            "bright smooth sandy terrain with fine sand grains on Mars",
            "isolated large individual boulders and rocks scattered on Mars soil",
        ],
    },
    "v6": {
        "description": "NAVCAM domain prefix (a photo of... pattern)",
        # CLIP trained on image-caption pairs — "a photo of X" aligns with training
        "prompts": [
            "a Mars rover NAVCAM grayscale image of loose soil and fine regolith on the ground",
            "a Mars rover NAVCAM grayscale image of solid flat bedrock forming the entire ground surface",
            "a Mars rover NAVCAM grayscale image of smooth sandy terrain and wind-blown sand",
            "a Mars rover NAVCAM grayscale image of large isolated boulders and rocks on Mars",
        ],
    },
}

# ── Multi-prompt ensemble sets ────────────────────────────────────────────────
# Each class gets a LIST of prompts; similarities are averaged before softmax.
# Structure: list of 4 lists (one list per class)
ENSEMBLE_SETS = {
    "v7": {
        "description": "Ensemble mean: top prompts per class (3 each)",
        "agg": "mean",
        "prompts": [
            # soil — combine v1 (64.2%) style + variations
            [
                "fine-grained loose soil on Mars surface",
                "fine-grained loose soil and dust covering Mars ground",
                "loose dusty regolith and fine soil covering the Mars ground",
            ],
            # bedrock — v4 won (91.3%); add variations to stabilise
            [
                "Mars rocky ground where the floor itself is made of solid flat rock",
                "continuous flat exposed rock layer covering the Mars ground like a floor",
                "solid flat rock pavement covering the entire Mars ground surface",
            ],
            # sand — v2 won (30.6%); add variations
            [
                "bright smooth sandy terrain with fine sand grains on Mars",
                "bright sandy terrain or sand dunes on Mars",
                "light-coloured smooth sand covering Mars terrain",
            ],
            # big_rock — no dominant test images; keep distinct to avoid stealing other classes
            [
                "isolated large individual boulders and rocks scattered on Mars soil",
                "Mars terrain with several large prominent rocks and boulders",
                "large Mars rocks and boulders sitting on the ground surface",
            ],
        ],
    },
    "v8": {
        "description": "Ensemble mean: strong bedrock + richer soil/sand prompts",
        "agg": "mean",
        "prompts": [
            # soil — richer descriptions to compete with strong bedrock prompts
            [
                "fine-grained loose soil on Mars surface",
                "dark loose unconsolidated soil and dust on Mars ground",
                "Mars surface covered in soft loose soil with no visible rock",
            ],
            # bedrock — aggressive v4-style
            [
                "Mars rocky ground where the floor itself is made of solid flat rock",
                "continuous unbroken rock layer forming the Mars ground floor",
                "flat solid bedrock outcrop covering the Mars surface",
            ],
            # sand — emphasise brightness and smoothness vs rocky terrain
            [
                "bright smooth sandy terrain with fine sand grains on Mars",
                "light bright sandy Mars surface with uniform fine texture",
                "wind-deposited smooth sand on bright Mars terrain",
            ],
            # big_rock
            [
                "isolated large individual boulders and rocks scattered on Mars soil",
                "large prominent rocks and boulders sitting on Mars ground",
                "distinct large rocks with shadows resting on Mars soil",
            ],
        ],
    },
    "v9": {
        # KEY INSIGHT from v4 vs v6:
        #   v6 NAVCAM prefix → Soil 96.7% but Bedrock 5.4%
        #   v4 texture+scale → Bedrock 91.3% but Soil 45.5%
        # v9 combines both: soil uses v6-style, bedrock uses v4-style
        "description": "Hybrid: v6-soil (96.7%) + v4-bedrock (91.3%) combined",
        "agg": "mean",
        "prompts": [
            # soil: use NAVCAM prefix (gave 96.7% in v6)
            [
                "a Mars rover NAVCAM grayscale image of loose soil and fine regolith on the ground",
                "fine-grained loose soil on Mars surface",
                "loose dusty regolith and fine soil covering the Mars ground",
            ],
            # bedrock: use v4 texture+scale (gave 91.3%)
            [
                "Mars rocky ground where the floor itself is made of solid flat rock",
                "continuous flat exposed rock layer covering the Mars ground like a floor",
                "solid flat rock pavement covering the entire Mars ground surface",
            ],
            # sand: mix v6-NAVCAM + v2 (v6 gave 31.9%, v2 gave 30.6%)
            [
                "a Mars rover NAVCAM grayscale image of smooth sandy terrain and wind-blown sand",
                "bright smooth sandy terrain with fine sand grains on Mars",
                "bright sandy terrain or sand dunes on Mars",
            ],
            # big_rock: v2 isolation emphasis
            [
                "isolated large individual boulders and rocks scattered on Mars soil",
                "Mars terrain with several large prominent rocks and boulders",
                "a Mars rover NAVCAM grayscale image of large isolated boulders and rocks on Mars",
            ],
        ],
    },
    "v10": {
        # Same as v9 but uses MAX aggregation instead of mean
        # MAX = more aggressive: any single strong prompt match wins
        "description": "Hybrid v9 with MAX aggregation (aggressive match)",
        "agg": "max",
        "prompts": [
            [
                "a Mars rover NAVCAM grayscale image of loose soil and fine regolith on the ground",
                "fine-grained loose soil on Mars surface",
                "loose dusty regolith and fine soil covering the Mars ground",
            ],
            [
                "Mars rocky ground where the floor itself is made of solid flat rock",
                "continuous flat exposed rock layer covering the Mars ground like a floor",
                "solid flat rock pavement covering the entire Mars ground surface",
            ],
            [
                "a Mars rover NAVCAM grayscale image of smooth sandy terrain and wind-blown sand",
                "bright smooth sandy terrain with fine sand grains on Mars",
                "bright sandy terrain or sand dunes on Mars",
            ],
            [
                "isolated large individual boulders and rocks scattered on Mars soil",
                "Mars terrain with several large prominent rocks and boulders",
                "a Mars rover NAVCAM grayscale image of large isolated boulders and rocks on Mars",
            ],
        ],
    },
    "v11": {
        # REBALANCED: reduce NAVCAM soil from 3→1 prompt so bedrock can compete back
        # soil: 1 NAVCAM + 2 regular (less dominant than v9's 3 NAVCAM)
        # bedrock: 4 prompts (strongest possible set from all experiments)
        "description": "Rebalanced: 1 NAVCAM-soil + 4 bedrock prompts",
        "agg": "mean",
        "prompts": [
            # soil: 1 NAVCAM to leverage 96.7% strength, + 2 regular for balance
            [
                "a Mars rover NAVCAM grayscale image of loose soil and fine regolith on the ground",
                "fine-grained loose soil on Mars surface",
                "loose fine-grained dusty soil covering the Mars ground",
            ],
            # bedrock: maximum number of prompts to compete with soil
            [
                "Mars rocky ground where the floor itself is made of solid flat rock",
                "continuous flat exposed rock layer covering the Mars ground like a floor",
                "solid flat rock pavement covering the entire Mars ground surface",
                "flat solid continuous bedrock forming the Mars terrain floor",
            ],
            # sand: v6 NAVCAM + v2 + v1 (best sand trio)
            [
                "a Mars rover NAVCAM grayscale image of smooth sandy terrain and wind-blown sand",
                "bright smooth sandy terrain with fine sand grains on Mars",
                "bright sandy terrain or sand dunes on Mars",
            ],
            # big_rock
            [
                "isolated large individual boulders and rocks scattered on Mars soil",
                "Mars terrain with several large prominent rocks and boulders",
                "a Mars rover NAVCAM grayscale image of large isolated boulders on Mars",
            ],
        ],
    },
    "v12": {
        # WEIGHTED ENSEMBLE: same prompts as v11 but use mean of NORMALIZED per-class scores
        # Adds per-class prompt count balancing — each class gets equal vote regardless of n prompts
        # Implementation: same as v11 (mean already normalises across prompts within a class)
        # Key difference: bedrock gets only 2 prompts (same count as soil) — purer competition
        "description": "Balanced count: 2 prompts per class (best 2 each)",
        "agg": "mean",
        "prompts": [
            # soil: NAVCAM (v6 best) + original (v1 best)
            [
                "a Mars rover NAVCAM grayscale image of loose soil and fine regolith on the ground",
                "fine-grained loose soil on Mars surface",
            ],
            # bedrock: v4 best + v2 second best
            [
                "Mars rocky ground where the floor itself is made of solid flat rock",
                "continuous flat exposed rock layer covering the Mars ground like a floor",
            ],
            # sand: v6 NAVCAM + v2 best
            [
                "a Mars rover NAVCAM grayscale image of smooth sandy terrain and wind-blown sand",
                "bright smooth sandy terrain with fine sand grains on Mars",
            ],
            # big_rock: v2 best + v4 second
            [
                "isolated large individual boulders and rocks scattered on Mars soil",
                "Mars terrain with several large prominent rocks and boulders",
            ],
        ],
    },
}


# ── Device ────────────────────────────────────────────────────────────────────

def get_device(force_cpu: bool) -> str:
    if force_cpu:
        return "cpu"
    if torch.cuda.is_available():
        try:
            torch.zeros(1).cuda()
            return "cuda"
        except Exception:
            print("CUDA available but failed (MX330 issue) — falling back to CPU")
    return "cpu"


# ── Ground truth ──────────────────────────────────────────────────────────────

def dominant_class(label_path: str):
    label = np.array(Image.open(label_path))
    valid = label[label != IGNORE_PIXEL]
    if len(valid) == 0:
        return None
    return int(np.argmax(np.bincount(valid, minlength=4)))


def load_test_set():
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
            pairs.append((image_path, gt))
    return pairs


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(model, preprocess, prompts, test_pairs, device, agg="single"):
    """
    prompts: list[str] of length 4 (single) OR list[list[str]] of length 4 (ensemble)
    agg: "single" | "mean" | "max"
    Returns dict with overall, per_class, avg_ms, fps, n_images, confusion.
    """
    n_classes = len(CLASS_NAMES)

    # Encode text features once
    if agg == "single":
        tokens = clip.tokenize(prompts).to(device)
        with torch.no_grad():
            txt = model.encode_text(tokens)
            txt /= txt.norm(dim=-1, keepdim=True)
        class_feats = [txt[i].unsqueeze(0) for i in range(n_classes)]
    else:
        class_feats = []
        for cls_prompts in prompts:
            tokens = clip.tokenize(cls_prompts).to(device)
            with torch.no_grad():
                feats = model.encode_text(tokens)
                feats /= feats.norm(dim=-1, keepdim=True)
            class_feats.append(feats)

    correct   = {c: 0 for c in range(n_classes)}
    total     = {c: 0 for c in range(n_classes)}
    times     = []
    confusion = np.zeros((n_classes, n_classes), dtype=int)

    for image_path, gt in test_pairs:
        image = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)

        t0 = time.perf_counter()
        with torch.no_grad():
            img_feat = model.encode_image(image)
            img_feat /= img_feat.norm(dim=-1, keepdim=True)   # [1, D]

            scores = []
            for feats in class_feats:                           # feats: [k, D]
                sims = (100.0 * img_feat @ feats.T).squeeze(0) # [k]
                scores.append(sims.mean() if agg != "max" else sims.max())

            pred = int(torch.stack(scores).softmax(dim=0).argmax())

        times.append((time.perf_counter() - t0) * 1000)
        correct[gt] += int(pred == gt)
        total[gt]   += 1
        confusion[gt][pred] += 1

    per_class = {
        name: (correct[i] / total[i] * 100 if total[i] > 0 else None)
        for i, name in enumerate(CLASS_NAMES)
    }
    n = sum(total.values())
    return {
        "overall":   sum(correct.values()) / n * 100 if n > 0 else 0,
        "per_class": per_class,
        "avg_ms":    float(np.mean(times)),
        "fps":       float(1000 / np.mean(times)),
        "n_images":  n,
        "confusion": confusion,
    }


# ── Reporting ─────────────────────────────────────────────────────────────────

def print_comparison(results: dict):
    keys  = list(results.keys())
    col_w = 10

    width = 18 + col_w * len(keys)
    print("\n" + "=" * width)
    print("PROMPT ENGINEERING RESULTS — CLIP on AI4Mars (287 images)")
    print("=" * width)

    header = f"{'':18}" + "".join(f"{k:>{col_w}}" for k in keys)
    print(header)
    print("-" * width)

    for name in CLASS_NAMES:
        row = f"{name:<18}"
        for k in keys:
            val = results[k]["per_class"].get(name)
            row += f"  {val:>6.1f}%" if val is not None else f"  {'N/A':>6}"
        print(row)

    print("-" * width)
    print(f"{'OVERALL':<18}" + "".join(f"  {results[k]['overall']:>6.1f}%" for k in keys))
    print(f"{'avg ms/img':<18}" + "".join(f"  {results[k]['avg_ms']:>7.1f}" for k in keys))
    print(f"{'gap vs 96.67%':<18}" +
          "".join(f"  {results[k]['overall'] - SUPERVISED_BASELINE:>+6.1f}%" for k in keys))
    print("=" * width)

    best_key = max(results, key=lambda k: results[k]["overall"])
    best_res = results[best_key]
    print(f"\nBest overall: {best_key}  →  {best_res['overall']:.1f}%  "
          f"(gap {best_res['overall'] - SUPERVISED_BASELINE:+.1f}%)")

    # Per-class best
    print("\nBest per class:")
    for name in CLASS_NAMES:
        vals = {k: results[k]["per_class"].get(name) for k in keys}
        vals = {k: v for k, v in vals.items() if v is not None}
        if vals:
            bk = max(vals, key=vals.get)
            print(f"  {name:<10} → {bk}  {vals[bk]:.1f}%")

    return best_key, best_res


def print_confusion(key: str, res: dict):
    cm = res["confusion"]
    short = [n[:4] for n in CLASS_NAMES]
    print(f"\nConfusion Matrix — {key}  (row=actual, col=predicted):")
    print(f"  {'':>10}" + "".join(f"  {s:>7}" for s in short))
    for i, name in enumerate(CLASS_NAMES):
        row_total = cm[i].sum()
        acc = cm[i][i] / row_total * 100 if row_total > 0 else 0
        print(f"  {name:>10}" +
              "".join(f"  {cm[i][j]:>7}" for j in range(4)) +
              f"   ({acc:.0f}%)")


def save_csv(results: dict, model_name: str):
    out_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"prompt_engineering_{model_name.replace('/', '-')}.csv")

    fieldnames = ["version", "overall", "gap"] + CLASS_NAMES + ["avg_ms"]
    rows = []
    for key, res in results.items():
        row = {
            "version": key,
            "overall": f"{res['overall']:.2f}",
            "gap":     f"{res['overall'] - SUPERVISED_BASELINE:.2f}",
            "avg_ms":  f"{res['avg_ms']:.1f}",
        }
        for name in CLASS_NAMES:
            val = res["per_class"].get(name)
            row[name] = f"{val:.2f}" if val is not None else "N/A"
        rows.append(row)

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nCSV saved → {path}")


def log_to_wandb(key: str, res: dict, model_name: str):
    run_name = f"clip-{model_name.replace('/', '-').lower()}-prompt-{key}"
    wandb.init(
        project="rover-autonomy-thesis",
        name=run_name,
        config={
            "model":      model_name,
            "experiment": "prompt_engineering",
            "prompt_set": key,
            "dataset":    "AI4Mars gold-standard (masked-gold-min3-100agree)",
            "n_images":   res["n_images"],
        },
    )
    pc = res["per_class"]
    wandb.log({
        "accuracy/overall":             res["overall"],
        "accuracy/soil":                pc.get("soil")     or 0,
        "accuracy/bedrock":             pc.get("bedrock")  or 0,
        "accuracy/sand":                pc.get("sand")     or 0,
        "accuracy/big_rock":            pc.get("big_rock") or 0,
        "accuracy/supervised_baseline": SUPERVISED_BASELINE,
        "accuracy/gap_vs_baseline":     res["overall"] - SUPERVISED_BASELINE,
        "inference/avg_ms":             res["avg_ms"],
        "inference/fps":                res["fps"],
    })
    wandb.finish()
    print(f"Logged → rover-autonomy-thesis / {run_name}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="CLIP prompt engineering for AI4Mars")
    parser.add_argument("--set",      choices=list(PROMPT_SETS.keys()),
                        help="Single prompt set to test (default: all v1-v6)")
    parser.add_argument("--ensemble", action="store_true",
                        help="Run ensemble sets (v7, v8) only")
    parser.add_argument("--all",      action="store_true",
                        help="Run all single-prompt AND ensemble sets")
    parser.add_argument("--log",      action="store_true",
                        help="Log best result to wandb")
    parser.add_argument("--log-all",  action="store_true",
                        help="Log every result to wandb")
    parser.add_argument("--csv",      action="store_true",
                        help="Save results table to CSV")
    parser.add_argument("--cpu",      action="store_true", help="Force CPU")
    parser.add_argument("--model",    default="ViT-B/32",
                        choices=["ViT-B/32", "ViT-L/14"],
                        help="CLIP model variant (ViT-L/14 is more accurate but slower)")
    args = parser.parse_args()

    device = get_device(args.cpu)
    print(f"Device: {device} | Model: {args.model}")

    model, preprocess = clip.load(args.model, device=device)
    model.eval()

    print("Loading test set...")
    test_pairs = load_test_set()
    print(f"Test set: {len(test_pairs)} images\n")

    # Decide which sets to run
    if args.all:
        single_keys  = list(PROMPT_SETS.keys())
        ensemble_keys = list(ENSEMBLE_SETS.keys())
    elif args.ensemble:
        single_keys, ensemble_keys = [], list(ENSEMBLE_SETS.keys())
    elif args.set:
        single_keys, ensemble_keys = [args.set], []
    else:
        single_keys, ensemble_keys = list(PROMPT_SETS.keys()), []

    results = {}

    for key in single_keys:
        ps = PROMPT_SETS[key]
        print(f"── {key}: {ps['description']}")
        res = evaluate(model, preprocess, ps["prompts"], test_pairs, device, agg="single")
        results[key] = res
        pc = res["per_class"]
        print(f"   Overall {res['overall']:.1f}%  |  "
              f"Soil {pc['soil']:.1f}%  Bedrock {pc['bedrock']:.1f}%  Sand {pc['sand']:.1f}%")

    for key in ensemble_keys:
        es = ENSEMBLE_SETS[key]
        print(f"── {key}: {es['description']}")
        res = evaluate(model, preprocess, es["prompts"], test_pairs, device, agg=es["agg"])
        results[key] = res
        pc = res["per_class"]
        print(f"   Overall {res['overall']:.1f}%  |  "
              f"Soil {pc['soil']:.1f}%  Bedrock {pc['bedrock']:.1f}%  Sand {pc['sand']:.1f}%")

    if len(results) > 1:
        best_key, best_res = print_comparison(results)
    else:
        best_key = list(results.keys())[0]
        best_res = results[best_key]

    print_confusion(best_key, best_res)

    if args.csv:
        save_csv(results, args.model)

    if args.log_all:
        for k, r in results.items():
            log_to_wandb(k, r, args.model)
    elif args.log:
        log_to_wandb(best_key, best_res, args.model)


if __name__ == "__main__":
    main()
