"""
Purpose:    Test SmolVLM (HuggingFaceTB/SmolVLM-Instruct, 256M params) on AI4Mars terrain images.
            Compares accuracy, inference time, and memory vs BLIP-2 and CLIP.
            SmolVLM is edge-native (256M params) — key comparison point for RPi deployment.
Inputs:     AI4Mars gold-standard test set (322 images, local)
Outputs:    Per-class accuracy, inference time, RAM — saved to experiments/results/smolvlm_terrain_test.csv
How to run:
    python3 experiments/smolvlm_terrain_test.py                # full test (all ~287 images)
    python3 experiments/smolvlm_terrain_test.py --n 20         # quick test: 20 images
    python3 experiments/smolvlm_terrain_test.py --n 5          # smoke test: 5 images
    python3 experiments/smolvlm_terrain_test.py --model 500m   # use 500M variant instead
Project: Onboard Visual Foundation Models for Mars Terrain Perception and Rover Navigation
"""

import argparse
import csv
import os
import time

import numpy as np
import psutil
import torch
from PIL import Image
from transformers import AutoProcessor, SmolVLMForConditionalGeneration

# ── Paths ─────────────────────────────────────────────────────────────────────
AI4MARS_BASE = "/mnt/c/Users/DELL/Desktop/Thesis/github source/ai4mars-dataset-merged-0.1"
IMAGES_DIR   = os.path.join(AI4MARS_BASE, "msl/images/edr")
TEST_LABELS  = os.path.join(AI4MARS_BASE, "msl/labels/test/masked-gold-min3-100agree")
RESULTS_DIR  = os.path.join(os.path.dirname(__file__), "results")

CLASS_NAMES         = ["soil", "bedrock", "sand", "big_rock"]
IGNORE_PIXEL        = 255
SUPERVISED_BASELINE = 96.67

# ── Model variants ─────────────────────────────────────────────────────────────
MODEL_IDS = {
    "256m": "HuggingFaceTB/SmolVLM-256M-Instruct",
    "500m": "HuggingFaceTB/SmolVLM-500M-Instruct",
    "2b":   "HuggingFaceTB/SmolVLM-Instruct",        # 2B — larger, not 256M
}

# ── Questions ─────────────────────────────────────────────────────────────────
QUESTION_SETS = {
    # q1: original prompt (contains "Mars" — causes model to echo "Mars." → 21.6% parse failure)
    "q1": "What type of terrain is visible in this Mars image? Answer with exactly one word: soil, bedrock, sand, or rock.",
    # q1b: improved prompt — removes "Mars" keyword to prevent model echoing it back
    "q1b": "Classify the ground surface in this image. Reply with exactly one word from this list: soil, bedrock, sand, rock.",
    "q2": "Is this Mars terrain safe for a rover to drive on? Answer: safe or unsafe.",
}

KEYWORD_MAP = {
    "soil":    0, "dirt":    0, "regolith": 0, "dust":    0, "ground":   0,
    "bedrock": 1, "rock":    1, "stone":    1, "flat":    1, "pavement": 1,
    "sand":    2, "sandy":   2, "dune":     2, "bright":  2,
    "big":     3, "boulder": 3, "large":    3,
}


# ── Ground truth ───────────────────────────────────────────────────────────────

def dominant_class(label_path):
    label = np.array(Image.open(label_path))
    valid = label[label != IGNORE_PIXEL]
    return int(np.argmax(np.bincount(valid, minlength=4))) if len(valid) > 0 else None


def load_test_set(max_n=None):
    pairs = []
    for fname in sorted(os.listdir(TEST_LABELS)):
        if not fname.endswith("_merged.png"):
            continue
        stem = fname.replace("_merged.png", "")
        img_path = os.path.join(IMAGES_DIR, stem + ".JPG")
        lbl_path = os.path.join(TEST_LABELS, fname)
        if not os.path.exists(img_path):
            continue
        gt = dominant_class(lbl_path)
        if gt is not None:
            pairs.append((img_path, gt))
        if max_n and len(pairs) >= max_n:
            break
    return pairs


# ── Answer parsing ─────────────────────────────────────────────────────────────

def parse_answer(answer):
    low = answer.lower().strip()
    # Direct class name anywhere in response (handles "The terrain is soil.")
    for i, name in enumerate(CLASS_NAMES):
        if name in low:
            return i
    # Word-by-word keyword match
    for word in low.split():
        word = word.strip(".,!?;:()")
        if word in KEYWORD_MAP:
            return KEYWORD_MAP[word]
    # Partial keyword match for compound words ("rocky" → bedrock)
    extended = {
        "rocky": 1, "rocks": 1, "rocky terrain": 1, "exposed": 1,
        "loose": 0, "sediment": 0, "granular": 0,
        "sandy": 2, "dunes": 2, "ripples": 2,
        "boulders": 3, "large rock": 3,
    }
    for phrase, cls in extended.items():
        if phrase in low:
            return cls
    return None


# ── Model ─────────────────────────────────────────────────────────────────────

def load_model(model_id, device):
    print(f"Loading SmolVLM processor ({model_id})...", flush=True)
    processor = AutoProcessor.from_pretrained(model_id)

    print(f"Loading SmolVLM model...", flush=True)
    t0 = time.perf_counter()
    model = SmolVLMForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        device_map={"": device},
    )
    model.eval()
    load_time = time.perf_counter() - t0
    ram_mb = psutil.Process().memory_info().rss / 1e6
    print(f"Loaded in {load_time:.1f}s  |  RAM: {ram_mb:.0f} MB", flush=True)
    return processor, model, load_time, ram_mb


def run_inference(processor, model, question, image_path, device):
    image = Image.open(image_path).convert("RGB")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": question},
            ],
        }
    ]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt, images=[image], return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    t0 = time.perf_counter()
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=20, do_sample=False)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    ram_mb = psutil.Process().memory_info().rss / 1e6

    input_len = inputs["input_ids"].shape[1]
    answer = processor.tokenizer.decode(
        output_ids[0][input_len:], skip_special_tokens=True
    ).strip()
    return answer, elapsed_ms, ram_mb


# ── Evaluation ─────────────────────────────────────────────────────────────────

def evaluate(processor, model, question_key, question, test_pairs, device):
    correct = {c: 0 for c in range(4)}
    total   = {c: 0 for c in range(4)}
    times_ms, rams, unparsed = [], [], 0
    sample_answers = []

    for i, (img_path, gt) in enumerate(test_pairs):
        answer, ms, ram = run_inference(processor, model, question, img_path, device)
        pred = parse_answer(answer)

        if i < 5:
            sample_answers.append(
                f"  GT={CLASS_NAMES[gt]:8s} | pred={CLASS_NAMES[pred] if pred is not None else '?':8s} | '{answer}'"
            )

        if pred is None:
            unparsed += 1
        else:
            correct[gt] += int(pred == gt)
            total[gt]   += 1

        times_ms.append(ms)
        rams.append(ram)

        if (i + 1) % 5 == 0 or (i + 1) == len(test_pairs):
            overall_so_far = sum(correct.values()) / max(sum(total.values()), 1) * 100
            print(f"  [{i+1:3d}/{len(test_pairs)}]  avg {np.mean(times_ms):.0f}ms/img  "
                  f"running acc {overall_so_far:.1f}%", flush=True)

    n_scored = sum(total.values())
    overall  = sum(correct.values()) / n_scored * 100 if n_scored > 0 else 0
    per_class = {
        name: (correct[i] / total[i] * 100 if total[i] > 0 else None)
        for i, name in enumerate(CLASS_NAMES)
    }
    return {
        "overall":      overall,
        "per_class":    per_class,
        "avg_ms":       float(np.mean(times_ms)),
        "fps":          float(1000 / np.mean(times_ms)),
        "avg_ram_mb":   float(np.mean(rams)),
        "unparsed":     unparsed,
        "n_scored":     n_scored,
        "sample_answers": sample_answers,
    }


# ── Reporting ─────────────────────────────────────────────────────────────────

def print_results(results, model_id, load_time, load_ram):
    print(f"\n{'='*70}")
    print(f"SmolVLM RESULTS — {model_id}")
    print(f"{'='*70}")
    header = f"{'Class':<12}" + "".join(f"  {k:>8}" for k in results)
    print(header)
    print("-" * len(header))
    for name in CLASS_NAMES:
        row = f"{name:<12}"
        for res in results.values():
            val = res["per_class"].get(name)
            row += f"  {val:>7.1f}%" if val is not None else f"  {'N/A':>8}"
        print(row)
    print("-" * len(header))
    print(f"{'OVERALL':<12}" + "".join(f"  {r['overall']:>7.1f}%" for r in results.values()))
    print(f"{'avg ms/img':<12}" + "".join(f"  {r['avg_ms']:>7.0f} " for r in results.values()))
    print(f"{'FPS':<12}"       + "".join(f"  {r['fps']:>7.3f} " for r in results.values()))
    print(f"{'RAM MB':<12}"    + "".join(f"  {r['avg_ram_mb']:>7.0f} " for r in results.values()))
    print(f"{'='*70}")
    print(f"\nModel load: {load_time:.1f}s  |  RAM at load: {load_ram:.0f} MB")
    print(f"\nReference baselines (Overall):")
    print(f"  CLIP zero-shot v1:         34.8%  @ 154ms/img")
    print(f"  CLIP prompt eng. v9:       54.4%  @  96ms/img")
    print(f"  CLIP 1000-shot linear:     87.8%  @ ~100ms/img")
    print(f"  BLIP-2 float16 CPU:        80.0%* @ 753,000ms/img")
    print(f"  Supervised (DeepLabv3+):   96.7%  (GPU)")
    print(f"  *BLIP-2 soil-dominant 20-img sample, bedrock 0%")
    print(f"\nSample answers (q1):")
    for line in results.get("q1", {}).get("sample_answers", []):
        print(line)


def save_csv(results, model_id, load_time, load_ram, n_images):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    if "256" in model_id:
        variant = "256m"
    elif "500" in model_id:
        variant = "500m"
    else:
        variant = "2b"
    path = os.path.join(RESULTS_DIR, f"smolvlm_{variant}_terrain_test.csv")
    fieldnames = ["question_set", "model_id", "overall", "gap"] + CLASS_NAMES + \
                 ["avg_ms", "fps", "avg_ram_mb", "load_time_s", "load_ram_mb",
                  "n_scored", "unparsed", "n_images"]
    rows = []
    for qkey, res in results.items():
        row = {
            "question_set": qkey,
            "model_id":     model_id,
            "overall":      f"{res['overall']:.2f}",
            "gap":          f"{res['overall'] - SUPERVISED_BASELINE:.2f}",
            "avg_ms":       f"{res['avg_ms']:.0f}",
            "fps":          f"{res['fps']:.4f}",
            "avg_ram_mb":   f"{res['avg_ram_mb']:.0f}",
            "load_time_s":  f"{load_time:.1f}",
            "load_ram_mb":  f"{load_ram:.0f}",
            "n_scored":     res["n_scored"],
            "unparsed":     res["unparsed"],
            "n_images":     n_images,
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
    return path


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SmolVLM terrain VQA on AI4Mars")
    parser.add_argument("--n",      type=int, default=None,
                        help="Max images to evaluate (default: all ~287)")
    parser.add_argument("--model",  choices=["256m", "500m", "2b"], default="256m",
                        help="SmolVLM variant (default: 256m = SmolVLM-256M-Instruct)")
    parser.add_argument("--questions", nargs="+", choices=list(QUESTION_SETS.keys()), default=["q1"],
                        help="Question sets to run (default: q1)")
    parser.add_argument("--cpu",    action="store_true", help="Force CPU (default: auto)")
    args = parser.parse_args()

    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    model_id = MODEL_IDS[args.model]

    print(f"\n{'='*60}")
    print(f"SmolVLM Terrain Test — AI4Mars")
    print(f"Model:  {model_id}")
    print(f"Device: {device}")
    print(f"{'='*60}\n")

    processor, model, load_time, load_ram = load_model(model_id, device)

    test_pairs = load_test_set(args.n)
    print(f"\nTest set: {len(test_pairs)} images\n")

    results = {}
    for qkey in args.questions:
        question = QUESTION_SETS[qkey]
        print(f"── {qkey}: {question[:60]}...", flush=True)
        res = evaluate(processor, model, qkey, question, test_pairs, device)
        results[qkey] = res

        pc = res["per_class"]
        print(f"\n   Overall {res['overall']:.1f}% | "
              f"Soil {(pc.get('soil') or 0):.1f}% | "
              f"Bedrock {(pc.get('bedrock') or 0):.1f}% | "
              f"Sand {(pc.get('sand') or 0):.1f}% | "
              f"{res['avg_ms']:.0f}ms/img | unparsed={res['unparsed']}\n", flush=True)

        # Save after every question set (OOM protection)
        save_csv(results, model_id, load_time, load_ram, len(test_pairs))

    print_results(results, model_id, load_time, load_ram)


if __name__ == "__main__":
    main()
