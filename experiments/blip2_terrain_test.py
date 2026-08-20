"""
Purpose: Test BLIP-2 (Salesforce/blip2-opt-2.7b) on AI4Mars terrain images using VQA.
         Evaluates scene description and terrain classification via natural language answers.
         Measures inference time and memory usage for RPi feasibility assessment.
Inputs:  AI4Mars gold-standard test set (322 images, local)
Outputs: Per-class accuracy, inference time, memory usage — logged to wandb, saved to CSV
How to run:
    python3 blip2_terrain_test.py --int4 --n 50 --questions q1 --log  # INT4 GPU (recommended, MX330 ~2GB)
    python3 blip2_terrain_test.py --int8 --n 50 --questions q1 --log  # INT8 GPU (needs >3GB VRAM)
    python3 blip2_terrain_test.py --cpu  --n 20 --questions q1        # float16 CPU (slow ~60s/img)
Project: Onboard Visual Foundation Models for Mars Terrain Perception and Rover Navigation
"""

import argparse
import csv
import os
import sys
import time

import numpy as np
import psutil
import torch
import wandb
from PIL import Image
from transformers import Blip2ForConditionalGeneration, Blip2Processor, BitsAndBytesConfig

# ── Dataset ───────────────────────────────────────────────────────────────────
AI4MARS_BASE = "/mnt/c/Users/DELL/Desktop/Thesis/github source/ai4mars-dataset-merged-0.1"
IMAGES_DIR   = os.path.join(AI4MARS_BASE, "msl/images/edr")
TEST_LABELS  = os.path.join(AI4MARS_BASE, "msl/labels/test/masked-gold-min3-100agree")

CLASS_NAMES         = ["soil", "bedrock", "sand", "big_rock"]
IGNORE_PIXEL        = 255
SUPERVISED_BASELINE = 96.67
CLIP_ZERO_SHOT      = 34.8
CLIP_PROMPT_ENG     = 54.4

# ── VQA questions to test ─────────────────────────────────────────────────────
# Question set: ask the same image multiple ways to find which framing works best
QUESTION_SETS = {
    "q1": {
        "description": "Direct terrain type question",
        "question": "Question: What type of terrain is visible? Answer with one word: soil, bedrock, sand, or rock.",
    },
    "q2": {
        "description": "Is it safe question",
        "question": "Question: Is this Mars terrain safe for a rover to drive on? Answer: safe or unsafe.",
    },
    "q3": {
        "description": "Surface description",
        "question": "Question: Describe the Mars surface texture in one word.",
    },
    "q4": {
        "description": "Open caption",
        "question": "Question: What is on the ground in this Mars image? Answer:",
    },
}

# Keyword mapping: BLIP-2 text answer → class index
KEYWORD_MAP = {
    "soil":    0, "dirt":    0, "regolith": 0, "dust": 0, "ground": 0, "loose": 0,
    "bedrock": 1, "rock":    1, "stone":    1, "flat": 1, "pavement": 1, "solid": 1,
    "sand":    2, "sandy":   2, "dune":     2, "bright": 2,
    "big":     3, "boulder": 3, "large":    3, "boulders": 3,
}


# ── Ground truth ──────────────────────────────────────────────────────────────

def dominant_class(label_path: str):
    label = np.array(Image.open(label_path))
    valid = label[label != IGNORE_PIXEL]
    if len(valid) == 0:
        return None
    return int(np.argmax(np.bincount(valid, minlength=4)))


def load_test_set(max_n: int = None):
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


# ── Answer parsing ────────────────────────────────────────────────────────────

def parse_answer(answer: str):
    """Map BLIP-2 text answer to a class index using keyword matching."""
    answer_lower = answer.lower().strip()
    # Direct class name match first
    for i, name in enumerate(CLASS_NAMES):
        if name in answer_lower:
            return i
    # Keyword fallback
    for word in answer_lower.split():
        if word in KEYWORD_MAP:
            return KEYWORD_MAP[word]
    return None   # unparseable


# ── Model loading ─────────────────────────────────────────────────────────────

def load_blip2(device: str, int8: bool = False, int4: bool = False):
    print("Loading BLIP-2 processor...")
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")

    if int4:
        dtype_label = "int4 (bitsandbytes NF4, GPU)"
    elif int8:
        dtype_label = "int8 (bitsandbytes, GPU)"
    else:
        dtype_label = f"float16 ({device})"
    print(f"Loading BLIP-2 model — {dtype_label} ...")
    t0 = time.perf_counter()

    if int4:
        # 4-bit NF4: LM quantised to ~1.35GB on GPU, vision encoder (~600MB) overflows to CPU
        # llm_int8_enable_fp32_cpu_offload allows mixed GPU/CPU dispatch for quantised models
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            llm_int8_enable_fp32_cpu_offload=True,
        )
        model = Blip2ForConditionalGeneration.from_pretrained(
            "Salesforce/blip2-opt-2.7b",
            quantization_config=bnb_config,
            device_map="auto",
            max_memory={0: "1.5GiB", "cpu": "6GiB"},
        )
    elif int8:
        # 8-bit: requires ~2.7GB VRAM; overflow to CPU if needed
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_enable_fp32_cpu_offload=True,
        )
        model = Blip2ForConditionalGeneration.from_pretrained(
            "Salesforce/blip2-opt-2.7b",
            quantization_config=bnb_config,
            device_map="auto",
            max_memory={0: "1.8GiB", "cpu": "6GiB"},
        )
    else:
        model = Blip2ForConditionalGeneration.from_pretrained(
            "Salesforce/blip2-opt-2.7b",
            torch_dtype=torch.float16,
            device_map={"": device},
        )

    model.eval()
    print(f"Model loaded in {time.perf_counter()-t0:.1f}s")
    return processor, model


# ── Evaluation ────────────────────────────────────────────────────────────────

def run_vqa(processor, model, question: str, image_path: str, device: str,
            max_new_tokens: int = 20, quantized: bool = False):
    """Run one BLIP-2 VQA inference. Return (answer_text, elapsed_ms, ram_mb)."""
    image = Image.open(image_path).convert("RGB")
    # Quantised models (INT4/INT8) manage dtype internally — do not cast inputs
    if quantized:
        inputs = processor(image, question, return_tensors="pt").to(device)
    else:
        inputs = processor(image, question, return_tensors="pt").to(device, torch.float16)

    proc = psutil.Process()
    ram_before = proc.memory_info().rss / 1024 / 1024

    t0 = time.perf_counter()
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=max_new_tokens)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    ram_after = proc.memory_info().rss / 1024 / 1024
    answer = processor.decode(output[0], skip_special_tokens=True).strip()
    return answer, elapsed_ms, ram_after


def evaluate_question_set(processor, model, qkey: str, question: str,
                           test_pairs: list, device: str, quantized: bool = False):
    """Evaluate a question set on all test images. Returns result dict."""
    n_classes = len(CLASS_NAMES)
    correct   = {c: 0 for c in range(n_classes)}
    total     = {c: 0 for c in range(n_classes)}
    times_ms  = []
    ram_mb    = []
    unparsed  = 0
    sample_answers = []   # store first 5 raw answers for inspection

    print(f"  Running {len(test_pairs)} images...")
    for i, (image_path, gt) in enumerate(test_pairs):
        answer, ms, ram = run_vqa(processor, model, question, image_path, device,
                                  quantized=quantized)
        pred = parse_answer(answer)

        if i < 5:
            sample_answers.append(
                f"GT={CLASS_NAMES[gt]} | pred={CLASS_NAMES[pred] if pred is not None else '?'} | raw='{answer}'"
            )

        if pred is None:
            unparsed += 1
        else:
            correct[gt] += int(pred == gt)
            total[gt]   += 1

        times_ms.append(ms)
        ram_mb.append(ram)

        if (i + 1) % 5 == 0:
            print(f"    {i+1}/{len(test_pairs)}  avg {np.mean(times_ms):.0f}ms/img", flush=True)

    per_class = {
        name: (correct[i] / total[i] * 100 if total[i] > 0 else None)
        for i, name in enumerate(CLASS_NAMES)
    }
    n_scored = sum(total.values())
    overall  = sum(correct.values()) / n_scored * 100 if n_scored > 0 else 0

    return {
        "overall":       overall,
        "per_class":     per_class,
        "avg_ms":        float(np.mean(times_ms)),
        "fps":           float(1000 / np.mean(times_ms)),
        "avg_ram_mb":    float(np.mean(ram_mb)),
        "unparsed":      unparsed,
        "n_scored":      n_scored,
        "sample_answers": sample_answers,
    }


# ── Reporting ─────────────────────────────────────────────────────────────────

def print_comparison(results: dict):
    keys  = list(results.keys())
    col_w = 14
    width = 18 + col_w * len(keys)

    print("\n" + "=" * width)
    print("BLIP-2 VQA RESULTS — AI4Mars terrain classification")
    print("=" * width)
    print(f"{'':18}" + "".join(f"  {k:>{col_w-2}}" for k in keys))
    print("-" * width)

    for name in CLASS_NAMES:
        row = f"{name:<18}"
        for k in keys:
            val = results[k]["per_class"].get(name)
            row += f"  {val:>{col_w-2}.1f}%" if val is not None else f"  {'N/A':>{col_w-2}}"
        print(row)

    print("-" * width)
    print(f"{'OVERALL':<18}" + "".join(f"  {results[k]['overall']:>{col_w-2}.1f}%" for k in keys))
    print(f"{'avg ms/image':<18}" + "".join(f"  {results[k]['avg_ms']:>{col_w-2}.0f}" for k in keys))
    print(f"{'FPS':<18}"          + "".join(f"  {results[k]['fps']:>{col_w-2}.3f}" for k in keys))
    print(f"{'RAM MB':<18}"       + "".join(f"  {results[k]['avg_ram_mb']:>{col_w-2}.0f}" for k in keys))
    print(f"{'unparsed':<18}"     + "".join(f"  {results[k]['unparsed']:>{col_w-2}}" for k in keys))
    print("=" * width)

    print(f"\nReference:")
    print(f"  CLIP zero-shot v1:    {CLIP_ZERO_SHOT:.1f}%")
    print(f"  CLIP prompt eng. v9:  {CLIP_PROMPT_ENG:.1f}%")
    print(f"  Supervised baseline:  {SUPERVISED_BASELINE:.1f}%")

    print("\nSample answers (first 5 images, best question set):")
    best_key = max(results, key=lambda k: results[k]["overall"])
    for ans in results[best_key]["sample_answers"]:
        print(f"  {ans}")


def save_csv(results: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = ["question_set", "overall", "gap"] + CLASS_NAMES + ["avg_ms", "fps", "avg_ram_mb", "unparsed"]
    rows = []
    for key, res in results.items():
        row = {
            "question_set": key,
            "overall":      f"{res['overall']:.2f}",
            "gap":          f"{res['overall'] - SUPERVISED_BASELINE:.2f}",
            "avg_ms":       f"{res['avg_ms']:.0f}",
            "fps":          f"{res['fps']:.4f}",
            "avg_ram_mb":   f"{res['avg_ram_mb']:.0f}",
            "unparsed":     res["unparsed"],
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


def log_to_wandb(results: dict, n_images: int, dtype_label: str = "float16"):
    for key, res in results.items():
        pc = res["per_class"]
        wandb.init(
            project="rover-autonomy-thesis",
            name=f"blip2-opt-2.7b-terrain-{dtype_label}-{key}",
            config={
                "model":        "Salesforce/blip2-opt-2.7b",
                "experiment":   "blip2_terrain_vqa",
                "question_set": key,
                "n_images":     n_images,
                "dtype":        dtype_label,
            },
        )
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
            "inference/avg_ram_mb":         res["avg_ram_mb"],
            "inference/unparsed":           res["unparsed"],
        })
        wandb.finish()
    print("Logged to wandb → rover-autonomy-thesis")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="BLIP-2 VQA terrain classification on AI4Mars")
    parser.add_argument("--n",       type=int, default=None,
                        help="Max images to evaluate (default: all ~287)")
    parser.add_argument("--questions", nargs="+", choices=list(QUESTION_SETS.keys()),
                        default=list(QUESTION_SETS.keys()),
                        help="Which question sets to run (default: all)")
    parser.add_argument("--log",     action="store_true", help="Log results to wandb")
    parser.add_argument("--csv",     action="store_true", help="Save results to CSV")
    parser.add_argument("--cpu",     action="store_true", help="Force CPU (float16)")
    parser.add_argument("--int8",    action="store_true", help="INT8 bitsandbytes (needs >3GB VRAM)")
    parser.add_argument("--int4",    action="store_true", help="INT4 NF4 bitsandbytes (fits MX330 2GB VRAM)")
    args = parser.parse_args()

    quantized = args.int4 or args.int8
    if args.int4:
        dtype_label = "int4"
        device = "cuda" if torch.cuda.is_available() else "cpu"
    elif args.int8:
        dtype_label = "int8"
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        dtype_label = "float16"
        device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}  |  dtype: {dtype_label}")
    processor, model = load_blip2(device, int8=args.int8, int4=args.int4)

    print("\nLoading test set...")
    test_pairs = load_test_set(args.n)
    print(f"Test set: {len(test_pairs)} images\n")

    csv_path = os.path.join(os.path.dirname(__file__), "results",
                            f"blip2_terrain_vqa_{dtype_label}.csv")

    results = {}
    for qkey in args.questions:
        qs = QUESTION_SETS[qkey]
        print(f"── {qkey}: {qs['description']}", flush=True)
        print(f"   Q: \"{qs['question']}\"", flush=True)
        res = evaluate_question_set(
            processor, model, qkey, qs["question"], test_pairs, device,
            quantized=quantized)
        results[qkey] = res
        pc = res["per_class"]
        print(f"   Overall {res['overall']:.1f}%  |  "
              f"Soil {pc.get('soil', 0) or 0:.1f}%  "
              f"Bedrock {pc.get('bedrock', 0) or 0:.1f}%  "
              f"Sand {pc.get('sand', 0) or 0:.1f}%  "
              f"| {res['avg_ms']:.0f}ms/img  unparsed={res['unparsed']}\n", flush=True)
        # Save after every question set — protects against OOM mid-run
        save_csv(results, csv_path)

    print_comparison(results)

    if args.log:
        log_to_wandb(results, len(test_pairs), dtype_label=dtype_label)


if __name__ == "__main__":
    main()
