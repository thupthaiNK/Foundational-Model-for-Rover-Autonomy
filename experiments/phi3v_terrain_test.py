"""
Purpose: Test Phi-3-V (microsoft/Phi-3-vision-128k-instruct) INT4 on AI4Mars terrain images.
         Compares against BLIP-2 on accuracy, inference time, and memory usage.
         Validates Phi-3-V as a lighter alternative (~2GB INT4 vs BLIP-2 ~5GB float16).
Inputs:  AI4Mars gold-standard test set (322 images, local)
Outputs: Per-class accuracy, inference time, memory — logged to wandb, saved to CSV
How to run:
    python3 phi3v_terrain_test.py --cpu --n 20        # quick test: 20 images
    python3 phi3v_terrain_test.py --cpu --n 100       # medium test
    python3 phi3v_terrain_test.py --cpu               # full test: all 287 images
    python3 phi3v_terrain_test.py --cpu --log         # + log to wandb
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
from transformers import AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig

# ── Dataset ───────────────────────────────────────────────────────────────────
AI4MARS_BASE = "/mnt/c/Users/DELL/Desktop/Thesis/github source/ai4mars-dataset-merged-0.1"
IMAGES_DIR   = os.path.join(AI4MARS_BASE, "msl/images/edr")
TEST_LABELS  = os.path.join(AI4MARS_BASE, "msl/labels/test/masked-gold-min3-100agree")

CLASS_NAMES         = ["soil", "bedrock", "sand", "big_rock"]
IGNORE_PIXEL        = 255
SUPERVISED_BASELINE = 96.67
CLIP_ZERO_SHOT      = 34.8
CLIP_PROMPT_ENG     = 54.4
FEWSHOT_1000        = 87.8

MODEL_ID = "microsoft/Phi-3-vision-128k-instruct"

# ── Questions ─────────────────────────────────────────────────────────────────
QUESTION_SETS = {
    "q1": {
        "description": "Direct terrain type (single word)",
        "prompt": "<|image_1|>\nWhat type of terrain is visible in this Mars image? "
                  "Answer with exactly one word: soil, bedrock, sand, or rock.",
    },
    "q2": {
        "description": "Safe/unsafe traversability",
        "prompt": "<|image_1|>\nIs this Mars terrain safe for a rover to drive on? "
                  "Answer: safe or unsafe.",
    },
    "q4": {
        "description": "Open description",
        "prompt": "<|image_1|>\nDescribe the ground surface visible in this Mars image in one sentence.",
    },
}

KEYWORD_MAP = {
    "soil":    0, "dirt":     0, "regolith": 0, "dust":  0, "ground": 0,
    "bedrock": 1, "rock":     1, "stone":    1, "flat":  1, "pavement": 1,
    "sand":    2, "sandy":    2, "dune":     2, "bright": 2,
    "big":     3, "boulder":  3, "boulders": 3, "large": 3,
}


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


# ── Answer parsing ────────────────────────────────────────────────────────────

def parse_answer(answer: str):
    answer_lower = answer.lower().strip()
    for i, name in enumerate(CLASS_NAMES):
        if name in answer_lower:
            return i
    for word in answer_lower.split():
        word = word.strip(".,!?")
        if word in KEYWORD_MAP:
            return KEYWORD_MAP[word]
    return None


# ── Model loading ─────────────────────────────────────────────────────────────

def load_phi3v(device: str, use_int4: bool = True):
    # Try local phi3v cache first; fall back to HuggingFace ID
    local_path = "/home/thupthai/.cache/huggingface/hub/phi3v"
    model_source = local_path if os.path.isdir(local_path) else MODEL_ID
    print(f"Loading Phi-3-V processor from {model_source}...")
    processor = AutoProcessor.from_pretrained(
        model_source,
        trust_remote_code=True,
        num_crops=4,
    )

    quant_str = "INT4 NF4" if use_int4 else "float16"
    print(f"Loading Phi-3-V model ({quant_str}) on {device}...")
    t0 = time.perf_counter()

    if use_int4:
        # INT4 NF4: ~2GB total — fits in MX330 2GB VRAM (tight) or CPU RAM
        # llm_int8_enable_fp32_cpu_offload allows GPU/CPU split if VRAM overflows
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            llm_int8_enable_fp32_cpu_offload=True,
        )
        cuda_ok = torch.cuda.is_available()
        if cuda_ok:
            # Try GPU-first with CPU overflow safety valve
            max_mem = {0: "1.8GiB", "cpu": "5GiB"}
        else:
            max_mem = {"cpu": "6GiB"}
        model = AutoModelForCausalLM.from_pretrained(
            model_source,
            quantization_config=quantization_config,
            device_map="auto",
            max_memory=max_mem,
            trust_remote_code=True,
            _attn_implementation="eager",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_source,
            torch_dtype=torch.float16,
            device_map={"": device},
            trust_remote_code=True,
            _attn_implementation="eager",
        )

    model.eval()
    load_time = time.perf_counter() - t0
    ram_mb = psutil.Process().memory_info().rss / 1024 / 1024
    print(f"Model loaded in {load_time:.1f}s  |  RAM: {ram_mb:.0f} MB")
    return processor, model, ram_mb


# ── Evaluation ────────────────────────────────────────────────────────────────

def run_inference(processor, model, prompt: str, image_path: str, device: str):
    image = Image.open(image_path).convert("RGB")
    messages = [{"role": "user", "content": prompt}]

    text = processor.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(text=text, images=[image], return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    proc = psutil.Process()
    t0 = time.perf_counter()
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=30,
            do_sample=False,
            temperature=None,
            top_p=None,
        )
    elapsed_ms = (time.perf_counter() - t0) * 1000
    ram_mb     = proc.memory_info().rss / 1024 / 1024

    # Decode only the newly generated tokens
    input_len  = inputs["input_ids"].shape[1]
    answer     = processor.tokenizer.decode(
        output_ids[0][input_len:], skip_special_tokens=True
    ).strip()
    return answer, elapsed_ms, ram_mb


def evaluate_question_set(processor, model, qkey, prompt, test_pairs, device):
    n_classes = len(CLASS_NAMES)
    correct   = {c: 0 for c in range(n_classes)}
    total     = {c: 0 for c in range(n_classes)}
    times_ms, ram_mb = [], []
    unparsed  = 0
    sample_answers = []

    print(f"  Running {len(test_pairs)} images...")
    for i, (image_path, gt) in enumerate(test_pairs):
        answer, ms, ram = run_inference(processor, model, prompt, image_path, device)
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
    n_scored  = sum(total.values())
    overall   = sum(correct.values()) / n_scored * 100 if n_scored > 0 else 0

    return {
        "overall":        overall,
        "per_class":      per_class,
        "avg_ms":         float(np.mean(times_ms)),
        "fps":            float(1000 / np.mean(times_ms)),
        "avg_ram_mb":     float(np.mean(ram_mb)),
        "unparsed":       unparsed,
        "n_scored":       n_scored,
        "sample_answers": sample_answers,
    }


# ── Reporting ─────────────────────────────────────────────────────────────────

def print_results(results: dict, model_load_ram_mb: float):
    keys  = list(results.keys())
    col_w = 14
    width = 18 + col_w * len(keys)

    print("\n" + "=" * width)
    print("Phi-3-V INT4 VQA RESULTS — AI4Mars terrain classification")
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
    print(f"{'model load RAM':<18}  {model_load_ram_mb:>{col_w-2}.0f} MB")
    print(f"{'unparsed':<18}"     + "".join(f"  {results[k]['unparsed']:>{col_w-2}}" for k in keys))
    print("=" * width)

    print(f"\nReference (overall accuracy):")
    print(f"  CLIP zero-shot v1:         {CLIP_ZERO_SHOT:.1f}%")
    print(f"  CLIP prompt eng. v9:       {CLIP_PROMPT_ENG:.1f}%")
    print(f"  CLIP 1000-shot linear:     {FEWSHOT_1000:.1f}%")
    print(f"  Supervised (DeepLabv3+):   {SUPERVISED_BASELINE:.1f}%")

    best_key = max(results, key=lambda k: results[k]["overall"])
    print(f"\nSample answers — {best_key}:")
    for ans in results[best_key]["sample_answers"]:
        print(f"  {ans}")


def save_csv(results: dict, model_load_ram_mb: float, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = ["question_set", "overall", "gap"] + CLASS_NAMES + \
                 ["avg_ms", "fps", "avg_ram_mb", "model_load_ram_mb", "unparsed"]
    rows = []
    for key, res in results.items():
        row = {
            "question_set":      key,
            "overall":           f"{res['overall']:.2f}",
            "gap":               f"{res['overall'] - SUPERVISED_BASELINE:.2f}",
            "avg_ms":            f"{res['avg_ms']:.0f}",
            "fps":               f"{res['fps']:.4f}",
            "avg_ram_mb":        f"{res['avg_ram_mb']:.0f}",
            "model_load_ram_mb": f"{model_load_ram_mb:.0f}",
            "unparsed":          res["unparsed"],
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


def log_to_wandb(results: dict, model_load_ram_mb: float, n_images: int):
    for key, res in results.items():
        pc = res["per_class"]
        wandb.init(
            project="rover-autonomy-thesis",
            name=f"phi3v-int4-terrain-{key}",
            config={
                "model":             MODEL_ID,
                "quantization":      "INT4 (nf4)",
                "experiment":        "phi3v_terrain_vqa",
                "question_set":      key,
                "n_images":          n_images,
                "model_load_ram_mb": model_load_ram_mb,
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
    parser = argparse.ArgumentParser(description="Phi-3-V INT4 terrain VQA on AI4Mars")
    parser.add_argument("--n",         type=int, default=None,
                        help="Max images to evaluate (default: all ~287)")
    parser.add_argument("--questions", nargs="+", choices=list(QUESTION_SETS.keys()),
                        default=list(QUESTION_SETS.keys()),
                        help="Question sets to run (default: all)")
    parser.add_argument("--no-int4",   action="store_true",
                        help="Use float16 instead of INT4 (requires more RAM)")
    parser.add_argument("--log",       action="store_true", help="Log results to wandb")
    parser.add_argument("--cpu",       action="store_true", help="Force CPU")
    args = parser.parse_args()

    # INT4 requires bitsandbytes + working CUDA GPU.
    # Test CUDA actually works (not just available) to catch MX330-class GPUs
    # where torch.cuda.is_available() is True but cuDNN/bitsandbytes fail.
    if args.cpu:
        cuda_ok = False
    elif torch.cuda.is_available():
        try:
            torch.zeros(1).cuda()
            cuda_ok = True
        except Exception:
            cuda_ok = False
    else:
        cuda_ok = False

    device   = "cuda" if cuda_ok else "cpu"
    use_int4 = (not args.no_int4) and cuda_ok
    if (not args.no_int4) and not cuda_ok:
        print("NOTE: INT4 requires working CUDA GPU. Falling back to float16 CPU.")
    mode_str = "INT4 NF4 (GPU→CPU auto)" if use_int4 else f"float16 ({device})"
    print(f"Mode: {mode_str}")

    processor, model, load_ram_mb = load_phi3v(device, use_int4)

    print("\nLoading test set...")
    test_pairs = load_test_set(args.n)
    print(f"Test set: {len(test_pairs)} images\n")

    csv_path = os.path.join(os.path.dirname(__file__), "results", "phi3v_terrain_vqa.csv")

    results = {}
    for qkey in args.questions:
        qs = QUESTION_SETS[qkey]
        print(f"── {qkey}: {qs['description']}", flush=True)
        res = evaluate_question_set(
            processor, model, qkey, qs["prompt"], test_pairs, device)
        results[qkey] = res
        pc = res["per_class"]
        print(f"   Overall {res['overall']:.1f}%  |  "
              f"Soil {pc.get('soil') or 0:.1f}%  "
              f"Bedrock {pc.get('bedrock') or 0:.1f}%  "
              f"Sand {pc.get('sand') or 0:.1f}%  "
              f"| {res['avg_ms']:.0f}ms/img  unparsed={res['unparsed']}\n", flush=True)
        # Save after every question set — protects against OOM mid-run
        save_csv(results, load_ram_mb, csv_path)

    print_results(results, load_ram_mb)

    if args.log:
        log_to_wandb(results, load_ram_mb, len(test_pairs))


if __name__ == "__main__":
    main()
