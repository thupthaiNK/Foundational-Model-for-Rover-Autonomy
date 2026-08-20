"""
Purpose: Test SmolVLM2-256M on AI4Mars terrain images as drop-in upgrade to SmolVLM-256M.
         Evaluates whether the newer model improves overall accuracy (>49.78%) and
         reduces parse failure rate (<21.6%) on the same 287-image gold-standard test set.
Inputs:  AI4Mars gold-standard test set (287 images, local)
Outputs: Per-class accuracy, inference time, parse failure rate — CSV saved to results/
How to run:
    python3 -u experiments/smolvlm2_terrain_test.py          # full 287-image eval
    python3 -u experiments/smolvlm2_terrain_test.py --n 20   # quick 20-image smoke test
    python3 -u experiments/smolvlm2_terrain_test.py --n 5    # 5-image sanity check
Reference: Marafioti et al. (2025) SmolVLM: Redefining small and efficient multimodal models.
           arXiv:2504.05299  https://huggingface.co/HuggingFaceTB/SmolVLM2-256M-Video-Instruct
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
from transformers import AutoModelForImageTextToText, AutoProcessor

# ── Paths ─────────────────────────────────────────────────────────────────────
AI4MARS_BASE = "/mnt/c/Users/DELL/Desktop/Thesis/github source/ai4mars-dataset-merged-0.1"
IMAGES_DIR   = os.path.join(AI4MARS_BASE, "msl/images/edr")
TEST_LABELS  = os.path.join(AI4MARS_BASE, "msl/labels/test/masked-gold-min3-100agree")
RESULTS_DIR  = os.path.join(os.path.dirname(__file__), "results")

MODEL_ID            = "HuggingFaceTB/SmolVLM2-256M-Video-Instruct"
CLASS_NAMES         = ["soil", "bedrock", "sand", "big_rock"]
IGNORE_PIXEL        = 255
SUPERVISED_BASELINE = 96.67

# ── SmolVLM-256M baseline (from experiments done 2026-06-03, n=287) ───────────
SMOLVLM_BASELINE = {
    "overall": 49.78, "soil": 67.3, "bedrock": 21.9, "sand": 50.0,
    "avg_ms": 60826, "parse_failure_pct": 21.6,
}

# SmolVLM2 parrots option lists back verbatim when given a closed-choice prompt.
# Solution: open-ended, single-word prompt — rely on keyword_map + extended_map to parse.
# Keeps "Mars" keyword (removing it collapses soil accuracy to ~0% in SmolVLM ablation).
# max_new_tokens=8 caps length so the model cannot generate multi-word option lists.
QUESTION = "Describe the ground surface in this Mars rover image in one word."

KEYWORD_MAP = {
    # specific terrain words only — generic words like "ground", "flat", "big" removed
    # to prevent early false matches in sentences like "The ground surface is rocky."
    "soil":    0, "dirt":    0, "regolith": 0, "dust":    0, "dusty":   0,
    "bedrock": 1, "rock":    1, "stone":    1,
    "sand":    2, "sandy":   2, "dune":     2, "desert":  2,
    "boulder": 3, "gravel":  3, "pebble":   3,
}
EXTENDED_MAP = {
    "rocky":    1, "rocks":   1, "exposed":  1, "stony":   1, "cracked": 1,
    "sandy":    2, "dunes":   2, "ripples":  2,
    "sediment": 0, "dusty":   0, "loose":    0,
    "boulders": 3, "pebbles": 3, "gravelly": 3,
}


# ── Ground truth ───────────────────────────────────────────────────────────────

def dominant_class(label_path: str):
    label = np.array(Image.open(label_path))
    valid = label[label != IGNORE_PIXEL]
    return int(np.argmax(np.bincount(valid, minlength=4))) if len(valid) > 0 else None


def load_test_set(max_n=None):
    pairs = []
    for fname in sorted(os.listdir(TEST_LABELS)):
        if not fname.endswith("_merged.png"):
            continue
        stem     = fname.replace("_merged.png", "")
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

def parse_answer(answer: str):
    low = answer.lower().strip()
    # 1. Exact class name anywhere in response
    for i, name in enumerate(CLASS_NAMES):
        if name in low:
            return i
    # 2. Scan words last→first: model puts terrain descriptor at end of sentence
    #    e.g. "The ground surface is rocky." → last word "rocky" → bedrock
    words = [w.strip(".,!?;:()") for w in low.split()]
    for word in reversed(words):
        if word in KEYWORD_MAP:
            return KEYWORD_MAP[word]
    # 3. Extended phrases anywhere
    for phrase, cls in EXTENDED_MAP.items():
        if phrase in low:
            return cls
    return None


# ── Model ─────────────────────────────────────────────────────────────────────

def load_model(device: str):
    print(f"Loading SmolVLM2-256M processor...", flush=True)
    processor = AutoProcessor.from_pretrained(MODEL_ID)

    print(f"Loading SmolVLM2-256M model...", flush=True)
    t0 = time.perf_counter()
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        dtype=torch.float32,
        device_map={"": device},
    )
    model.eval()
    load_s = time.perf_counter() - t0
    ram_mb = psutil.Process().memory_info().rss / 1e6
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Loaded in {load_s:.1f}s  |  Params: {n_params:.1f}M  "
          f"|  RAM: {ram_mb:.0f}MB  |  Device: {device}", flush=True)
    return processor, model


# ── Inference ─────────────────────────────────────────────────────────────────

def run_inference(processor, model, image_path: str, device: str):
    image    = Image.open(image_path).convert("RGB")
    messages = [{"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": QUESTION},
    ]}]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt, images=[image], return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    t0 = time.perf_counter()
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=8, do_sample=False)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    ram_mb     = psutil.Process().memory_info().rss / 1e6

    input_len = inputs["input_ids"].shape[1]
    answer    = processor.tokenizer.decode(
        output_ids[0][input_len:], skip_special_tokens=True
    ).strip()
    return answer, elapsed_ms, ram_mb


# ── Evaluation ─────────────────────────────────────────────────────────────────

def evaluate(processor, model, test_pairs, device: str):
    correct  = {c: 0 for c in range(4)}
    total    = {c: 0 for c in range(4)}
    times_ms = []
    unparsed = 0
    rows     = []

    print(f"\n{'Image':<45} {'GT':<10} {'Pred':<10} {'ms':>7}  Answer")
    print("=" * 100)

    for i, (img_path, gt) in enumerate(test_pairs):
        stem   = os.path.basename(img_path).replace(".JPG", "")
        answer, ms, ram = run_inference(processor, model, img_path, device)
        pred   = parse_answer(answer)

        times_ms.append(ms)
        gt_name   = CLASS_NAMES[gt]
        pred_name = CLASS_NAMES[pred] if pred is not None else "?"
        marker    = "✓" if (pred == gt) else ("?" if pred is None else "✗")

        print(f"{stem[:44]:<45} {gt_name:<10} {pred_name:<10} {ms:>6.0f}  '{answer[:30]}' {marker}",
              flush=True)

        if pred is None:
            unparsed += 1
        else:
            correct[gt] += int(pred == gt)
            total[gt]   += 1

        rows.append({
            "image": stem, "gt": gt_name, "pred": pred_name,
            "ms": f"{ms:.0f}", "answer": answer[:60], "correct": (pred == gt) if pred is not None else "parse_fail",
        })

        if (i + 1) % 10 == 0:
            scored = sum(total.values())
            acc_so_far = sum(correct.values()) / scored * 100 if scored > 0 else 0
            print(f"  ── [{i+1}/{len(test_pairs)}]  running acc: {acc_so_far:.1f}%  "
                  f"parse_fail: {unparsed}/{i+1}  avg: {np.mean(times_ms):.0f}ms ──",
                  flush=True)

    n_scored = sum(total.values())
    overall  = sum(correct.values()) / n_scored * 100 if n_scored > 0 else 0
    per_class = {
        name: (correct[i] / total[i] * 100 if total[i] > 0 else None)
        for i, name in enumerate(CLASS_NAMES)
    }
    return overall, per_class, times_ms, unparsed, rows


# ── Reporting ─────────────────────────────────────────────────────────────────

def print_results(overall, per_class, times_ms, unparsed, n_total):
    avg_ms = np.mean(times_ms)
    print(f"\n{'='*65}")
    print(f"{'Class':<12} {'SmolVLM2':>10} {'SmolVLM (old)':>14}  {'delta':>8}")
    print("-" * 65)
    for name in CLASS_NAMES:
        val  = per_class.get(name)
        base = SMOLVLM_BASELINE.get(name, 0)
        if val is not None:
            diff = val - base
            print(f"{name:<12} {val:>9.1f}%  {base:>12.1f}%  {diff:>+8.1f}%")
        else:
            print(f"{name:<12}  {'N/A':>9}  {base:>12.1f}%  {'N/A':>8}")
    print("-" * 65)
    diff_overall = overall - SMOLVLM_BASELINE["overall"]
    print(f"{'OVERALL':<12} {overall:>9.1f}%  "
          f"{SMOLVLM_BASELINE['overall']:>12.1f}%  {diff_overall:>+8.1f}%")

    parse_pct = unparsed / n_total * 100
    diff_parse = parse_pct - SMOLVLM_BASELINE["parse_failure_pct"]

    print(f"\n{'Metric':<25} {'SmolVLM2':>12} {'SmolVLM':>12}")
    print("-" * 52)
    print(f"{'avg ms / img':<25} {avg_ms:>11.0f}  {SMOLVLM_BASELINE['avg_ms']:>11,}")
    print(f"{'parse failure %':<25} {parse_pct:>10.1f}%  "
          f"{SMOLVLM_BASELINE['parse_failure_pct']:>10.1f}%  ({diff_parse:+.1f}%)")
    print(f"{'vs supervised (96.67%)':<25} {overall-SUPERVISED_BASELINE:>+11.2f}%")
    print("=" * 65)


def save_csv(overall, per_class, times_ms, unparsed, rows, n_total):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, "smolvlm2_256m_terrain_test.csv")
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["image", "gt", "pred", "ms", "answer", "correct"])
        writer.writeheader()
        writer.writerows(rows)
        writer.writerow({})
        writer.writerow({"image": "SUMMARY", "gt": "overall",
                         "pred": f"{overall:.2f}%",
                         "ms": f"{np.mean(times_ms):.0f}",
                         "answer": f"parse_fail={unparsed}/{n_total}",
                         "correct": ""})
        for name in CLASS_NAMES:
            val = per_class.get(name)
            writer.writerow({"image": "", "gt": name,
                             "pred": f"{val:.2f}%" if val is not None else "N/A",
                             "ms": "", "answer": "", "correct": ""})
    print(f"\nCSV saved → {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SmolVLM2-256M terrain classification")
    parser.add_argument("--n",   type=int, default=None,
                        help="Limit test images (default: all ~287)")
    args = parser.parse_args()

    # MX330 GPU does not support all ops — force CPU (same as SmolVLM original)
    device = "cpu"

    processor, model = load_model(device)

    print(f"\nLoading test set...")
    test_pairs = load_test_set(max_n=args.n)
    print(f"Images to evaluate: {len(test_pairs)}")
    print(f"Prompt (q1): '{QUESTION}'")

    overall, per_class, times_ms, unparsed, rows = evaluate(
        processor, model, test_pairs, device)

    print_results(overall, per_class, times_ms, unparsed, len(test_pairs))
    save_csv(overall, per_class, times_ms, unparsed, rows, len(test_pairs))

    print(f"\nDone. Key comparison for thesis:")
    print(f"  SmolVLM2-256M overall: {overall:.1f}%  vs  SmolVLM-256M: {SMOLVLM_BASELINE['overall']}%  "
          f"({'improved' if overall > SMOLVLM_BASELINE['overall'] else 'not improved'})")
    parse_pct = unparsed / len(test_pairs) * 100
    print(f"  Parse failure: {parse_pct:.1f}%  vs  SmolVLM: {SMOLVLM_BASELINE['parse_failure_pct']}%  "
          f"({'reduced' if parse_pct < SMOLVLM_BASELINE['parse_failure_pct'] else 'increased'})")


if __name__ == "__main__":
    main()
