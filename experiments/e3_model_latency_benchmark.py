"""
Purpose: E3 — PC inference latency and memory benchmark for all three Gazebo models:
         DINOv2+reg ViT-S/14, CLIP ViT-B/32, SmolVLM-256M.
         Provides a structured comparison table for Ch4 §4.9 and extrapolates
         estimated RPi 4 latency using the measured DINOv2 PC→RPi speedup factor (3.2×).
Inputs:  None (generates random 480×640 noise images matching AI4Mars/Gazebo resolution)
Outputs: experiments/results/e3_model_latency_benchmark.csv
         Printed table suitable for Ch4
How to run:
    python3 experiments/e3_model_latency_benchmark.py
    # DINOv2 + CLIP: ~2 min total
    # SmolVLM: adds ~3 min (3 images × ~60s)
    # Skip SmolVLM with --skip-smolvlm flag for faster run
Project: Onboard Visual Foundation Models for Mars Terrain Perception and Rover Navigation
"""

import argparse, csv, os, time, gc
import numpy as np
import psutil
import torch
from PIL import Image

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

N_DINOV2 = 20   # inference runs for DINOv2 + CLIP
N_SMOLVLM = 3   # fewer because SmolVLM is very slow
IMG_H, IMG_W = 480, 640
RPi_FACTOR = 3.2   # measured: DINOv2 PC-ONNX 178ms → RPi ONNX 570ms


def ram_mb():
    return psutil.Process().memory_info().rss / 1e6


def make_image():
    arr = (np.random.rand(IMG_H, IMG_W, 3) * 255).astype(np.uint8)
    return Image.fromarray(arr)


# ── DINOv2+reg ViT-S/14 ────────────────────────────────────────────────────────
def bench_dinov2(n):
    print(f"\n[1/3] DINOv2+reg ViT-S/14 (PyTorch, CPU, n={n})")
    import torch
    from transformers import AutoModel, AutoImageProcessor

    ram_before = ram_mb()
    t0 = time.time()
    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-with-registers-small")
    model = AutoModel.from_pretrained("facebook/dinov2-with-registers-small")
    model.eval()
    load_s = time.time() - t0
    ram_loaded = ram_mb()
    print(f"  Load: {load_s:.1f}s  RAM: {ram_loaded - ram_before:.0f} MB")

    img = make_image()
    times = []
    with torch.no_grad():
        for i in range(n):
            t0 = time.time()
            inputs = processor(images=img, return_tensors="pt")
            outputs = model(**inputs)
            _ = outputs.last_hidden_state[:, 0, :].numpy()
            times.append((time.time() - t0) * 1000)
    ram_peak = ram_mb()

    mean_ms = np.mean(times[2:])   # skip first 2 warmup
    std_ms  = np.std(times[2:])
    print(f"  Inference: {mean_ms:.0f}±{std_ms:.0f} ms/img  RAM peak: {ram_peak - ram_before:.0f} MB")
    del model, processor
    gc.collect()
    return mean_ms, ram_peak - ram_before, load_s


# ── CLIP ViT-B/32 ─────────────────────────────────────────────────────────────
def bench_clip(n):
    print(f"\n[2/3] CLIP ViT-B/32 (CPU, n={n})")
    import clip

    ram_before = ram_mb()
    t0 = time.time()
    model, preprocess = clip.load("ViT-B/32", device="cpu")
    model = model.float().eval()
    # pre-encode text (same as deployed usage)
    PROMPTS = [
        "fine-grained loose soil on Mars surface",
        "flat smooth bedrock rock surface on Mars",
        "bright sandy terrain or sand dunes on Mars",
        "large boulders or big rocks on Mars terrain",
    ]
    with torch.no_grad():
        text_tok = clip.tokenize(PROMPTS)
        text_feat = model.encode_text(text_tok)
        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
    load_s = time.time() - t0
    ram_loaded = ram_mb()
    print(f"  Load+encode text: {load_s:.1f}s  RAM: {ram_loaded - ram_before:.0f} MB")

    img = make_image()
    times = []
    with torch.no_grad():
        for i in range(n):
            t0 = time.time()
            img_tensor = preprocess(img).unsqueeze(0)
            img_feat = model.encode_image(img_tensor)
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
            _ = (100.0 * img_feat @ text_feat.T).softmax(dim=-1)
            times.append((time.time() - t0) * 1000)
    ram_peak = ram_mb()

    mean_ms = np.mean(times[2:])
    std_ms  = np.std(times[2:])
    print(f"  Inference: {mean_ms:.0f}±{std_ms:.0f} ms/img  RAM peak: {ram_peak - ram_before:.0f} MB")
    del model
    gc.collect()
    return mean_ms, ram_peak - ram_before, load_s


# ── SmolVLM-256M ──────────────────────────────────────────────────────────────
def bench_smolvlm(n):
    print(f"\n[3/3] SmolVLM-256M (CPU, n={n})")
    from transformers import AutoProcessor, AutoModelForVision2Seq

    ram_before = ram_mb()
    t0 = time.time()
    processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-256M-Instruct")
    model = AutoModelForVision2Seq.from_pretrained(
        "HuggingFaceTB/SmolVLM-256M-Instruct",
        torch_dtype=torch.float32
    )
    model.eval()
    load_s = time.time() - t0
    ram_loaded = ram_mb()
    print(f"  Load: {load_s:.1f}s  RAM: {ram_loaded - ram_before:.0f} MB")

    PROMPT = "What is the terrain type in this Mars rover image? Answer with one word: soil, bedrock, sand, or rock."
    img = make_image()
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}]
    text_input = processor.apply_chat_template(messages, add_generation_prompt=True)

    times = []
    for i in range(n):
        print(f"  Run {i+1}/{n}...", end=" ", flush=True)
        t0 = time.time()
        inputs = processor(text=text_input, images=[img], return_tensors="pt")
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=20, do_sample=False)
        elapsed = (time.time() - t0) * 1000
        times.append(elapsed)
        print(f"{elapsed/1000:.1f}s")
    ram_peak = ram_mb()

    mean_ms = np.mean(times)
    std_ms  = np.std(times)
    print(f"  Inference: {mean_ms/1000:.1f}±{std_ms/1000:.1f} s/img  RAM peak: {ram_peak - ram_before:.0f} MB")
    del model, processor
    gc.collect()
    return mean_ms, ram_peak - ram_before, load_s


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-smolvlm", action="store_true", help="Skip SmolVLM benchmark")
    args = parser.parse_args()

    print("E3 — Multi-Model PC Latency Benchmark")
    print(f"Image size: {IMG_W}×{IMG_H}px  n_dinov2/clip={N_DINOV2}  n_smolvlm={N_SMOLVLM}")
    print(f"RPi extrapolation factor: {RPi_FACTOR}× (from DINOv2 PC-ONNX 178ms → RPi 570ms)")

    results = []

    # DINOv2
    d_ms, d_ram, d_load = bench_dinov2(N_DINOV2)
    results.append({
        "model": "DINOv2+reg ViT-S/14",
        "backend": "PyTorch CPU",
        "n_runs": N_DINOV2,
        "mean_ms": round(d_ms),
        "fps_pc": round(1000 / d_ms, 2),
        "ram_mb": round(d_ram),
        "load_s": round(d_load, 1),
        "rpi_est_ms": round(d_ms * RPi_FACTOR),
        "rpi_est_fps": round(1000 / (d_ms * RPi_FACTOR), 2),
        "note": "Deployed model; RPi ONNX measured=570ms"
    })

    # CLIP
    c_ms, c_ram, c_load = bench_clip(N_DINOV2)
    results.append({
        "model": "CLIP ViT-B/32",
        "backend": "PyTorch CPU",
        "n_runs": N_DINOV2,
        "mean_ms": round(c_ms),
        "fps_pc": round(1000 / c_ms, 2),
        "ram_mb": round(c_ram),
        "load_s": round(c_load, 1),
        "rpi_est_ms": round(c_ms * RPi_FACTOR),
        "rpi_est_fps": round(1000 / (c_ms * RPi_FACTOR), 2),
        "note": "Used in X5 cascade; RPi estimate via 3.2x factor"
    })

    # SmolVLM
    if not args.skip_smolvlm:
        s_ms, s_ram, s_load = bench_smolvlm(N_SMOLVLM)
        results.append({
            "model": "SmolVLM-256M-Instruct",
            "backend": "PyTorch CPU",
            "n_runs": N_SMOLVLM,
            "mean_ms": round(s_ms),
            "fps_pc": round(1000 / s_ms, 4),
            "ram_mb": round(s_ram),
            "load_s": round(s_load, 1),
            "rpi_est_ms": round(s_ms * RPi_FACTOR),
            "rpi_est_fps": round(1000 / (s_ms * RPi_FACTOR), 4),
            "note": "Used in Exp 3a; not viable for real-time"
        })
    else:
        print("\n[3/3] SmolVLM-256M — SKIPPED (--skip-smolvlm)")
        results.append({
            "model": "SmolVLM-256M-Instruct",
            "backend": "PyTorch CPU",
            "n_runs": 0,
            "mean_ms": 61000,
            "fps_pc": 0.016,
            "ram_mb": "~1500",
            "load_s": "~30",
            "rpi_est_ms": round(61000 * RPi_FACTOR),
            "rpi_est_fps": round(1000 / (61000 * RPi_FACTOR), 5),
            "note": "Skipped — from Exp 3a Gazebo run (~61s/img observed)"
        })

    # Print summary table
    print("\n" + "=" * 80)
    print("E3 RESULTS — PC Latency Comparison")
    print("=" * 80)
    print(f"{'Model':<26} {'PC ms/img':>10} {'PC FPS':>8} {'RAM MB':>8} {'RPi est ms':>12} {'RPi FPS':>9}")
    print("-" * 80)
    for r in results:
        print(f"{r['model']:<26} {str(r['mean_ms']):>10} {str(r['fps_pc']):>8} {str(r['ram_mb']):>8} "
              f"{str(r['rpi_est_ms']):>12} {str(r['rpi_est_fps']):>9}")
    print("-" * 80)
    print(f"RPi 4 reference: DINOv2 ONNX measured = 570ms (1.75 FPS), 258 MB RAM")

    # Save CSV
    csv_path = os.path.join(RESULTS_DIR, "e3_model_latency_benchmark.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)
    print(f"\nSaved → {csv_path}")


if __name__ == "__main__":
    main()
