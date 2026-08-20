"""
Purpose:    B1 (RPi side) — Benchmark FP32 vs INT8 DINOv2+reg ViT-S on Raspberry Pi 4.
            Measures per-image latency, FPS, and RAM for both precisions.
            Run on RPi 4 AFTER copying files from PC.
Inputs:     dinov2_reg_small_encoder.onnx      (FP32 ONNX — copy from PC results/)
            dinov2_reg_small_encoder_int8.onnx (INT8 ONNX — copy from PC results/)
            dinov2_reg_small_probe.npz          (LogReg weights — copy from PC results/)
Outputs:    b1_rpi_benchmark.csv  (saved next to this script)
How to run:
    # On RPi, after scp of all 4 files:
    python3 int8_benchmark_rpi.py
    python3 int8_benchmark_rpi.py --n 50          # more images, more stable mean
    python3 int8_benchmark_rpi.py --no-fp32       # INT8 only (skip slow FP32 baseline)
Project: Onboard Visual Foundation Models for Mars Terrain Perception and Rover Navigation
"""

import argparse
import csv
import os
import platform
import socket
import tempfile
import time

import numpy as np
import psutil
from PIL import Image

# ── Config ─────────────────────────────────────────────────────────────────────
CLASS_NAMES = ["soil", "bedrock", "sand", "big_rock"]
WARMUP      = 3     # images used to warm up ONNX Runtime (not timed)
N_DEFAULT   = 20

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# ── Helpers ────────────────────────────────────────────────────────────────────

def _resolve(filename: str) -> str:
    """Find a file next to the script, or raise with a clear message."""
    p = os.path.join(_SCRIPT_DIR, filename)
    if not os.path.exists(p):
        raise FileNotFoundError(
            f"Required file not found: {p}\n"
            "Copy it from PC:  scp pi@<pc-ip>:<path>/<file> ~/"
        )
    return p


def _make_noise_images(n: int) -> list:
    """Generate n random 224×224 noise images for latency measurement."""
    tmp = tempfile.mkdtemp(prefix="b1_bench_")
    rng = np.random.default_rng(42)
    paths = []
    for i in range(n):
        arr = rng.integers(0, 256, (224, 224, 3), dtype=np.uint8)
        p   = os.path.join(tmp, f"rand_{i:03d}.png")
        Image.fromarray(arr).save(p)
        paths.append(p)
    return paths


def _preprocess(img_path: str) -> np.ndarray:
    img = Image.open(img_path).convert("RGB").resize((224, 224))
    x   = np.array(img, dtype=np.float32) / 255.0
    x   = (x - _MEAN) / _STD
    return x.transpose(2, 0, 1)[np.newaxis, :]   # (1, 3, 224, 224)


def _probe_from_npz(path: str):
    """Reconstruct LogisticRegression from .npz (no pickle, no sklearn import needed)."""
    data = np.load(path)
    return data["coef"], data["intercept"], data["classes"]


def _predict(feat: np.ndarray, coef: np.ndarray, intercept: np.ndarray, classes: np.ndarray) -> str:
    """Manual LogReg forward pass — avoids sklearn dependency on RPi."""
    logits = feat @ coef.T + intercept   # (1, n_classes)
    return CLASS_NAMES[int(classes[np.argmax(logits[0])])]


# ── Benchmark ──────────────────────────────────────────────────────────────────

def benchmark_onnx(onnx_path: str, probe_npz: str, images: list, warmup: int) -> dict:
    import onnxruntime as ort

    label    = os.path.basename(onnx_path)
    size_mb  = os.path.getsize(onnx_path) / 1e6
    print(f"\n  Model:    {label}  ({size_mb:.1f} MB)")

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    coef, intercept, classes = _probe_from_npz(probe_npz)

    proc     = psutil.Process()
    times_ms = []
    ram_mb   = []

    for i, img_path in enumerate(images):
        x    = _preprocess(img_path)
        t0   = time.perf_counter()
        feat = sess.run(None, {"pixel_values": x})[0]
        _predict(feat, coef, intercept, classes)
        elapsed = (time.perf_counter() - t0) * 1000

        if i >= warmup:
            times_ms.append(elapsed)
            ram_mb.append(proc.memory_info().rss / 1e6)

        if (i + 1) % 10 == 0:
            print(f"    {i+1}/{len(images)}  last={elapsed:.0f}ms")

    arr = np.array(times_ms)
    return {
        "model":      label,
        "precision":  "INT8" if "int8" in label else "FP32",
        "size_mb":    round(size_mb, 1),
        "avg_ms":     round(float(np.mean(arr)), 1),
        "p50_ms":     round(float(np.percentile(arr, 50)), 1),
        "p95_ms":     round(float(np.percentile(arr, 95)), 1),
        "fps":        round(float(1000 / np.mean(arr)), 3),
        "avg_ram_mb": round(float(np.mean(ram_mb)), 1),
        "n_measured": len(times_ms),
    }


# ── Reporting ──────────────────────────────────────────────────────────────────

def print_table(rows: list) -> None:
    cols = ["model", "precision", "size_mb", "avg_ms", "p50_ms", "p95_ms", "fps", "avg_ram_mb"]
    headers = ["Model", "Precision", "Size MB", "Avg ms", "p50 ms", "p95 ms", "FPS", "RAM MB"]
    widths  = [max(len(headers[i]), max(len(str(r[cols[i]])) for r in rows)) + 2
               for i in range(len(cols))]
    sep = "+" + "+".join("-" * w for w in widths) + "+"
    hdr = "|" + "|".join(f" {headers[i]:<{widths[i]-1}}" for i in range(len(cols))) + "|"
    print(sep); print(hdr); print(sep)
    for r in rows:
        print("|" + "|".join(f" {str(r[cols[i]]):<{widths[i]-1}}" for i in range(len(cols))) + "|")
    print(sep)

    if len(rows) == 2:
        fp32 = next((r for r in rows if r["precision"] == "FP32"), None)
        int8 = next((r for r in rows if r["precision"] == "INT8"), None)
        if fp32 and int8 and fp32["avg_ms"] > 0:
            speedup = fp32["avg_ms"] / int8["avg_ms"]
            ram_red = fp32["avg_ram_mb"] / int8["avg_ram_mb"] if int8["avg_ram_mb"] > 0 else 0
            siz_red = fp32["size_mb"] / int8["size_mb"] if int8["size_mb"] > 0 else 0
            print(f"\n  INT8 speedup:        {speedup:.2f}×  (FP32 {fp32['avg_ms']:.0f}ms → INT8 {int8['avg_ms']:.0f}ms)")
            print(f"  INT8 RAM reduction:  {ram_red:.2f}×  (FP32 {fp32['avg_ram_mb']:.0f}MB → INT8 {int8['avg_ram_mb']:.0f}MB)")
            print(f"  Model size ratio:    {siz_red:.2f}×  (FP32 {fp32['size_mb']:.1f}MB → INT8 {int8['size_mb']:.1f}MB)")


def save_csv(rows: list, hostname: str) -> str:
    out = os.path.join(_SCRIPT_DIR, "b1_rpi_benchmark.csv")
    fieldnames = ["model", "precision", "size_mb", "avg_ms", "p50_ms", "p95_ms",
                  "fps", "avg_ram_mb", "n_measured", "hostname", "platform", "cpu_cores", "ram_gb"]
    sys_info = {
        "hostname":  hostname,
        "platform":  platform.platform(),
        "cpu_cores": psutil.cpu_count(logical=False),
        "ram_gb":    round(psutil.virtual_memory().total / 1e9, 1),
    }
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({**r, **sys_info})
    return out


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="B1: FP32 vs INT8 benchmark on RPi 4")
    parser.add_argument("--n",       type=int, default=N_DEFAULT, help="Images to benchmark")
    parser.add_argument("--no-fp32", action="store_true",         help="Skip FP32 (time-saving)")
    args = parser.parse_args()

    fp32_path  = _resolve("dinov2_reg_small_encoder.onnx")
    int8_path  = _resolve("dinov2_reg_small_encoder_int8.onnx")
    probe_path = _resolve("dinov2_reg_small_probe.npz")

    hostname = socket.gethostname()
    mem_gb   = round(psutil.virtual_memory().total / 1e9, 1)
    cores    = psutil.cpu_count(logical=False)

    print("=" * 60)
    print(f"B1: INT8 Benchmark — {hostname}")
    print(f"CPU cores: {cores}  |  RAM: {mem_gb} GB")
    print(f"Images:    {args.n} measured + {WARMUP} warmup")
    print("=" * 60)

    total_images = args.n + WARMUP
    images = _make_noise_images(total_images)

    rows = []

    if not args.no_fp32:
        print("\n[1/2] FP32 ONNX baseline …")
        rows.append(benchmark_onnx(fp32_path, probe_path, images, WARMUP))
    else:
        print("\n[1/2] FP32 skipped (--no-fp32)")

    print("\n[2/2] INT8 ONNX …")
    rows.append(benchmark_onnx(int8_path, probe_path, images, WARMUP))

    print("\n\n=== RESULTS ===")
    print_table(rows)

    csv_path = save_csv(rows, hostname)
    print(f"\nCSV saved → {csv_path}")

    if rows:
        print("\nThesis targets (Ch5 §5.1):")
        print("  FP32 ONNX RPi 4:  ~500–600 ms/img  (measured previously: 570ms)")
        print("  INT8 ONNX RPi 4:  ~200–350 ms/img  (expected ~2× speedup)")
        print("  INT8 RAM:         ~130–180 MB       (expected ~1.5–2× reduction)")


if __name__ == "__main__":
    main()
