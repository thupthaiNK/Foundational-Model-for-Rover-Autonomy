"""
Purpose:    Benchmark FM inference speed and memory usage on target hardware (PC / RPi 4).
            Measures latency, RAM, CPU%, and FPS for each model to validate RPi deployment feasibility.
            Part of Experiment: FM Efficiency Benchmark (plan.md Phase 2 Week 3).
Inputs:     20 AI4Mars test images (random sample), local model caches
Outputs:    Console table + experiments/results/rpi_benchmark_<hostname>.csv
How to run:
    # On PC (CPU):
    python3 experiments/rpi_benchmark.py

    # On RPi 4 (copy script + run):
    python3 rpi_benchmark.py

    # Select which models to run:
    python3 experiments/rpi_benchmark.py --models clip onnx mobileclip

    # Quick smoke test (5 images):
    python3 experiments/rpi_benchmark.py --n 5 --models clip

Project: Onboard Visual Foundation Models for Mars Terrain Perception and Rover Navigation
"""

import argparse
import csv
import os
import pickle
import platform
import socket
import tempfile
import time

import numpy as np
import psutil
from PIL import Image

# ── Paths ─────────────────────────────────────────────────────────────────────
AI4MARS_BASE = "/mnt/c/Users/DELL/Desktop/Thesis/github source/ai4mars-dataset-merged-0.1"
IMAGES_DIR   = os.path.join(AI4MARS_BASE, "msl/images/edr")
TEST_LABELS  = os.path.join(AI4MARS_BASE, "msl/labels/test/masked-gold-min3-100agree")
RESULTS_DIR  = os.path.join(os.path.dirname(__file__), "results")

# CLIP prompts (v9 best-overall)
CLIP_PROMPTS = [
    "a NAVCAM image of fine-grained Martian soil or regolith covering the ground",
    "a NAVCAM image of flat continuous exposed bedrock with no loose material",
    "a NAVCAM image of bright sandy terrain or sand dune ripples on Mars",
    "a NAVCAM image of a large boulder or big isolated rock on Mars terrain",
]
LABEL_NAMES = ["soil", "bedrock", "sand", "big_rock"]

WARMUP_IMAGES = 2   # number of warmup passes before timing
N_DEFAULT     = 20  # default images to benchmark

# DINOv2 preprocessing constants (ImageNet stats, same as HuggingFace processor)
_DINOV2_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_DINOV2_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ── Image loading ──────────────────────────────────────────────────────────────

def _generate_random_image_files(n: int) -> list:
    """Create n temporary 640×480 noise PNG files for latency-only benchmarks."""
    tmp_dir = tempfile.mkdtemp(prefix="rpi_bench_")
    paths = []
    rng = np.random.default_rng(42)
    for i in range(n):
        arr = (rng.integers(0, 256, (480, 640, 3), dtype=np.uint8))
        p = os.path.join(tmp_dir, f"rand_{i:03d}.png")
        Image.fromarray(arr).save(p)
        paths.append(p)
    return paths


def load_sample_images(n: int, images_dir: str = "") -> list:
    """Return up to n image paths. Falls back to random noise images on RPi."""
    # Allow explicit override (--images-dir flag)
    src_dir = images_dir or IMAGES_DIR
    label_dir = TEST_LABELS

    paths = []
    if os.path.isdir(src_dir) and os.path.isdir(label_dir):
        for fname in sorted(os.listdir(label_dir)):
            if not fname.endswith("_merged.png"):
                continue
            stem = fname.replace("_merged.png", "")
            for ext in (".JPG", ".jpg", ".png", ".PNG"):
                img_path = os.path.join(src_dir, stem + ext)
                if os.path.exists(img_path):
                    paths.append(img_path)
                    break
            if len(paths) >= n:
                break
    elif os.path.isdir(src_dir):
        # images-dir provided but no label dir — load directly
        for fname in sorted(os.listdir(src_dir)):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                paths.append(os.path.join(src_dir, fname))
            if len(paths) >= n:
                break

    if not paths:
        print(f"  [INFO] AI4Mars dataset not found at {src_dir}")
        print(f"  [INFO] Using {n} random noise images for latency measurement")
        paths = _generate_random_image_files(n)

    return paths


# ── Benchmark helpers ──────────────────────────────────────────────────────────

def _measure_loop(fn, images: list, warmup: int = WARMUP_IMAGES) -> dict:
    """
    Call fn(image_path) for each image, skip warmup images for timing.
    Returns dict with timing and memory stats.
    """
    proc = psutil.Process()
    times_ms = []
    ram_mb   = []

    for i, img_path in enumerate(images):
        t0  = time.perf_counter()
        fn(img_path)
        elapsed = (time.perf_counter() - t0) * 1000
        mem     = proc.memory_info().rss / 1e6

        if i >= warmup:
            times_ms.append(elapsed)
            ram_mb.append(mem)

    times_arr = np.array(times_ms) if times_ms else np.array([0.0])
    return {
        "avg_ms":  float(np.mean(times_arr)),
        "p50_ms":  float(np.percentile(times_arr, 50)),
        "p95_ms":  float(np.percentile(times_arr, 95)),
        "fps":     float(1000 / np.mean(times_arr)) if np.mean(times_arr) > 0 else 0,
        "avg_ram_mb": float(np.mean(ram_mb)) if ram_mb else 0,
        "n_measured": len(times_ms),
    }


# ── CLIP ViT-B/32 (vanilla PyTorch) ──────────────────────────────────────────

def benchmark_clip(images: list) -> dict:
    try:
        import clip
        import torch
    except ImportError:
        print("  [SKIP] clip or torch not installed. Run: pip install git+https://github.com/openai/CLIP.git torch")
        return None

    print("  Loading CLIP ViT-B/32...")
    t0 = time.perf_counter()
    model, preprocess = clip.load("ViT-B/32", device="cpu")
    load_time = time.perf_counter() - t0
    model.eval()

    # Pre-encode text features once (not counted in per-image timing)
    text_tokens = clip.tokenize(CLIP_PROMPTS)
    with torch.no_grad():
        txt_feats = model.encode_text(text_tokens)
        txt_feats /= txt_feats.norm(dim=-1, keepdim=True)

    def infer(img_path: str):
        img  = preprocess(Image.open(img_path).convert("RGB")).unsqueeze(0)
        with torch.no_grad():
            img_feat = model.encode_image(img)
            img_feat /= img_feat.norm(dim=-1, keepdim=True)
            logits   = (img_feat @ txt_feats.T) * 100
        probs = logits.softmax(dim=-1)[0]
        return LABEL_NAMES[int(probs.argmax())]

    stats = _measure_loop(infer, images)
    stats["load_time_s"] = round(load_time, 1)
    stats["model_size_mb"] = 338
    return stats


# ── CLIP ViT-B/32 (ONNX Runtime) ─────────────────────────────────────────────

def benchmark_clip_onnx(images: list, onnx_path: str = "clip_visual_vit_b32.onnx") -> dict:
    """
    Export CLIP visual encoder to ONNX once, then benchmark ONNX Runtime inference.
    Expected speedup vs PyTorch CPU: 2–3× on x86; higher on ARM.
    """
    try:
        import clip
        import torch
        import onnxruntime as ort
    except ImportError:
        print("  [SKIP] clip, torch, or onnxruntime not installed.")
        return None

    # Export ONNX model if not already done
    if not os.path.exists(onnx_path):
        print(f"  Exporting CLIP visual encoder to {onnx_path}...")
        model, preprocess = clip.load("ViT-B/32", device="cpu")
        model.eval()
        dummy = torch.zeros(1, 3, 224, 224)
        torch.onnx.export(
            model.visual, dummy, onnx_path,
            input_names=["image"], output_names=["features"],
            opset_version=14,
        )
        print(f"  Exported → {onnx_path}")

    import clip as clip_module
    model_ref, preprocess = clip_module.load("ViT-B/32", device="cpu")
    model_ref.eval()
    text_tokens = clip_module.tokenize(CLIP_PROMPTS)
    with torch.no_grad():
        txt_feats = model_ref.encode_text(text_tokens).numpy()
        txt_feats /= np.linalg.norm(txt_feats, axis=-1, keepdims=True)

    print(f"  Loading ONNX session from {onnx_path}...")
    t0 = time.perf_counter()
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    load_time = time.perf_counter() - t0

    def infer(img_path: str):
        img_tensor = preprocess(Image.open(img_path).convert("RGB")).unsqueeze(0).numpy()
        img_feat   = session.run(None, {"image": img_tensor})[0]
        img_feat  /= np.linalg.norm(img_feat, axis=-1, keepdims=True)
        logits     = (img_feat @ txt_feats.T) * 100
        return LABEL_NAMES[int(np.argmax(logits[0]))]

    stats = _measure_loop(infer, images)
    stats["load_time_s"] = round(load_time, 1)
    stats["model_size_mb"] = 300  # approximate ONNX file size
    return stats


# ── MobileCLIP S0 ─────────────────────────────────────────────────────────────

def benchmark_mobileclip(images: list) -> dict:
    """
    Apple MobileCLIP S0 — ~100MB, designed for mobile/edge deployment.
    Install: pip install open-clip-torch
    """
    try:
        import open_clip
        import torch
    except ImportError:
        print("  [SKIP] open_clip or torch not installed. Run: pip install open-clip-torch torch")
        return None

    print("  Loading MobileCLIP-S0 (open_clip)...")
    t0 = time.perf_counter()
    try:
        model, _, preprocess = open_clip.create_model_and_transforms(
            "MobileCLIP-S0", pretrained="datacompdr", device="cpu"
        )
        tokenizer = open_clip.get_tokenizer("MobileCLIP-S0")
    except Exception as e:
        print(f"  [SKIP] MobileCLIP load failed: {e}")
        return None
    load_time = time.perf_counter() - t0
    model.eval()

    text_tokens = tokenizer(CLIP_PROMPTS)
    with torch.no_grad():
        txt_feats = model.encode_text(text_tokens)
        txt_feats /= txt_feats.norm(dim=-1, keepdim=True)

    def infer(img_path: str):
        img = preprocess(Image.open(img_path).convert("RGB")).unsqueeze(0)
        with torch.no_grad():
            img_feat = model.encode_image(img)
            img_feat /= img_feat.norm(dim=-1, keepdim=True)
            logits   = (img_feat @ txt_feats.T) * 100
        return LABEL_NAMES[int(logits.softmax(dim=-1)[0].argmax())]

    stats = _measure_loop(infer, images)
    stats["load_time_s"] = round(load_time, 1)
    stats["model_size_mb"] = 100
    return stats


# ── SmolVLM-256M (periodic VQA) ───────────────────────────────────────────────

def benchmark_smolvlm(images: list,
                      model_id: str = "HuggingFaceTB/SmolVLM-256M-Instruct") -> dict:
    """
    SmolVLM-256M-Instruct — supervisor-recommended edge VLM for RPi 4.
    ~256M params, ~2GB RAM, ~55s/img on PC CPU, ~30-60s/img expected on RPi 4 ARM.
    NOTE: This is slow — 20 images ≈ 20 minutes on PC, ~10-20 min on RPi 4.
    """
    try:
        import torch
        from transformers import AutoProcessor, SmolVLMForConditionalGeneration
    except ImportError:
        print("  [SKIP] torch or transformers not installed.")
        return None

    QUESTION = (
        "What type of terrain is visible in this Mars image? "
        "Answer with exactly one word: soil, bedrock, sand, or rock."
    )

    print(f"  Loading SmolVLM-256M ({model_id})...")
    print("  WARNING: ~55s/image on PC CPU — this benchmark will take several minutes.")
    t0 = time.perf_counter()
    try:
        processor = AutoProcessor.from_pretrained(model_id)
        model = SmolVLMForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch.float32, device_map={"": "cpu"}
        )
    except Exception as e:
        print(f"  [SKIP] SmolVLM load failed: {e}")
        return None
    load_time = time.perf_counter() - t0
    model.eval()

    messages = [{"role": "user", "content": [
        {"type": "image"}, {"type": "text", "text": QUESTION}
    ]}]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)

    def infer(img_path: str):
        from PIL import Image as PILImage
        pil_img = PILImage.open(img_path).convert("RGB")
        inputs = processor(text=prompt, images=[pil_img], return_tensors="pt")
        with torch.no_grad():
            out_ids = model.generate(**inputs, max_new_tokens=20, do_sample=False)
        n_in = inputs["input_ids"].shape[1]
        return processor.tokenizer.decode(
            out_ids[0][n_in:], skip_special_tokens=True
        ).strip()

    stats = _measure_loop(infer, images)
    stats["load_time_s"] = round(load_time, 1)
    stats["model_size_mb"] = 1000  # ~1GB on disk
    return stats


# ── DINOv2 ViT-S/14 (vanilla PyTorch) ────────────────────────────────────────

def benchmark_dinov2(images: list,
                     probe_path: str = "dinov2_logreg_probe.pkl") -> dict:
    """
    DINOv2 ViT-S/14 (facebook/dinov2-small) + pre-trained LogReg probe.
    This is the model deployed in the Gazebo traversability experiment (§4.8).
    Expected: ~181ms/img on PC CPU; ~500-900ms/img on RPi 4 ARMv8.
    Requires:
      - torch, transformers, sklearn
      - dinov2_logreg_probe.pkl (generated by export_dinov2_onnx.py)
    """
    try:
        import torch
        from transformers import AutoImageProcessor, AutoModel
        from sklearn.preprocessing import normalize as sk_normalize
    except ImportError:
        print("  [SKIP] torch, transformers, or sklearn not installed")
        return None

    probe_path = probe_path if os.path.exists(probe_path) else \
        os.path.join(os.path.dirname(__file__), "results", "dinov2_logreg_probe.pkl")
    if not os.path.exists(probe_path):
        print(f"  [SKIP] Probe not found: {probe_path}  — run export_dinov2_onnx.py first")
        return None

    print("  Loading DINOv2 ViT-S/14 (PyTorch) + LogReg probe …")
    t0 = time.perf_counter()
    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
    model     = AutoModel.from_pretrained("facebook/dinov2-small")
    model.eval()
    with open(probe_path, "rb") as f:
        clf = pickle.load(f)
    load_time = time.perf_counter() - t0

    def infer(img_path: str):
        img    = Image.open(img_path).convert("RGB")
        inputs = processor(images=img, return_tensors="pt")
        with torch.no_grad():
            out = model(**inputs)
            cls = out.last_hidden_state[:, 0, :].numpy()  # (1, 384)
        cls = cls / np.linalg.norm(cls, axis=-1, keepdims=True)
        return clf.predict(cls)[0]

    stats = _measure_loop(infer, images)
    stats["load_time_s"]    = round(load_time, 1)
    stats["model_size_mb"]  = 86
    return stats


# ── DINOv2 ViT-S/14 (ONNX Runtime) ───────────────────────────────────────────

def benchmark_dinov2_onnx(images: list,
                           onnx_path: str  = "dinov2_small_encoder.onnx",
                           probe_path: str = "dinov2_logreg_probe.pkl") -> dict:
    """
    DINOv2 ViT-S/14 via ONNX Runtime + pre-trained LogReg probe.
    Preprocessing done with PIL + numpy (no transformers dependency at inference).
    Expected speedup vs PyTorch on ARM: 2–4× due to optimised NEON kernels.
    Requires:
      - onnxruntime, sklearn
      - dinov2_small_encoder.onnx + dinov2_logreg_probe.pkl (from export_dinov2_onnx.py)
    """
    try:
        import onnxruntime as ort
    except ImportError:
        print("  [SKIP] onnxruntime not installed. Run: pip install onnxruntime")
        return None

    # Resolve paths: try local first, then results/
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    if not os.path.exists(onnx_path):
        onnx_path = os.path.join(results_dir, "dinov2_small_encoder.onnx")
    if not os.path.exists(probe_path):
        probe_path = os.path.join(results_dir, "dinov2_logreg_probe.pkl")

    if not os.path.exists(onnx_path):
        print(f"  [SKIP] ONNX model not found: {onnx_path}  — run export_dinov2_onnx.py first")
        return None
    if not os.path.exists(probe_path):
        print(f"  [SKIP] Probe not found: {probe_path}  — run export_dinov2_onnx.py first")
        return None

    print(f"  Loading DINOv2 ONNX session ({onnx_path}) + LogReg probe …")
    t0      = time.perf_counter()
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    with open(probe_path, "rb") as f:
        clf = pickle.load(f)
    load_time = time.perf_counter() - t0

    def infer(img_path: str):
        # Manual ImageNet preprocessing — no transformers import needed on RPi
        img = Image.open(img_path).convert("RGB").resize((224, 224))
        x   = np.array(img, dtype=np.float32) / 255.0          # (224, 224, 3)
        x   = (x - _DINOV2_MEAN) / _DINOV2_STD                 # normalise
        x   = x.transpose(2, 0, 1)[np.newaxis, :]              # (1, 3, 224, 224)
        feats = session.run(None, {"pixel_values": x})[0]       # (1, 384)
        return clf.predict(feats)[0]

    stats = _measure_loop(infer, images)
    stats["load_time_s"]   = round(load_time, 1)
    stats["model_size_mb"] = int(os.path.getsize(onnx_path) / 1e6) \
                             if os.path.exists(onnx_path) else 90
    return stats


# ── System info ────────────────────────────────────────────────────────────────

def get_system_info() -> dict:
    mem = psutil.virtual_memory()
    cpu = platform.processor() or platform.machine()
    try:
        import torch as _torch
        torch_ver = _torch.__version__
    except ImportError:
        torch_ver = "not installed"
    return {
        "hostname":    socket.gethostname(),
        "platform":    platform.platform(),
        "cpu":         cpu,
        "cpu_cores":   psutil.cpu_count(logical=False),
        "ram_total_gb": round(mem.total / 1e9, 1),
        "torch":       torch_ver,
        "python":      platform.python_version(),
    }


# ── Reporting ──────────────────────────────────────────────────────────────────

def print_table(results: dict):
    col_names = ["Model", "avg ms", "p50 ms", "p95 ms", "FPS", "RAM MB", "Load s", "Size MB"]
    rows = []
    for name, stats in results.items():
        if stats is None:
            continue
        rows.append([
            name,
            f"{stats['avg_ms']:.0f}",
            f"{stats['p50_ms']:.0f}",
            f"{stats['p95_ms']:.0f}",
            f"{stats['fps']:.2f}",
            f"{stats['avg_ram_mb']:.0f}",
            f"{stats['load_time_s']:.1f}",
            f"{stats['model_size_mb']}",
        ])

    col_w = [max(len(col_names[i]), max((len(r[i]) for r in rows), default=0)) + 2
             for i in range(len(col_names))]
    sep = "+" + "+".join("-" * w for w in col_w) + "+"
    header = "|" + "|".join(f" {col_names[i]:<{col_w[i]-1}}" for i in range(len(col_names))) + "|"

    print(sep)
    print(header)
    print(sep)
    for row in rows:
        print("|" + "|".join(f" {row[i]:<{col_w[i]-1}}" for i in range(len(col_names))) + "|")
    print(sep)


def save_csv(results: dict, sysinfo: dict, n_images: int):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    hostname = sysinfo["hostname"].replace(" ", "_")
    path     = os.path.join(RESULTS_DIR, f"rpi_benchmark_{hostname}.csv")
    fieldnames = ["model", "hostname", "platform", "cpu", "ram_total_gb",
                  "avg_ms", "p50_ms", "p95_ms", "fps", "avg_ram_mb",
                  "load_time_s", "model_size_mb", "n_measured"]
    rows = []
    for name, stats in results.items():
        if stats is None:
            continue
        row = {"model": name, **{k: sysinfo.get(k, "") for k in
                                 ["hostname", "platform", "cpu", "ram_total_gb"]}}
        row.update({k: stats.get(k, "") for k in fieldnames if k not in row})
        rows.append(row)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nCSV saved → {path}")
    return path


# ── Main ───────────────────────────────────────────────────────────────────────

AVAILABLE_MODELS = {
    "clip":            benchmark_clip,
    "onnx":            benchmark_clip_onnx,
    "mobileclip":      benchmark_mobileclip,
    "dinov2_pytorch":  benchmark_dinov2,
    "dinov2_onnx":     benchmark_dinov2_onnx,
    "smolvlm":         benchmark_smolvlm,
}

# Recommended set for RPi thesis benchmark (§4.9)
RPi_MODELS = ["clip", "onnx", "dinov2_pytorch", "dinov2_onnx"]


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark FM inference speed on CPU hardware (PC or RPi 4). "
                    "Works without AI4Mars dataset on RPi — uses random noise images "
                    "for latency measurement. Prerequisite for DINOv2 models: "
                    "run export_dinov2_onnx.py on PC first."
    )
    parser.add_argument("--n", type=int, default=N_DEFAULT,
                        help=f"Number of images to benchmark (default: {N_DEFAULT})")
    parser.add_argument("--models", nargs="+", choices=list(AVAILABLE_MODELS.keys()),
                        default=RPi_MODELS,
                        help=f"Models to benchmark (default: {RPi_MODELS}). "
                             "smolvlm is very slow (~55s/img on PC, ~2-4 min/img on RPi)")
    parser.add_argument("--images-dir", default="",
                        help="Path to directory of test images. "
                             "Overrides default AI4Mars path. "
                             "If omitted and AI4Mars not found, uses random noise images.")
    parser.add_argument("--onnx-path", default="clip_visual_vit_b32.onnx",
                        help="ONNX model path for --models onnx (CLIP ONNX)")
    parser.add_argument("--dinov2-onnx-path", default="dinov2_small_encoder.onnx",
                        help="ONNX model path for --models dinov2_onnx")
    parser.add_argument("--probe-path", default="dinov2_logreg_probe.pkl",
                        help="LogReg probe .pkl for DINOv2 models")
    args = parser.parse_args()

    sysinfo = get_system_info()
    print("\n=== RPi Benchmark — FM Inference Speed ===")
    print(f"Host:     {sysinfo['hostname']}")
    print(f"CPU:      {sysinfo['cpu']} ({sysinfo['cpu_cores']} cores)")
    print(f"RAM:      {sysinfo['ram_total_gb']} GB")
    print(f"PyTorch:  {sysinfo['torch']}")
    print(f"Models:   {args.models}")
    print(f"Images:   {args.n} measured + {WARMUP_IMAGES} warmup")
    print()

    images = load_sample_images(args.n + WARMUP_IMAGES, args.images_dir)

    results = {}
    for model_name in args.models:
        print(f"── {model_name.upper()} ──")
        fn = AVAILABLE_MODELS[model_name]
        if model_name == "onnx":
            stats = fn(images, args.onnx_path)
        elif model_name == "dinov2_pytorch":
            stats = fn(images, args.probe_path)
        elif model_name == "dinov2_onnx":
            stats = fn(images, args.dinov2_onnx_path, args.probe_path)
        else:
            stats = fn(images)
        results[model_name] = stats
        if stats:
            print(f"  avg {stats['avg_ms']:.0f}ms | {stats['fps']:.2f} FPS | "
                  f"RAM {stats['avg_ram_mb']:.0f}MB\n")

    print("\n=== RESULTS ===")
    print_table(results)
    save_csv(results, sysinfo, args.n)

    print("\nThesis targets (§4.9):")
    print("  CLIP PyTorch RPi 4:      ~500–800 ms/img")
    print("  CLIP ONNX RPi 4:         ~200–350 ms/img  (2–3× PyTorch speedup)")
    print("  DINOv2-S PyTorch RPi 4:  ~500–900 ms/img")
    print("  DINOv2-S ONNX RPi 4:     ~200–400 ms/img  (2–4× PyTorch speedup on ARM)")
    print("  SmolVLM-256M RPi 4:      ~30,000–120,000 ms/img  (periodic slow-path only)")


if __name__ == "__main__":
    main()
