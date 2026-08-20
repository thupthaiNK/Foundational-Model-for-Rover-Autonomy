"""
Purpose:    B1 — Export DINOv2+reg ViT-S/14 (the thesis deployment model) to FP32 ONNX,
            apply INT8 dynamic quantization, then compare FP32 vs INT8 accuracy on
            AI4Mars gold test set. Run on PC first; then transfer outputs to RPi.
Inputs:     facebook/dinov2-with-registers-small (HuggingFace — downloads automatically)
            experiments/results/feature_cache/dinov2_reg_small_train_1000shot.npz
            AI4Mars msl/images/edr/ + msl/labels/test/masked-gold-min3-100agree/
Outputs:    experiments/results/dinov2_reg_small_encoder.onnx      (FP32 ONNX, ~86 MB)
            experiments/results/dinov2_reg_small_encoder_int8.onnx (INT8 ONNX, ~25 MB)
            experiments/results/dinov2_reg_small_probe.npz          (LogReg weights, ~6 KB)
            experiments/results/b1_accuracy_comparison_pc.csv
How to run:
    python3 experiments/int8_quantization_rpi.py
    # Outputs to copy to RPi afterwards:
    #   dinov2_reg_small_encoder.onnx
    #   dinov2_reg_small_encoder_int8.onnx
    #   dinov2_reg_small_probe.npz
Project: Onboard Visual Foundation Models for Mars Terrain Perception and Rover Navigation
"""

import csv
import os
import time

import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize
from transformers import AutoImageProcessor, AutoModel

# ── Config ─────────────────────────────────────────────────────────────────────
MODEL_ID    = "facebook/dinov2-with-registers-small"   # thesis deployment model, 90.24%
LOGR_C      = 0.316
N_SHOT      = 1000
CLASS_NAMES = ["soil", "bedrock", "sand", "big_rock"]
N_TEST_MAX  = 200   # max test images for accuracy comparison (more = slower but more reliable)

_HERE       = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(_HERE, "results")
CACHE_DIR   = os.path.join(RESULTS_DIR, "feature_cache")

FP32_ONNX   = os.path.join(RESULTS_DIR, "dinov2_reg_small_encoder.onnx")
INT8_ONNX   = os.path.join(RESULTS_DIR, "dinov2_reg_small_encoder_int8.onnx")
PROBE_NPZ   = os.path.join(RESULTS_DIR, "dinov2_reg_small_probe.npz")
TRAIN_CACHE = os.path.join(CACHE_DIR, "dinov2_reg_small_train_1000shot.npz")
OUTPUT_CSV  = os.path.join(RESULTS_DIR, "b1_accuracy_comparison_pc.csv")

AI4MARS_BASE = "/mnt/c/Users/DELL/Desktop/Thesis/github source/ai4mars-dataset-merged-0.1"
IMAGES_DIR   = os.path.join(AI4MARS_BASE, "msl/images/edr")
TEST_LABELS  = os.path.join(AI4MARS_BASE, "msl/labels/test/masked-gold-min3-100agree")

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ── ONNX wrapper ───────────────────────────────────────────────────────────────

class _DINOv2RegEncoder(torch.nn.Module):
    """Wraps HuggingFace DINOv2-reg: pixel_values → L2-normalised CLS token."""

    def __init__(self, hf_model):
        super().__init__()
        self.model = hf_model

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        out = self.model(pixel_values=pixel_values)
        cls = out.last_hidden_state[:, 0, :]          # (B, 384) CLS token
        return cls / cls.norm(dim=-1, keepdim=True)   # L2 normalise


# ── Step 1: Export FP32 ONNX ──────────────────────────────────────────────────

def export_fp32_onnx() -> None:
    if os.path.exists(FP32_ONNX):
        print(f"[1/4] FP32 ONNX already exists ({os.path.getsize(FP32_ONNX)/1e6:.1f} MB) — skipping export")
        return

    print(f"[1/4] Exporting {MODEL_ID} → FP32 ONNX …")
    t0    = time.perf_counter()
    model = AutoModel.from_pretrained(MODEL_ID)
    model.eval()
    encoder = _DINOv2RegEncoder(model)
    dummy   = torch.zeros(1, 3, 224, 224)

    # DINOv2-with-registers traces a bicubic AA interpolation path for position
    # embeddings even when input is 224×224 (identity op — no actual resize).
    # PyTorch's TorchScript ONNX exporter has no symbolic for _upsample_bicubic2d_aa,
    # so we temporarily strip the antialias kwarg during tracing only.
    import torch.nn.functional as _F
    _orig_interpolate = _F.interpolate
    def _interpolate_no_aa(x, *args, **kwargs):
        kwargs.pop("antialias", None)
        return _orig_interpolate(x, *args, **kwargs)
    _F.interpolate = _interpolate_no_aa
    try:
        torch.onnx.export(
            encoder,
            dummy,
            FP32_ONNX,
            input_names=["pixel_values"],
            output_names=["features"],
            dynamic_axes={"pixel_values": {0: "batch"}, "features": {0: "batch"}},
            opset_version=17,
            do_constant_folding=True,
        )
    finally:
        _F.interpolate = _orig_interpolate
    size_mb = os.path.getsize(FP32_ONNX) / 1e6
    print(f"  Done in {time.perf_counter()-t0:.1f}s  →  {FP32_ONNX}  ({size_mb:.1f} MB)")

    # Sanity check
    import onnxruntime as ort
    sess = ort.InferenceSession(FP32_ONNX, providers=["CPUExecutionProvider"])
    out  = sess.run(None, {"pixel_values": dummy.numpy()})[0]
    assert out.shape == (1, 384), f"Unexpected shape: {out.shape}"
    norm = float(np.linalg.norm(out[0]))
    assert abs(norm - 1.0) < 1e-3, f"Not L2-normalised (norm={norm:.4f})"
    print(f"  Sanity check PASSED — shape {out.shape}, L2-norm {norm:.4f}")


# ── Step 2: INT8 Dynamic Quantization ────────────────────────────────────────

def quantize_int8() -> None:
    if os.path.exists(INT8_ONNX):
        print(f"[2/4] INT8 ONNX already exists ({os.path.getsize(INT8_ONNX)/1e6:.1f} MB) — skipping")
        return

    print("[2/4] Applying INT8 dynamic quantization …")
    from onnxruntime.quantization import quantize_dynamic, QuantType
    t0 = time.perf_counter()
    # Quantize only MatMul/Gemm (transformer attention ops) — skip Conv.
    # ConvInteger is not implemented in onnxruntime CPU EP, and the patch-embedding
    # Conv runs only once per image so it contributes negligible latency.
    quantize_dynamic(
        model_input=FP32_ONNX,
        model_output=INT8_ONNX,
        weight_type=QuantType.QInt8,
        op_types_to_quantize=["MatMul", "Gemm"],
    )
    size_mb = os.path.getsize(INT8_ONNX) / 1e6
    fp32_mb = os.path.getsize(FP32_ONNX) / 1e6
    ratio   = fp32_mb / size_mb
    print(f"  Done in {time.perf_counter()-t0:.1f}s")
    print(f"  FP32: {fp32_mb:.1f} MB  →  INT8: {size_mb:.1f} MB  ({ratio:.1f}× smaller)")


# ── Step 3: Train + save LogReg probe ─────────────────────────────────────────
# Probe is serialised as .npz (coef + intercept + classes arrays) — no pickle needed.

def _probe_from_npz(path: str) -> LogisticRegression:
    """Reconstruct a fitted LogisticRegression from saved numpy arrays."""
    data = np.load(path)
    clf = LogisticRegression()
    clf.coef_      = data["coef"]
    clf.intercept_ = data["intercept"]
    clf.classes_   = data["classes"]
    return clf


def build_probe() -> LogisticRegression:
    if os.path.exists(PROBE_NPZ):
        print(f"[3/4] Loading existing probe from {PROBE_NPZ}")
        return _probe_from_npz(PROBE_NPZ)

    print(f"[3/4] Training LogReg probe from {TRAIN_CACHE} …")
    if not os.path.exists(TRAIN_CACHE):
        raise FileNotFoundError(
            f"Feature cache not found: {TRAIN_CACHE}\n"
            "Run dinov2_terrain_test.py with the reg-small model first."
        )
    data   = np.load(TRAIN_CACHE)
    feats  = data["feats"]
    labels = data["labels"]

    idx = []
    for c in np.unique(labels):
        c_idx = np.where(labels == c)[0]
        idx.extend(c_idx[:N_SHOT].tolist() if len(c_idx) >= N_SHOT else c_idx.tolist())
    idx = np.array(idx)

    X  = normalize(feats[idx], norm="l2")
    y  = labels[idx]
    lc = {CLASS_NAMES[k]: v for k, v in zip(*np.unique(y, return_counts=True))}
    print(f"  Training on {len(idx)} samples: {lc}")

    t0  = time.perf_counter()
    clf = LogisticRegression(C=LOGR_C, max_iter=1000, solver="lbfgs", random_state=42)
    clf.fit(X, y)
    print(f"  Trained in {time.perf_counter()-t0:.1f}s")

    np.savez(PROBE_NPZ, coef=clf.coef_, intercept=clf.intercept_, classes=clf.classes_)
    print(f"  Saved → {PROBE_NPZ}")
    return clf


# ── Step 4: Accuracy comparison on AI4Mars test set ───────────────────────────

def _preprocess(img_path: str) -> np.ndarray:
    img = Image.open(img_path).convert("RGB").resize((224, 224))
    x   = np.array(img, dtype=np.float32) / 255.0
    x   = (x - _MEAN) / _STD
    return x.transpose(2, 0, 1)[np.newaxis, :]   # (1, 3, 224, 224)


def _label_from_mask(mask_path: str) -> int:
    """Majority AI4Mars class (0-3) from label mask PNG. Returns -1 if unlabelled."""
    mask  = np.array(Image.open(mask_path).convert("L"))
    valid = mask[mask < 4]   # 0=soil 1=bedrock 2=sand 3=big_rock; 255=no_label
    return int(np.bincount(valid).argmax()) if len(valid) > 0 else -1


def _load_test_pairs(n_max: int):
    """Return list of (img_path, gt_label) from AI4Mars gold test set."""
    if not os.path.isdir(IMAGES_DIR) or not os.path.isdir(TEST_LABELS):
        print(f"  [WARN] AI4Mars not found — skipping accuracy test")
        return []
    pairs = []
    for fname in sorted(os.listdir(TEST_LABELS)):
        if not fname.endswith("_merged.png"):
            continue
        mask_path = os.path.join(TEST_LABELS, fname)
        stem      = fname.replace("_merged.png", "")
        img_path  = None
        for ext in (".JPG", ".jpg", ".png", ".PNG"):
            p = os.path.join(IMAGES_DIR, stem + ext)
            if os.path.exists(p):
                img_path = p
                break
        if img_path is None:
            continue
        gt = _label_from_mask(mask_path)
        if gt < 0:
            continue
        pairs.append((img_path, gt))
        if len(pairs) >= n_max:
            break
    return pairs


def accuracy_comparison(clf: LogisticRegression) -> dict:
    print(f"[4/4] Comparing FP32 vs INT8 accuracy on AI4Mars test set (up to {N_TEST_MAX} images) …")
    import onnxruntime as ort

    pairs = _load_test_pairs(N_TEST_MAX)
    if not pairs:
        print("  No test images found — accuracy comparison skipped")
        return {}

    sess_fp32 = ort.InferenceSession(FP32_ONNX, providers=["CPUExecutionProvider"])
    sess_int8 = ort.InferenceSession(INT8_ONNX, providers=["CPUExecutionProvider"])

    fp32_correct = 0
    int8_correct = 0
    fp32_agree   = 0   # times both models agree
    n            = len(pairs)
    per_class_fp32 = {c: [0, 0] for c in CLASS_NAMES}   # [correct, total]
    per_class_int8 = {c: [0, 0] for c in CLASS_NAMES}

    print(f"  Running on {n} test images …")
    for i, (img_path, gt) in enumerate(pairs):
        x = _preprocess(img_path)
        gt_name = CLASS_NAMES[gt]

        feat_fp32 = sess_fp32.run(None, {"pixel_values": x})[0]
        feat_int8 = sess_int8.run(None, {"pixel_values": x})[0]

        pred_fp32 = int(clf.predict(feat_fp32)[0])
        pred_int8 = int(clf.predict(feat_int8)[0])

        if pred_fp32 == gt:
            fp32_correct += 1
            per_class_fp32[gt_name][0] += 1
        per_class_fp32[gt_name][1] += 1

        if pred_int8 == gt:
            int8_correct += 1
            per_class_int8[gt_name][0] += 1
        per_class_int8[gt_name][1] += 1

        if pred_fp32 == pred_int8:
            fp32_agree += 1

        if (i + 1) % 50 == 0:
            print(f"    {i+1}/{n}  FP32 acc={fp32_correct/(i+1)*100:.1f}%  "
                  f"INT8 acc={int8_correct/(i+1)*100:.1f}%")

    acc_fp32 = fp32_correct / n * 100
    acc_int8 = int8_correct / n * 100
    agree    = fp32_agree / n * 100

    print(f"\n  FP32 accuracy:  {acc_fp32:.2f}%")
    print(f"  INT8 accuracy:  {acc_int8:.2f}%")
    print(f"  Accuracy drop:  {acc_fp32 - acc_int8:+.2f} pp")
    print(f"  Agreement:      {agree:.1f}% (images where FP32 and INT8 predict same class)")

    return {
        "n_images":   n,
        "acc_fp32":   round(acc_fp32, 2),
        "acc_int8":   round(acc_int8, 2),
        "acc_drop_pp": round(acc_fp32 - acc_int8, 2),
        "agreement_pct": round(agree, 1),
        "per_class_fp32": {c: (v[0]/v[1]*100 if v[1] else 0) for c, v in per_class_fp32.items()},
        "per_class_int8": {c: (v[0]/v[1]*100 if v[1] else 0) for c, v in per_class_int8.items()},
    }


# ── Save CSV ──────────────────────────────────────────────────────────────────

def save_csv(acc_results: dict) -> None:
    fp32_mb = os.path.getsize(FP32_ONNX) / 1e6 if os.path.exists(FP32_ONNX) else 0
    int8_mb = os.path.getsize(INT8_ONNX) / 1e6 if os.path.exists(INT8_ONNX) else 0

    rows = [
        {
            "model":       "DINOv2+reg ViT-S FP32 ONNX",
            "precision":   "FP32",
            "size_mb":     round(fp32_mb, 1),
            "acc_pc_pct":  acc_results.get("acc_fp32", "N/A"),
            "note":        "baseline deployment model",
        },
        {
            "model":       "DINOv2+reg ViT-S INT8 ONNX",
            "precision":   "INT8",
            "size_mb":     round(int8_mb, 1),
            "acc_pc_pct":  acc_results.get("acc_int8", "N/A"),
            "note":        "dynamic quantization (weights INT8, activations FP32)",
        },
    ]

    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nCSV saved → {OUTPUT_CSV}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 65)
    print("B1: INT8 Quantization — DINOv2+reg ViT-S/14")
    print("=" * 65)

    export_fp32_onnx()
    quantize_int8()
    clf = build_probe()
    acc = accuracy_comparison(clf)

    fp32_mb = os.path.getsize(FP32_ONNX) / 1e6
    int8_mb = os.path.getsize(INT8_ONNX) / 1e6

    print("\n" + "=" * 65)
    print("SUMMARY (PC — size and accuracy only; latency measured on RPi)")
    print(f"  FP32 ONNX:  {fp32_mb:.1f} MB  |  accuracy: {acc.get('acc_fp32', 'N/A')}")
    print(f"  INT8 ONNX:  {int8_mb:.1f} MB  |  accuracy: {acc.get('acc_int8', 'N/A')}")
    print(f"  Size reduction: {fp32_mb/int8_mb:.1f}×")
    if acc:
        print(f"  Accuracy drop:  {acc['acc_drop_pp']:+.2f} pp")
        print(f"  Agreement:      {acc['agreement_pct']:.1f}%")

    save_csv(acc)

    print("\nFiles to copy to RPi:")
    print(f"  {FP32_ONNX}")
    print(f"  {INT8_ONNX}")
    print(f"  {PROBE_NPZ}")
    print("\nThen on RPi, run:")
    print("  python3 int8_benchmark_rpi.py")
    print("=" * 65)


if __name__ == "__main__":
    main()
