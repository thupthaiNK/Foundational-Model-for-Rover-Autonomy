"""
Purpose:    Export DINOv2 ViT-S/14 visual encoder to ONNX format and save the
            pre-trained LogReg terrain probe as a .pkl file. The two output files
            together constitute the RPi-deployable DINOv2 pipeline and are
            consumed by rpi_benchmark.py --models dinov2_pytorch dinov2_onnx.
Inputs:     experiments/results/feature_cache/dinov2_train_1000shot.npz
Outputs:    experiments/results/dinov2_small_encoder.onnx  (~86 MB)
            experiments/results/dinov2_logreg_probe.pkl    (~1 KB)
How to run:
    python3 experiments/export_dinov2_onnx.py
    # Then copy both output files to RPi:
    #   scp experiments/results/dinov2_small_encoder.onnx pi@<rpi-ip>:~/
    #   scp experiments/results/dinov2_logreg_probe.pkl   pi@<rpi-ip>:~/
Project: Onboard Visual Foundation Models for Mars Terrain Perception and Rover Navigation
"""

import os
import pickle
import time

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize
from transformers import AutoImageProcessor, AutoModel

# ── Config ─────────────────────────────────────────────────────────────────────
DINOV2_MODEL_ID = "facebook/dinov2-small"   # 384-d CLS token, 21M params
LOGR_C          = 0.316                      # matches dinov2_terrain_node.py
N_SHOT          = 1000
CLASS_NAMES     = ["soil", "bedrock", "sand", "big_rock"]

_HERE        = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR  = os.path.join(_HERE, "results")
CACHE_PATH   = os.path.join(RESULTS_DIR, "feature_cache", "dinov2_train_1000shot.npz")
ONNX_PATH    = os.path.join(RESULTS_DIR, "dinov2_small_encoder.onnx")
PROBE_PATH   = os.path.join(RESULTS_DIR, "dinov2_logreg_probe.pkl")


# ── ONNX wrapper ───────────────────────────────────────────────────────────────

class _DINOv2Encoder(torch.nn.Module):
    """Wraps HuggingFace DINOv2: pixel_values → L2-normalised CLS token."""

    def __init__(self, hf_model):
        super().__init__()
        self.model = hf_model

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        out = self.model(pixel_values=pixel_values)
        cls = out.last_hidden_state[:, 0, :]          # (B, 384)
        return cls / cls.norm(dim=-1, keepdim=True)   # L2 normalise


# ── Step 1: ONNX export ────────────────────────────────────────────────────────

def export_onnx() -> None:
    print(f"[1/2] Loading {DINOV2_MODEL_ID} …")
    t0 = time.perf_counter()
    model = AutoModel.from_pretrained(DINOV2_MODEL_ID)
    model.eval()
    encoder = _DINOv2Encoder(model)
    print(f"  Model loaded in {time.perf_counter()-t0:.1f}s")

    dummy = torch.zeros(1, 3, 224, 224)

    print(f"  Exporting to ONNX …")
    t0 = time.perf_counter()
    torch.onnx.export(
        encoder,
        dummy,
        ONNX_PATH,
        input_names=["pixel_values"],
        output_names=["features"],
        dynamic_axes={
            "pixel_values": {0: "batch_size"},
            "features":     {0: "batch_size"},
        },
        opset_version=17,
        do_constant_folding=True,
    )
    elapsed = time.perf_counter() - t0
    size_mb = os.path.getsize(ONNX_PATH) / 1e6
    print(f"  Done in {elapsed:.1f}s  →  {ONNX_PATH}  ({size_mb:.1f} MB)")

    # Quick sanity check with ONNX Runtime
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
        out = sess.run(None, {"pixel_values": dummy.numpy()})[0]
        assert out.shape == (1, 384), f"Unexpected output shape: {out.shape}"
        norm = float(np.linalg.norm(out[0]))
        assert abs(norm - 1.0) < 1e-4, f"Output not L2-normalised (norm={norm:.4f})"
        print(f"  ONNX sanity check PASSED — output shape {out.shape}, L2-norm {norm:.4f}")
    except ImportError:
        print("  onnxruntime not installed — skipping sanity check")


# ── Step 2: Train + save LogReg probe ─────────────────────────────────────────

def export_probe() -> None:
    print(f"\n[2/2] Building LogReg probe from feature cache …")
    if not os.path.exists(CACHE_PATH):
        raise FileNotFoundError(
            f"Feature cache not found: {CACHE_PATH}\n"
            "Run dinov2_terrain_test.py first to generate it."
        )

    data   = np.load(CACHE_PATH)
    feats  = data["feats"]    # (N, 384)
    labels = data["labels"]   # (N,) int

    # Sample N_SHOT per class — identical protocol to dinov2_terrain_node.py
    idx = []
    for c in np.unique(labels):
        c_idx  = np.where(labels == c)[0]
        chosen = c_idx[:N_SHOT] if len(c_idx) >= N_SHOT else c_idx
        idx.extend(chosen.tolist())
    idx = np.array(idx)

    X = normalize(feats[idx], norm="l2")
    y = labels[idx]

    label_counts = dict(zip(*np.unique(y, return_counts=True)))
    print(f"  Training on {len(idx)} samples: {label_counts}")

    t0  = time.perf_counter()
    clf = LogisticRegression(C=LOGR_C, max_iter=1000, solver="lbfgs", random_state=42)
    clf.fit(X, y)
    print(f"  LogReg trained in {time.perf_counter()-t0:.1f}s")

    with open(PROBE_PATH, "wb") as f:
        pickle.dump(clf, f)
    size_kb = os.path.getsize(PROBE_PATH) / 1e3
    print(f"  Probe saved → {PROBE_PATH}  ({size_kb:.1f} KB)")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)

    export_onnx()
    export_probe()

    print("\n" + "=" * 60)
    print("Export complete. Copy both files to RPi:")
    print(f"  scp {ONNX_PATH}  pi@<rpi-ip>:~/")
    print(f"  scp {PROBE_PATH} pi@<rpi-ip>:~/")
    print("\nOn RPi, benchmark with:")
    print("  python3 rpi_benchmark.py --models dinov2_pytorch dinov2_onnx --n 10")
    print("  python3 rpi_benchmark.py --models clip onnx dinov2_pytorch dinov2_onnx --n 10")
    print("=" * 60)


if __name__ == "__main__":
    main()
