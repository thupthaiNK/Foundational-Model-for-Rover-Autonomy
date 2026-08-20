"""
Purpose:    Visualise DINOv2+registers ViT-S/14 self-attention maps on representative
            AI4Mars terrain images. Shows where the model focuses when classifying each
            terrain type. Uses last-layer CLS-to-patch attention averaged over all 6 heads.
Inputs:     AI4Mars gold-standard test images (1 per class: soil, bedrock, sand)
            + 1 misclassified example found by a 1000-shot LogReg probe.
            Model: facebook/dinov2-with-registers-small (frozen, same as deployed model)
Outputs:    experiments/results/figures/dinov2_attention_maps.png  (4-panel figure)
How to run:
    python3 -u experiments/dinov2_attention_maps.py
Project: Onboard Visual Foundation Models for Mars Terrain Perception and Rover Navigation
"""

import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize
from transformers import AutoImageProcessor, AutoModel

# ── Paths ─────────────────────────────────────────────────────────────────────
AI4MARS_BASE = "/mnt/c/Users/DELL/Desktop/Thesis/github source/ai4mars-dataset-merged-0.1"
IMAGES_DIR   = os.path.join(AI4MARS_BASE, "msl/images/edr")
TEST_LABELS  = os.path.join(AI4MARS_BASE, "msl/labels/test/masked-gold-min3-100agree")
CACHE_DIR    = os.path.join(os.path.dirname(__file__), "results", "feature_cache")
FIGURES_DIR  = os.path.join(os.path.dirname(__file__), "results", "figures")

MODEL_ID    = "facebook/dinov2-with-registers-small"
CLASS_NAMES = ["soil", "bedrock", "sand", "big_rock"]
IGNORE_PX   = 255
SEED        = 42


# ── Helpers ───────────────────────────────────────────────────────────────────

def dominant_class(label_path):
    lbl = np.array(Image.open(label_path))
    valid = lbl[lbl != IGNORE_PX]
    if len(valid) == 0:
        return None
    return int(np.argmax(np.bincount(valid, minlength=4)))


def load_test_pairs():
    pairs = []
    for fname in sorted(os.listdir(TEST_LABELS)):
        if not fname.endswith(".png"):
            continue
        stem = fname.replace("_merged.png", "").replace(".png", "")
        img_path = os.path.join(IMAGES_DIR, stem + ".JPG")
        lbl_path = os.path.join(TEST_LABELS, fname)
        if not os.path.exists(img_path):
            continue
        gt = dominant_class(lbl_path)
        if gt is not None:
            pairs.append((img_path, gt))
    return pairs


def load_model(device):
    print(f"Loading {MODEL_ID}...", flush=True)
    t0 = time.perf_counter()
    processor = AutoImageProcessor.from_pretrained(MODEL_ID)
    model     = AutoModel.from_pretrained(MODEL_ID, output_attentions=True).to(device).eval()
    print(f"  Loaded in {time.perf_counter()-t0:.1f}s | device={device}", flush=True)
    return model, processor


def extract_attention(model, processor, img_path, device):
    """Forward one image; return (attention_map [H,W], orig_image PIL)."""
    image  = Image.open(img_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

    # Last-layer attention: [1, heads, seq_len, seq_len]
    last_attn = outputs.attentions[-1][0]   # [heads, seq_len, seq_len]

    # CLS token → all other tokens
    cls_row = last_attn[:, 0, :]            # [heads, seq_len]

    # Infer number of patch tokens from processor crop size & patch size
    crop_h = processor.crop_size.get("height", 224)
    crop_w = processor.crop_size.get("width",  224)
    patch  = model.config.patch_size        # 14 for ViT-S/14
    n_patches = (crop_h // patch) * (crop_w // patch)   # 256 for 224×224
    grid_h    = crop_h // patch             # 16
    grid_w    = crop_w // patch             # 16

    # Skip CLS + register tokens; keep patches
    n_prefix   = cls_row.shape[1] - n_patches   # 1 CLS + 4 regs = 5
    patch_attn = cls_row[:, n_prefix:].cpu().numpy()   # [heads, 256]

    avg = patch_attn.mean(axis=0)           # [256]  — average over heads
    avg = (avg - avg.min()) / (avg.max() - avg.min() + 1e-8)  # normalise
    attn_map = avg.reshape(grid_h, grid_w)  # [16, 16]

    return attn_map, image


def overlay_attention(ax, image, attn_map, title, pred_label=None, gt_label=None):
    """Plot original image with attention heatmap overlay."""
    # Resize attention to image display size using PIL
    h, w = 224, 224
    orig_resized = image.resize((w, h), Image.LANCZOS)

    attn_pil = Image.fromarray((attn_map * 255).astype(np.uint8)).resize(
        (w, h), Image.BILINEAR
    )
    attn_np  = np.array(attn_pil) / 255.0

    ax.imshow(orig_resized)
    ax.imshow(attn_np, cmap="inferno", alpha=0.50, vmin=0, vmax=1)

    if pred_label is not None and gt_label is not None:
        color = "lime" if pred_label == gt_label else "red"
        ax.set_title(
            f"{title}\nGT: {gt_label}  Pred: {pred_label}",
            fontsize=10, color=color, fontweight="bold"
        )
    else:
        ax.set_title(title, fontsize=11, fontweight="bold")

    ax.axis("off")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    device = "cpu"   # attention extraction; CUDA conv2d engine issue on this system

    # Load test pairs (same order as feature cache)
    print("Loading test image list...", flush=True)
    pairs = load_test_pairs()
    print(f"  {len(pairs)} test images", flush=True)

    # Load cached features to find a misclassified example via LogReg
    tr_feats  = np.load(os.path.join(CACHE_DIR, "dinov2_reg_small_train_1000_feats.npy"))
    tr_labels = np.load(os.path.join(CACHE_DIR, "dinov2_reg_small_train_1000_labels.npy"))
    te_feats  = np.load(os.path.join(CACHE_DIR, "dinov2_reg_small_test_287_feats.npy"))
    te_labels = np.load(os.path.join(CACHE_DIR, "dinov2_reg_small_test_287_labels.npy"))

    clf = LogisticRegression(C=0.316, max_iter=1000, random_state=SEED,
                             multi_class="multinomial", solver="lbfgs")
    clf.fit(tr_feats, tr_labels)
    preds = clf.predict(te_feats)

    # Select display images:
    #   - 1 correct example per class (soil, bedrock, sand)
    #   - 1 misclassified example
    by_class  = {c: [] for c in range(3)}   # only 3 classes in test set
    for i, (path, gt) in enumerate(pairs):
        if gt < 3:
            by_class[gt].append(i)

    sel_correct = []
    for c in range(3):
        for idx in by_class[c]:
            if preds[idx] == te_labels[idx]:
                sel_correct.append(idx)
                break

    sel_wrong = next(
        (i for i, (p, gt) in enumerate(zip(preds, te_labels)) if p != gt),
        None
    )

    # Load model for attention extraction
    model, processor = load_model(device)

    # Build figure
    n_panels  = len(sel_correct) + (1 if sel_wrong is not None else 0)
    fig, axes = plt.subplots(1, n_panels, figsize=(4 * n_panels, 5))
    if n_panels == 1:
        axes = [axes]

    panel = 0
    for idx in sel_correct:
        img_path = pairs[idx][0]
        gt_name  = CLASS_NAMES[pairs[idx][1]]
        pred_name = CLASS_NAMES[preds[idx]]
        print(f"  Panel {panel+1}: {gt_name} ({os.path.basename(img_path)})", flush=True)
        attn_map, image = extract_attention(model, processor, img_path, device)
        overlay_attention(axes[panel], image, attn_map,
                          title=f"Terrain: {gt_name}",
                          pred_label=pred_name, gt_label=gt_name)
        panel += 1

    if sel_wrong is not None:
        img_path  = pairs[sel_wrong][0]
        gt_name   = CLASS_NAMES[te_labels[sel_wrong]]
        pred_name = CLASS_NAMES[preds[sel_wrong]]
        print(f"  Panel {panel+1}: MISCLASSIFIED {gt_name}→{pred_name} "
              f"({os.path.basename(img_path)})", flush=True)
        attn_map, image = extract_attention(model, processor, img_path, device)
        overlay_attention(axes[panel], image, attn_map,
                          title="Misclassified",
                          pred_label=pred_name, gt_label=gt_name)
        panel += 1

    # Colorbar legend
    sm = plt.cm.ScalarMappable(cmap="inferno", norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, shrink=0.6, pad=0.02)
    cbar.set_label("Attention weight (normalised)", fontsize=9)

    fig.suptitle(
        "DINOv2+registers ViT-S/14 — Last-layer CLS Attention Maps\n"
        "AI4Mars Test Images  |  Averaged over 6 attention heads",
        fontsize=12, y=1.02
    )

    out_path = os.path.join(FIGURES_DIR, "dinov2_attention_maps.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved → {out_path}", flush=True)


if __name__ == "__main__":
    main()
