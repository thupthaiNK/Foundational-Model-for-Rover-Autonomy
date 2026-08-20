"""
Purpose:    Evaluate Apple AIMv2-L (aimv2-large-patch14-224, 307M params, 1024-d) on
            AI4Mars 287-image test set using 1000-shot frozen linear probe.
            AIMv2 uses multimodal autoregressive pretraining, achieving 89.5% ImageNet
            frozen-trunk accuracy vs DINOv2-g's 83.0% — testing whether this translates
            to better Mars terrain features despite different pretraining objective.
            Hypothesis: AIMv2's richer autoregressive feature may close gap vs DINOv2 ViT-L.
Inputs:     AI4Mars train labels + gold-standard test set (287 images)
            Model: apple/aimv2-large-patch14-224 (trust_remote_code=True)
Outputs:    experiments/results/aimv2_terrain_few_shot.csv
            experiments/results/feature_cache/aimv2_large_{train,test}_*.npy
            Printed per-class accuracy vs DINOv2 ViT-L (93.73%) and Ensemble B (94.43%)
How to run:
    python3 -u experiments/aimv2_terrain_test.py | tee /tmp/aimv2_log.txt
Project: Onboard Visual Foundation Models for Mars Terrain Perception and Rover Navigation
"""

import csv
import os
import re
import random
import time

import numpy as np
import torch
from PIL import Image
from safetensors import safe_open
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize
from torchvision import transforms
from transformers import AutoImageProcessor

# ── Paths ─────────────────────────────────────────────────────────────────────
AI4MARS_BASE = "/mnt/c/Users/DELL/Desktop/Thesis/github source/ai4mars-dataset-merged-0.1"
IMAGES_DIR   = os.path.join(AI4MARS_BASE, "msl/images/edr")
TRAIN_LABELS = os.path.join(AI4MARS_BASE, "msl/labels/train")
TEST_LABELS  = os.path.join(AI4MARS_BASE, "msl/labels/test/masked-gold-min3-100agree")
RESULTS_DIR  = os.path.join(os.path.dirname(__file__), "results")
CACHE_DIR    = os.path.join(RESULTS_DIR, "feature_cache")

MODEL_ID    = "apple/aimv2-large-patch14-224"
PREFIX      = "aimv2_large"
CLASS_NAMES = ["soil", "bedrock", "sand", "big_rock"]
IGNORE_PX   = 255
SEED        = 42
SHOTS_LIST  = [10, 100, 1000]

BASELINES = {
    "dinov2_vitl_1000":  {"overall": 93.73, "bedrock": 90.94},
    "ensemble_b_1000":   {"overall": 94.43, "bedrock": 91.32},
}


# ── Data loading ───────────────────────────────────────────────────────────────

def dominant_class(label_path):
    lbl = np.array(Image.open(label_path))
    valid = lbl[lbl != IGNORE_PX]
    if len(valid) == 0:
        return None
    return int(np.argmax(np.bincount(valid, minlength=4)))


def load_split(label_dir):
    pairs = []
    for fname in sorted(os.listdir(label_dir)):
        if not fname.endswith(".png"):
            continue
        stem     = fname.replace("_merged.png", "").replace(".png", "")
        img_path = os.path.join(IMAGES_DIR, stem + ".JPG")
        lbl_path = os.path.join(label_dir, fname)
        if not os.path.exists(img_path):
            continue
        gt = dominant_class(lbl_path)
        if gt is not None:
            pairs.append((img_path, gt))
    return pairs


def sample_n_per_class(pairs, n_per_class, seed=SEED):
    rng = random.Random(seed)
    by_class = {c: [] for c in range(len(CLASS_NAMES))}
    for pair in pairs:
        by_class[pair[1]].append(pair)
    sampled = []
    for c in range(len(CLASS_NAMES)):
        pool = by_class[c]
        rng.shuffle(pool)
        sampled.extend(pool[:n_per_class])
    return sampled


# ── Model loading (manual key-remap required) ──────────────────────────────────
# The apple/aimv2-large-patch14-224 checkpoint uses encoder.layers.* / separate q,k,v,
# but the published modeling_aimv2.py expects trunk.blocks.* / combined qkv.
# We remap checkpoint keys → model keys when loading the state dict.

CKPT_PATH = (
    "/home/thupthai/.cache/huggingface/hub/"
    "models--apple--aimv2-large-patch14-224/snapshots/"
    "fcb5093a9c5b3efb7db5acee213849967fd18210/model.safetensors"
)

# ImageNet normalisation used by AIMv2
TRANSFORM = transforms.Compose([
    transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])


def remap_aimv2_checkpoint(ckpt_path):
    """
    Remap checkpoint keys (encoder.layers.* / separate q,k,v) to
    AIMv2Model keys (trunk.blocks.* / combined qkv).

    Key mappings:
      embeddings.patch_embed.weight    → preprocessor.patchifier.proj.weight
      embeddings.patch_embed.bias      → preprocessor.patchifier.proj.bias
      embeddings.rms_norm.weight       → preprocessor.patchifier.norm.weight
      embeddings.position_embedding.weight → preprocessor.pos_embed  (unsqueeze 0)
      rms_norm.weight                  → trunk.post_trunk_norm.weight
      encoder.layers.X.rms_norm1.weight → trunk.blocks.X.norm_1.weight
      encoder.layers.X.rms_norm2.weight → trunk.blocks.X.norm_2.weight
      encoder.layers.X.attention.q_proj.weight }
      encoder.layers.X.attention.k_proj.weight } → trunk.blocks.X.attn.qkv.weight (concat)
      encoder.layers.X.attention.v_proj.weight }
      encoder.layers.X.attention.out_proj.weight → trunk.blocks.X.attn.proj.weight
      encoder.layers.X.ffn.gate_proj.weight     → trunk.blocks.X.mlp.fc1.weight
      encoder.layers.X.ffn.up_proj.weight       → trunk.blocks.X.mlp.fc3.weight
      encoder.layers.X.ffn.down_proj.weight     → trunk.blocks.X.mlp.fc2.weight
    """
    print(f"  Loading safetensors from: {ckpt_path}", flush=True)
    raw = {}
    with safe_open(ckpt_path, framework="pt") as f:
        for key in f.keys():
            raw[key] = f.get_tensor(key)

    # Collect q,k,v tensors per layer to merge later
    qkv_parts = {}  # {layer_idx: {'q': T, 'k': T, 'v': T}}
    sd = {}

    for key, tensor in raw.items():
        # ── Embeddings ──
        if key == "embeddings.patch_embed.weight":
            sd["preprocessor.patchifier.proj.weight"] = tensor
        elif key == "embeddings.patch_embed.bias":
            sd["preprocessor.patchifier.proj.bias"] = tensor
        elif key == "embeddings.rms_norm.weight":
            sd["preprocessor.patchifier.norm.weight"] = tensor
        elif key == "embeddings.position_embedding.weight":
            # Model expects (1, N, D); checkpoint has (N, D)
            sd["preprocessor.pos_embed"] = tensor.unsqueeze(0)

        # ── Final norm ──
        elif key == "rms_norm.weight":
            sd["trunk.post_trunk_norm.weight"] = tensor

        # ── Layer-wise ──
        else:
            m = re.match(r"encoder\.layers\.(\d+)\.(.*)", key)
            if not m:
                print(f"  [skip] {key}", flush=True)
                continue
            idx, rest = int(m.group(1)), m.group(2)
            prefix = f"trunk.blocks.{idx}"

            if rest == "rms_norm1.weight":
                sd[f"{prefix}.norm_1.weight"] = tensor
            elif rest == "rms_norm2.weight":
                sd[f"{prefix}.norm_2.weight"] = tensor
            elif rest == "attention.out_proj.weight":
                sd[f"{prefix}.attn.proj.weight"] = tensor
            elif rest == "attention.q_proj.weight":
                qkv_parts.setdefault(idx, {})["q"] = tensor
            elif rest == "attention.k_proj.weight":
                qkv_parts.setdefault(idx, {})["k"] = tensor
            elif rest == "attention.v_proj.weight":
                qkv_parts.setdefault(idx, {})["v"] = tensor
            elif rest == "ffn.gate_proj.weight":
                sd[f"{prefix}.mlp.fc1.weight"] = tensor   # SwiGLU gate
            elif rest == "ffn.up_proj.weight":
                sd[f"{prefix}.mlp.fc3.weight"] = tensor   # SwiGLU up
            elif rest == "ffn.down_proj.weight":
                sd[f"{prefix}.mlp.fc2.weight"] = tensor   # SwiGLU down
            else:
                print(f"  [skip] {key}", flush=True)

    # Merge q,k,v → combined qkv (row-concat)
    for idx, parts in qkv_parts.items():
        qkv = torch.cat([parts["q"], parts["k"], parts["v"]], dim=0)
        sd[f"trunk.blocks.{idx}.attn.qkv.weight"] = qkv

    print(f"  Remapped {len(sd)} keys ({len(qkv_parts)} qkv merges)", flush=True)
    return sd


def _import_aimv2_classes():
    """Dynamically locate the cached AIMv2 module (revision-hash-agnostic)."""
    import importlib, sys
    cache_base = os.path.expanduser(
        "~/.cache/huggingface/modules/transformers_modules/apple/"
        "aimv2_hyphen_large_hyphen_patch14_hyphen_224"
    )
    rev = next(
        d for d in sorted(os.listdir(cache_base))
        if os.path.isdir(os.path.join(cache_base, d)) and not d.startswith("_")
    )
    hf_modules = os.path.expanduser("~/.cache/huggingface/modules")
    if hf_modules not in sys.path:
        sys.path.insert(0, hf_modules)
    base = (f"transformers_modules.apple"
            f".aimv2_hyphen_large_hyphen_patch14_hyphen_224.{rev}")
    cfg_mod   = importlib.import_module(f"{base}.configuration_aimv2")
    model_mod = importlib.import_module(f"{base}.modeling_aimv2")
    return cfg_mod.AIMv2Config, model_mod.AIMv2Model


def load_model(device):
    AIMv2Config, AIMv2Model = _import_aimv2_classes()

    print(f"Loading {MODEL_ID} (manual key-remap)...", flush=True)
    t0     = time.perf_counter()
    config = AIMv2Config.from_pretrained(MODEL_ID, trust_remote_code=True)
    model  = AIMv2Model(config)

    sd = remap_aimv2_checkpoint(CKPT_PATH)
    missing, unexpected = model.load_state_dict(sd, strict=True)
    if missing:
        print(f"  Missing: {missing[:5]}", flush=True)
    if unexpected:
        print(f"  Unexpected: {unexpected[:5]}", flush=True)

    model = model.to(device).eval()
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Loaded {n_params:.1f}M params in {time.perf_counter()-t0:.1f}s | device={device}",
          flush=True)
    return model


BATCH_SIZE = 4  # batch inference for ~4× speedup on CPU


def extract_features(model, pairs, device, desc=""):
    features, labels = [], []
    n  = len(pairs)
    t0 = time.perf_counter()

    for batch_start in range(0, n, BATCH_SIZE):
        batch = pairs[batch_start: batch_start + BATCH_SIZE]
        batch_i = batch_start + len(batch)
        if batch_i % 200 == 0 or batch_i >= n:
            elapsed = time.perf_counter() - t0
            eta     = elapsed / batch_i * (n - batch_i) if batch_i < n else 0
            print(f"  {desc} {batch_i}/{n}  ({elapsed:.0f}s elapsed, ~{eta:.0f}s ETA)", flush=True)

        imgs  = [Image.open(p).convert("RGB") for p, _ in batch]
        pixel = torch.stack([TRANSFORM(img) for img in imgs]).to(device)
        with torch.no_grad():
            out  = model(pixel_values=pixel)
            feats = out.last_hidden_state.mean(dim=1).cpu().numpy()  # (B, 1024)

        for feat, (_, gt) in zip(feats, batch):
            features.append(feat)
            labels.append(gt)

    features = normalize(np.array(features, dtype=np.float32))
    return features, np.array(labels)


# ── Probe training ─────────────────────────────────────────────────────────────

def run_probe(tr_feats, tr_labels, te_feats, te_labels, n_shots):
    rng = np.random.RandomState(SEED)
    idx = []
    for c in range(len(CLASS_NAMES)):
        c_idx = np.where(tr_labels == c)[0]
        if len(c_idx) > 0:
            chosen = rng.choice(c_idx, size=min(n_shots, len(c_idx)), replace=False)
            idx.extend(chosen.tolist())

    clf = LogisticRegression(C=0.316, max_iter=1000, random_state=SEED,
                             multi_class="multinomial", solver="lbfgs")
    clf.fit(tr_feats[idx], tr_labels[idx])
    preds   = clf.predict(te_feats)
    overall = (preds == te_labels).mean() * 100
    per_cls = {}
    for c, name in enumerate(CLASS_NAMES):
        mask = te_labels == c
        per_cls[name] = (preds[mask] == te_labels[mask]).mean() * 100 if mask.sum() > 0 else None
    return overall, per_cls


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    device = "cpu"  # CUDA conv2d engine fails for ViT Conv2d on this machine (same as DINOv2WithRegisters)

    print("Loading AI4Mars data...", flush=True)
    train_pairs = load_split(TRAIN_LABELS)
    test_pairs  = load_split(TEST_LABELS)
    max_shots   = max(SHOTS_LIST)
    train_sample = sample_n_per_class(train_pairs, max_shots)
    print(f"  Train pool: {len(train_sample)} | Test: {len(test_pairs)}", flush=True)

    # Cache paths
    c_trf = os.path.join(CACHE_DIR, f"{PREFIX}_train_{max_shots}_feats.npy")
    c_trl = os.path.join(CACHE_DIR, f"{PREFIX}_train_{max_shots}_labels.npy")
    c_tef = os.path.join(CACHE_DIR, f"{PREFIX}_test_{len(test_pairs)}_feats.npy")
    c_tel = os.path.join(CACHE_DIR, f"{PREFIX}_test_{len(test_pairs)}_labels.npy")

    if os.path.exists(c_tef) and os.path.exists(c_trf):
        print("Loading cached features...", flush=True)
        tr_feats  = np.load(c_trf); tr_labels = np.load(c_trl)
        te_feats  = np.load(c_tef); te_labels = np.load(c_tel)
    else:
        model = load_model(device)
        print("\nExtracting train features...", flush=True)
        tr_feats, tr_labels = extract_features(model, train_sample, device, "train")
        print("Extracting test features...", flush=True)
        te_feats, te_labels = extract_features(model, test_pairs, device, "test")
        np.save(c_trf, tr_feats); np.save(c_trl, tr_labels)
        np.save(c_tef, te_feats); np.save(c_tel, te_labels)
        print("Features cached.", flush=True)
        del model

    print(f"\nTrain feats: {tr_feats.shape}  Test: {te_feats.shape}", flush=True)
    feat_dim = tr_feats.shape[1]

    results = {}
    for shots in SHOTS_LIST:
        overall, per_cls = run_probe(tr_feats, tr_labels, te_feats, te_labels, shots)
        results[shots] = {"overall": overall, "per_class": per_cls}
        print(f"\n{shots}-shot:  Overall={overall:.2f}%  "
              f"Bedrock={per_cls.get('bedrock', 0):.2f}%  "
              f"Soil={per_cls.get('soil', 0):.2f}%  "
              f"Sand={per_cls.get('sand', 0):.2f}%", flush=True)

    # Summary
    print("\n" + "=" * 70)
    print(f"AIMv2-L ({feat_dim}-d)  vs  DINOv2 ViT-L (1024-d)  vs  Ensemble B (1792-d)")
    print("=" * 70)
    r1k = results[1000]
    print(f"AIMv2-L  1000-shot:  {r1k['overall']:.2f}%  "
          f"(Bedrock {r1k['per_class'].get('bedrock',0):.2f}%)")
    print(f"DINOv2 ViT-L 1000-shot: {BASELINES['dinov2_vitl_1000']['overall']:.2f}%  "
          f"(Bedrock {BASELINES['dinov2_vitl_1000']['bedrock']:.2f}%)")
    print(f"Ensemble B 1000-shot:   {BASELINES['ensemble_b_1000']['overall']:.2f}%  "
          f"(Bedrock {BASELINES['ensemble_b_1000']['bedrock']:.2f}%)")
    delta = r1k['overall'] - BASELINES['dinov2_vitl_1000']['overall']
    print(f"vs DINOv2 ViT-L: {delta:+.2f}%")
    print("=" * 70)

    # Save CSV
    csv_path = os.path.join(RESULTS_DIR, "aimv2_terrain_few_shot.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["shots", "overall", "soil", "bedrock", "sand", "feat_dim"])
        for shots in SHOTS_LIST:
            r = results[shots]
            w.writerow([shots, round(r["overall"], 4),
                        round(r["per_class"].get("soil") or 0, 4),
                        round(r["per_class"].get("bedrock") or 0, 4),
                        round(r["per_class"].get("sand") or 0, 4),
                        feat_dim])
    print(f"\nSaved CSV → {csv_path}", flush=True)


if __name__ == "__main__":
    main()
