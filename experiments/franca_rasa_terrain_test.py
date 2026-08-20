"""
Purpose:    Evaluate Franca ViT-L/14 (Venkataramanan et al., arXiv:2507.14137) with its
            RASA (positional disentanglement) head on AI4Mars 287-image test set.
            Franca uses nested Matryoshka clustering + position-debiased RASA patch tokens.
            Three feature variants tested:
              A) CLS-only (1024-d) — apple-to-apple vs DINOv2 ViT-L
              B) RASA mean-pool (1024-d) — debiased patch spatial aggregation
              C) CLS + RASA mean-pool (2048-d) — combined global+spatial
            Hypothesis: RASA debiasing removes positional noise → better texture separation.
Inputs:     AI4Mars train labels + gold-standard test set (287 images)
            Model: valeoai/Franca (franca_vitl14, Laion600M checkpoint — no In21K for ViT-L)
Outputs:    experiments/results/franca_rasa_few_shot.csv
            experiments/results/feature_cache/franca_vitl14_{cls,rasa,combined}_{train,test}_*.npy
            Summary printed vs DINOv2 ViT-L (93.73%) and Ensemble B (94.43%)
How to run:
    python3 -u experiments/franca_rasa_terrain_test.py | tee /tmp/franca_log.txt
Project: Onboard Visual Foundation Models for Mars Terrain Perception and Rover Navigation
"""

import csv
import os
import random
import time

import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize
from torchvision import transforms

# ── Paths ─────────────────────────────────────────────────────────────────────
AI4MARS_BASE = "/mnt/c/Users/DELL/Desktop/Thesis/github source/ai4mars-dataset-merged-0.1"
IMAGES_DIR   = os.path.join(AI4MARS_BASE, "msl/images/edr")
TRAIN_LABELS = os.path.join(AI4MARS_BASE, "msl/labels/train")
TEST_LABELS  = os.path.join(AI4MARS_BASE, "msl/labels/test/masked-gold-min3-100agree")
RESULTS_DIR  = os.path.join(os.path.dirname(__file__), "results")
CACHE_DIR    = os.path.join(RESULTS_DIR, "feature_cache")

SHOTS_LIST  = [10, 100, 1000]
CLASS_NAMES = ["soil", "bedrock", "sand", "big_rock"]
IGNORE_PX   = 255
SEED        = 42

BASELINES = {
    "dinov2_vitl_1000":  {"overall": 93.73, "bedrock": 90.94},
    "ensemble_b_1000":   {"overall": 94.43, "bedrock": 91.32},
}

# Use 224x224 for inference speed (Franca ViT interpolates pos_embed at runtime).
# Matches the input size used for all other models in this thesis (DINOv2, AIMv2, etc.)
# enabling direct accuracy comparison under identical input conditions.
TRANSFORM = transforms.Compose([
    transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])


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


# ── Model loading ──────────────────────────────────────────────────────────────

def load_franca(device):
    """Load Franca ViT-L/14 + RASA head.

    The hub code hardcodes n_pos_layers=9 but the released ViT-L RASA checkpoint
    was saved with n_pos_layers=8 (pre_pos_layers.0..7). We therefore:
      1. Load the backbone without RASA via torch.hub
      2. Manually build RASAHead(n_pos_layers=8) and load its weights
    """
    print("Loading Franca ViT-L/14 via torch.hub (backbone, first run ~1.3 GB)...", flush=True)
    t0 = time.perf_counter()

    # Step 1: backbone only (no RASA head)
    model = torch.hub.load(
        "valeoai/Franca",
        "franca_vitl14",
        weights="LAION",
        img_size=518,
        use_rasa_head=False,
        pretrained=True,
        force_reload=False,
    )

    # Step 2: attach RASA head manually with correct n_pos_layers=8
    import sys
    sys.path.insert(0, "/home/thupthai/.cache/torch/hub/valeoai_Franca_main")
    from rasa.src.rasa_head import RASAHead
    from torch.hub import load_state_dict_from_url

    rasa_url = "https://github.com/valeoai/Franca/releases/download/v1.0.0/franca_vitl14_Laion600M_rasa.pth"
    rasa_ckpt_path = "/home/thupthai/.cache/torch/hub/checkpoints/franca_vitl14_Laion600M_rasa.pth"
    if os.path.exists(rasa_ckpt_path):
        rasa_state = torch.load(rasa_ckpt_path, map_location="cpu", weights_only=True)
    else:
        rasa_state = load_state_dict_from_url(rasa_url, map_location="cpu", weights_only=True)

    rasa_head = RASAHead(input_dim=model.embed_dim, n_pos_layers=8, pos_out_dim=2)
    rasa_head.load_state_dict(rasa_state)
    model.rasa_head = rasa_head

    model = model.to(device).eval()
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Loaded {n_params:.1f}M params in {time.perf_counter()-t0:.1f}s | device={device}",
          flush=True)
    return model


# ── Feature extraction ─────────────────────────────────────────────────────────

def extract_features(model, pairs, device, desc=""):
    """
    Returns three feature matrices (all L2-normalised):
      feats_cls     (N, 1024)   — CLS token
      feats_rasa    (N, 1024)   — RASA mean-pooled patches
      feats_comb    (N, 2048)   — CLS ‖ RASA
    """
    cls_list, rasa_list, labels = [], [], []
    n  = len(pairs)
    t0 = time.perf_counter()

    for i, (img_path, gt) in enumerate(pairs):
        if (i + 1) % 200 == 0 or i == n - 1:
            elapsed = time.perf_counter() - t0
            eta     = elapsed / (i + 1) * (n - i - 1)
            print(f"  {desc} {i+1}/{n}  ({elapsed:.0f}s elapsed, ~{eta:.0f}s ETA)", flush=True)

        img = Image.open(img_path).convert("RGB")
        x   = TRANSFORM(img).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model.forward_features(x, use_rasa_head=True)

        cls  = out["x_norm_clstoken"].squeeze().cpu().numpy()   # (1024,)
        rasa = out["patch_token_rasa"].squeeze()                 # (256, 1024) at 224x224
        rasa = rasa.mean(dim=0).cpu().numpy()                    # (1024,) mean-pool

        cls_list.append(cls)
        rasa_list.append(rasa)
        labels.append(gt)

    cls_arr  = normalize(np.array(cls_list,  dtype=np.float32))
    rasa_arr = normalize(np.array(rasa_list, dtype=np.float32))
    comb_arr = normalize(np.concatenate([cls_arr, rasa_arr], axis=1))
    return cls_arr, rasa_arr, comb_arr, np.array(labels)


# ── Probe ─────────────────────────────────────────────────────────────────────

def run_probe(tr_feats, tr_labels, te_feats, te_labels, n_shots, tag=""):
    rng = np.random.RandomState(SEED)
    idx = []
    for c in range(len(CLASS_NAMES)):
        c_idx  = np.where(tr_labels == c)[0]
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
        per_cls[name] = (preds[mask] == te_labels[mask]).mean() * 100 if mask.sum() > 0 else 0.0
    if tag:
        print(f"    {tag}: Overall={overall:.2f}%  Bedrock={per_cls['bedrock']:.2f}%", flush=True)
    return overall, per_cls


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    # Force CPU: CUDA conv2d (patch_embed projection) fails on this machine.
    # Same issue as AIMv2 and DINOv2WithRegisters — CUDA is available but unusable.
    device = "cpu"

    print("Loading AI4Mars data...", flush=True)
    train_pairs  = load_split(TRAIN_LABELS)
    test_pairs   = load_split(TEST_LABELS)
    max_shots    = max(SHOTS_LIST)
    train_sample = sample_n_per_class(train_pairs, max_shots)
    n_test       = len(test_pairs)
    n_train      = len(train_sample)
    print(f"  Train: {n_train}  Test: {n_test}", flush=True)

    # Cache paths
    pfx = f"franca_vitl14"
    c_tr_cls  = os.path.join(CACHE_DIR, f"{pfx}_train_{n_train}_cls_feats.npy")
    c_te_cls  = os.path.join(CACHE_DIR, f"{pfx}_test_{n_test}_cls_feats.npy")
    c_tr_rasa = os.path.join(CACHE_DIR, f"{pfx}_train_{n_train}_rasa_feats.npy")
    c_te_rasa = os.path.join(CACHE_DIR, f"{pfx}_test_{n_test}_rasa_feats.npy")
    c_tr_comb = os.path.join(CACHE_DIR, f"{pfx}_train_{n_train}_comb_feats.npy")
    c_te_comb = os.path.join(CACHE_DIR, f"{pfx}_test_{n_test}_comb_feats.npy")
    c_tr_lbl  = os.path.join(CACHE_DIR, f"{pfx}_train_{n_train}_labels.npy")
    c_te_lbl  = os.path.join(CACHE_DIR, f"{pfx}_test_{n_test}_labels.npy")

    if os.path.exists(c_te_cls) and os.path.exists(c_tr_cls):
        print("Loading cached Franca features...", flush=True)
        tr_cls  = np.load(c_tr_cls);  te_cls  = np.load(c_te_cls)
        tr_rasa = np.load(c_tr_rasa); te_rasa = np.load(c_te_rasa)
        tr_comb = np.load(c_tr_comb); te_comb = np.load(c_te_comb)
        tr_lbl  = np.load(c_tr_lbl);  te_lbl  = np.load(c_te_lbl)
    else:
        model = load_franca(device)
        print("\nExtracting train features...", flush=True)
        tr_cls, tr_rasa, tr_comb, tr_lbl = extract_features(model, train_sample, device, "train")
        print("Extracting test features...", flush=True)
        te_cls, te_rasa, te_comb, te_lbl = extract_features(model, test_pairs, device, "test")
        np.save(c_tr_cls, tr_cls);   np.save(c_te_cls, te_cls)
        np.save(c_tr_rasa, tr_rasa); np.save(c_te_rasa, te_rasa)
        np.save(c_tr_comb, tr_comb); np.save(c_te_comb, te_comb)
        np.save(c_tr_lbl, tr_lbl);   np.save(c_te_lbl, te_lbl)
        print("Features cached.", flush=True)
        del model

    VARIANTS = [
        ("CLS",      tr_cls,  te_cls,  1024),
        ("RASA",     tr_rasa, te_rasa, 1024),
        ("CLS+RASA", tr_comb, te_comb, 2048),
    ]

    all_results = {}
    for variant, tr_f, te_f, fdim in VARIANTS:
        print(f"\n=== Variant: Franca-{variant} ({fdim}-d) ===", flush=True)
        shots_res = {}
        for shots in SHOTS_LIST:
            overall, per_cls = run_probe(tr_f, tr_lbl, te_f, te_lbl, shots,
                                         tag=f"{shots}-shot")
            shots_res[shots] = {"overall": overall, "per_class": per_cls, "feat_dim": fdim}
        all_results[variant] = shots_res

    # Print final summary
    print("\n" + "=" * 70)
    print("Franca ViT-L/14  (Venkataramanan et al. arXiv:2507.14137)")
    print("=" * 70)
    for variant, shots_res in all_results.items():
        r1k = shots_res[1000]
        fdim = r1k["feat_dim"]
        print(f"  Franca-{variant:10s} ({fdim}-d)  1000-shot: "
              f"Overall={r1k['overall']:.2f}%  "
              f"Bedrock={r1k['per_class']['bedrock']:.2f}%")
    print(f"  DINOv2 ViT-L (1024-d)       1000-shot: "
          f"Overall={BASELINES['dinov2_vitl_1000']['overall']:.2f}%  "
          f"Bedrock={BASELINES['dinov2_vitl_1000']['bedrock']:.2f}%")
    print(f"  Ensemble B   (1792-d)        1000-shot: "
          f"Overall={BASELINES['ensemble_b_1000']['overall']:.2f}%  "
          f"Bedrock={BASELINES['ensemble_b_1000']['bedrock']:.2f}%")

    best_var   = max(all_results.items(), key=lambda kv: kv[1][1000]["overall"])
    best_name  = best_var[0]
    best_acc   = best_var[1][1000]["overall"]
    delta_dino = best_acc - BASELINES["dinov2_vitl_1000"]["overall"]
    print(f"\nBest Franca variant: {best_name}  {best_acc:.2f}%  "
          f"(vs DINOv2 ViT-L: {delta_dino:+.2f}%)")
    print("=" * 70)

    # Save CSV
    csv_path = os.path.join(RESULTS_DIR, "franca_rasa_few_shot.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["variant", "feat_dim", "shots",
                    "overall", "soil", "bedrock", "sand"])
        for variant, shots_res in all_results.items():
            for shots in SHOTS_LIST:
                r = shots_res[shots]
                pc = r["per_class"]
                w.writerow([f"Franca_{variant}", r["feat_dim"], shots,
                            round(r["overall"], 4),
                            round(pc.get("soil") or 0, 4),
                            round(pc.get("bedrock") or 0, 4),
                            round(pc.get("sand") or 0, 4)])
    print(f"\nSaved CSV → {csv_path}", flush=True)


if __name__ == "__main__":
    main()
