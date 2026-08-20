"""
Purpose:    Statistical seed stability test for key thesis models.
            Repeats 1000-shot LogReg evaluation 5× with different random seeds to report
            mean ± std dev instead of a single point estimate.
            Answers examiner question: "Is 90.24% / 93.73% / 94.43% stable or a lucky run?"
Inputs:     Feature caches in experiments/results/feature_cache/ (no model re-run needed)
Outputs:    experiments/results/seed_stability_results.csv
            Console table with mean ± std per class and overall
How to run:
    python3 -u experiments/seed_stability_test.py
    python3 -u experiments/seed_stability_test.py --shots 100 1000
    python3 -u experiments/seed_stability_test.py --seeds 42 123 456 789 1337 2024
Project: Onboard Visual Foundation Models for Mars Terrain Perception and Rover Navigation
"""

import argparse
import csv
import os

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize

# ── Paths ─────────────────────────────────────────────────────────────────────
_HERE      = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR  = os.path.join(_HERE, "results", "feature_cache")
RESULTS_DIR = os.path.join(_HERE, "results")

CLASS_NAMES = ["soil", "bedrock", "sand", "big_rock"]
LOGR_C      = 0.316   # same as all thesis experiments

DEFAULT_SEEDS = [42, 123, 456, 789, 1337]
DEFAULT_SHOTS = [1000]


# ── Cache loaders ─────────────────────────────────────────────────────────────

def _load_npz(train_path, test_path):
    """Load old-format .npz cache (DINOv2 ViT-S)."""
    tr = np.load(train_path)
    te = np.load(test_path)
    return tr["feats"], tr["labels"], te["feats"], te["labels"]


def _load_npy(train_feats, train_labels, test_feats, test_labels):
    """Load new-format separate .npy files."""
    return (np.load(train_feats), np.load(train_labels),
            np.load(test_feats),  np.load(test_labels))


def load_dinov2_vits():
    return _load_npz(
        os.path.join(CACHE_DIR, "dinov2_train_1000shot.npz"),
        os.path.join(CACHE_DIR, "dinov2_test_1000shot.npz"),
    )


def load_dinov2_vitl():
    return _load_npy(
        os.path.join(CACHE_DIR, "dinov2_vitl_train_1000_feats.npy"),
        os.path.join(CACHE_DIR, "dinov2_vitl_train_1000_labels.npy"),
        os.path.join(CACHE_DIR, "dinov2_vitl_test_287_feats.npy"),
        os.path.join(CACHE_DIR, "dinov2_vitl_test_287_labels.npy"),
    )


def load_ensemble_b():
    """Ensemble B: concat DINOv2 ViT-L (1024-d) + DINOv2 ViT-B (768-d) = 1792-d."""
    vitl_tr_f = np.load(os.path.join(CACHE_DIR, "dinov2_vitl_train_1000_feats.npy"))
    vitl_tr_l = np.load(os.path.join(CACHE_DIR, "dinov2_vitl_train_1000_labels.npy"))
    vitl_te_f = np.load(os.path.join(CACHE_DIR, "dinov2_vitl_test_287_feats.npy"))
    vitl_te_l = np.load(os.path.join(CACHE_DIR, "dinov2_vitl_test_287_labels.npy"))

    vitb_tr_f = np.load(os.path.join(CACHE_DIR, "dinov2_vitb_train_1000_feats.npy"))
    vitb_te_f = np.load(os.path.join(CACHE_DIR, "dinov2_vitb_test_287_feats.npy"))

    train_feats = np.hstack([vitl_tr_f, vitb_tr_f])   # (N, 1792)
    test_feats  = np.hstack([vitl_te_f, vitb_te_f])   # (287, 1792)
    return train_feats, vitl_tr_l, test_feats, vitl_te_l


MODELS = {
    "dinov2_vits":  ("DINOv2 ViT-S/14  (384-d)",  load_dinov2_vits),
    "dinov2_vitl":  ("DINOv2 ViT-L/14 (1024-d)",  load_dinov2_vitl),
    "ensemble_b":   ("Ensemble B ViT-L+B (1792-d)", load_ensemble_b),
}


# ── Core evaluation ───────────────────────────────────────────────────────────

def sample_and_eval(train_feats, train_labels, test_feats, test_labels,
                    n_shot: int, seed: int) -> dict:
    """Sample n_shot per class, train LogReg, evaluate on full test set."""
    rng = np.random.default_rng(seed)

    idx = []
    for c in np.unique(train_labels):
        c_idx  = np.where(train_labels == c)[0]
        chosen = rng.choice(c_idx, size=min(n_shot, len(c_idx)), replace=False)
        idx.extend(chosen.tolist())
    idx = np.array(idx)

    X_tr = normalize(train_feats[idx], norm="l2")
    y_tr = train_labels[idx]
    X_te = normalize(test_feats,       norm="l2")
    y_te = test_labels

    clf = LogisticRegression(C=LOGR_C, max_iter=1000, solver="lbfgs",
                             random_state=seed, multi_class="multinomial")
    clf.fit(X_tr, y_tr)
    preds = clf.predict(X_te)

    results = {}
    correct_all = preds == y_te
    results["overall"] = float(correct_all.mean() * 100)

    for ci, cname in enumerate(CLASS_NAMES):
        mask = y_te == ci
        if mask.sum() > 0:
            results[cname] = float((preds[mask] == y_te[mask]).mean() * 100)
        else:
            results[cname] = float("nan")
    return results


# ── Reporting ─────────────────────────────────────────────────────────────────

def print_summary(model_name: str, shot: int, per_seed: list[dict], seeds: list[int]):
    metrics = ["overall"] + CLASS_NAMES
    print(f"\n{'─'*65}")
    print(f"  {model_name}  |  {shot}-shot  |  {len(seeds)} seeds: {seeds}")
    print(f"{'─'*65}")
    print(f"  {'Metric':<12}  {'Mean':>7}  {'Std':>6}  {'Min':>7}  {'Max':>7}")
    print(f"  {'──────':<12}  {'────':>7}  {'───':>6}  {'───':>7}  {'───':>7}")
    for m in metrics:
        vals = np.array([r[m] for r in per_seed if not np.isnan(r.get(m, float("nan")))])
        if len(vals) == 0:
            continue
        print(f"  {m:<12}  {vals.mean():>6.2f}%  {vals.std():>5.2f}  "
              f"{vals.min():>6.2f}%  {vals.max():>6.2f}%")


def save_csv(all_results: list[dict], path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not all_results:
        return
    fieldnames = list(all_results[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)
    print(f"\nCSV saved → {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Seed stability test for DINOv2 ViT-S, ViT-L, and Ensemble B. "
                    "Runs N-shot LogReg over multiple random seeds using cached features. "
                    "No model inference needed — pure numpy/sklearn."
    )
    parser.add_argument("--shots", nargs="+", type=int, default=DEFAULT_SHOTS,
                        help=f"Shot counts to test (default: {DEFAULT_SHOTS})")
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS,
                        help=f"Random seeds (default: {DEFAULT_SEEDS})")
    parser.add_argument("--models", nargs="+", choices=list(MODELS.keys()),
                        default=list(MODELS.keys()),
                        help="Models to test (default: all 3)")
    args = parser.parse_args()

    print("=== Seed Stability Test ===")
    print(f"Seeds:  {args.seeds}")
    print(f"Shots:  {args.shots}")
    print(f"Models: {args.models}")
    print(f"LogReg: C={LOGR_C}, max_iter=1000, solver=lbfgs")

    csv_rows = []

    for model_key in args.models:
        label, loader_fn = MODELS[model_key]
        print(f"\n{'═'*65}")
        print(f"Loading {label} ...")
        try:
            train_feats, train_labels, test_feats, test_labels = loader_fn()
        except FileNotFoundError as e:
            print(f"  [SKIP] Cache not found: {e}")
            continue
        print(f"  Train: {train_feats.shape}  Test: {test_feats.shape}")

        for shot in args.shots:
            per_seed = []
            for seed in args.seeds:
                r = sample_and_eval(train_feats, train_labels,
                                    test_feats, test_labels, shot, seed)
                per_seed.append(r)
                print(f"  seed={seed:5d}  overall={r['overall']:.2f}%  "
                      f"bedrock={r.get('bedrock', float('nan')):.2f}%")

            print_summary(label, shot, per_seed, args.seeds)

            # Collect CSV rows (one row per seed + summary row)
            for seed, r in zip(args.seeds, per_seed):
                row = {"model": model_key, "label": label, "shot": shot, "seed": seed}
                row.update({k: round(v, 4) for k, v in r.items()})
                csv_rows.append(row)

            # Summary row (mean ± std)
            metrics = ["overall"] + CLASS_NAMES
            summary = {"model": model_key, "label": label, "shot": shot, "seed": "mean±std"}
            for m in metrics:
                vals = np.array([r[m] for r in per_seed
                                 if not np.isnan(r.get(m, float("nan")))])
                summary[m] = f"{vals.mean():.2f}±{vals.std():.2f}" if len(vals) else "N/A"
            csv_rows.append(summary)

    out_path = os.path.join(RESULTS_DIR, "seed_stability_results.csv")
    save_csv(csv_rows, out_path)

    print("\n=== DONE ===")
    print("Use these mean±std values to replace single-run point estimates in Ch4 §4.7.")


if __name__ == "__main__":
    main()
