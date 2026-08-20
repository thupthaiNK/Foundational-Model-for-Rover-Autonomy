"""
Purpose: Compute Wilson score 95% confidence intervals for this thesis's headline
         accuracy figures, using each result's own reported test-set size (n).
         Post-hoc statistical analysis on already-reported numbers -- no model
         re-run, no new inference.
Inputs: Hard-coded (accuracy_pct, n) pairs taken directly from each experiment's
        own results file/section, cited per row below.
Outputs: experiments/results/confidence_intervals.csv
How to run: python experiments/confidence_intervals.py
Project: Onboard Visual Foundation Models for Mars Terrain Perception and Rover Navigation
"""

import csv
import math

# Wilson score interval for a binomial proportion (Wilson, 1927).
# Preferred over the normal (Wald) approximation at small n or accuracy
# near 0%/100%, both of which occur among these headline results
# (Exp 5b n=20; Earth+Mars earth->mars near-chance/below-chance).
def wilson_interval(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z ** 2 / n
    centre = p + z ** 2 / (2 * n)
    margin = z * math.sqrt((p * (1 - p) + z ** 2 / (4 * n)) / n)
    lo = (centre - margin) / denom
    hi = (centre + margin) / denom
    return (max(0.0, lo) * 100, min(1.0, hi) * 100)


# (label, accuracy_pct, n, source section)
ROWS = [
    ("CLIP zero-shot (v1)", 34.8, 287, "§4.2, Exp 1"),
    ("CLIP prompt engineering (v9)", 54.4, 287, "§4.3, Exp 2"),
    ("DINOv2+reg ViT-S 1000-shot (deployed)", 90.24, 287, "§4.7.21, Exp 7"),
    ("DINOv2 ViT-L 1000-shot", 93.73, 287, "§4.7.21, Exp 7"),
    ("Ensemble B (DINOv2 ViT-L+ViT-B)", 94.43, 287, "§4.7.21/§4.7.28, Exp 7"),
    ("Supervised DeepLabv3+ ceiling", 96.67, 287, "Swan et al. 2021, ceiling"),
    ("Mars-Bench zero-shot transfer", 24.94, 1594, "§4.7.32, Exp 10"),
    ("Mars-Bench in-domain probe", 84.07, 1594, "§4.7.33, Exp 10"),
    ("E1 joint probe @ AI4Mars (mapped)", 86.41, 810, "§4.7.34, Exp E1"),
    ("E1 joint probe @ Mars-Bench (mapped)", 91.98, 810, "§4.7.34, Exp E1"),
    ("Real ExoMy hardware (Exp 5b)", 20.0, 20, "§4.8.12, Exp 5b"),
    ("Earth+Mars: mars_only@mars (sanity gate)", 97.19, 498, "§4.8.34, Exp H22"),
    ("Earth+Mars: earth_only@earth (sanity gate)", 96.68, 512, "§4.8.34, Exp H22"),
    ("Earth+Mars: earth_only@mars (transfer)", 13.45, 498, "§4.8.34, Exp H22"),
    ("Earth+Mars: mars_only@earth (transfer)", 35.74, 512, "§4.8.34, Exp H22"),
    ("Earth+Mars: joint@mars", 96.39, 498, "§4.8.34, Exp H22"),
    ("Earth+Mars: joint@earth", 95.31, 512, "§4.8.34, Exp H22"),
]


def main():
    out_rows = []
    for label, acc_pct, n, source in ROWS:
        k = round(acc_pct / 100 * n)
        lo, hi = wilson_interval(k, n)
        half_width = (hi - lo) / 2
        out_rows.append(
            {
                "result": label,
                "accuracy_pct": acc_pct,
                "n": n,
                "k_correct": k,
                "ci95_lower_pct": round(lo, 2),
                "ci95_upper_pct": round(hi, 2),
                "half_width_pp": round(half_width, 2),
                "source": source,
            }
        )
        print(
            f"{label:45s} {acc_pct:6.2f}% (n={n:5d})  "
            f"95% CI [{lo:6.2f}, {hi:6.2f}]  (+/-{half_width:.2f} pp)"
        )

    out_path = "experiments/results/confidence_intervals.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
