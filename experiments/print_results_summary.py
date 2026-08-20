"""
Purpose: Print a combined results summary table for all terrain classification experiments.
         Reads all CSV files from experiments/results/ and prints a thesis-ready comparison.
Inputs:  experiments/results/*.csv
Outputs: Console table (copy-paste ready for thesis Chapter 4)
How to run:
    python3 experiments/print_results_summary.py
Project: Onboard Visual Foundation Models for Mars Terrain Perception and Rover Navigation
"""

import csv
import os

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

# ── Supervised baseline (from Swan et al. 2021 AI4MARS paper) ─────────────────
SUPERVISED = {
    "model": "DeepLabv3+ (supervised)",
    "overall": 96.67, "soil": 99.1, "bedrock": 94.9, "sand": 93.5,
    "speed_ms": None, "n": 287, "note": "Swan et al. 2021 — full fine-tune"
}

# ── Manual entries (experiments without structured CSV output) ─────────────────
MANUAL_ROWS = [
    {
        "model": "CLIP ViT-B/32 zero-shot v1",
        "overall": 34.8, "soil": 64.2, "bedrock": 2.2, "sand": 26.4,
        "speed_ms": 154, "n": 287, "note": "prompt_engineering v1 baseline"
    },
    {
        "model": "CLIP ViT-B/32 + prompt eng. v9",
        "overall": 54.4, "soil": 95.9, "bedrock": 18.5, "sand": 29.2,
        "speed_ms": 96, "n": 287, "note": "hybrid NAVCAM-soil + texture-bedrock"
    },
    {
        "model": "SAM2 + CLIP cascade",
        "overall": 76.7, "soil": 95.8, "bedrock": 0.0, "sand": None,
        "speed_ms": 5274, "n": 30, "note": "SAM2 adds 34x overhead, no accuracy gain"
    },
]


def load_csv_results(filename: str, model_label: str, note: str = "") -> list[dict]:
    """Load rows from a results CSV and format them consistently."""
    path = os.path.join(RESULTS_DIR, filename)
    if not os.path.exists(path):
        return []
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            q = row.get("question_set", row.get("n_shots", ""))
            rows.append({
                "model":     f"{model_label} [{q}]" if q else model_label,
                "overall":   float(row["overall"]),
                "soil":      float(row["soil"])    if row.get("soil")    not in (None, "N/A", "") else None,
                "bedrock":   float(row["bedrock"]) if row.get("bedrock") not in (None, "N/A", "") else None,
                "sand":      float(row["sand"])    if row.get("sand")    not in (None, "N/A", "") else None,
                "speed_ms":  float(row["avg_ms"])  if row.get("avg_ms")  not in (None, "N/A", "") else None,
                "n":         int(row.get("n_scored", row.get("n_images", 0)) or 0),
                "note":      note,
            })
    return rows


def fmt(val, unit="", decimals=1):
    if val is None:
        return "N/A"
    return f"{val:.{decimals}f}{unit}"


def print_table(rows: list[dict]):
    cols = ["Model", "Overall", "Soil", "Bedrock", "Sand", "ms/img", "n", "Notes"]
    widths = [42, 9, 8, 9, 8, 8, 6, 35]

    header = "  ".join(c.ljust(w) for c, w in zip(cols, widths))
    sep    = "  ".join("-" * w for w in widths)

    print("\n" + "=" * len(header))
    print("TERRAIN CLASSIFICATION RESULTS — AI4Mars dataset")
    print("=" * len(header))
    print(header)
    print(sep)

    for r in rows:
        print("  ".join([
            r["model"][:widths[0]].ljust(widths[0]),
            fmt(r["overall"], "%").rjust(widths[1]),
            fmt(r["soil"], "%").rjust(widths[2]),
            fmt(r["bedrock"], "%").rjust(widths[3]),
            fmt(r["sand"], "%").rjust(widths[4]),
            fmt(r["speed_ms"]).rjust(widths[5]),
            str(r["n"]).rjust(widths[6]),
            (r["note"] or "")[:widths[7]].ljust(widths[7]),
        ]))

    print(sep)
    print(f"  {'Supervised baseline':42}  {fmt(SUPERVISED['overall'], '%'):>9}  "
          f"{fmt(SUPERVISED['soil'], '%'):>8}  {fmt(SUPERVISED['bedrock'], '%'):>9}  "
          f"{fmt(SUPERVISED['sand'], '%'):>8}  {'GPU':>8}  {str(SUPERVISED['n']):>6}  "
          f"{SUPERVISED['note']}")
    print("=" * len(header))


def main():
    rows = list(MANUAL_ROWS)

    # Few-shot results — CSV has 'shots' column and baseline comparison rows mixed in
    fewshot_path = os.path.join(RESULTS_DIR, "few_shot_linear_probe_ViT-B-32.csv")
    if os.path.exists(fewshot_path):
        with open(fewshot_path) as f:
            for row in csv.DictReader(f):
                shots = row.get("shots", "")
                try:
                    n_shots = int(shots)   # skip non-numeric rows (baselines)
                except ValueError:
                    continue
                rows.append({
                    "model":    f"CLIP ViT-B/32 + LogReg {n_shots}-shot",
                    "overall":  float(row["overall"]),
                    "soil":     float(row["soil"])    if row.get("soil")    not in (None, "N/A", "") else None,
                    "bedrock":  float(row["bedrock"]) if row.get("bedrock") not in (None, "N/A", "") else None,
                    "sand":     float(row["sand"])    if row.get("sand")    not in (None, "N/A", "") else None,
                    "speed_ms": None,
                    "n":        n_shots * 4,  # 4 classes × n shots per class
                    "note":     "frozen CLIP encoder + LogReg",
                })

    # BLIP-2 results (float16 and/or int4)
    for fname, label in [
        ("blip2_terrain_vqa_float16.csv", "BLIP-2 OPT-2.7B float16 CPU"),
        ("blip2_terrain_vqa_int4.csv",    "BLIP-2 OPT-2.7B INT4 GPU"),
        ("blip2_terrain_vqa_int8.csv",    "BLIP-2 OPT-2.7B INT8 GPU"),
    ]:
        rows += load_csv_results(fname, label, "VQA")

    # SmolVLM results
    for fname, label in [
        ("smolvlm_256m_terrain_test.csv", "SmolVLM-256M (256M)"),
        ("smolvlm_500m_terrain_test.csv", "SmolVLM-500M (500M)"),
    ]:
        rows += load_csv_results(fname, label, "VQA edge SLM")

    # Phi-3-V results
    rows += load_csv_results(
        "phi3v_terrain_vqa.csv",
        "Phi-3-V INT4 (4.2B)",
        "VQA — microsoft/Phi-3-vision-128k"
    )

    if not rows:
        print("No results found in experiments/results/")
        return

    print_table(rows)

    # Gap summary
    print("\nGap vs supervised baseline (96.67%):")
    for r in rows:
        gap = r["overall"] - SUPERVISED["overall"]
        bar = "█" * int(r["overall"] / 5)
        print(f"  {r['model'][:45]:<45} {r['overall']:5.1f}%  gap={gap:+.1f}%  {bar}")


if __name__ == "__main__":
    main()
