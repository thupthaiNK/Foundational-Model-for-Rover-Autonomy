"""
Purpose: Compute-variance meta-analysis across archived turning-RTF diagnostic
         runs -- backlog item 18, scoped 2026-07-20. Quantifies Gazebo
         real-time-factor (RTF = sim_time / real_elapsed_time) variance on
         this development machine, using the already-recorded
         turning_rtf_diagnostic_*.csv logs (5 configs, produced while
         investigating the L4 in-place-rotation issue, §4.8.23). This is a
         methodology point supporting the standing "dev-machine compute
         throttling" explanation used elsewhere in this thesis for D1's null
         result (§4.8.13) and L4's compute isolation test (§4.8.24) --
         quantifying how much RTF actually varies run-to-run on this host,
         rather than asserting it qualitatively each time.
Inputs:  experiments/results/turning_rtf_diagnostic_*.csv (real_elapsed_s,
         sim_time_s columns; 5 configs: baseline, frictiononly, torqueonly,
         multiwheel, multiwheel_run2, hightorque_lowfriction).
Outputs: experiments/results/compute_variance_meta_analysis.csv
         Printed summary for Ch4/Ch5 methodology discussion.
How to run:
    python3 experiments/compute_variance_meta_analysis.py
Project: Onboard Visual Foundation Models for Mars Terrain Perception and Rover Navigation
"""
import csv
import glob
import os
import statistics

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def load_run(path):
    rows = []
    with open(path) as f:
        for row in csv.DictReader(f):
            if not row.get("real_elapsed_s") or not row.get("sim_time_s"):
                continue
            rows.append((float(row["real_elapsed_s"]), float(row["sim_time_s"])))
    return rows


def windowed_rtf(rows, window_rows=10):
    """Instantaneous RTF over sliding windows of `window_rows` samples, to
    capture within-run variance (not just a single overall-run average)."""
    rtfs = []
    for i in range(0, len(rows) - window_rows, window_rows):
        (real0, sim0) = rows[i]
        (real1, sim1) = rows[i + window_rows]
        d_real = real1 - real0
        d_sim = sim1 - sim0
        if d_real > 0:
            rtfs.append(d_sim / d_real)
    return rtfs


def main():
    paths = sorted(glob.glob(os.path.join(RESULTS_DIR, "turning_rtf_diagnostic_*.csv")))
    out_path = os.path.join(RESULTS_DIR, "compute_variance_meta_analysis.csv")

    all_run_means = []
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["config", "n_samples", "n_windows", "overall_rtf",
                          "windowed_rtf_mean", "windowed_rtf_stdev", "windowed_rtf_min",
                          "windowed_rtf_max"])
        for path in paths:
            config = os.path.basename(path).replace("turning_rtf_diagnostic_", "").replace(".csv", "")
            rows = load_run(path)
            overall_rtf = (rows[-1][1] - rows[0][1]) / (rows[-1][0] - rows[0][0])
            rtfs = windowed_rtf(rows)
            mean_rtf = statistics.mean(rtfs)
            stdev_rtf = statistics.stdev(rtfs) if len(rtfs) > 1 else 0.0
            writer.writerow([config, len(rows), len(rtfs), round(overall_rtf, 3),
                              round(mean_rtf, 3), round(stdev_rtf, 3),
                              round(min(rtfs), 3), round(max(rtfs), 3)])
            all_run_means.append(mean_rtf)
            print(f"{config:30s} overall RTF={overall_rtf:.3f}  "
                  f"windowed mean={mean_rtf:.3f} stdev={stdev_rtf:.3f} "
                  f"range=[{min(rtfs):.3f}, {max(rtfs):.3f}]")

    print(f"\nAcross {len(all_run_means)} runs/configs: "
          f"RTF mean-of-means={statistics.mean(all_run_means):.3f}, "
          f"stdev-of-means={statistics.stdev(all_run_means):.3f}")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
