"""
Purpose: Offline ablation, max-combine (deployed) vs. equal-weighted-sum, for
         traversability_score_fusion_node.py's fuse_traversability_score() --
         backlog item 26, scoped 2026-07-20. Replays the real recorded
         (dinov2_score, lidar_risk, imu_risk) triples from the existing H13
         live verification logs (Ch4 S4.8.25) through both combination rules
         and reports where they diverge, in particular any case where a
         single hard-STOP signal (risk=1.0) would be diluted below the
         deployed STOP threshold by weighted-sum -- the concrete safety
         argument for the "worst signal wins" design already deployed.
         H13's own live runs tested each risk channel in isolation (§4.8.25's
         isolated-launch pattern), so no single recorded run has all three
         channels simultaneously nonzero; this ablation is therefore run
         once per available log (LiDAR channel live, IMU channel live) with
         the untested channel held at its actual recorded value of 0.0,
         not fabricated -- consistent with how these systems were actually
         verified.
Inputs:  experiments/results/traversability_fusion_lidar_live_test.csv
         experiments/results/traversability_fusion_imu_live_test.csv
Outputs: experiments/results/fusion_ablation_max_vs_weighted.csv
         Printed summary for Ch4.
How to run:
    python3 experiments/fusion_ablation_max_vs_weighted.py
Project: Onboard Visual Foundation Models for Mars Terrain Perception and Rover Navigation
"""
import csv
import os

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

STOP_THRESHOLD = 1.0  # fuse_traversability_score's own STOP convention: 1.0 = hard stop
WEIGHTS = (1 / 3, 1 / 3, 1 / 3)  # dinov2, lidar, imu -- equal weights, the natural
                                  # baseline for an ablation with no other principled
                                  # prior (unlike traversability_grid.py's risk weights,
                                  # which are calibrated to a discrete speed policy,
                                  # not to this fusion node's three risk channels)


def max_combine(dinov2_score: float, lidar_risk: float, imu_risk: float) -> float:
    return max(dinov2_score, lidar_risk, imu_risk)


def weighted_sum(dinov2_score: float, lidar_risk: float, imu_risk: float) -> float:
    w_d, w_l, w_i = WEIGHTS
    return w_d * dinov2_score + w_l * lidar_risk + w_i * imu_risk


def load_lidar_log():
    path = os.path.join(RESULTS_DIR, "traversability_fusion_lidar_live_test.csv")
    rows = []
    with open(path) as f:
        for row in csv.DictReader(f):
            rows.append((float(row["dinov2_score"]), float(row["expected_lidar_risk"]), 0.0))
    return rows


def load_imu_log():
    path = os.path.join(RESULTS_DIR, "traversability_fusion_imu_live_test.csv")
    rows = []
    with open(path) as f:
        for row in csv.DictReader(f):
            rows.append((0.0, 0.0, float(row["expected_imu_risk"])))
    return rows


def main():
    all_rows = [("lidar_log", *r) for r in load_lidar_log()] + \
               [("imu_log", *r) for r in load_imu_log()]

    out_path = os.path.join(RESULTS_DIR, "fusion_ablation_max_vs_weighted.csv")
    n_diverge = 0
    n_diluted_stop = 0
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["source", "dinov2_score", "lidar_risk", "imu_risk",
                          "max_combine", "weighted_sum", "diluted_a_stop_signal"])
        for source, d, l, i in all_rows:
            m = max_combine(d, l, i)
            w = weighted_sum(d, l, i)
            diverges = abs(m - w) > 1e-9
            diluted = (m >= STOP_THRESHOLD) and (w < STOP_THRESHOLD)
            if diverges:
                n_diverge += 1
            if diluted:
                n_diluted_stop += 1
            writer.writerow([source, d, l, i, round(m, 4), round(w, 4), diluted])

    n_total = len(all_rows)
    print(f"Rows: {n_total}")
    print(f"Rows where max-combine and weighted-sum disagree: {n_diverge} "
          f"({100 * n_diverge / n_total:.1f}%)")
    print(f"Rows where a genuine STOP-level signal (risk=1.0) would be diluted "
          f"below STOP by weighted-sum: {n_diluted_stop} "
          f"({100 * n_diluted_stop / n_total:.1f}%)")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
