#!/usr/bin/env python3
"""
Purpose: A3 boundary-band uncertainty quantification (offline, no Gazebo) --
         item 9 of the L1-L6 further-work plan (2026-07-19). Attempts to
         directly quantify, from already-recorded data, the "mixed-view"
         mechanism inferred (not measured) during the semantic-frontier
         deadlock investigation (grid_with_start_freed, §3.11.12/§4.8.28):
         the hypothesis that DINOv2 classification confidence drops nearer
         the soil/bedrock zone boundary because a single camera frame
         straddling both classes gives the classifier a genuinely mixed,
         ambiguous view.
         RESULT (see the module's own printed summary and §4.8.28's
         write-up): this specific test -- confidence vs. the rover's raw
         Euclidean distance from the boundary line -- found NO significant
         relationship (r=-0.078, n=578, pooled; -0.008 and -0.077 per run
         individually). This does not refute the mixed-view mechanism
         itself, which the two live deadlocks still directly evidence; it
         means position alone is too weak a proxy for what the camera
         actually saw. DINOv2 classifies a whole forward-camera frame
         (live_traversability_costmap_node.py's own docstring already
         states this is a deliberate simplification), so what determines a
         "mixed view" is the classified frame's field of view -- a function
         of position AND heading AND the 0.6 m lookahead projection -- not
         raw position alone; heading was not recorded in either source
         CSV, so a position-only proxy is the strongest test this existing
         data supports. Reported as a null result rather than reframed or
         dropped, per this thesis's standing practice (Config B, §4.8.8;
         D1, §4.8.13).
         Method: correlate each DINOv2 classification's logged confidence
         against the rover's ground-truth distance from the class boundary
         (x=0 in the shared arena convention: soil_zone x<0, bedrock_zone
         x>0, §3.4/§4.8.1) at the nearest-in-time recorded pose, across two
         independent official missions that both operated at this boundary:
         the semantic-frontier official run 2 (§4.8.28, PASS) and the A2
         re-observation official run (§4.8.30, PASS). Both raw ROS launch
         logs were captured this session and are committed under
         raw_logs/ below, since the scratchpad directory they were
         originally written to does not persist across sessions.
         DINOv2's own log line is throttled to every 10th classification
         (dinov2_terrain_node.py's internal counter) -- this analysis uses
         exactly that subsampled stream, not a hidden full-rate one; the
         sampling is a property of the source log, disclosed as such.
Inputs:  experiments/results/raw_logs/semantic_frontier_run2_launch.log
         experiments/results/raw_logs/reobservation_official_launch.log
         experiments/results/raw_logs/semantic_frontier_run2_recorder.log
         experiments/results/raw_logs/reobservation_official_recorder.log
             (recorder logs supply the epoch anchor for their own CSV's
             t_s=0, from each recorder's own "Recording..." startup line)
         experiments/results/semantic_frontier_test.csv (ground-truth path)
         experiments/results/reobservation_test.csv (ground-truth path)
Outputs: experiments/results/boundary_band_uncertainty.csv
         experiments/results/figures/boundary_band_uncertainty.png
How to run:
    python3 experiments/boundary_band_uncertainty.py
Project: Onboard Visual Foundation Models for Mars Terrain Perception and Rover Navigation
"""
import csv
import os
import re
from bisect import bisect_left
from typing import List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
RAW_LOGS_DIR = os.path.join(RESULTS_DIR, "raw_logs")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")

CONFIDENCE_THRESHOLD = 0.40  # deployed classification threshold, dinov2_terrain_node.py

# (log file, position csv, recorder's own "Recording..." line -- gives the
# epoch anchor for that CSV's t_s column, read directly from the log rather
# than assumed, since the recorder's own __init__ timestamp is what t_s=0
# actually corresponds to)
RUNS = [
    {
        "name": "semantic_frontier_run2",
        "log": os.path.join(RAW_LOGS_DIR, "semantic_frontier_run2_launch.log"),
        "recorder_log": os.path.join(RAW_LOGS_DIR, "semantic_frontier_run2_recorder.log"),
        "csv": os.path.join(RESULTS_DIR, "semantic_frontier_test.csv"),
        "anchor_re": re.compile(
            r"\[(\d+\.\d+)\] \[semantic_frontier_recorder\]: Recording"),
    },
    {
        "name": "reobservation_official",
        "log": os.path.join(RAW_LOGS_DIR, "reobservation_official_launch.log"),
        "recorder_log": os.path.join(RAW_LOGS_DIR, "reobservation_official_recorder.log"),
        "csv": os.path.join(RESULTS_DIR, "reobservation_test.csv"),
        "anchor_re": re.compile(
            r"\[(\d+\.\d+)\] \[reobservation_test\]: Recording"),
    },
]

CLASSIFICATION_RE = re.compile(
    r"\[(\d+\.\d+)\] \[dinov2_terrain_node\]: \[\d+\] (\w+):([\d.]+) \|")

# Distance bands (metres from the x=0 boundary), chosen to resolve the
# lookahead/patch-radius geometry (0.6m lookahead, 0.3m patch radius,
# §3.11.12) rather than picked arbitrarily: the paint disc itself spans
# roughly 0.3m either side of its centre, so bands narrower than that
# would mostly reflect noise, not a real spatial distinction.
BANDS = [(0.0, 0.3), (0.3, 0.6), (0.6, 1.0), (1.0, 2.0), (2.0, float("inf"))]


def find_anchor_epoch(log_path: str, anchor_re: re.Pattern) -> float:
    with open(log_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = anchor_re.search(line)
            if m:
                return float(m.group(1))
    raise ValueError(f"No recorder anchor line found in {log_path}")


def load_position_csv(csv_path: str) -> Tuple[List[float], List[float], List[float]]:
    """Returns (t_s list, x list, y list), sorted by t_s (already ascending
    in these recorders' own output, but sorted defensively)."""
    ts, xs, ys = [], [], []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ts.append(float(row["t_s"]))
            xs.append(float(row["gt_x"]))
            ys.append(float(row["gt_y"]))
    order = sorted(range(len(ts)), key=lambda i: ts[i])
    return [ts[i] for i in order], [xs[i] for i in order], [ys[i] for i in order]


def nearest_position(t_query: float, ts: List[float], xs: List[float]) -> float:
    """Nearest-in-time x position to t_query (simple binary search;
    ts is sorted)."""
    i = bisect_left(ts, t_query)
    if i == 0:
        return xs[0]
    if i == len(ts):
        return xs[-1]
    before, after = ts[i - 1], ts[i]
    return xs[i - 1] if (t_query - before) <= (after - t_query) else xs[i]


def extract_classifications(log_path: str) -> List[Tuple[float, str, float]]:
    """Returns [(epoch, label, confidence), ...] in log order."""
    events = []
    with open(log_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = CLASSIFICATION_RE.search(line)
            if m:
                events.append((float(m.group(1)), m.group(2), float(m.group(3))))
    return events


def band_label(d: float) -> str:
    for lo, hi in BANDS:
        if lo <= d < hi:
            return f"{lo:.1f}-{hi:.1f}m" if hi != float("inf") else f">={lo:.1f}m"
    return "?"


def main() -> None:
    os.makedirs(FIGURES_DIR, exist_ok=True)
    rows = []  # (run, t_s, distance_to_boundary_m, label, confidence, band)

    for run in RUNS:
        anchor_epoch = find_anchor_epoch(run["recorder_log"], run["anchor_re"])
        ts, xs, _ys = load_position_csv(run["csv"])
        classifications = extract_classifications(run["log"])
        for epoch, label, conf in classifications:
            t_s = epoch - anchor_epoch
            x = nearest_position(t_s, ts, xs)
            dist = abs(x)
            rows.append((run["name"], round(t_s, 2), round(dist, 3), label,
                         conf, band_label(dist)))
        print(f"{run['name']}: {len(classifications)} classification samples "
              f"(log-throttled to every 10th frame), anchor epoch={anchor_epoch:.3f}")

    csv_path = os.path.join(RESULTS_DIR, "boundary_band_uncertainty.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["run", "t_s", "distance_to_boundary_m", "label",
                         "confidence", "band"])
        writer.writerows(rows)

    # -- Per-band summary ------------------------------------------------
    print("\n" + "=" * 78)
    print("BOUNDARY-BAND UNCERTAINTY SUMMARY (both official runs combined)")
    print(f"Total classification samples: {len(rows)}")
    print(f"{'Band':<12} {'N':>5} {'mean conf':>10} {'std':>7} "
          f"{'%below 0.40':>12}")
    band_order = [band_label(lo) if hi != float("inf") else band_label(lo)
                  for lo, hi in BANDS]
    band_stats = []
    for lo, hi in BANDS:
        label = band_label(lo)
        confs = [r[4] for r in rows if r[5] == label]
        if not confs:
            continue
        arr = np.array(confs)
        pct_uncertain = 100.0 * (arr < CONFIDENCE_THRESHOLD).mean()
        band_stats.append((label, len(confs), arr.mean(), arr.std(), pct_uncertain))
        print(f"{label:<12} {len(confs):>5} {arr.mean():>10.3f} {arr.std():>7.3f} "
              f"{pct_uncertain:>11.1f}%")

    all_dist = np.array([r[2] for r in rows])
    all_conf = np.array([r[4] for r in rows])
    if len(all_dist) > 2:
        corr = np.corrcoef(all_dist, all_conf)[0, 1]
        print(f"\nPearson correlation (distance-to-boundary, confidence): "
              f"r={corr:+.3f} (n={len(rows)})")
    print("=" * 78)

    # -- Figure: scatter + per-band mean/std ------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    colors = {"semantic_frontier_run2": "tab:blue",
              "reobservation_official": "tab:orange"}
    for run in RUNS:
        run_rows = [r for r in rows if r[0] == run["name"]]
        if not run_rows:
            continue
        d = [r[2] for r in run_rows]
        c = [r[4] for r in run_rows]
        ax1.scatter(d, c, s=18, alpha=0.6, color=colors[run["name"]],
                    label=run["name"])
    ax1.axhline(CONFIDENCE_THRESHOLD, color="red", linestyle="--", linewidth=1,
                label=f"threshold ({CONFIDENCE_THRESHOLD})")
    ax1.set_xlabel("distance from soil/bedrock boundary (m)")
    ax1.set_ylabel("classification confidence")
    ax1.set_title("Confidence vs distance-to-boundary (per sample)")
    ax1.legend(fontsize=8)

    if band_stats:
        labels = [b[0] for b in band_stats]
        means = [b[2] for b in band_stats]
        stds = [b[3] for b in band_stats]
        ns = [b[1] for b in band_stats]
        x_pos = range(len(labels))
        ax2.bar(x_pos, means, yerr=stds, capsize=4, color="tab:purple", alpha=0.75)
        ax2.axhline(CONFIDENCE_THRESHOLD, color="red", linestyle="--", linewidth=1,
                    label=f"threshold ({CONFIDENCE_THRESHOLD})")
        ax2.set_xticks(list(x_pos))
        ax2.set_xticklabels(labels, fontsize=8)
        for i, n in enumerate(ns):
            ax2.annotate(f"n={n}", (i, means[i] + stds[i] + 0.02),
                        ha="center", fontsize=7)
        ax2.set_ylabel("mean confidence (+/- 1 std)")
        ax2.set_title("Mean confidence by distance band")
        ax2.legend(fontsize=8)
        ax2.set_ylim(0, 1.05)

    fig.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, "boundary_band_uncertainty.png")
    fig.savefig(fig_path, dpi=150)
    print(f"\nSaved: {csv_path}")
    print(f"Saved: {fig_path}")


if __name__ == "__main__":
    main()
