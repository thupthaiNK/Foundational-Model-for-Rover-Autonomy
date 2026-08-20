"""
Purpose: Auto-generate a short human-readable mission report from an
         existing ground-truth trajectory recorder CSV -- backlog item 15
         (A5), scoped 2026-07-20. Closes the "report findings" piece of the
         original mission-autonomy vision as a -lite version: duration,
         path length, zones visited, and (if an events/selections CSV is
         also given) a count and timeline of autonomous decision points.
         Works on any of this thesis's existing trajectory logs that share
         the t_s,gt_x,gt_y[,zone] column convention (explore_return_home,
         l6_lite_roundtrip, abort_to_home, etc.) -- no new data collection,
         purely a summarisation layer over what these missions already log.
Inputs:  A trajectory CSV with columns t_s, gt_x, gt_y (zone optional).
         Optionally an events CSV with a t_s column (e.g.
         explore_return_home_selections.csv) for a selection timeline.
Outputs: Printed report; optionally saved to a .txt file.
How to run:
    python3 experiments/mission_report.py experiments/results/l6_lite_roundtrip_test.csv
    python3 experiments/mission_report.py experiments/results/explore_return_home_test.csv \
        --events experiments/results/explore_return_home_selections.csv
Project: Onboard Visual Foundation Models for Mars Terrain Perception and Rover Navigation
"""
import argparse
import csv
import math
import os


def _read_trajectory(path):
    rows = []
    with open(path) as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def _read_events(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def generate_report(trajectory_csv: str, events_csv: str = None, mission_name: str = None) -> str:
    rows = _read_trajectory(trajectory_csv)
    if not rows:
        raise ValueError(f"{trajectory_csv} has no data rows")

    mission_name = mission_name or os.path.splitext(os.path.basename(trajectory_csv))[0]
    t0, t1 = float(rows[0]["t_s"]), float(rows[-1]["t_s"])
    duration_s = t1 - t0

    path_length_m = 0.0
    prev = None
    zones_visited = []
    for row in rows:
        x, y = float(row["gt_x"]), float(row["gt_y"])
        if prev is not None:
            path_length_m += math.hypot(x - prev[0], y - prev[1])
        prev = (x, y)
        zone = row.get("zone")
        if zone and (not zones_visited or zones_visited[-1] != zone):
            zones_visited.append(zone)

    lines = [
        f"Mission report: {mission_name}",
        "=" * (16 + len(mission_name)),
        f"Duration: {duration_s:.1f} s ({t0:.1f}s -> {t1:.1f}s)",
        f"Ground-truth path length (recomputed from this CSV): {path_length_m:.2f} m",
        "  CAVEAT: path length recomputed from a rounded CSV export can read up to "
        "~13% higher than a script's own live-printed total (see "
        "feedback_verify_against_source_log_not_recomputed_csv memory) -- treat this "
        "figure as an approximate cross-check, not a replacement for the mission's "
        "own reported number if one exists in the thesis.",
        f"Trajectory samples: {len(rows)}",
    ]
    if zones_visited:
        lines.append(f"Zone sequence ({len(zones_visited)} transitions): {' -> '.join(zones_visited)}")

    if events_csv:
        events = _read_events(events_csv)
        lines.append(f"Autonomous decision points ({os.path.basename(events_csv)}): {len(events)}")
        for i, ev in enumerate(events[:5]):
            lines.append(f"  [{i}] t={float(ev['t_s']):.1f}s  " +
                          ", ".join(f"{k}={v}" for k, v in ev.items() if k != "t_s"))
        if len(events) > 5:
            lines.append(f"  ... and {len(events) - 5} more")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trajectory_csv")
    parser.add_argument("--events", default=None)
    parser.add_argument("--name", default=None)
    parser.add_argument("--out", default=None, help="also write the report to this .txt file")
    args = parser.parse_args()

    report = generate_report(args.trajectory_csv, args.events, args.name)
    print(report)
    if args.out:
        with open(args.out, "w") as f:
            f.write(report + "\n")
        print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
