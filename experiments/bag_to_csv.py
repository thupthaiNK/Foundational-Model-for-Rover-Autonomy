#!/usr/bin/env python3
"""
Purpose: Turn a recorded ros2 bag into one analysis-ready CSV, sampled onto a
         fixed time grid, ready for the MATLAB figure scripts.

         The bag is the source of truth for a run; this is derived from it. That
         is deliberate. An earlier design wrote CSV live on the rover, and it was
         dropped because a field test cannot be re-run while a converter can be
         re-run without limit, and because writing it live would have added CPU
         and card load to the machine that is already the bottleneck.

         Every rule about what a reading means comes from
         fm_perception.sensor_snapshot -- the same module the live RViz2 overlay
         uses, through the same TOPIC_EXTRACTORS table. A graph produced here
         therefore cannot disagree with what was on the screen during the run.

         Staleness carries through: a field whose last message is older than its
         timeout is written empty rather than carried forward. Carrying a value
         forward across a sensor dropout turns a gap into a flat line, and a flat
         line in a plot reads as a measurement nobody thinks to question.

Inputs:  a bag directory, e.g. bags/real_hardware_20260813_101500/
Outputs: experiments/results/<bag name>.csv, one row per grid step
         plus a short summary to stdout

How to run:
    source /opt/ros/humble/setup.bash && source ros2_ws/install/setup.bash
    python3 experiments/bag_to_csv.py bags/real_hardware_20260813_101500

    # every bag from the field test at once, at 2 Hz
    python3 experiments/bag_to_csv.py bags/real_hardware_* --hz 2.0

Project: Onboard Visual Foundation Models for Mars Terrain Perception and Rover Navigation
"""

import argparse
import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "ros2_ws" / "src" / "fm_perception"))

from fm_perception.sensor_snapshot import (  # noqa: E402
    CSV_COLUMNS,
    TOPIC_EXTRACTORS,
    SensorSnapshot,
)

DEFAULT_HZ = 2.0
RESULTS = REPO_ROOT / "experiments" / "results"


def _readers():
    """Import rosbag2 lazily so --help works without a sourced workspace."""
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message
    return rosbag2_py, deserialize_message, get_message


def read_bag(bag_dir: Path):
    """Yield (topic, deserialised message, seconds) for topics we understand.

    Timestamps are the bag's own receive times, converted to seconds. They are
    monotonic within a run, which is all the grid needs; they are not wall-clock
    aligned with anything else.
    """
    rosbag2_py, deserialize_message, get_message = _readers()

    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(bag_dir), storage_id="sqlite3"),
        rosbag2_py.ConverterOptions("", ""),
    )

    # Trust the bag's own type for a topic rather than the name in
    # TOPIC_EXTRACTORS: a bag recorded before a type changed should fail loudly
    # here rather than deserialise into something that looks plausible.
    recorded = {t.name: t.type for t in reader.get_all_topics_and_types()}
    types = {}
    for topic, (expected, _) in TOPIC_EXTRACTORS.items():
        actual = recorded.get(topic)
        if actual is None:
            continue
        if actual != expected:
            raise SystemExit(
                f"{bag_dir.name}: {topic} is {actual} in the bag but "
                f"{expected} in TOPIC_EXTRACTORS")
        types[topic] = get_message(actual)

    while reader.has_next():
        topic, data, stamp_ns = reader.read_next()
        msg_type = types.get(topic)
        if msg_type is None:
            continue
        yield topic, deserialize_message(data, msg_type), stamp_ns * 1e-9


def convert(bag_dir: Path, out_dir: Path, hz: float = DEFAULT_HZ) -> Path:
    """Write one CSV for one bag. Returns the path written."""
    step = 1.0 / hz
    snap = SensorSnapshot()
    rows: list[list] = []
    t0: float | None = None
    next_sample: float | None = None
    n_msgs = 0

    for topic, msg, t in read_bag(bag_dir):
        if t0 is None:
            t0, next_sample = t, t
        # Emit every grid step up to this message's time, so a silent stretch
        # produces rows full of STALE rather than no rows at all. A gap that
        # leaves no rows is invisible in a plot; a run of empty cells is not.
        while next_sample is not None and next_sample <= t:
            row = snap.as_csv_row(now=next_sample)
            row[0] = round(next_sample - t0, 3)   # seconds since the run began
            rows.append(row)
            next_sample += step
        snap.apply(topic, msg, t)
        n_msgs += 1

    if t0 is None:
        raise SystemExit(f"{bag_dir.name}: no messages on any known topic")

    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{bag_dir.name}.csv"
    with out.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(CSV_COLUMNS)
        writer.writerows(rows)

    duration = rows[-1][0] if rows else 0.0
    filled = _fill_rates(rows)
    try:
        shown = out.relative_to(REPO_ROOT)
    except ValueError:
        shown = out          # --out can point outside the repo
    print(f"{bag_dir.name}: {n_msgs} messages, {len(rows)} rows, "
          f"{duration:.1f} s -> {shown}")
    for col, pct in sorted(filled.items(), key=lambda kv: kv[1]):
        if pct < 50.0:
            # Worth saying out loud: a column that is mostly empty means the
            # sensor was absent or dropping out, not that the run was quiet.
            print(f"    {col:22s} filled {pct:5.1f}% of rows")
    return out


def _fill_rates(rows: list[list]) -> dict[str, float]:
    if not rows:
        return {}
    n = len(rows)
    return {
        col: 100.0 * sum(1 for r in rows if r[i] != "") / n
        for i, col in enumerate(CSV_COLUMNS)
        if col != "t_s"
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("Inputs:")[0])
    ap.add_argument("bags", nargs="+", type=Path,
                    help="bag directories, e.g. bags/real_hardware_*")
    ap.add_argument("--hz", type=float, default=DEFAULT_HZ,
                    help=f"sampling rate of the output grid (default {DEFAULT_HZ})")
    ap.add_argument("--out", type=Path, default=RESULTS,
                    help="output directory (default experiments/results)")
    args = ap.parse_args(argv)

    if args.hz <= 0:
        ap.error("--hz must be positive")

    written = 0
    for bag in args.bags:
        if not bag.is_dir():
            print(f"skipping {bag}: not a directory")
            continue
        convert(bag, args.out, args.hz)
        written += 1
    if written == 0:
        print("nothing converted")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
