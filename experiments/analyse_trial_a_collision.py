#!/usr/bin/env python3
"""
Purpose: Answer one question with data instead of a guess: when the rover's
         front-left wheel reached the lab cupboard at the end of Trial A
         (2026-07-29), what did the LiDAR actually see in that direction?

         Two hypotheses are on the table. (1) The forward corridor check is a
         narrow 9.1 deg cone, so a cupboard approached obliquely sits outside
         it and is never counted as blocking. (2) Grazing incidence: a flat
         specular panel struck at a shallow angle reflects the beam away
         rather than back, so the LiDAR returns nothing at all there until it
         is nearly perpendicular. The two call for completely different fixes,
         and only the recorded scans can separate them.

         Prints, for the last `--window` seconds of the bag, a per-scan
         summary of the nearest return in each 15 deg sector, plus the
         fraction of beams in the rover's left-front quadrant that came back
         as no-return. Hypothesis (2) shows up as that fraction climbing
         towards 1.0 while the rover closes on the cupboard; hypothesis (1)
         shows up as a healthy, steady return at a bearing outside +/-9.1 deg.

Inputs:  a rosbag2 directory containing /scan (sensor_msgs/msg/LaserScan).
Outputs: a per-scan table on stdout, then a verdict summary.
How to run (on the Pi, inside the exomy_ros container):
    source /opt/ros/humble/setup.bash && source /ws/install/setup.bash
    python3 /ws/experiments/analyse_trial_a_collision.py \
        /ws/bags/real_hardware_20260729_144735 --window 10
Project: Onboard Visual Foundation Models for Mars Terrain Perception and Rover Navigation
"""
import argparse
import glob
import math
import os
import sqlite3
import sys

from rclpy.serialization import deserialize_message
from sensor_msgs.msg import LaserScan

# Measured on the rover, 2026-07-29. Every bearing below is in the PHYSICAL
# frame, i.e. already corrected for where the LiDAR's own zero sits.
LIDAR_YAW_OFFSET_DEG = 3.5
SECTOR_DEG = 15.0
# The quadrant the front-left wheel swings through, which is where the
# cupboard was according to the user's sketch (Picture/127.jpg).
LEFT_FRONT_SECTOR = (0.0, 90.0)


def _sector_index(bearing_deg: float) -> int:
    return int((bearing_deg % 360.0) // SECTOR_DEG)


def read_scans(bag_path: str):
    """Yield (timestamp_ns, LaserScan) for every /scan message in the bag.

    Reads the .db3 files with sqlite3 directly instead of going through
    rosbag2_py's SequentialReader. That reader opens the database read-only,
    and a bag whose recorder was killed rather than shut down cleanly is left
    with a hot journal that SQLite must replay before any read can succeed --
    which a read-only handle cannot do, so it fails with
    "SQLite error (10): disk I/O error" and the recording looks lost when it
    is not. Trial A's bag ended exactly that way (the rover's power switch was
    hit mid-recording). Opening read-write lets SQLite recover the journal.

    The rosbag2 sqlite3 schema is stable and small: a `topics` table naming
    each topic, and a `messages` table of (topic_id, timestamp, blob).
    """
    if os.path.isdir(bag_path):
        db_files = sorted(glob.glob(os.path.join(bag_path, "*.db3")))
    else:
        db_files = [bag_path]
    if not db_files:
        raise SystemExit(f"no .db3 file found under {bag_path}")

    found_any_topic = False
    for db in db_files:
        conn = sqlite3.connect(db)          # read-write: recovers a hot journal
        try:
            names = [r[0] for r in conn.execute("SELECT name FROM topics")]
            found_any_topic = found_any_topic or bool(names)
            rows = conn.execute(
                "SELECT m.timestamp, m.data FROM messages m "
                "JOIN topics t ON m.topic_id = t.id "
                "WHERE t.name = '/scan' ORDER BY m.timestamp"
            )
            for stamp_ns, blob in rows:
                yield stamp_ns, deserialize_message(bytes(blob), LaserScan)
        finally:
            conn.close()

    if not found_any_topic:
        raise SystemExit(f"{bag_path} has no topics table; is it a rosbag2 bag?")


def summarise(scan: LaserScan):
    """Return (nearest_by_sector, left_front_noreturn_fraction)."""
    nearest = {}
    lf_total = 0
    lf_noreturn = 0

    for i, r in enumerate(scan.ranges):
        lidar_bearing = math.degrees(scan.angle_min + i * scan.angle_increment)
        # Physical bearing: the offset is defined as the LiDAR-frame bearing at
        # which the rover's physical front appears, so subtract it to convert.
        bearing = (lidar_bearing - LIDAR_YAW_OFFSET_DEG) % 360.0

        in_left_front = LEFT_FRONT_SECTOR[0] <= bearing <= LEFT_FRONT_SECTOR[1]
        valid = (
            r == r                                  # not NaN
            and math.isfinite(r)
            and scan.range_min <= r <= scan.range_max
        )
        if in_left_front:
            lf_total += 1
            if not valid:
                lf_noreturn += 1
        if not valid:
            continue

        s = _sector_index(bearing)
        if s not in nearest or r < nearest[s]:
            nearest[s] = r

    frac = (lf_noreturn / lf_total) if lf_total else float("nan")
    return nearest, frac


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("bag")
    ap.add_argument("--window", type=float, default=10.0,
                    help="seconds before the end of the bag to report on")
    args = ap.parse_args()

    scans = list(read_scans(args.bag))
    if not scans:
        raise SystemExit("bag contained no /scan messages")

    end_ns = scans[-1][0]
    cutoff_ns = end_ns - int(args.window * 1e9)
    tail = [(t, s) for t, s in scans if t >= cutoff_ns]

    print(f"{len(scans)} scans in bag, {len(tail)} in the last {args.window:.0f} s")
    print(f"bearings are PHYSICAL (LiDAR yaw offset {LIDAR_YAW_OFFSET_DEG} deg removed)")
    print()
    print(f"{'t-end(s)':>9} | {'ahead':>7} {'L15':>7} {'L30':>7} {'L45':>7} "
          f"{'L60':>7} {'L75':>7} | {'min360':>7} | left-front no-return")
    print("-" * 96)

    for t_ns, scan in tail:
        nearest, frac = summarise(scan)
        dt = (t_ns - end_ns) / 1e9

        def cell(bearing_deg):
            v = nearest.get(_sector_index(bearing_deg))
            return f"{v:7.3f}" if v is not None else "      -"

        overall = min(nearest.values()) if nearest else float("nan")
        print(f"{dt:9.2f} | {cell(0)} {cell(15)} {cell(30)} {cell(45)} "
              f"{cell(60)} {cell(75)} | {overall:7.3f} | {frac * 100:5.1f}%")

    print()
    first_frac = summarise(tail[0][1])[1]
    last_frac = summarise(tail[-1][1])[1]
    print(f"left-front no-return fraction: {first_frac * 100:.1f}% at the start "
          f"of the window, {last_frac * 100:.1f}% at the end")
    print()
    print("How to read this:")
    print("  Hypothesis 1 (corridor too narrow): the cupboard shows up as a")
    print("  solid, shrinking distance in L30/L45/L60 while 'ahead' stays")
    print("  clear. The LiDAR saw it; the planner was not looking there.")
    print("  Hypothesis 2 (grazing incidence): the left-front columns go to")
    print("  '-' and the no-return fraction climbs as the rover closes. The")
    print("  LiDAR never saw it, and widening the corridor would not have")
    print("  helped.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
