#!/usr/bin/env python3
"""
Purpose: Measure which LiDAR bearings are permanently blocked by ExoMy's own
         structure, and print a ready-to-paste `blocked_sectors_deg` mask for
         lidar_proximity_guard_node.py.

         Why this is needed: the RPLIDAR C1 is mounted at the centre of the
         top plate, so it sees the rover's own mast and servo arms at
         ~0.20-0.23 m. That is beyond the guard's radial `min_ignore_m`
         (0.2 m, sized for the chassis at ~0.075 m) but inside
         `stop_distance_m` (0.4 m), so a 360-degree distance-only guard
         latches STOP forever on the rover itself (verified live
         2026-07-23/24). The obstruction is angular, not radial, so the fix
         is an angular mask. This script derives that mask from recorded
         scans instead of guessing it.

         Method: accumulate N scans with the rover STATIONARY and nothing
         within about a metre of it. A beam index is called "self-view" when
         a close return (< self_view_max_m) appears in at least
         `persistence` of those scans. Real obstacles and people move or are
         further away, so they do not persist across every scan; the rover's
         own structure does. Contiguous self-view indices are merged into
         sectors, padded by `pad_deg`, and printed in degrees.

         Read the printed table before trusting the mask. Every masked
         bearing is a direction in which the proximity guard is blind, so
         the mask must cover the structure and nothing more.

Inputs:  /scan (sensor_msgs/LaserScan) from a running sllidar_ros2 driver.
         Parameters: num_scans, self_view_max_m, persistence, pad_deg,
         min_sector_deg.
Outputs: A per-sector table on stdout plus the flat `blocked_sectors_deg`
         list to paste into the launch file or pass with --ros-args.
         Writes nothing to disk.
How to run:
    # on the Pi, inside the ROS2 container, with the LiDAR already publishing
    source /opt/ros/humble/setup.bash
    source /ws/install/setup.bash
    python3 lidar_self_view_characterisation.py --ros-args \
        -p num_scans:=100 -p self_view_max_m:=0.35
Project: Onboard Visual Foundation Models for Mars Terrain Perception and Rover Navigation
"""
import math

from sensor_msgs.msg import LaserScan
import rclpy
from rclpy.node import Node


def self_view_indices(scans, self_view_max_m: float, persistence: float):
    """Beam indices whose return is close in at least `persistence` of scans.

    `scans` is a list of equal-length ranges arrays. A NaN or +Inf entry is
    "no return", which counts as not-close, so a beam that only sometimes
    sees something near is not called structure.
    """
    if not scans:
        return []
    n_beams = min(len(s) for s in scans)
    out = []
    for i in range(n_beams):
        close = sum(
            1 for s in scans
            if math.isfinite(s[i]) and s[i] < self_view_max_m
        )
        if close >= persistence * len(scans):
            out.append(i)
    return out


def dilate(indices, n_beams: int, pad_beams: int):
    """Grow each blocked beam by pad_beams on both sides, wrapping the circle.

    Padding is applied here rather than per-sector at the end, because the
    structure's returns are not perfectly contiguous: a beam grazing the edge
    of the mast sometimes reports no return at all, splitting one physical
    obstruction into many one-beam runs. Padding first merges those back into
    a single sector by construction, so the output cannot contain overlapping
    or duplicated sectors.
    """
    blocked = set()
    for i in indices:
        for d in range(-pad_beams, pad_beams + 1):
            blocked.add((i + d) % n_beams)
    return sorted(blocked)


def group_contiguous(indices, n_beams: int):
    """Merge sorted beam indices into (start, end) runs, joining across 0/N."""
    if not indices:
        return []
    if len(indices) >= n_beams:
        return [(0, n_beams - 1)]
    runs = []
    start = prev = indices[0]
    for i in indices[1:]:
        if i == prev + 1:
            prev = i
            continue
        runs.append((start, prev))
        start = prev = i
    runs.append((start, prev))
    # A run touching both ends of the array is one run wrapping through the
    # scan's angle_min boundary, not two.
    if len(runs) > 1 and runs[0][0] == 0 and runs[-1][1] == n_beams - 1:
        runs[0] = (runs[-1][0] - n_beams, runs[0][1])
        runs.pop()
    return runs


def runs_to_sectors_deg(runs, angle_min: float, angle_increment: float,
                        min_sector_deg: float):
    """Convert beam-index runs into (start_deg, end_deg) sectors.

    Padding is already baked in by dilate(), so this only converts units and
    enforces a floor width for a run that is a single beam wide.
    """
    sectors = []
    for lo, hi in runs:
        start = math.degrees(angle_min + lo * angle_increment)
        end = math.degrees(angle_min + hi * angle_increment)
        if end - start < min_sector_deg:
            mid = 0.5 * (start + end)
            start, end = mid - min_sector_deg / 2.0, mid + min_sector_deg / 2.0
        sectors.append((_wrap_deg(start), _wrap_deg(end)))
    return sectors


def _wrap_deg(deg: float) -> float:
    return (deg + 180.0) % 360.0 - 180.0


class SelfViewCharacterisationNode(Node):

    def __init__(self):
        super().__init__("lidar_self_view_characterisation")
        self.declare_parameter("num_scans", 100)
        # Above the guard's stop_distance_m (0.4) would sweep in real
        # obstacles, so keep this at or below it.
        self.declare_parameter("self_view_max_m", 0.35)
        self.declare_parameter("persistence", 0.9)
        self.declare_parameter("pad_deg", 3.0)
        self.declare_parameter("min_sector_deg", 4.0)

        self.num_scans = self.get_parameter("num_scans").value
        self.self_view_max = self.get_parameter("self_view_max_m").value
        self.persistence = self.get_parameter("persistence").value
        self.pad_deg = self.get_parameter("pad_deg").value
        self.min_sector_deg = self.get_parameter("min_sector_deg").value

        self._scans = []
        self._proto = None
        self.create_subscription(LaserScan, "/scan", self._cb, 10)
        self.get_logger().info(
            f"Collecting {self.num_scans} scans. Keep the rover STATIONARY "
            "and keep everything (including yourself) more than about a "
            "metre away, or real returns will be recorded as structure."
        )

    def _cb(self, msg: LaserScan) -> None:
        if len(self._scans) >= self.num_scans:
            return
        self._proto = msg
        self._scans.append(list(msg.ranges))
        if len(self._scans) % 20 == 0:
            self.get_logger().info(f"  {len(self._scans)}/{self.num_scans}")

    def done(self) -> bool:
        return len(self._scans) >= self.num_scans

    def report(self) -> None:
        idx = self_view_indices(self._scans, self.self_view_max, self.persistence)
        n_beams = len(self._proto.ranges)
        beam_deg = math.degrees(self._proto.angle_increment)
        pad_beams = max(1, int(round(self.pad_deg / beam_deg))) if beam_deg else 1
        runs = group_contiguous(dilate(idx, n_beams, pad_beams), n_beams)
        sectors = runs_to_sectors_deg(runs, self._proto.angle_min,
                                      self._proto.angle_increment,
                                      self.min_sector_deg)

        print()
        print(f"Scans analysed:   {len(self._scans)}")
        print(f"Beams per scan:   {n_beams}")
        print(f"Self-view beams:  {len(idx)} ({100.0 * len(idx) / n_beams:.1f}% of the scan)")
        print(f"Criterion:        range < {self.self_view_max:.2f} m in "
              f">= {self.persistence * 100:.0f}% of scans")
        # Coarse ring of median distance, so the operator can see the whole
        # environment and judge whether a sector is the rover or the room.
        print("Median range per 15-degree bin (inf = nothing within 16m):")
        for b in range(24):
            lo_deg, hi_deg = -180.0 + 15.0 * b, -180.0 + 15.0 * (b + 1)
            vals = []
            for i in range(n_beams):
                d = math.degrees(self._proto.angle_min + i * self._proto.angle_increment)
                if lo_deg <= _wrap_deg(d) < hi_deg:
                    col = sorted(s[i] for s in self._scans if math.isfinite(s[i]))
                    if col:
                        vals.append(col[len(col) // 2])
            med = sorted(vals)[len(vals) // 2] if vals else float("inf")
            print(f"  {lo_deg:>6.0f} to {hi_deg:>4.0f} deg : {med:>7.3f} m")

        print()
        if not sectors:
            print("No persistent close returns found. Either the mount is "
                  "clear or self_view_max_m is too small.")
            return

        print(f"{'sector':>8}  {'start_deg':>10}  {'end_deg':>10}  {'width_deg':>10}  {'median_m':>9}")
        for k, ((lo, hi), (s_deg, e_deg)) in enumerate(zip(runs, sectors), 1):
            width = (e_deg - s_deg) % 360.0
            meds = []
            for i in range(lo, hi + 1):
                vals = sorted(s[i % n_beams] for s in self._scans
                              if math.isfinite(s[i % n_beams]))
                if vals:
                    meds.append(vals[len(vals) // 2])
            med = sum(meds) / len(meds) if meds else float("nan")
            print(f"{k:>8}  {s_deg:>10.1f}  {e_deg:>10.1f}  {width:>10.1f}  {med:>9.3f}")

        flat = []
        for s_deg, e_deg in sectors:
            flat += [round(s_deg, 1), round(e_deg, 1)]
        total = sum((e - s) % 360.0 for s, e in sectors)
        print()
        print(f"Total masked: {total:.1f} deg ({100.0 * total / 360.0:.1f}% of the scan)")
        print("Paste into the launch file:")
        print(f'    "blocked_sectors_deg": {flat},')
        print("Or pass on the command line:")
        print(f"    -p blocked_sectors_deg:={flat}".replace(" ", ""))
        print()
        print("CHECK BEFORE USING: each sector above should match a real part "
              "of the rover (mast, servo arm, camera post). A sector you "
              "cannot account for is probably a real object that was left "
              "sitting next to the rover during the recording.")


def main(args=None):
    rclpy.init(args=args)
    node = SelfViewCharacterisationNode()
    try:
        while rclpy.ok() and not node.done():
            rclpy.spin_once(node, timeout_sec=1.0)
        node.report()
    except KeyboardInterrupt:
        if node._scans:
            node.report()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
