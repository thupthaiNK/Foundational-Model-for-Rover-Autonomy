#!/usr/bin/env python3
"""
Purpose: Directly measure the LiDAR's mounting yaw offset relative to the
         rover's physical front, using a wall placed at a known distance
         directly ahead as ground truth -- much simpler than the earlier
         wall-fit/segment-voting approach in
         experiments/lidar_yaw_from_drive.py (which never got below a
         4-8% sharpness ceiling, see project_lidar_yaw_offset_20260724).
         Rather than fitting an angle from noisy geometry, this looks for
         the largest contiguous run of beams reading close to
         EXPECTED_WALL_DISTANCE_M (a flat wall generates many adjacent
         beams at roughly the same range; clutter like a mounting-stand
         leg generates only one or two isolated beams) and reports the
         bearing at the centre of that run -- that bearing IS the yaw
         offset, since we know by construction the wall is at bearing 0
         in the physical (rover-front) frame.
         v2 (2026-07-26, same night): v1 took the single globally-closest
         point, which on the real rover picked up the mast self-view
         (~0.072m) on the first run, and then an isolated mounting-stand
         leg (~0.473m, spread 0.25m among the 20 closest -- clearly not a
         flat surface) on the second, once the self-view was filtered out.
         Both were closer than the actual wall. Searching for a WIDE flat
         cluster near the known wall distance instead of "whatever is
         closest" avoids both failure modes.
Inputs:  /scan (sensor_msgs/LaserScan) -- run this with a flat wall placed
         directly in front of the rover's physical chassis (not the
         LiDAR's guessed "front"), at EXPECTED_WALL_DISTANCE_M below
         (edit this to match your actual measured setup distance before
         running), with as little other clutter as practical within
         +/-1m of that distance in the swept region.
Outputs: Prints the bearing (in degrees, normalised to [-180, 180)) at the
         centre of the largest contiguous run of beams found near
         EXPECTED_WALL_DISTANCE_M, the run's size and mean range, and the
         top few runs found (for sanity-checking against duplicates/noise).
How to run:
    source /opt/ros/humble/setup.bash
    source ros2_ws/install/setup.bash
    python3 experiments/measure_lidar_yaw_offset_direct.py [distance_m] [tolerance_m]
    # distance_m defaults to 1.0, tolerance_m defaults to 0.15 -- pass both
    # to check for a different known landmark (e.g. a person sitting at a
    # known distance to one side) without editing/redeploying this file.
    # (on the Pi, inside the exomy_ros container, after scp-ing this file
    # over -- do not paste this script into a live SSH session, see
    # project_lidar_avoidance_hardware_trial_20260726 on why)
Project: Onboard Visual Foundation Models for Mars Terrain Perception and Rover Navigation
"""
import math
import sys

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan

EXPECTED_WALL_DISTANCE_M = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
TOLERANCE_M = float(sys.argv[2]) if len(sys.argv) > 2 else 0.15
MIN_RUN_LENGTH = 5


class YawOffsetMeasurer(Node):

    def __init__(self):
        super().__init__("lidar_yaw_offset_measurer")
        self.create_subscription(LaserScan, "/scan", self._cb, 10)
        self.done = False

    def _cb(self, msg: LaserScan) -> None:
        if self.done:
            return
        self.done = True

        lo = EXPECTED_WALL_DISTANCE_M - TOLERANCE_M
        hi = EXPECTED_WALL_DISTANCE_M + TOLERANCE_M
        in_band = [
            math.isfinite(r) and lo <= r <= hi for r in msg.ranges
        ]

        # Group contiguous True runs (no wraparound across the array ends --
        # good enough here since we expect the wall near bearing 0, not at
        # the -180/180 boundary).
        runs = []
        start = None
        for i, ok in enumerate(in_band):
            if ok and start is None:
                start = i
            elif not ok and start is not None:
                runs.append((start, i - 1))
                start = None
        if start is not None:
            runs.append((start, len(in_band) - 1))

        runs = [r for r in runs if (r[1] - r[0] + 1) >= MIN_RUN_LENGTH]

        if not runs:
            print(f"No contiguous run of >= {MIN_RUN_LENGTH} beams found within "
                  f"{lo:.2f}-{hi:.2f}m. Either the wall isn't at "
                  f"{EXPECTED_WALL_DISTANCE_M:.2f}m, or it's not flat/large "
                  "enough to register multiple adjacent beams. Edit "
                  "EXPECTED_WALL_DISTANCE_M at the top of this script to match "
                  "your actual setup and re-run, or widen TOLERANCE_M.")
            return

        runs.sort(key=lambda r: r[1] - r[0], reverse=True)

        print("=" * 70)
        print(f"Found {len(runs)} candidate run(s) of >= {MIN_RUN_LENGTH} beams "
              f"within {lo:.2f}-{hi:.2f}m:")
        for start_i, end_i in runs[:5]:
            length = end_i - start_i + 1
            center_i = (start_i + end_i) / 2.0
            bearing_rad = msg.angle_min + center_i * msg.angle_increment
            bearing_deg = math.degrees(bearing_rad)
            bearing_deg_norm = ((bearing_deg + 180.0) % 360.0) - 180.0
            mean_range = sum(msg.ranges[start_i:end_i + 1]) / length
            print(f"  length={length:3d} beams  indices=[{start_i},{end_i}]  "
                  f"mean_range={mean_range:.3f}m  bearing={bearing_deg_norm:.1f} deg")

        best_start, best_end = runs[0]
        best_length = best_end - best_start + 1
        best_center_i = (best_start + best_end) / 2.0
        best_bearing_rad = msg.angle_min + best_center_i * msg.angle_increment
        best_bearing_deg = math.degrees(best_bearing_rad)
        best_bearing_norm = ((best_bearing_deg + 180.0) % 360.0) - 180.0

        print()
        print(f"Largest run: {best_length} beams, bearing = {best_bearing_norm:.1f} deg")
        print()
        print("If the wall was placed directly in front of the rover's "
              f"PHYSICAL chassis: lidar_yaw_offset_deg ~= {best_bearing_norm:.1f}")
        if len(runs) > 1 and (runs[1][1] - runs[1][0]) > best_length * 0.6:
            print("NOTE: a second run of comparable size was also found -- "
                  "check the table above, there may be two flat surfaces "
                  "(e.g. wall + a large box) in range. Prefer the one whose "
                  "mean_range most closely matches your actual measured "
                  "wall distance.")
        print("=" * 70)


def main(args=None):
    rclpy.init(args=args)
    node = YawOffsetMeasurer()
    while not node.done:
        rclpy.spin_once(node, timeout_sec=1.0)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
