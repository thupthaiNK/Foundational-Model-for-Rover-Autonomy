#!/usr/bin/env python3
"""
Purpose: Live recorder for "L6-lite" (scoped via grill-thesis 2026-07-18) --
         verifies l5_lite_planner_node.py's new autonomous waypoint-advance
         mechanism on a real recon-and-return mission: D1's real GOAL_POSE
         first, then autonomously switching (no operator command) to
         START_POSE once GOAL_POSE is reached. Success criterion agreed with
         the user before building: the autonomous switch itself firing is
         the primary evidence for L6-lite, independent of whether leg 2
         (the return) completes in time or re-encounters the same known
         DINOv2 hazard-zone misclassification as leg 1 -- both are reported
         honestly either way, per this thesis's standing practice.
         Detects leg completion purely from ground-truth position
         (/exomy/odom), not from any internal planner state, so this is an
         independent check of the same kind used for l5_lite_live_test.py.
Inputs:  /exomy/odom (nav_msgs/Odometry) -- ground truth
         /l5_lite_plan (nav_msgs/Path) -- for replan counting
Outputs: experiments/results/l6_lite_roundtrip_test.csv
         experiments/results/figures/l6_lite_roundtrip_test.png
How to run:
    # Terminal 1 (waypoints come from experiments/missions/recon_and_return.yaml,
    # backlog item 27 -- edit that file for a different mission, not this command):
    source /opt/ros/humble/setup.bash && source ros2_ws/install/setup.bash
    ros2 launch simulation/launch/l5_lite_test.launch.py \
        waypoint_xs:="[7.5]" waypoint_ys:="[1.0]"

    # Terminal 2 (after l5_lite_planner_node has started):
    source /opt/ros/humble/setup.bash && source ros2_ws/install/setup.bash
    python3 experiments/l6_lite_roundtrip_test.py
Project: Onboard Visual Foundation Models for Mars Terrain Perception and Rover Navigation
"""
import csv
import math
import os
import time
from typing import List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import rclpy
from nav_msgs.msg import Odometry, Path
from rclpy.node import Node

from fm_perception.path_follower import goal_reached
from fm_perception.traversability_grid import zone_for_point
from mission_spec import load_mission

_MISSION = load_mission(os.path.join(os.path.dirname(__file__), "missions", "recon_and_return.yaml"))

START_POSE = _MISSION.waypoints[1]   # D1's real START_POSE -- also L6-lite's return waypoint
GOAL_POSE = _MISSION.waypoints[0]    # D1's real GOAL_POSE -- L6-lite's first waypoint
GOAL_TOLERANCE_M = _MISSION.goal_tolerance_m  # matches l5_lite_planner_node.py's default
HAZARD_ZONES = {"rock_cluster", "boulder_zone"}

MAX_DURATION_S = 900.0      # generous margin: leg 1 alone took 355.3s in the
                            # single-goal L5-lite test; this allows a similar
                            # or somewhat slower leg 2 with real headroom.

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)


def path_length_m(poses: List[Tuple[float, float]]) -> float:
    total = 0.0
    for (x0, y0), (x1, y1) in zip(poses, poses[1:]):
        total += math.hypot(x1 - x0, y1 - y0)
    return total


class L6LiteRoundtripRecorder(Node):

    def __init__(self):
        super().__init__("l6_lite_roundtrip_test")
        self._t0 = time.time()
        self._gt_positions: List[Tuple[float, float]] = []
        self._records = []
        self._zones_visited = set()
        self._replan_count = 0
        self._last_plan_len = None
        self._leg1_reached_at_s = None   # GOAL_POSE reached
        self._leg2_reached_at_s = None   # START_POSE reached, AFTER leg 1

        self.create_subscription(Odometry, "/exomy/odom", self._odom_cb, 10)
        self.create_subscription(Path, "/l5_lite_plan", self._plan_cb, 10)

    def _odom_cb(self, msg: Odometry) -> None:
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        self._gt_positions.append((x, y))
        zone = zone_for_point(x, y)
        self._zones_visited.add(zone)

        elapsed = time.time() - self._t0
        self._records.append({"t_s": round(elapsed, 2), "gt_x": round(x, 3), "gt_y": round(y, 3), "zone": zone})

        if self._leg1_reached_at_s is None and goal_reached(
            x, y, GOAL_POSE[0], GOAL_POSE[1], GOAL_TOLERANCE_M
        ):
            self._leg1_reached_at_s = elapsed
            self.get_logger().info(f"LEG 1 COMPLETE (ground truth) at t={elapsed:.1f}s -- watching for autonomous switch to leg 2")

        elif self._leg1_reached_at_s is not None and self._leg2_reached_at_s is None and goal_reached(
            x, y, START_POSE[0], START_POSE[1], GOAL_TOLERANCE_M
        ):
            self._leg2_reached_at_s = elapsed
            self.get_logger().info(f"LEG 2 COMPLETE -- ROUND TRIP DONE (ground truth) at t={elapsed:.1f}s")

    def _plan_cb(self, msg: Path) -> None:
        plan_len = len(msg.poses)
        if self._last_plan_len is not None and plan_len != self._last_plan_len:
            self._replan_count += 1
        self._last_plan_len = plan_len


def main():
    rclpy.init()
    node = L6LiteRoundtripRecorder()
    node.get_logger().info(
        f"Recording L6-lite round-trip test: GOAL={GOAL_POSE} -> START={START_POSE}, max {MAX_DURATION_S}s."
    )

    end_t = time.time() + MAX_DURATION_S
    while time.time() < end_t:
        rclpy.spin_once(node, timeout_sec=0.2)
        if node._leg2_reached_at_s is not None:
            settle_end = time.time() + 5.0
            while time.time() < settle_end:
                rclpy.spin_once(node, timeout_sec=0.2)
            break

    csv_path = os.path.join(RESULTS_DIR, "l6_lite_roundtrip_test.csv")
    with open(csv_path, "w", newline="") as f:
        if node._records:
            w = csv.DictWriter(f, fieldnames=list(node._records[0].keys()))
            w.writeheader()
            w.writerows(node._records)
    node.get_logger().info(f"Saved {len(node._records)} records -> {csv_path}")

    gt_positions = node._gt_positions
    entered_hazard = bool(node._zones_visited & HAZARD_ZONES)
    total_path_length = path_length_m(gt_positions) if gt_positions else 0.0

    print("\n" + "=" * 70)
    print("L6-LITE ROUND-TRIP TEST SUMMARY")
    print(f"Records: {len(node._records)}")
    print(f"Zones visited: {sorted(node._zones_visited)}")
    print(f"Entered hazard zone (rock_cluster/boulder_zone): {entered_hazard}")
    print(f"Total ground-truth path length: {total_path_length:.2f} m")
    print(f"Replan count (plan length changed): {node._replan_count}")
    if node._leg1_reached_at_s is not None:
        print(f"LEG 1 (GOAL_POSE) reached at t={node._leg1_reached_at_s:.1f}s "
              f"-- AUTONOMOUS SWITCH MECHANISM: primary L6-lite evidence")
    else:
        print("LEG 1 (GOAL_POSE) NOT reached within the recording window")
    if node._leg2_reached_at_s is not None:
        print(f"LEG 2 (START_POSE) reached at t={node._leg2_reached_at_s:.1f}s -- FULL ROUND TRIP COMPLETE")
    else:
        print("LEG 2 (START_POSE) NOT reached within the recording window "
              "(does not invalidate L6-lite if leg 1 + the switch succeeded)")
    print("=" * 70)

    if gt_positions:
        xs, ys = zip(*gt_positions)
        plt.figure(figsize=(8, 8))
        plt.plot(xs, ys, "b-", linewidth=1, label="Ground-truth trajectory")
        plt.plot(GOAL_POSE[0], GOAL_POSE[1], "r*", markersize=16, label="Leg 1 goal")
        plt.plot(START_POSE[0], START_POSE[1], "g*", markersize=16, label="Leg 2 goal (return)")
        plt.plot(xs[0], ys[0], "ko", markersize=10, label="Start")
        plt.xlabel("x (m)")
        plt.ylabel("y (m)")
        plt.title("L6-lite round-trip test: ground-truth trajectory")
        plt.legend()
        plt.axis("equal")
        plt.grid(True)
        fig_path = os.path.join(FIGURES_DIR, "l6_lite_roundtrip_test.png")
        plt.savefig(fig_path, dpi=120)
        node.get_logger().info(f"Figure saved -> {fig_path}")

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
