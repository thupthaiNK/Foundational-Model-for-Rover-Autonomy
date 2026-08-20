#!/usr/bin/env python3
"""
Purpose: Live recorder + failsafe injector for "mission-level failsafe
         reaction" (item 12, grill-scoped 2026-07-20): a reactive_explorer
         terminal FAILSAFE (rover still mobile, unlike stuck_detection's
         FAILSAFE) should autonomously abort the current frontier
         exploration and drive back to the rover's own recorded start pose,
         instead of holding zero Twist forever until the mission timeout.
         reactive_explorer_node is deliberately NOT running in this test's
         launch file (abort_to_home_test.launch.py) -- the planner cannot
         tell an injected /reactive_explorer/failsafe Bool from a genuine
         one, and reactive_explorer's own hazard-triggering logic is
         already verified elsewhere (28 unit tests + live). This recorder
         publishes /reactive_explorer/failsafe=True itself, once, after a
         fixed number of autonomous frontier selections have been observed
         on /rosout (so exploration is genuinely under way before the
         injection, not an immediate abort at t=0), then verifies the
         planner's response.
         PRE-REGISTERED CRITERIA (fixed before this run):
           P1. The planner logs its own autonomous abort-to-home transition
               ("Reactive failsafe -- aborting exploration, returning
               home...") within MAX_DURATION_S of the injection.
           P2. The rover's ground-truth position is within
               HOME_TOLERANCE_M of its own recorded first pose (home) by
               the end of the run, AND the planner logs the distinct
               "mission aborted after reactive failsafe, returned home"
               completion line (not the ordinary explore-then-return-home
               line, which this run's abort_to_home=True/return_home=False
               configuration cannot produce).
         SECONDARY (descriptive, not pass/fail): frontier selections before
         injection, time from injection to abort log, time from abort to
         arrival, path length, replans.
Inputs:  /exomy/odom (nav_msgs/Odometry) -- ground truth
         /rosout (rcl_interfaces/Log) -- P1/P2 evidence + frontier count
         /l5_lite_plan (nav_msgs/Path) -- replan counting
Outputs: /reactive_explorer/failsafe (std_msgs/Bool) -- this script injects
             the one and only publish of True
         experiments/results/abort_to_home_test.csv
         experiments/results/figures/abort_to_home_test.png
How to run:
    # Terminal 1:
    source /opt/ros/humble/setup.bash && source ros2_ws/install/setup.bash
    ros2 launch simulation/launch/abort_to_home_test.launch.py

    # Terminal 2 (after l5_lite_planner_node has started):
    source /opt/ros/humble/setup.bash && source ros2_ws/install/setup.bash
    python3 experiments/abort_to_home_test.py
Project: Onboard Visual Foundation Models for Mars Terrain Perception and Rover Navigation
"""
import csv
import math
import os
import re
import time
from typing import List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import rclpy
from nav_msgs.msg import Odometry, Path
from rcl_interfaces.msg import Log
from rclpy.node import Node
from std_msgs.msg import Bool

# Inject after this many autonomous frontier selections (must be under way,
# not an immediate abort at t=0).
INJECT_AFTER_N_FRONTIERS = 3
# Generous cap: the home pose is the spawn point inside a 6x6 m box, so the
# return leg is short; margin covers slow exploration before injection too.
MAX_DURATION_S = 400.0
HOME_TOLERANCE_M = 0.3  # slightly looser than the planner's own goal_tolerance_m
                        # default (0.2) to give this recorder's own distance
                        # check margin against odom/pose noise

FRONTIER_RE = re.compile(r"Frontier (\d+) selected \(no operator command\)")
ABORT_RE = re.compile(
    r"Reactive failsafe -- aborting exploration, returning home "
    r"to \(([-\d.]+), ([-\d.]+)\)")
RETURNED_ABORT_RE = re.compile(
    r"Returned home \(([-\d.]+), ([-\d.]+)\) -- mission aborted after "
    r"reactive failsafe, returned home")

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")


def path_length_m(poses: List[Tuple[float, float]]) -> float:
    total = 0.0
    for (x0, y0), (x1, y1) in zip(poses, poses[1:]):
        total += math.hypot(x1 - x0, y1 - y0)
    return total


class AbortToHomeRecorder(Node):

    def __init__(self):
        super().__init__("abort_to_home_test")
        self._t0 = time.time()
        self._gt_positions: List[Tuple[float, float]] = []
        self._records = []
        self._replan_count = 0
        self._last_plan_len = None
        self._home_xy: Optional[Tuple[float, float]] = None
        self._first_pose_seen = False
        self._frontier_count = 0
        self._injected = False
        self._inject_t: Optional[float] = None
        self._abort_t: Optional[float] = None
        self._abort_goal: Optional[Tuple[float, float]] = None
        self._arrived_t: Optional[float] = None
        self._arrived_goal: Optional[Tuple[float, float]] = None

        self.create_subscription(Odometry, "/exomy/odom", self._on_odom, 50)
        self.create_subscription(Path, "/l5_lite_plan", self._on_plan, 10)
        self.create_subscription(Log, "/rosout", self._on_rosout, 50)
        self.pub_failsafe = self.create_publisher(Bool, "/reactive_explorer/failsafe", 10)

        self.get_logger().info(
            f"Recording abort-to-home run: inject after "
            f"{INJECT_AFTER_N_FRONTIERS} frontier selections, "
            f"cap={MAX_DURATION_S:.0f}s. PRIMARY: P1 abort log fires, "
            f"P2 arrives home within {HOME_TOLERANCE_M}m and logs the "
            f"distinct abort-recovery completion line."
        )

    def elapsed(self) -> float:
        return time.time() - self._t0

    def _on_odom(self, msg: Odometry) -> None:
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        if not self._first_pose_seen:
            self._first_pose_seen = True
            self._home_xy = (x, y)
            self.get_logger().info(f"First /exomy/odom pose recorded as home: ({x:.2f}, {y:.2f})")
        self._gt_positions.append((x, y))
        self._records.append((round(self.elapsed(), 2), round(x, 3), round(y, 3)))

    def _on_plan(self, msg: Path) -> None:
        n = len(msg.poses)
        if self._last_plan_len is not None and n != self._last_plan_len:
            self._replan_count += 1
        self._last_plan_len = n

    def _on_rosout(self, msg: Log) -> None:
        if msg.name != "l5_lite_planner_node":
            return
        m = FRONTIER_RE.search(msg.msg)
        if m:
            self._frontier_count = int(m.group(1))
            if not self._injected and self._frontier_count >= INJECT_AFTER_N_FRONTIERS:
                self._inject_failsafe()
            return
        m = ABORT_RE.search(msg.msg)
        if m and self._abort_t is None:
            self._abort_t = self.elapsed()
            self._abort_goal = (float(m.group(1)), float(m.group(2)))
            self.get_logger().info(
                f"P1 EVENT at t={self._abort_t:.1f}s: planner logged autonomous "
                f"abort-to-home, goal=({self._abort_goal[0]:.2f}, {self._abort_goal[1]:.2f})"
            )
            return
        m = RETURNED_ABORT_RE.search(msg.msg)
        if m and self._arrived_t is None:
            self._arrived_t = self.elapsed()
            self._arrived_goal = (float(m.group(1)), float(m.group(2)))
            self.get_logger().info(
                f"P2 EVENT at t={self._arrived_t:.1f}s: planner logged the distinct "
                f"abort-recovery completion line"
            )

    def _inject_failsafe(self) -> None:
        self._injected = True
        self._inject_t = self.elapsed()
        msg = Bool()
        msg.data = True
        self.pub_failsafe.publish(msg)
        self.get_logger().info(
            f"INJECTED /reactive_explorer/failsafe=True at t={self._inject_t:.1f}s "
            f"(after {self._frontier_count} frontier selections)"
        )


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    rclpy.init()
    node = AbortToHomeRecorder()
    try:
        while rclpy.ok() and node.elapsed() < MAX_DURATION_S:
            rclpy.spin_once(node, timeout_sec=0.2)
    except KeyboardInterrupt:
        pass

    csv_path = os.path.join(RESULTS_DIR, "abort_to_home_test.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["t_s", "gt_x", "gt_y"])
        writer.writerows(node._records)

    fig, ax = plt.subplots(figsize=(7, 6))
    if node._gt_positions:
        xs, ys = zip(*node._gt_positions)
        ax.plot(xs, ys, "b-", linewidth=1, label="ground-truth path")
        ax.plot(xs[0], ys[0], "go", markersize=10, label="start / home")
    if node._abort_goal:
        ax.plot(node._abort_goal[0], node._abort_goal[1], "r*", markersize=14,
                 label="abort-to-home goal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Mission-level failsafe: abort-to-home test")
    ax.legend(fontsize=8)
    ax.axis("equal")
    fig_path = os.path.join(FIGURES_DIR, "abort_to_home_test.png")
    fig.tight_layout()
    fig.savefig(fig_path, dpi=100)

    # -- Verdict -------------------------------------------------------------
    p1 = node._abort_t is not None
    final_pos = node._gt_positions[-1] if node._gt_positions else None
    home_dist = (math.hypot(final_pos[0] - node._home_xy[0], final_pos[1] - node._home_xy[1])
                 if final_pos and node._home_xy else None)
    p2 = (node._arrived_t is not None and home_dist is not None
          and home_dist <= HOME_TOLERANCE_M)

    print("\n" + "=" * 70)
    print("MISSION-LEVEL FAILSAFE (ITEM 12) -- ABORT-TO-HOME TEST SUMMARY")
    print(f"Injected after {INJECT_AFTER_N_FRONTIERS} frontier selections, cap={MAX_DURATION_S:.0f}s")
    if node._injected:
        print(f"Injection: t={node._inject_t:.1f}s (after {node._frontier_count} selections)")
    else:
        print("Injection: NEVER FIRED -- exploration never reached the selection threshold")
    if p1:
        print(f"P1 (abort log fires): PASS at t={node._abort_t:.1f}s "
              f"({node._abort_t - node._inject_t:.1f}s after injection)")
    else:
        print("P1 (abort log fires): FAIL -- never observed")
    if p2:
        print(f"P2 (arrives home within {HOME_TOLERANCE_M}m + distinct completion log): "
              f"PASS at t={node._arrived_t:.1f}s, final distance to home={home_dist:.3f}m")
    else:
        reason = []
        if node._arrived_t is None:
            reason.append("completion log never observed")
        if home_dist is not None and home_dist > HOME_TOLERANCE_M:
            reason.append(f"final distance to home {home_dist:.3f}m > {HOME_TOLERANCE_M}m")
        print(f"P2 (arrives home): FAIL -- {', '.join(reason) if reason else 'unknown'}")
    print(f"OVERALL: {'PASS' if (p1 and p2) else 'FAIL'}")
    print("-- Descriptive --")
    print(f"Frontier selections before injection: {node._frontier_count}")
    if node._abort_t is not None and node._arrived_t is not None:
        print(f"Abort -> arrival duration: {node._arrived_t - node._abort_t:.1f}s")
    print(f"Ground-truth path length: {path_length_m(node._gt_positions):.2f} m")
    print(f"Replans: {node._replan_count}")
    print(f"Saved: {csv_path}")
    print(f"Saved: {fig_path}")
    print("=" * 70)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
