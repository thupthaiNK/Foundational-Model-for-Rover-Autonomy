#!/usr/bin/env python3
"""
Purpose: Live recorder for "semantic-biased frontier selection" (scoped via
         grill-thesis 2026-07-18 night) -- frontier exploration-lite with
         lexicographic bedrock-adjacency priority, the L1-drives-L6 sliver.
         Success criteria fixed BEFORE running, per this thesis's standing
         practice (and the frontier-lite v1/v2 lesson: prefer logical
         invariants over trajectory-shape rules that can fail on benign
         plateaus):
         PRIMARY (pass/fail, both required):
           P1. Zero invariant violations: every planner selection event
               that reports bedrock_candidates>0 must report
               selected_bedrock=True. This is a logical guarantee of the
               lexicographic design -- any violation is a real bug, there
               is no benign way to fail it.
           P2. >=10 selection events with bedrock_candidates>0 within the
               time cap, so P1 is exercised enough times to be meaningful
               evidence rather than a vacuous pass.
         SECONDARY (descriptive, NOT pass/fail): total selections, how many
         selected goals fall in bedrock_zone vs soil_zone, % coverage of
         the 6x6 m box at the cap, ground-truth path length, replan count.
         The comparison against the non-semantic baseline (be39399) is
         illustrative only -- different box, different run, physics not
         deterministic -- and is written up as such, not as a controlled
         comparison.
Inputs:  /exomy/odom (nav_msgs/Odometry) -- ground truth
         /traversability_costmap (nav_msgs/OccupancyGrid) -- -1 = unexplored
         /l5_lite_frontier_goal (geometry_msgs/PointStamped) -- selections
         /l5_lite_plan (nav_msgs/Path) -- for replan counting
         /rosout (rcl_interfaces/Log) -- planner "Frontier N selected" lines
         carrying bedrock_candidates=/selected_bedrock= invariant evidence
Outputs: experiments/results/semantic_frontier_test.csv
         experiments/results/semantic_frontier_selections.csv
         experiments/results/figures/semantic_frontier_test.png
How to run:
    # Terminal 1:
    source /opt/ros/humble/setup.bash && source ros2_ws/install/setup.bash
    ros2 launch simulation/launch/semantic_frontier_test.launch.py

    # Terminal 2 (after l5_lite_planner_node has started):
    source /opt/ros/humble/setup.bash && source ros2_ws/install/setup.bash
    python3 experiments/semantic_frontier_test.py
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
from geometry_msgs.msg import PointStamped
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from rcl_interfaces.msg import Log
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy

from fm_perception.frontier_explorer import count_known_cells_in_box
from fm_perception.traversability_grid import (
    WIDTH_CELLS, world_to_cell, zone_for_point,
)

# Must match semantic_frontier_test.launch.py exactly.
BOX_WORLD = (-3.0, 3.0, 3.0, 9.0)   # x_min, y_min, x_max, y_max
BOX_TOTAL_CELLS = 3600              # 6m x 6m at 0.1m resolution (60x60)
MAX_DURATION_S = 300.0
MIN_BEDROCK_CANDIDATE_EVENTS = 10   # P2

# Planner log line, e.g. "... bedrock_candidates=4, selected_bedrock=True"
SELECTION_RE = re.compile(
    r"Frontier (\d+) selected .*world=\(([-\d.]+), ([-\d.]+)\).*"
    r"bedrock_candidates=(\d+), selected_bedrock=(True|False)")

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")


def cell_box() -> Tuple[int, int, int, int]:
    c1 = world_to_cell(BOX_WORLD[0], BOX_WORLD[1])
    c2 = world_to_cell(BOX_WORLD[2], BOX_WORLD[3])
    return (min(c1[0], c2[0]), min(c1[1], c2[1]),
            max(c1[0], c2[0]), max(c1[1], c2[1]))


class SemanticFrontierRecorder(Node):
    def __init__(self) -> None:
        super().__init__("semantic_frontier_recorder")
        self._box = cell_box()
        self._t0 = time.time()
        self._records: List[Tuple[float, float, float, str, int]] = []
        # Selection events parsed from /rosout:
        # (t_s, index, world_x, world_y, goal_zone, n_bedrock, sel_bedrock)
        self._selections: List[Tuple[float, int, float, float, str, int, bool]] = []
        self._violations = 0
        self._known_cells = 0
        self._gt_xy: Optional[Tuple[float, float]] = None
        self._path_len_m = 0.0
        self._replan_count = 0
        self._goal_events = 0

        costmap_qos = QoSProfile(depth=1)
        costmap_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        costmap_qos.reliability = ReliabilityPolicy.RELIABLE
        self.create_subscription(OccupancyGrid, "/traversability_costmap",
                                  self._costmap_cb, costmap_qos)
        self.create_subscription(Odometry, "/exomy/odom", self._odom_cb, 10)
        self.create_subscription(PointStamped, "/l5_lite_frontier_goal",
                                  self._goal_cb, 10)
        self.create_subscription(Path, "/l5_lite_plan", self._plan_cb, 10)
        self.create_subscription(Log, "/rosout", self._rosout_cb, 50)
        self.create_timer(1.0, self._sample)
        self.get_logger().info(
            f"Recording semantic frontier run: box={BOX_WORLD}, "
            f"cap={MAX_DURATION_S:.0f}s. PRIMARY: P1 zero invariant "
            f"violations, P2 >={MIN_BEDROCK_CANDIDATE_EVENTS} "
            f"bedrock-candidate selection events."
        )

    def _elapsed(self) -> float:
        return time.time() - self._t0

    def _costmap_cb(self, msg: OccupancyGrid) -> None:
        self._known_cells = count_known_cells_in_box(
            list(msg.data), WIDTH_CELLS, self._box)

    def _odom_cb(self, msg: Odometry) -> None:
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        if self._gt_xy is not None:
            self._path_len_m += math.hypot(x - self._gt_xy[0],
                                            y - self._gt_xy[1])
        self._gt_xy = (x, y)

    def _goal_cb(self, msg: PointStamped) -> None:
        self._goal_events += 1

    def _plan_cb(self, msg: Path) -> None:
        self._replan_count += 1

    def _rosout_cb(self, msg: Log) -> None:
        if msg.name != "l5_lite_planner_node":
            return
        m = SELECTION_RE.search(msg.msg)
        if not m:
            return
        idx = int(m.group(1))
        wx, wy = float(m.group(2)), float(m.group(3))
        n_bedrock = int(m.group(4))
        sel_bedrock = m.group(5) == "True"
        zone = zone_for_point(wx, wy)
        self._selections.append(
            (self._elapsed(), idx, wx, wy, zone, n_bedrock, sel_bedrock))
        if n_bedrock > 0 and not sel_bedrock:
            self._violations += 1
            self.get_logger().error(
                f"INVARIANT VIOLATION at selection {idx}: "
                f"bedrock_candidates={n_bedrock} but selected_bedrock=False")
        else:
            self.get_logger().info(
                f"Selection {idx}: goal=({wx:.2f},{wy:.2f}) zone={zone} "
                f"bedrock_candidates={n_bedrock} selected_bedrock={sel_bedrock}")

    def _sample(self) -> None:
        if self._gt_xy is None:
            return
        x, y = self._gt_xy
        self._records.append(
            (self._elapsed(), x, y, zone_for_point(x, y), self._known_cells))


def main() -> None:
    os.makedirs(FIGURES_DIR, exist_ok=True)
    rclpy.init()
    node = SemanticFrontierRecorder()
    try:
        while rclpy.ok() and node._elapsed() < MAX_DURATION_S:
            rclpy.spin_once(node, timeout_sec=0.2)
    except KeyboardInterrupt:
        pass

    # -- Save raw records ----------------------------------------------------
    csv_path = os.path.join(RESULTS_DIR, "semantic_frontier_test.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["t_s", "gt_x", "gt_y", "zone", "known_cells_in_box"])
        writer.writerows(node._records)

    sel_path = os.path.join(RESULTS_DIR, "semantic_frontier_selections.csv")
    with open(sel_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["t_s", "selection", "goal_x", "goal_y", "goal_zone",
                         "bedrock_candidates", "selected_bedrock"])
        writer.writerows(node._selections)

    # -- Figure: trajectory + selections, coverage over time -----------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    xs = [r[1] for r in node._records]
    ys = [r[2] for r in node._records]
    ax1.plot(xs, ys, "-", linewidth=1, label="ground-truth path")
    bx0, by0, bx1, by1 = BOX_WORLD
    ax1.plot([bx0, bx1, bx1, bx0, bx0], [by0, by0, by1, by1, by0],
             "k--", linewidth=1, label="exploration box")
    ax1.axvline(0.0, color="grey", linewidth=0.8, linestyle=":",
                label="soil/bedrock boundary")
    sel_b = [(s[2], s[3]) for s in node._selections if s[6]]
    sel_p = [(s[2], s[3]) for s in node._selections if not s[6]]
    if sel_b:
        ax1.scatter(*zip(*sel_b), marker="*", s=80, zorder=3,
                    label="bedrock-adjacent goal")
    if sel_p:
        ax1.scatter(*zip(*sel_p), marker="o", s=30, zorder=3,
                    label="fallback goal")
    ax1.set_xlabel("x (m)")
    ax1.set_ylabel("y (m)")
    ax1.set_title("Semantic frontier run: path + selected goals")
    ax1.legend(fontsize=8)
    ax1.set_aspect("equal")

    ts = [r[0] for r in node._records]
    ks = [r[4] for r in node._records]
    ax2.plot(ts, ks, "-", linewidth=1.2)
    for s in node._selections:
        ax2.axvline(s[0], color="grey", linewidth=0.5, alpha=0.5)
    ax2.set_xlabel("t (s)")
    ax2.set_ylabel("known cells in box")
    ax2.set_title("Coverage over time (grey = selections)")
    fig.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, "semantic_frontier_test.png")
    fig.savefig(fig_path, dpi=150)

    # -- Verdict (live summary is authoritative, not the rounded CSVs) -------
    n_sel = len(node._selections)
    n_bedrock_events = sum(1 for s in node._selections if s[5] > 0)
    n_goal_bedrock_zone = sum(1 for s in node._selections
                               if s[4] == "bedrock_zone")
    p1 = node._violations == 0
    p2 = n_bedrock_events >= MIN_BEDROCK_CANDIDATE_EVENTS
    print("=" * 70)
    print(f"Selections: {n_sel} "
          f"(goal in bedrock_zone: {n_goal_bedrock_zone}, "
          f"soil_zone: {n_sel - n_goal_bedrock_zone})")
    print(f"P1 zero invariant violations: {node._violations} violations "
          f"-> {'PASS' if p1 else 'FAIL'}")
    print(f"P2 bedrock-candidate events: {n_bedrock_events} "
          f"(need >={MIN_BEDROCK_CANDIDATE_EVENTS}) "
          f"-> {'PASS' if p2 else 'FAIL'}")
    print(f"OVERALL: {'PASS' if (p1 and p2) else 'FAIL'}")
    print(f"Coverage at cap: {node._known_cells}/{BOX_TOTAL_CELLS} "
          f"({100.0 * node._known_cells / BOX_TOTAL_CELLS:.1f}%)")
    print(f"Ground-truth path length: {node._path_len_m:.2f} m")
    print(f"Replans: {node._replan_count}, goal events: {node._goal_events}")
    print(f"Saved: {csv_path}")
    print(f"Saved: {sel_path}")
    print(f"Saved: {fig_path}")
    print("=" * 70)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
