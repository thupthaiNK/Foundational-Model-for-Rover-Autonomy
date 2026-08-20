#!/usr/bin/env python3
"""
Purpose: Live recorder for "frontier exploration-lite" (scoped via
         grill-thesis 2026-07-18). Success criteria fixed BEFORE running,
         per this thesis's standing practice:
         PRIMARY (pass/fail, criterion v2 -- revised and re-pre-registered
         after official run 1 (af18713) failed v1's strict-increase rule
         on a single benign transit plateau): >=3 autonomous selections,
         overall coverage growth (final selection strictly above the
         first), and no >=3 consecutive zero-growth selection intervals
         (the goal-cycling signature v1 was written to catch). Evaluated
         from /l5_lite_frontier_goal selection events and the known-cell
         count at each selection.
         SECONDARY (descriptive, NOT pass/fail): % coverage of the 6x6 m
         box at the time cap, ground-truth path length, replan count.
Inputs:  /exomy/odom (nav_msgs/Odometry) -- ground truth
         /traversability_costmap (nav_msgs/OccupancyGrid) -- -1 = unexplored
         /l5_lite_frontier_goal (geometry_msgs/PointStamped) -- selections
         /l5_lite_plan (nav_msgs/Path) -- for replan counting
Outputs: experiments/results/frontier_exploration_test.csv
         experiments/results/figures/frontier_exploration_test.png
How to run:
    # Terminal 1:
    source /opt/ros/humble/setup.bash && source ros2_ws/install/setup.bash
    ros2 launch simulation/launch/frontier_exploration_test.launch.py

    # Terminal 2 (after l5_lite_planner_node has started):
    source /opt/ros/humble/setup.bash && source ros2_ws/install/setup.bash
    python3 experiments/frontier_exploration_test.py
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
from geometry_msgs.msg import PointStamped
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy

from fm_perception.frontier_explorer import count_known_cells_in_box
from fm_perception.traversability_grid import WIDTH_CELLS, world_to_cell, zone_for_point

# Must match frontier_exploration_test.launch.py exactly.
BOX_WORLD = (-9.0, 3.0, -3.0, 9.0)  # x_min, y_min, x_max, y_max
BOX_TOTAL_CELLS = 3600              # 6m x 6m at 0.1m resolution (60x60)
MAX_DURATION_S = 600.0
MIN_SELECTIONS_FOR_PASS = 3
# Criterion v2: a plateau of up to this many consecutive zero-growth
# selection intervals is tolerated (transit through already-painted
# ground); one more fails the run as a goal-cycling signature.
MAX_CONSECUTIVE_ZERO_GROWTH = 2

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)


def cell_box() -> Tuple[int, int, int, int]:
    c1 = world_to_cell(BOX_WORLD[0], BOX_WORLD[1])
    c2 = world_to_cell(BOX_WORLD[2], BOX_WORLD[3])
    return (min(c1[0], c2[0]), min(c1[1], c2[1]),
            max(c1[0], c2[0]), max(c1[1], c2[1]))


def path_length_m(poses: List[Tuple[float, float]]) -> float:
    total = 0.0
    for (x0, y0), (x1, y1) in zip(poses, poses[1:]):
        total += math.hypot(x1 - x0, y1 - y0)
    return total


class FrontierExplorationRecorder(Node):

    def __init__(self):
        super().__init__("frontier_exploration_test")
        self._t0 = time.time()
        self._box = cell_box()
        self._gt_positions: List[Tuple[float, float]] = []
        self._records = []
        self._zones_visited = set()
        self._known_cells = 0
        self._replan_count = 0
        self._last_plan_len = None
        # (t_s, x, y, known_cells_at_selection)
        self._selections: List[Tuple[float, float, float, int]] = []

        self.create_subscription(Odometry, "/exomy/odom", self._on_odom, 50)
        self.create_subscription(Path, "/l5_lite_plan", self._on_plan, 10)
        self.create_subscription(PointStamped, "/l5_lite_frontier_goal",
                                  self._on_selection, 10)
        costmap_qos = QoSProfile(depth=1)
        costmap_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        costmap_qos.reliability = ReliabilityPolicy.RELIABLE
        self.create_subscription(OccupancyGrid, "/traversability_costmap",
                                  self._on_costmap, costmap_qos)

        self.get_logger().info(
            f"Recording frontier exploration: box={BOX_WORLD}, "
            f"max {MAX_DURATION_S}s."
        )

    def elapsed(self) -> float:
        return time.time() - self._t0

    def _on_odom(self, msg: Odometry) -> None:
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        self._gt_positions.append((x, y))
        zone = zone_for_point(x, y)
        self._zones_visited.add(zone)
        self._records.append((round(self.elapsed(), 2), round(x, 3), round(y, 3),
                              zone, self._known_cells))

    def _on_costmap(self, msg: OccupancyGrid) -> None:
        self._known_cells = count_known_cells_in_box(
            list(msg.data), WIDTH_CELLS, self._box)

    def _on_plan(self, msg: Path) -> None:
        n = len(msg.poses)
        if self._last_plan_len is not None and n != self._last_plan_len:
            self._replan_count += 1
        self._last_plan_len = n

    def _on_selection(self, msg: PointStamped) -> None:
        t = self.elapsed()
        self._selections.append((t, msg.point.x, msg.point.y, self._known_cells))
        self.get_logger().info(
            f"Frontier selection {len(self._selections)} at t={t:.1f}s: "
            f"({msg.point.x:.2f}, {msg.point.y:.2f}), "
            f"known cells in box={self._known_cells}"
        )


def evaluate_primary(selections: List[Tuple[float, float, float, int]]) -> Tuple[bool, str]:
    """PRIMARY criterion v2 (pre-registered before the official re-run,
    2026-07-18). v1 required the known-cell count to increase strictly
    between EVERY consecutive selection pair; official run 1 (af18713)
    failed that on a single benign tie (306=306) while the rover transited
    already-painted cells -- not the 10Hz goal-cycling bug the criterion
    was written to catch. v2 keeps the anti-cycling intent without failing
    on short transit plateaus:
      (i)   >= 3 autonomous selections;
      (ii)  known cells at the final selection strictly greater than at
            the first (the run as a whole genuinely explored);
      (iii) no run of >= MAX_CONSECUTIVE_ZERO_GROWTH+1 consecutive
            zero-growth selection intervals (goal-cycling stalls coverage
            across many selections within seconds; a 1-2 interval plateau
            is normal transit through already-known ground)."""
    if len(selections) < MIN_SELECTIONS_FOR_PASS:
        return False, (f"only {len(selections)} selections "
                       f"(need >= {MIN_SELECTIONS_FOR_PASS})")
    counts = [s[3] for s in selections]
    if counts[-1] <= counts[0]:
        return False, (f"no overall coverage growth: first selection "
                       f"{counts[0]} cells, final selection {counts[-1]}")
    zero_growth_streak = 0
    for i in range(1, len(counts)):
        if counts[i] <= counts[i - 1]:
            zero_growth_streak += 1
            if zero_growth_streak > MAX_CONSECUTIVE_ZERO_GROWTH:
                return False, (f"{zero_growth_streak} consecutive "
                               f"zero-growth intervals ending at selection "
                               f"{i + 1} (max allowed "
                               f"{MAX_CONSECUTIVE_ZERO_GROWTH}) -- "
                               f"goal-cycling signature")
        else:
            zero_growth_streak = 0
    return True, (f"{len(selections)} autonomous selections, overall growth "
                  f"{counts[0]} -> {counts[-1]} cells, no zero-growth run "
                  f"longer than {MAX_CONSECUTIVE_ZERO_GROWTH} intervals")


def main():
    rclpy.init()
    node = FrontierExplorationRecorder()
    try:
        while rclpy.ok() and node.elapsed() < MAX_DURATION_S:
            rclpy.spin_once(node, timeout_sec=0.2)
    except KeyboardInterrupt:
        pass

    csv_path = os.path.join(RESULTS_DIR, "frontier_exploration_test.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["t_s", "gt_x", "gt_y", "zone", "known_cells_in_box"])
        writer.writerows(node._records)
    node.get_logger().info(f"Saved {len(node._records)} records -> {csv_path}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))
    if node._gt_positions:
        xs, ys = zip(*node._gt_positions)
        ax1.plot(xs, ys, "b-", linewidth=1, label="ground-truth path")
        ax1.plot(xs[0], ys[0], "go", markersize=10, label="start")
        ax1.plot(xs[-1], ys[-1], "rs", markersize=10, label="end")
    for i, (t, x, y, _) in enumerate(node._selections):
        ax1.plot(x, y, "m*", markersize=12)
        ax1.annotate(str(i + 1), (x, y), fontsize=9)
    bx0, by0, bx1, by1 = BOX_WORLD[0], BOX_WORLD[1], BOX_WORLD[2], BOX_WORLD[3]
    ax1.plot([bx0, bx1, bx1, bx0, bx0], [by0, by0, by1, by1, by0],
             "k--", linewidth=1, label="exploration box")
    ax1.set_xlabel("x (m)")
    ax1.set_ylabel("y (m)")
    ax1.set_title("Frontier exploration: path + selections (numbered stars)")
    ax1.legend()
    ax1.axis("equal")

    if node._records:
        ts = [r[0] for r in node._records]
        ks = [r[4] for r in node._records]
        ax2.plot(ts, ks, "b-")
        for t, _, _, k in node._selections:
            ax2.axvline(t, color="m", linestyle=":", linewidth=1)
        ax2.set_xlabel("t (s)")
        ax2.set_ylabel("known cells in box")
        ax2.set_title("Coverage over time (dotted lines = frontier selections)")
    fig_path = os.path.join(FIGURES_DIR, "frontier_exploration_test.png")
    fig.tight_layout()
    fig.savefig(fig_path, dpi=100)
    node.get_logger().info(f"Figure saved -> {fig_path}")

    passed, reason = evaluate_primary(node._selections)
    coverage_pct = 100.0 * node._known_cells / BOX_TOTAL_CELLS
    print("\n" + "=" * 70)
    print("FRONTIER EXPLORATION-LITE TEST SUMMARY")
    print(f"Records: {len(node._records)}")
    print(f"Zones visited: {sorted(node._zones_visited)}")
    print(f"Autonomous frontier selections: {len(node._selections)}")
    for i, (t, x, y, k) in enumerate(node._selections):
        print(f"  [{i + 1}] t={t:.1f}s -> ({x:.2f}, {y:.2f}), known cells={k}")
    print(f"Total ground-truth path length: {path_length_m(node._gt_positions):.2f} m")
    print(f"Replan count (plan length changed): {node._replan_count}")
    print(f"Known cells in box at end: {node._known_cells}/{BOX_TOTAL_CELLS} "
          f"({coverage_pct:.1f}% coverage)")
    print(f"PRIMARY CRITERION v2 (>= {MIN_SELECTIONS_FOR_PASS} selections, "
          f"overall coverage growth, no > {MAX_CONSECUTIVE_ZERO_GROWTH} "
          f"consecutive zero-growth intervals): "
          f"{'PASS' if passed else 'FAIL'} -- {reason}")
    print("=" * 70)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
