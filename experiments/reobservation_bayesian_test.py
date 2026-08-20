#!/usr/bin/env python3
"""
Purpose: Live recorder for "A2 re-observation mode" WITH bayesian_fusion
         enabled (SuperMap-inspired log-odds label fusion) -- an honest,
         direct comparison against reobservation_test.py's latest-write
         baseline (§4.8.30) on the identical arena, box, mission, and
         success criteria; the only pipeline change is the costmap's
         bayesian_fusion:=true parameter (reobservation_bayesian_test.
         launch.py). Same code, same recorder logic, different output
         filenames, so both runs' evidence coexists rather than one
         overwriting the other.
         PRIMARY (pass/fail, all three required, identical to baseline):
           P1. The exploration -> re-observation transition fires
               autonomously within MAX_DURATION_S.
           P2. Zero invariant violations: every "Reobservation N selected"
               line must report selected_conf == pool_min.
           P3. >= 10 re-observation selections within the cap.
         SECONDARY (descriptive, NOT pass/fail -- comparison against the
         baseline's mean confidence delta is the point of this run, not a
         pre-registered "fusion must win" claim): per-revisited-cell
         confidence delta (value at selection vs final grid value at cap),
         counts raised/lowered/unchanged, frontier selections, coverage,
         path length, replans, and grid_with_start_freed escape count.
Inputs:  /exomy/odom (nav_msgs/Odometry) -- ground truth
         /traversability_costmap (nav_msgs/OccupancyGrid) -- coverage
         /traversability_confidence (nav_msgs/OccupancyGrid) -- final
             per-cell confidence for the delta analysis
         /l5_lite_frontier_goal, /l5_lite_reobserve_goal (PointStamped) --
             topic-level counts, cross-checked against the planner's own
             /rosout log which is authoritative
         /l5_lite_plan (nav_msgs/Path) -- replan counting
         /rosout (rcl_interfaces/Log) -- P1/P2/P3 evidence
Outputs: experiments/results/reobservation_bayesian_test.csv
         experiments/results/reobservation_bayesian_selections.csv
         experiments/results/figures/reobservation_bayesian_test.png
How to run:
    # Terminal 1:
    source /opt/ros/humble/setup.bash && source ros2_ws/install/setup.bash
    ros2 launch simulation/launch/reobservation_bayesian_test.launch.py

    # Terminal 2 (after l5_lite_planner_node has started):
    source /opt/ros/humble/setup.bash && source ros2_ws/install/setup.bash
    python3 experiments/reobservation_bayesian_test.py
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
from fm_perception.traversability_grid import WIDTH_CELLS, world_to_cell

# Must match reobservation_test.launch.py exactly.
BOX_WORLD = (-1.5, 4.5, 1.5, 7.5)   # x_min, y_min, x_max, y_max
# ~46% margin over the smoke test's observed 1085.7s exploration phase,
# leaving ~600s of re-observation at the observed ~31.7s/revisit pace.
MAX_DURATION_S = 1800.0
MIN_REOBSERVATIONS = 10             # P3 (smoke pace projects ~20)

TRANSITION_RE = re.compile(
    r"Exploration complete: no frontiers remain after (\d+) autonomous "
    r"selections \((\d+) known cells in box, (\d+) blacklisted\) -- "
    r"entering re-observation mode")
REOBS_RE = re.compile(
    r"Reobservation (\d+) selected \(no operator command\): "
    r"cell=\((\d+), (\d+)\) world=\(([-\d.]+), ([-\d.]+)\), "
    r"selected_conf=(-?\d+), pool_min=(-?\d+), pool_size=(\d+)")
FRONTIER_RE = re.compile(r"Frontier (\d+) selected \(no operator command\)")
ESCAPE_RE = re.compile(r"was hazard-painted -- escaped")

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")


def cell_box() -> Tuple[int, int, int, int]:
    c1 = world_to_cell(BOX_WORLD[0], BOX_WORLD[1])
    c2 = world_to_cell(BOX_WORLD[2], BOX_WORLD[3])
    return (min(c1[0], c2[0]), min(c1[1], c2[1]),
            max(c1[0], c2[0]), max(c1[1], c2[1]))


BOX_CELLS = cell_box()
BOX_TOTAL_CELLS = (BOX_CELLS[2] - BOX_CELLS[0] + 1) * (BOX_CELLS[3] - BOX_CELLS[1] + 1)


def path_length_m(poses: List[Tuple[float, float]]) -> float:
    total = 0.0
    for (x0, y0), (x1, y1) in zip(poses, poses[1:]):
        total += math.hypot(x1 - x0, y1 - y0)
    return total


class ReobservationBayesianRecorder(Node):

    def __init__(self):
        super().__init__("reobservation_bayesian_test")
        self._t0 = time.time()
        self._box = BOX_CELLS
        self._gt_positions: List[Tuple[float, float]] = []
        self._records = []
        self._known_cells = 0
        self._replan_count = 0
        self._last_plan_len = None
        self._confidence_grid: Optional[List[int]] = None
        self._frontier_topic_count = 0
        self._reobserve_topic_count = 0
        # From /rosout (authoritative): (t_s, index, col, row, wx, wy,
        # selected_conf, pool_min, pool_size)
        self._reobs_selections: List[Tuple] = []
        self._frontier_log_count = 0
        self._violations = 0
        self._escape_count = 0
        self._transition_t: Optional[float] = None
        self._transition_info: Optional[Tuple[int, int, int]] = None

        self.create_subscription(Odometry, "/exomy/odom", self._on_odom, 50)
        self.create_subscription(Path, "/l5_lite_plan", self._on_plan, 10)
        self.create_subscription(PointStamped, "/l5_lite_frontier_goal",
                                  self._on_frontier_goal, 10)
        self.create_subscription(PointStamped, "/l5_lite_reobserve_goal",
                                  self._on_reobserve_goal, 10)
        self.create_subscription(Log, "/rosout", self._on_rosout, 50)
        costmap_qos = QoSProfile(depth=1)
        costmap_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        costmap_qos.reliability = ReliabilityPolicy.RELIABLE
        self.create_subscription(OccupancyGrid, "/traversability_costmap",
                                  self._on_costmap, costmap_qos)
        self.create_subscription(OccupancyGrid, "/traversability_confidence",
                                  self._on_confidence, costmap_qos)

        self.get_logger().info(
            f"Recording re-observation run: box={BOX_WORLD} "
            f"({BOX_TOTAL_CELLS} cells), cap={MAX_DURATION_S:.0f}s. "
            f"PRIMARY: P1 transition fires, P2 zero invariant violations, "
            f"P3 >={MIN_REOBSERVATIONS} re-observation selections."
        )

    def elapsed(self) -> float:
        return time.time() - self._t0

    def _on_odom(self, msg: Odometry) -> None:
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        self._gt_positions.append((x, y))
        self._records.append((round(self.elapsed(), 2), round(x, 3), round(y, 3),
                              self._known_cells))

    def _on_costmap(self, msg: OccupancyGrid) -> None:
        self._known_cells = count_known_cells_in_box(
            list(msg.data), WIDTH_CELLS, self._box)

    def _on_confidence(self, msg: OccupancyGrid) -> None:
        self._confidence_grid = list(msg.data)

    def _on_plan(self, msg: Path) -> None:
        n = len(msg.poses)
        if self._last_plan_len is not None and n != self._last_plan_len:
            self._replan_count += 1
        self._last_plan_len = n

    def _on_frontier_goal(self, msg: PointStamped) -> None:
        self._frontier_topic_count += 1

    def _on_reobserve_goal(self, msg: PointStamped) -> None:
        self._reobserve_topic_count += 1

    def _on_rosout(self, msg: Log) -> None:
        if msg.name != "l5_lite_planner_node":
            return
        if FRONTIER_RE.search(msg.msg):
            self._frontier_log_count += 1
            return
        if ESCAPE_RE.search(msg.msg):
            self._escape_count += 1
            return
        if self._transition_t is None:
            m = TRANSITION_RE.search(msg.msg)
            if m:
                self._transition_t = self.elapsed()
                self._transition_info = (int(m.group(1)), int(m.group(2)),
                                          int(m.group(3)))
                self.get_logger().info(
                    f"P1 EVENT at t={self._transition_t:.1f}s: entering "
                    f"re-observation mode")
                return
        m = REOBS_RE.search(msg.msg)
        if m:
            idx = int(m.group(1))
            col, row = int(m.group(2)), int(m.group(3))
            wx, wy = float(m.group(4)), float(m.group(5))
            sel_conf = int(m.group(6))
            pool_min = int(m.group(7))
            pool_size = int(m.group(8))
            t = self.elapsed()
            self._reobs_selections.append(
                (t, idx, col, row, wx, wy, sel_conf, pool_min, pool_size))
            if sel_conf != pool_min:
                self._violations += 1
                self.get_logger().error(
                    f"INVARIANT VIOLATION at reobservation {idx}: "
                    f"selected_conf={sel_conf} != pool_min={pool_min}")
            else:
                self.get_logger().info(
                    f"Reobservation {idx} at t={t:.1f}s: cell=({col},{row}) "
                    f"conf={sel_conf} (== pool_min, invariant holds)")


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    rclpy.init()
    node = ReobservationBayesianRecorder()
    try:
        while rclpy.ok() and node.elapsed() < MAX_DURATION_S:
            rclpy.spin_once(node, timeout_sec=0.2)
    except KeyboardInterrupt:
        pass

    # -- Save raw records ----------------------------------------------------
    csv_path = os.path.join(RESULTS_DIR, "reobservation_bayesian_test.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["t_s", "gt_x", "gt_y", "known_cells_in_box"])
        writer.writerows(node._records)

    # Per-revisited-cell confidence delta: value recorded at selection time
    # (from the planner's own log) vs the final confidence grid at the cap.
    deltas = []
    sel_rows = []
    for (t, idx, col, row, wx, wy, sel_conf, pool_min, pool_size) in node._reobs_selections:
        final_conf = None
        if node._confidence_grid is not None:
            final_conf = node._confidence_grid[row * WIDTH_CELLS + col]
        delta = (final_conf - sel_conf) if final_conf is not None else None
        if delta is not None:
            deltas.append(delta)
        sel_rows.append([f"{t:.1f}", idx, col, row, wx, wy, sel_conf,
                         pool_min, pool_size, final_conf,
                         delta if delta is not None else ""])
    sel_path = os.path.join(RESULTS_DIR, "reobservation_bayesian_selections.csv")
    with open(sel_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["t_s", "index", "cell_col", "cell_row", "goal_x",
                         "goal_y", "conf_at_selection", "pool_min",
                         "pool_size", "conf_at_end", "delta"])
        writer.writerows(sel_rows)

    # -- Figure --------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))
    if node._gt_positions:
        xs, ys = zip(*node._gt_positions)
        ax1.plot(xs, ys, "b-", linewidth=1, label="ground-truth path")
        ax1.plot(xs[0], ys[0], "go", markersize=10, label="start")
        ax1.plot(xs[-1], ys[-1], "rs", markersize=10, label="end")
    for (t, idx, col, row, wx, wy, sc, pm, ps) in node._reobs_selections:
        ax1.plot(wx, wy, "m*", markersize=12)
    bx0, by0, bx1, by1 = BOX_WORLD
    ax1.plot([bx0, bx1, bx1, bx0, bx0], [by0, by0, by1, by1, by0],
             "k--", linewidth=1, label="exploration box")
    ax1.axvline(0.0, color="grey", linewidth=0.8, linestyle=":",
                label="soil/bedrock boundary")
    ax1.set_xlabel("x (m)")
    ax1.set_ylabel("y (m)")
    ax1.set_title("Re-observation run: path + re-observation targets (stars)")
    ax1.legend(fontsize=8)
    ax1.axis("equal")

    if node._records:
        ts = [r[0] for r in node._records]
        ks = [r[3] for r in node._records]
        ax2.plot(ts, ks, "b-", label="known cells in box")
        if node._transition_t is not None:
            ax2.axvline(node._transition_t, color="green", linewidth=1.5,
                        label="P1: re-observation begins")
        for (t, *_rest) in node._reobs_selections:
            ax2.axvline(t, color="m", linestyle=":", linewidth=0.5)
        ax2.set_xlabel("t (s)")
        ax2.set_ylabel("known cells in box")
        ax2.set_title("Coverage over time (dotted = re-observation selections)")
        ax2.legend(fontsize=8)
    fig_path = os.path.join(FIGURES_DIR, "reobservation_bayesian_test.png")
    fig.tight_layout()
    fig.savefig(fig_path, dpi=100)

    # -- Verdict -------------------------------------------------------------
    p1 = node._transition_t is not None
    p2 = node._violations == 0
    n_reobs = len(node._reobs_selections)
    p3 = n_reobs >= MIN_REOBSERVATIONS
    raised = sum(1 for d in deltas if d > 0)
    lowered = sum(1 for d in deltas if d < 0)
    unchanged = sum(1 for d in deltas if d == 0)
    coverage_pct = 100.0 * node._known_cells / BOX_TOTAL_CELLS
    print("\n" + "=" * 70)
    print("A2 RE-OBSERVATION TEST SUMMARY (bayesian_fusion=True)")
    print(f"Box: {BOX_WORLD} ({BOX_TOTAL_CELLS} cells), cap={MAX_DURATION_S:.0f}s")
    if p1:
        n_sel, known, blacklisted = node._transition_info
        print(f"P1 (transition fires by {MAX_DURATION_S:.0f}s): PASS at "
              f"t={node._transition_t:.1f}s ({n_sel} frontier selections, "
              f"{known} known cells, {blacklisted} blacklisted)")
    else:
        print(f"P1 (transition fires by {MAX_DURATION_S:.0f}s): FAIL -- never observed")
    print(f"P2 (zero invariant violations): {node._violations} violations "
          f"-> {'PASS' if p2 else 'FAIL'}")
    print(f"P3 (re-observation selections >= {MIN_REOBSERVATIONS}): {n_reobs} "
          f"-> {'PASS' if p3 else 'FAIL'}")
    print(f"OVERALL: {'PASS' if (p1 and p2 and p3) else 'FAIL'}")
    print(f"-- Descriptive --")
    print(f"Frontier selections: {node._frontier_log_count} (log) / "
          f"{node._frontier_topic_count} (topic)")
    print(f"Re-observation selections: {n_reobs} (log) / "
          f"{node._reobserve_topic_count} (topic; log is authoritative)")
    if deltas:
        print(f"Confidence deltas of revisited cells (selection -> end of run): "
              f"raised={raised}, lowered={lowered}, unchanged={unchanged}, "
              f"mean={sum(deltas)/len(deltas):+.1f} (0-100 scale)")
    print(f"Coverage at cap: {node._known_cells}/{BOX_TOTAL_CELLS} "
          f"({coverage_pct:.1f}%)")
    print(f"Ground-truth path length: {path_length_m(node._gt_positions):.2f} m")
    print(f"Replans: {node._replan_count}")
    print(f"grid_with_start_freed escapes: {node._escape_count}")
    print(f"Saved: {csv_path}")
    print(f"Saved: {sel_path}")
    print(f"Saved: {fig_path}")
    print("=" * 70)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
