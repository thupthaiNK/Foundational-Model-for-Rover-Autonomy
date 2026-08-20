#!/usr/bin/env python3
"""
Purpose: Live recorder for "explore-then-return-home" (scoped via
         grill-thesis 2026-07-19, item 5 of the L1-L6 further-work plan) --
         wires frontier exploration-lite's own "no frontiers remain" signal
         into L6-lite's autonomous-switch mechanism: once the rover finishes
         exploring the bounded box it autonomously drives back to its own
         recorded start pose, with no operator command at either transition.
         Success criteria fixed BEFORE this official run, per this thesis's
         standing practice: a smoke test (2026-07-19, ~1400s launch,
         explore_return_smoke2.log) on this exact box/launch observed a
         1194.2s exploration phase followed by a 34.7s return leg (1229.0s
         total from the rover's first recorded pose), zero blacklist events,
         zero errors -- so the cap below has real margin, not a guess.
         PRIMARY (pass/fail, both required):
           P1. The exploration -> return-home transition fires autonomously
               (parsed from the planner's own "Exploration complete...--
               returning home" /rosout line) within MAX_DURATION_S.
           P2. The return-home leg itself completes (parsed from the
               planner's own "Returned home...mission complete" /rosout
               line) within MAX_DURATION_S. Mirrors L6-lite's own precedent
               (§4.8.27): P1 alone already matches L6-lite's own success
               bar (the autonomous switch itself); P2 is pre-registered as
               its own binary check, not left merely descriptive, because
               the smoke test already showed it working end-to-end.
         SECONDARY (descriptive, NOT pass/fail): total frontier selections,
         % coverage of the box at completion, ground-truth path length,
         replan count, and the wall-clock split between the two phases.
Inputs:  /exomy/odom (nav_msgs/Odometry) -- ground truth
         /traversability_costmap (nav_msgs/OccupancyGrid) -- -1 = unexplored
         /l5_lite_frontier_goal (geometry_msgs/PointStamped) -- frontier
             selection events only; the return-home goal is not published
             here (it is set internally, not via a fresh frontier
             selection), so it is detected via /rosout instead, below
         /l5_lite_plan (nav_msgs/Path) -- for replan counting
         /rosout (rcl_interfaces/Log) -- the planner's own "Exploration
             complete...returning home" / "Returned home...mission
             complete" lines carry the two pass/fail events directly, no
             new topic needed -- same pattern semantic_frontier_test.py
             already uses for its own invariant-evidence log fields
Outputs: experiments/results/explore_return_home_test.csv
         experiments/results/explore_return_home_selections.csv
         experiments/results/figures/explore_return_home_test.png
How to run:
    # Terminal 1:
    source /opt/ros/humble/setup.bash && source ros2_ws/install/setup.bash
    ros2 launch simulation/launch/explore_return_home_test.launch.py

    # Terminal 2 (after l5_lite_planner_node has started):
    source /opt/ros/humble/setup.bash && source ros2_ws/install/setup.bash
    python3 experiments/explore_return_home_test.py
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
from fm_perception.traversability_grid import WIDTH_CELLS, world_to_cell, zone_for_point

# Must match explore_return_home_test.launch.py exactly.
BOX_WORLD = (-7.5, 4.5, -4.5, 7.5)  # x_min, y_min, x_max, y_max
# ~46% margin over the smoke test's observed 1229.0s total (2026-07-19),
# not a round-number guess -- see the module docstring.
MAX_DURATION_S = 1800.0

EXPLORATION_COMPLETE_RE = re.compile(
    r"Exploration complete: no frontiers remain after (\d+) autonomous "
    r"selections \((\d+) known cells in box, (\d+) blacklisted\) -- "
    r"returning home")
RETURNED_HOME_RE = re.compile(
    r"Returned home .* explore-then-return-home mission complete")

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")


def cell_box() -> Tuple[int, int, int, int]:
    c1 = world_to_cell(BOX_WORLD[0], BOX_WORLD[1])
    c2 = world_to_cell(BOX_WORLD[2], BOX_WORLD[3])
    return (min(c1[0], c2[0]), min(c1[1], c2[1]),
            max(c1[0], c2[0]), max(c1[1], c2[1]))


BOX_CELLS = cell_box()
# Inclusive bounds at each edge (matches count_known_cells_in_box's own
# convention) -- for a 3m x 3m box at 0.1m resolution this is 31x31=961,
# not the naive 30x30=900 a plain width/height product would suggest.
BOX_TOTAL_CELLS = (BOX_CELLS[2] - BOX_CELLS[0] + 1) * (BOX_CELLS[3] - BOX_CELLS[1] + 1)


def path_length_m(poses: List[Tuple[float, float]]) -> float:
    total = 0.0
    for (x0, y0), (x1, y1) in zip(poses, poses[1:]):
        total += math.hypot(x1 - x0, y1 - y0)
    return total


class ExploreReturnHomeRecorder(Node):

    def __init__(self):
        super().__init__("explore_return_home_test")
        self._t0 = time.time()
        self._box = BOX_CELLS
        self._gt_positions: List[Tuple[float, float]] = []
        self._records = []
        self._zones_visited = set()
        self._known_cells = 0
        self._replan_count = 0
        self._last_plan_len = None
        # (t_s, x, y, known_cells_at_selection)
        self._selections: List[Tuple[float, float, float, int]] = []
        self._exploration_complete_t: Optional[float] = None
        self._exploration_complete_info: Optional[Tuple[int, int, int]] = None
        self._returned_home_t: Optional[float] = None

        self.create_subscription(Odometry, "/exomy/odom", self._on_odom, 50)
        self.create_subscription(Path, "/l5_lite_plan", self._on_plan, 10)
        self.create_subscription(PointStamped, "/l5_lite_frontier_goal",
                                  self._on_selection, 10)
        self.create_subscription(Log, "/rosout", self._on_rosout, 50)
        costmap_qos = QoSProfile(depth=1)
        costmap_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        costmap_qos.reliability = ReliabilityPolicy.RELIABLE
        self.create_subscription(OccupancyGrid, "/traversability_costmap",
                                  self._on_costmap, costmap_qos)

        self.get_logger().info(
            f"Recording explore-then-return-home: box={BOX_WORLD} "
            f"({BOX_TOTAL_CELLS} cells), cap={MAX_DURATION_S:.0f}s. "
            f"PRIMARY: P1 exploration->return-home transition fires, "
            f"P2 return-home leg completes."
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

    def _on_rosout(self, msg: Log) -> None:
        if msg.name != "l5_lite_planner_node":
            return
        if self._exploration_complete_t is None:
            m = EXPLORATION_COMPLETE_RE.search(msg.msg)
            if m:
                self._exploration_complete_t = self.elapsed()
                self._exploration_complete_info = (
                    int(m.group(1)), int(m.group(2)), int(m.group(3)))
                self.get_logger().info(
                    f"P1 EVENT at t={self._exploration_complete_t:.1f}s: "
                    f"exploration complete, transitioning to return-home"
                )
        if self._returned_home_t is None:
            if RETURNED_HOME_RE.search(msg.msg):
                self._returned_home_t = self.elapsed()
                self.get_logger().info(
                    f"P2 EVENT at t={self._returned_home_t:.1f}s: returned home"
                )


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    rclpy.init()
    node = ExploreReturnHomeRecorder()
    try:
        while rclpy.ok() and node.elapsed() < MAX_DURATION_S:
            rclpy.spin_once(node, timeout_sec=0.2)
    except KeyboardInterrupt:
        pass

    # -- Save raw records ----------------------------------------------------
    csv_path = os.path.join(RESULTS_DIR, "explore_return_home_test.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["t_s", "gt_x", "gt_y", "zone", "known_cells_in_box"])
        writer.writerows(node._records)

    sel_path = os.path.join(RESULTS_DIR, "explore_return_home_selections.csv")
    with open(sel_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["t_s", "goal_x", "goal_y", "known_cells_in_box"])
        writer.writerows(node._selections)

    # -- Figure: trajectory + selections, coverage over time -----------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))
    if node._gt_positions:
        xs, ys = zip(*node._gt_positions)
        ax1.plot(xs, ys, "b-", linewidth=1, label="ground-truth path")
        ax1.plot(xs[0], ys[0], "go", markersize=10, label="start / home")
        ax1.plot(xs[-1], ys[-1], "rs", markersize=10, label="end")
    for i, (t, x, y, _) in enumerate(node._selections):
        ax1.plot(x, y, "m*", markersize=12)
        ax1.annotate(str(i + 1), (x, y), fontsize=8)
    bx0, by0, bx1, by1 = BOX_WORLD
    ax1.plot([bx0, bx1, bx1, bx0, bx0], [by0, by0, by1, by1, by0],
             "k--", linewidth=1, label="exploration box")
    ax1.set_xlabel("x (m)")
    ax1.set_ylabel("y (m)")
    ax1.set_title("Explore-then-return-home: path + selections (numbered stars)")
    ax1.legend(fontsize=8)
    ax1.axis("equal")

    if node._records:
        ts = [r[0] for r in node._records]
        ks = [r[4] for r in node._records]
        ax2.plot(ts, ks, "b-")
        for t, _, _, _ in node._selections:
            ax2.axvline(t, color="m", linestyle=":", linewidth=0.6)
        if node._exploration_complete_t is not None:
            ax2.axvline(node._exploration_complete_t, color="green", linewidth=1.5,
                        label="P1: exploration complete / return-home starts")
        if node._returned_home_t is not None:
            ax2.axvline(node._returned_home_t, color="red", linewidth=1.5,
                        label="P2: returned home / mission complete")
        ax2.set_xlabel("t (s)")
        ax2.set_ylabel("known cells in box")
        ax2.set_title("Coverage over time")
        ax2.legend(fontsize=8)
    fig_path = os.path.join(FIGURES_DIR, "explore_return_home_test.png")
    fig.tight_layout()
    fig.savefig(fig_path, dpi=100)

    # -- Verdict ---------------------------------------------------------
    p1 = node._exploration_complete_t is not None
    p2 = node._returned_home_t is not None
    coverage_pct = 100.0 * node._known_cells / BOX_TOTAL_CELLS
    print("\n" + "=" * 70)
    print("EXPLORE-THEN-RETURN-HOME TEST SUMMARY")
    print(f"Box: {BOX_WORLD} ({BOX_TOTAL_CELLS} cells), cap={MAX_DURATION_S:.0f}s")
    print(f"Zones visited: {sorted(node._zones_visited)}")
    print(f"Autonomous frontier selections: {len(node._selections)}")
    for i, (t, x, y, k) in enumerate(node._selections):
        print(f"  [{i + 1}] t={t:.1f}s -> ({x:.2f}, {y:.2f}), known cells={k}")
    print(f"Total ground-truth path length: {path_length_m(node._gt_positions):.2f} m")
    print(f"Replan count (plan length changed): {node._replan_count}")
    print(f"Known cells in box at end: {node._known_cells}/{BOX_TOTAL_CELLS} "
          f"({coverage_pct:.1f}% coverage)")
    if p1:
        n_sel, known_at_complete, n_blacklisted = node._exploration_complete_info
        print(f"P1 (exploration->return-home transition fires by "
              f"{MAX_DURATION_S:.0f}s): PASS at t={node._exploration_complete_t:.1f}s "
              f"({n_sel} selections, {known_at_complete} known cells, "
              f"{n_blacklisted} blacklisted)")
    else:
        print(f"P1 (exploration->return-home transition fires by "
              f"{MAX_DURATION_S:.0f}s): FAIL -- never observed")
    if p2:
        return_leg_s = node._returned_home_t - node._exploration_complete_t if p1 else None
        extra = f" (return leg took {return_leg_s:.1f}s)" if return_leg_s is not None else ""
        print(f"P2 (return-home leg completes by {MAX_DURATION_S:.0f}s): "
              f"PASS at t={node._returned_home_t:.1f}s{extra}")
    else:
        print(f"P2 (return-home leg completes by {MAX_DURATION_S:.0f}s): "
              f"FAIL -- never observed")
    print(f"OVERALL: {'PASS' if (p1 and p2) else 'FAIL'}")
    print(f"Saved: {csv_path}")
    print(f"Saved: {sel_path}")
    print(f"Saved: {fig_path}")
    print("=" * 70)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
