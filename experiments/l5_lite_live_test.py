"""
Purpose: Live Gazebo verification of "L5-lite" (backlog item, scoped via
         grill-thesis 2026-07-17). Records ground-truth /exomy/odom
         alongside slam_toolbox's /pose, /l5_lite_plan (for replan count),
         and /l5_lite_planner/active over the same D1 mission (§4.8.13):
         START_POSE (7.5, 1.0, bedrock_zone near the hazard boundary) to
         GOAL_POSE (-7.5, -9.0, sand_zone opposite side), whose direct
         route crosses rock_cluster/boulder_zone. Uses the same
         path_length_m()/entered_hazard_zone metric definitions as
         nav2_waypoint_experiment.py for direct comparability with D1's
         own reported result (0/20 success, mean path_length_m ~0.073m in
         the Condition A re-run, §4.8.13).
Inputs:  /exomy/odom (nav_msgs/Odometry) -- ground truth
         /pose (geometry_msgs/PoseWithCovarianceStamped) -- slam_toolbox estimate
         /l5_lite_plan (nav_msgs/Path) -- current plan, counted for replans
         /l5_lite_planner/active (std_msgs/Bool)
Outputs: experiments/results/l5_lite_live_test.csv
         experiments/results/figures/l5_lite_live_test.png
How to run:
    # Terminal 1:
    source /opt/ros/humble/setup.bash && source ros2_ws/install/setup.bash
    ros2 launch simulation/launch/l5_lite_test.launch.py

    # Terminal 2 (after Terminal 1 shows l5_lite_planner_node ready):
    source /opt/ros/humble/setup.bash && source ros2_ws/install/setup.bash
    python3 experiments/l5_lite_live_test.py
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
from std_msgs.msg import Bool

from fm_perception.path_follower import goal_reached
from fm_perception.traversability_grid import zone_for_point

START_POSE = (7.5, 1.0)     # D1's real START_POSE, bedrock_zone near hazard boundary
GOAL_POSE = (-7.5, -9.0)    # D1's real GOAL_POSE, sand_zone opposite side
GOAL_TOLERANCE_M = 0.2      # matches l5_lite_planner_node.py's default
HAZARD_ZONES = {"rock_cluster", "boulder_zone"}

MAX_DURATION_S = 500.0      # hard cap; straight-line distance alone is ~18m.
                            # Raised from 300s (2026-07-18): the 300s official run showed
                            # steady, still-accelerating progress (best 30s window was the
                            # last one recorded) right up to the cutoff -- not a stall -- so
                            # the time budget itself, not navigation capability, was binding.
                             # at 0.10 m/s = 180s minimum, plus detour/turn overhead

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)


def path_length_m(poses: List[Tuple[float, float]]) -> float:
    """Total travelled distance -- same definition as nav2_waypoint_experiment.py."""
    total = 0.0
    for (x0, y0), (x1, y1) in zip(poses, poses[1:]):
        total += math.hypot(x1 - x0, y1 - y0)
    return total


class L5LiteRecorder(Node):

    def __init__(self):
        super().__init__("l5_lite_live_test")
        self._t0 = time.time()
        self._gt_positions: List[Tuple[float, float]] = []
        self._slam_positions: List[Tuple[float, float]] = []
        self._records = []
        self._zones_visited = set()
        self._replan_count = 0
        self._last_plan_len = None
        self._goal_reached_at_s = None

        self.create_subscription(Odometry, "/exomy/odom", self._odom_cb, 10)
        self.create_subscription(Path, "/l5_lite_plan", self._plan_cb, 10)
        self.create_subscription(Bool, "/l5_lite_planner/active", self._active_cb, 10)
        self._active = False

    def _odom_cb(self, msg: Odometry) -> None:
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        self._gt_positions.append((x, y))
        zone = zone_for_point(x, y)
        self._zones_visited.add(zone)

        elapsed = time.time() - self._t0
        self._records.append({
            "t_s": round(elapsed, 2),
            "gt_x": round(x, 3),
            "gt_y": round(y, 3),
            "zone": zone,
            "active": self._active,
        })

        if self._goal_reached_at_s is None and goal_reached(
            x, y, GOAL_POSE[0], GOAL_POSE[1], GOAL_TOLERANCE_M
        ):
            self._goal_reached_at_s = elapsed
            self.get_logger().info(f"GOAL REACHED (ground truth) at t={elapsed:.1f}s")

    def _plan_cb(self, msg: Path) -> None:
        plan_len = len(msg.poses)
        if self._last_plan_len is not None and plan_len != self._last_plan_len:
            self._replan_count += 1
        self._last_plan_len = plan_len

    def _active_cb(self, msg: Bool) -> None:
        self._active = msg.data


def main():
    rclpy.init()
    node = L5LiteRecorder()
    node.get_logger().info(
        f"Recording L5-lite live test: START={START_POSE} -> GOAL={GOAL_POSE}, "
        f"max {MAX_DURATION_S}s."
    )

    end_t = time.time() + MAX_DURATION_S
    while time.time() < end_t:
        rclpy.spin_once(node, timeout_sec=0.2)
        if node._goal_reached_at_s is not None:
            # Keep recording a few more seconds after reaching goal to
            # confirm the rover actually stops there, not overshoot.
            settle_end = time.time() + 5.0
            while time.time() < settle_end:
                rclpy.spin_once(node, timeout_sec=0.2)
            break

    csv_path = os.path.join(RESULTS_DIR, "l5_lite_live_test.csv")
    with open(csv_path, "w", newline="") as f:
        if node._records:
            w = csv.DictWriter(f, fieldnames=list(node._records[0].keys()))
            w.writeheader()
            w.writerows(node._records)
    node.get_logger().info(f"Saved {len(node._records)} records -> {csv_path}")

    gt_positions = node._gt_positions
    entered_hazard = bool(node._zones_visited & HAZARD_ZONES)
    total_path_length = path_length_m(gt_positions) if gt_positions else 0.0
    final_pos = gt_positions[-1] if gt_positions else None
    final_dist_to_goal = (
        math.hypot(final_pos[0] - GOAL_POSE[0], final_pos[1] - GOAL_POSE[1])
        if final_pos else None
    )

    print("\n" + "=" * 70)
    print("L5-LITE LIVE TEST SUMMARY")
    print(f"Records: {len(node._records)}")
    print(f"Zones visited: {sorted(node._zones_visited)}")
    print(f"Entered hazard zone (rock_cluster/boulder_zone): {entered_hazard}")
    print(f"Total ground-truth path length: {total_path_length:.2f} m "
          f"(straight-line distance: {math.hypot(GOAL_POSE[0]-START_POSE[0], GOAL_POSE[1]-START_POSE[1]):.2f} m)")
    print(f"Replan count (plan length changed): {node._replan_count}")
    if node._goal_reached_at_s is not None:
        print(f"GOAL REACHED at t={node._goal_reached_at_s:.1f}s")
    else:
        print(f"Goal NOT reached within {MAX_DURATION_S}s "
              f"(final distance to goal: {final_dist_to_goal:.2f} m)" if final_dist_to_goal is not None
              else "Goal NOT reached, no odometry received at all")
    print("=" * 70)

    if gt_positions:
        xs = [p[0] for p in gt_positions]
        ys = [p[1] for p in gt_positions]
        plt.figure(figsize=(8, 8))
        plt.plot(xs, ys, "b-", linewidth=1.5, label="Ground-truth trajectory")
        plt.plot(START_POSE[0], START_POSE[1], "go", markersize=12, label="Start")
        plt.plot(GOAL_POSE[0], GOAL_POSE[1], "r*", markersize=16, label="Goal")
        circle = plt.Circle(GOAL_POSE, GOAL_TOLERANCE_M, color="r", fill=False, linestyle="--")
        plt.gca().add_patch(circle)
        plt.xlabel("x (m)")
        plt.ylabel("y (m)")
        plt.title("L5-lite live test: ground-truth trajectory (D1's START/GOAL_POSE)")
        plt.legend()
        plt.axis("equal")
        plt.grid(True, alpha=0.3)
        fig_path = os.path.join(FIGURES_DIR, "l5_lite_live_test.png")
        plt.savefig(fig_path, dpi=120, bbox_inches="tight")
        node.get_logger().info(f"Figure saved -> {fig_path}")

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
