#!/usr/bin/env python3
"""
Purpose: Quantitative comparison of Nav2 waypoint navigation under a static
         known-hazard costmap (Condition A) vs a live DINOv2-populated
         costmap (Condition B), in the validated 5-zone Gazebo arena.
         Mission: start in bedrock_zone near the hazard boundary (7.5, 1.0),
         goal in sand_zone on the opposite side (-7.5, -9.0) — the direct
         route crosses rock_cluster/boulder_zone, so the only safe path is
         a detour north through soil_zone.
Inputs:  Gazebo + Nav2 stack running (via
         simulation/launch/nav2_waypoint_test.launch.py costmap_mode:=<X>)
Outputs: experiments/results/nav2_waypoint_experiment.csv
How to run:
    # Terminal 1 (per condition):
    source /opt/ros/humble/setup.bash && source ros2_ws/install/setup.bash
    ros2 launch simulation/launch/nav2_waypoint_test.launch.py costmap_mode:=static
    # Terminal 2:
    source /opt/ros/humble/setup.bash && source ros2_ws/install/setup.bash
    python3 experiments/nav2_waypoint_experiment.py --condition static --trials 10
    # repeat with costmap_mode:=live --condition live
Project: Onboard Visual Foundation Models for Mars Terrain Perception and Rover Navigation
"""
import argparse
import csv
import math
import os
import time
from typing import List, Tuple

from fm_perception.traversability_grid import zone_for_point

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy
from gazebo_msgs.srv import DeleteEntity, SpawnEntity
from nav_msgs.msg import Path, Odometry
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped
from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult

START_POSE = (7.5, 1.0)    # bedrock_zone, near the hazard boundary
GOAL_POSE = (-7.5, -9.0)   # sand_zone, opposite side
HAZARD_ZONES = {"rock_cluster", "boulder_zone"}

RESULTS_CSV = os.path.join(
    os.path.dirname(__file__), "results", "nav2_waypoint_experiment.csv"
)


def path_length_m(poses: List[Tuple[float, float]]) -> float:
    """Sum of Euclidean distances between consecutive (x, y) poses."""
    if len(poses) < 2:
        return 0.0
    total = 0.0
    for (x0, y0), (x1, y1) in zip(poses, poses[1:]):
        total += math.hypot(x1 - x0, y1 - y0)
    return total


def _paths_differ(path_a: List[Tuple[float, float]], path_b: List[Tuple[float, float]],
                   position_tol_m: float) -> bool:
    if len(path_a) != len(path_b):
        return True
    for (xa, ya), (xb, yb) in zip(path_a, path_b):
        if math.hypot(xa - xb, ya - yb) > position_tol_m:
            return True
    return False


def count_replans(plans: List[List[Tuple[float, float]]], position_tol_m: float = 0.05) -> int:
    """Count how many times the published /plan path materially changed."""
    if len(plans) <= 1:
        return 0
    replans = 0
    for prev, curr in zip(plans, plans[1:]):
        if _paths_differ(prev, curr, position_tol_m):
            replans += 1
    return replans


def path_entered_hazard_zone(poses: List[Tuple[float, float]]) -> bool:
    """True if any executed pose falls inside rock_cluster or boulder_zone."""
    return any(zone_for_point(x, y) in HAZARD_ZONES for x, y in poses)


def trial_result_row(condition: str, trial: int, success: bool, path_length_m: float,
                      time_to_goal_s: float, replan_count: int, entered_hazard_zone: bool) -> dict:
    return {
        "condition": condition,
        "trial": trial,
        "success": success,
        "path_length_m": path_length_m,
        "time_to_goal_s": time_to_goal_s,
        "replan_count": replan_count,
        "entered_hazard_zone": entered_hazard_zone,
    }


def append_row_to_csv(row: dict, csv_path: str = RESULTS_CSV) -> None:
    file_exists = os.path.isfile(csv_path)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


class TrialRunnerNode(Node):
    """Teleports the rover, tracks /plan and /exomy/odom during one trial."""

    def __init__(self):
        super().__init__("nav2_waypoint_trial_runner")
        self._urdf_xml = None
        self._poses: List[Tuple[float, float]] = []
        self._plans: List[List[Tuple[float, float]]] = []

        # /robot_description is published once, latched (transient_local) by
        # robot_state_publisher — a volatile (default) subscription would
        # never see it if this node starts after that initial publish.
        urdf_qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(String, "/robot_description", self._on_urdf, urdf_qos)
        self.create_subscription(Odometry, "/exomy/odom", self._on_odom, 10)
        self.create_subscription(Path, "/plan", self._on_plan, 10)
        self._delete_cli = self.create_client(DeleteEntity, "/delete_entity")
        self._spawn_cli = self.create_client(SpawnEntity, "/spawn_entity")

    def _on_urdf(self, msg: String):
        self._urdf_xml = msg.data

    def _on_odom(self, msg: Odometry):
        self._poses.append((msg.pose.pose.position.x, msg.pose.pose.position.y))

    def _on_plan(self, msg: Path):
        self._plans.append([(p.pose.position.x, p.pose.position.y) for p in msg.poses])

    def reset_trial_state(self):
        self._poses = []
        self._plans = []

    def teleport(self, x: float, y: float):
        if self._urdf_xml is None:
            self.get_logger().info("Waiting for /robot_description...")
            deadline = time.time() + 8.0
            while self._urdf_xml is None and time.time() < deadline:
                rclpy.spin_once(self, timeout_sec=0.2)
        if self._urdf_xml is None:
            self.get_logger().error("robot_description unavailable; cannot teleport")
            return

        if not self._delete_cli.wait_for_service(timeout_sec=10.0):
            self.get_logger().error("delete_entity not available")
            return
        del_req = DeleteEntity.Request()
        del_req.name = "exomy"
        future = self._delete_cli.call_async(del_req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=15.0)
        time.sleep(1.5)

        if not self._spawn_cli.wait_for_service(timeout_sec=10.0):
            self.get_logger().error("spawn_entity not available")
            return
        sp_req = SpawnEntity.Request()
        sp_req.name = "exomy"
        sp_req.xml = self._urdf_xml
        sp_req.initial_pose.position.x = x
        sp_req.initial_pose.position.y = y
        sp_req.initial_pose.position.z = 0.15
        sp_req.initial_pose.orientation.w = 1.0
        future = self._spawn_cli.call_async(sp_req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=15.0)
        time.sleep(1.5)


def run_trial(runner: TrialRunnerNode, nav: BasicNavigator, trial: int,
              condition: str, timeout_s: float = 90.0) -> dict:
    runner.teleport(*START_POSE)
    runner.reset_trial_state()

    goal = PoseStamped()
    goal.header.frame_id = "map"
    goal.header.stamp = nav.get_clock().now().to_msg()
    goal.pose.position.x = GOAL_POSE[0]
    goal.pose.position.y = GOAL_POSE[1]
    goal.pose.orientation.w = 1.0

    t_start = time.time()
    nav.goToPose(goal)
    while not nav.isTaskComplete():
        rclpy.spin_once(runner, timeout_sec=0.5)
        if time.time() - t_start > timeout_s:
            nav.cancelTask()
            break
    t_end = time.time()

    result = nav.getResult()
    success = result == TaskResult.SUCCEEDED

    return trial_result_row(
        condition=condition,
        trial=trial,
        success=success,
        path_length_m=round(path_length_m(runner._poses), 3),
        time_to_goal_s=round(t_end - t_start, 2),
        replan_count=count_replans(runner._plans),
        entered_hazard_zone=path_entered_hazard_zone(runner._poses),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", choices=["static", "live"], required=True)
    parser.add_argument("--trials", type=int, default=10)
    args = parser.parse_args()

    rclpy.init()
    runner = TrialRunnerNode()
    nav = BasicNavigator()
    # localizer='bt_navigator' REQUIRED — discovered during Task 6 execution
    # 2026-07-01: the default localizer='amcl' makes this call hang forever
    # on a nonexistent amcl/get_state service (this setup has no AMCL by
    # design — ground-truth odom + identity map->odom transform instead).
    nav.waitUntilNav2Active(localizer='bt_navigator')

    try:
        for trial in range(1, args.trials + 1):
            try:
                row = run_trial(runner, nav, trial, args.condition)
            except Exception as exc:
                # A single trial's Gazebo/Nav2 service call can fail (e.g. a
                # flaky delete_entity/spawn_entity RPC); given each trial
                # takes minutes on this dev machine, one exception must not
                # discard every trial run so far — record it as failed and
                # keep going with the batch.
                runner.get_logger().error(
                    f"Trial {trial}/{args.trials} raised {exc!r}; "
                    "recording as failed and continuing"
                )
                row = trial_result_row(
                    condition=args.condition, trial=trial, success=False,
                    path_length_m=round(path_length_m(runner._poses), 3),
                    time_to_goal_s=-1.0,
                    replan_count=count_replans(runner._plans),
                    entered_hazard_zone=path_entered_hazard_zone(runner._poses),
                )
            append_row_to_csv(row)
            print(f"[{args.condition}] trial {trial}/{args.trials}: {row}")
    finally:
        runner.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
