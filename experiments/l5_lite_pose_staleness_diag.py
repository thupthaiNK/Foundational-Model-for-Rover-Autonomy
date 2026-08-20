"""
Purpose: Diagnostic-only script (systematic-debugging, 2026-07-17 reopened
         session) testing whether L5-lite's earlier "rotation doesn't work"
         conclusion was actually a SLAM pose-staleness artifact instead.
         l5_lite_planner_node.py's control loop uses slam_toolbox's /pose
         for robot_yaw, but slam_toolbox only publishes an updated pose
         once minimum_travel_heading=0.2 rad (~11.5deg) has accumulated
         since the last update (matching the known update-sparsity already
         documented for Phase A, §4.8.22: "10-11 pose updates per 90s").
         If the rover is genuinely rotating but /pose just hasn't updated
         yet, ground-truth /exomy/odom yaw will diverge from /pose yaw
         during the same window. NOT a claim about which is correct in
         general -- purely a diagnostic comparison for this one question.
Inputs:  /exomy/odom (nav_msgs/Odometry) -- ground truth, diagnostic only
         /pose (geometry_msgs/PoseWithCovarianceStamped) -- slam_toolbox estimate
Outputs: Console log comparing both yaw values over time.
How to run:
    # Terminal 1:
    source /opt/ros/humble/setup.bash && source ros2_ws/install/setup.bash
    ros2 launch simulation/launch/l5_lite_minimal_diag.launch.py

    # Terminal 2 (after l5_lite_planner_node ready):
    source /opt/ros/humble/setup.bash && source ros2_ws/install/setup.bash
    python3 experiments/l5_lite_pose_staleness_diag.py
Project: Onboard Visual Foundation Models for Mars Terrain Perception and Rover Navigation
"""
import math
import time

import rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node

DURATION_S = 90.0


def _yaw_from_quaternion(q) -> float:
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


class PoseStalenessDiag(Node):

    def __init__(self):
        super().__init__("l5_lite_pose_staleness_diag")
        self._t0 = time.time()
        self._gt_yaw = None
        self._gt_count = 0
        self._slam_yaw = None
        self._slam_count = 0
        self._last_slam_change_t = None
        self.create_subscription(Odometry, "/exomy/odom", self._gt_cb, 10)
        self.create_subscription(PoseWithCovarianceStamped, "/pose", self._slam_cb, 10)
        self.create_timer(2.0, self._report)

    def _gt_cb(self, msg: Odometry) -> None:
        self._gt_yaw = _yaw_from_quaternion(msg.pose.pose.orientation)
        self._gt_count += 1

    def _slam_cb(self, msg: PoseWithCovarianceStamped) -> None:
        new_yaw = _yaw_from_quaternion(msg.pose.pose.orientation)
        if self._slam_yaw is None or abs(new_yaw - self._slam_yaw) > 1e-6:
            self._last_slam_change_t = time.time() - self._t0
        self._slam_yaw = new_yaw
        self._slam_count += 1

    def _report(self) -> None:
        elapsed = time.time() - self._t0
        gt_deg = math.degrees(self._gt_yaw) if self._gt_yaw is not None else None
        slam_deg = math.degrees(self._slam_yaw) if self._slam_yaw is not None else None
        self.get_logger().info(
            f"t={elapsed:.1f}s | ground_truth_yaw={gt_deg} (n={self._gt_count}) | "
            f"slam_yaw={slam_deg} (n={self._slam_count}, last_changed_at={self._last_slam_change_t})"
        )


def main():
    rclpy.init()
    node = PoseStalenessDiag()
    node.get_logger().info(f"Comparing ground-truth vs SLAM yaw for {DURATION_S}s...")
    end_t = time.time() + DURATION_S
    while time.time() < end_t:
        rclpy.spin_once(node, timeout_sec=0.2)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
