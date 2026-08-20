"""
Purpose: Live Gazebo verification of traversability_score_fusion_node.py's
         LiDAR term (backlog item 8, scoped via grill-thesis 2026-07-17).
         Drives straight into the Q4 rock_cluster (spawn 2.0,-4.0, §3.11.4)
         at a fixed low speed -- cmd_vel is published directly by this
         script, bypassing terrain_controller_node.py entirely (not run by
         traversability_fusion_lidar_test.launch.py), mirroring the L4
         Phase A isolation precedent so the reactive safety stack cannot
         fight this script for /exomy/cmd_vel. Records /scan,
         /traversability_score, and /traversability_score_fused together,
         independently computes the expected fused value using the node's
         own already-unit-tested lidar_range_risk()/fuse_traversability_score()
         functions applied to the raw /scan reading (not read from the node
         under test), and compares -- checks the live wiring (subscriptions,
         callback correctness, timing), not the formula itself (already
         unit-tested in test_traversability_score_fusion.py).
Inputs:  /scan (sensor_msgs/LaserScan)
         /traversability_score (std_msgs/Float64)
         /traversability_score_fused (std_msgs/Float64)
         /exomy/cmd_vel (published by this script, not subscribed)
Outputs: experiments/results/traversability_fusion_lidar_live_test.csv
         experiments/results/figures/traversability_fusion_lidar_live_test.png
How to run:
    # Terminal 1:
    source /opt/ros/humble/setup.bash && source ros2_ws/install/setup.bash
    ros2 launch simulation/launch/traversability_fusion_lidar_test.launch.py

    # Terminal 2 (after Terminal 1 shows the fusion node has started):
    source /opt/ros/humble/setup.bash && source ros2_ws/install/setup.bash
    python3 experiments/traversability_fusion_lidar_live_test.py
Project: Onboard Visual Foundation Models for Mars Terrain Perception and Rover Navigation
"""
import csv
import math
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float64

from fm_imu_fusion.traversability_score_fusion_node import (
    lidar_range_risk, fuse_traversability_score,
)

DRIVE_LINEAR_SPEED = 0.10   # m/s -- matches this thesis's established speed ceiling
DURATION_S = 25.0           # §3.11.4: 25s at 0.10 m/s toward Q4 already confirmed a collision

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)


def _min_valid_range(ranges, range_min: float, range_max: float) -> float:
    valid = [r for r in ranges if math.isfinite(r) and range_min <= r <= range_max]
    return min(valid) if valid else float("inf")


class LidarFusionRecorder(Node):

    def __init__(self):
        super().__init__("traversability_fusion_lidar_live_test")
        self._records = []
        self._t0 = time.time()
        self._latest_dinov2_score = None
        self._latest_min_range = None

        self.create_subscription(LaserScan, "/scan", self._scan_cb, 10)
        self.create_subscription(Float64, "/traversability_score", self._score_cb, 10)
        self.create_subscription(Float64, "/traversability_score_fused", self._fused_cb, 10)
        self.pub_cmd = self.create_publisher(Twist, "/exomy/cmd_vel", 10)

    def _scan_cb(self, msg: LaserScan) -> None:
        self._latest_min_range = _min_valid_range(msg.ranges, msg.range_min, msg.range_max)

    def _score_cb(self, msg: Float64) -> None:
        self._latest_dinov2_score = msg.data

    def _fused_cb(self, msg: Float64) -> None:
        if self._latest_dinov2_score is None or self._latest_min_range is None:
            return  # not enough data yet to compute an expected value
        expected_lidar_risk = lidar_range_risk(self._latest_min_range)
        expected_fused = fuse_traversability_score(
            self._latest_dinov2_score, expected_lidar_risk, imu_risk=0.0
        )
        self._records.append({
            "t_s": round(time.time() - self._t0, 3),
            "min_range_m": round(self._latest_min_range, 3),
            "dinov2_score": round(self._latest_dinov2_score, 4),
            "expected_lidar_risk": round(expected_lidar_risk, 4),
            "expected_fused": round(expected_fused, 4),
            "actual_fused": round(msg.data, 4),
        })

    def send_twist(self, linear_x: float) -> None:
        msg = Twist()
        msg.linear.x = linear_x
        self.pub_cmd.publish(msg)


def main():
    rclpy.init()
    node = LidarFusionRecorder()
    node.get_logger().info(
        f"Driving forward at {DRIVE_LINEAR_SPEED} m/s toward Q4 rock_cluster for "
        f"{DURATION_S}s, recording LiDAR-fusion live verification data."
    )

    end_t = time.time() + DURATION_S
    while time.time() < end_t:
        node.send_twist(DRIVE_LINEAR_SPEED)
        rclpy.spin_once(node, timeout_sec=0.1)
    node.send_twist(0.0)

    csv_path = os.path.join(RESULTS_DIR, "traversability_fusion_lidar_live_test.csv")
    with open(csv_path, "w", newline="") as f:
        if node._records:
            w = csv.DictWriter(f, fieldnames=list(node._records[0].keys()))
            w.writeheader()
            w.writerows(node._records)
    node.get_logger().info(f"Saved {len(node._records)} records -> {csv_path}")

    print("\n" + "=" * 70)
    print("LIDAR-FUSION LIVE TEST SUMMARY")
    print(f"Total /traversability_score_fused messages recorded: {len(node._records)}")
    if node._records:
        errors = [abs(r["actual_fused"] - r["expected_fused"]) for r in node._records]
        min_range_seen = min(r["min_range_m"] for r in node._records)
        max_lidar_risk_seen = max(r["expected_lidar_risk"] for r in node._records)
        times_fused_exceeds_dinov2 = sum(
            1 for r in node._records if r["actual_fused"] > r["dinov2_score"] + 1e-6
        )
        print(f"Closest LiDAR range observed: {min_range_seen:.3f} m")
        print(f"Max LiDAR risk term observed: {max_lidar_risk_seen:.4f}")
        print(f"Messages where fused > dinov2-only score: {times_fused_exceeds_dinov2}/{len(node._records)}")
        print(f"Mean |actual_fused - expected_fused|: {sum(errors)/len(errors):.4f}")
        print(f"Max  |actual_fused - expected_fused|: {max(errors):.4f}")
        if max_lidar_risk_seen < 0.01:
            print("WARNING: LiDAR risk term never rose above ~0 -- rover may not have "
                  "gotten close enough to Q4's rocks during this run.")
    print("=" * 70)

    if node._records:
        t = [r["t_s"] for r in node._records]
        actual = [r["actual_fused"] for r in node._records]
        expected = [r["expected_fused"] for r in node._records]
        dinov2 = [r["dinov2_score"] for r in node._records]
        plt.figure(figsize=(10, 5))
        plt.plot(t, actual, label="/traversability_score_fused (actual)", marker=".")
        plt.plot(t, expected, label="expected fused (independent calc)", linestyle="--")
        plt.plot(t, dinov2, label="/traversability_score (DINOv2-only)", linestyle=":")
        plt.xlabel("Time (s)")
        plt.ylabel("Risk score (0=safe, 1=stop)")
        plt.title("LiDAR-fusion live test: actual vs. expected (Q4 rock_cluster approach)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        fig_path = os.path.join(FIGURES_DIR, "traversability_fusion_lidar_live_test.png")
        plt.savefig(fig_path, dpi=120, bbox_inches="tight")
        node.get_logger().info(f"Figure saved -> {fig_path}")

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
