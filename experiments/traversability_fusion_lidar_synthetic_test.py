"""
Purpose: Supplementary live verification of traversability_score_fusion_node.py's
         LiDAR risk ramp (backlog item 8, scoped via grill-thesis 2026-07-17).
         The Gazebo physical test (traversability_fusion_lidar_live_test.py,
         driving into the Q4 rock_cluster) proved the fusion arithmetic
         correct against real /scan data (near-zero error throughout) but
         never exercised the rising part of the ramp: Q4's rock collision
         geometry (spheres centred slightly below ground level, radius up
         to ~0.29m even for "large boulders") tops out around 0.17-0.19m,
         while lidar_link is mounted at ground_clearance + body_height +
         0.030 = 0.215m -- the entire Q4 rock field sits below this
         horizontal-plane LiDAR's scan height. This is a genuine,
         previously-undocumented finding affecting the whole LiDAR safety
         subsystem (lidar_proximity_guard_node.py likely can't see Q4's
         rocks either), not something introduced by this feature, and out
         of scope to fix here (would mean touching the shared URDF or
         world file every other Gazebo result depends on). This script
         applies the same "ROS2-live, no Gazebo physics" pattern already
         used for the IMU term: runs the fusion node for real, feeds it a
         sequence of known /scan ranges directly, and confirms the ramp's
         rising branch (0.4m..2.0m) actually responds live -- closing the
         verification gap the Gazebo run's negative geometric finding left
         open, without needing to alter shared simulation infrastructure.
Inputs:  /traversability_score_fused (std_msgs/Float64) -- subscribed
         /traversability_score (std_msgs/Float64) -- published by this script
         /scan (sensor_msgs/LaserScan) -- published by this script
Outputs: experiments/results/traversability_fusion_lidar_synthetic_test.csv
How to run:
    # Terminal 1:
    source /opt/ros/humble/setup.bash && source ros2_ws/install/setup.bash
    ros2 run fm_imu_fusion traversability_score_fusion_node.py

    # Terminal 2:
    source /opt/ros/humble/setup.bash && source ros2_ws/install/setup.bash
    python3 experiments/traversability_fusion_lidar_synthetic_test.py
Project: Onboard Visual Foundation Models for Mars Terrain Perception and Rover Navigation
"""
import csv
import os
import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float64

from fm_imu_fusion.traversability_score_fusion_node import (
    lidar_range_risk, fuse_traversability_score,
)

DINOV2_SCORE_HELD = 0.0   # constant low score -- never masks the LiDAR term
# Spans clear/beyond, midpoint, at-threshold, and inside the 0.4m..2.0m ramp.
RANGE_SEQUENCE_M = [float("inf"), 3.0, 1.2, 0.4, 0.1]
HOLD_S_PER_RANGE = 4.0
LIDAR_RANGE_MIN = 0.05  # matches exomy.urdf.xacro's lidar_sensor <range><min>
LIDAR_RANGE_MAX = 12.0  # matches exomy.urdf.xacro's lidar_sensor <range><max>

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


class LidarSyntheticFusionRecorder(Node):

    def __init__(self):
        super().__init__("traversability_fusion_lidar_synthetic_test")
        self._records = []
        self._t0 = time.time()
        self._current_range_m = float("inf")

        self.create_subscription(Float64, "/traversability_score_fused", self._fused_cb, 10)
        self.pub_score = self.create_publisher(Float64, "/traversability_score", 10)
        self.pub_scan = self.create_publisher(LaserScan, "/scan", 10)
        self.create_timer(0.1, self._publish_stimuli)

    def _publish_stimuli(self) -> None:
        self.pub_score.publish(Float64(data=DINOV2_SCORE_HELD))

        scan = LaserScan()
        scan.range_min = LIDAR_RANGE_MIN
        scan.range_max = LIDAR_RANGE_MAX
        # A single-ray scan is enough -- _min_valid_range() only needs one
        # in-spec reading. inf correctly represents "nothing detected"
        # per the LaserScan convention (excluded from valid candidates).
        scan.ranges = [self._current_range_m]
        self.pub_scan.publish(scan)

    def _fused_cb(self, msg: Float64) -> None:
        expected_lidar_risk = lidar_range_risk(self._current_range_m)
        expected_fused = fuse_traversability_score(
            DINOV2_SCORE_HELD, expected_lidar_risk, imu_risk=0.0
        )
        self._records.append({
            "t_s": round(time.time() - self._t0, 3),
            "commanded_range_m": self._current_range_m,
            "expected_lidar_risk": round(expected_lidar_risk, 4),
            "expected_fused": round(expected_fused, 4),
            "actual_fused": round(msg.data, 4),
        })


def main():
    rclpy.init()
    node = LidarSyntheticFusionRecorder()
    node.get_logger().info(
        f"Feeding /scan range sequence {RANGE_SEQUENCE_M} m, {HOLD_S_PER_RANGE}s each, "
        "to verify the fusion node's LiDAR ramp live (no Gazebo)."
    )

    for range_m in RANGE_SEQUENCE_M:
        node._current_range_m = range_m
        end_t = time.time() + HOLD_S_PER_RANGE
        while time.time() < end_t:
            rclpy.spin_once(node, timeout_sec=0.1)

    csv_path = os.path.join(RESULTS_DIR, "traversability_fusion_lidar_synthetic_test.csv")
    with open(csv_path, "w", newline="") as f:
        if node._records:
            w = csv.DictWriter(f, fieldnames=list(node._records[0].keys()))
            w.writeheader()
            w.writerows(node._records)
    node.get_logger().info(f"Saved {len(node._records)} records -> {csv_path}")

    print("\n" + "=" * 70)
    print("LIDAR-SYNTHETIC-FUSION LIVE TEST SUMMARY")
    print(f"Total /traversability_score_fused messages recorded: {len(node._records)}")
    if node._records:
        errors = [abs(r["actual_fused"] - r["expected_fused"]) for r in node._records]
        by_range = {}
        for r in node._records:
            by_range.setdefault(r["commanded_range_m"], []).append(r["actual_fused"])
        print("Mean actual_fused by commanded range:")
        for range_m in RANGE_SEQUENCE_M:
            vals = by_range.get(range_m, [])
            if vals:
                print(f"  {range_m!s:>6} m -> mean actual_fused = {sum(vals)/len(vals):.4f} "
                      f"(n={len(vals)})")
        print(f"Mean |actual_fused - expected_fused|: {sum(errors)/len(errors):.4f}")
        print(f"Max  |actual_fused - expected_fused|: {max(errors):.4f}")
    print("=" * 70)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
