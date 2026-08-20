"""
Purpose: Live verification of traversability_score_fusion_node.py's IMU term
         (backlog item 8, scoped via grill-thesis 2026-07-17). No Gazebo --
         slope_zone (referenced in imu_slope_fusion_node.py's own docstring)
         was checked and confirmed to no longer exist in mars_terrain.world
         (removed in the June 23 quadrant restructuring, never rebuilt), and
         Gazebo's own simulated IMU sensor already continuously publishes
         real /exomy/imu_raw, which would race a synthetic publisher on the
         same topic rather than cleanly inject a known tilt. Instead, this
         script runs the fusion node for real (a live ROS2 process, not a
         unit test) with no Gazebo robot spawned at all, and feeds it a
         sequence of known tilt values directly -- verifying the node's
         actual subscribe/quaternion-decode/compute/publish wiring
         end-to-end, without re-deriving slope physics (already established
         separately in Exp 7b, §4.8.x) or Gazebo's IMU sensor accuracy
         (already established elsewhere). /traversability_score is held at
         a constant low value throughout so it never masks the IMU term;
         no /scan is published, so the LiDAR term stays at its documented
         default (0.0, "no data yet").
Inputs:  /traversability_score_fused (std_msgs/Float64) -- subscribed
         /traversability_score (std_msgs/Float64) -- published by this script
         /exomy/imu_raw (sensor_msgs/Imu) -- published by this script
Outputs: experiments/results/traversability_fusion_imu_live_test.csv
How to run:
    # Terminal 1:
    source /opt/ros/humble/setup.bash && source ros2_ws/install/setup.bash
    ros2 run fm_imu_fusion traversability_score_fusion_node.py

    # Terminal 2:
    source /opt/ros/humble/setup.bash && source ros2_ws/install/setup.bash
    python3 experiments/traversability_fusion_imu_live_test.py
Project: Onboard Visual Foundation Models for Mars Terrain Perception and Rover Navigation
"""
import csv
import os
import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from std_msgs.msg import Float64

from fm_imu_fusion.icm20948_driver_node import euler_to_quaternion
from fm_imu_fusion.traversability_score_fusion_node import (
    imu_tilt_risk, fuse_traversability_score,
)

DINOV2_SCORE_HELD = 0.0   # constant low score -- never masks the IMU term
TILT_SEQUENCE_DEG = [0.0, 5.0, 10.0, 15.0, 20.0, 25.0]  # spans below/at/above both thresholds
HOLD_S_PER_TILT = 4.0     # >= a few publish cycles (5Hz default) per stimulus

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


class ImuFusionRecorder(Node):

    def __init__(self):
        super().__init__("traversability_fusion_imu_live_test")
        self._records = []
        self._t0 = time.time()
        self._current_tilt_deg = 0.0

        self.create_subscription(Float64, "/traversability_score_fused", self._fused_cb, 10)
        self.pub_score = self.create_publisher(Float64, "/traversability_score", 10)
        self.pub_imu = self.create_publisher(Imu, "/exomy/imu_raw", 10)
        self.create_timer(0.1, self._publish_stimuli)

    def _publish_stimuli(self) -> None:
        self.pub_score.publish(Float64(data=DINOV2_SCORE_HELD))

        qx, qy, qz, qw = euler_to_quaternion(roll_deg=self._current_tilt_deg, pitch_deg=0.0)
        msg = Imu()
        msg.orientation.x = qx
        msg.orientation.y = qy
        msg.orientation.z = qz
        msg.orientation.w = qw
        self.pub_imu.publish(msg)

    def _fused_cb(self, msg: Float64) -> None:
        expected_imu_risk = imu_tilt_risk(self._current_tilt_deg)
        expected_fused = fuse_traversability_score(
            DINOV2_SCORE_HELD, lidar_risk=0.0, imu_risk=expected_imu_risk
        )
        self._records.append({
            "t_s": round(time.time() - self._t0, 3),
            "commanded_tilt_deg": self._current_tilt_deg,
            "expected_imu_risk": round(expected_imu_risk, 4),
            "expected_fused": round(expected_fused, 4),
            "actual_fused": round(msg.data, 4),
        })


def main():
    rclpy.init()
    node = ImuFusionRecorder()
    node.get_logger().info(
        f"Feeding tilt sequence {TILT_SEQUENCE_DEG} deg, {HOLD_S_PER_TILT}s each, "
        "to verify the fusion node's IMU term live (no Gazebo)."
    )

    for tilt_deg in TILT_SEQUENCE_DEG:
        node._current_tilt_deg = tilt_deg
        end_t = time.time() + HOLD_S_PER_TILT
        while time.time() < end_t:
            rclpy.spin_once(node, timeout_sec=0.1)

    csv_path = os.path.join(RESULTS_DIR, "traversability_fusion_imu_live_test.csv")
    with open(csv_path, "w", newline="") as f:
        if node._records:
            w = csv.DictWriter(f, fieldnames=list(node._records[0].keys()))
            w.writeheader()
            w.writerows(node._records)
    node.get_logger().info(f"Saved {len(node._records)} records -> {csv_path}")

    print("\n" + "=" * 70)
    print("IMU-FUSION LIVE TEST SUMMARY")
    print(f"Total /traversability_score_fused messages recorded: {len(node._records)}")
    if node._records:
        errors = [abs(r["actual_fused"] - r["expected_fused"]) for r in node._records]
        by_tilt = {}
        for r in node._records:
            by_tilt.setdefault(r["commanded_tilt_deg"], []).append(r["actual_fused"])
        print("Mean actual_fused by commanded tilt:")
        for tilt_deg in TILT_SEQUENCE_DEG:
            vals = by_tilt.get(tilt_deg, [])
            if vals:
                print(f"  {tilt_deg:5.1f} deg -> mean actual_fused = {sum(vals)/len(vals):.4f} "
                      f"(n={len(vals)})")
        print(f"Mean |actual_fused - expected_fused|: {sum(errors)/len(errors):.4f}")
        print(f"Max  |actual_fused - expected_fused|: {max(errors):.4f}")
    print("=" * 70)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
