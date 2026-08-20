"""
Purpose: IMU-enhanced slope detection — fuses DINOv2 terrain classification with
         IMU tilt angle (max of |pitch| and |roll|) to override traversability on
         inclined terrain. Addresses Exp 7b finding: DINOv2 texture-only classification
         misclassifies 15° soil slope as 'sand' (conf 0.55–0.64) because camera
         inclination shifts feature distribution. An IMU tilt threshold catches slope
         geometry that appearance-only models cannot detect.
         NOTE: v1 used pitch-only (Y-axis). Gazebo slope_zone uses roll=-15° (X-axis),
         so pitch-only detection reads 0° on slope. Fixed in v2: use max(|pitch|, |roll|)
         for full 3-DOF tilt coverage.
Inputs:  /terrain_classification  (std_msgs/String)   "label:confidence"  from DINOv2 node
         /exomy/imu_raw           (sensor_msgs/Imu)   IMU orientation quaternion @ 100 Hz
Outputs: /traversability_fused    (std_msgs/String)   "label:confidence:tilt_deg:source"
         Console log comparing DINOv2-only vs fused policy at each frame
How to run:
    # Terminal 1: launch Gazebo with DINOv2 node
    source /opt/ros/humble/setup.bash && source ros2_ws/install/setup.bash
    ros2 launch simulation/launch/dinov2_controller_test.launch.py

    # Terminal 2: run IMU slope fusion node
    source /opt/ros/humble/setup.bash && source ros2_ws/install/setup.bash
    python3 experiments/imu_slope_detection.py

    # Verification: rostopic echo /traversability_fused
    ros2 topic echo /traversability_fused

Project: Onboard Visual Foundation Models for Mars Terrain Perception and Rover Navigation
"""

import math
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from std_msgs.msg import String

# ── Policy thresholds ────────────────────────────────────────────────────────
SLOPE_CAUTION_DEG  = 10.0   # pitch above this → downgrade to CAUTION minimum
SLOPE_STOP_DEG     = 20.0   # pitch above this → STOP regardless of terrain class
CONF_THRESHOLD     = 0.40   # DINOv2 confidence below this → uncertain → STOP

# Terrain → base policy (matches dinov2_traversability_experiment.py)
TERRAIN_POLICY = {
    "soil":      ("SAFE",    0.10),
    "bedrock":   ("HAZARD",  0.03),
    "sand":      ("CAUTION", 0.05),
    "big_rock":  ("STOP",    0.00),
    "uncertain": ("STOP",    0.00),
}

# When slope overrides terrain label → minimum policy
SLOPE_CAUTION_POLICY = ("CAUTION", 0.05)
SLOPE_STOP_POLICY    = ("STOP",    0.00)


def quaternion_to_pitch(qx: float, qy: float, qz: float, qw: float) -> float:
    """Extract pitch angle in degrees (rotation about Y-axis)."""
    sinp = 2.0 * (qw * qy - qz * qx)
    sinp = max(-1.0, min(1.0, sinp))
    return math.degrees(math.asin(sinp))


def quaternion_to_roll(qx: float, qy: float, qz: float, qw: float) -> float:
    """Extract roll angle in degrees (rotation about X-axis)."""
    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
    return math.degrees(math.atan2(sinr_cosp, cosr_cosp))


def quaternion_to_max_tilt(qx: float, qy: float, qz: float, qw: float) -> float:
    """Return the maximum absolute tilt (max of |pitch|, |roll|) in degrees.

    Required for full slope detection: Gazebo slope_zone uses roll=-15° (X-axis tilt),
    which produces pitch=0° if only Y-axis rotation is extracted.
    """
    pitch = quaternion_to_pitch(qx, qy, qz, qw)
    roll  = quaternion_to_roll(qx, qy, qz, qw)
    return max(abs(pitch), abs(roll))


class ImuSlopeFusion(Node):
    """
    ROS2 node that fuses DINOv2 terrain classification with IMU slope measurement.

    Logic per frame:
    1. Read latest DINOv2 label + confidence from /terrain_classification.
    2. Read latest IMU quaternion → compute max_tilt = max(|pitch|, |roll|).
    3. If max_tilt > SLOPE_STOP_DEG   → override policy to STOP.
       If max_tilt > SLOPE_CAUTION_DEG → downgrade to CAUTION if current policy is SAFE.
       Otherwise → use DINOv2 policy as-is.
    4. Publish fused decision to /traversability_fused.
    """

    def __init__(self):
        super().__init__("imu_slope_fusion_node")

        self._terrain_label = "uncertain"
        self._terrain_conf  = 0.0
        self._tilt_deg      = 0.0

        self.create_subscription(String, "/terrain_classification", self._terrain_cb, 10)
        self.create_subscription(Imu,    "/exomy/imu_raw",          self._imu_cb,     10)
        self._fused_pub = self.create_publisher(String, "/traversability_fused", 10)

        self.create_timer(0.5, self._fuse_and_publish)
        self.get_logger().info("IMU slope fusion node started (v2: max-tilt = max(|pitch|,|roll|)).")
        self.get_logger().info(f"Thresholds: CAUTION >tilt{SLOPE_CAUTION_DEG}°, STOP >tilt{SLOPE_STOP_DEG}°")

    def _terrain_cb(self, msg: String):
        parts = msg.data.split(":")
        if len(parts) >= 2:
            self._terrain_label = parts[0].strip().lower()
            try:
                self._terrain_conf = float(parts[1])
            except ValueError:
                self._terrain_conf = 0.0

    def _imu_cb(self, msg: Imu):
        q = msg.orientation
        self._tilt_deg = quaternion_to_max_tilt(q.x, q.y, q.z, q.w)

    def _fuse_and_publish(self):
        label = self._terrain_label
        conf  = self._terrain_conf
        tilt  = self._tilt_deg   # max(|pitch|, |roll|) — covers all slope orientations

        # Step 1: DINOv2-only policy
        if conf < CONF_THRESHOLD or label not in TERRAIN_POLICY:
            dino_policy, dino_speed = TERRAIN_POLICY["uncertain"]
            label = "uncertain"
        else:
            dino_policy, dino_speed = TERRAIN_POLICY[label]

        # Step 2: IMU slope override
        if tilt > SLOPE_STOP_DEG:
            fused_policy, fused_speed = SLOPE_STOP_POLICY
            source = "IMU_STOP"
        elif tilt > SLOPE_CAUTION_DEG and dino_policy == "SAFE":
            fused_policy, fused_speed = SLOPE_CAUTION_POLICY
            source = "IMU_CAUTION"
        else:
            fused_policy, fused_speed = dino_policy, dino_speed
            source = "DINOV2"

        out = f"{label}:{conf:.3f}:{tilt:.1f}:{source}"
        self._fused_pub.publish(String(data=out))

        # Log when slope override changes the decision
        if source.startswith("IMU"):
            self.get_logger().warn(
                f"SLOPE OVERRIDE: tilt={tilt:.1f}° | DINOv2={label}({dino_policy},{dino_speed:.2f}m/s)"
                f" → FUSED={fused_policy},{fused_speed:.2f}m/s [{source}]"
            )
        else:
            self.get_logger().info(
                f"tilt={tilt:.1f}° | terrain={label}(conf={conf:.2f}) → {fused_policy},{fused_speed:.2f}m/s [{source}]"
            )


def main():
    rclpy.init()
    node = ImuSlopeFusion()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
