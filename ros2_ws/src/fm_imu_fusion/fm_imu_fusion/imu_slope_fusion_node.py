#!/usr/bin/env python3
"""
Purpose: IMU-terrain fusion ROS2 node — fuses DINOv2 terrain classification
         with IMU tilt angle (max of |pitch|, |roll|) to override traversability
         decisions on inclined terrain. Addresses Exp 7b finding: DINOv2 texture-
         only classification misclassifies a 15-degree soil slope as sand (confidence
         0.55-0.64) because camera inclination shifts the feature distribution. IMU
         tilt threshold catches slope geometry that appearance-only models cannot detect.
         v2 fix: uses max(|pitch|,|roll|) instead of pitch-only so X-axis slopes
         (roll=-15 degrees in Gazebo slope_zone) are correctly detected.
Inputs:  /terrain_classification  (std_msgs/String)  "label:confidence" from DINOv2 node
         /exomy/imu_raw           (sensor_msgs/Imu)  IMU orientation quaternion at 100 Hz
Outputs: /traversability_fused   (std_msgs/String)  "label:confidence:tilt_deg:source"
         source is one of: DINOV2 | IMU_CAUTION | IMU_STOP
         /imu_slope_stop          (std_msgs/Bool)    true when tilt > slope_stop_deg --
         a plain boolean mirror of the IMU_STOP decision above, in the same style as
         lidar_proximity_guard_node.py's /lidar_proximity_stop, so reactive_explorer_node
         can consume it directly as a third hazard trigger without parsing the
         label:confidence:tilt_deg:source string format.
ROS2 parameters:
         slope_caution_deg  (float, default 10.0) — tilt above this downgrades SAFE to CAUTION
         slope_stop_deg     (float, default 20.0) — tilt above this forces STOP
         conf_threshold     (float, default 0.40) — DINOv2 confidence below this → uncertain
         publish_hz         (float, default 2.0)  — output publish rate in Hz
         rotation_gate_rad_s (float, default 0.05) — while |gyro_z| exceeds this, the
                              accel-only tilt reading is held at its last pre-rotation
                              value instead of being trusted (see gate_tilt())
How to run:
    source /opt/ros/humble/setup.bash
    cd ros2_ws && colcon build --packages-select fm_imu_fusion
    source install/setup.bash
    ros2 launch fm_imu_fusion imu_fusion.launch.py
    # Or run node directly:
    ros2 run fm_imu_fusion imu_slope_fusion_node.py
    # Verify output:
    ros2 topic echo /traversability_fused
Project: Onboard Visual Foundation Models for Mars Terrain Perception and Rover Navigation
"""

import math

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from std_msgs.msg import Bool, String

# Terrain → base traversability policy (matches dinov2_terrain_node.py)
TERRAIN_POLICY = {
    "soil":      ("SAFE",    0.10),
    "bedrock":   ("HAZARD",  0.03),
    "sand":      ("CAUTION", 0.05),
    "big_rock":  ("STOP",    0.00),
    "uncertain": ("STOP",    0.00),
}

SLOPE_CAUTION_POLICY = ("CAUTION", 0.05)
SLOPE_STOP_POLICY    = ("STOP",    0.00)

# How far the measured acceleration may depart from 1 g before its tilt angle
# is refused. Chosen from the geometry rather than by taste: an artifact that
# only just reaches the 20 deg STOP threshold implies 1/cos(20 deg) = 1.064 g,
# so the gate has to sit below 0.064 to catch the weakest spike that can still
# trigger a false stop, while staying above the few-hundredths-of-a-g sensor
# noise that would otherwise block every update.
ACCEL_GATE_G = 0.05

# Standard gravity, for turning the incoming linear_acceleration (m/s^2 per
# REP-103) back into g. Deliberately duplicated from icm20948_driver_node
# rather than imported: this node must stay importable without smbus2 and the
# rest of the I2C stack, which is what keeps its tests pure Python.
G_TO_M_S2 = 9.80665


def _quaternion_to_pitch(qx: float, qy: float, qz: float, qw: float) -> float:
    sinp = max(-1.0, min(1.0, 2.0 * (qw * qy - qz * qx)))
    return math.degrees(math.asin(sinp))


def _quaternion_to_roll(qx: float, qy: float, qz: float, qw: float) -> float:
    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
    return math.degrees(math.atan2(sinr_cosp, cosr_cosp))


def _max_tilt(qx: float, qy: float, qz: float, qw: float) -> float:
    """Return max(|pitch|, |roll|) in degrees — covers both X- and Y-axis slopes."""
    return max(abs(_quaternion_to_pitch(qx, qy, qz, qw)),
               abs(_quaternion_to_roll(qx, qy, qz, qw)))


def is_slope_stop(tilt_deg: float, stop_deg: float) -> bool:
    """True once tilt exceeds the STOP threshold -- the same condition already
    used for the IMU_STOP source in _publish(), exposed as a plain bool."""
    return tilt_deg > stop_deg


def gate_tilt(new_tilt_deg: float, held_tilt_deg: float,
              gyro_z_rad_s: float, gate_rad_s: float,
              accel_magnitude_g: float = 1.0,
              accel_gate_g: float = ACCEL_GATE_G) -> float:
    """Hold the last tilt reading whenever the fresh one cannot be trusted.

    The tilt estimate is accelerometer-only (no gyro fusion), so it is really
    a measurement of "which way is the total specific force pointing". That
    equals gravity only when nothing else is pushing on the rover. Two
    separate things break that assumption and each gets its own gate.

    ROTATION (added 2026-07-27). Angular acceleration during a real POINT_TURN
    corrupts the gravity-vector reading into a spurious 30-40 deg tilt -- the
    root cause of the 2026-07-27 wall collision. While |gyro_z_rad_s| exceeds
    gate_rad_s, hold.

    LINEAR ACCELERATION AND VIBRATION (added 2026-07-29). The rotation gate
    covers only yaw rate, so it did nothing while the rover drove STRAIGHT.
    Trial A on a flat lab floor logged apparent tilts of 15.8, 20.5, 25.6,
    31.7, 33.5 and 62.9 deg in FAKE_ACKERMANN with gyro_z near zero, and every
    one of them sailed through. The discriminator is magnitude, not direction:
    gravity is 1 g whatever the rover's orientation, so a rover parked on a
    genuine 25 deg slope reads |a| = 1.000 g, whereas 25 deg of apparent tilt
    manufactured by horizontal force alone requires |a| = 1/cos(25 deg) =
    1.103 g. Any sample whose magnitude departs from 1 g by more than
    accel_gate_g therefore contains a non-gravitational force and its angle is
    not a slope measurement. Under-reading is gated too (a wheel dropping off
    a lip unloads the IMU below 1 g).

    Callers that pass no acceleration get the historical two-gate behaviour,
    since the default magnitude of 1.0 can never trip the second gate.
    """
    if abs(gyro_z_rad_s) > gate_rad_s:
        return held_tilt_deg
    if abs(accel_magnitude_g - 1.0) > accel_gate_g:
        return held_tilt_deg
    return new_tilt_deg


def apply_slope_override(base_policy: str, base_speed: float, tilt_deg: float,
                         caution_deg: float, stop_deg: float,
                         enabled: bool = True):
    """Apply the IMU's veto to a terrain decision, or pass it through.

    Returns (policy, speed, source). `enabled` exists because the tilt estimate
    is accelerometer-only and, measured from Trial A run 2's bag on 2026-07-29,
    unusable while this rover drives: /exomy/imu_raw was live at 50 Hz with
    every message different, but chassis vibration swung |a| between 0.521 g and
    2.306 g against a true 1.000 g and produced apparent tilts of 10-74 deg on a
    flat lab floor. accel_gate_g cannot filter that, because it gates magnitude
    and not direction -- the recorded sample tilt=53.04 deg at |a|=0.979 g
    passes a 1.000 +/- 0.05 g gate while pointing nowhere near down.

    A false STOP is not harmless here. Over-tilt is handled like a blocked
    corridor, so it forces reactive_explorer to re-decide its heading, and run 2
    ended in FAILSAFE(boxed_in) when that happened in front of a cupboard where
    the LiDAR had no clear heading to offer.

    Defaults to enabled: every Gazebo result this thesis reports ran with the
    override live. It is switched off in real_hardware_deployment.launch.py
    only, and the node still computes and publishes tilt either way so the
    limitation stays measurable.
    """
    if not enabled:
        return base_policy, base_speed, "DINOV2"
    if tilt_deg > stop_deg:
        return (*SLOPE_STOP_POLICY, "IMU_STOP")
    if tilt_deg > caution_deg and base_policy == "SAFE":
        return (*SLOPE_CAUTION_POLICY, "IMU_CAUTION")
    return base_policy, base_speed, "DINOV2"


class SlopeStopLatch:
    """Require an over-tilt to persist for `hold_s` before believing it."""

    def __init__(self, stop_deg: float, hold_s: float):
        self.stop_deg = stop_deg
        self.hold_s = hold_s
        self._above_since_s = None

    def update(self, tilt_deg: float, now_s: float) -> bool:
        if not is_slope_stop(tilt_deg, self.stop_deg):
            self._above_since_s = None
            return False
        if self._above_since_s is None:
            self._above_since_s = now_s
        return (now_s - self._above_since_s) >= self.hold_s


class ImuSlopeFusionNode(Node):
    """
    Fuses DINOv2 terrain classification with IMU tilt to produce a safe traversability
    decision. Published at a fixed rate so downstream controllers receive a steady stream
    even when IMU or terrain topics are temporarily silent.

    Decision logic (per publish tick):
    1. If DINOv2 confidence < conf_threshold → treat terrain as uncertain → STOP.
    2. If tilt > slope_stop_deg             → override to STOP (IMU_STOP).
    3. If tilt > slope_caution_deg and base policy is SAFE → downgrade to CAUTION (IMU_CAUTION).
    4. Otherwise                            → pass DINOv2 decision unchanged (DINOV2).
    """

    def __init__(self):
        super().__init__("imu_slope_fusion_node")

        self.declare_parameter("slope_caution_deg", 10.0)
        self.declare_parameter("slope_stop_deg",    20.0)
        self.declare_parameter("conf_threshold",    0.40)
        self.declare_parameter("publish_hz",         2.0)
        self.declare_parameter("rotation_gate_rad_s", 0.05)
        self.declare_parameter("accel_gate_g", ACCEL_GATE_G)
        self.declare_parameter("slope_stop_hold_s", 1.5)
        # Whether the IMU may override the terrain decision at all. See
        # apply_slope_override for the measurement behind switching it off on
        # real hardware. Default True so every reported Gazebo result stands.
        self.declare_parameter("slope_override_enabled", True)

        self._caution_deg   = self.get_parameter("slope_caution_deg").value
        self._stop_deg      = self.get_parameter("slope_stop_deg").value
        self._conf_thresh   = self.get_parameter("conf_threshold").value
        publish_hz          = self.get_parameter("publish_hz").value
        self._rotation_gate = self.get_parameter("rotation_gate_rad_s").value
        self._accel_gate_g  = self.get_parameter("accel_gate_g").value
        self._slope_override_enabled = self.get_parameter(
            "slope_override_enabled").value
        self._slope_latch   = SlopeStopLatch(
            stop_deg=self._stop_deg,
            hold_s=self.get_parameter("slope_stop_hold_s").value,
        )

        self._terrain_label = "uncertain"
        self._terrain_conf  = 0.0
        self._tilt_deg      = 0.0
        self._gyro_z        = 0.0

        self.create_subscription(String, "/terrain_classification",
                                 self._terrain_cb, 10)
        self.create_subscription(Imu, "/exomy/imu_raw",
                                 self._imu_cb, 10)
        self._pub = self.create_publisher(String, "/traversability_fused", 10)
        self._pub_slope_stop = self.create_publisher(Bool, "/imu_slope_stop", 10)
        self.create_timer(1.0 / publish_hz, self._publish)

        self.get_logger().info(
            f"fm_imu_fusion started | "
            f"caution>{self._caution_deg}° stop>{self._stop_deg}° "
            f"conf_thresh={self._conf_thresh} rate={publish_hz}Hz"
        )

    def _terrain_cb(self, msg: String) -> None:
        parts = msg.data.split(":")
        if len(parts) >= 2:
            self._terrain_label = parts[0].strip().lower()
            try:
                self._terrain_conf = float(parts[1])
            except ValueError:
                self._terrain_conf = 0.0

    def _imu_cb(self, msg: Imu) -> None:
        q = msg.orientation
        raw_tilt = _max_tilt(q.x, q.y, q.z, q.w)
        self._gyro_z = msg.angular_velocity.z
        a = msg.linear_acceleration
        # A driver that does not populate linear_acceleration leaves all three
        # components at zero, which would read as 0 g and gate every sample
        # forever. Fall back to the neutral 1.0 g in that case so the accel
        # gate simply does not engage, rather than silently freezing the tilt.
        magnitude_m_s2 = math.sqrt(a.x * a.x + a.y * a.y + a.z * a.z)
        accel_g = (magnitude_m_s2 / G_TO_M_S2) if magnitude_m_s2 > 0.0 else 1.0
        self._tilt_deg = gate_tilt(raw_tilt, self._tilt_deg,
                                    self._gyro_z, self._rotation_gate,
                                    accel_magnitude_g=accel_g,
                                    accel_gate_g=self._accel_gate_g)

    def _publish(self) -> None:
        label = self._terrain_label
        conf  = self._terrain_conf
        tilt  = self._tilt_deg

        # Step 1: DINOv2-only policy
        if conf < self._conf_thresh or label not in TERRAIN_POLICY:
            base_policy, base_speed = TERRAIN_POLICY["uncertain"]
            label = "uncertain"
        else:
            base_policy, base_speed = TERRAIN_POLICY[label]

        # Step 2: IMU override
        fused_policy, fused_speed, source = apply_slope_override(
            base_policy, base_speed, tilt,
            caution_deg=self._caution_deg, stop_deg=self._stop_deg,
            enabled=self._slope_override_enabled,
        )

        self._pub.publish(String(data=f"{label}:{conf:.3f}:{tilt:.1f}:{source}"))
        # /imu_slope_stop is the topic other nodes act on, so it goes through
        # the dwell latch as well as the per-sample gates: a spike that
        # survives both gates still has to persist before the rest of the
        # stack is told the rover is on a slope.
        now_s = self.get_clock().now().nanoseconds / 1e9
        latched = self._slope_latch.update(tilt, now_s)
        self._pub_slope_stop.publish(
            Bool(data=latched and self._slope_override_enabled)
        )

        if source.startswith("IMU"):
            self.get_logger().warn(
                f"SLOPE OVERRIDE tilt={tilt:.1f}° | "
                f"DINOv2={label}({base_policy},{base_speed:.2f}m/s) "
                f"→ {fused_policy},{fused_speed:.2f}m/s [{source}]"
            )
        else:
            self.get_logger().info(
                f"tilt={tilt:.1f}° terrain={label}(conf={conf:.2f}) "
                f"→ {fused_policy},{fused_speed:.2f}m/s [{source}]"
            )


def main(args=None):
    rclpy.init(args=args)
    node = ImuSlopeFusionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
