#!/usr/bin/env python3
"""
Purpose: Additive extension of H3f's continuous /traversability_score
         (dinov2_terrain_node.py) with LiDAR range and IMU tilt risk terms,
         producing one richer continuous /traversability_score_fused signal.
         Backlog item 8, scoped via grill-thesis 2026-07-17. Deliberately
         additive: the existing independent Bool E-stop triggers
         (/terrain_controller/stopped, /lidar_proximity_stop,
         /imu_slope_stop) and the nodes that publish them
         (lidar_proximity_guard_node.py, imu_slope_fusion_node.py) are
         completely untouched -- this node only adds a new, optional
         continuous signal for proportional speed control
         (terrain_controller_node.py's score_source param), it does not
         replace or weaken the existing any-one-of-three-stops-it safety
         redundancy.
         Risk mapping reuses already-established thresholds rather than
         inventing new ones where possible: the LiDAR term's near end
         matches lidar_proximity_guard_node.py's stop_distance_m (0.4m);
         the IMU term's caution/stop ends match imu_slope_fusion_node.py's
         slope_caution_deg/slope_stop_deg (10/20 degrees) exactly, so no
         new IMU parameters exist at all. The LiDAR term's far end
         (2.0m, "no longer relevant") is a new parameter with no prior
         precedent in this codebase -- chosen as 5x the stop distance,
         giving several seconds of travel time at this thesis's
         established 0.10 m/s speed ceiling before the term starts rising.
         Combination is max(dinov2_score, lidar_risk, imu_risk) -- the
         worst signal wins, matching the existing philosophy that a single
         hazard indicator (the obstacle-gate override, the low-confidence
         forced-STOP rule) is never diluted by averaging against other,
         calmer channels.
Inputs:  /traversability_score (std_msgs/Float64)  DINOv2-only continuous
             score from dinov2_terrain_node.py (H3f)
         /scan                 (sensor_msgs/LaserScan)  LiDAR range data
         /exomy/imu_raw        (sensor_msgs/Imu)  IMU orientation quaternion
Outputs: /traversability_score_fused (std_msgs/Float64)  fused continuous
             risk score, 0.0=safe .. 1.0=stop
ROS2 parameters:
         lidar_stop_distance_m  (float, default 0.4)  LiDAR risk = 1.0 at/inside this range
         lidar_clear_distance_m (float, default 2.0)  LiDAR risk = 0.0 at/beyond this range
         imu_caution_deg        (float, default 10.0) IMU risk = 0.0 at/below this tilt
         imu_stop_deg           (float, default 20.0) IMU risk = 1.0 at/above this tilt
         publish_hz             (float, default 5.0)  output publish rate in Hz
How to run:
    source /opt/ros/humble/setup.bash
    cd ros2_ws && colcon build --packages-select fm_imu_fusion
    source install/setup.bash
    ros2 run fm_imu_fusion traversability_score_fusion_node.py
    ros2 topic echo /traversability_score_fused
Project: Onboard Visual Foundation Models for Mars Terrain Perception and Rover Navigation
"""
import math

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu, LaserScan
from std_msgs.msg import Float64

LIDAR_STOP_DISTANCE_M  = 0.4   # matches lidar_proximity_guard_node.py's stop_distance_m
LIDAR_CLEAR_DISTANCE_M = 2.0   # new parameter, see module docstring
IMU_CAUTION_DEG = 10.0         # matches imu_slope_fusion_node.py's slope_caution_deg
IMU_STOP_DEG    = 20.0         # matches imu_slope_fusion_node.py's slope_stop_deg


def _min_valid_range(ranges, range_min: float, range_max: float) -> float:
    """Closest in-spec reading in a LaserScan's ranges array. Mirrors
    lidar_proximity_guard_node.py's min_valid_range() -- duplicated rather
    than imported to keep fm_imu_fusion independent of fm_perception
    (existing convention: fm_* packages only couple via ROS2 topics)."""
    valid = [r for r in ranges if math.isfinite(r) and range_min <= r <= range_max]
    return min(valid) if valid else float("inf")


def _quaternion_to_pitch(qx: float, qy: float, qz: float, qw: float) -> float:
    sinp = max(-1.0, min(1.0, 2.0 * (qw * qy - qz * qx)))
    return math.degrees(math.asin(sinp))


def _quaternion_to_roll(qx: float, qy: float, qz: float, qw: float) -> float:
    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
    return math.degrees(math.atan2(sinr_cosp, cosr_cosp))


def _max_tilt(qx: float, qy: float, qz: float, qw: float) -> float:
    """max(|pitch|, |roll|) in degrees -- identical to imu_slope_fusion_node.py's
    _max_tilt(), duplicated for the same independence reason as above."""
    return max(abs(_quaternion_to_pitch(qx, qy, qz, qw)),
               abs(_quaternion_to_roll(qx, qy, qz, qw)))


def lidar_range_risk(min_range_m: float,
                      stop_distance_m: float = LIDAR_STOP_DISTANCE_M,
                      clear_distance_m: float = LIDAR_CLEAR_DISTANCE_M) -> float:
    """Linear ramp: 1.0 at/inside stop_distance_m, 0.0 at/beyond clear_distance_m."""
    if min_range_m <= stop_distance_m:
        return 1.0
    if min_range_m >= clear_distance_m:
        return 0.0
    return (clear_distance_m - min_range_m) / (clear_distance_m - stop_distance_m)


def imu_tilt_risk(tilt_deg: float,
                   caution_deg: float = IMU_CAUTION_DEG,
                   stop_deg: float = IMU_STOP_DEG) -> float:
    """Linear ramp: 0.0 at/below caution_deg, 1.0 at/above stop_deg."""
    if tilt_deg <= caution_deg:
        return 0.0
    if tilt_deg >= stop_deg:
        return 1.0
    return (tilt_deg - caution_deg) / (stop_deg - caution_deg)


def fuse_traversability_score(dinov2_score: float, lidar_risk: float, imu_risk: float) -> float:
    """Worst signal wins -- see module docstring for the rationale."""
    return max(dinov2_score, lidar_risk, imu_risk)


class TraversabilityScoreFusionNode(Node):

    def __init__(self):
        super().__init__("traversability_score_fusion_node")

        self.declare_parameter("lidar_stop_distance_m", LIDAR_STOP_DISTANCE_M)
        self.declare_parameter("lidar_clear_distance_m", LIDAR_CLEAR_DISTANCE_M)
        self.declare_parameter("imu_caution_deg", IMU_CAUTION_DEG)
        self.declare_parameter("imu_stop_deg", IMU_STOP_DEG)
        self.declare_parameter("publish_hz", 5.0)

        self._lidar_stop_distance  = self.get_parameter("lidar_stop_distance_m").value
        self._lidar_clear_distance = self.get_parameter("lidar_clear_distance_m").value
        self._imu_caution_deg      = self.get_parameter("imu_caution_deg").value
        self._imu_stop_deg         = self.get_parameter("imu_stop_deg").value
        publish_hz                 = self.get_parameter("publish_hz").value

        self._dinov2_score = 1.0   # uncertain/no data yet -> treat as unsafe, matches
                                    # dinov2_terrain_node.py's own uncertain->STOP convention
        self._lidar_risk = 0.0     # no /scan yet -- absence of LiDAR data is not itself
                                    # treated as unsafe here (lidar_proximity_guard_node.py's
                                    # own stale-data watchdog already covers that hard-stop case
                                    # independently; this node only adds a continuous term)
        self._imu_risk = 0.0       # no /exomy/imu_raw yet -- same reasoning as above

        self.create_subscription(Float64, "/traversability_score", self._score_cb, 10)
        self.create_subscription(LaserScan, "/scan", self._scan_cb, 10)
        self.create_subscription(Imu, "/exomy/imu_raw", self._imu_cb, 10)
        self._pub = self.create_publisher(Float64, "/traversability_score_fused", 10)
        self.create_timer(1.0 / publish_hz, self._publish)

        self.get_logger().info(
            f"traversability_score_fusion_node ready | "
            f"lidar=[{self._lidar_stop_distance:.2f}m..{self._lidar_clear_distance:.2f}m] "
            f"imu=[{self._imu_caution_deg:.1f}..{self._imu_stop_deg:.1f}]deg rate={publish_hz}Hz"
        )

    def _score_cb(self, msg: Float64) -> None:
        self._dinov2_score = msg.data

    def _scan_cb(self, msg: LaserScan) -> None:
        min_range = _min_valid_range(msg.ranges, msg.range_min, msg.range_max)
        self._lidar_risk = lidar_range_risk(
            min_range, self._lidar_stop_distance, self._lidar_clear_distance)

    def _imu_cb(self, msg: Imu) -> None:
        q = msg.orientation
        tilt = _max_tilt(q.x, q.y, q.z, q.w)
        self._imu_risk = imu_tilt_risk(tilt, self._imu_caution_deg, self._imu_stop_deg)

    def _publish(self) -> None:
        fused = fuse_traversability_score(self._dinov2_score, self._lidar_risk, self._imu_risk)
        self._pub.publish(Float64(data=fused))


def main(args=None):
    rclpy.init(args=args)
    node = TraversabilityScoreFusionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
