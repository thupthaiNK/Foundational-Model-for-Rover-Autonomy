#!/usr/bin/env python3
"""
Purpose: Show every sensor on the rover at once, in one RViz2 window, during a
         field test: what DINOv2 concludes about the terrain, how far the LiDAR
         sees ahead, what angles the IMU reads, and what the rover is being
         commanded to do.

         Read-only. It subscribes to topics that already exist and publishes a
         single text marker; no node that produced a number in the thesis is
         touched. All of the logic lives in sensor_snapshot.py, which has no ROS
         import and is tested directly, so this file stays a thin adapter:
         message in, snapshot field out.

         TEXT_VIEW_FACING is a built-in RViz2 marker type, chosen so nothing has
         to be installed on the laptop in the field. A field is rendered as
         "STALE 4.2s" once it is past its timeout rather than holding its last
         value, because the IMU has dropped off I2C mid-run before and a
         frozen angle is indistinguishable from a rover that is genuinely level.

         This runs alongside traversability_viz_node, which is left exactly as it
         was: that node produced thesis figures and answers the narrower question
         of the current policy verdict. Colours are taken from it so the two
         cannot disagree on screen.

Inputs:  /terrain_classification      (std_msgs/String)  "label:confidence"
         /terrain_class_probs         (std_msgs/Float32MultiArray)
         /traversability_score        (std_msgs/Float64)
         /inference_latency_ms        (std_msgs/Float64)
         /terrain_frame_informative   (std_msgs/Bool)
         /scan                        (sensor_msgs/LaserScan)
         /lidar_proximity_stop        (std_msgs/Bool)
         /lidar_proximity_reason      (std_msgs/String)
         /exomy/imu_raw               (sensor_msgs/Imu)
         /imu_slope_stop              (std_msgs/Bool)
         /traversability_fused        (std_msgs/String)
         /exomy/cmd_vel               (geometry_msgs/Twist)
         /reactive_explorer/active    (std_msgs/Bool)
         /e_stop, /e_stop_reason      (std_msgs/Bool, std_msgs/String)
Outputs: /sensor_overlay/markers      (visualization_msgs/MarkerArray)

Parameters:
    publish_rate_hz  (float, default 2.0)  marker refresh rate
    frame_id         (str,   default "base_link")
    text_height_m    (float, default 0.20)  see the note below on width
    text_z_m         (float, default 0.0)   the view is top-down; height gains nothing

         Sizing note, measured in RViz2 on 2026-08-11 rather than guessed. Marker
         text is laid out in world units, so its on-screen width is
         text_height_m x characters x the view's Scale (pixels per metre). At the
         shipped Scale of 120 and a 22-character line, 0.20 m fills roughly 320
         px, which fits the render panel next to two image panels. Raise it only
         if the render panel is widened to match, or the lines run off the edge.

How to run:
    source /opt/ros/humble/setup.bash && source ros2_ws/install/setup.bash
    ros2 run fm_perception sensor_overlay_node.py

    # in RViz2 on the laptop: add a MarkerArray display on
    # /sensor_overlay/markers, or load the shipped config:
    rviz2 -d ros2_ws/src/fm_perception/rviz/field_test.rviz

Project: Onboard Visual Foundation Models for Mars Terrain Perception and Rover Navigation
"""

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import Imu, LaserScan
from std_msgs.msg import Bool, Float32MultiArray, Float64, String
from visualization_msgs.msg import Marker, MarkerArray

from fm_perception import sensor_snapshot as ss
from fm_perception.sensor_snapshot import SensorSnapshot

# The ROS types named in TOPIC_EXTRACTORS. That table stays free of ROS imports
# so it can be tested without a workspace; this is where the names are bound.
MSG_TYPES = {
    "std_msgs/msg/String": String,
    "std_msgs/msg/Bool": Bool,
    "std_msgs/msg/Float64": Float64,
    "std_msgs/msg/Float32MultiArray": Float32MultiArray,
    "sensor_msgs/msg/LaserScan": LaserScan,
    "sensor_msgs/msg/Imu": Imu,
    "geometry_msgs/msg/Twist": Twist,
}

# Taken from traversability_viz_node so the two overlays cannot show different
# colours for the same state on the same screen.
COLOUR_OK = (0.10, 0.85, 0.10)      # green, nothing asserting a stop
COLOUR_STOP = (0.90, 0.10, 0.10)    # red, something is stopping the rover
COLOUR_UNKNOWN = (0.55, 0.55, 0.55)  # grey, no terrain verdict to show


class SensorOverlayNode(Node):

    def __init__(self) -> None:
        super().__init__("sensor_overlay_node")

        self.declare_parameter("publish_rate_hz", 2.0)
        self.declare_parameter("frame_id", "base_link")
        self.declare_parameter("text_height_m", 0.20)
        self.declare_parameter("text_z_m", 0.0)

        rate = float(self.get_parameter("publish_rate_hz").value)
        self.frame_id = str(self.get_parameter("frame_id").value)
        self.text_height_m = float(self.get_parameter("text_height_m").value)
        self.text_z_m = float(self.get_parameter("text_z_m").value)

        self.snap = SensorSnapshot()

        # Best-effort on the scan, matching how the LiDAR driver publishes it and
        # how lidar_proximity_guard_node consumes it. A missed scan is replaced
        # 100 ms later; a queued one is stale by the time it is drawn.
        scan_qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
        )

        # One subscription per entry in TOPIC_EXTRACTORS, so this node and
        # experiments/bag_to_csv.py cannot develop different ideas about what a
        # topic means. Adding a sensor is a change to that table, not to this
        # file. /scan is the only one needing its own QoS: the driver publishes
        # best-effort, and a queued scan is stale by the time it is drawn.
        for topic, (type_name, _) in ss.TOPIC_EXTRACTORS.items():
            msg_type = MSG_TYPES[type_name]
            qos = scan_qos if topic == "/scan" else 10
            self.create_subscription(
                msg_type, topic,
                lambda msg, t=topic: self._on_message(t, msg), qos)

        self._pub = self.create_publisher(MarkerArray, "/sensor_overlay/markers", 10)
        self.create_timer(1.0 / rate if rate > 0 else 0.5, self._publish)

        self.get_logger().info(
            "sensor_overlay_node ready. Add MarkerArray /sensor_overlay/markers in RViz2."
        )

    # ── clock ────────────────────────────────────────────────────────────────

    def _now_s(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    # ── callbacks ────────────────────────────────────────────────────────────

    def _on_message(self, topic: str, msg) -> None:
        handled = self.snap.apply(topic, msg, self._now_s())
        if not handled:
            return
        if topic == "/terrain_classification" and not ss.parse_terrain_classification(
                msg.data):
            # Deliberately left to age out rather than shown. A malformed
            # verdict rendered as "soil 0.00" looks like a real reading.
            self.get_logger().warn(
                f"unparseable terrain classification {msg.data!r}",
                throttle_duration_sec=10.0,
            )

    # ── publishing ───────────────────────────────────────────────────────────

    def _colour(self, now: float) -> tuple[float, float, float]:
        """Red if anything is asserting a stop, grey with no verdict, else green.

        Stale counts as no assertion, not as safe: a stale stop flag is shown as
        STALE in the text, and colouring the whole panel red off a reading that
        may be seconds old would train the operator to ignore red.
        """
        for flag in ("e_stop", "lidar_stop", "imu_slope_stop"):
            if self.snap.get(flag, now):
                return COLOUR_STOP
        if self.snap.get("terrain_label", now) is None:
            return COLOUR_UNKNOWN
        return COLOUR_OK

    def _publish(self) -> None:
        now = self._now_s()

        marker = Marker()
        marker.header.frame_id = self.frame_id
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "sensor_overlay"
        marker.id = 0
        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD
        marker.pose.position.x = 0.0
        marker.pose.position.y = 0.0
        marker.pose.position.z = self.text_z_m
        marker.pose.orientation.w = 1.0
        marker.scale.z = self.text_height_m

        r, g, b = self._colour(now)
        marker.color.r, marker.color.g, marker.color.b = r, g, b
        marker.color.a = 1.0

        marker.text = self.snap.as_text(now)
        # Outlive one publish interval but not much more, so the panel disappears
        # if this node dies rather than freezing the last frame on screen.
        marker.lifetime.sec = 2

        array = MarkerArray()
        array.markers = [marker]
        self._pub.publish(array)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SensorOverlayNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
