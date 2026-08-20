#!/usr/bin/env python3
"""
Purpose: Publishes a latched MarkerArray of ground-truth zone slabs on /zone_markers
         so RViz2 shows the 5 terrain zones with colour-coded labels. Each zone is a
         flat 3×3m slab at ground level (z=0.05) with a text label floating above it.
         Designed to run alongside dinov2_rviz2.launch.py for the §4.8 thesis figure.
Inputs:  None (standalone node)
Outputs: /zone_markers (visualization_msgs/MarkerArray) — latched, republished every 5s
How to run:
    source /opt/ros/humble/setup.bash
    source ros2_ws/install/setup.bash
    python3 experiments/publish_zone_markers.py
    # Or launch via dinov2_rviz2.launch.py (included automatically)
Project: Onboard Visual Foundation Models for Mars Terrain Perception and Rover Navigation
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, HistoryPolicy
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point

# ── Zone definitions: (name, x, y, r, g, b) ─────────────────────────────────
# Positions match ZONE_POSES in dinov2_traversability_experiment.py
ZONES = [
    {"name": "soil_zone",    "x": -7.5, "y":  6.0, "r": 0.20, "g": 0.70, "b": 0.20},  # green
    {"name": "bedrock_zone", "x":  7.5, "y":  6.0, "r": 0.90, "g": 0.50, "b": 0.10},  # orange
    {"name": "sand_zone",    "x": -7.5, "y": -6.0, "r": 0.90, "g": 0.85, "b": 0.15},  # yellow
    {"name": "rock_cluster", "x":  2.0, "y": -4.0, "r": 0.85, "g": 0.15, "b": 0.15},  # red
    {"name": "boulder_zone", "x":  2.0, "y": -8.0, "r": 0.50, "g": 0.05, "b": 0.05},  # dark red
]

SLAB_SIZE  = 3.0   # metres — slab side length
SLAB_Z     = 0.02  # metres — slab height (thin, sits on ground)
TEXT_Z     = 1.2   # metres — label height above ground


class ZoneMarkerPublisher(Node):

    def __init__(self):
        super().__init__("zone_marker_publisher")

        # Transient-local (latched) so RViz2 receives markers on connection
        qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
        )
        self._pub = self.create_publisher(MarkerArray, "/zone_markers", qos)

        # Publish immediately, then every 5s to handle late subscribers
        self._publish()
        self.create_timer(5.0, self._publish)
        self.get_logger().info(
            f"Zone marker publisher ready — {len(ZONES)} zones on /zone_markers"
        )

    def _publish(self) -> None:
        msg = MarkerArray()
        now = self.get_clock().now().to_msg()

        for i, zone in enumerate(ZONES):
            # ── Slab marker ──────────────────────────────────────────────────
            slab = Marker()
            slab.header.frame_id = "world"
            slab.header.stamp    = now
            slab.ns              = "zones"
            slab.id              = i * 2
            slab.type            = Marker.CUBE
            slab.action          = Marker.ADD
            slab.pose.position.x = zone["x"]
            slab.pose.position.y = zone["y"]
            slab.pose.position.z = SLAB_Z / 2
            slab.pose.orientation.w = 1.0
            slab.scale.x = SLAB_SIZE
            slab.scale.y = SLAB_SIZE
            slab.scale.z = SLAB_Z
            slab.color.r = zone["r"]
            slab.color.g = zone["g"]
            slab.color.b = zone["b"]
            slab.color.a = 0.55
            slab.lifetime.sec = 0  # never expires

            # ── Text label marker ────────────────────────────────────────────
            label = Marker()
            label.header.frame_id = "world"
            label.header.stamp    = now
            label.ns              = "zone_labels"
            label.id              = i * 2 + 1
            label.type            = Marker.TEXT_VIEW_FACING
            label.action          = Marker.ADD
            label.pose.position.x = zone["x"]
            label.pose.position.y = zone["y"]
            label.pose.position.z = TEXT_Z
            label.pose.orientation.w = 1.0
            label.scale.z = 0.5  # text height in metres
            label.color.r = zone["r"]
            label.color.g = zone["g"]
            label.color.b = zone["b"]
            label.color.a = 1.0
            label.text    = zone["name"].replace("_", "\n")
            label.lifetime.sec = 0

            msg.markers.append(slab)
            msg.markers.append(label)

        self._pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = ZoneMarkerPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
