"""
Purpose: RViz2 traversability visualisation node — subscribes to /terrain_classification
         and publishes colour-coded markers (green=SAFE, yellow=CAUTION, orange=HAZARD,
         red=STOP) for display in RViz2. Provides a thesis figure showing the perception
         pipeline output as a real-time traversability status overlay.
Inputs:  /terrain_classification  (std_msgs/String)   "label:confidence"
         /exomy/cmd_vel           (geometry_msgs/Twist) current speed command
Outputs: /traversability_marker   (visualization_msgs/MarkerArray)
             — sphere marker: colour = policy
             — text marker: "label | POLICY | speed m/s | conf"
How to run:
    # Terminal 1: launch Gazebo + DINOv2 node
    source /opt/ros/humble/setup.bash && source ros2_ws/install/setup.bash
    ros2 launch simulation/launch/dinov2_controller_test.launch.py

    # Terminal 2: run visualisation node
    source /opt/ros/humble/setup.bash && source ros2_ws/install/setup.bash
    ros2 run fm_perception traversability_viz_node

    # Terminal 3: open RViz2 and add MarkerArray topic /traversability_marker
    rviz2

Project: Onboard Visual Foundation Models for Mars Terrain Perception and Rover Navigation
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray
from builtin_interfaces.msg import Duration

# Policy colour map  (R, G, B)
POLICY_COLOUR = {
    "SAFE":    (0.10, 0.85, 0.10),   # green
    "CAUTION": (1.00, 0.85, 0.00),   # yellow
    "HAZARD":  (1.00, 0.45, 0.00),   # orange
    "STOP":    (0.90, 0.10, 0.10),   # red
}
UNKNOWN_COLOUR = (0.55, 0.55, 0.55)  # grey for uncertain / no data

TERRAIN_POLICY = {
    "soil":      ("SAFE",    0.10),
    "bedrock":   ("HAZARD",  0.03),
    "sand":      ("CAUTION", 0.05),
    "big_rock":  ("STOP",    0.00),
    "uncertain": ("STOP",    0.00),
}

CONF_THRESHOLD = 0.40   # below this → treat as uncertain


class TraversabilityVizNode(Node):
    """Publishes colour-coded RViz2 markers for the current traversability state."""

    def __init__(self):
        super().__init__("traversability_viz_node")

        self._label  = "uncertain"
        self._conf   = 0.0
        self._speed  = 0.0

        self.create_subscription(String, "/terrain_classification", self._terrain_cb, 10)
        self.create_subscription(Twist,  "/exomy/cmd_vel",          self._vel_cb,     10)
        self._pub = self.create_publisher(MarkerArray, "/traversability_marker", 10)

        self.create_timer(0.2, self._publish_markers)
        self.get_logger().info("Traversability viz node ready. Add /traversability_marker in RViz2.")

    def _terrain_cb(self, msg: String):
        parts = msg.data.split(":")
        if len(parts) >= 2:
            self._label = parts[0].strip().lower()
            try:
                self._conf = float(parts[1])
            except ValueError:
                self._conf = 0.0

    def _vel_cb(self, msg: Twist):
        self._speed = msg.linear.x

    def _publish_markers(self):
        label = self._label
        conf  = self._conf

        if conf < CONF_THRESHOLD or label not in TERRAIN_POLICY:
            policy, speed_target = "STOP", 0.00
            label = "uncertain"
        else:
            policy, speed_target = TERRAIN_POLICY[label]

        r, g, b = POLICY_COLOUR.get(policy, UNKNOWN_COLOUR)
        lifetime = Duration(sec=1, nanosec=0)

        # Marker 0 — coloured sphere representing current traversability state
        sphere = Marker()
        sphere.header.frame_id = "base_link"
        sphere.header.stamp    = self.get_clock().now().to_msg()
        sphere.ns              = "traversability"
        sphere.id              = 0
        sphere.type            = Marker.SPHERE
        sphere.action          = Marker.ADD
        sphere.pose.position.x = 0.0
        sphere.pose.position.y = 0.0
        sphere.pose.position.z = 0.6
        sphere.pose.orientation.w = 1.0
        sphere.scale.x = sphere.scale.y = sphere.scale.z = 0.4
        sphere.color.r = r
        sphere.color.g = g
        sphere.color.b = b
        sphere.color.a = 0.85
        sphere.lifetime = lifetime

        # Marker 1 — text label above sphere
        text = Marker()
        text.header.frame_id = "base_link"
        text.header.stamp    = self.get_clock().now().to_msg()
        text.ns              = "traversability"
        text.id              = 1
        text.type            = Marker.TEXT_VIEW_FACING
        text.action          = Marker.ADD
        text.pose.position.x = 0.0
        text.pose.position.y = 0.0
        text.pose.position.z = 1.0
        text.pose.orientation.w = 1.0
        text.scale.z           = 0.18
        text.color.r = text.color.g = text.color.b = 1.0
        text.color.a = 1.0
        text.text    = f"{label.upper()} | {policy}\nconf={conf:.2f}  cmd={self._speed:.2f}m/s"
        text.lifetime = lifetime

        array = MarkerArray()
        array.markers = [sphere, text]
        self._pub.publish(array)


def main():
    rclpy.init()
    node = TraversabilityVizNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
