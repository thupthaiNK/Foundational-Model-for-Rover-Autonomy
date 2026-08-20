#!/usr/bin/env python3
"""
Purpose: Bridges Nav2's global /cmd_vel output (published by velocity_smoother
         inside nav2_bringup's navigation_launch.py, which has no built-in way
         to retarget this topic without namespacing the entire Nav2 stack) to
         /exomy/cmd_vel, which is what the libgazebo_ros_diff_drive plugin in
         exomy.urdf.xacro (and the real ExoMy driver) subscribes to.
         Originally a 2-line unconditional forwarder with no speed limiting
         and no awareness of the rover's own safety signals -- any Nav2 or
         teleop command was relayed verbatim, including straight through an
         active E-stop or a too-close LiDAR reading. This version clamps
         linear/angular velocity to conservative real-world-safe limits
         (slow forward speed, slow turns to avoid flinging sand off the
         wheels) and forces a zero Twist whenever safety_watchdog_node's
         /e_stop or lidar_proximity_guard_node's /lidar_proximity_stop is
         active, regardless of what Nav2/teleop requested.
Inputs:  /cmd_vel (geometry_msgs/Twist)                 raw Nav2/teleop command
         /e_stop (std_msgs/Bool)                         from safety_watchdog_node.py
         /lidar_proximity_stop (std_msgs/Bool)            from lidar_proximity_guard_node.py
Outputs: /exomy/cmd_vel (geometry_msgs/Twist)            clamped, safety-gated command
Parameters:
    max_linear_velocity  (float, default 0.10): m/s, matches the existing
        terrain_controller_node.py soil-speed ceiling used everywhere else
        in this system, so the relay path is never faster than the reactive
        path.
    max_angular_velocity (float, default 0.3): rad/s (~17 deg/s) -- slow
        enough to keep wheel scrub/sand-throw low during a turn on loose
        terrain. Tune via ROS2 parameter if real-hardware testing shows a
        different value is needed.
How to run:
    source /opt/ros/humble/setup.bash
    source ros2_ws/install/setup.bash
    ros2 run fm_perception cmd_vel_relay_node.py
Project: Onboard Visual Foundation Models for Mars Terrain Perception and Rover Navigation
"""
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool


def clamp_twist_values(linear_x: float, angular_z: float,
                        max_linear: float, max_angular: float) -> tuple:
    """Symmetric clamp of a requested (linear_x, angular_z) pair."""
    clamped_linear = max(-max_linear, min(max_linear, linear_x))
    clamped_angular = max(-max_angular, min(max_angular, angular_z))
    return clamped_linear, clamped_angular


def should_gate_stop(e_stop_active: bool, lidar_stop_active: bool) -> bool:
    """True if any independent safety source demands a full stop."""
    return e_stop_active or lidar_stop_active


class CmdVelRelayNode(Node):

    def __init__(self):
        super().__init__("cmd_vel_relay_node")

        self.declare_parameter("max_linear_velocity", 0.10)
        self.declare_parameter("max_angular_velocity", 0.3)

        self.max_linear = self.get_parameter("max_linear_velocity").get_parameter_value().double_value
        self.max_angular = self.get_parameter("max_angular_velocity").get_parameter_value().double_value

        self._e_stop_active = False
        self._lidar_stop_active = False

        self.pub = self.create_publisher(Twist, "/exomy/cmd_vel", 10)
        self.create_subscription(Twist, "/cmd_vel", self._on_cmd_vel, 10)
        self.create_subscription(Bool, "/e_stop", self._on_e_stop, 10)
        self.create_subscription(Bool, "/lidar_proximity_stop", self._on_lidar_stop, 10)

        self.get_logger().info(
            f"cmd_vel_relay_node ready\n"
            f"  Relaying: /cmd_vel -> /exomy/cmd_vel\n"
            f"  max_linear_velocity={self.max_linear:.2f} m/s | "
            f"max_angular_velocity={self.max_angular:.2f} rad/s\n"
            "  Forces zero Twist when /e_stop or /lidar_proximity_stop is active"
        )

    def _on_e_stop(self, msg: Bool) -> None:
        self._e_stop_active = msg.data

    def _on_lidar_stop(self, msg: Bool) -> None:
        self._lidar_stop_active = msg.data

    def _on_cmd_vel(self, msg: Twist) -> None:
        out = Twist()

        if should_gate_stop(self._e_stop_active, self._lidar_stop_active):
            self.pub.publish(out)  # zero Twist -- gated
            return

        out.linear.x, out.angular.z = clamp_twist_values(
            msg.linear.x, msg.angular.z, self.max_linear, self.max_angular
        )
        self.pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = CmdVelRelayNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutdown -- final STOP command sent")
        node.pub.publish(Twist())
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
