#!/usr/bin/env python3
"""
Purpose:    ROS2 robot node — converts RoverCommand (joystick input) to
            MotorCommands (per-wheel angles + speeds) using ExoMy kinematics.
Inputs:     /rover_command (exomy_ros2/RoverCommand)
Outputs:    /motor_commands (exomy_ros2/MotorCommands)
How to run: ros2 run exomy_ros2 robot_node.py
Project:    Onboard Visual Foundation Models for Mars Terrain Perception and Rover Navigation
Original:   github.com/esa-prl/ExoMy_Software robot_node.py (rospy → rclpy)
"""

import rclpy
from rclpy.node import Node

from exomy_ros2_msgs.msg import MotorCommands, RoverCommand
try:
    from .rover import Rover
except ImportError:
    # ros2run intermittently execs installed nodes without package context in
    # this environment (same issue documented for
    # cmd_vel_to_rovercommand_bridge_node.py) -- fall back to an absolute
    # import so this node doesn't depend on that context being set up
    # correctly.
    from exomy_ros2.rover import Rover


class RobotNode(Node):
    def __init__(self):
        super().__init__("robot_node")
        self.rover = Rover(logger=self.get_logger())

        self.sub = self.create_subscription(
            RoverCommand, "/rover_command", self._joy_callback, 1
        )
        self.pub = self.create_publisher(MotorCommands, "/motor_commands", 1)
        self.get_logger().info("robot_node started — waiting for /rover_command")

    def _joy_callback(self, msg: RoverCommand):
        cmds = MotorCommands()

        if msg.motors_enabled:
            self.rover.setLocomotionMode(msg.locomotion_mode)
            cmds.motor_angles = self.rover.joystickToSteeringAngle(
                msg.vel, msg.steering
            )
            cmds.motor_speeds = self.rover.joystickToVelocity(
                msg.vel, msg.steering
            )
        else:
            cmds.motor_angles = self.rover.joystickToSteeringAngle(0.0, 0.0)
            cmds.motor_speeds = self.rover.joystickToVelocity(0.0, 0.0)

        self.pub.publish(cmds)


def main(args=None):
    rclpy.init(args=args)
    node = RobotNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
