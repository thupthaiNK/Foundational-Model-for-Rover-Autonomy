#!/usr/bin/env python3
"""
Purpose: Bridges the fm_perception perception/control stack's
         geometry_msgs/Twist output (/exomy/cmd_vel, from
         terrain_controller_node.py and reactive_explorer_node.py, built for
         Gazebo's diff-drive plugin) to the real-hardware
         exomy_ros2_msgs/RoverCommand interface (/rover_command, consumed by
         robot_node.py -> motor_node.py -> real PWM). Without this bridge,
         nothing built for the Gazebo pipeline can drive the real rover --
         no such bridge existed anywhere in this repo before.
         Every current sender only ever commands one of: pure forward/backward
         (angular.z==0), pure in-place rotation (linear.x==0), or full stop --
         never a combined arc -- so this only needs to produce FAKE_ACKERMANN
         (straight) or POINT_TURN (in-place) RoverCommands, not general
         Ackermann steering.
         Speed magnitude is preserved (not collapsed to on/off at a fixed
         50%): linear.x is scaled against max_linear_velocity (default
         0.10 m/s, matching terrain_controller_node's soil policy speed and
         cmd_vel_relay_node's own conservative limit) and angular.z against
         max_angular_velocity (default 0.3 rad/s, matching
         reactive_explorer_node's/cmd_vel_relay_node's turn speed) into the
         RoverCommand vel field's -100..100 range. This relies on the
         companion fix in rover.py making joystickToVelocity() scale
         proportionally instead of hardcoding +/-50.0 -- Motors.setDriving()
         (the actual PWM hardware layer, unchanged, already hardware-tested)
         already applies motor_speeds proportionally, so this preserves real
         speed differentiation (soil/sand/bedrock) all the way to the wheels.
         KNOWN LIMITATION: the mapping from a rotation's sign (angular.z > 0
         vs < 0) to which POINT_TURN steering value (STEERING_TURN_A=0.0 vs
         STEERING_TURN_B=180.0) is physically left vs right is UNVERIFIED --
         no documented convention exists anywhere in this repo (it traces
         back to the original ESA joystick_parser_node.py, which was never
         ported to this ROS2 workspace). The mapping is internally consistent
         (same sign always produces the same physical direction), but which
         physical direction that is needs confirming once real hardware is
         accessible again.
         Cedes /rover_command to real_stuck_detection_node.py whenever
         /stuck_detection/active is true -- added 2026-07-26 after a live
         hardware trial found this bridge kept republishing
         reactive_explorer_node.py's turn commands every tick regardless of
         who else wanted the topic, racing directly against
         real_stuck_detection_node's own independent publisher on the same
         /rover_command topic (visible on real hardware as robot_node's
         locomotion mode flapping between POINT_TURN and FAKE_ACKERMANN
         every ~50ms). This matches the priority already stated in this
         launch file's own docstring ("real_stuck_detection_node -- highest
         /rover_command priority") but never actually implemented anywhere
         until now. See project_lidar_avoidance_hardware_trial_20260726.
         Cannot be tested on real hardware right now (no lab access) -- same
         status as icm20948_driver_node.py: written and unit-tested in
         isolation, hardware verification deferred.
Inputs:  /exomy/cmd_vel (geometry_msgs/Twist)
         /stuck_detection/active (std_msgs/Bool) -- cede /rover_command while true
Outputs: /rover_command (exomy_ros2_msgs/RoverCommand)
How to run:
    source /opt/ros/humble/setup.bash
    source ros2_ws/install/setup.bash
    ros2 run exomy_ros2 cmd_vel_to_rovercommand_bridge_node.py
Project: Onboard Visual Foundation Models for Mars Terrain Perception and Rover Navigation
"""
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool
import rclpy
from rclpy.node import Node

from exomy_ros2_msgs.msg import RoverCommand
try:
    from .locomotion_modes import LocomotionMode
except ImportError:
    # ros2run intermittently execs installed nodes without package context in
    # this environment (same latent issue exists in robot_node.py's `from
    # .rover import Rover`) -- fall back to an absolute import so this node
    # doesn't depend on that context being set up correctly.
    from exomy_ros2.locomotion_modes import LocomotionMode


STEERING_STRAIGHT = 90.0
STEERING_TURN_A   = 0.0    # angular_z > 0 -- UNVERIFIED physical direction, see module docstring
STEERING_TURN_B   = 180.0  # angular_z < 0 -- UNVERIFIED physical direction, see module docstring


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def twist_to_rover_command(linear_x: float, angular_z: float,
                            max_linear_mps: float = 0.10,
                            max_angular_rps: float = 0.3):
    """Convert a (linear_x, angular_z) Twist snapshot into RoverCommand fields.

    Returns (motors_enabled, locomotion_mode, vel, steering). Pure function,
    no ROS2/rclpy dependency -- fully unit-testable.
    """
    if linear_x != 0.0 and angular_z != 0.0:
        # Not something any current sender does -- defensive stop rather
        # than guessing which one should win.
        return False, LocomotionMode.FAKE_ACKERMANN.value, 0.0, STEERING_STRAIGHT

    if angular_z != 0.0:
        vel = _clamp(abs(angular_z) / max_angular_rps * 100.0, 0.0, 100.0)
        steering = STEERING_TURN_A if angular_z > 0.0 else STEERING_TURN_B
        return True, LocomotionMode.POINT_TURN.value, vel, steering

    if linear_x != 0.0:
        vel = _clamp(linear_x / max_linear_mps * 100.0, -100.0, 100.0)
        return True, LocomotionMode.FAKE_ACKERMANN.value, vel, STEERING_STRAIGHT

    return False, LocomotionMode.FAKE_ACKERMANN.value, 0.0, STEERING_STRAIGHT


class CmdVelToRoverCommandBridgeNode(Node):

    def __init__(self):
        super().__init__("cmd_vel_to_rovercommand_bridge_node")

        self.declare_parameter("cmd_vel_topic", "/exomy/cmd_vel")
        self.declare_parameter("rover_command_topic", "/rover_command")
        self.declare_parameter("max_linear_velocity", 0.10)
        self.declare_parameter("max_angular_velocity", 0.3)
        self.declare_parameter("stale_timeout_s", 3.0)
        self.declare_parameter("publish_rate_hz", 10.0)
        self.declare_parameter("stuck_detection_active_topic", "/stuck_detection/active")
        # If real_stuck_detection_node dies mid-recovery, this bridge must
        # NOT stay silent forever -- before this cede logic existed, the
        # bridge was the last node that would still command a stop in that
        # scenario, and robot_node has no command watchdog of its own.
        # Treat the flag as inactive (resume normal publishing, which stops
        # the rover the moment /exomy/cmd_vel itself goes stale) if no
        # /stuck_detection/active message has arrived recently -- that node
        # publishes on every tick, both True and False, so this only fires
        # on a genuine dropout, not a normal state change.
        self.declare_parameter("stuck_detection_active_stale_timeout_s", 1.0)

        cmd_vel_topic        = self.get_parameter("cmd_vel_topic").value
        rover_command_topic  = self.get_parameter("rover_command_topic").value
        self.max_linear      = self.get_parameter("max_linear_velocity").value
        self.max_angular     = self.get_parameter("max_angular_velocity").value
        self.stale_timeout_s = self.get_parameter("stale_timeout_s").value
        self.stuck_detection_active_stale_timeout_s = self.get_parameter(
            "stuck_detection_active_stale_timeout_s"
        ).value
        publish_hz           = self.get_parameter("publish_rate_hz").value

        self._last_linear_x  = 0.0
        self._last_angular_z = 0.0
        self._last_msg_time  = None  # None until the first /exomy/cmd_vel arrives
        self._stuck_detection_active = False
        self._last_stuck_active_msg_time = None

        stuck_detection_active_topic = self.get_parameter("stuck_detection_active_topic").value
        self.sub = self.create_subscription(Twist, cmd_vel_topic, self._cmd_vel_cb, 10)
        self.create_subscription(
            Bool, stuck_detection_active_topic, self._stuck_detection_active_cb, 10
        )
        self.pub = self.create_publisher(RoverCommand, rover_command_topic, 10)
        self.create_timer(1.0 / publish_hz, self._tick)

        self.get_logger().info(
            "cmd_vel_to_rovercommand_bridge_node ready\n"
            f"  {cmd_vel_topic} -> {rover_command_topic}\n"
            f"  max_linear_velocity={self.max_linear:.2f} m/s | "
            f"max_angular_velocity={self.max_angular:.2f} rad/s | "
            f"stale_timeout_s={self.stale_timeout_s:.1f}"
        )

    def _cmd_vel_cb(self, msg: Twist) -> None:
        self._last_linear_x  = msg.linear.x
        self._last_angular_z = msg.angular.z
        self._last_msg_time  = self.get_clock().now()

    def _stuck_detection_active_cb(self, msg: Bool) -> None:
        self._stuck_detection_active = msg.data
        self._last_stuck_active_msg_time = self.get_clock().now()

    def _tick(self) -> None:
        stuck_flag_stale = (
            self._last_stuck_active_msg_time is None
            or (self.get_clock().now() - self._last_stuck_active_msg_time).nanoseconds / 1e9
               > self.stuck_detection_active_stale_timeout_s
        )
        if self._stuck_detection_active and not stuck_flag_stale:
            # real_stuck_detection_node is the sole /rover_command publisher
            # while it's recovering -- do not also publish here, or the two
            # independent 10Hz timers race on the same topic (see module
            # docstring). Just skip this tick entirely; there is nothing
            # useful for this bridge to say while it doesn't own the topic.
            # If that node's own topic has gone stale (crashed/killed
            # mid-recovery), fall through to normal publishing instead of
            # staying silent forever -- see stuck_detection_active_stale_timeout_s.
            return

        stale = (
            self._last_msg_time is None
            or (self.get_clock().now() - self._last_msg_time).nanoseconds / 1e9
               > self.stale_timeout_s
        )

        if stale:
            enabled, mode, vel, steering = (
                False, LocomotionMode.FAKE_ACKERMANN.value, 0.0, STEERING_STRAIGHT
            )
        else:
            enabled, mode, vel, steering = twist_to_rover_command(
                self._last_linear_x, self._last_angular_z,
                self.max_linear, self.max_angular,
            )

        cmd = RoverCommand()
        cmd.motors_enabled  = enabled
        cmd.locomotion_mode = mode
        cmd.vel             = vel
        cmd.steering        = steering
        self.pub.publish(cmd)


def main(args=None):
    rclpy.init(args=args)
    node = CmdVelToRoverCommandBridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
