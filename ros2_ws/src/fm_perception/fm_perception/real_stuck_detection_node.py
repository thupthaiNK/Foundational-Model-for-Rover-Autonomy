#!/usr/bin/env python3
"""
Purpose: Real-hardware "commanded to move but not actually moving" detector
         and recovery node -- the real-rover counterpart to
         stuck_detection_node.py's Gazebo-only StuckDetectionFSM.
         The real rover has no wheel encoders and no gyro, so there is no
         odom to compare against a commanded velocity. This node instead
         judges forward progress from the closest LiDAR return directly
         ahead (front_sector_min_range, RealStuckDetectionFSM,
         is_stuck_lidar_front in stuck_detection_node.py). An earlier version
         used a whole-scan range diff instead; live testing on the real
         rover (2026-07-25) found that a rover yawing in place against an
         obstacle (never actually progressing) shifts every beam's bearing
         just as much as real forward motion does, so a whole-scan diff
         read that as "moved" and released back to MONITORING even though
         the rover was still stuck. Only the closest return dead ahead is
         immune to that: it only shrinks if the rover actually got closer to
         whatever is blocking it. Only detects the linear-stuck, forward-
         command case -- there is no sensor to confirm a wiggle turn
         actually rotated the rover, so recovery only escalates
         BOOST_SPEED -> WIGGLE_TURN (fixed duration, not yaw-checked) ->
         WIGGLE_RETRY -> STUCK_FAILSAFE, never the rotation-only path.
         Real hardware also cannot produce a zero-linear-velocity in-place
         turn: rover.py's FAKE_ACKERMANN kinematics return zero motor angles
         whenever the driving command is zero (`if driving_command == 0:
         return angles`), so WIGGLE_TURN here creeps forward at
         wiggle_vel with steering held at one of FAKE_ACKERMANN's two
         discrete turn bands (170 deg or 10 deg) instead of true angular-
         only rotation.
Inputs:  /scan (sensor_msgs/LaserScan)
         /rover_command (exomy_ros2_msgs/RoverCommand) -- observed while
         inactive, to read whichever node is currently driving.
Outputs: /rover_command (exomy_ros2_msgs/RoverCommand) -- only while active
         /stuck_detection/active (std_msgs/Bool)
         /stuck_detection/failsafe (std_msgs/Bool)
How to run:
    source /opt/ros/humble/setup.bash
    source ros2_ws/install/setup.bash
    ros2 run fm_perception real_stuck_detection_node.py
Project: Onboard Visual Foundation Models for Mars Terrain Perception and Rover Navigation
"""
import math

from exomy_ros2_msgs.msg import RoverCommand
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool
import rclpy
from rclpy.node import Node

from fm_perception.stuck_detection_node import RealStuckDetectionFSM, front_sector_min_range

LOCOMOTION_FAKE_ACKERMANN = 0
STEERING_STRAIGHT = 90.0
# FAKE_ACKERMANN's two discrete turn bands (rover.py joystickToSteeringAngle):
# deg > 100 and 0 <= deg <= 80 turn opposite ways. Used for WIGGLE_TURN since
# a true zero-vel in-place rotation isn't available on this kinematics.
STEERING_TURN_POSITIVE = 170.0
STEERING_TURN_NEGATIVE = 10.0


def rover_command_to_commanded_linear_x(vel: float, locomotion_mode: int) -> float:
    """RoverCommand.vel is only a forward/backward speed in FAKE_ACKERMANN
    mode. In POINT_TURN (and any other mode), vel is a rotation-speed
    magnitude -- reading it as commanded_linear_x makes any legitimate
    in-place turn (e.g. reactive_explorer_node.py's SCAN_AVOID_TURN) look
    like "commanded to drive forward fast", and since a pure rotation
    correctly does not shrink front_sector_min_range, that falsely trips
    this node's stuck detector on a turn that was never stuck at all.
    Confirmed live 2026-07-26 -- see
    project_lidar_avoidance_hardware_trial_20260726: reactive_explorer_node
    entered SCAN_AVOID_TURN (POINT_TURN, vel=100 meaning "full rotation
    speed") and this node misread that as a forward-stuck condition within
    one stuck_window_s, escalating into BOOST_SPEED/WIGGLE/RETREAT and
    fighting reactive_explorer_node for /rover_command the whole time.
    Pure function, no rclpy dependency -- fully unit-testable.
    """
    if locomotion_mode != LOCOMOTION_FAKE_ACKERMANN:
        return 0.0
    return vel


class RealStuckDetectionNode(Node):

    def __init__(self):
        super().__init__("real_stuck_detection_node")

        # min_commanded_speed/boost_max_speed/wiggle_vel are on
        # RoverCommand.vel's own 0-100 scale, NOT metres/second. Verified
        # live on the real rover (2026-07-25): a value like 0.10 sits inside
        # the drive servos' dead-band on this scale and produces stochastic
        # per-wheel spin instead of real forward motion. Every
        # verified-working real drive command this session used 15-30.
        self.declare_parameter("stuck_window_s", 4.0)
        self.declare_parameter("stuck_progress_threshold_m", 0.05)
        self.declare_parameter("min_commanded_speed", 1.0)
        self.declare_parameter("boost_max_speed", 25.0)
        self.declare_parameter("boost_duration_s", 4.0)
        self.declare_parameter("wiggle_angle_deg", 20.0)
        self.declare_parameter("angular_speed", 0.3)
        # 4 (not 3) gives an even left/right wiggle split -- direction
        # alternates each retry starting from +1, so 3 attempts give
        # +1,-1,+1 (one side tried twice), 4 give +1,-1,+1,-1 (even).
        self.declare_parameter("max_wiggle_attempts", 4)
        self.declare_parameter("wiggle_vel", 20.0)
        # Backs the rover straight away for a fixed distance/time before
        # finally giving up, so it doesn't sit unattended in whatever pose
        # it was fighting an obstacle in (e.g. front wheels resting mid-climb
        # on a wall or sand mound) -- added 2026-07-25 after live testing
        # ended in exactly that pose more than once.
        self.declare_parameter("retreat_speed", 20.0)
        self.declare_parameter("retreat_duration_s", 2.0)
        # 2026-08-02: the retreat used to run with zero sensing at all. Now
        # gated on the closest return in the rear sector before and during
        # every retreat tick -- 0.30 m matches lidar_proximity_guard_node's
        # own stop_distance_m, the safety clearance already established
        # elsewhere in this stack, rather than inventing a separate number.
        self.declare_parameter("retreat_min_clearance_m", 0.30)
        # 2026-08-02: the centre-mounted C1 sees its own mast at ~0.072 m in a
        # narrow sector somewhere on the rover, measured live in the sandpit
        # (see lidar_proximity_guard_node's min_ignore_m, same rover, same
        # sensor, same fix). Without this floor, front/rear sector reads can
        # land on that self-return instead of the environment. 0.2 m matches
        # the guard's own default rather than inventing a separate number.
        self.declare_parameter("min_ignore_m", 0.2)
        self.declare_parameter("sector_half_width_deg", 20.0)
        self.declare_parameter("publish_rate_hz", 10.0)
        self.declare_parameter("scan_topic", "/scan")
        self.declare_parameter("rover_command_topic", "/rover_command")

        self.fsm = RealStuckDetectionFSM(
            stuck_window_s=self.get_parameter("stuck_window_s").value,
            stuck_progress_threshold_m=self.get_parameter("stuck_progress_threshold_m").value,
            min_commanded_speed=self.get_parameter("min_commanded_speed").value,
            boost_max_speed=self.get_parameter("boost_max_speed").value,
            boost_duration_s=self.get_parameter("boost_duration_s").value,
            wiggle_angle_deg=self.get_parameter("wiggle_angle_deg").value,
            angular_speed=self.get_parameter("angular_speed").value,
            retreat_speed=self.get_parameter("retreat_speed").value,
            retreat_duration_s=self.get_parameter("retreat_duration_s").value,
            max_wiggle_attempts=self.get_parameter("max_wiggle_attempts").value,
            retreat_min_clearance_m=self.get_parameter("retreat_min_clearance_m").value,
        )

        self.wiggle_vel = self.get_parameter("wiggle_vel").value
        self.sector_half_width_deg = self.get_parameter("sector_half_width_deg").value
        self.min_ignore_m = self.get_parameter("min_ignore_m").value
        scan_topic = self.get_parameter("scan_topic").value
        rover_command_topic = self.get_parameter("rover_command_topic").value
        publish_hz = self.get_parameter("publish_rate_hz").value

        self._angle_min = None
        self._angle_increment = None
        self._ranges = None
        self._commanded_linear_x = 0.0
        self._prev_state = self.fsm.state

        self.create_subscription(LaserScan, scan_topic, self._scan_cb, 10)
        # Subscribes to the SAME topic it publishes to: while inactive, this
        # observes whichever upstream node is currently driving. Once active,
        # this node is the sole publisher, so its own commands are what's
        # read back -- same pattern as stuck_detection_node.py's Gazebo node.
        self.create_subscription(RoverCommand, rover_command_topic,
                                 self._rover_command_cb, 10)

        self.pub_rover_command = self.create_publisher(
            RoverCommand, rover_command_topic, 10)
        self.pub_active = self.create_publisher(Bool, "/stuck_detection/active", 10)
        self.pub_failsafe = self.create_publisher(Bool, "/stuck_detection/failsafe", 10)

        self.create_timer(1.0 / publish_hz, self._tick)

        self.get_logger().info(
            "real_stuck_detection_node ready (LiDAR front-sector-based, no odom/gyro)\n"
            f"  stuck_window_s={self.fsm.stuck_window_s:.1f} "
            f"stuck_progress_threshold_m={self.fsm.stuck_progress_threshold_m:.3f} "
            f"boost_max_speed={self.fsm.boost_max_speed:.2f} "
            f"max_wiggle_attempts={self.fsm.max_wiggle_attempts}"
        )

    def _scan_cb(self, msg: LaserScan) -> None:
        self._ranges = msg.ranges
        self._angle_min = msg.angle_min
        self._angle_increment = msg.angle_increment

    def _rover_command_cb(self, msg: RoverCommand) -> None:
        self._commanded_linear_x = rover_command_to_commanded_linear_x(
            msg.vel, msg.locomotion_mode
        )

    def _tick(self) -> None:
        if self._ranges is None:
            return

        front_range = front_sector_min_range(
            self._ranges, self._angle_min, self._angle_increment,
            self.sector_half_width_deg, min_ignore_m=self.min_ignore_m)
        rear_range = front_sector_min_range(
            self._ranges, self._angle_min, self._angle_increment,
            self.sector_half_width_deg, center_bearing_rad=math.pi,
            min_ignore_m=self.min_ignore_m)

        now_s = self.get_clock().now().nanoseconds / 1e9
        out = self.fsm.step(now_s=now_s, front_range=front_range,
                            commanded_linear_x=self._commanded_linear_x,
                            rear_range=rear_range)

        active_msg = Bool()
        active_msg.data = out["active"]
        self.pub_active.publish(active_msg)

        failsafe_msg = Bool()
        failsafe_msg.data = out["failsafe"]
        self.pub_failsafe.publish(failsafe_msg)

        if out["active"]:
            cmd = RoverCommand()
            cmd.motors_enabled = True
            cmd.locomotion_mode = LOCOMOTION_FAKE_ACKERMANN
            if out["angular_z"] != 0.0:
                cmd.vel = self.wiggle_vel
                cmd.steering = (STEERING_TURN_POSITIVE if out["angular_z"] > 0.0
                                else STEERING_TURN_NEGATIVE)
            else:
                cmd.vel = out["linear_x"]
                cmd.steering = STEERING_STRAIGHT
            self.pub_rover_command.publish(cmd)

        if out["state"] != self._prev_state:
            self.get_logger().info(f"real_stuck_detection: {self._prev_state} -> {out['state']}")
            self._prev_state = out["state"]


def main(args=None):
    rclpy.init(args=args)
    node = RealStuckDetectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
