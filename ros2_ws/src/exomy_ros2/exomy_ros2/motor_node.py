#!/usr/bin/env python3
"""
Purpose:    ROS2 motor node — receives MotorCommands and drives the ExoMy
            wheels via Adafruit PCA9685 PWM over I2C (Raspberry Pi only).
            On non-RPi machines the Motors class will fail to import — that
            is expected; this node only runs on the real hardware.
Inputs:     /motor_commands (exomy_ros2/MotorCommands)
Outputs:    Motors move via I2C PWM signals
How to run: ros2 run exomy_ros2 motor_node.py   (on Raspberry Pi 4 only)
Project:    Onboard Visual Foundation Models for Mars Terrain Perception and Rover Navigation
Original:   github.com/esa-prl/ExoMy_Software motor_node.py (rospy → rclpy)
"""

import rclpy
from rclpy.node import Node

from exomy_ros2_msgs.msg import MotorCommands

# `ros2 run` executes this file as a top-level script, not as a package
# submodule, so a relative `from .motors import Motors` raises ImportError
# ("attempted relative import with no known parent package") and would drop us
# into DRY-RUN even on the real rover. Try the relative form first (used when
# the module is imported as part of the package, e.g. tests), then fall back to
# the absolute form (used by `ros2 run`). The outer guard still catches a
# genuinely missing Adafruit_PCA9685 on a dev machine and keeps DRY-RUN there.
try:
    try:
        from .motors import Motors
    except ImportError:
        from exomy_ros2.motors import Motors
    HARDWARE_AVAILABLE = True
except ImportError:
    Motors = None
    HARDWARE_AVAILABLE = False


class MotorNode(Node):

    # PWM pin defaults (can be overridden via ROS2 parameters).
    #
    # These are ExoMy's AS-BUILT channels, not the upstream ExoMy defaults.
    # Until 2026-07-23 this dict held the upstream sequential layout
    # (drive 0,2,4,6,8,10 / steer 1,3,5,7,9,11), which does not match how
    # this rover is actually wired. Sending a continuous-rotation throttle
    # to a positional steering servo can stall it against its end stop, so
    # the mismatch was a real hardware risk, never a cosmetic one.
    #
    # Verified live on the rover (2026-07-23) by driving one PCA9685 channel
    # at a time with adafruit_servokit and recording which wheel responded.
    # All 12 channels confirmed. Channels 3, 11, 12, 13 are unused.
    DEFAULT_PINS = {
        "pin_drive_fl": 14, "pin_steer_fl": 8,
        "pin_drive_fr": 2,  "pin_steer_fr": 0,
        "pin_drive_cl": 6,  "pin_steer_cl": 7,
        "pin_drive_cr": 4,  "pin_steer_cr": 5,
        "pin_drive_rl": 10, "pin_steer_rl": 9,
        "pin_drive_rr": 15, "pin_steer_rr": 1,
        # All six steer channels trimmed off the textbook 307 on 2026-07-25:
        # at 307 across the board, the rover consistently drifted left on a
        # flat floor, visibly worst at FL and CL (photo-verified against a
        # straight-edge laid along the chassis centreline). FL/CL trimmed
        # first and confirmed by driving straight before/after (untrimmed
        # drifted left throughout; trimmed drove straight for 30-50 cm with
        # a small residual right drift). FR/RL/RR then trimmed +5 each to
        # remove that residual drift -- with all five trims in place the
        # rover drove straight over a full test run. See
        # project_wheel_steering_trim_20260725 memory for the full session.
        "steer_pwm_neutral_fl": 322, "steer_pwm_neutral_fr": 312,
        "steer_pwm_neutral_cl": 317, "steer_pwm_neutral_cr": 307,
        "steer_pwm_neutral_rl": 312, "steer_pwm_neutral_rr": 312,
        "steer_pwm_range":  100,
        "drive_pwm_neutral": 307,
        "drive_pwm_range":   200,
    }

    def __init__(self):
        super().__init__("motor_node")

        # Declare all parameters with defaults
        for name, default in self.DEFAULT_PINS.items():
            self.declare_parameter(name, default)

        # I2C bus the PCA9685 sits on. Explicit because Adafruit_GPIO can't
        # auto-detect the bus on this Pi 4 / Debian trixie. Bus 1 = hardware
        # I2C1, where the Servo HAT lives (i2cdetect -y 1 shows 0x40).
        self.declare_parameter("i2c_bus", 1)

        if not HARDWARE_AVAILABLE:
            self.get_logger().warn(
                "Adafruit_PCA9685 not available — running in DRY-RUN mode "
                "(motor commands logged but not sent to hardware). "
                "Deploy on Raspberry Pi 4 for real motor control."
            )
            self.motors = None
        else:
            self.motors = Motors(node=self)
            self.get_logger().info("Motors initialised — hardware ready")

        self.sub = self.create_subscription(
            MotorCommands, "/motor_commands", self._cmd_callback, 1
        )
        # Safety watchdog: if no command received for 5s, stop motors
        self._watchdog = self.create_timer(5.0, self._watchdog_cb)
        self.get_logger().info("motor_node started")

    def _cmd_callback(self, msg: MotorCommands):
        # Reset watchdog on every received command
        self._watchdog.cancel()
        self._watchdog = self.create_timer(5.0, self._watchdog_cb)

        if self.motors:
            self.motors.setSteering(list(msg.motor_angles))
            self.motors.setDriving(list(msg.motor_speeds))
        else:
            self.get_logger().debug(
                f"[DRY-RUN] angles={list(msg.motor_angles)[:4]}... "
                f"speeds={list(msg.motor_speeds)[:4]}..."
            )

    def _watchdog_cb(self):
        self.get_logger().warn("Watchdog fired — stopping motors")
        if self.motors:
            self.motors.stopMotors()

    def destroy_node(self):
        if self.motors:
            self.motors.stopMotors()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = MotorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
