#!/usr/bin/env python3
"""
Purpose:    Rover kinematics — converts joystick commands to per-wheel steering
            angles and speeds for the 6-wheel ExoMy rover.
Inputs:     driving_command (-100 to 100), steering_command (degrees)
Outputs:    motor_angles[6], motor_speeds[6]
How to run: Imported by robot_node.py — not run directly.
Project:    Onboard Visual Foundation Models for Mars Terrain Perception and Rover Navigation
Original:   github.com/esa-prl/ExoMy_Software (rospy removed, logger passed in)
"""

import math
import numpy as np
from .locomotion_modes import LocomotionMode


class Rover:
    """6-wheel ExoMy kinematics.  Logger is optional (pass a rclpy node logger)."""

    FL, FR, CL, CR, RL, RR = range(6)
    FAKE_ACKERMANN, ACKERMANN, POINT_TURN, CRABBING = range(4)

    def __init__(self, logger=None):
        self.logger = logger
        self.locomotion_mode = LocomotionMode.FAKE_ACKERMANN

        self.wheel_x = 12.0
        self.wheel_y = 20.0

        max_steering_angle = 45
        self.ackermann_r_min = (
            abs(self.wheel_y) / math.tan(math.radians(max_steering_angle))
            + self.wheel_x
        )
        self.ackermann_r_max = 250

    def setLocomotionMode(self, locomotion_mode_command):
        if self.locomotion_mode.value != locomotion_mode_command:
            self.locomotion_mode = LocomotionMode(locomotion_mode_command)
            if self.logger:
                self.logger.info(
                    f"Locomotion mode → {self.locomotion_mode.name}"
                )

    def joystickToSteeringAngle(self, driving_command, steering_command):
        # float32[6] ROS2 message field (exomy_ros2_msgs/MotorCommands) --
        # every element assigned here must be a Python float, never an int.
        angles = [0.0] * 6
        deg = steering_command

        if self.locomotion_mode == LocomotionMode.FAKE_ACKERMANN:
            if driving_command == 0:
                return angles
            if 80 < deg < 100:
                pass  # straight forward — all zeros
            elif -80 < deg < -100:
                pass  # straight backward
            elif deg > 100:
                angles[self.FL] = 45.0; angles[self.FR] = 45.0
                angles[self.RL] = -45.0; angles[self.RR] = -45.0
            elif deg < -100:
                angles[self.FL] = 45.0; angles[self.FR] = 45.0
                angles[self.RL] = -45.0; angles[self.RR] = -45.0
            elif 0 <= deg <= 80:
                angles[self.FL] = -45.0; angles[self.FR] = -45.0
                angles[self.RL] = 45.0; angles[self.RR] = 45.0
            elif -80 < deg < 0:
                angles[self.FL] = -45.0; angles[self.FR] = -45.0
                angles[self.RL] = 45.0; angles[self.RR] = 45.0
            return angles

        if self.locomotion_mode == LocomotionMode.ACKERMANN:
            if driving_command == 0:
                return angles
            cos_steer = math.cos(math.radians(steering_command))
            r = (self.ackermann_r_max
                 if cos_steer == 0
                 else self.ackermann_r_max - abs(cos_steer) * (self.ackermann_r_max - self.ackermann_r_min))
            if r == self.ackermann_r_max:
                return angles
            inner = float(int(math.degrees(math.atan(self.wheel_x / (abs(r) - self.wheel_y)))))
            outer = float(int(math.degrees(math.atan(self.wheel_x / (abs(r) + self.wheel_y)))))
            if steering_command > 90 or steering_command < -90:
                angles[self.FL] = outer; angles[self.FR] = inner
                angles[self.RL] = -outer; angles[self.RR] = -inner
            else:
                angles[self.FL] = -inner; angles[self.FR] = -outer
                angles[self.RL] = inner; angles[self.RR] = outer
            return angles

        if self.locomotion_mode == LocomotionMode.POINT_TURN:
            angles[self.FL] = 45.0; angles[self.FR] = -45.0
            angles[self.RL] = -45.0; angles[self.RR] = 45.0
            return angles

        if self.locomotion_mode == LocomotionMode.CRABBING:
            if driving_command != 0:
                d = steering_command - 90 if steering_command > 0 else steering_command + 90
                d = float(np.clip(d, -75, 75))
                angles = [d] * 6
        return angles

    def joystickToVelocity(self, driving_command, steering_command):
        speeds = [0.0] * 6

        if self.locomotion_mode == LocomotionMode.FAKE_ACKERMANN:
            if driving_command > 0 and steering_command >= 0:
                speeds = [driving_command] * 6
            elif driving_command > 0 and steering_command < 0:
                speeds = [-driving_command] * 6
            elif driving_command < 0 and steering_command >= 0:
                # Straight reverse -- driving_command is already negative,
                # so this naturally reverses all wheels uniformly. Needed
                # for reactive_explorer_node's slope retreat, which backs
                # straight away from an over-tilted heading without turning
                # (turning in place while already tilted risks tipping).
                speeds = [driving_command] * 6
            return speeds

        if self.locomotion_mode == LocomotionMode.ACKERMANN:
            v = driving_command * (-1 if steering_command < 0 else 1)
            radius = (self.ackermann_r_max
                      - abs(math.cos(math.radians(steering_command)))
                      * (self.ackermann_r_max - self.ackermann_r_min))
            if v == 0:
                return speeds
            if radius == self.ackermann_r_max:
                return [float(v)] * 6
            rmax = radius + self.wheel_x
            a = self.wheel_y ** 2
            b = (abs(radius) + self.wheel_x) ** 2
            c = (abs(radius) - self.wheel_x) ** 2
            r1 = math.sqrt(a + b)
            r2 = float(rmax)
            r4 = math.sqrt(a + c)
            r5 = abs(radius) - self.wheel_x
            v1, v2 = float(v), float(v * r2 / r1)
            v4, v5 = float(v * r4 / r1), float(v * r5 / r1)
            if steering_command > 90 or steering_command < -90:
                speeds = [v1, v2, v1, v4, v5, v4]
            else:
                speeds = [v4, v5, v4, v1, v2, v1]
            return speeds

        if self.locomotion_mode == LocomotionMode.POINT_TURN:
            if driving_command != 0:
                mag = abs(driving_command)
                if -85 < steering_command < 85:
                    speeds = [-mag, mag, -mag, mag, -mag, mag]
                elif steering_command > 95 or steering_command < -95:
                    speeds = [mag, -mag, mag, -mag, mag, -mag]
            return speeds

        if self.locomotion_mode == LocomotionMode.CRABBING:
            if driving_command > 0:
                speeds = [50.0 if steering_command > 0 else -50.0] * 6
        return speeds
