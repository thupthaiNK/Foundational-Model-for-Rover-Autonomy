#!/usr/bin/env python3
"""
Purpose: Shortest possible real-hardware drive test. Drives the rover
         straight for a few seconds, then stops, and is meant to be watched
         rather than logged.

         It exists because of the 2026-07-24 stop-path bug, where the rover
         span in place whenever anything told it to stop. The lesson recorded
         from that is that "a command makes the wheels turn" and "a stop makes
         the wheels stop" are two different claims, and the second one is
         where the safety behaviour lives. This script is the second claim's
         test: the thing to watch is not that the rover moves, it is whether
         the wheels stop the instant the drive window ends.

         Run this before any longer drive, on any day the motor code changed.

Inputs:  None. Publishes /rover_command (exomy_ros2_msgs/RoverCommand), so
         robot_node and motor_node must both be running.
         Positional arguments: drive seconds (default 1.0), velocity 0-100
         (default 20).
Outputs: Progress on stdout. Writes nothing to disk. The real output is what
         you see the rover do.
Safety:  Drive time is hard-capped at 5 s regardless of the argument, and the
         stop is published ten times on every exit path including exceptions
         and Ctrl+C. Stand next to the rover with the battery accessible.
         Fastest certain stop is still cutting battery power.
How to run:
    # on the Pi, inside the ROS2 container
    source /opt/ros/humble/setup.bash
    source /ws/install/setup.bash
    python3 drive_smoke_test.py 1.0 20
Project: Onboard Visual Foundation Models for Mars Terrain Perception and Rover Navigation
"""
import sys
import time

from exomy_ros2_msgs.msg import RoverCommand
import rclpy

# FAKE_ACKERMANN with steering 90 puts all six wheels straight and drives them
# at the same speed. See rover.py joystickToSteeringAngle/joystickToVelocity.
LOCOMOTION_FAKE_ACKERMANN = 0
STEERING_STRAIGHT = 90.0
HARD_MAX_DRIVE_S = 5.0


def command(pub, vel: float, enabled: bool) -> None:
    msg = RoverCommand()
    msg.motors_enabled = enabled
    msg.locomotion_mode = LOCOMOTION_FAKE_ACKERMANN
    msg.vel = vel
    msg.steering = STEERING_STRAIGHT
    pub.publish(msg)


def main() -> None:
    drive_s = min(float(sys.argv[1]) if len(sys.argv) > 1 else 1.0,
                  HARD_MAX_DRIVE_S)
    vel = float(sys.argv[2]) if len(sys.argv) > 2 else 20.0

    rclpy.init()
    node = rclpy.create_node("drive_smoke_test")
    pub = node.create_publisher(RoverCommand, "/rover_command", 1)
    time.sleep(1.0)   # let the publisher connect before the first command
    try:
        print(f"DRIVE {drive_s:.1f}s at vel {vel:.0f}", flush=True)
        # motor_node stops the wheels if no command arrives for 5 s, so the
        # command has to be republished rather than sent once.
        t0 = time.time()
        while time.time() - t0 < drive_s:
            command(pub, vel, True)
            rclpy.spin_once(node, timeout_sec=0.1)
        print("STOP -- watch whether the wheels stop right now", flush=True)
    finally:
        for _ in range(10):
            command(pub, 0.0, False)
            rclpy.spin_once(node, timeout_sec=0.05)
        node.destroy_node()
        rclpy.shutdown()
        print("stop published 10x", flush=True)


if __name__ == "__main__":
    main()
