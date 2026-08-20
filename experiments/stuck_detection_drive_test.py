#!/usr/bin/env python3
"""
Purpose: Safely drive the rover into a real obstacle to test
         real_stuck_detection_node.py's recovery escalation
         (BOOST_SPEED -> WIGGLE_TURN -> WIGGLE_RETRY -> STUCK_FAILSAFE) on
         real hardware, without depending on a human hitting Ctrl+C in time.

         Built after a 2026-07-25 test using a raw `ros2 topic pub -r 10`
         loop instead of a real script: it had no time cap, kept publishing
         straight-ahead commands the whole time real_stuck_detection_node
         was active (a race between the two publishers on /rover_command,
         since a raw topic pub has no idea what /stuck_detection/active
         means), and the rover rammed a wall for 30+ seconds before anyone
         intervened. This script fixes both problems: it stops publishing
         the moment /stuck_detection/active goes true (ceding /rover_command
         to the recovery node, matching the architecture's actual design),
         and it is hard-capped regardless of what the FSM does.

Inputs:  /stuck_detection/active (std_msgs/Bool)
         /stuck_detection/failsafe (std_msgs/Bool)
         Publishes /rover_command (exomy_ros2_msgs/RoverCommand), so
         robot_node, motor_node, and real_stuck_detection_node must all be
         running already.
Outputs: Progress on stdout. Writes nothing to disk. The real output is
         whether the rover recovers or reaches STUCK_FAILSAFE and stops.
Safety:  Total run time is hard-capped at HARD_MAX_S (30 s) regardless of
         FSM state. This script stops publishing straight-ahead commands
         the instant /stuck_detection/active is true, so it can never
         fight the recovery node for /rover_command. An explicit stop is
         published on every exit path (cap reached, failsafe seen,
         Ctrl+C, exception). Stand next to the rover with the battery
         accessible; fastest certain stop is still cutting battery power.
How to run:
    # on the Pi, inside the ROS2 container, with sllidar/robot_node/
    # motor_node/real_stuck_detection_node all already running, rover
    # pressed against a real solid obstacle
    source /opt/ros/humble/setup.bash
    source /ws/install/setup.bash
    python3 stuck_detection_drive_test.py 25
Project: Onboard Visual Foundation Models for Mars Terrain Perception and Rover Navigation
"""
import sys
import time

from exomy_ros2_msgs.msg import RoverCommand
from std_msgs.msg import Bool
import rclpy

# FAKE_ACKERMANN with steering 90 puts all six wheels straight and drives them
# at the same speed. See rover.py joystickToSteeringAngle/joystickToVelocity.
LOCOMOTION_FAKE_ACKERMANN = 0
STEERING_STRAIGHT = 90.0
HARD_MAX_S = 30.0


def command(pub, vel: float, enabled: bool) -> None:
    msg = RoverCommand()
    msg.motors_enabled = enabled
    msg.locomotion_mode = LOCOMOTION_FAKE_ACKERMANN
    msg.vel = vel
    msg.steering = STEERING_STRAIGHT
    pub.publish(msg)


def main() -> None:
    vel = float(sys.argv[1]) if len(sys.argv) > 1 else 25.0

    rclpy.init()
    node = rclpy.create_node("stuck_detection_drive_test")
    pub = node.create_publisher(RoverCommand, "/rover_command", 1)

    state = {"active": False, "failsafe": False}

    def on_active(msg: Bool) -> None:
        if msg.data != state["active"]:
            print(f"/stuck_detection/active -> {msg.data}", flush=True)
        state["active"] = msg.data

    def on_failsafe(msg: Bool) -> None:
        if msg.data and not state["failsafe"]:
            print("/stuck_detection/failsafe -> True", flush=True)
        state["failsafe"] = msg.data

    node.create_subscription(Bool, "/stuck_detection/active", on_active, 10)
    node.create_subscription(Bool, "/stuck_detection/failsafe", on_failsafe, 10)

    time.sleep(1.0)   # let pub/sub connect before the first command
    try:
        print(f"DRIVE at vel {vel:.0f}, hard-capped at {HARD_MAX_S:.0f}s total",
              flush=True)
        t0 = time.time()
        while time.time() - t0 < HARD_MAX_S:
            if state["failsafe"]:
                print("Recovery gave up (STUCK_FAILSAFE) -- stopping.", flush=True)
                break
            if state["active"]:
                # real_stuck_detection_node owns /rover_command now -- do not
                # publish anything, or the two publishers race for the same
                # topic exactly like the 2026-07-25 test that motivated this
                # script. Just watch and wait for it to resolve or fail.
                rclpy.spin_once(node, timeout_sec=0.1)
                continue
            command(pub, vel, True)
            rclpy.spin_once(node, timeout_sec=0.1)
        else:
            print(f"Hit the {HARD_MAX_S:.0f}s hard cap -- stopping.", flush=True)
    finally:
        for _ in range(10):
            command(pub, 0.0, False)
            rclpy.spin_once(node, timeout_sec=0.05)
        node.destroy_node()
        rclpy.shutdown()
        print("stop published 10x", flush=True)


if __name__ == "__main__":
    main()
