#!/usr/bin/env python3
"""
Purpose: Measure real ExoMy wheel speed (m/s) vs. commanded RoverCommand.vel
         (0-100 scale) on the real hardware, on a short (~3m) flat straight
         course. Drives straight for a fixed, script-controlled duration
         (not distance -- there is no odometry/distance sensing on this
         real-hardware path), then stops. The operator measures the actual
         distance travelled with a tape measure and enters it; the script
         computes speed = distance / duration.

         Two-stage protocol (per the plan agreed with the user, 2026-07-26):
         1. PROBE run: a short 1.0s pulse at the requested vel, to estimate
            speed before committing to a longer run.
         2. OFFICIAL run: uses compute_official_duration_s() to pick a
            duration (up to --max-duration, default 6.0s) that keeps the
            expected travel distance at or below --target-distance (default
            2.2m), leaving a safety buffer on the 3m course. If the probe
            speed is 0 (rover barely moved), the full max duration is used.

Inputs:  CLI args: vel (float, 0-100), --duration (float s, required --
         pass 1.0 for a probe run, or the value compute_official_duration_s
         suggests for the official run).
         Publishes /rover_command (exomy_ros2_msgs/RoverCommand); requires
         robot_node and motor_node already running. Does NOT require
         real_stuck_detection_node (this test does not exercise recovery).
Outputs: Prints the exact commanded duration, then prompts for the measured
         distance (metres) and prints the resulting speed and (for a probe
         run) the suggested official duration for the next run. Appends one
         row to wheel_speed_calibration_results.csv (vel, duration_s,
         distance_m, speed_m_s, is_probe).
Safety:  Total publish time is hard-capped at duration + HARD_CAP_MARGIN_S
         regardless of anything else, matching the pattern in
         stuck_detection_drive_test.py (built after a 2026-07-25 incident
         where an uncapped raw publish loop drove the rover into a person
         and a wall). An explicit stop is published on every exit path.
         Course must be a flat, obstacle-free 3m straight line with a soft
         buffer (cushion/box) at the far end; operator stands with the
         battery accessible.
How to run:
    # on the Pi, inside the ROS2 container, with robot_node/motor_node
    # already running, rover at the start of a clear 3m straight course
    source /opt/ros/humble/setup.bash
    source /ws/install/setup.bash
    # Stage 1 -- probe:
    python3 wheel_speed_calibration_test.py 25 --duration 1.0 --probe
    # Stage 2 -- official run, using the duration the probe suggested:
    python3 wheel_speed_calibration_test.py 25 --duration <suggested>
Project: Onboard Visual Foundation Models for Mars Terrain Perception and Rover Navigation
"""
import argparse
import csv
import os
import sys
import time

HARD_CAP_MARGIN_S = 2.0
RESULTS_CSV = os.path.join(os.path.dirname(__file__), "results",
                            "wheel_speed_calibration_results.csv")


def compute_speed_m_s(distance_m: float, duration_s: float) -> float:
    if duration_s <= 0:
        raise ValueError("duration_s must be positive")
    return distance_m / duration_s


def compute_official_duration_s(
    probe_speed_m_s: float,
    target_distance_m: float = 2.2,
    max_duration_s: float = 6.0,
    min_duration_s: float = 0.5,
) -> float:
    if probe_speed_m_s < 0:
        raise ValueError("probe_speed_m_s cannot be negative")
    if probe_speed_m_s == 0:
        return max_duration_s
    duration = target_distance_m / probe_speed_m_s
    return max(min_duration_s, min(duration, max_duration_s))


def _append_csv_row(vel: float, duration_s: float, distance_m: float,
                     speed_m_s: float, is_probe: bool) -> None:
    os.makedirs(os.path.dirname(RESULTS_CSV), exist_ok=True)
    write_header = not os.path.exists(RESULTS_CSV)
    with open(RESULTS_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["vel", "duration_s", "distance_m", "speed_m_s", "is_probe"])
        writer.writerow([vel, duration_s, distance_m, speed_m_s, is_probe])


def _drive_straight_for(pub, node, vel: float, duration_s: float) -> None:
    import rclpy
    from exomy_ros2_msgs.msg import RoverCommand

    LOCOMOTION_FAKE_ACKERMANN = 0
    STEERING_STRAIGHT = 90.0
    hard_cap_s = duration_s + HARD_CAP_MARGIN_S

    def command(v: float, enabled: bool) -> None:
        msg = RoverCommand()
        msg.motors_enabled = enabled
        msg.locomotion_mode = LOCOMOTION_FAKE_ACKERMANN
        msg.vel = v
        msg.steering = STEERING_STRAIGHT
        pub.publish(msg)

    print(f"DRIVE at vel {vel:.0f} for {duration_s:.2f}s "
          f"(hard cap {hard_cap_s:.2f}s)", flush=True)
    t0 = time.time()
    try:
        while time.time() - t0 < duration_s:
            if time.time() - t0 >= hard_cap_s:
                print("Hit hard cap -- stopping early.", flush=True)
                break
            command(vel, True)
            rclpy.spin_once(node, timeout_sec=0.05)
    finally:
        for _ in range(10):
            command(0.0, False)
            rclpy.spin_once(node, timeout_sec=0.02)
        print("stop published 10x", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("vel", type=float)
    parser.add_argument("--duration", type=float, required=True)
    parser.add_argument("--probe", action="store_true",
                         help="tag this run as a probe in the CSV log")
    args = parser.parse_args()

    import rclpy
    from exomy_ros2_msgs.msg import RoverCommand

    rclpy.init()
    node = rclpy.create_node("wheel_speed_calibration_test")
    pub = node.create_publisher(RoverCommand, "/rover_command", 1)
    time.sleep(1.0)  # let pub connect before the first command

    _drive_straight_for(pub, node, args.vel, args.duration)

    node.destroy_node()
    rclpy.shutdown()

    distance_m = float(input("Measured distance travelled (m): "))
    speed_m_s = compute_speed_m_s(distance_m, args.duration)
    _append_csv_row(args.vel, args.duration, distance_m, speed_m_s, args.probe)
    print(f"speed = {speed_m_s:.3f} m/s", flush=True)

    if args.probe:
        suggested = compute_official_duration_s(speed_m_s)
        print(f"Suggested official duration for vel={args.vel:.0f}: "
              f"{suggested:.2f}s (capped at 6.0s, target 2.2m)", flush=True)


if __name__ == "__main__":
    main()
