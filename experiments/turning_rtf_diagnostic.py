"""
Purpose: Diagnoses the confirmed (3x independent, §4.8.23) turning-
         unreliability problem by directly measuring Gazebo's real-time
         factor (RTF = simulated time elapsed / real wall-clock time
         elapsed) during a FORWARD-driving phase and a TURNING phase,
         commanded back to back under otherwise identical conditions.
         Tests hypothesis (a) -- Gazebo RTF throttling under load -- by
         checking whether RTF is measurably lower during the turn than
         during forward driving. If RTF is similar in both phases but
         actual yaw change per unit of SIMULATED time is still far below
         the commanded rate, that points away from (a) and toward
         hypothesis (b): a genuine skid-steer torque/friction limitation
         in the URDF's diff_drive configuration. Deliberately does not
         modify simulation/urdf/exomy.urdf.xacro's physics parameters
         (mu1/mu2, max_wheel_torque, max_wheel_acceleration) -- those are
         shared by every already-validated Gazebo result in this thesis
         and must not be changed for a diagnostic test.
Inputs:  /clock (rosgraph_msgs/Clock) -- Gazebo's simulated time
         /exomy/odom (nav_msgs/Odometry) -- ground truth position/yaw
Outputs: experiments/results/turning_rtf_diagnostic.csv
         experiments/results/figures/turning_rtf_diagnostic.png
How to run:
    # Terminal 1:
    source /opt/ros/humble/setup.bash && source ros2_ws/install/setup.bash
    ros2 launch simulation/launch/turning_diagnostic.launch.py

    # Terminal 2 (after Terminal 1 shows "Spawn status: ... Successfully spawned"):
    source /opt/ros/humble/setup.bash && source ros2_ws/install/setup.bash
    python3 experiments/turning_rtf_diagnostic.py
Project: Onboard Visual Foundation Models for Mars Terrain Perception and Rover Navigation
"""

import csv
import math
import os
import sys
import time

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rosgraph_msgs.msg import Clock

LINEAR_SPEED  = 0.10
ANGULAR_SPEED = 0.30   # matches the value already used throughout this codebase
                        # (reactive_explorer_node, stuck_detection_node) for turns
PHASE_DURATION_S = 20.0  # real wall-clock seconds budgeted per phase

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)


def yaw_from_quaternion(q) -> float:
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


class RtfDiagnostic(Node):

    def __init__(self):
        super().__init__("turning_rtf_diagnostic")
        self._records = []
        self._real_t0 = time.time()
        self._sim_time_s = None   # from /clock
        self._gt_pos = None       # (x, y, yaw)
        self._phase = "idle"

        # Gazebo's /clock publisher uses BEST_EFFORT reliability; the rclpy
        # default subscription QoS is RELIABLE, which is incompatible and
        # silently receives nothing (confirmed via the QoS-mismatch warning
        # on a first attempt, not assumed).
        clock_qos = QoSProfile(depth=10)
        clock_qos.reliability = ReliabilityPolicy.BEST_EFFORT
        self.create_subscription(Clock, "/clock", self._clock_cb, clock_qos)
        self.create_subscription(Odometry, "/exomy/odom", self._odom_cb, 10)
        self.pub_cmd = self.create_publisher(Twist, "/exomy/cmd_vel", 10)
        self.create_timer(0.2, self._record_tick)

    def _clock_cb(self, msg: Clock) -> None:
        self._sim_time_s = msg.clock.sec + msg.clock.nanosec / 1e9

    def _odom_cb(self, msg: Odometry) -> None:
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        self._gt_pos = (p.x, p.y, yaw_from_quaternion(q))

    def _record_tick(self) -> None:
        real_elapsed = time.time() - self._real_t0
        self._records.append({
            "real_elapsed_s": round(real_elapsed, 3),
            "sim_time_s": round(self._sim_time_s, 3) if self._sim_time_s is not None else None,
            "phase": self._phase,
            "gt_x": round(self._gt_pos[0], 4) if self._gt_pos else None,
            "gt_y": round(self._gt_pos[1], 4) if self._gt_pos else None,
            "gt_yaw_deg": round(math.degrees(self._gt_pos[2]), 2) if self._gt_pos else None,
        })

    def send_twist(self, linear_x: float, angular_z: float) -> None:
        msg = Twist()
        msg.linear.x = linear_x
        msg.angular.z = angular_z
        self.pub_cmd.publish(msg)

    def run_phase(self, name: str, linear_x: float, angular_z: float, duration_s: float) -> None:
        self._phase = name
        self.get_logger().info(f"Phase '{name}' starting: linear={linear_x} angular={angular_z}")
        end_t = time.time() + duration_s
        while time.time() < end_t:
            self.send_twist(linear_x, angular_z)
            rclpy.spin_once(self, timeout_sec=0.1)
        self.send_twist(0.0, 0.0)
        self._phase = "idle"

    def save_csv(self, path: str) -> None:
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(self._records[0].keys()))
            w.writeheader()
            w.writerows(self._records)
        self.get_logger().info(f"Saved {len(self._records)} records -> {path}")


def summarize_phase(records, phase_name):
    phase_records = [r for r in records if r["phase"] == phase_name and r["sim_time_s"] is not None]
    if len(phase_records) < 2:
        return None
    real_span = phase_records[-1]["real_elapsed_s"] - phase_records[0]["real_elapsed_s"]
    sim_span = phase_records[-1]["sim_time_s"] - phase_records[0]["sim_time_s"]
    rtf = sim_span / real_span if real_span > 0 else 0.0

    first_gt = next((r for r in phase_records if r["gt_x"] is not None), None)
    last_gt = next((r for r in reversed(phase_records) if r["gt_x"] is not None), None)
    displacement_m = None
    yaw_change_deg = None
    if first_gt and last_gt:
        displacement_m = math.hypot(last_gt["gt_x"] - first_gt["gt_x"], last_gt["gt_y"] - first_gt["gt_y"])
        yaw_change_deg = last_gt["gt_yaw_deg"] - first_gt["gt_yaw_deg"]

    return {
        "phase": phase_name,
        "real_span_s": round(real_span, 2),
        "sim_span_s": round(sim_span, 2),
        "rtf": round(rtf, 3),
        "displacement_m": round(displacement_m, 3) if displacement_m is not None else None,
        "yaw_change_deg": round(yaw_change_deg, 1) if yaw_change_deg is not None else None,
    }


def main():
    rclpy.init()
    node = RtfDiagnostic()
    node.get_logger().info(
        "Turning RTF diagnostic starting -- forward phase, then turn phase, "
        "measuring Gazebo real-time-factor (/clock vs wall time) in each."
    )

    # Wait for /clock and /exomy/odom to arrive before starting either phase.
    deadline = time.time() + 15.0
    while (node._sim_time_s is None or node._gt_pos is None) and time.time() < deadline:
        rclpy.spin_once(node, timeout_sec=0.2)
    if node._sim_time_s is None:
        node.get_logger().error("Never received /clock -- is use_sim_time wired correctly? Aborting.")
        node.destroy_node()
        rclpy.shutdown()
        return

    node.run_phase("forward", LINEAR_SPEED, 0.0, PHASE_DURATION_S)
    time.sleep(2.0)  # let physics settle between phases (real time, deliberately not spun)
    rclpy.spin_once(node, timeout_sec=2.0)
    node.run_phase("turn", 0.0, ANGULAR_SPEED, PHASE_DURATION_S)

    # Optional tag (sys.argv[1]) picks a distinct output filename so a new
    # diagnostic variant never silently overwrites a previously archived run
    # (this exact mistake happened once before, see project memory).
    tag = sys.argv[1] if len(sys.argv) > 1 else None
    csv_name = f"turning_rtf_diagnostic_{tag}.csv" if tag else "turning_rtf_diagnostic.csv"
    csv_path = os.path.join(RESULTS_DIR, csv_name)
    node.save_csv(csv_path)

    forward_summary = summarize_phase(node._records, "forward")
    turn_summary = summarize_phase(node._records, "turn")

    print("\n" + "=" * 70)
    print("TURNING RTF DIAGNOSTIC SUMMARY")
    for s in (forward_summary, turn_summary):
        if s is None:
            continue
        expected_forward = LINEAR_SPEED * s["sim_span_s"] if s["phase"] == "forward" else None
        expected_turn_deg = math.degrees(ANGULAR_SPEED * s["sim_span_s"]) if s["phase"] == "turn" else None
        print(f"\nPhase: {s['phase']}")
        print(f"  Real wall-clock elapsed: {s['real_span_s']}s")
        print(f"  Gazebo simulated time elapsed: {s['sim_span_s']}s")
        print(f"  Real-time factor (RTF): {s['rtf']}")
        if s["phase"] == "forward":
            print(f"  Ground-truth displacement: {s['displacement_m']}m "
                  f"(expected at commanded speed x sim_time: {expected_forward:.3f}m)")
        else:
            print(f"  Ground-truth yaw change: {s['yaw_change_deg']} deg "
                  f"(expected at commanded rate x sim_time: {expected_turn_deg:.1f} deg)")
    print("=" * 70)

    if forward_summary and turn_summary:
        rtf_ratio = turn_summary["rtf"] / forward_summary["rtf"] if forward_summary["rtf"] > 0 else float("nan")
        print(f"\nRTF ratio (turn / forward): {rtf_ratio:.3f}")
        print("  ~1.0 => RTF is similar in both phases -> turning problem is NOT primarily RTF-driven,")
        print("          points toward a torque/friction limitation instead (hypothesis b).")
        print("  << 1.0 => RTF drops specifically during turning -> supports RTF throttling")
        print("          as at least a major contributor (hypothesis a).")

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
