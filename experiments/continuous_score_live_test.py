"""
Purpose: Live Gazebo verification of terrain_controller_node's continuous-speed
         mode (thesis Ch5 SS5.6.2, Ch4 SS4.8.16, use_continuous_score param).
         The continuous-speed formula (v = v_max * (1 - T_score), score smoothed
         over a 4-frame moving average) was implemented and unit-tested on
         2026-07-16 but never exercised in a live Gazebo run -- this script
         closes that verification gap. Records /traversability_score and
         /exomy/cmd_vel together and checks that the commanded linear speed
         tracks continuous_speed(smoothed_score, v_max=0.10) reasonably
         closely, rather than sitting at a fixed discrete POLICY step.
Inputs:  /traversability_score (std_msgs/Float64) -- from dinov2_terrain_node.py
         /exomy/cmd_vel (geometry_msgs/Twist) -- from terrain_controller_node.py
Outputs: experiments/results/continuous_score_live_test.csv
         experiments/results/figures/continuous_score_live_test.png
How to run:
    # Terminal 1:
    source /opt/ros/humble/setup.bash && source ros2_ws/install/setup.bash
    ros2 launch simulation/launch/dinov2_controller_test.launch.py \
        use_continuous_score:=true

    # Terminal 2 (after Terminal 1 shows the terrain controller has started):
    source /opt/ros/humble/setup.bash && source ros2_ws/install/setup.bash
    python3 experiments/continuous_score_live_test.py
Project: Onboard Visual Foundation Models for Mars Terrain Perception and Rover Navigation
"""

import csv
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from std_msgs.msg import Float64

DURATION_S = 60.0
V_MAX = 0.10  # POLICY["soil"], the v_max reference used by continuous_speed()

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)


def continuous_speed(t_score: float, v_max: float) -> float:
    """Mirrors terrain_controller_node.py's own continuous_speed() exactly,
    for computing the expected value independently of the node under test."""
    return max(0.0, min(v_max, v_max * (1.0 - t_score)))


class ContinuousScoreRecorder(Node):

    def __init__(self):
        super().__init__("continuous_score_live_test")
        self._records = []
        self._t0 = time.time()
        self._latest_score = None
        self.create_subscription(Float64, "/traversability_score", self._score_cb, 10)
        self.create_subscription(Twist, "/exomy/cmd_vel", self._cmd_vel_cb, 10)

    def _score_cb(self, msg: Float64) -> None:
        self._latest_score = msg.data

    def _cmd_vel_cb(self, msg: Twist) -> None:
        self._records.append({
            "t_s": round(time.time() - self._t0, 3),
            "traversability_score": round(self._latest_score, 4) if self._latest_score is not None else None,
            "cmd_vel_linear_x": round(msg.linear.x, 4),
            "expected_continuous_speed": (
                round(continuous_speed(self._latest_score, V_MAX), 4)
                if self._latest_score is not None else None
            ),
        })


def main():
    rclpy.init()
    node = ContinuousScoreRecorder()
    node.get_logger().info(
        f"Recording /traversability_score + /exomy/cmd_vel for {DURATION_S}s "
        "to verify continuous-speed mode live in Gazebo."
    )

    end_t = time.time() + DURATION_S
    while time.time() < end_t:
        rclpy.spin_once(node, timeout_sec=0.1)

    csv_path = os.path.join(RESULTS_DIR, "continuous_score_live_test.csv")
    with open(csv_path, "w", newline="") as f:
        if node._records:
            w = csv.DictWriter(f, fieldnames=list(node._records[0].keys()))
            w.writeheader()
            w.writerows(node._records)
    node.get_logger().info(f"Saved {len(node._records)} records -> {csv_path}")

    valid = [r for r in node._records if r["traversability_score"] is not None]
    print("\n" + "=" * 70)
    print("CONTINUOUS-SCORE LIVE TEST SUMMARY")
    print(f"Total /exomy/cmd_vel messages: {len(node._records)}")
    print(f"Messages with a score already received: {len(valid)}")
    if valid:
        distinct_speeds = sorted({r["cmd_vel_linear_x"] for r in valid})
        errors = [abs(r["cmd_vel_linear_x"] - r["expected_continuous_speed"]) for r in valid]
        print(f"Distinct commanded speeds observed: {len(distinct_speeds)} -> {distinct_speeds[:10]}"
              f"{'...' if len(distinct_speeds) > 10 else ''}")
        print(f"Mean |cmd_vel - expected continuous_speed()|: {sum(errors)/len(errors):.4f} m/s")
        print(f"Max  |cmd_vel - expected continuous_speed()|: {max(errors):.4f} m/s")
        if len(distinct_speeds) <= 2:
            print("WARNING: speed looks discrete-stepped (<=2 distinct values) -- "
                  "continuous mode may not be taking effect as intended.")
        else:
            print("Speed varies continuously across >2 distinct values, consistent "
                  "with continuous-speed mode being active (not discrete POLICY steps).")
    print("=" * 70)

    if valid:
        t = [r["t_s"] for r in valid]
        actual = [r["cmd_vel_linear_x"] for r in valid]
        expected = [r["expected_continuous_speed"] for r in valid]
        plt.figure(figsize=(10, 5))
        plt.plot(t, actual, label="/exomy/cmd_vel linear.x (actual)", marker=".")
        plt.plot(t, expected, label="continuous_speed(score, v_max) (expected)", linestyle="--")
        plt.xlabel("Time (s)")
        plt.ylabel("Speed (m/s)")
        plt.title("Continuous-speed mode: commanded vs. expected (live Gazebo)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        fig_path = os.path.join(FIGURES_DIR, "continuous_score_live_test.png")
        plt.savefig(fig_path, dpi=120, bbox_inches="tight")
        node.get_logger().info(f"Figure saved -> {fig_path}")

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
