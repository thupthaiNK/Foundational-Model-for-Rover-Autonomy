"""
Purpose: X4 follow-up -- repeats the exact X4 sequential-drive protocol
         (Ch4 SS4.8.9: spawn at (-3.0, 6.0) inside the soil zone, facing +x
         toward the soil/bedrock boundary at x=0, record for 90s) but with
         terrain_controller_node's use_continuous_score mode enabled instead
         of the original discrete policy. Tests the specific claim the
         continuous-speed mode was proposed to address (Ch5 SS5.6.2): "this
         would reduce unnecessary stops at zone boundaries." X4's original
         result already establishes that the STOP at t=69s is triggered by
         confidence falling below the deployment threshold (T=0.40), which
         forces T_score=1.0 in continuous mode too (dinov2_terrain_node.py's
         traversability_score() forces score=1.0 below threshold, unchanged
         by which controller mode consumes it) -- so continuous mode is NOT
         expected to prevent that final stop. What it can plausibly change
         is the speed PROFILE above the threshold: X4's Phase 1 (t=0-60s)
         shows a flat 0.05 m/s CAUTION speed throughout despite confidence
         varying 0.76-0.85, and the confidence decay from 0.85->0.41
         (t=50-65s) also produced no discrete-policy speed change until the
         label itself flipped. This script measures whether continuous mode
         shows graded speed change in that same region instead.
Inputs:  /terrain_classification (std_msgs/String), /traversability_score
         (std_msgs/Float64), /exomy/cmd_vel (geometry_msgs/Twist)
Outputs: experiments/results/sequential_drive_continuous.csv
         experiments/results/figures/sequential_drive_continuous.png
How to run:
    # Terminal 1:
    source /opt/ros/humble/setup.bash && source ros2_ws/install/setup.bash
    ros2 launch simulation/launch/dinov2_controller_test.launch.py \
        spawn_x:=-3.0 spawn_y:=6.0 use_continuous_score:=true

    # Terminal 2 (after terrain_controller_node has started):
    source /opt/ros/humble/setup.bash && source ros2_ws/install/setup.bash
    python3 experiments/sequential_drive_continuous_experiment.py
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
from std_msgs.msg import Float64, String

RECORD_SECONDS = 90.0  # matches X4's original recording duration exactly
SPAWN_X = -3.0         # matches X4 -- for the CSV's x_est column only, not used to spawn here

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)


class SequentialDriveContinuousRecorder(Node):

    def __init__(self):
        super().__init__("sequential_drive_continuous_recorder")
        self._records = []
        self._terrain = "uncertain"
        self._conf = 0.0
        self._score = None
        self._speed = 0.0
        self._t0 = None

        self.create_subscription(String, "/terrain_classification", self._terrain_cb, 10)
        self.create_subscription(Float64, "/traversability_score", self._score_cb, 10)
        self.create_subscription(Twist, "/exomy/cmd_vel", self._vel_cb, 10)

    def _terrain_cb(self, msg: String) -> None:
        parts = msg.data.split(":")
        self._terrain = parts[0].strip().lower() if parts else "uncertain"
        try:
            self._conf = float(parts[1]) if len(parts) > 1 else 0.0
        except ValueError:
            self._conf = 0.0

    def _score_cb(self, msg: Float64) -> None:
        self._score = msg.data

    def _vel_cb(self, msg: Twist) -> None:
        self._speed = msg.linear.x

    def record_tick(self) -> None:
        if self._t0 is None:
            self._t0 = time.time()
        elapsed = time.time() - self._t0
        self._records.append({
            "elapsed_s": round(elapsed, 2),
            "terrain": self._terrain,
            "confidence": round(self._conf, 3),
            "traversability_score": round(self._score, 4) if self._score is not None else None,
            "speed_ms": round(self._speed, 4),
            "x_est": round(SPAWN_X + self._integrated_x(), 2),
        })

    def _integrated_x(self) -> float:
        total = 0.0
        for i in range(1, len(self._records)):
            dt = self._records[i]["elapsed_s"] - self._records[i - 1]["elapsed_s"]
            total += self._records[i - 1]["speed_ms"] * dt
        return total

    def save_csv(self, path: str) -> None:
        if not self._records:
            return
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(self._records[0].keys()))
            w.writeheader()
            w.writerows(self._records)
        self.get_logger().info(f"Saved {len(self._records)} records -> {path}")


def main():
    rclpy.init()
    node = SequentialDriveContinuousRecorder()
    node.get_logger().info(
        f"X4 continuous-mode follow-up starting -- recording for {RECORD_SECONDS}s. "
        "Assumes rover already spawned at (-3.0, 6.0) via dinov2_controller_test.launch.py "
        "spawn_x:=-3.0 spawn_y:=6.0 use_continuous_score:=true"
    )

    # Wait for the first real classification before starting the clock -- DINOv2
    # model load + first-frame inference latency otherwise silently eats into the
    # fixed recording window (observed: ~40s dead time in an earlier run here,
    # confirmed via node._terrain/_conf still at their "uncertain"/0.0 defaults).
    node.get_logger().info("Waiting for first real /traversability_score before starting clock...")
    wait_deadline = time.time() + 60.0
    while node._score is None and time.time() < wait_deadline:
        rclpy.spin_once(node, timeout_sec=0.2)
    if node._score is None:
        node.get_logger().error("Never received /traversability_score within 60s -- aborting.")
        node.destroy_node()
        rclpy.shutdown()
        return
    node.get_logger().info(f"First score received ({node._score:.3f}) -- starting {RECORD_SECONDS}s clock now.")

    end_t = time.time() + RECORD_SECONDS
    while time.time() < end_t:
        node.record_tick()
        rclpy.spin_once(node, timeout_sec=0.1)

    csv_path = os.path.join(RESULTS_DIR, "sequential_drive_continuous.csv")
    node.save_csv(csv_path)

    records = node._records
    terrains = [r["terrain"] for r in records]
    unique_terrains = list(dict.fromkeys(terrains))

    speeds_pre_stop = [r["speed_ms"] for r in records if r["speed_ms"] > 0.0]
    distinct_speeds = sorted({round(s, 3) for s in speeds_pre_stop})
    stop_records = [r for r in records if r["speed_ms"] == 0.0]
    first_stop = stop_records[0]["elapsed_s"] if stop_records else None

    print("\n" + "=" * 70)
    print("X4 CONTINUOUS-MODE FOLLOW-UP SUMMARY")
    print(f"Total records: {len(records)}")
    print(f"Terrain sequence: {' -> '.join(unique_terrains)}")
    print(f"Distinct nonzero commanded speeds: {len(distinct_speeds)} -> {distinct_speeds[:10]}"
          f"{'...' if len(distinct_speeds) > 10 else ''}")
    print(f"First STOP (speed==0.0) at t={first_stop}s" if first_stop is not None
          else "No STOP observed during recording.")
    print("Compare against X4's original discrete-mode result (Ch4 SS4.8.9, Table 4.8.9.1): "
          "flat 0.05 m/s CAUTION for t=0-60s (1 distinct nonzero speed only), "
          "STOP at t=69s.")
    print("=" * 70)

    if records:
        t = [r["elapsed_s"] for r in records]
        speed = [r["speed_ms"] for r in records]
        conf = [r["confidence"] for r in records]
        fig, ax1 = plt.subplots(figsize=(10, 5))
        ax1.plot(t, speed, label="cmd_vel linear.x", color="tab:blue")
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Speed (m/s)", color="tab:blue")
        ax2 = ax1.twinx()
        ax2.plot(t, conf, label="confidence", color="tab:orange", alpha=0.6)
        ax2.set_ylabel("Confidence", color="tab:orange")
        plt.title("X4 continuous-mode follow-up: speed vs confidence over time")
        fig_path = os.path.join(FIGURES_DIR, "sequential_drive_continuous.png")
        plt.savefig(fig_path, dpi=120, bbox_inches="tight")
        node.get_logger().info(f"Figure saved -> {fig_path}")

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
