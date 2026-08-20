"""
Purpose: DINOv2 lighting robustness experiment — runs the standard 5-zone
         traversability test under dawn, noon, or dusk lighting conditions
         and compares accuracy against the baseline (mars_terrain.world).
         Tests whether DINOv2 frozen features trained on AI4Mars are
         sensitive to simulated sun angle changes in Gazebo.
         Output from the three variants feeds Table §4.9 / §5.X in the thesis:
         "Lighting robustness: terrain accuracy across sun angles."
Inputs:  --variant dawn|noon|dusk  (must match the running Gazebo world)
         Gazebo with dinov2_lighting_test.launch.py running:
           ros2 launch simulation/launch/dinov2_lighting_test.launch.py variant:=<VARIANT>
         /terrain_classification (std_msgs/String) "label:confidence"
         /terrain_class_probs    (std_msgs/Float32MultiArray)
         /inference_latency_ms   (std_msgs/Float64)
         /exomy/cmd_vel          (geometry_msgs/Twist)
         /measurement_mode       (std_msgs/Bool)
Outputs: experiments/results/gazebo_traversability_<variant>.csv
         (e.g. gazebo_traversability_dawn.csv, _noon.csv, _dusk.csv)
How to run:
    # Terminal 2 — launch Gazebo with chosen variant:
    source /opt/ros/humble/setup.bash && source ros2_ws/install/setup.bash
    ros2 launch simulation/launch/dinov2_lighting_test.launch.py variant:=dawn

    # Terminal 3 — run matching experiment:
    source /opt/ros/humble/setup.bash && source ros2_ws/install/setup.bash
    python3 experiments/lighting_traversability_experiment.py --variant dawn

    # Repeat for noon and dusk.

Project: Onboard Visual Foundation Models for Mars Terrain Perception and Rover Navigation
"""

import argparse
import csv
import os
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy
from gazebo_msgs.srv import DeleteEntity, SpawnEntity
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool, Float32MultiArray, Float64, String

# ── Zone definitions — identical to dinov2_traversability_experiment.py ───────
ZONES = [
    {
        "name":           "soil_zone",
        "x": -7.5, "y": 6.0,
        "ground_truth":   "soil",
        "expected_policy": "SAFE",
        "expected_speed_ms": 0.10,
        "description":    "Soil zone Q2",
    },
    {
        "name":           "bedrock_zone",
        "x": 7.5, "y": 6.0,
        "ground_truth":   "bedrock",
        "expected_policy": "HAZARD",
        "expected_speed_ms": 0.03,
        "description":    "Bedrock zone Q1",
    },
    {
        "name":           "sand_zone",
        "x": -7.5, "y": -6.0,
        "ground_truth":   "sand",
        "expected_policy": "CAUTION",
        "expected_speed_ms": 0.05,
        "description":    "Sand zone Q3",
    },
    {
        "name":           "rock_cluster",
        "x": 2.0, "y": -4.0,
        "ground_truth":   "big_rock",
        "expected_policy": "STOP",
        "expected_speed_ms": 0.00,
        "description":    "Rock cluster Q4 centre",
    },
    {
        "name":           "boulder_zone",
        "x": 2.0, "y": -8.0,
        "ground_truth":   "big_rock",
        "expected_policy": "STOP",
        "expected_speed_ms": 0.00,
        "description":    "Boulder zone Q4 south",
    },
]

# Timing — same as baseline experiment
WAIT_STABLE_S  = 3
N_READINGS     = 8
SPIN_TIMEOUT   = 0.5
MAX_WAIT_S     = 30
SPEED_WAIT_S   = 5.0

OUT_DIR = os.path.join(os.path.dirname(__file__), "results")


class LightingExperiment(Node):
    """5-zone DINOv2 traversability experiment under a specific lighting variant."""

    def __init__(self):
        super().__init__("lighting_experiment_node")

        self._terrain_label: str   = ""
        self._terrain_conf: float  = 0.0
        self._terrain_count: int   = 0
        self._latency_ms: float    = 0.0
        self._cmd_vel_x: float     = 0.0
        self._urdf_xml: str | None = None

        self.create_subscription(String,            "/terrain_classification", self._terrain_cb, 10)
        self.create_subscription(Float64,           "/inference_latency_ms",   self._latency_cb, 10)
        self.create_subscription(Twist,             "/exomy/cmd_vel",          self._vel_cb,     10)
        qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(String, "/robot_description", self._urdf_cb, qos)

        self._mode_pub = self.create_publisher(Bool, "/measurement_mode", 10)

        self._delete_cli = self.create_client(DeleteEntity, "/delete_entity")
        self._spawn_cli  = self.create_client(SpawnEntity,  "/spawn_entity")

    def _terrain_cb(self, msg: String):
        parts = msg.data.split(":")
        if len(parts) >= 2:
            self._terrain_label = parts[0].strip().lower()
            try:
                self._terrain_conf = float(parts[1])
            except ValueError:
                self._terrain_conf = 0.0
        self._terrain_count += 1

    def _latency_cb(self, msg: Float64):
        self._latency_ms = msg.data

    def _vel_cb(self, msg: Twist):
        self._cmd_vel_x = msg.linear.x

    def _urdf_cb(self, msg: String):
        self._urdf_xml = msg.data

    def _set_freeze(self, freeze: bool):
        msg = Bool()
        msg.data = freeze
        for _ in range(3):
            self._mode_pub.publish(msg)
            rclpy.spin_once(self, timeout_sec=0.05)

    def _teleport(self, x: float, y: float):
        self._set_freeze(True)
        if self._urdf_xml is None:
            self.get_logger().info("Waiting for /robot_description...")
            deadline = time.time() + 8.0
            while self._urdf_xml is None and time.time() < deadline:
                rclpy.spin_once(self, timeout_sec=0.2)
        if self._urdf_xml is None:
            self.get_logger().error("robot_description unavailable; cannot respawn full rover")
            return

        if not self._delete_cli.wait_for_service(timeout_sec=10.0):
            self.get_logger().error("delete_entity not available")
            return

        self.get_logger().info(f"Teleport: deleting exomy before respawn to ({x:.1f}, {y:.1f})")
        del_req = DeleteEntity.Request()
        del_req.name = "exomy"
        deleted = False
        for attempt in range(2):
            future = self._delete_cli.call_async(del_req)
            rclpy.spin_until_future_complete(self, future, timeout_sec=15.0)
            if future.done() and future.result() and future.result().success:
                deleted = True
                break
            self.get_logger().warn(
                f"Delete attempt {attempt + 1} failed"
                + (" — retrying in 2s" if attempt == 0 else " — continuing")
            )
            if attempt == 0:
                time.sleep(2.0)
        if deleted:
            self.get_logger().info("Teleport: delete_entity succeeded")
        time.sleep(1.5)

        if not self._spawn_cli.wait_for_service(timeout_sec=10.0):
            self.get_logger().error("spawn_entity not available")
            return

        self.get_logger().info("Teleport: respawning full rover from /robot_description")
        sp_req = SpawnEntity.Request()
        sp_req.name = "exomy"
        sp_req.xml  = self._urdf_xml
        sp_req.initial_pose.position.x = x
        sp_req.initial_pose.position.y = y
        sp_req.initial_pose.position.z = 0.15
        sp_req.initial_pose.orientation.w = 1.0
        future = self._spawn_cli.call_async(sp_req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=15.0)
        if future.done() and future.result() and future.result().success:
            self.get_logger().info("Teleport: spawn_entity succeeded")
        else:
            self.get_logger().error("Teleport: spawn_entity failed or timed out")
        time.sleep(1.5)

    def _collect_readings(self, n: int) -> tuple[list[str], list[float], list[float]]:
        """Collect n terrain readings during freeze window."""
        labels, confs, latencies = [], [], []
        last_count = self._terrain_count
        t_end = time.time() + MAX_WAIT_S
        while len(labels) < n and time.time() < t_end:
            rclpy.spin_once(self, timeout_sec=SPIN_TIMEOUT)
            if self._terrain_count > last_count:
                labels.append(self._terrain_label)
                confs.append(self._terrain_conf)
                latencies.append(self._latency_ms)
                last_count = self._terrain_count
        return labels, confs, latencies

    def _measure_speed(self) -> float:
        self._set_freeze(False)
        samples = []
        t_end = time.time() + SPEED_WAIT_S
        while time.time() < t_end:
            rclpy.spin_once(self, timeout_sec=0.2)
            samples.append(self._cmd_vel_x)
        self._set_freeze(True)
        return round(max((abs(s) for s in samples), default=0.0), 4)

    def run_experiment(self) -> list[dict]:
        results = []
        for zone in ZONES:
            name = zone["name"]
            gt   = zone["ground_truth"]

            self.get_logger().info(f"\n{'='*55}")
            self.get_logger().info(f"Zone: {name}  GT={gt}")
            self._teleport(zone["x"], zone["y"])

            # Stable wait
            self.get_logger().info(f"Waiting {WAIT_STABLE_S}s for camera to stabilise...")
            t_end = time.time() + WAIT_STABLE_S
            while time.time() < t_end:
                rclpy.spin_once(self, timeout_sec=0.3)

            self.get_logger().info(
                f"Collecting {N_READINGS} terrain readings (timeout {MAX_WAIT_S}s)..."
            )
            labels, confs, latencies = self._collect_readings(N_READINGS)
            self.get_logger().info(
                f"Collected {len(labels)}/{N_READINGS} readings; measuring controller speed..."
            )
            actual_speed = self._measure_speed()

            from collections import Counter
            if not labels:
                pred       = "timeout"
                pred_conf  = 0.0
                consistent = False
            else:
                cnt        = Counter(labels)
                pred       = cnt.most_common(1)[0][0]
                pred_conf  = round(
                    sum(c for l, c in zip(labels, confs) if l == pred)
                    / max(1, cnt[pred]), 3
                )
                consistent = (len(set(labels)) == 1)

            mean_latency  = round(sum(latencies) / len(latencies), 1) if latencies else 0.0
            terrain_correct = (pred == gt)
            exp_spd         = zone["expected_speed_ms"]
            speed_correct   = abs(actual_speed - exp_spd) <= 0.02

            self.get_logger().info(
                f"pred={pred!r} conf={pred_conf:.2f} → terrain_correct={terrain_correct}  "
                f"speed={actual_speed:.3f} → speed_correct={speed_correct}  "
                f"latency={mean_latency:.0f}ms"
            )

            results.append({
                "zone":              name,
                "ground_truth":      gt,
                "pred":              pred,
                "confidence":        pred_conf,
                "terrain_correct":   terrain_correct,
                "consistent":        consistent,
                "n_readings":        len(labels),
                "all_labels":        ",".join(labels),
                "expected_policy":   zone["expected_policy"],
                "expected_speed_ms": exp_spd,
                "actual_speed_ms":   actual_speed,
                "speed_correct":     speed_correct,
                "mean_latency_ms":   mean_latency,
                "description":       zone["description"],
            })

        return results


def _summarise(results: list[dict], variant: str):
    terrain_ok = sum(1 for r in results if r["terrain_correct"] is True)
    speed_ok   = sum(1 for r in results if r["speed_correct"]   is True)
    n          = len(results)

    print(f"\n{'='*65}")
    print(f"LIGHTING ROBUSTNESS — {variant.upper()} — DINOv2 5-ZONE RESULTS")
    print(f"{'='*65}")
    print(
        f"{'Zone':<20} {'GT':<10} {'Pred':<10} {'Conf':<7} "
        f"{'T?':<5} {'Speed':<7} {'S?'}"
    )
    print("-" * 65)
    for r in results:
        t = "YES" if r["terrain_correct"] is True else "NO "
        s = "YES" if r["speed_correct"]   is True else "NO "
        print(
            f"{r['zone']:<20} {r['ground_truth']:<10} {r['pred']:<10} "
            f"{r['confidence']:<7.2f} {t:<5} {r['actual_speed_ms']:<7.3f} {s}"
        )
    print("-" * 65)
    print(f"Terrain correct: {terrain_ok}/{n}  |  Speed correct: {speed_ok}/{n}")
    print("=" * 65)

    # Comparison baseline (authoritative 2026-06-24)
    print("\nComparison (thesis §4.9 Lighting Robustness Table):")
    print(f"  Baseline (afternoon): 3/5 terrain, 5/5 speed")
    print(f"  {variant.capitalize()}: {terrain_ok}/5 terrain, {speed_ok}/5 speed")


def save_csv(results: list[dict], path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = [
        "zone", "ground_truth", "pred", "confidence", "terrain_correct",
        "consistent", "n_readings", "all_labels", "expected_policy",
        "expected_speed_ms", "actual_speed_ms", "speed_correct",
        "mean_latency_ms", "description",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"Results saved → {path}")


def main():
    parser = argparse.ArgumentParser(description="DINOv2 lighting robustness experiment")
    parser.add_argument(
        "--variant", choices=["dawn", "noon", "dusk"], default="noon",
        help="Lighting variant (must match the running Gazebo world)"
    )
    args = parser.parse_args()
    variant = args.variant

    out_csv = os.path.join(OUT_DIR, f"gazebo_traversability_{variant}.csv")

    print(f"\n{'='*55}")
    print(f"DINOv2 Lighting Robustness Experiment — {variant.upper()}")
    print(f"{'='*55}")
    print(f"Output CSV: {out_csv}")
    print(f"Ensure Gazebo is running with variant:={variant}\n")

    rclpy.init()
    node = LightingExperiment()

    # Warm-up
    t_end = time.time() + 5.0
    while time.time() < t_end:
        rclpy.spin_once(node, timeout_sec=0.5)

    results = node.run_experiment()
    _summarise(results, variant)
    save_csv(results, out_csv)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
