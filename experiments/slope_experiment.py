"""
Purpose: Step 7b — Slope zone qualitative experiment.
         Tests whether DINOv2 terrain classification is texture-based or
         geometry/pose-aware by placing the rover on a 15° inclined soil surface
         (slope_zone in mars_terrain.world) and recording the terrain prediction.
         Expected result: DINOv2 classifies the slope as 'soil' regardless of
         inclination — confirming the model is texture-only and cannot detect slopes.
         This is a documented thesis finding (§5.X): slope recognition requires
         geometric sensing (IMU, depth) and cannot be inferred from RGB texture alone.
Inputs:  Gazebo with dinov2_controller_test.launch.py running
         /terrain_classification (std_msgs/String) "label:confidence"
         /inference_latency_ms   (std_msgs/Float64)
         /exomy/cmd_vel          (geometry_msgs/Twist)
         /measurement_mode       (std_msgs/Bool)
Outputs: experiments/results/gazebo_traversability_slope.csv
How to run:
    # Terminal 2 — launch Gazebo with DINOv2 (uses mars_terrain.world which has slope_zone):
    source /opt/ros/humble/setup.bash && source ros2_ws/install/setup.bash
    ros2 launch simulation/launch/dinov2_controller_test.launch.py

    # Terminal 3 — run slope experiment after DINOv2 node is ready:
    source /opt/ros/humble/setup.bash && source ros2_ws/install/setup.bash
    python3 experiments/slope_experiment.py

Project: Onboard Visual Foundation Models for Mars Terrain Perception and Rover Navigation
"""

import csv
import os
import time
from collections import Counter

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy
from gazebo_msgs.srv import DeleteEntity, SpawnEntity
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool, Float32MultiArray, Float64, String

# ── Slope zone definition ─────────────────────────────────────────────────────
# Slope is at (0, -20, ~1.55m) in mars_terrain.world.
# Rover spawns at z=1.70 (mid-slope surface + 0.15m clearance) and settles
# onto the 15° inclined surface under gravity.
# Ground_truth = 'soil' (same texture as Q2 soil zone, different geometry).
SLOPE_ZONE = {
    "name":            "slope_zone",
    "x": 0.0, "y": -20.0, "z": 1.70,   # spawn ON the slope surface (z>0)
    "ground_truth":    "soil",
    "expected_policy": "SAFE",
    "expected_speed_ms": 0.10,
    "description":     "15° inclined soil surface — DINOv2 texture-vs-geometry test",
}

# Multiple spawn positions on the slope (to vary camera angle + texture sample)
SLOPE_POSITIONS = [
    {"label": "mid_slope",   "x": 0.0,  "y": -20.0, "z": 1.70},
    {"label": "lower_slope", "x": 0.0,  "y": -16.0, "z": 0.60},
    {"label": "upper_slope", "x": 0.0,  "y": -22.5, "z": 2.20},
]

WAIT_STABLE_S = 3
N_READINGS    = 8
SPIN_TIMEOUT  = 0.5
MAX_WAIT_S    = 30
SPEED_WAIT_S  = 5.0

OUT_DIR = os.path.join(os.path.dirname(__file__), "results")
OUT_CSV = os.path.join(OUT_DIR, "gazebo_traversability_slope.csv")


class SlopeExperiment(Node):
    """Qualitative slope experiment: records DINOv2 output at three slope positions."""

    def __init__(self):
        super().__init__("slope_experiment_node")

        self._terrain_label: str   = ""
        self._terrain_conf: float  = 0.0
        self._terrain_count: int   = 0
        self._latency_ms: float    = 0.0
        self._cmd_vel_x: float     = 0.0
        self._urdf_xml: str | None = None

        self.create_subscription(String,           "/terrain_classification", self._terrain_cb, 10)
        self.create_subscription(Float64,          "/inference_latency_ms",   self._latency_cb, 10)
        self.create_subscription(Twist,            "/exomy/cmd_vel",          self._vel_cb,     10)
        qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(String, "/robot_description", self._urdf_cb, qos)

        self._mode_pub   = self.create_publisher(Bool, "/measurement_mode", 10)
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

    def _teleport(self, x: float, y: float, z: float = 0.15):
        self._set_freeze(True)
        if self._urdf_xml is None:
            self.get_logger().info("Waiting for /robot_description...")
            deadline = time.time() + 8.0
            while self._urdf_xml is None and time.time() < deadline:
                rclpy.spin_once(self, timeout_sec=0.2)
        if self._urdf_xml is None:
            self.get_logger().error("robot_description unavailable")
            return

        if not self._delete_cli.wait_for_service(timeout_sec=10.0):
            self.get_logger().error("delete_entity not available")
            return

        self.get_logger().info(f"Teleport: deleting exomy → respawn to ({x:.1f}, {y:.1f}, {z:.2f})")
        del_req = DeleteEntity.Request()
        del_req.name = "exomy"
        deleted = False
        for attempt in range(2):
            future = self._delete_cli.call_async(del_req)
            rclpy.spin_until_future_complete(self, future, timeout_sec=15.0)
            if future.done() and future.result() and future.result().success:
                deleted = True
                break
            self.get_logger().warn(f"Delete attempt {attempt + 1} failed"
                                   + (" — retrying" if attempt == 0 else ""))
            if attempt == 0:
                time.sleep(2.0)
        if deleted:
            self.get_logger().info("Teleport: delete_entity succeeded")
        time.sleep(1.5)

        if not self._spawn_cli.wait_for_service(timeout_sec=10.0):
            self.get_logger().error("spawn_entity not available")
            return

        sp_req = SpawnEntity.Request()
        sp_req.name = "exomy"
        sp_req.xml  = self._urdf_xml
        sp_req.initial_pose.position.x = x
        sp_req.initial_pose.position.y = y
        sp_req.initial_pose.position.z = z
        sp_req.initial_pose.orientation.w = 1.0
        future = self._spawn_cli.call_async(sp_req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=15.0)
        if future.done() and future.result() and future.result().success:
            self.get_logger().info("Teleport: spawn_entity succeeded")
        else:
            self.get_logger().error("Teleport: spawn_entity failed or timed out")
        time.sleep(2.0)  # extra wait — rover settles onto slope under gravity

    def _collect_readings(self, n: int) -> tuple[list[str], list[float], list[float]]:
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

        for pos in SLOPE_POSITIONS:
            label = pos["label"]
            self.get_logger().info(f"\n{'='*55}")
            self.get_logger().info(f"Slope position: {label}  ({pos['x']}, {pos['y']}, z={pos['z']:.2f})")
            self._teleport(pos["x"], pos["y"], pos["z"])

            self.get_logger().info(f"Waiting {WAIT_STABLE_S}s for camera to stabilise...")
            t_end = time.time() + WAIT_STABLE_S
            while time.time() < t_end:
                rclpy.spin_once(self, timeout_sec=0.3)

            self.get_logger().info(f"Collecting {N_READINGS} DINOv2 readings...")
            labels, confs, latencies = self._collect_readings(N_READINGS)
            actual_speed = self._measure_speed()

            if not labels:
                pred, pred_conf, consistent = "timeout", 0.0, False
            else:
                cnt        = Counter(labels)
                pred       = cnt.most_common(1)[0][0]
                pred_conf  = round(
                    sum(c for l, c in zip(labels, confs) if l == pred)
                    / max(1, cnt[pred]), 3
                )
                consistent = (len(set(labels)) == 1)

            mean_latency    = round(sum(latencies) / len(latencies), 1) if latencies else 0.0
            terrain_correct = (pred == SLOPE_ZONE["ground_truth"])
            speed_correct   = abs(actual_speed - SLOPE_ZONE["expected_speed_ms"]) <= 0.02

            self.get_logger().info(
                f"pred={pred!r} conf={pred_conf:.2f} → terrain_correct={terrain_correct}  "
                f"speed={actual_speed:.3f} → speed_correct={speed_correct}  "
                f"latency={mean_latency:.0f}ms"
            )

            results.append({
                "position_label":    label,
                "x":                 pos["x"],
                "y":                 pos["y"],
                "spawn_z":           pos["z"],
                "ground_truth":      SLOPE_ZONE["ground_truth"],
                "pred":              pred,
                "confidence":        pred_conf,
                "terrain_correct":   terrain_correct,
                "consistent":        consistent,
                "n_readings":        len(labels),
                "all_labels":        ",".join(labels),
                "expected_policy":   SLOPE_ZONE["expected_policy"],
                "expected_speed_ms": SLOPE_ZONE["expected_speed_ms"],
                "actual_speed_ms":   actual_speed,
                "speed_correct":     speed_correct,
                "mean_latency_ms":   mean_latency,
                "slope_angle_deg":   15.0,
                "description":       SLOPE_ZONE["description"],
            })

        return results


def _summarise(results: list[dict]):
    terrain_ok = sum(1 for r in results if r["terrain_correct"] is True)
    n          = len(results)

    print(f"\n{'='*65}")
    print("SLOPE ZONE EXPERIMENT — DINOv2 texture-vs-geometry test")
    print(f"{'='*65}")
    print(f"{'Position':<16} {'GT':<10} {'Pred':<12} {'Conf':<7} {'T?':<5} {'Speed':<7} {'S?'}")
    print("-" * 65)
    for r in results:
        t = "YES" if r["terrain_correct"] is True else "NO "
        s = "YES" if r["speed_correct"]   is True else "NO "
        print(
            f"{r['position_label']:<16} {r['ground_truth']:<10} {r['pred']:<12} "
            f"{r['confidence']:<7.2f} {t:<5} {r['actual_speed_ms']:<7.3f} {s}"
        )
    print("-" * 65)
    print(f"Terrain correct: {terrain_ok}/{n}")
    print("=" * 65)

    if terrain_ok == n:
        print("\nFinding: DINOv2 correctly classifies soil texture on 15° slope.")
        print("Interpretation: model is TEXTURE-BASED — slope inclination does not affect")
        print("terrain label. This is expected. Slope SAFETY cannot be inferred from RGB alone.")
        print("IMU (Step 6a / /exomy/imu_raw) required for slope-hazard detection.")
    else:
        print("\nFinding: DINOv2 misclassifies slope — likely due to unusual texture angle.")
        print("Camera pose change at steep incline changes feature distribution.")


def save_csv(results: list[dict], path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = [
        "position_label", "x", "y", "spawn_z", "ground_truth", "pred",
        "confidence", "terrain_correct", "consistent", "n_readings", "all_labels",
        "expected_policy", "expected_speed_ms", "actual_speed_ms", "speed_correct",
        "mean_latency_ms", "slope_angle_deg", "description",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"\nResults saved → {path}")


def main():
    rclpy.init()
    node = SlopeExperiment()

    print(f"\n{'='*60}")
    print("DINOv2 Slope Zone Experiment (Step 7b)")
    print(f"{'='*60}")
    print("Slope zone: mars_terrain.world at y=-20, 15° incline, soil texture")
    print("Tests: does DINOv2 classify texture-only or detect slope geometry?")
    print("Expected: 'soil' at all positions (texture-based model)")
    print(f"Positions: {[p['label'] for p in SLOPE_POSITIONS]}")
    print("Waiting 5s for DINOv2 node warm-up...\n")
    t_end = time.time() + 5.0
    while time.time() < t_end:
        rclpy.spin_once(node, timeout_sec=0.5)

    results = node.run_experiment()
    _summarise(results)
    save_csv(results, OUT_CSV)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
