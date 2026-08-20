"""
Purpose: 5-zone CASCADE pipeline traversability experiment for Gazebo simulation.
         Compares DINOv2-only (Exp 8, 3/5 terrain, 4/5 traversability) against
         the cascade architecture: CLIP fast path + SmolVLM slow path.
         Two metrics are measured per zone:
           (A) terrain_correct — does CLIP's /terrain_classification match GT?
           (B) cascade_trav_correct — does /traversability match expected cascade output?
         Speed is driven by terrain_controller_node using CLIP terrain (fast path).
         The SmolVLM slow path runs asynchronously in the background (every 5s) and
         updates /traversability; it does not block the controller.
Inputs:  Gazebo with cascade_test.launch.py already running
         /terrain_classification (std_msgs/String)  CLIP output "label:confidence"
         /traversability         (std_msgs/String)  cascade "safe"|"caution"|"hazard"
         /exomy/cmd_vel          (geometry_msgs/Twist) controller output
         /measurement_mode       (std_msgs/Bool)    experiment freeze control
Outputs: experiments/results/gazebo_traversability_cascade.csv
How to run:
    # Launch Gazebo (Terminal 2):
    source /opt/ros/humble/setup.bash && source ros2_ws/install/setup.bash
    ros2 launch simulation/launch/cascade_test.launch.py

    # Run experiment (Terminal 3, after controller prints "ready"):
    source /opt/ros/humble/setup.bash && source ros2_ws/install/setup.bash
    python3 experiments/cascade_traversability_experiment.py

Project: Onboard Visual Foundation Models for Mars Terrain Perception and Rover Navigation
"""

import csv
import os
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from gazebo_msgs.srv import DeleteEntity, SpawnEntity
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool, String

# ── Zone definitions ──────────────────────────────────────────────────────────
# CLIP terrain prediction → terrain_correct (metric A)
# Cascade /traversability output → cascade_trav_correct (metric B)
# Controller speed driven by CLIP terrain (same policy as DINOv2 experiment)
ZONES = [
    {
        "name":           "soil_zone",
        "x": -7.5, "y": 6.0,
        "ground_truth":   "soil",
        "expected_trav":  "safe",      # cascade rule: soil → safe
        "expected_speed_ms": 0.10,     # terrain_controller: soil → safe_speed
        "description":    "Soil zone Q2 — CLIP→soil, cascade→safe, ctrl→0.10m/s",
    },
    {
        "name":           "bedrock_zone",
        "x": 7.5, "y": 6.0,
        "ground_truth":   "bedrock",
        "expected_trav":  "caution",   # cascade rule: bedrock → caution
        "expected_speed_ms": 0.03,     # terrain_controller: bedrock → hazard_speed
        "description":    "Bedrock zone Q1 — CLIP→bedrock, cascade→caution, ctrl→0.03m/s",
    },
    {
        "name":           "sand_zone",
        "x": -7.5, "y": -6.0,
        "ground_truth":   "sand",
        "expected_trav":  "safe",      # cascade: sand → safe (more permissive than DINOv2)
        "expected_speed_ms": 0.05,     # terrain_controller: sand → caution_speed
        "description":    "Sand zone Q3 — cascade says safe (vs DINOv2 CAUTION for sand)",
    },
    {
        "name":           "rock_cluster",
        "x": 2.0, "y": -4.0,
        "ground_truth":   "big_rock",
        "expected_trav":  "hazard",    # cascade rule: big_rock → hazard
        "expected_speed_ms": 0.00,     # terrain_controller: big_rock/uncertain → stop
        "description":    "Rock cluster Q4 centre — cascade→hazard, ctrl→STOP",
    },
    {
        "name":           "boulder_zone",
        "x": 2.0, "y": -8.0,
        "ground_truth":   "big_rock",
        "expected_trav":  "hazard",
        "expected_speed_ms": 0.00,
        "description":    "Boulder zone Q4 south — cascade→hazard, ctrl→STOP",
    },
]

# Cascade traversability → DINOv2-equivalent severity (for thesis comparison table)
CASCADE_TO_SEVERITY = {"safe": "SAFE", "caution": "CAUTION", "hazard": "HAZARD"}

# Timing (CLIP is fast ~136ms, so standard DINOv2 protocol works)
WAIT_STABLE_S  = 3     # seconds after teleport before collecting readings
N_READINGS     = 8     # per zone (CLIP readings — fast enough to get 8 quickly)
SPIN_TIMEOUT   = 0.5   # rclpy.spin_once timeout
MAX_WAIT_S     = 30    # hard cap per CLIP reading collection window
SPEED_WAIT_S   = 5.0   # seconds of speed measurement after unfreeze

OUT_DIR = os.path.join(os.path.dirname(__file__), "results")
OUT_CSV = os.path.join(OUT_DIR, "gazebo_traversability_cascade.csv")


class CascadeExperiment(Node):
    """5-zone cascade pipeline experiment: measures CLIP terrain + cascade traversability."""

    def __init__(self):
        super().__init__("cascade_experiment_node")

        # State — CLIP terrain
        self._clip_label: str   = ""
        self._clip_conf: float  = 0.0
        self._clip_count: int   = 0

        # State — cascade traversability (SmolVLM slow path, async)
        self._cascade_trav: str = ""
        self._cascade_count: int = 0

        # State — controller speed
        self._cmd_vel_x: float  = 0.0
        self._urdf_xml: str | None = None

        self.create_subscription(String, "/terrain_classification", self._clip_cb, 10)
        self.create_subscription(String, "/traversability",         self._trav_cb, 10)
        self.create_subscription(Twist,  "/exomy/cmd_vel",          self._vel_cb, 10)
        qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(String, "/robot_description", self._urdf_cb, qos)

        # Measurement mode publisher (freeze controller during readings)
        self._mode_pub = self.create_publisher(Bool, "/measurement_mode", 10)

        # Teleport services
        self._delete_cli = self.create_client(DeleteEntity, "/delete_entity")
        self._spawn_cli  = self.create_client(SpawnEntity,  "/spawn_entity")

    # ── Callbacks ────────────────────────────────────────────────────────────

    def _clip_cb(self, msg: String):
        # format: "label:confidence"
        parts = msg.data.split(":")
        if len(parts) >= 2:
            self._clip_label = parts[0].strip().lower()
            try:
                self._clip_conf = float(parts[1])
            except ValueError:
                self._clip_conf = 0.0
        elif parts:
            self._clip_label = parts[0].strip().lower()
            self._clip_conf  = 0.0
        self._clip_count += 1

    def _trav_cb(self, msg: String):
        self._cascade_trav  = msg.data.strip().lower()
        self._cascade_count += 1

    def _vel_cb(self, msg: Twist):
        self._cmd_vel_x = msg.linear.x

    def _urdf_cb(self, msg: String):
        self._urdf_xml = msg.data

    # ── Freeze control ────────────────────────────────────────────────────────

    def _set_freeze(self, freeze: bool):
        """Publish measurement_mode to stop/start the terrain_controller_node."""
        msg = Bool()
        msg.data = freeze
        for _ in range(3):
            self._mode_pub.publish(msg)
            rclpy.spin_once(self, timeout_sec=0.05)

    # ── Teleport ─────────────────────────────────────────────────────────────

    def _teleport(self, x: float, y: float):
        self._set_freeze(True)  # freeze before teleport
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

    # ── Collect CLIP terrain readings (fast path) ─────────────────────────────

    def _collect_clip_readings(self, n: int) -> tuple[list[str], list[float]]:
        """Collect n CLIP /terrain_classification readings during freeze window.

        Returns (labels, confidences).
        """
        labels:  list[str]   = []
        confs:   list[float] = []
        last_count = self._clip_count
        t_end = time.time() + MAX_WAIT_S

        while len(labels) < n and time.time() < t_end:
            rclpy.spin_once(self, timeout_sec=SPIN_TIMEOUT)
            if self._clip_count > last_count:
                labels.append(self._clip_label)
                confs.append(self._clip_conf)
                last_count = self._clip_count

        return labels, confs

    # ── Collect speed measurement (unfreeze, measure actual cmd_vel) ──────────

    def _measure_speed(self) -> float:
        """Unfreeze controller and record cmd_vel_x for SPEED_WAIT_S seconds."""
        self._set_freeze(False)
        speed_samples: list[float] = []
        t_end = time.time() + SPEED_WAIT_S
        while time.time() < t_end:
            rclpy.spin_once(self, timeout_sec=0.2)
            speed_samples.append(self._cmd_vel_x)
        self._set_freeze(True)  # freeze again for next zone

        if not speed_samples:
            return 0.0
        # Use the non-zero peak or the max (controller settles quickly)
        peak = max(abs(s) for s in speed_samples)
        return round(peak, 4)

    # ── Main experiment ───────────────────────────────────────────────────────

    def run_experiment(self) -> list[dict]:
        results = []

        for zone in ZONES:
            name = zone["name"]
            gt   = zone["ground_truth"]

            self.get_logger().info(f"\n{'='*60}")
            self.get_logger().info(f"Zone: {name}  GT={gt}")
            self.get_logger().info(f"Teleporting to ({zone['x']}, {zone['y']})...")

            self._teleport(zone["x"], zone["y"])

            # Stable-wait with freeze on
            self.get_logger().info(f"Waiting {WAIT_STABLE_S}s for CLIP to stabilise...")
            t_end = time.time() + WAIT_STABLE_S
            while time.time() < t_end:
                rclpy.spin_once(self, timeout_sec=0.3)

            # Collect N_READINGS of CLIP terrain (frozen, so rover stays put)
            self.get_logger().info(f"Collecting {N_READINGS} CLIP readings...")
            clip_labels, clip_confs = self._collect_clip_readings(N_READINGS)

            # Snapshot the cascade traversability that arrived in this window
            # (SmolVLM slow path may not have published yet — use last known)
            cascade_trav_snapshot = self._cascade_trav or "unknown"

            # Measure controller speed (unfreeze briefly)
            actual_speed = self._measure_speed()

            # ── Evaluate ────────────────────────────────────────────────────
            from collections import Counter

            if not clip_labels:
                pred_clip     = "timeout"
                pred_clip_conf = 0.0
                consistent     = False
            else:
                cnt           = Counter(clip_labels)
                pred_clip     = cnt.most_common(1)[0][0]
                pred_clip_conf = round(
                    sum(c for l, c in zip(clip_labels, clip_confs) if l == pred_clip)
                    / max(1, cnt[pred_clip]), 3
                )
                consistent = (len(set(clip_labels)) == 1)

            terrain_correct     = (pred_clip == gt)
            cascade_trav_correct = (cascade_trav_snapshot == zone["expected_trav"])

            # Speed: within ±0.02 m/s tolerance
            exp_spd = zone["expected_speed_ms"]
            speed_correct = abs(actual_speed - exp_spd) <= 0.02

            self.get_logger().info(
                f"CLIP pred={pred_clip!r} (GT={gt!r}) → terrain_correct={terrain_correct}  "
                f"cascade_trav={cascade_trav_snapshot!r} → trav_correct={cascade_trav_correct}  "
                f"speed={actual_speed:.3f} (exp={exp_spd}) → speed_correct={speed_correct}"
            )

            results.append({
                "zone":                 name,
                "ground_truth":         gt,
                "clip_pred":            pred_clip,
                "clip_conf":            pred_clip_conf,
                "terrain_correct":      terrain_correct,
                "consistent":           consistent,
                "n_readings":           len(clip_labels),
                "all_clip_labels":      ",".join(clip_labels),
                "cascade_trav":         cascade_trav_snapshot,
                "expected_cascade_trav": zone["expected_trav"],
                "cascade_trav_correct": cascade_trav_correct,
                "expected_speed_ms":    exp_spd,
                "actual_speed_ms":      actual_speed,
                "speed_correct":        speed_correct,
                "description":          zone["description"],
            })

        return results


def _summarise(results: list[dict]):
    terrain_ok  = sum(1 for r in results if r["terrain_correct"]      is True)
    trav_ok     = sum(1 for r in results if r["cascade_trav_correct"] is True)
    speed_ok    = sum(1 for r in results if r["speed_correct"]         is True)
    n           = len(results)

    print("\n" + "=" * 80)
    print("CASCADE PIPELINE 5-ZONE EXPERIMENT SUMMARY")
    print("=" * 80)
    print(
        f"{'Zone':<20} {'GT':<10} {'CLIP_pred':<12} {'T?':<5} "
        f"{'Cascade_trav':<14} {'V?':<5} {'Speed':<7} {'S?'}"
    )
    print("-" * 80)
    for r in results:
        t = "YES" if r["terrain_correct"]      is True else "NO "
        v = "YES" if r["cascade_trav_correct"] is True else "NO "
        s = "YES" if r["speed_correct"]         is True else "NO "
        print(
            f"{r['zone']:<20} {r['ground_truth']:<10} {r['clip_pred']:<12} {t:<5} "
            f"{r['cascade_trav']:<14} {v:<5} {r['actual_speed_ms']:<7.3f} {s}"
        )
    print("-" * 80)
    print(f"CLIP terrain correct:      {terrain_ok}/{n}")
    print(f"Cascade trav correct:      {trav_ok}/{n}")
    print(f"Speed correct:             {speed_ok}/{n}")
    print("=" * 80)
    print("\nComparison (thesis Table §4.8):")
    print(f"  DINOv2 baseline:   3/5 terrain, 4/5 traversability, 4/5 speed")
    print(f"  Cascade pipeline:  {terrain_ok}/5 terrain (CLIP), {trav_ok}/5 traversability, {speed_ok}/5 speed")


def save_csv(results: list[dict], path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = [
        "zone", "ground_truth", "clip_pred", "clip_conf", "terrain_correct",
        "consistent", "n_readings", "all_clip_labels", "cascade_trav",
        "expected_cascade_trav", "cascade_trav_correct", "expected_speed_ms",
        "actual_speed_ms", "speed_correct", "description",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"\nResults saved → {path}")


def main():
    rclpy.init()
    node = CascadeExperiment()

    print("\n" + "=" * 60)
    print("Cascade Pipeline 5-Zone Traversability Experiment")
    print("=" * 60)
    print("Waiting 5s for CLIP + cascade pipeline to warm up...")
    t_end = time.time() + 5.0
    while time.time() < t_end:
        rclpy.spin_once(node, timeout_sec=0.5)

    print("Starting 5-zone experiment...\n")
    print(
        f"Config: N_READINGS={N_READINGS} (CLIP), "
        f"WAIT_STABLE_S={WAIT_STABLE_S}, SPEED_WAIT_S={SPEED_WAIT_S}"
    )
    print("Note: SmolVLM slow path runs asynchronously (~55s/img) in background.\n")

    results = node.run_experiment()
    _summarise(results)
    save_csv(results, OUT_CSV)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
