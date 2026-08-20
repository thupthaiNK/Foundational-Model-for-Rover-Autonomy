"""
Purpose: 5-zone SmolVLM-256M-Instruct traversability experiment for Gazebo simulation.
         Tests whether a VLM (SmolVLM) can classify terrain in the 5-zone Mars world
         and compares accuracy against the DINOv2 baseline (3/5 terrain correct).
         SmolVLM publishes to /smolvlm_terrain ("label:smolvlm") and /traversability
         ("safe"|"caution"|"hazard"). Inference takes ~55s/image on CPU — too slow for
         real-time reactive control. This is a documented thesis finding: VLM direct
         inference on CPU is not viable for real-time rover autonomy without acceleration.
         No terrain_controller_node is used; speed column is N/A.
Inputs:  Gazebo with smolvlm_controller_test.launch.py already running
         /smolvlm_terrain (std_msgs/String)  "label:smolvlm"
         /traversability  (std_msgs/String)  "safe"|"caution"|"hazard"
         /exomy/camera/image_raw (visual confirmation only)
Outputs: experiments/results/gazebo_traversability_smolvlm.csv
How to run:
    # Launch Gazebo (Terminal 2):
    source /opt/ros/humble/setup.bash && source ros2_ws/install/setup.bash
    ros2 launch simulation/launch/smolvlm_controller_test.launch.py

    # Run experiment (Terminal 3, after "smolvlm_terrain_node ready" message):
    source /opt/ros/humble/setup.bash && source ros2_ws/install/setup.bash
    python3 experiments/smolvlm_traversability_experiment.py

Project: Onboard Visual Foundation Models for Mars Terrain Perception and Rover Navigation
"""

import csv
import os
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy
from gazebo_msgs.srv import DeleteEntity, SpawnEntity
from std_msgs.msg import String

# ── Zone definitions (quadrant layout matching mars_terrain.world) ────────────
ZONES = [
    {
        "name": "soil_zone",
        "x": -7.5, "y": 6.0,
        "ground_truth": "soil",
        "expected_trav": "safe",       # SmolVLM cascade rule: soil → safe
        "description": "Soil zone Q2 — expect safe traversal",
    },
    {
        "name": "bedrock_zone",
        "x": 7.5, "y": 6.0,
        "ground_truth": "bedrock",
        "expected_trav": "caution",    # cascade rule: bedrock → caution
        "description": "Bedrock zone Q1 — expect caution",
    },
    {
        "name": "sand_zone",
        "x": -7.5, "y": -6.0,
        "ground_truth": "sand",
        "expected_trav": "safe",       # cascade rule: sand → safe (more permissive than DINOv2)
        "description": "Sand zone Q3 — cascade says safe (vs DINOv2 CAUTION)",
    },
    {
        "name": "rock_cluster",
        "x": 2.0, "y": -4.0,
        "ground_truth": "big_rock",
        "expected_trav": "hazard",     # cascade rule: big_rock → hazard
        "description": "Rock cluster Q4 centre — expect hazard",
    },
    {
        "name": "boulder_zone",
        "x": 2.0, "y": -8.0,
        "ground_truth": "big_rock",
        "expected_trav": "hazard",
        "description": "Boulder zone Q4 south — expect hazard",
    },
]

# Timing constants — tuned for ~55s SmolVLM inference on CPU
WAIT_STABLE_S  = 15    # seconds after teleport before collecting readings
N_READINGS     = 3     # readings to collect per zone (3 × ~55s ≈ 3 min/zone)
MAX_WAIT_S     = 300   # hard cap per zone (5 minutes)
SPIN_TIMEOUT   = 2.0   # rclpy.spin_once timeout — short to stay responsive

OUT_DIR = os.path.join(os.path.dirname(__file__), "results")
OUT_CSV = os.path.join(OUT_DIR, "gazebo_traversability_smolvlm.csv")


class SmolVLMExperiment(Node):
    """Subscribes to SmolVLM topics and runs the 5-zone experiment."""

    def __init__(self):
        super().__init__("smolvlm_experiment_node")

        self.create_subscription(String, "/smolvlm_terrain", self._terrain_cb, 10)
        self.create_subscription(String, "/traversability", self._trav_cb, 10)

        # Teleport services
        self._delete_cli = self.create_client(DeleteEntity, "/delete_entity")
        self._spawn_cli  = self.create_client(SpawnEntity, "/spawn_entity")

        # State
        self._terrain_label: str | None = None
        self._trav_label: str | None = None
        self._terrain_count = 0   # incremented per new /smolvlm_terrain message
        self._trav_count    = 0
        self._urdf_xml: str | None = None

        qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(String, "/robot_description", self._urdf_cb, qos)

    # ── Callbacks ────────────────────────────────────────────────────────────

    def _terrain_cb(self, msg: String):
        # format: "label:smolvlm"
        parts = msg.data.split(":")
        if parts:
            self._terrain_label = parts[0].strip().lower()
            self._terrain_count += 1

    def _trav_cb(self, msg: String):
        self._trav_label = msg.data.strip().lower()
        self._trav_count += 1

    def _urdf_cb(self, msg: String):
        self._urdf_xml = msg.data

    # ── Teleport ─────────────────────────────────────────────────────────────

    def _teleport(self, x: float, y: float):
        if self._urdf_xml is None:
            self.get_logger().info("Waiting for /robot_description...")
            deadline = time.time() + 8.0
            while self._urdf_xml is None and time.time() < deadline:
                rclpy.spin_once(self, timeout_sec=0.2)
        if self._urdf_xml is None:
            self.get_logger().error("robot_description unavailable; cannot respawn full rover")
            return

        if not self._delete_cli.wait_for_service(timeout_sec=10.0):
            self.get_logger().error("delete_entity service not available")
            return

        self.get_logger().info(f"Teleport: deleting exomy before respawn to ({x:.1f}, {y:.1f})")
        # Delete
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
        time.sleep(1.0)

        # Respawn
        if not self._spawn_cli.wait_for_service(timeout_sec=10.0):
            self.get_logger().error("spawn_entity service not available")
            return

        self.get_logger().info("Teleport: respawning full rover from /robot_description")
        sp_req = SpawnEntity.Request()
        sp_req.name = "exomy"
        sp_req.xml = self._urdf_xml
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
        time.sleep(1.0)

    # ── Collect SmolVLM terrain readings ─────────────────────────────────────

    def _collect_terrain_readings(self, n: int, max_wait_s: float) -> tuple[list[str], list[str]]:
        """Spin until n readings collected or max_wait_s elapsed.

        Returns (terrain_labels, trav_labels) — same length.
        SmolVLM publishes both on the same cadence so we pair them.
        """
        terrain_labels: list[str] = []
        trav_labels:    list[str] = []

        last_terrain_count = self._terrain_count
        last_trav_count    = self._trav_count

        t_end = time.time() + max_wait_s
        while time.time() < t_end and len(terrain_labels) < n:
            rclpy.spin_once(self, timeout_sec=SPIN_TIMEOUT)

            # New terrain reading arrived
            if self._terrain_count > last_terrain_count:
                terrain_labels.append(self._terrain_label or "unknown")
                last_terrain_count = self._terrain_count
                self.get_logger().info(
                    f"  Reading {len(terrain_labels)}/{n}: "
                    f"terrain={self._terrain_label!r}"
                )

            # New traversability reading arrived (may arrive at different cadence)
            if self._trav_count > last_trav_count:
                trav_labels.append(self._trav_label or "unknown")
                last_trav_count = self._trav_count

        # Pad trav_labels to match terrain_labels length if needed
        while len(trav_labels) < len(terrain_labels):
            trav_labels.append(self._trav_label or "unknown")

        return terrain_labels, trav_labels

    # ── Main experiment ───────────────────────────────────────────────────────

    def run_experiment(self) -> list[dict]:
        results = []

        for zone in ZONES:
            name = zone["name"]
            gt   = zone["ground_truth"]

            self.get_logger().info(f"\n{'='*60}")
            self.get_logger().info(f"Zone: {name}  GT={gt}  ({zone['description']})")
            self.get_logger().info(f"Teleporting to ({zone['x']}, {zone['y']})...")

            self._teleport(zone["x"], zone["y"])

            # Wait for image to stabilise and SmolVLM to process first frame
            self.get_logger().info(
                f"Waiting {WAIT_STABLE_S}s for camera + SmolVLM to settle..."
            )
            t_end = time.time() + WAIT_STABLE_S
            while time.time() < t_end:
                rclpy.spin_once(self, timeout_sec=0.5)

            # Collect readings — SmolVLM ~55s/reading so max_wait covers N × 55s
            self.get_logger().info(
                f"Collecting up to {N_READINGS} readings "
                f"(max {MAX_WAIT_S}s, ~55s each)..."
            )
            terrain_labels, trav_labels = self._collect_terrain_readings(
                N_READINGS, MAX_WAIT_S
            )

            if not terrain_labels:
                self.get_logger().warning(
                    f"No readings collected for {name} within {MAX_WAIT_S}s"
                )
                terrain_labels = ["timeout"]
                trav_labels    = ["unknown"]

            # Majority vote
            from collections import Counter
            terrain_counter = Counter(terrain_labels)
            trav_counter    = Counter(trav_labels)
            pred_terrain = terrain_counter.most_common(1)[0][0]
            pred_trav    = trav_counter.most_common(1)[0][0]

            terrain_correct = (pred_terrain == gt)
            trav_correct    = (pred_trav    == zone["expected_trav"])
            consistent      = (len(set(terrain_labels)) == 1)

            # Inference latency note — computed from observation, not measured here
            inference_note = f"~55s/img on CPU, {len(terrain_labels)} readings in {MAX_WAIT_S}s"

            self.get_logger().info(
                f"Result: pred_terrain={pred_terrain!r} (GT={gt!r}) → "
                f"terrain_correct={terrain_correct}  "
                f"pred_trav={pred_trav!r} → trav_correct={trav_correct}"
            )

            results.append({
                "zone":             name,
                "ground_truth":     gt,
                "smolvlm_pred":     pred_terrain,
                "terrain_correct":  terrain_correct,
                "consistent":       consistent,
                "n_readings":       len(terrain_labels),
                "all_terrain_labels": ",".join(terrain_labels),
                "expected_trav":    zone["expected_trav"],
                "pred_trav":        pred_trav,
                "trav_correct":     trav_correct,
                "all_trav_labels":  ",".join(trav_labels),
                # Speed: N/A — no terrain_controller (SmolVLM too slow for reactive control)
                "expected_speed_ms": "N/A",
                "actual_speed_ms":   "N/A",
                "speed_correct":     "N/A",
                "inference_note":    inference_note,
                "description":       zone["description"],
            })

        return results


def _summarise(results: list[dict]):
    """Print a table and accuracy summary to stdout."""
    terrain_correct  = sum(1 for r in results if r["terrain_correct"] is True)
    trav_correct     = sum(1 for r in results if r["trav_correct"] is True)
    n = len(results)

    print("\n" + "=" * 70)
    print("SMOLVLM 5-ZONE EXPERIMENT SUMMARY")
    print("=" * 70)
    print(f"{'Zone':<20} {'GT':<10} {'Pred':<10} {'Terr?':<8} {'Trav_pred':<12} {'Trav?':<8}")
    print("-" * 70)
    for r in results:
        t_ok = "YES" if r["terrain_correct"] is True else "NO "
        v_ok = "YES" if r["trav_correct"]    is True else "NO "
        print(
            f"{r['zone']:<20} {r['ground_truth']:<10} {r['smolvlm_pred']:<10} "
            f"{t_ok:<8} {r['pred_trav']:<12} {v_ok:<8}"
        )
    print("-" * 70)
    print(f"Terrain correct:       {terrain_correct}/{n}")
    print(f"Traversability correct:{trav_correct}/{n}")
    print(f"Speed correct:         N/A (no terrain_controller — VLM too slow for real-time control)")
    print("=" * 70)
    print("\nThesis note: SmolVLM ~55s/img on CPU is not viable for reactive rover control.")
    print("Use cascade pipeline (Exp 3b) where CLIP (~136ms) drives the controller.")


def save_csv(results: list[dict], path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = [
        "zone", "ground_truth", "smolvlm_pred", "terrain_correct", "consistent",
        "n_readings", "all_terrain_labels", "expected_trav", "pred_trav",
        "trav_correct", "all_trav_labels", "expected_speed_ms", "actual_speed_ms",
        "speed_correct", "inference_note", "description",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"\nResults saved → {path}")


def main():
    rclpy.init()
    node = SmolVLMExperiment()

    print("\n" + "=" * 60)
    print("SmolVLM 5-Zone Traversability Experiment")
    print("=" * 60)
    print("Waiting 5s for SmolVLM node to warm up...")
    t_end = time.time() + 5.0
    while time.time() < t_end:
        rclpy.spin_once(node, timeout_sec=0.5)

    print("Starting 5-zone experiment...\n")
    print(f"Config: N_READINGS={N_READINGS}, MAX_WAIT_S={MAX_WAIT_S}, "
          f"WAIT_STABLE_S={WAIT_STABLE_S}")
    print("Expected total runtime: ~25 minutes\n")

    results = node.run_experiment()
    _summarise(results)
    save_csv(results, OUT_CSV)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
