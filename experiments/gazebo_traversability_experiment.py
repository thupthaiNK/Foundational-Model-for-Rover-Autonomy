"""
Purpose:    Gazebo traversability experiment — move ExoMy rover to predefined
            positions in the terrain world and record CLIP + cascade pipeline
            traversability decisions for each terrain zone.
            Implements Experiment 7 from plan.md: "Traversability in Gazebo".
Inputs:     ROS2 topics (requires Gazebo + cascade_pipeline running):
              /terrain_classification  (CLIP output)
              /traversability          (cascade decision)
              /smolvlm_terrain         (SmolVLM output, if available)
            Rover positions defined per terrain zone in ZONE_POSES.
Outputs:    experiments/results/gazebo_traversability_experiment.csv
            Console table: zone × prediction × ground truth × correct?
How to run:
    # Terminal 1: start simulation + cascade pipeline
    source /opt/ros/humble/setup.bash
    source ros2_ws/install/setup.bash
    ros2 launch simulation/launch/cascade_test.launch.py

    # Terminal 2: run this script
    source /opt/ros/humble/setup.bash
    source ros2_ws/install/setup.bash
    python3 experiments/gazebo_traversability_experiment.py

    # Quick test with CLIP only (no SmolVLM):
    python3 experiments/gazebo_traversability_experiment.py --clip-only

Project:    Onboard Visual Foundation Models for Mars Terrain Perception and Rover Navigation
"""

import argparse
import csv
import os
import time

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose
from gazebo_msgs.srv import SetEntityState
from std_msgs.msg import String

# ── Terrain zones in mars_terrain.world ───────────────────────────────────────
# Each zone defines: rover spawn position + ground truth terrain + expected traversability
# Positions verified against world file (2026-06-03):
#   bedrock_zone model: pose (4,0), size 4×6  → covers X: 2–6, Y: -3–3
#   sand_zone model:    pose (-4,0), size 4×6 → covers X: -6 to -2, Y: -3–3
#   rocks: (1.5,1), (2,-1.5), (-1,2), boulder_1 (0.5,-2.5), boulder_2 (3,2)
#   soil: default ground, everything else (20×20 plane)
ZONE_POSES = [
    {
        "name":            "soil_zone",
        "x":  0.0, "y":  7.0, "z": 0.15,
        "ground_truth":    "soil",
        "expected_trav":   "safe",
        "description":     "Open regolith, far from bedrock/sand zones — safe to drive on",
    },
    {
        "name":            "bedrock_zone",
        "x":  4.5, "y":  0.0, "z": 0.15,
        "ground_truth":    "bedrock",
        "expected_trav":   "caution",
        "description":     "Centre of bedrock slab (world pose 4,0, size 4×6)",
    },
    {
        "name":            "sand_zone",
        "x": -4.5, "y":  0.0, "z": 0.15,
        "ground_truth":    "sand",
        "expected_trav":   "safe",
        "description":     "Centre of sand zone (world pose -4,0, size 4×6)",
    },
    {
        "name":            "rock_cluster",
        "x":  1.5, "y": -2.0, "z": 0.15,
        "ground_truth":    "big_rock",
        "expected_trav":   "hazard",
        "description":     "Near rock_2 (2,-1.5) and boulder_1 (0.5,-2.5) cluster",
    },
    {
        "name":            "boulder_zone",
        "x":  0.5, "y": -2.5, "z": 0.15,
        "ground_truth":    "big_rock",
        "expected_trav":   "hazard",
        "description":     "At boulder_1 (0.5,-2.5) — large rock, avoid",
    },
]

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
WAIT_CLIP_S = 3.0    # seconds to wait for stable CLIP readings per zone
N_READINGS  = 5      # number of CLIP readings to collect per zone


class TraversabilityExperiment(Node):

    def __init__(self, clip_only: bool = False):
        super().__init__("traversability_experiment")
        self.clip_only = clip_only

        # Latest topic values
        self._clip_label = None
        self._clip_conf  = 0.0
        self._trav       = None
        self._smolvlm    = None

        # Subscriptions
        self.create_subscription(
            String, "/terrain_classification", self._clip_cb, 10
        )
        self.create_subscription(
            String, "/traversability", self._trav_cb, 10
        )
        if not clip_only:
            self.create_subscription(
                String, "/smolvlm_terrain", self._smolvlm_cb, 10
            )

        # Gazebo set_entity_state service
        self._set_state = self.create_client(
            SetEntityState, "/gazebo/set_entity_state"
        )
        self.get_logger().info("Traversability experiment node ready")

    def _clip_cb(self, msg):
        parts = msg.data.split(":")
        if len(parts) == 2:
            self._clip_label = parts[0]
            try:
                self._clip_conf = float(parts[1])
            except ValueError:
                pass

    def _trav_cb(self, msg):
        self._trav = msg.data

    def _smolvlm_cb(self, msg):
        parts = msg.data.split(":")
        self._smolvlm = parts[0] if parts else msg.data

    def teleport(self, x: float, y: float, z: float) -> bool:
        """Move rover to (x, y, z) via Gazebo set_entity_state service."""
        if not self._set_state.wait_for_service(timeout_sec=3.0):
            self.get_logger().warn("set_entity_state service not available — skipping teleport")
            return False

        req = SetEntityState.Request()
        req.state.name = "exomy"
        req.state.pose.position.x = float(x)
        req.state.pose.position.y = float(y)
        req.state.pose.position.z = float(z)
        req.state.pose.orientation.w = 1.0

        future = self._set_state.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=3.0)
        if future.done():
            return future.result().success
        return False

    def collect_readings(self, n: int, wait_s: float) -> dict:
        """Collect n CLIP readings after waiting wait_s seconds."""
        # Wait for camera to stabilise at new position
        deadline = time.time() + wait_s
        while time.time() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)

        labels, confs = [], []
        for _ in range(n):
            rclpy.spin_once(self, timeout_sec=0.5)
            if self._clip_label:
                labels.append(self._clip_label)
                confs.append(self._clip_conf)

        if not labels:
            return {"label": "unknown", "conf": 0.0, "consistent": False}

        # Majority vote
        from collections import Counter
        majority = Counter(labels).most_common(1)[0][0]
        avg_conf = sum(confs) / len(confs)
        consistent = labels.count(majority) / len(labels) >= 0.6

        return {
            "label":      majority,
            "conf":       avg_conf,
            "consistent": consistent,
            "n_readings": len(labels),
            "all_labels": ",".join(labels),
        }

    def run_experiment(self) -> list[dict]:
        results = []
        self.get_logger().info(
            f"Starting traversability experiment — {len(ZONE_POSES)} zones"
        )

        for zone in ZONE_POSES:
            print(f"\n{'─'*60}")
            print(f"Zone: {zone['name']}  ({zone['description']})")
            print(f"  Expected terrain: {zone['ground_truth']} → {zone['expected_trav']}")

            # Teleport rover
            ok = self.teleport(zone["x"], zone["y"], zone["z"])
            if not ok:
                print(f"  ⚠ Teleport failed — recording with current position")

            # Collect readings
            readings = self.collect_readings(N_READINGS, WAIT_CLIP_S)
            clip_correct  = (readings["label"] == zone["ground_truth"])
            trav_correct  = (self._trav == zone["expected_trav"]) if self._trav else None

            # Build result row
            row = {
                "zone":           zone["name"],
                "ground_truth":   zone["ground_truth"],
                "expected_trav":  zone["expected_trav"],
                "clip_pred":      readings["label"],
                "clip_conf":      f"{readings['conf']:.3f}",
                "clip_correct":   clip_correct,
                "clip_consistent": readings["consistent"],
                "n_clip_readings": readings["n_readings"],
                "trav_pred":      self._trav or "N/A",
                "trav_correct":   trav_correct,
                "smolvlm_pred":   self._smolvlm or "N/A",
                "description":    zone["description"],
            }
            results.append(row)

            status = "✅" if clip_correct else "❌"
            print(f"  CLIP: {status} {readings['label']} (conf={readings['conf']:.2f}, "
                  f"consistent={readings['consistent']})")
            print(f"  Traversability: {self._trav}  (expected: {zone['expected_trav']})")
            if self._smolvlm:
                print(f"  SmolVLM: {self._smolvlm}")

        return results


def save_csv(results: list[dict], path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not results:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"\nCSV saved → {path}")


def print_summary(results: list[dict]):
    n = len(results)
    clip_correct  = sum(1 for r in results if r["clip_correct"])
    trav_correct  = sum(1 for r in results if r.get("trav_correct") is True)
    trav_n        = sum(1 for r in results if r.get("trav_correct") is not None)

    print(f"\n{'='*60}")
    print("TRAVERSABILITY EXPERIMENT SUMMARY")
    print(f"{'='*60}")
    print(f"  Zones tested:           {n}")
    print(f"  CLIP correct:           {clip_correct}/{n} = {100*clip_correct/n:.0f}%")
    if trav_n:
        print(f"  Traversability correct: {trav_correct}/{trav_n} = {100*trav_correct/trav_n:.0f}%")

    print(f"\n  {'Zone':<20} {'GT':>10} {'CLIP':>10} {'Trav':>8} {'Correct':>8}")
    print(f"  {'─'*20} {'─'*10} {'─'*10} {'─'*8} {'─'*8}")
    for r in results:
        ok = "✅" if r["clip_correct"] else "❌"
        print(f"  {r['zone']:<20} {r['ground_truth']:>10} {r['clip_pred']:>10} "
              f"{r['trav_pred']:>8} {ok:>8}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="Gazebo traversability experiment"
    )
    parser.add_argument("--clip-only", action="store_true",
                        help="Skip SmolVLM subscription (CLIP + traversability only)")
    args = parser.parse_args()

    rclpy.init()
    node = TraversabilityExperiment(clip_only=args.clip_only)

    # Wait for topics to be available
    print("Waiting for /terrain_classification topic...")
    deadline = time.time() + 15
    while node._clip_label is None and time.time() < deadline:
        rclpy.spin_once(node, timeout_sec=0.5)

    if node._clip_label is None:
        print("ERROR: /terrain_classification not publishing. "
              "Is cascade_test.launch.py running?")
        node.destroy_node()
        rclpy.shutdown()
        return

    results = node.run_experiment()

    print_summary(results)
    save_csv(
        results,
        os.path.join(RESULTS_DIR, "gazebo_traversability_experiment.csv")
    )

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
