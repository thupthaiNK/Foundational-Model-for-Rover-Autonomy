"""
Purpose: Model-agnostic Gazebo 5-zone traversability experiment. Identical zone
         logic to dinov2_traversability_experiment.py but accepts --model flag
         to produce per-model CSVs. Prints a multi-model comparison table at
         the end (including hardcoded CLIP 2/5 and DINOv2 ViT-S 3/5 baselines).
         Used to fill §4.8 Table X in the thesis (Experiment 9 — multi-model).
Inputs:  ROS2 topics (requires the corresponding launch file):
           /terrain_classification  ("label:confidence")
           /exomy/cmd_vel           (geometry_msgs/Twist)
           /robot_description       (URDF XML, for respawning)
         --model: dinov2_vits | dinov2_vitl | dinov3_vits
Outputs: experiments/results/gazebo_traversability_{model}.csv
How to run:
    # Terminal 1 — launch the model-specific stack
    ros2 launch simulation/launch/dinov3_vits_controller_test.launch.py
    # OR
    ros2 launch simulation/launch/dinov2_vitl_controller_test.launch.py

    # Terminal 2 — wait until "generic_terrain_node ready", then:
    python3 experiments/multimodel_traversability_experiment.py --model dinov3_vits
    # OR
    python3 experiments/multimodel_traversability_experiment.py --model dinov2_vitl

    # Show comparison table only (no experiment run):
    python3 experiments/multimodel_traversability_experiment.py --summary

Project: Onboard Visual Foundation Models for Mars Terrain Perception and Rover Navigation
"""

import argparse
import csv
import os
import time
from collections import Counter

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy
from geometry_msgs.msg import Twist
from gazebo_msgs.srv import SpawnEntity, DeleteEntity
from std_msgs.msg import String

RESULTS_DIR   = os.path.join(os.path.dirname(__file__), "results")
WAIT_STABLE_S = 8.0
N_READINGS    = 8

ZONE_POSES = [
    {
        "name":           "soil_zone",
        "x":  0.0, "y":  7.0, "z": 0.15,
        "ground_truth":   "soil",
        "expected_trav":  "SAFE",
        "expected_speed": 0.10,
        "description":    "Open regolith far from bedrock/sand — safe to drive",
    },
    {
        "name":           "bedrock_zone",
        "x":  4.5, "y":  0.0, "z": 0.15,
        "ground_truth":   "bedrock",
        "expected_trav":  "HAZARD",
        "expected_speed": 0.03,
        "description":    "Centre of bedrock slab (world pose 4,0, size 4x6)",
    },
    {
        "name":           "sand_zone",
        "x": -4.5, "y":  0.0, "z": 0.15,
        "ground_truth":   "sand",
        "expected_trav":  "CAUTION",
        "expected_speed": 0.05,
        "description":    "Centre of sand zone (world pose -4,0, size 4x6)",
    },
    {
        "name":           "rock_cluster",
        "x": -0.7, "y": -2.0, "z": 0.15,
        "ground_truth":   "big_rock",
        "expected_trav":  "STOP",
        "expected_speed": 0.00,
        "description":    "1.0m behind rock_foreground: rocks fill 25% FOV",
    },
    {
        "name":           "boulder_zone",
        "x": -1.3, "y": -2.5, "z": 0.15,
        "ground_truth":   "big_rock",
        "expected_trav":  "STOP",
        "expected_speed": 0.00,
        "description":    "1.3m behind boulder_1: boulder fills 26% FOV",
    },
]

POLICY_LABEL = {
    "soil":      "SAFE",
    "sand":      "CAUTION",
    "bedrock":   "HAZARD",
    "big_rock":  "STOP",
    "uncertain": "STOP",
    "unknown":   "STOP",
}

# Hardcoded baselines — already completed and saved to CSV
KNOWN_RESULTS = {
    "clip_vit32": {
        "display":        "CLIP ViT-B/32",
        "ai4mars_acc":    54.4,
        "zones_correct":  2,
        "zones_total":    5,
        "csv":            "gazebo_traversability_clip_vit32.csv",
        "date":           "2026-06-03",
    },
    "dinov2_vits": {
        "display":        "DINOv2+reg ViT-S (authoritative)",
        "ai4mars_acc":    90.24,
        "zones_correct":  3,
        "zones_total":    5,
        "csv":            "dinov2_traversability_experiment.csv",
        "date":           "2026-06-22",
    },
}

VALID_MODELS = ["dinov2_vits", "dinov2_vitl", "dinov3_vits"]

MODEL_META = {
    "dinov2_vits": {
        "display":     "DINOv2+reg ViT-S",
        "ai4mars_acc": 90.24,
    },
    "dinov2_vitl": {
        "display":     "DINOv2 ViT-L",
        "ai4mars_acc": 93.73,
    },
    "dinov3_vits": {
        "display":     "DINOv3 ViT-S/16",
        "ai4mars_acc": 90.20,
    },
}


class TraversabilityExperiment(Node):

    def __init__(self, model_name: str):
        super().__init__("multimodel_traversability_experiment")
        self.model_name     = model_name
        self._terrain_label = None
        self._terrain_conf  = 0.0
        self._cmd_vel_x     = 0.0
        self._urdf_xml      = None

        self.create_subscription(String, "/terrain_classification", self._terrain_cb, 10)
        self.create_subscription(Twist, "/exomy/cmd_vel", self._cmd_vel_cb, 10)

        qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(String, "/robot_description", self._urdf_cb, qos)

        self._delete_client = self.create_client(DeleteEntity, "/delete_entity")
        self._spawn_client  = self.create_client(SpawnEntity,  "/spawn_entity")

        self.get_logger().info(f"Experiment ready | model={model_name}")

    def _terrain_cb(self, msg):
        parts = msg.data.split(":")
        if len(parts) == 2:
            self._terrain_label = parts[0]
            try:
                self._terrain_conf = float(parts[1])
            except ValueError:
                pass

    def _cmd_vel_cb(self, msg):
        self._cmd_vel_x = msg.linear.x

    def _urdf_cb(self, msg):
        self._urdf_xml = msg.data

    def teleport(self, x: float, y: float, z: float) -> bool:
        if self._urdf_xml is None:
            print("  Waiting for /robot_description...", end="", flush=True)
            deadline = time.time() + 8
            while self._urdf_xml is None and time.time() < deadline:
                rclpy.spin_once(self, timeout_sec=0.2)
            print(" done" if self._urdf_xml else " timeout")

        if self._urdf_xml is None:
            self.get_logger().warn("No URDF — cannot respawn")
            return False

        if not self._delete_client.wait_for_service(timeout_sec=3.0):
            self.get_logger().warn("/delete_entity not available")
            return False

        del_req      = DeleteEntity.Request()
        del_req.name = "exomy"
        future = self._delete_client.call_async(del_req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
        if not (future.done() and future.result().success):
            self.get_logger().warn("Delete exomy failed — may already be deleted")

        time.sleep(1.5)

        if not self._spawn_client.wait_for_service(timeout_sec=3.0):
            self.get_logger().warn("/spawn_entity not available")
            return False

        spawn_req = SpawnEntity.Request()
        spawn_req.name  = "exomy"
        spawn_req.xml   = self._urdf_xml
        spawn_req.initial_pose.position.x    = float(x)
        spawn_req.initial_pose.position.y    = float(y)
        spawn_req.initial_pose.position.z    = float(z)
        spawn_req.initial_pose.orientation.w = 1.0

        future = self._spawn_client.call_async(spawn_req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=10.0)

        if future.done() and future.result().success:
            self.get_logger().info(f"Respawned exomy at ({x}, {y}, {z})")
            return True

        self.get_logger().warn(f"Respawn failed at ({x}, {y}, {z})")
        return False

    def collect_readings(self, n: int, wait_s: float) -> dict:
        print(f"  Waiting {wait_s:.0f}s for camera to stabilise...", end="", flush=True)
        deadline = time.time() + wait_s
        while time.time() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
        print(" done")

        labels, confs, speeds = [], [], []
        for _ in range(n):
            rclpy.spin_once(self, timeout_sec=0.5)
            if self._terrain_label and self._terrain_label not in ("unknown",):
                labels.append(self._terrain_label)
                confs.append(self._terrain_conf)
                speeds.append(self._cmd_vel_x)

        if not labels:
            return {"label": "unknown", "conf": 0.0, "speed": 0.0,
                    "consistent": False, "n": 0, "all_labels": ""}

        majority   = Counter(labels).most_common(1)[0][0]
        avg_conf   = sum(confs)  / len(confs)
        avg_speed  = sum(speeds) / len(speeds)
        consistent = labels.count(majority) / len(labels) >= 0.6

        return {
            "label":      majority,
            "conf":       avg_conf,
            "speed":      avg_speed,
            "consistent": consistent,
            "n":          len(labels),
            "all_labels": ",".join(labels),
        }

    def run_experiment(self) -> list[dict]:
        results  = []
        display  = MODEL_META.get(self.model_name, {}).get("display", self.model_name)
        print(f"\n{display} — Gazebo 5-Zone Traversability Experiment")
        print("=" * 65)

        for zone in ZONE_POSES:
            print(f"\nZone: {zone['name']}")
            print(f"  Position: ({zone['x']}, {zone['y']})  |  GT: {zone['ground_truth']}")
            print(f"  Expected: {zone['expected_trav']}  @  {zone['expected_speed']} m/s")

            ok = self.teleport(zone["x"], zone["y"], zone["z"])
            if not ok:
                print("  Warning: teleport failed — recording from current position")

            readings = self.collect_readings(N_READINGS, WAIT_STABLE_S)

            gt_label       = zone["ground_truth"]
            pred_label     = readings["label"]
            terrain_correct = (pred_label == gt_label)
            pred_trav      = POLICY_LABEL.get(pred_label, "STOP")
            trav_correct   = (pred_trav == zone["expected_trav"])
            expected_speed = zone["expected_speed"]
            actual_speed   = readings["speed"]
            speed_correct  = abs(actual_speed - expected_speed) < 0.02

            row = {
                "model":             self.model_name,
                "zone":              zone["name"],
                "ground_truth":      gt_label,
                "model_pred":        pred_label,
                "model_conf":        f"{readings['conf']:.3f}",
                "terrain_correct":   terrain_correct,
                "consistent":        readings["consistent"],
                "n_readings":        readings["n"],
                "expected_trav":     zone["expected_trav"],
                "pred_trav":         pred_trav,
                "trav_correct":      trav_correct,
                "expected_speed_ms": expected_speed,
                "actual_speed_ms":   f"{actual_speed:.3f}",
                "speed_correct":     speed_correct,
                "description":       zone["description"],
                "all_labels":        readings["all_labels"],
            }
            results.append(row)

            t_ok = "✅" if terrain_correct else "❌"
            s_ok = "✅" if speed_correct   else "❌"
            print(f"  {display}: {t_ok} {pred_label} "
                  f"(conf={readings['conf']:.2f}, consistent={readings['consistent']})")
            print(f"  Controller: {s_ok} speed={actual_speed:.3f} m/s "
                  f"(expected={expected_speed:.2f}) → {pred_trav}")

        return results


def save_csv(results: list[dict], model: str):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, f"gazebo_traversability_{model}.csv")
    if not results:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"\nResults saved → {path}")
    return path


def load_csv_zones_correct(path: str) -> int | None:
    """Return number of terrain-correct zones from a saved CSV."""
    if not os.path.exists(path):
        return None
    correct = 0
    try:
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                val = row.get("terrain_correct", "False")
                if val in ("True", "1", "true"):
                    correct += 1
    except Exception:
        return None
    return correct


def print_single_summary(results: list[dict], model: str):
    n       = len(results)
    t_cor   = sum(1 for r in results if r["terrain_correct"])
    display = MODEL_META.get(model, {}).get("display", model)

    print(f"\n{'='*65}")
    print(f"{display.upper()} — SUMMARY")
    print(f"{'='*65}")
    print(f"  Terrain correct:   {t_cor}/{n} = {100*t_cor/n:.0f}%")

    print(f"\n  {'Zone':<18} {'GT':>9} {'Pred':>9} {'Trav':>8} {'Speed':>7} {'OK?':>5}")
    print(f"  {'─'*18} {'─'*9} {'─'*9} {'─'*8} {'─'*7} {'─'*5}")
    for r in results:
        ok    = "✅" if r["terrain_correct"] else "❌"
        speed = r["actual_speed_ms"]
        print(f"  {r['zone']:<18} {r['ground_truth']:>9} {r['model_pred']:>9} "
              f"{r['pred_trav']:>8} {speed:>7} {ok:>5}")


def print_comparison_table(new_results: list[dict] | None, new_model: str | None):
    """Print a combined table of all model results (known baselines + new run)."""
    print(f"\n{'='*70}")
    print("MULTI-MODEL GAZEBO 5-ZONE COMPARISON")
    print(f"{'='*70}")
    print(f"  {'Model':<28} {'AI4Mars':>9} {'Zones':>7} {'%':>6}")
    print(f"  {'─'*28} {'─'*9} {'─'*7} {'─'*6}")

    for key, meta in KNOWN_RESULTS.items():
        path = os.path.join(RESULTS_DIR, meta["csv"])
        zones = load_csv_zones_correct(path)
        if zones is None:
            zones = meta["zones_correct"]   # fallback to hardcoded
        pct = 100 * zones / meta["zones_total"]
        print(f"  {meta['display']:<28} {meta['ai4mars_acc']:>8.2f}% {zones}/{meta['zones_total']:>4}  {pct:.0f}%")

    if new_results and new_model:
        n      = len(new_results)
        t_cor  = sum(1 for r in new_results if r["terrain_correct"])
        pct    = 100 * t_cor / n if n else 0
        meta_m = MODEL_META.get(new_model, {})
        disp   = meta_m.get("display", new_model)
        acc    = meta_m.get("ai4mars_acc", float("nan"))
        print(f"  {disp:<28} {acc:>8.2f}% {t_cor}/{n:>4}  {pct:.0f}%  << THIS RUN")

    # Also check for any previously saved new-model CSVs on disk
    for model_key, meta_m in MODEL_META.items():
        if model_key == new_model:
            continue  # already printed above
        if model_key in KNOWN_RESULTS:
            continue  # already printed in baselines
        path = os.path.join(RESULTS_DIR, f"gazebo_traversability_{model_key}.csv")
        zones = load_csv_zones_correct(path)
        if zones is not None:
            pct  = 100 * zones / 5
            disp = meta_m.get("display", model_key)
            acc  = meta_m.get("ai4mars_acc", float("nan"))
            print(f"  {disp:<28} {acc:>8.2f}% {zones}/5    {pct:.0f}%  (from {os.path.basename(path)})")

    print(f"\n  Note: AI4Mars accuracy = 1000-shot LogReg probe on AI4Mars test set.")
    print(f"  Gazebo result = terrain classification correct / 5 zones.")
    print(f"  Confidence threshold = 0.40 (Gazebo domain gap adjustment).")
    print(f"{'='*70}")


def main():
    parser = argparse.ArgumentParser(description="Multi-model Gazebo traversability experiment")
    parser.add_argument(
        "--model", choices=VALID_MODELS,
        help="Model to test. Must match the running launch file.",
    )
    parser.add_argument(
        "--summary", action="store_true",
        help="Print comparison table from existing CSVs only, no experiment run.",
    )
    args = parser.parse_args()

    if args.summary:
        print_comparison_table(None, None)
        return

    if not args.model:
        parser.error("--model is required unless --summary is used")

    rclpy.init()
    node = TraversabilityExperiment(args.model)

    print(f"Waiting for /terrain_classification ({args.model})...")
    deadline = time.time() + 20
    while node._terrain_label is None and time.time() < deadline:
        rclpy.spin_once(node, timeout_sec=0.5)

    if node._terrain_label is None:
        print("ERROR: /terrain_classification not publishing.\n"
              "  Is the launch file running?\n"
              "  generic_terrain_node needs ~25s to start.")
        node.destroy_node()
        rclpy.shutdown()
        return

    print(f"Topic active — first reading: "
          f"{node._terrain_label}:{node._terrain_conf:.2f}")

    results = node.run_experiment()
    print_single_summary(results, args.model)
    save_csv(results, args.model)
    print_comparison_table(results, args.model)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
