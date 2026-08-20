"""
Purpose: Gazebo traversability re-evaluation with DINOv2 + terrain controller.
         Teleports ExoMy to 5 predefined terrain zones using delete+respawn
         (since libgazebo_ros_state.so is incompatible with Gazebo 11.10.2).
         Collects DINOv2 terrain classifications and controller velocity decisions,
         and compares with ground truth. Produces Experiment 8 results for §4.8.

         Zone results (expected at threshold=0.40):
           soil_zone    → soil     → SAFE    → 0.10 m/s
           bedrock_zone → bedrock  → HAZARD  → 0.03 m/s
           sand_zone    → sand     → CAUTION → 0.05 m/s
           rock_cluster → big_rock → STOP    → 0.00 m/s
           boulder_zone → big_rock → STOP    → 0.00 m/s

Inputs:  ROS2 topics (requires dinov2_controller_test.launch.py running):
           /terrain_classification  (DINOv2 output: "label:confidence")
           /exomy/cmd_vel           (controller output: geometry_msgs/Twist)
           /robot_description       (URDF XML, for respawning)
Outputs: experiments/results/dinov2_traversability_experiment.csv
              experiments/results/dinov2_traversability_timeseries.csv
              experiments/results/figures/speed_profile.png
              experiments/results/figures/confidence_timeseries.png
How to run:
    # Terminal 1
    ros2 launch simulation/launch/dinov2_controller_test.launch.py

    # Terminal 2 (after Terminal 1 shows "dinov2_terrain_node ready")
    python3 experiments/dinov2_traversability_experiment.py

Project: Onboard Visual Foundation Models for Mars Terrain Perception and Rover Navigation
"""

import csv
import os
import time
from collections import Counter

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy
from geometry_msgs.msg import Twist
from gazebo_msgs.srv import SpawnEntity, DeleteEntity
from std_msgs.msg import Bool, String

RESULTS_DIR    = os.path.join(os.path.dirname(__file__), "results")
PRE_DRIVE_S    = 15.0  # seconds rover drives freely in zone before freeze (for video)
WAIT_STABLE_S  = 3.0   # seconds after respawn to clear stale frames (rover frozen)
N_READINGS     = 8     # DINOv2 readings per zone (majority vote, rover frozen)
SPEED_SETTLE_S = 2.0   # seconds after unfreeze before measuring controller speed
N_SPEED        = 3     # cmd_vel samples to average for actual_speed_ms
STARTUP_TIMEOUT_S = 120.0  # wait for DINOv2 warm-up + Gazebo services
READY_CONSECUTIVE_READS = 2
READY_GRACE_S = 3.0

ZONE_POSES = [
    {
        "name":           "soil_zone",
        "x": -7.5, "y":  6.0, "z": 0.15,
        "ground_truth":   "soil",
        "expected_trav":  "SAFE",
        "expected_speed": 0.10,
        "description":    "Centre of soil zone Q2 (x<0,y>0) 15x12m — facing +x into open regolith",
    },
    {
        "name":           "bedrock_zone",
        "x":  7.5, "y":  6.0, "z": 0.15,
        "ground_truth":   "bedrock",
        "expected_trav":  "HAZARD",
        "expected_speed": 0.03,
        "description":    "Centre of bedrock zone Q1 (x>0,y>0) 15x12m — facing +x into cracked slab",
    },
    {
        "name":           "sand_zone",
        "x": -7.5, "y": -6.0, "z": 0.15,
        "ground_truth":   "sand",
        "expected_trav":  "CAUTION",
        "expected_speed": 0.05,
        "description":    "Centre of sand zone Q3 (x<0,y<0) 15x12m — facing +x into smooth sand",
    },
    {
        "name":           "rock_cluster",
        "x":  2.0, "y": -4.0, "z": 0.15,
        "ground_truth":   "big_rock",
        "expected_trav":  "STOP",
        "expected_speed": 0.00,
        "description":    "Inside rock zone Q4 — facing +x; boulder_1(4.5,-3.5) at 2.7m, 12 rocks in FOV",
    },
    {
        "name":           "boulder_zone",
        "x":  2.0, "y": -8.0, "z": 0.15,
        "ground_truth":   "big_rock",
        "expected_trav":  "STOP",
        "expected_speed": 0.00,
        "description":    "Inside rock zone Q4 — facing +x; boulder_2(4.5,-8.0) at 2.5m, 13 rocks in FOV",
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

SPEED_POLICY = [
    ("STOP", 0.00),
    ("HAZARD", 0.03),
    ("CAUTION", 0.05),
    ("SAFE", 0.10),
]


class DINOv2TraversabilityExperiment(Node):

    def __init__(self):
        super().__init__("dinov2_traversability_experiment")

        self._terrain_label  = None
        self._terrain_conf   = 0.0
        self._cmd_vel_x      = 0.0
        self._urdf_xml       = None
        self._e_stop_active  = False
        self._e_stop_reason  = "none"

        # Terrain + cmd_vel subscriptions
        self.create_subscription(String, "/terrain_classification", self._terrain_cb, 10)
        self.create_subscription(Twist, "/exomy/cmd_vel", self._cmd_vel_cb, 10)
        self.create_subscription(Bool,   "/e_stop",        self._estop_cb,  10)
        self.create_subscription(String, "/e_stop_reason", self._ereason_cb, 10)

        # Freeze publisher — freezes terrain_controller_node at 0 m/s during classification
        self._freeze_pub = self.create_publisher(Bool, "/measurement_mode", 10)

        # robot_description — transient local QoS (latched topic)
        qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(String, "/robot_description", self._urdf_cb, qos)

        # Delete + spawn clients (provided by libgazebo_ros_factory.so)
        self._delete_client = self.create_client(DeleteEntity, "/delete_entity")
        self._spawn_client  = self.create_client(SpawnEntity,  "/spawn_entity")

        self.get_logger().info("DINOv2 traversability experiment ready")

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

    def _estop_cb(self, msg: Bool) -> None:
        self._e_stop_active = msg.data

    def _ereason_cb(self, msg: String) -> None:
        self._e_stop_reason = msg.data

    @staticmethod
    def _policy_from_speed(speed_m_s: float) -> str:
        """Map measured controller speed to policy bands used by the experiment.

        Keep the STOP boundary aligned with the experiment's own speed tolerance:
        values within 0.02 m/s of zero are operationally treated as STOP.
        """
        speed_m_s = abs(speed_m_s)
        if speed_m_s < 0.02:
            return "STOP"
        if speed_m_s < 0.04:
            return "HAZARD"
        if speed_m_s < 0.075:
            return "CAUTION"
        return "SAFE"

    # ── Teleport via delete + respawn ──────────────────────────────────────

    def wait_until_ready(self, timeout_s: float) -> bool:
        """Wait for Gazebo services, robot_description, and a real classifier output.

        The launch file starts Gazebo, spawns the rover, then warms up DINOv2 with
        long delays. If the experiment begins as soon as any topic exists, the first
        teleport can race the initial robot spawn and trigger duplicate Gazebo plugin
        nodes. We therefore require:
          1. /robot_description received
          2. /delete_entity and /spawn_entity services available
          3. at least READY_CONSECUTIVE_READS non-"unknown" terrain messages
        """
        print("Waiting for Gazebo services + DINOv2 warm-up...", flush=True)
        deadline = time.time() + timeout_s
        consecutive_real_reads = 0
        last_label = None
        status_print_t = 0.0

        while time.time() < deadline:
            rclpy.spin_once(self, timeout_sec=0.5)

            delete_ok = self._delete_client.wait_for_service(timeout_sec=0.0)
            spawn_ok = self._spawn_client.wait_for_service(timeout_sec=0.0)
            urdf_ok = self._urdf_xml is not None

            if self._terrain_label not in (None, "unknown"):
                if self._terrain_label == last_label:
                    consecutive_real_reads += 1
                else:
                    consecutive_real_reads = 1
                    last_label = self._terrain_label
            else:
                consecutive_real_reads = 0
                last_label = self._terrain_label

            if urdf_ok and delete_ok and spawn_ok and consecutive_real_reads >= READY_CONSECUTIVE_READS:
                print(
                    f"System ready — first stable reading: "
                    f"{self._terrain_label}:{self._terrain_conf:.2f}"
                )
                print(f"Waiting {READY_GRACE_S:.0f}s extra for Gazebo plugins to settle...", flush=True)
                grace_deadline = time.time() + READY_GRACE_S
                while time.time() < grace_deadline:
                    rclpy.spin_once(self, timeout_sec=0.1)
                return True

            now = time.time()
            if now - status_print_t >= 5.0:
                status_print_t = now
                terrain_state = (
                    f"{self._terrain_label}:{self._terrain_conf:.2f}"
                    if self._terrain_label is not None else "none"
                )
                print(
                    "  status:"
                    f" urdf={'yes' if urdf_ok else 'no'}"
                    f" delete_srv={'yes' if delete_ok else 'no'}"
                    f" spawn_srv={'yes' if spawn_ok else 'no'}"
                    f" terrain={terrain_state}"
                    f" stable_reads={consecutive_real_reads}/{READY_CONSECUTIVE_READS}",
                    flush=True,
                )

        return False

    def teleport(self, x: float, y: float, z: float) -> bool:
        # Wait for URDF to be available
        if self._urdf_xml is None:
            print("  Waiting for /robot_description...", end="", flush=True)
            deadline = time.time() + 8
            while self._urdf_xml is None and time.time() < deadline:
                rclpy.spin_once(self, timeout_sec=0.2)
            print(" done" if self._urdf_xml else " timeout")

        if self._urdf_xml is None:
            self.get_logger().warn("No URDF — cannot respawn rover")
            return False

        # 1. Delete exomy — retry once; Gazebo can be slow with 21 mesh rocks
        if not self._delete_client.wait_for_service(timeout_sec=10.0):
            self.get_logger().warn("/delete_entity not available")
            return False

        del_req = DeleteEntity.Request()
        del_req.name = "exomy"
        deleted = False
        for attempt in range(2):
            future = self._delete_client.call_async(del_req)
            rclpy.spin_until_future_complete(self, future, timeout_sec=15.0)
            if future.done() and future.result().success:
                deleted = True
                break
            self.get_logger().warn(
                f"Delete attempt {attempt+1} failed — retrying in 2s"
                if attempt == 0 else "Delete exomy failed — may already be gone"
            )
            if attempt == 0:
                time.sleep(2.0)

        time.sleep(1.5)  # let Gazebo finish removing the entity

        # 2. Respawn at new position
        if not self._spawn_client.wait_for_service(timeout_sec=10.0):
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

    # ── Measurement mode control ───────────────────────────────────────

    def _set_freeze(self, frozen: bool) -> None:
        """Publish /measurement_mode to freeze (True) or unfreeze (False) the controller."""
        msg = Bool()
        msg.data = frozen
        # Publish 3× to ensure the controller receives it despite any timing race
        for _ in range(3):
            self._freeze_pub.publish(msg)
            rclpy.spin_once(self, timeout_sec=0.05)

    def _measure_speed(self) -> float:
        """Unfreeze controller and sample N_SPEED cmd_vel readings after SPEED_SETTLE_S."""
        self._set_freeze(False)
        print(f"  Settling {SPEED_SETTLE_S:.0f}s for controller to reach steady speed...",
              end="", flush=True)
        deadline = time.time() + SPEED_SETTLE_S
        while time.time() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
        print(" done")

        speeds = []
        for _ in range(N_SPEED):
            rclpy.spin_once(self, timeout_sec=0.3)
            speeds.append(self._cmd_vel_x)

        return sum(speeds) / len(speeds) if speeds else 0.0

    # ── Collect readings ───────────────────────────────────────────────────

    def collect_readings(self, n: int, wait_s: float) -> dict:
        """Three-phase measurement: (0) drive freely PRE_DRIVE_S; (1) freeze → classify; (2) unfreeze → speed."""

        # Phase 0: let rover drive freely so video shows real motion per zone
        self._set_freeze(False)
        print(f"  Driving freely {PRE_DRIVE_S:.0f}s...", end="", flush=True)
        deadline = time.time() + PRE_DRIVE_S
        while time.time() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
        print(" done")

        # Phase 1: freeze rover so camera frame stays static during classification
        self._set_freeze(True)
        print(f"  Waiting {wait_s:.0f}s (frozen) for camera to stabilise...", end="", flush=True)
        deadline = time.time() + wait_s
        while time.time() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
        print(" done")

        labels, confs = [], []
        ts_rows: list[dict] = []
        t_start = time.time()
        for _ in range(n):
            rclpy.spin_once(self, timeout_sec=0.5)
            if self._terrain_label and self._terrain_label not in ("unknown",):
                labels.append(self._terrain_label)
                confs.append(self._terrain_conf)
                ts_rows.append({
                    "t_offset_s":  round(time.time() - t_start, 3),
                    "label":       self._terrain_label,
                    "confidence":  round(self._terrain_conf, 3),
                    "cmd_vel_x":   round(self._cmd_vel_x, 4),
                })

        if not labels:
            self._set_freeze(False)
            return {"label": "unknown", "conf": 0.0, "speed": 0.0,
                    "observed_trav": "STOP",
                    "consistent": False, "n": 0, "all_labels": "",
                    "timeseries": [], "e_stop_triggered": self._e_stop_active,
                    "e_stop_reason": self._e_stop_reason}

        majority     = Counter(labels).most_common(1)[0][0]
        majority_idx = [i for i, label in enumerate(labels) if label == majority]
        avg_conf     = sum(confs[i] for i in majority_idx) / len(majority_idx)
        consistent   = labels.count(majority) / len(labels) >= 0.6

        # Phase 2: unfreeze and measure actual controller speed
        avg_speed = self._measure_speed()
        observed_trav = self._policy_from_speed(avg_speed)

        return {
            "label":            majority,
            "conf":             avg_conf,
            "speed":            avg_speed,
            "observed_trav":    observed_trav,
            "consistent":       consistent,
            "n":                len(labels),
            "all_labels":       ",".join(labels),
            "timeseries":       ts_rows,
            "e_stop_triggered": self._e_stop_active,
            "e_stop_reason":    self._e_stop_reason,
        }

    # ── Main experiment loop ───────────────────────────────────────────────

    def run_experiment(self) -> tuple[list[dict], list[dict]]:
        results: list[dict] = []
        timeseries_rows: list[dict] = []
        print(f"\nDINOv2 Traversability Experiment — {len(ZONE_POSES)} zones")
        print("=" * 65)

        for zone in ZONE_POSES:
            print(f"\nZone: {zone['name']}")
            print(f"  Position: ({zone['x']}, {zone['y']})  |  GT: {zone['ground_truth']}")
            print(f"  Expected: {zone['expected_trav']}  @  {zone['expected_speed']} m/s")

            ok = self.teleport(zone["x"], zone["y"], zone["z"])
            if not ok:
                print("  Warning: teleport failed — recording from current position")

            readings = self.collect_readings(N_READINGS, WAIT_STABLE_S)
            for ts in readings["timeseries"]:
                timeseries_rows.append({"zone": zone["name"], **ts})

            gt_label       = zone["ground_truth"]
            pred_label     = readings["label"]
            terrain_correct = (pred_label == gt_label)
            pred_trav      = POLICY_LABEL.get(pred_label, "STOP")
            expected_speed = zone["expected_speed"]
            actual_speed   = readings["speed"]
            observed_trav  = readings["observed_trav"]
            trav_correct   = (observed_trav == zone["expected_trav"])
            speed_correct  = abs(actual_speed - expected_speed) < 0.02

            row = {
                "zone":              zone["name"],
                "ground_truth":      gt_label,
                "dinov2_pred":       pred_label,
                "dinov2_conf":       f"{readings['conf']:.3f}",
                "terrain_correct":   terrain_correct,
                "consistent":        readings["consistent"],
                "n_readings":        readings["n"],
                "expected_trav":     zone["expected_trav"],
                "label_trav":        pred_trav,
                "observed_trav":     observed_trav,
                "trav_correct":      trav_correct,
                "expected_speed_ms": expected_speed,
                "actual_speed_ms":   f"{actual_speed:.3f}",
                "speed_correct":     speed_correct,
                "description":       zone["description"],
                "all_labels":        readings["all_labels"],
                "e_stop_triggered":  readings["e_stop_triggered"],
                "e_stop_reason":     readings["e_stop_reason"],
            }
            results.append(row)

            t_ok = "✅" if terrain_correct else "❌"
            tr_ok = "✅" if trav_correct else "❌"
            c_ok = "✅" if speed_correct else "❌"
            print(f"  DINOv2: {t_ok} {pred_label} "
                  f"(conf={readings['conf']:.2f}, consistent={readings['consistent']})")
            print(f"  Traversability: {tr_ok} observed={observed_trav} "
                  f"(label_policy={pred_trav})")
            print(f"  Controller: {c_ok} speed={actual_speed:.3f} m/s "
                  f"(expected={expected_speed:.2f})")

        return results, timeseries_rows


def save_csv(results: list[dict], path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not results:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"\nResults saved → {path}")


def save_timeseries(rows: list[dict], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Timeseries saved → {path}")


def plot_results(results: list[dict], timeseries: list[dict], figures_dir: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("matplotlib/seaborn not available — skipping plots")
        return

    os.makedirs(figures_dir, exist_ok=True)
    sns.set_theme(style="whitegrid", font_scale=1.1)
    zones = [r["zone"] for r in results]

    # ── Plot 1: Speed profile (actual vs expected per zone) ────────────────
    fig, ax = plt.subplots(figsize=(8, 4))
    x        = list(range(len(zones)))
    expected = [float(r["expected_speed_ms"]) for r in results]
    actual   = [float(r["actual_speed_ms"])   for r in results]
    ax.bar([xi - 0.2 for xi in x], expected, width=0.35, label="Expected",
           color="#4C72B0", alpha=0.85)
    ax.bar([xi + 0.2 for xi in x], actual,   width=0.35, label="Actual",
           color="#DD8452", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([z.replace("_", "\n") for z in zones], fontsize=9)
    ax.set_ylabel("cmd_vel x (m/s)")
    ax.set_title("Controller Speed Profile — DINOv2 5-Zone Experiment")
    ax.legend()
    ax.set_ylim(0, 0.14)
    plt.tight_layout()
    speed_path = os.path.join(figures_dir, "speed_profile.png")
    plt.savefig(speed_path, dpi=150)
    plt.close()
    print(f"Speed profile → {speed_path}")

    # ── Plot 2: Confidence timeseries (per-reading per zone) ───────────────
    if not timeseries:
        return
    fig, ax = plt.subplots(figsize=(10, 4))
    palette = sns.color_palette("husl", len(zones))
    colors  = dict(zip(zones, palette))

    # Concatenate zone time offsets so all zones appear left-to-right
    t_global: float = 0.0
    zone_offsets: dict[str, float] = {}
    for zone in zones:
        rows = [r for r in timeseries if r["zone"] == zone]
        zone_offsets[zone] = t_global
        if rows:
            t_global += rows[-1]["t_offset_s"] + 1.0

    for zone in zones:
        rows = [r for r in timeseries if r["zone"] == zone]
        if not rows:
            continue
        t    = [r["t_offset_s"] + zone_offsets[zone] for r in rows]
        conf = [r["confidence"] for r in rows]
        ax.plot(t, conf, "o-", color=colors[zone], markersize=5, lw=1.5,
                label=zone.replace("_", " "))

    ax.axhline(y=0.40, color="red", linestyle="--", lw=1.5, label="threshold (0.40)")
    ax.set_xlabel("Time (s, zone-concatenated)")
    ax.set_ylabel("DINOv2 confidence")
    ax.set_title("Confidence Timeseries — DINOv2 5-Zone Experiment")
    ax.legend(fontsize=8, ncol=3)
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    conf_path = os.path.join(figures_dir, "confidence_timeseries.png")
    plt.savefig(conf_path, dpi=150)
    plt.close()
    print(f"Confidence timeseries → {conf_path}")


def print_summary(results: list[dict]):
    n      = len(results)
    t_cor  = sum(1 for r in results if r["terrain_correct"])
    tr_cor = sum(1 for r in results if r["trav_correct"])
    sp_cor = sum(1 for r in results if r["speed_correct"])

    print(f"\n{'='*65}")
    print("DINOV2 TRAVERSABILITY EXPERIMENT SUMMARY")
    print(f"{'='*65}")
    print(f"  Zones tested:              {n}")
    print(f"  DINOv2 terrain correct:    {t_cor}/{n} = {100*t_cor/n:.0f}%")
    print(f"  Traversability correct:    {tr_cor}/{n} = {100*tr_cor/n:.0f}%")
    print(f"  Controller speed correct:  {sp_cor}/{n} = {100*sp_cor/n:.0f}%")
    print(f"\n  {'Zone':<18} {'GT':>9} {'DINOv2':>9} "
          f"{'Trav':>8} {'Speed':>7} {'OK?':>5}")
    print(f"  {'─'*18} {'─'*9} {'─'*9} {'─'*8} {'─'*7} {'─'*5}")
    for r in results:
        ok    = "✅" if r["trav_correct"] else "❌"
        speed = r["actual_speed_ms"]
        print(f"  {r['zone']:<18} {r['ground_truth']:>9} {r['dinov2_pred']:>9} "
              f"{r['observed_trav']:>8} {speed:>7} {ok:>5}")
    print(f"\n  Baseline (CLIP):  2/5 = 40%")
    print(f"  DINOv2 (this):   {t_cor}/{n} = {100*t_cor/n:.0f}%")
    print(f"{'='*65}")


def main():
    rclpy.init()
    node = DINOv2TraversabilityExperiment()

    if not node.wait_until_ready(STARTUP_TIMEOUT_S):
        print("ERROR: system not ready for safe teleport experiment.\n"
              "  Required: /robot_description, /delete_entity, /spawn_entity,\n"
              "  and at least two non-'unknown' terrain readings.\n"
              "  Check dinov2_controller_test.launch.py logs for Gazebo/DINOv2 startup.")
        node.destroy_node()
        rclpy.shutdown()
        return

    results, timeseries = node.run_experiment()
    print_summary(results)
    csv_path = os.path.join(RESULTS_DIR, "dinov2_traversability_experiment.csv")
    save_csv(results, csv_path)
    if timeseries:
        ts_path = os.path.join(RESULTS_DIR, "dinov2_traversability_timeseries.csv")
        save_timeseries(timeseries, ts_path)
        plot_results(results, timeseries, os.path.join(RESULTS_DIR, "figures"))

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
