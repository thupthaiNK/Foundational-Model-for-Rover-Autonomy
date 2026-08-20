"""
Purpose: X3 Domain Randomisation — measure DINOv2 terrain classification variance
         across 10 random rover positions within the Q4 rock zone (x>0, y<0).
         Tests spatial robustness: does the classifier produce consistent rock-zone
         labels regardless of viewpoint, or does accuracy depend on exact position?
         Expected: uncertain/big_rock labels throughout Q4 (rock textures visible
         from all positions); variance in confidence reveals classifier sensitivity
         to viewpoint/rock density.
Inputs:  /terrain_classification  (std_msgs/String)  DINOv2 + LogReg probe
         /robot_description       (std_msgs/String)  URDF for respawn
         /delete_entity, /spawn_entity               Gazebo services
Outputs: experiments/results/domain_randomisation.csv
         experiments/results/figures/domain_randomisation.png
How to run:
    # Terminal 1: Gazebo + DINOv2 already running (run_exp_launch.sh)
    # Terminal 2:
    source /opt/ros/humble/setup.bash && source ros2_ws/install/setup.bash
    python3 experiments/domain_randomisation_experiment.py
Project: Onboard Visual Foundation Models for Mars Terrain Perception and Rover Navigation
"""

import csv, os, time, random
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from gazebo_msgs.srv import SpawnEntity, DeleteEntity
import xacro

# Q4 rock zone: x ∈ (0, 15), y ∈ (-12, 0)
# Safe margins keep rover 1.5m from each wall
Q4_X_MIN, Q4_X_MAX = 1.5, 12.0
Q4_Y_MIN, Q4_Y_MAX = -10.5, -1.5
SPAWN_Z             = 0.15
N_POSITIONS         = 10
READINGS_PER_POS    = 8
SETTLE_S            = 3.5   # wait after respawn for camera to stabilise
RANDOM_SEED         = 42    # reproducible positions

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

SIM_DIR   = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "simulation"))
URDF_PATH = os.path.join(SIM_DIR, "urdf", "exomy.urdf.xacro")

COLOUR = {
    "soil":      "#2ecc71",
    "bedrock":   "#e67e22",
    "sand":      "#f1c40f",
    "big_rock":  "#e74c3c",
    "uncertain": "#95a5a6",
}


class DomainRandNode(Node):
    def __init__(self):
        super().__init__("domain_rand_node")
        self._terrain  = "unknown"
        self._conf     = 0.0
        self._urdf_xml = xacro.process_file(URDF_PATH).toxml()

        self.create_subscription(String, "/terrain_classification", self._terrain_cb, 10)

        self._delete_client = self.create_client(DeleteEntity, "/delete_entity")
        self._spawn_client  = self.create_client(SpawnEntity,  "/spawn_entity")

    def _terrain_cb(self, msg):
        parts = msg.data.split(":")
        self._terrain = parts[0].strip().lower() if parts else "uncertain"
        try:
            self._conf = float(parts[1]) if len(parts) > 1 else 0.0
        except ValueError:
            self._conf = 0.0

    def collect_readings(self, n: int) -> list[dict]:
        readings = []
        deadline = time.time() + n * 1.0 + 5.0
        while len(readings) < n and time.time() < deadline:
            rclpy.spin_once(self, timeout_sec=0.5)
            if self._terrain != "unknown":
                readings.append({"terrain": self._terrain, "conf": round(self._conf, 3)})
        return readings

    def teleport(self, x: float, y: float) -> bool:
        if not self._delete_client.wait_for_service(timeout_sec=10.0):
            return False
        del_req = DeleteEntity.Request()
        del_req.name = "exomy"
        for attempt in range(2):
            future = self._delete_client.call_async(del_req)
            rclpy.spin_until_future_complete(self, future, timeout_sec=15.0)
            if future.done() and future.result().success:
                break
            if attempt == 0:
                time.sleep(2.0)
        time.sleep(1.5)

        if not self._spawn_client.wait_for_service(timeout_sec=10.0):
            return False
        req = SpawnEntity.Request()
        req.name = "exomy"
        req.xml  = self._urdf_xml
        req.initial_pose.position.x    = float(x)
        req.initial_pose.position.y    = float(y)
        req.initial_pose.position.z    = SPAWN_Z
        req.initial_pose.orientation.w = 1.0
        future = self._spawn_client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=10.0)
        if future.done() and future.result().success:
            self.get_logger().info(f"  Spawned at ({x:.2f}, {y:.2f})")
            return True
        return False


def summarise(readings: list[dict]) -> dict:
    if not readings:
        return {"mode_terrain": "no_data", "mean_conf": 0.0, "n_readings": 0,
                "n_uncertain": 0, "n_big_rock": 0, "n_other": 0}
    from collections import Counter
    counts = Counter(r["terrain"] for r in readings)
    mode_terrain = counts.most_common(1)[0][0]
    mean_conf = round(sum(r["conf"] for r in readings) / len(readings), 3)
    n_uncertain = counts.get("uncertain", 0)
    n_big_rock  = counts.get("big_rock", 0)
    n_other = len(readings) - n_uncertain - n_big_rock
    return {
        "mode_terrain": mode_terrain,
        "mean_conf":    mean_conf,
        "n_readings":   len(readings),
        "n_uncertain":  n_uncertain,
        "n_big_rock":   n_big_rock,
        "n_other":      n_other,
    }


def main():
    rclpy.init()
    node = DomainRandNode()

    # Generate reproducible random positions in Q4
    rng = random.Random(RANDOM_SEED)
    positions = [
        (round(rng.uniform(Q4_X_MIN, Q4_X_MAX), 2),
         round(rng.uniform(Q4_Y_MIN, Q4_Y_MAX), 2))
        for _ in range(N_POSITIONS)
    ]

    node.get_logger().info(
        f"X3 Domain Randomisation — {N_POSITIONS} positions in Q4 rock zone (seed={RANDOM_SEED})"
    )
    node.get_logger().info(f"Q4 bounds: x=[{Q4_X_MIN},{Q4_X_MAX}]  y=[{Q4_Y_MIN},{Q4_Y_MAX}]")

    # Wait for DINOv2 to warm up
    node.get_logger().info("Waiting for /terrain_classification...")
    deadline = time.time() + 30.0
    while node._terrain == "unknown" and time.time() < deadline:
        rclpy.spin_once(node, timeout_sec=0.5)
    if node._terrain == "unknown":
        node.get_logger().warn("No terrain classification received — is DINOv2 running?")

    results = []
    for i, (x, y) in enumerate(positions):
        node.get_logger().info(f"\n[{i+1}/{N_POSITIONS}] pos=({x}, {y})")
        ok = node.teleport(x, y)
        if not ok:
            node.get_logger().warn("  Teleport failed — skipping")
            continue

        time.sleep(SETTLE_S)
        readings = node.collect_readings(READINGS_PER_POS)
        s = summarise(readings)

        row = {"pos_id": i+1, "x": x, "y": y, **s}
        results.append(row)
        node.get_logger().info(
            f"  mode={s['mode_terrain']}  mean_conf={s['mean_conf']:.3f}  "
            f"uncertain={s['n_uncertain']}/{s['n_readings']}  "
            f"big_rock={s['n_big_rock']}/{s['n_readings']}"
        )

    # Save CSV
    csv_path = os.path.join(RESULTS_DIR, "domain_randomisation.csv")
    if results:
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            w.writeheader()
            w.writerows(results)
        node.get_logger().info(f"\nSaved {len(results)} rows → {csv_path}")

    # Summary statistics
    if results:
        n_correct = sum(1 for r in results if r["mode_terrain"] in ("uncertain", "big_rock"))
        n_wrong   = len(results) - n_correct
        node.get_logger().info(
            f"\n=== X3 SUMMARY ===\n"
            f"  Positions tested:          {len(results)}\n"
            f"  Correct rock-zone label:   {n_correct}/{len(results)}\n"
            f"    (uncertain or big_rock)\n"
            f"  Wrong label (soil/sand/..): {n_wrong}/{len(results)}\n"
            f"  Mean confidence across all: "
            f"{sum(r['mean_conf'] for r in results)/len(results):.3f}"
        )

    # Plot
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import numpy as np

        fig, (ax_map, ax_bar) = plt.subplots(1, 2, figsize=(13, 6))
        fig.suptitle("X3 Domain Randomisation — Q4 Rock Zone (N=10 random positions)", fontsize=13)

        # Left: scatter map of Q4
        ax_map.set_xlim(-1, 16)
        ax_map.set_ylim(-13, 1)
        ax_map.set_xlabel("World X (m)")
        ax_map.set_ylabel("World Y (m)")
        ax_map.set_title("Terrain label at each random position")

        # Draw Q4 zone boundary
        from matplotlib.patches import Rectangle
        ax_map.add_patch(Rectangle((0, -12), 15, 12,
                                   linewidth=2, edgecolor="#e74c3c", facecolor="#fde8e8",
                                   alpha=0.3, zorder=0))
        ax_map.text(7.5, -0.3, "Q4 — Rock Zone", ha="center", fontsize=9, color="#c0392b")
        ax_map.axvline(0, color="k", linewidth=0.8, linestyle="--", alpha=0.4)
        ax_map.axhline(0, color="k", linewidth=0.8, linestyle="--", alpha=0.4)

        for r in results:
            c = COLOUR.get(r["mode_terrain"], "#95a5a6")
            ax_map.scatter(r["x"], r["y"], c=c, s=200, zorder=5, edgecolors="k", linewidths=0.8)
            ax_map.annotate(
                f"#{r['pos_id']}\n{r['mode_terrain'][:4]}\n{r['mean_conf']:.2f}",
                (r["x"], r["y"]), textcoords="offset points", xytext=(6, 6),
                fontsize=7, color="k"
            )

        patches = [mpatches.Patch(color=c, label=l) for l, c in COLOUR.items()]
        ax_map.legend(handles=patches, fontsize=8, loc="upper left")
        ax_map.set_aspect("equal")
        ax_map.grid(True, alpha=0.3)

        # Right: bar chart of classification distribution
        labels_count = {}
        for r in results:
            labels_count[r["mode_terrain"]] = labels_count.get(r["mode_terrain"], 0) + 1
        bars = ax_bar.bar(
            list(labels_count.keys()),
            list(labels_count.values()),
            color=[COLOUR.get(l, "#95a5a6") for l in labels_count],
            edgecolor="k", linewidth=0.8
        )
        ax_bar.set_xlabel("Mode terrain label (per position)")
        ax_bar.set_ylabel("Number of positions")
        ax_bar.set_title("Classification distribution across 10 positions")
        ax_bar.set_ylim(0, N_POSITIONS + 1)
        for bar, count in zip(bars, labels_count.values()):
            ax_bar.text(bar.get_x() + bar.get_width()/2, count + 0.1,
                        str(count), ha="center", va="bottom", fontsize=11)
        ax_bar.grid(True, axis="y", alpha=0.3)

        plt.tight_layout()
        fig_path = os.path.join(FIGURES_DIR, "domain_randomisation.png")
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        node.get_logger().info(f"Figure saved → {fig_path}")
        plt.close()
    except Exception as e:
        node.get_logger().warn(f"Plot failed: {e}")

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
