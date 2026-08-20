"""
Purpose: X4 Sequential drive experiment — rover drives continuously from soil zone
         into bedrock zone without teleport, recording terrain classification and
         cmd_vel as terrain transitions. Demonstrates real-time classification
         response to terrain change (not zone-by-zone teleport).
         Path: spawn at (-3, 6) facing +x → soil zone 3m → cross x=0 → bedrock zone.
Inputs:  /terrain_classification  (std_msgs/String)  from DINOv2 node
         /exomy/cmd_vel           (geometry_msgs/Twist) from terrain controller
         Gazebo spawn/delete services (for initial placement)
Outputs: experiments/results/sequential_drive.csv
         experiments/results/figures/sequential_drive.png
How to run:
    # Terminal 1: Gazebo + DINOv2 already running (run_exp_launch.sh)
    # Terminal 2:
    source /opt/ros/humble/setup.bash && source ros2_ws/install/setup.bash
    python3 experiments/sequential_drive_experiment.py
Project: Onboard Visual Foundation Models for Mars Terrain Perception and Rover Navigation
"""

import csv, os, time, math
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Twist
from gazebo_msgs.srv import SpawnEntity, DeleteEntity
import xacro

RECORD_SECONDS = 90        # total recording time after rover spawns
SPAWN_X        = -3.0      # 3 m inside soil zone (soil extends from x=-15 to x=0)
SPAWN_Y        =  6.0      # centre of soil/bedrock corridor (y=6)
SPAWN_Z        =  0.15

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

SIM_DIR   = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "simulation"))
URDF_PATH = os.path.join(SIM_DIR, "urdf", "exomy.urdf.xacro")


class SequentialDriveRecorder(Node):
    def __init__(self):
        super().__init__("sequential_drive_recorder")
        self._records = []
        self._terrain  = "uncertain"
        self._conf     = 0.0
        self._speed    = 0.0
        self._t0       = None

        self.create_subscription(String, "/terrain_classification", self._terrain_cb, 10)
        self.create_subscription(Twist,  "/exomy/cmd_vel",          self._vel_cb,     10)

    def _terrain_cb(self, msg):
        parts = msg.data.split(":")
        self._terrain = parts[0].strip().lower() if parts else "uncertain"
        try:
            self._conf = float(parts[1]) if len(parts) > 1 else 0.0
        except ValueError:
            self._conf = 0.0

    def _vel_cb(self, msg):
        self._speed = msg.linear.x

    def record_tick(self):
        if self._t0 is None:
            self._t0 = time.time()
        elapsed = time.time() - self._t0
        self._records.append({
            "elapsed_s":  round(elapsed, 2),
            "terrain":    self._terrain,
            "confidence": round(self._conf, 3),
            "speed_ms":   round(self._speed, 4),
            "x_est":      round(SPAWN_X + self._integrated_x(), 2),
        })

    def _integrated_x(self):
        # approximate x position by integrating logged speed over time
        total = 0.0
        for i in range(1, len(self._records)):
            dt = self._records[i]["elapsed_s"] - self._records[i-1]["elapsed_s"]
            total += self._records[i-1]["speed_ms"] * dt
        return total

    def save_csv(self, path):
        if not self._records:
            return
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(self._records[0].keys()))
            w.writeheader()
            w.writerows(self._records)
        self.get_logger().info(f"Saved {len(self._records)} records → {path}")

    def summary(self):
        terrains = [r["terrain"] for r in self._records]
        unique = list(dict.fromkeys(terrains))
        self.get_logger().info(f"Terrain sequence: {' → '.join(unique)}")
        self.get_logger().info(f"Total records: {len(self._records)}")


def respawn(node, x, y, z):
    urdf_xml = xacro.process_file(URDF_PATH).toxml()

    del_cli   = node.create_client(DeleteEntity, "/delete_entity")
    spawn_cli = node.create_client(SpawnEntity, "/spawn_entity")

    if del_cli.wait_for_service(timeout_sec=5.0):
        req = DeleteEntity.Request()
        req.name = "exomy"
        future = del_cli.call_async(req)
        rclpy.spin_until_future_complete(node, future, timeout_sec=5.0)
        time.sleep(1.0)
    else:
        node.get_logger().warn("delete_entity not available — skipping delete")

    if not spawn_cli.wait_for_service(timeout_sec=10.0):
        node.get_logger().warn("spawn_entity not available — skipping spawn, recording from current pose")
        return

    req = SpawnEntity.Request()
    req.name            = "exomy"
    req.xml             = urdf_xml
    req.initial_pose.position.x = x
    req.initial_pose.position.y = y
    req.initial_pose.position.z = z
    req.initial_pose.orientation.w = 1.0  # facing +x
    future = spawn_cli.call_async(req)
    rclpy.spin_until_future_complete(node, future, timeout_sec=10.0)
    node.get_logger().info(f"Respawned exomy at ({x}, {y}, {z})")
    time.sleep(3.0)  # let physics settle


def main():
    rclpy.init()
    node = SequentialDriveRecorder()
    node.get_logger().info("Sequential drive experiment starting...")
    node.get_logger().info(f"Path: ({SPAWN_X}, {SPAWN_Y}) → +x → soil → bedrock boundary at x=0")

    # Spawn rover 3m inside soil zone facing +x
    respawn(node, SPAWN_X, SPAWN_Y, SPAWN_Z)

    node.get_logger().info(f"Recording {RECORD_SECONDS}s... (rover driving autonomously)")

    end_time = time.time() + RECORD_SECONDS
    while time.time() < end_time:
        rclpy.spin_once(node, timeout_sec=0.5)
        node.record_tick()
        remaining = end_time - time.time()
        if int(remaining) % 10 == 0 and remaining % 10 < 0.6:
            node.get_logger().info(
                f"  t={node._records[-1]['elapsed_s']:.0f}s | "
                f"terrain={node._terrain}(conf={node._conf:.2f}) | "
                f"speed={node._speed:.3f}m/s | x_est={node._records[-1]['x_est']:.1f}m"
            )

    node.summary()

    csv_path = os.path.join(RESULTS_DIR, "sequential_drive.csv")
    node.save_csv(csv_path)

    # Plot
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        records = node._records
        t   = [r["elapsed_s"]  for r in records]
        spd = [r["speed_ms"]   for r in records]
        trr = [r["terrain"]    for r in records]
        x   = [r["x_est"]      for r in records]

        COLOUR = {
            "soil":      "#2ecc71",
            "bedrock":   "#e67e22",
            "sand":      "#f1c40f",
            "big_rock":  "#e74c3c",
            "uncertain": "#95a5a6",
        }

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
        fig.suptitle("X4 Sequential Drive — Soil→Bedrock Transition (no teleport)", fontsize=13)

        # Speed trace coloured by terrain
        for i in range(len(t) - 1):
            c = COLOUR.get(trr[i], "#95a5a6")
            ax1.plot(t[i:i+2], spd[i:i+2], color=c, linewidth=2)
        ax1.axvline(x=0, color="k", linestyle=":", alpha=0.3)
        ax1.axhline(y=0.10, color="#2ecc71", linestyle="--", alpha=0.5, label="SAFE (0.10)")
        ax1.axhline(y=0.05, color="#f1c40f", linestyle="--", alpha=0.5, label="CAUTION (0.05)")
        ax1.axhline(y=0.03, color="#e67e22", linestyle="--", alpha=0.5, label="HAZARD (0.03)")
        ax1.set_ylabel("cmd_vel (m/s)")
        ax1.set_ylim(-0.01, 0.13)
        ax1.legend(loc="upper right", fontsize=8)
        ax1.grid(True, alpha=0.3)

        # Terrain label over time
        unique_t = sorted(set(trr))
        t_idx = {t: i for i, t in enumerate(unique_t)}
        y_vals = [t_idx[t] for t in trr]
        scatter_colours = [COLOUR.get(t, "#95a5a6") for t in trr]
        ax2.scatter(t, y_vals, c=scatter_colours, s=20, zorder=3)
        ax2.set_yticks(range(len(unique_t)))
        ax2.set_yticklabels(unique_t)
        ax2.axvline(x=0, color="k", linestyle=":", alpha=0.3)
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("DINOv2 terrain label")
        ax2.grid(True, alpha=0.3)

        patches = [mpatches.Patch(color=c, label=l) for l, c in COLOUR.items() if l in unique_t]
        ax2.legend(handles=patches, loc="upper right", fontsize=8)

        plt.tight_layout()
        fig_path = os.path.join(FIGURES_DIR, "sequential_drive.png")
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        node.get_logger().info(f"Figure saved → {fig_path}")
        plt.close()
    except Exception as e:
        node.get_logger().warn(f"Plot failed: {e}")

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
