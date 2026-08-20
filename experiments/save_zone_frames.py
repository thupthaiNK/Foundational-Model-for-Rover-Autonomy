"""
Purpose: Teleport rover to the five Gazebo traversability zones and save camera
         frames. Used to verify what DINOv2 actually sees at the same positions
         used by dinov2_traversability_experiment.py.
Inputs:  Running ROS2 stack (dinov2_controller_test.launch.py)
Outputs: docs/figures/gazebo_demo_latest/<zone>_view.png
How to run:
    python3 experiments/save_zone_frames.py
Project: Onboard Visual Foundation Models for Mars Terrain Perception and Rover Navigation
"""

import os
import csv
import time
from collections import Counter
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import String
from gazebo_msgs.srv import SpawnEntity, DeleteEntity
from PIL import Image as PIL

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
OUT_DIR = os.path.join(_REPO_ROOT, "docs", "figures", "gazebo_demo_latest")
LABEL_CSV = os.path.join(OUT_DIR, "zone_frame_labels.csv")
WAIT_STABLE_S = 8.0
N_READINGS = 8

ZONES = [
    {"name": "soil_zone", "x": 0.0, "y": 7.0, "z": 0.15,
     "out": os.path.join(OUT_DIR, "soil_zone_view.png")},
    {"name": "bedrock_zone", "x": 4.5, "y": 0.0, "z": 0.15,
     "out": os.path.join(OUT_DIR, "bedrock_zone_view.png")},
    {"name": "sand_zone", "x": -4.5, "y": 0.0, "z": 0.15,
     "out": os.path.join(OUT_DIR, "sand_zone_view.png")},
    {"name": "rock_cluster", "x": -0.7, "y": -2.0, "z": 0.15,
     "out": os.path.join(OUT_DIR, "rock_cluster_view.png")},
    {"name": "boulder_zone", "x": -1.3, "y": -2.5, "z": 0.15,
     "out": os.path.join(OUT_DIR, "boulder_zone_view.png")},
]


class FrameDiag(Node):
    def __init__(self):
        super().__init__("frame_diag")
        self._img = None
        self._urdf = None
        self._label = None
        self._conf = 0.0
        self.create_subscription(Image, "/exomy/camera/image_raw", self._img_cb, 1)
        self.create_subscription(String, "/terrain_classification", self._lbl_cb, 10)
        qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(String, "/robot_description", self._urdf_cb, qos)
        self._del = self.create_client(DeleteEntity, "/delete_entity")
        self._spn = self.create_client(SpawnEntity, "/spawn_entity")

    def _img_cb(self, msg):
        arr = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width, -1)[:, :, :3]
        if "bgr" in msg.encoding:
            arr = arr[:, :, ::-1].copy()
        self._img = arr.copy()

    def _lbl_cb(self, msg):
        self._label = msg.data
        parts = msg.data.split(":")
        if len(parts) == 2:
            try:
                self._conf = float(parts[1])
            except ValueError:
                self._conf = 0.0

    def _urdf_cb(self, msg):
        self._urdf = msg.data

    def wait_urdf(self):
        deadline = time.time() + 10
        while self._urdf is None and time.time() < deadline:
            rclpy.spin_once(self, timeout_sec=0.2)
        return self._urdf is not None

    def teleport(self, x, y, z):
        self._del.wait_for_service(timeout_sec=3.0)
        req = DeleteEntity.Request()
        req.name = "exomy"
        f = self._del.call_async(req)
        rclpy.spin_until_future_complete(self, f, timeout_sec=5.0)
        time.sleep(1.5)

        self._spn.wait_for_service(timeout_sec=3.0)
        req2 = SpawnEntity.Request()
        req2.name = "exomy"
        req2.xml = self._urdf
        req2.initial_pose.position.x = float(x)
        req2.initial_pose.position.y = float(y)
        req2.initial_pose.position.z = float(z)
        req2.initial_pose.orientation.w = 1.0
        f2 = self._spn.call_async(req2)
        rclpy.spin_until_future_complete(self, f2, timeout_sec=10.0)
        return f2.done() and f2.result().success

    def grab_frame(self, wait=8.0):
        self._img = None
        deadline = time.time() + wait
        while time.time() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
        # grab a fresh frame after wait
        self._img = None
        deadline2 = time.time() + 3.0
        while self._img is None and time.time() < deadline2:
            rclpy.spin_once(self, timeout_sec=0.1)
        return self._img

    def collect_labels(self, n=N_READINGS):
        labels, confs = [], []
        for _ in range(n):
            rclpy.spin_once(self, timeout_sec=0.5)
            if self._label:
                label = self._label.split(":")[0]
                labels.append(label)
                confs.append(self._conf)
        if not labels:
            return {
                "label": "unknown",
                "conf": 0.0,
                "consistent": False,
                "n": 0,
                "all_labels": "",
            }
        majority = Counter(labels).most_common(1)[0][0]
        return {
            "label": majority,
            "conf": sum(confs) / len(confs),
            "consistent": labels.count(majority) / len(labels) >= 0.6,
            "n": len(labels),
            "all_labels": ",".join(labels),
        }


def main():
    rclpy.init()
    node = FrameDiag()
    os.makedirs(OUT_DIR, exist_ok=True)
    print("Waiting for /robot_description...")
    if not node.wait_urdf():
        print("ERROR: no URDF")
        return

    rows = []
    for zone in ZONES:
        print(f"\n--- {zone['name']} ---")
        ok = node.teleport(zone["x"], zone["y"], zone["z"])
        print(f"  Spawn: {'ok' if ok else 'failed'}")
        img = node.grab_frame(wait=WAIT_STABLE_S)
        reading = node.collect_labels()
        print(
            f"  DINOv2: {reading['label']}:{reading['conf']:.3f} "
            f"(consistent={reading['consistent']}, n={reading['n']})"
        )
        if img is not None:
            PIL.fromarray(img).save(zone["out"])
            print(f"  Saved → {zone['out']}")
        else:
            print("  No frame captured")
        rows.append({
            "zone": zone["name"],
            "image": os.path.relpath(zone["out"], _REPO_ROOT),
            **reading,
        })

    with open(LABEL_CSV, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["zone", "image", "label", "conf", "consistent", "n", "all_labels"],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved label evidence → {LABEL_CSV}")

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
