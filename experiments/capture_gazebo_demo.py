"""
Purpose: Capture Gazebo rover-camera views for all 5 terrain zones used in the
         DINOv2 traversability experiment (§4.8). Saves, per zone, the raw camera
         frame and the annotated /terrain_viz frame (label + confidence bars), then
         builds a single 5-panel contact sheet for the thesis figure and demo.
Inputs:  Running ROS2 stack (dinov2_controller_test.launch.py) publishing
           /exomy/camera/image_raw, /terrain_viz, /terrain_classification,
           /robot_description, and the /spawn_entity, /delete_entity services.
Outputs: docs/figures/gazebo_demo/<zone>_raw.png
         docs/figures/gazebo_demo/<zone>_viz.png
         docs/figures/gazebo_demo/gazebo_zones_contact_sheet.png  (thesis figure)
How to run:
    # Terminal 1
    source /opt/ros/humble/setup.bash && source ros2_ws/install/setup.bash
    ros2 launch simulation/launch/dinov2_controller_test.launch.py
    # Terminal 2 (after "dinov2_terrain_node ready")
    source /opt/ros/humble/setup.bash && source ros2_ws/install/setup.bash
    python3 experiments/capture_gazebo_demo.py
Project: Onboard Visual Foundation Models for Mars Terrain Perception and Rover Navigation
"""

import os
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import String
from gazebo_msgs.srv import SpawnEntity, DeleteEntity
from PIL import Image as PIL, ImageDraw

# Zone positions — identical to experiments/dinov2_traversability_experiment.py
ZONES = [
    {"name": "soil_zone",    "x":  0.0, "y":  7.0, "gt": "soil",     "exp": "SAFE @ 0.10"},
    {"name": "bedrock_zone", "x":  4.5, "y":  0.0, "gt": "bedrock",  "exp": "HAZARD @ 0.03"},
    {"name": "sand_zone",    "x": -4.5, "y":  0.0, "gt": "sand",     "exp": "CAUTION @ 0.05"},
    {"name": "rock_cluster", "x": -0.7, "y": -2.0, "gt": "big_rock", "exp": "STOP @ 0.00"},
    {"name": "boulder_zone", "x": -1.3, "y": -2.5, "gt": "big_rock", "exp": "STOP @ 0.00"},
]

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "docs", "figures", "gazebo_demo")
WAIT_STABLE_S = 8.0


class DemoCapture(Node):
    def __init__(self):
        super().__init__("gazebo_demo_capture")
        self._raw = None
        self._viz = None
        self._label = None
        self._urdf = None
        self.create_subscription(Image, "/exomy/camera/image_raw", self._raw_cb, 1)
        self.create_subscription(Image, "/terrain_viz", self._viz_cb, 1)
        self.create_subscription(String, "/terrain_classification", self._lbl_cb, 10)
        qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(String, "/robot_description", self._urdf_cb, qos)
        self._del = self.create_client(DeleteEntity, "/delete_entity")
        self._spn = self.create_client(SpawnEntity, "/spawn_entity")

    @staticmethod
    def _to_rgb(msg):
        arr = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width, -1)[:, :, :3]
        if "bgr" in msg.encoding:
            arr = arr[:, :, ::-1]
        return arr.copy()

    def _raw_cb(self, msg):  self._raw = self._to_rgb(msg)
    def _viz_cb(self, msg):  self._viz = self._to_rgb(msg)
    def _lbl_cb(self, msg):  self._label = msg.data
    def _urdf_cb(self, msg): self._urdf = msg.data

    def wait_urdf(self):
        deadline = time.time() + 10
        while self._urdf is None and time.time() < deadline:
            rclpy.spin_once(self, timeout_sec=0.2)
        return self._urdf is not None

    def teleport(self, x, y, z=0.15):
        self._del.wait_for_service(timeout_sec=3.0)
        req = DeleteEntity.Request(); req.name = "exomy"
        rclpy.spin_until_future_complete(self, self._del.call_async(req), timeout_sec=5.0)
        time.sleep(1.5)
        self._spn.wait_for_service(timeout_sec=3.0)
        req2 = SpawnEntity.Request()
        req2.name = "exomy"; req2.xml = self._urdf
        req2.initial_pose.position.x = float(x)
        req2.initial_pose.position.y = float(y)
        req2.initial_pose.position.z = float(z)
        req2.initial_pose.orientation.w = 1.0
        f = self._spn.call_async(req2)
        rclpy.spin_until_future_complete(self, f, timeout_sec=10.0)
        return f.done() and f.result().success

    def settle_and_grab(self, wait=WAIT_STABLE_S):
        deadline = time.time() + wait
        while time.time() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
        self._raw = self._viz = None
        deadline2 = time.time() + 4.0
        while (self._raw is None or self._viz is None) and time.time() < deadline2:
            rclpy.spin_once(self, timeout_sec=0.1)
        # capture last seen label across a few spins
        labels = []
        for _ in range(6):
            rclpy.spin_once(self, timeout_sec=0.3)
            if self._label:
                labels.append(self._label)
        return self._raw, self._viz, (labels[-1] if labels else "unknown")


def build_contact_sheet(results):
    """results: list of dicts with raw(np), label(str), gt, exp, name."""
    pad, header_h, label_h = 12, 36, 64
    thumbs = [r for r in results if r["raw"] is not None]
    if not thumbs:
        print("No frames to build contact sheet."); return
    w = thumbs[0]["raw"].shape[1]
    h = thumbs[0]["raw"].shape[0]
    n = len(thumbs)
    sheet_w = n * w + (n + 1) * pad
    sheet_h = header_h + h + label_h + 2 * pad
    sheet = PIL.new("RGB", (sheet_w, sheet_h), (24, 22, 20))
    draw = ImageDraw.Draw(sheet)
    draw.text((pad, 10),
              "Gazebo terrain zones — ExoMy rover camera + DINOv2 ViT-S classification",
              fill=(235, 235, 235))
    for i, r in enumerate(thumbs):
        x0 = pad + i * (w + pad)
        y0 = header_h + pad
        sheet.paste(PIL.fromarray(r["raw"]), (x0, y0))
        pred = r["label"].split(":")[0]
        correct = pred == r["gt"]
        bar_col = (60, 150, 60) if correct else (170, 60, 60)
        ly = y0 + h
        draw.rectangle([(x0, ly), (x0 + w, ly + label_h)], fill=bar_col)
        draw.text((x0 + 6, ly + 4),  f"{r['name']}", fill=(255, 255, 255))
        draw.text((x0 + 6, ly + 22), f"GT: {r['gt']}  pred: {r['label']}",
                  fill=(255, 255, 255))
        draw.text((x0 + 6, ly + 40), f"{'OK' if correct else 'MISS'} | {r['exp']}",
                  fill=(255, 255, 255))
    out = os.path.join(OUT_DIR, "gazebo_zones_contact_sheet.png")
    sheet.save(out)
    print(f"\nContact sheet → {out}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rclpy.init()
    node = DemoCapture()
    print("Waiting for /robot_description...")
    if not node.wait_urdf():
        print("ERROR: no URDF on /robot_description — is the launch running?")
        return

    results = []
    for z in ZONES:
        print(f"\n--- {z['name']} ({z['x']}, {z['y']}) GT={z['gt']} ---")
        ok = node.teleport(z["x"], z["y"])
        print(f"  spawn: {'ok' if ok else 'FAILED'}")
        raw, viz, label = node.settle_and_grab()
        print(f"  DINOv2: {label}")
        if raw is not None:
            PIL.fromarray(raw).save(os.path.join(OUT_DIR, f"{z['name']}_raw.png"))
        if viz is not None:
            PIL.fromarray(viz).save(os.path.join(OUT_DIR, f"{z['name']}_viz.png"))
        print(f"  saved raw={raw is not None} viz={viz is not None}")
        results.append({"raw": raw, "label": label, "gt": z["gt"],
                        "exp": z["exp"], "name": z["name"]})

    build_contact_sheet(results)
    node.destroy_node()
    rclpy.shutdown()
    print(f"\nDone. All frames in: {OUT_DIR}")


if __name__ == "__main__":
    main()
