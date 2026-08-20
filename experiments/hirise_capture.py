"""
Purpose: Teleport rover to 5 positions in HiRISE terrain and capture one
         terrain_viz frame at each position for thesis figures.
How to run:
    source /opt/ros/humble/setup.bash && source ros2_ws/install/setup.bash
    python3 experiments/hirise_capture.py
Project: Onboard Visual Foundation Models for Mars Terrain Perception and Rover Navigation
"""

import os, time, cv2, numpy as np, rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import String
from gazebo_msgs.srv import DeleteEntity, SpawnEntity

POSITIONS = [
    {"label": "p01_origin",      "x":  0.0, "y":  0.0, "z": 5.0},  # confirmed good
    {"label": "p02_southwest",   "x": -6.0, "y": -6.0, "z": 5.0},  # confirmed good ridges
    {"label": "p03_southeast",   "x":  6.0, "y": -8.0, "z": 5.0},  # confirmed good ridge line
    {"label": "p04_west",        "x": -8.0, "y":  0.0, "z": 5.0},  # unexplored west ridge
    {"label": "p05_south",       "x":  0.0, "y": -10.0, "z": 5.0}, # south edge
    {"label": "p06_northwest",   "x": -5.0, "y":  6.0, "z": 5.0},  # northwest quadrant
    {"label": "p07_northeast",   "x":  4.0, "y":  6.0, "z": 5.0},  # northeast (safer than east)
    {"label": "p08_ssw",         "x": -3.0, "y": -9.0, "z": 5.0},  # south-southwest
    {"label": "p09_ene",         "x":  5.0, "y":  2.0, "z": 5.0},  # gentle east (closer than 8m)
    {"label": "p10_center_north","x": -2.0, "y":  5.0, "z": 5.0},  # slightly north of center
]

OUT_DIR = "docs/figures"
os.makedirs(OUT_DIR, exist_ok=True)


class HiRISECapture(Node):
    def __init__(self):
        super().__init__("hirise_capture")
        self._latest_viz: Image | None = None
        self._latest_label: str = ""
        self._urdf_xml: str | None = None

        self.create_subscription(Image, "/terrain_viz", self._viz_cb, 10)
        self.create_subscription(String, "/terrain_classification", self._cls_cb, 10)
        qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(String, "/robot_description", self._urdf_cb, qos)

        self._del = self.create_client(DeleteEntity, "/delete_entity")
        self._spn = self.create_client(SpawnEntity, "/spawn_entity")

    def _viz_cb(self, msg): self._latest_viz = msg
    def _cls_cb(self, msg): self._latest_label = msg.data
    def _urdf_cb(self, msg): self._urdf_xml = msg.data

    def _teleport(self, x, y, z):
        # Wait for URDF
        deadline = time.time() + 8.0
        while self._urdf_xml is None and time.time() < deadline:
            rclpy.spin_once(self, timeout_sec=0.2)
        if self._urdf_xml is None:
            self.get_logger().error("No robot_description — cannot teleport")
            return False

        # Delete
        self._del.wait_for_service(timeout_sec=5.0)
        req = DeleteEntity.Request(); req.name = "exomy"
        fut = self._del.call_async(req)
        rclpy.spin_until_future_complete(self, fut, timeout_sec=10.0)
        time.sleep(1.5)

        # Spawn
        self._spn.wait_for_service(timeout_sec=5.0)
        req2 = SpawnEntity.Request()
        req2.name = "exomy"; req2.xml = self._urdf_xml
        req2.initial_pose.position.x = x
        req2.initial_pose.position.y = y
        req2.initial_pose.position.z = z
        req2.initial_pose.orientation.w = 1.0
        fut2 = self._spn.call_async(req2)
        rclpy.spin_until_future_complete(self, fut2, timeout_sec=10.0)
        time.sleep(3.0)  # settle on terrain
        return True

    def _save_frame(self, label):
        # Spin until fresh frame arrives
        deadline = time.time() + 15.0
        while time.time() < deadline:
            rclpy.spin_once(self, timeout_sec=0.3)
            if self._latest_viz is not None:
                break
        if self._latest_viz is None:
            self.get_logger().warn(f"No frame for {label}")
            return
        msg = self._latest_viz
        img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
        if msg.encoding == "rgb8":
            img = img[:, :, ::-1]  # RGB → BGR for cv2.imwrite
        path = os.path.join(OUT_DIR, f"hirise_{label}.jpg")
        cv2.imwrite(path, img)
        print(f"  Saved: {path}  |  DINOv2: {self._latest_label}")


def main():
    rclpy.init()
    node = HiRISECapture()

    print("\n=== HiRISE 5-position capture ===")
    print("Warming up 5s...")
    t = time.time() + 5
    while time.time() < t:
        rclpy.spin_once(node, timeout_sec=0.3)

    for pos in POSITIONS:
        print(f"\n[{pos['label']}] teleporting to ({pos['x']}, {pos['y']}, z={pos['z']})...")
        ok = node._teleport(pos["x"], pos["y"], pos["z"])
        if not ok:
            continue
        print("  Waiting 5s for camera to stabilise...")
        t = time.time() + 5
        while time.time() < t:
            rclpy.spin_once(node, timeout_sec=0.3)
        node._save_frame(pos["label"])

    print(f"\nDone — frames saved in {OUT_DIR}/")
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
