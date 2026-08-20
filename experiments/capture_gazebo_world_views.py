"""
Purpose: Capture reproducible views of the Gazebo Mars world for the thesis. The
         world is the basis of every simulation result in Chapter 4, but no figure
         in the thesis has ever shown it. Rather than driving the Gazebo GUI by
         hand, this spawns a temporary camera-sensor model at scripted poses and
         saves what it renders, so the same figure can be regenerated exactly.
Inputs:  A running Gazebo with mars_terrain.world loaded (gzserver is enough, no GUI).
Outputs: docs/figures/gazebo_world_views/*.png
How to run (two terminals, from the repo root):
    # terminal 1
    source /opt/ros/humble/setup.bash && source ros2_ws/install/setup.bash
    gzserver simulation/worlds/mars_terrain.world -s libgazebo_ros_init.so \
        -s libgazebo_ros_factory.so
    # terminal 2
    source /opt/ros/humble/setup.bash && source ros2_ws/install/setup.bash
    python3 experiments/capture_gazebo_world_views.py
Project: Onboard Visual Foundation Models for Mars Terrain Perception and Rover Navigation
"""

import os
import time

import rclpy
from rclpy.node import Node
from gazebo_msgs.srv import SpawnEntity, DeleteEntity
from sensor_msgs.msg import Image
import numpy as np
import cv2

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "docs", "figures",
                       "gazebo_world_views")

# name, pose "x y z roll pitch yaw", horizontal FOV in radians, image size
# Poses are derived from the world file's own zone centres: soil (-7.5, 6.0),
# bedrock (7.5, 6.0), sand (-7.5, -6.0), rock (7.5, -6.0) on a 30x24 m base.
VIEWS = [
    # Straight down, so the quadrant layout reads as a plan view.
    ("world_overhead", "0 0 31 0 1.5707 0", 1.15, (1600, 1280)),
    # Low oblique from the sand corner, framing the whole base with the rock
    # field visible at the far side.
    ("world_oblique", "-19 -16 6 0 0.236 0.700", 0.95, (1600, 1100)),
    # Rover-eye height inside the rock quadrant. This is the geometry behind the
    # sphere-primitive versus mesh finding in Chapter 4.
    ("rock_zone_close", "3.0 -3.0 0.6 0 0.055 -0.588", 1.30, (1600, 1100)),
    # Rover-eye view across a zone boundary, showing the texture tiling that
    # Chapter 4 traces part of the sim-to-real gap to.
    ("zone_boundary", "-2.0 2.0 0.6 0 0.05 -0.785", 1.30, (1600, 1100)),
]

CAMERA_SDF = """<?xml version="1.0"?>
<sdf version="1.6">
  <model name="{name}">
    <static>true</static>
    <pose>{pose}</pose>
    <link name="link">
      <sensor name="cam" type="camera">
        <always_on>1</always_on>
        <update_rate>10</update_rate>
        <camera>
          <horizontal_fov>{fov}</horizontal_fov>
          <image><width>{w}</width><height>{h}</height><format>R8G8B8</format></image>
          <clip><near>0.05</near><far>500</far></clip>
        </camera>
        <plugin name="cam_plugin_{name}" filename="libgazebo_ros_camera.so">
          <ros><namespace>/shot</namespace></ros>
          <camera_name>{name}</camera_name>
        </plugin>
      </sensor>
    </link>
  </model>
</sdf>
"""


class Shooter(Node):
    def __init__(self):
        super().__init__("gazebo_world_shooter")
        self.spawn = self.create_client(SpawnEntity, "/spawn_entity")
        self.delete = self.create_client(DeleteEntity, "/delete_entity")
        for c, n in ((self.spawn, "/spawn_entity"), (self.delete, "/delete_entity")):
            if not c.wait_for_service(timeout_sec=20.0):
                raise RuntimeError(f"{n} unavailable. Is gzserver running with "
                                   f"libgazebo_ros_factory.so?")
        self.frame = None

    def _cb(self, msg):
        self.frame = msg

    def shoot(self, name, pose, fov, size):
        w, h = size
        sdf = CAMERA_SDF.format(name=name, pose=pose, fov=fov, w=w, h=h)
        req = SpawnEntity.Request()
        req.name, req.xml = name, sdf
        fut = self.spawn.call_async(req)
        rclpy.spin_until_future_complete(self, fut, timeout_sec=30.0)

        topic = f"/shot/{name}/image_raw"
        self.frame = None
        sub = self.create_subscription(Image, topic, self._cb, 10)
        deadline = time.time() + 25.0
        # Discard the first frames: the renderer needs a moment to load textures,
        # and an early grab comes back untextured.
        seen = 0
        while time.time() < deadline:
            rclpy.spin_once(self, timeout_sec=0.2)
            if self.frame is not None:
                seen += 1
                if seen >= 15:
                    break
                self.frame = None
        ok = self.frame is not None
        if ok:
            os.makedirs(OUT_DIR, exist_ok=True)
            m = self.frame
            arr = np.frombuffer(m.data, dtype=np.uint8).reshape(m.height, m.width, 3)
            img = arr[:, :, ::-1] if m.encoding == "rgb8" else arr
            path = os.path.normpath(os.path.join(OUT_DIR, name + ".png"))
            cv2.imwrite(path, img)
            print(f"  wrote {path}  ({img.shape[1]}x{img.shape[0]})")
        else:
            print(f"  FAILED: no frame on {topic}")
        self.destroy_subscription(sub)

        dreq = DeleteEntity.Request()
        dreq.name = name
        dfut = self.delete.call_async(dreq)
        rclpy.spin_until_future_complete(self, dfut, timeout_sec=20.0)
        time.sleep(1.0)
        return ok


def main():
    rclpy.init()
    node = Shooter()
    results = {}
    try:
        for name, pose, fov, size in VIEWS:
            print(f"capturing {name} at pose ({pose})...")
            results[name] = node.shoot(name, pose, fov, size)
    finally:
        node.destroy_node()
        rclpy.shutdown()
    print("\nsummary:", {k: ("ok" if v else "FAILED") for k, v in results.items()})


if __name__ == "__main__":
    main()
