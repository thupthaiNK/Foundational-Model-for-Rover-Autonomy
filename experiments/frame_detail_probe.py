#!/usr/bin/env python3
"""
Purpose: Print the frame-detail score of whatever the rover's camera is
         seeing right now, next to the threshold the terrain node rejects
         frames below. Covering the lens and reading the number is the only
         way to tell "the blank-frame gate is not running" apart from "the
         gate is running and this frame is simply not blank" -- a finger held
         against the lens still passes red light through the flesh and is not
         a dark frame at all.

         Reads the same fm_perception.frame_quality functions the deployed
         node uses, so the number printed here is the number the gate sees.
         It only subscribes, it never publishes, so it is safe to run beside
         a live dinov2_terrain_node.

Inputs:  /camera/image_raw (sensor_msgs/Image, rgb8 or bgr8)
Outputs: one line per frame on stdout: detail score, verdict, and the min,
         max and mean pixel values that explain the score.
How to run:
    # laptop
    scp experiments/frame_detail_probe.py pi@172.20.10.13:~/ros2_ws/
    # container (camera_node must already be publishing)
    source /ws/install/setup.bash
    python3 /ws/frame_detail_probe.py
Project: Onboard Visual Foundation Models for Mars Terrain Perception and Rover Navigation
"""

import sys

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image

try:
    from fm_perception.frame_quality import DEFAULT_MIN_DETAIL, frame_detail
except ImportError:
    # Older installed builds have no frame_quality module at all. Say so
    # loudly rather than silently measuring with a local copy, because
    # "the module is missing" is itself the answer we may be looking for.
    print("fm_perception.frame_quality NOT IMPORTABLE from the installed build.")
    print("That means the blank-frame gate is not in the running node.")
    print("Rebuild on the Pi:  cd /ws && colcon build --packages-select fm_perception")
    sys.exit(3)


class FrameDetailProbe(Node):
    def __init__(self):
        super().__init__("frame_detail_probe")
        self.declare_parameter("min_frame_detail", DEFAULT_MIN_DETAIL)
        self.threshold = (
            self.get_parameter("min_frame_detail").get_parameter_value().double_value
        )
        self.count = 0
        self.create_subscription(Image, "/camera/image_raw", self._on_image, 1)
        self.get_logger().info(
            f"probing /camera/image_raw | reject below detail {self.threshold}"
        )

    def _on_image(self, msg):
        arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(
            msg.height, msg.width, -1
        )
        detail = frame_detail(arr)
        self.count += 1
        verdict = "REJECT (blank)" if detail < self.threshold else "accept"
        print(
            f"[{self.count:4d}] detail {detail:8.3f}  {verdict:15s} "
            f"min {int(arr.min()):3d}  max {int(arr.max()):3d}  "
            f"mean {arr.mean():6.1f}  {msg.encoding} {msg.width}x{msg.height}",
            flush=True,
        )


def main():
    rclpy.init()
    node = FrameDetailProbe()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        # Ctrl+C can already have torn the context down, and calling shutdown
        # a second time raises RCLError over the top of a clean exit.
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
