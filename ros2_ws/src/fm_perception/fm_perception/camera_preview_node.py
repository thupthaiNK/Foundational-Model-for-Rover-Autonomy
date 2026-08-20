#!/usr/bin/env python3
"""
Purpose: Publish a bandwidth-cheap JPEG preview of the rover camera so RViz2 on a
         laptop can show a smooth live feed over the Pi's WiFi hotspot during a
         field test.

         The raw camera runs at 640x480 RGB888 15 fps, which is about 110 Mbps
         uncompressed -- far past what the hotspot carries, and recording it once
         produced an 11 GB bag for a 573 s run. This node drops frames to a target
         rate BEFORE encoding anything, then JPEG-encodes what survives, so the
         cost is set by target_fps and not by the camera's rate.

         It exists alongside, not instead of, /terrain_classified_image. That topic
         carries the frame each DINOv2 verdict was actually made on, at the ~0.5 Hz
         the full stack manages on a Pi; this one answers the different question of
         where the rover is looking right now. RViz2 shows both side by side.

         Off by default in the launch file. The Pi has 4 cores and 13 nodes
         competing for them, so anything added here is taken from inference. Run
         the stack once with use_preview:=false and once with true and compare
         /inference_latency_ms before trusting field numbers against the report's.

Inputs:  /camera/image_raw   (sensor_msgs/Image, rgb8)
Outputs: /camera/preview/compressed  (sensor_msgs/CompressedImage, jpeg)
             -- the base topic name /camera/preview is what an RViz2 Image display
                is pointed at, with transport set to "compressed".

Parameters:
    target_fps    (float, default 5.0)  frames per second to publish; <= 0 passes all
    jpeg_quality  (int,   default 80)   matches encode_jpeg() in dinov2_terrain_node

How to run:
    source /opt/ros/humble/setup.bash && source ros2_ws/install/setup.bash
    ros2 run fm_perception camera_preview_node.py --ros-args -p target_fps:=5.0

    # or, as part of the full rover stack
    ros2 launch exomy_ros2 real_hardware_deployment.launch.py use_preview:=true

Project: Onboard Visual Foundation Models for Mars Terrain Perception and Rover Navigation
"""

import io

import numpy as np
import rclpy
from PIL import Image as PILImage
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import CompressedImage, Image


class CameraPreviewNode(Node):

    def __init__(self) -> None:
        super().__init__("camera_preview_node")

        self.declare_parameter("target_fps", 5.0)
        self.declare_parameter("jpeg_quality", 80)

        self.target_fps = float(self.get_parameter("target_fps").value)
        self.jpeg_quality = int(self.get_parameter("jpeg_quality").value)

        # Minimum wall-clock gap between published frames. A target of 0 or less
        # disables throttling entirely, which is only sensible on a wired link.
        self.min_period_s = 1.0 / self.target_fps if self.target_fps > 0.0 else 0.0

        # Best-effort with depth 1 on both sides. A preview frame that arrives late
        # is worthless -- the operator wants the current view, not a replay of a
        # queue -- and reliable delivery over a marginal field hotspot would stall
        # the publisher rather than drop, which is the opposite of what is wanted.
        qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
        )

        self.sub = self.create_subscription(
            Image, "/camera/image_raw", self._image_callback, qos
        )
        self.pub = self.create_publisher(
            CompressedImage, "/camera/preview/compressed", qos
        )

        self._last_pub_s: float | None = None
        self.n_received = 0
        self.n_published = 0
        self.create_timer(10.0, self._report)

        self.get_logger().info(
            f"camera_preview_node ready | target_fps={self.target_fps:.1f} "
            f"| quality={self.jpeg_quality} | -> /camera/preview/compressed"
        )

    def _image_callback(self, msg: Image) -> None:
        self.n_received += 1

        # Throttle first. Everything below this point costs CPU that DINOv2 needs,
        # so a dropped frame must cost nothing but the callback itself.
        now_s = self.get_clock().now().nanoseconds * 1e-9
        if self._last_pub_s is not None and self.min_period_s > 0.0:
            if now_s - self._last_pub_s < self.min_period_s:
                return

        rgb = self._to_rgb_array(msg)
        if rgb is None:
            return

        buf = io.BytesIO()
        PILImage.fromarray(rgb, "RGB").save(
            buf, format="JPEG", quality=self.jpeg_quality
        )

        out = CompressedImage()
        # Carry the camera's own stamp through rather than restamping. The preview
        # is then on the same clock as every other topic in the bag, so a frame can
        # be lined up against the verdict, scan and IMU sample of that instant.
        out.header = msg.header
        out.format = "jpeg"
        out.data = buf.getvalue()
        self.pub.publish(out)

        self._last_pub_s = now_s
        self.n_published += 1

    def _to_rgb_array(self, msg: Image) -> np.ndarray | None:
        """Return an (h, w, 3) uint8 RGB array, or None if the encoding is unusable.

        camera_ros is configured for RGB888 in the deployment launch file, which
        arrives as encoding "rgb8". bgr8 is accepted as well so the node is still
        useful against a bag or a different camera driver. Anything else -- Bayer
        in particular, which is what /dev/video0 offers on this rover without
        libcamera -- is refused loudly rather than reshaped into garbage.
        """
        enc = msg.encoding.lower()
        if enc not in ("rgb8", "bgr8"):
            self.get_logger().error(
                f"unsupported encoding '{msg.encoding}'; expected rgb8 or bgr8",
                throttle_duration_sec=10.0,
            )
            return None

        arr = np.frombuffer(msg.data, dtype=np.uint8)
        expected = msg.height * msg.width * 3
        if arr.size < expected:
            self.get_logger().error(
                f"short image buffer: {arr.size} bytes for {msg.width}x{msg.height}",
                throttle_duration_sec=10.0,
            )
            return None

        # Reshape via step, not width, so a driver that pads rows does not shear
        # the image. step is bytes per row and may exceed width * 3.
        row_bytes = msg.step if msg.step >= msg.width * 3 else msg.width * 3
        if arr.size < msg.height * row_bytes:
            row_bytes = msg.width * 3
        arr = arr[: msg.height * row_bytes].reshape(msg.height, row_bytes)
        rgb = arr[:, : msg.width * 3].reshape(msg.height, msg.width, 3)

        if enc == "bgr8":
            rgb = rgb[:, :, ::-1]

        return np.ascontiguousarray(rgb)

    def _report(self) -> None:
        if self.n_received == 0:
            self.get_logger().warn("no camera frames received yet")
            return
        self.get_logger().info(
            f"preview: {self.n_published} published / {self.n_received} received"
        )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = CameraPreviewNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
