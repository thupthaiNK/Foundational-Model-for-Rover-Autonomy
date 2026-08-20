#!/usr/bin/env python3
"""
Purpose: ROS2 node that subscribes to /camera/image_raw, classifies terrain
         using CLIP ViT-B/32 with v9 hybrid ensemble prompts, and publishes
         terrain label + confidence to /terrain_classification.
Inputs:  /camera/image_raw (sensor_msgs/Image)
Outputs: /terrain_classification (std_msgs/String)  format: "label:confidence"
         /terrain_viz (sensor_msgs/Image)            annotated image for debugging
How to run:
    source /opt/ros/humble/setup.bash
    cd ros2_ws && colcon build --packages-select fm_perception
    source install/setup.bash
    ros2 run fm_perception clip_terrain_node
Project: Onboard Visual Foundation Models for Mars Terrain Perception and Rover Navigation
"""

import time

import clip
import numpy as np
import rclpy
import torch
from PIL import Image as PILImage
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String

# ── Best prompts from Experiment 2 — v9 hybrid ensemble ───────────────────────
CLIP_PROMPTS = [
    # soil: NAVCAM prefix (v6 gave 96.7% soil)
    "a Mars rover NAVCAM grayscale image of loose soil and fine regolith on the ground",
    # bedrock: texture+scale (v4 gave 91.3% bedrock)
    "Mars rocky ground where the floor itself is made of solid flat rock",
    # sand: NAVCAM prefix + v2
    "a Mars rover NAVCAM grayscale image of smooth sandy terrain and wind-blown sand",
    # big_rock: isolation emphasis
    "isolated large individual boulders and rocks scattered on Mars soil",
]
CLASS_NAMES = ["soil", "bedrock", "sand", "big_rock"]
CONFIDENCE_THRESHOLD = 0.40  # below this → publish "uncertain"


class ClipTerrainNode(Node):

    def __init__(self):
        super().__init__("clip_terrain_node")

        # ── Parameters ─────────────────────────────────────────────────
        self.declare_parameter("device", "cpu")
        self.declare_parameter("confidence_threshold", CONFIDENCE_THRESHOLD)
        self.declare_parameter("publish_viz", True)

        device_str = self.get_parameter("device").get_parameter_value().string_value
        self.conf_threshold = (
            self.get_parameter("confidence_threshold")
            .get_parameter_value()
            .double_value
        )
        publish_viz = (
            self.get_parameter("publish_viz").get_parameter_value().bool_value
        )

        # ── Device ─────────────────────────────────────────────────────
        if device_str == "cuda" and torch.cuda.is_available():
            try:
                torch.zeros(1).cuda()
                self.device = "cuda"
            except Exception:
                self.device = "cpu"
        else:
            self.device = "cpu"
        self.get_logger().info(f"Using device: {self.device}")

        # ── Load CLIP ───────────────────────────────────────────────────
        self.get_logger().info("Loading CLIP ViT-B/32...")
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        self.model.eval()

        # Pre-encode text features (do once at startup — saves ~30% per inference)
        text_tokens = clip.tokenize(CLIP_PROMPTS).to(self.device)
        with torch.no_grad():
            self.txt_feat = self.model.encode_text(text_tokens)
            self.txt_feat /= self.txt_feat.norm(dim=-1, keepdim=True)
        self.get_logger().info("CLIP loaded — text features pre-encoded")

        # ── ROS2 publishers / subscribers ──────────────────────────────
        self.sub = self.create_subscription(
            Image, "/camera/image_raw", self._image_callback, 1
        )
        self.pub = self.create_publisher(String, "/terrain_classification", 10)
        if publish_viz:
            self.pub_viz = self.create_publisher(Image, "/terrain_viz", 1)
        else:
            self.pub_viz = None

        # ── Safety watchdog (from ExoMy motor_node pattern) ────────────
        # If no image received in 5 seconds → publish "unknown"
        self.last_image_time = self.get_clock().now()
        self.create_timer(5.0, self._watchdog_callback)

        # ── Stats ───────────────────────────────────────────────────────
        self.n_processed = 0
        self.total_ms = 0.0

        self.get_logger().info(
            "clip_terrain_node ready — "
            f"subscribed to /camera/image_raw, "
            f"publishing to /terrain_classification"
        )

    def _image_callback(self, msg: Image) -> None:
        """Process each incoming camera frame."""
        self.last_image_time = self.get_clock().now()

        # Convert ROS Image → PIL (avoid cv_bridge NumPy 2.x incompatibility)
        np_arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(
            (msg.height, msg.width, -1)
        )
        if msg.encoding in ("bgr8", "bgra8"):
            np_arr = np_arr[:, :, :3][:, :, ::-1].copy()
        else:
            np_arr = np_arr[:, :, :3].copy()
        pil_img = PILImage.fromarray(np_arr, "RGB")

        # CLIP inference
        t0 = time.perf_counter()
        label, confidence, probs = self._classify(pil_img)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        # Update stats
        self.n_processed += 1
        self.total_ms += elapsed_ms
        avg_ms = self.total_ms / self.n_processed

        # Publish terrain label
        out_msg = String()
        if confidence >= self.conf_threshold:
            out_msg.data = f"{label}:{confidence:.3f}"
        else:
            out_msg.data = f"uncertain:{confidence:.3f}"
        self.pub.publish(out_msg)

        # Log every 10 frames
        if self.n_processed % 10 == 0:
            self.get_logger().info(
                f"[{self.n_processed}] {out_msg.data}  "
                f"| {elapsed_ms:.0f}ms  avg {avg_ms:.0f}ms"
            )

        # Publish annotated image for debugging
        if self.pub_viz is not None:
            self._publish_viz(np_arr, label, confidence, probs)

    def _classify(self, pil_img: PILImage.Image):
        """Run CLIP on a PIL image. Returns (label, confidence, all_probs)."""
        img_tensor = self.preprocess(pil_img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            img_feat = self.model.encode_image(img_tensor)
            img_feat /= img_feat.norm(dim=-1, keepdim=True)
            probs = (100.0 * img_feat @ self.txt_feat.T).softmax(dim=-1)[0]

        pred_idx = int(probs.argmax())
        confidence = float(probs[pred_idx])
        return CLASS_NAMES[pred_idx], confidence, probs.cpu().numpy()

    def _publish_viz(self, rgb_arr: np.ndarray, label: str,
                     confidence: float, probs: np.ndarray) -> None:
        """Draw classification result on image and publish (PIL-only, no cv_bridge)."""
        from PIL import ImageDraw

        colour_map = {
            "soil":      (139, 90,  43),
            "bedrock":   (160, 150, 140),
            "sand":      (210, 190, 120),
            "big_rock":  (80,  80,  80),
            "uncertain": (200, 50,  50),
        }
        colour = colour_map.get(label, (200, 50, 50))

        pil_viz = PILImage.fromarray(rgb_arr.copy(), "RGB")
        draw = ImageDraw.Draw(pil_viz)
        h, w = rgb_arr.shape[:2]

        # Header bar + label text
        draw.rectangle([(0, 0), (w, 60)], fill=colour)
        draw.text((10, 15), f"{label} ({confidence:.2f})", fill=(255, 255, 255))

        # Per-class probability bars
        bar_h, bar_w = 15, max(1, w // len(CLASS_NAMES) - 5)
        for i, (name, prob) in enumerate(zip(CLASS_NAMES, probs)):
            x = i * (bar_w + 5) + 5
            y = h - 25
            filled = int(float(prob) * bar_w)
            draw.rectangle([(x, y), (x + bar_w, y + bar_h)], fill=(80, 80, 80))
            draw.rectangle([(x, y), (x + filled, y + bar_h)],
                           fill=colour_map.get(name, (150, 150, 150)))
            draw.text((x, h - 12), name[:3], fill=(220, 220, 220))

        # Convert PIL → ROS Image manually (avoids cv_bridge / NumPy 2.x issue)
        out_arr = np.asarray(pil_viz)
        viz_msg = Image()
        viz_msg.height = out_arr.shape[0]
        viz_msg.width  = out_arr.shape[1]
        viz_msg.encoding = "rgb8"
        viz_msg.is_bigendian = False
        viz_msg.step = out_arr.shape[1] * 3
        viz_msg.data = out_arr.tobytes()
        self.pub_viz.publish(viz_msg)

    def _watchdog_callback(self) -> None:
        """Publish 'unknown' if no image received for 5 seconds."""
        elapsed = (self.get_clock().now() - self.last_image_time).nanoseconds / 1e9
        if elapsed > 5.0:
            msg = String()
            msg.data = "unknown:0.000"
            self.pub.publish(msg)
            self.get_logger().warn(
                f"No image for {elapsed:.1f}s — publishing unknown"
            )


def main(args=None):
    rclpy.init(args=args)
    node = ClipTerrainNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
