#!/usr/bin/env python3
"""
Purpose: ROS2 node that runs BLIP-2 VQA on camera images at a configurable
         interval (default 5s). Publishes scene description and traversability
         assessment to /scene_description and /traversability.
         Designed as the "slow path" in the dual-process cascade pipeline:
         CLIP runs every frame; BLIP-2 runs every 5s or when CLIP is uncertain.
Inputs:  /camera/image_raw (sensor_msgs/Image)
         /terrain_classification (std_msgs/String) — from clip_terrain_node
Outputs: /scene_description (std_msgs/String)  — natural language description
         /traversability (std_msgs/String)      — "safe" | "caution" | "hazard"
How to run:
    ros2 run fm_perception blip2_scene_node
Project: Onboard Visual Foundation Models for Mars Terrain Perception and Rover Navigation
"""

import time

import numpy as np
import rclpy
import torch
from PIL import Image as PILImage
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from transformers import Blip2ForConditionalGeneration, Blip2Processor

# Traversability question asked every interval
TERRAIN_QUESTION = (
    "Question: Is this Mars terrain safe for a rover to drive on? "
    "Describe what you see and whether it is safe, requires caution, or is a hazard. Answer:"
)

# Simple keyword-based traversability parser
HAZARD_KEYWORDS  = ["hazard", "dangerous", "unsafe", "obstacle", "boulder", "cliff", "crater"]
CAUTION_KEYWORDS = ["caution", "rocky", "rough", "uneven", "slope", "rock"]
SAFE_KEYWORDS    = ["safe", "clear", "flat", "smooth", "soil", "sand"]


def parse_traversability(text: str) -> str:
    text_lower = text.lower()
    for kw in HAZARD_KEYWORDS:
        if kw in text_lower:
            return "hazard"
    for kw in CAUTION_KEYWORDS:
        if kw in text_lower:
            return "caution"
    for kw in SAFE_KEYWORDS:
        if kw in text_lower:
            return "safe"
    return "caution"  # conservative default


class Blip2SceneNode(Node):

    def __init__(self):
        super().__init__("blip2_scene_node")

        # ── Parameters ─────────────────────────────────────────────────
        self.declare_parameter("device", "cpu")
        self.declare_parameter("interval_sec", 5.0)
        self.declare_parameter("clip_uncertainty_threshold", 0.40)
        self.declare_parameter("use_int8", True)

        device_str = self.get_parameter("device").get_parameter_value().string_value
        self.interval = (
            self.get_parameter("interval_sec").get_parameter_value().double_value
        )
        self.clip_threshold = (
            self.get_parameter("clip_uncertainty_threshold")
            .get_parameter_value()
            .double_value
        )
        use_int8 = self.get_parameter("use_int8").get_parameter_value().bool_value

        self.device = (
            "cuda"
            if device_str == "cuda" and torch.cuda.is_available()
            else "cpu"
        )
        self.get_logger().info(f"Device: {self.device}")

        # ── Load BLIP-2 ─────────────────────────────────────────────────
        model_id = "Salesforce/blip2-opt-2.7b"
        self.get_logger().info(f"Loading BLIP-2 ({model_id})...")
        self.processor = Blip2Processor.from_pretrained(model_id)

        dtype = torch.int8 if use_int8 else torch.float16
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map={"": self.device},
        )
        self.model.eval()
        self.get_logger().info("BLIP-2 loaded")

        # ── State ───────────────────────────────────────────────────────
        self.latest_image: PILImage.Image = None
        self.last_clip_conf: float = 1.0
        self.last_clip_label: str = "unknown"

        # ── ROS2 subscriptions ──────────────────────────────────────────
        self.sub_img = self.create_subscription(
            Image, "/camera/image_raw", self._image_callback, 1
        )
        self.sub_clip = self.create_subscription(
            String, "/terrain_classification", self._clip_callback, 10
        )

        # ── Publishers ──────────────────────────────────────────────────
        self.pub_desc  = self.create_publisher(String, "/scene_description", 10)
        self.pub_trav  = self.create_publisher(String, "/traversability",    10)

        # ── Timer: run BLIP-2 every interval_sec ───────────────────────
        self.create_timer(self.interval, self._run_blip2)

        self.get_logger().info(
            f"blip2_scene_node ready — interval {self.interval}s, "
            f"clip uncertainty threshold {self.clip_threshold}"
        )

    def _image_callback(self, msg: Image) -> None:
        """Store the latest image for periodic BLIP-2 inference."""
        np_arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(
            (msg.height, msg.width, -1)
        )
        if msg.encoding in ("bgr8", "bgra8"):
            np_arr = np_arr[:, :, :3][:, :, ::-1].copy()
        else:
            np_arr = np_arr[:, :, :3].copy()
        self.latest_image = PILImage.fromarray(np_arr, "RGB")

    def _clip_callback(self, msg: String) -> None:
        """Track CLIP confidence to trigger BLIP-2 when uncertain."""
        parts = msg.data.split(":")
        if len(parts) == 2:
            self.last_clip_label = parts[0]
            try:
                self.last_clip_conf = float(parts[1])
            except ValueError:
                pass

        # Immediately trigger BLIP-2 if CLIP is uncertain
        if self.last_clip_conf < self.clip_threshold:
            self.get_logger().info(
                f"CLIP uncertain ({self.last_clip_conf:.2f}) — triggering BLIP-2"
            )
            self._run_blip2()

    def _run_blip2(self) -> None:
        """Run BLIP-2 on the latest stored image."""
        if self.latest_image is None:
            return

        t0 = time.perf_counter()
        inputs = self.processor(
            self.latest_image, TERRAIN_QUESTION, return_tensors="pt"
        ).to(self.device, torch.float16)

        with torch.no_grad():
            output = self.model.generate(**inputs, max_new_tokens=60)

        elapsed_ms = (time.perf_counter() - t0) * 1000
        description = self.processor.decode(output[0], skip_special_tokens=True).strip()
        traversability = parse_traversability(description)

        # Publish
        desc_msg = String()
        desc_msg.data = description
        self.pub_desc.publish(desc_msg)

        trav_msg = String()
        trav_msg.data = traversability
        self.pub_trav.publish(trav_msg)

        self.get_logger().info(
            f"BLIP-2 [{elapsed_ms:.0f}ms] → {traversability}: {description[:80]}..."
        )


def main(args=None):
    rclpy.init(args=args)
    node = Blip2SceneNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
