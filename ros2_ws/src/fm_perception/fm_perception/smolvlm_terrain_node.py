#!/usr/bin/env python3
"""
Purpose: ROS2 node running SmolVLM-256M-Instruct as the "slow path" in the
         dual-process cascade pipeline.  Runs every 5s (or when CLIP is
         uncertain) and answers two questions in one forward pass:
           Q1 — terrain type (soil / bedrock / sand / rock)
           Q2 — traversability (safe / caution / hazard)
         SmolVLM-256M is the supervisor-recommended small VLM for RPi 4
         deployment: 256M params, ~2GB RAM, ~55s/img on PC CPU.
Inputs:  /camera/image_raw          (sensor_msgs/Image)
         /terrain_classification     (std_msgs/String) — from clip_terrain_node
Outputs: /smolvlm_terrain           (std_msgs/String) — "label:confidence"
         /scene_description         (std_msgs/String) — free-text answer
         /traversability            (std_msgs/String) — "safe" | "caution" | "hazard"
How to run:
    ros2 run fm_perception smolvlm_terrain_node
    # Parameters:
    ros2 run fm_perception smolvlm_terrain_node \
        --ros-args -p interval_sec:=5.0 -p device:=cpu
Project: Onboard Visual Foundation Models for Mars Terrain Perception and Rover Navigation
"""

import threading
import time

import numpy as np
import rclpy
import torch
from PIL import Image as PILImage
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from transformers import AutoProcessor, SmolVLMForConditionalGeneration

# ── Model ─────────────────────────────────────────────────────────────────────
MODEL_ID = "HuggingFaceTB/SmolVLM-256M-Instruct"

# ── Questions (sent as separate chat turns in one forward pass) ────────────────
Q_TERRAIN = (
    "What type of terrain is visible in this Mars image? "
    "Answer with exactly one word: soil, bedrock, sand, or rock."
)
Q_SAFETY = (
    "Is this Mars terrain safe for a rover to drive on? "
    "Answer in one word: safe, caution, or hazard."
)

# ── Terrain label parsing ─────────────────────────────────────────────────────
CLASS_NAMES = ["soil", "bedrock", "sand", "big_rock"]
KEYWORD_MAP = {
    "soil": 0, "dirt": 0, "regolith": 0, "dust": 0, "ground": 0,
    "bedrock": 1, "rock": 1, "stone": 1, "flat": 1, "rocky": 1,
    "sand": 2, "sandy": 2, "dune": 2,
    "big": 3, "boulder": 3, "boulders": 3, "large": 3,
}

# ── Traversability parsing ────────────────────────────────────────────────────
HAZARD_KW  = ["hazard", "dangerous", "unsafe", "obstacle", "boulder",
               "cliff", "crater", "impassable"]
CAUTION_KW = ["caution", "rocky", "rough", "uneven", "slope", "rocks",
               "bedrock", "challenging"]
SAFE_KW    = ["safe", "clear", "flat", "smooth", "soil", "sand", "traversable"]


def parse_terrain(text: str):
    low = text.lower().strip()
    for i, name in enumerate(CLASS_NAMES):
        if name in low:
            return i, name
    for word in low.split():
        w = word.strip(".,!?;:()")
        if w in KEYWORD_MAP:
            return KEYWORD_MAP[w], CLASS_NAMES[KEYWORD_MAP[w]]
    # Partial matches for compound words
    extra = {"rocky terrain": 1, "large rock": 3, "loose": 0, "sediment": 0,
             "sandy": 2, "dunes": 2}
    for phrase, cls in extra.items():
        if phrase in low:
            return cls, CLASS_NAMES[cls]
    return None, "uncertain"


def parse_traversability(text: str) -> str:
    low = text.lower()
    for kw in HAZARD_KW:
        if kw in low:
            return "hazard"
    for kw in CAUTION_KW:
        if kw in low:
            return "caution"
    for kw in SAFE_KW:
        if kw in low:
            return "safe"
    return "caution"  # conservative default


class SmolVLMTerrainNode(Node):

    def __init__(self):
        super().__init__("smolvlm_terrain_node")

        # ── Parameters ─────────────────────────────────────────────────
        self.declare_parameter("device", "cpu")
        self.declare_parameter("interval_sec", 5.0)
        self.declare_parameter("clip_uncertainty_threshold", 0.40)
        self.declare_parameter("model_id", MODEL_ID)

        device_str = self.get_parameter("device").get_parameter_value().string_value
        self.interval = self.get_parameter("interval_sec").get_parameter_value().double_value
        self.clip_threshold = (
            self.get_parameter("clip_uncertainty_threshold")
            .get_parameter_value().double_value
        )
        model_id = self.get_parameter("model_id").get_parameter_value().string_value

        self.device = (
            "cuda" if device_str == "cuda" and torch.cuda.is_available() else "cpu"
        )
        self.get_logger().info(f"Device: {self.device}  |  Model: {model_id}")

        # ── Load SmolVLM ────────────────────────────────────────────────
        self.get_logger().info("Loading SmolVLM processor...")
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.get_logger().info("Loading SmolVLM model (256M params, ~2GB RAM)...")
        t0 = time.perf_counter()
        self.model = SmolVLMForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
            device_map={"": self.device},
        )
        self.model.eval()
        load_s = time.perf_counter() - t0
        self.get_logger().info(f"SmolVLM loaded in {load_s:.1f}s")

        # ── State ───────────────────────────────────────────────────────
        self.latest_image: PILImage.Image = None
        self.last_clip_conf: float = 1.0
        self.last_clip_label: str = "unknown"
        self._busy = False  # prevent overlapping slow-path calls

        # ── Subscriptions ───────────────────────────────────────────────
        self.sub_img = self.create_subscription(
            Image, "/camera/image_raw", self._image_cb, 1
        )
        self.sub_clip = self.create_subscription(
            String, "/terrain_classification", self._clip_cb, 10
        )

        # ── Publishers ──────────────────────────────────────────────────
        self.pub_terrain = self.create_publisher(String, "/smolvlm_terrain", 10)
        self.pub_desc    = self.create_publisher(String, "/scene_description", 10)
        self.pub_trav    = self.create_publisher(String, "/traversability", 10)

        # ── Periodic timer ──────────────────────────────────────────────
        self.create_timer(self.interval, self._run_smolvlm)

        self.get_logger().info(
            f"smolvlm_terrain_node ready — "
            f"interval {self.interval}s, "
            f"CLIP uncertainty threshold {self.clip_threshold}"
        )

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def _image_cb(self, msg: Image) -> None:
        np_arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(
            (msg.height, msg.width, -1)
        )
        if msg.encoding in ("bgr8", "bgra8"):
            np_arr = np_arr[:, :, :3][:, :, ::-1].copy()
        else:
            np_arr = np_arr[:, :, :3].copy()
        self.latest_image = PILImage.fromarray(np_arr, "RGB")

    def _clip_cb(self, msg: String) -> None:
        parts = msg.data.split(":")
        if len(parts) == 2:
            self.last_clip_label = parts[0]
            try:
                self.last_clip_conf = float(parts[1])
            except ValueError:
                pass
        # Immediately trigger SmolVLM if CLIP is uncertain
        if self.last_clip_conf < self.clip_threshold and not self._busy:
            self.get_logger().info(
                f"CLIP uncertain ({self.last_clip_conf:.2f} < {self.clip_threshold}) "
                "— triggering SmolVLM"
            )
            self._run_smolvlm()

    # ── Inference ─────────────────────────────────────────────────────────────

    def _ask(self, image: PILImage.Image, question: str) -> str:
        """Run one VQA forward pass. Returns the model's answer string."""
        messages = [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}
        ]
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(text=prompt, images=[image], return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            out_ids = self.model.generate(**inputs, max_new_tokens=20, do_sample=False)
        input_len = inputs["input_ids"].shape[1]
        return self.processor.tokenizer.decode(
            out_ids[0][input_len:], skip_special_tokens=True
        ).strip()

    def _run_smolvlm(self) -> None:
        """Spawn a background thread for inference — does not block the spin loop."""
        if self.latest_image is None or self._busy:
            return
        threading.Thread(target=self._do_infer, daemon=True).start()

    def _do_infer(self) -> None:
        self._busy = True
        t0 = time.perf_counter()

        try:
            image = self.latest_image  # snapshot (avoid race with _image_cb)

            # Q1 — terrain type
            terrain_ans = self._ask(image, Q_TERRAIN)
            cls_idx, cls_name = parse_terrain(terrain_ans)

            # Q2 — traversability
            safety_ans = self._ask(image, Q_SAFETY)
            traversability = parse_traversability(safety_ans)

            elapsed_ms = (time.perf_counter() - t0) * 1000

            # Publish terrain label (format matches clip_terrain_node)
            terrain_msg = String()
            terrain_msg.data = f"{cls_name}:smolvlm"
            self.pub_terrain.publish(terrain_msg)

            # Publish combined scene description
            desc_msg = String()
            desc_msg.data = f"terrain={terrain_ans} | safety={safety_ans}"
            self.pub_desc.publish(desc_msg)

            # Publish traversability
            trav_msg = String()
            trav_msg.data = traversability
            self.pub_trav.publish(trav_msg)

            self.get_logger().info(
                f"SmolVLM [{elapsed_ms:.0f}ms] "
                f"terrain='{terrain_ans}' ({cls_name}) | "
                f"safety='{safety_ans}' → {traversability}"
            )

        except Exception as e:
            self.get_logger().error(f"SmolVLM inference error: {e}")
        finally:
            self._busy = False


def main(args=None):
    rclpy.init(args=args)
    node = SmolVLMTerrainNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
