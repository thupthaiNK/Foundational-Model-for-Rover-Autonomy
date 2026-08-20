#!/usr/bin/env python3
"""
Purpose:    Dual-process cascade perception pipeline for the ExoMy rover.
            Implements the "fast path / slow path" architecture (Paper #210):
              FAST PATH — CLIP runs on every camera frame (~136ms on PC CPU)
                → publishes terrain label + confidence
                → if confidence < threshold, flags for slow path
              SLOW PATH — SmolVLM-256M runs every 5s OR when CLIP is uncertain
                → publishes VQA terrain answer + safety assessment
                → triggers traversability decision
            Combined output: /traversability ("safe" | "caution" | "hazard")
Inputs:     /camera/image_raw (sensor_msgs/Image)
Outputs:    /terrain_classification  (std_msgs/String) — "label:confidence" (CLIP)
            /smolvlm_terrain         (std_msgs/String) — "label:smolvlm" (SmolVLM)
            /scene_description       (std_msgs/String) — "terrain=X | safety=Y"
            /traversability          (std_msgs/String) — "safe"|"caution"|"hazard"
            /terrain_viz             (sensor_msgs/Image) — annotated frame (debug)
How to run:
    ros2 run fm_perception cascade_pipeline
    ros2 run fm_perception cascade_pipeline \
        --ros-args -p smolvlm_interval_sec:=5.0 -p clip_threshold:=0.40
Project:    Onboard Visual Foundation Models for Mars Terrain Perception and Rover Navigation
Reference:  Dual-process pattern — Paper #210 (SAS on RPi 5): 88% fast / 12% VLM slow
"""

import threading
import time

import clip
import numpy as np
import rclpy
import torch
from PIL import Image as PILImage
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from transformers import AutoProcessor, SmolVLMForConditionalGeneration

# ── CLIP: v9 hybrid ensemble prompts (best overall from Experiment 2) ─────────
CLIP_PROMPTS = [
    "a Mars rover NAVCAM grayscale image of loose soil and fine regolith on the ground",
    "Mars rocky ground where the floor itself is made of solid flat rock",
    "a Mars rover NAVCAM grayscale image of smooth sandy terrain and wind-blown sand",
    "isolated large individual boulders and rocks scattered on Mars soil",
]
CLIP_CLASSES = ["soil", "bedrock", "sand", "big_rock"]

# ── SmolVLM questions ─────────────────────────────────────────────────────────
Q_TERRAIN  = ("What type of terrain is visible in this Mars image? "
              "Answer with exactly one word: soil, bedrock, sand, or rock.")
Q_SAFETY   = ("Is this Mars terrain safe for a rover to drive on? "
              "Answer in one word: safe, caution, or hazard.")

# ── Traversability rules (CLIP fast-path decision) ────────────────────────────
TRAVERSABILITY_RULES = {
    "soil":    "safe",
    "sand":    "safe",
    "bedrock": "caution",
    "big_rock": "hazard",
}

# ── SmolVLM answer parsing ────────────────────────────────────────────────────
SMOLVLM_TERRAIN_MAP = {
    "soil": 0, "dirt": 0, "regolith": 0, "dust": 0, "ground": 0,
    "bedrock": 1, "rock": 1, "stone": 1, "rocky": 1,
    "sand": 2, "sandy": 2,
    "big": 3, "boulder": 3, "boulders": 3, "large": 3,
}
HAZARD_KW  = ["hazard", "dangerous", "unsafe", "obstacle", "boulder", "cliff"]
CAUTION_KW = ["caution", "rocky", "rough", "uneven", "slope", "rocks", "bedrock"]
SAFE_KW    = ["safe", "clear", "flat", "smooth", "soil", "sand", "traversable"]


def _parse_smolvlm_terrain(text: str) -> str:
    low = text.lower().strip()
    for name in CLIP_CLASSES:
        if name in low:
            return name
    for word in low.split():
        w = word.strip(".,!?;:()")
        if w in SMOLVLM_TERRAIN_MAP:
            return CLIP_CLASSES[SMOLVLM_TERRAIN_MAP[w]]
    return "uncertain"


def _parse_smolvlm_safety(text: str) -> str:
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


class CascadePipelineNode(Node):

    def __init__(self):
        super().__init__("cascade_pipeline")

        # ── Parameters ─────────────────────────────────────────────────
        self.declare_parameter("device", "cpu")
        self.declare_parameter("clip_threshold", 0.40)
        self.declare_parameter("smolvlm_interval_sec", 5.0)
        self.declare_parameter("smolvlm_model_id", "HuggingFaceTB/SmolVLM-256M-Instruct")
        self.declare_parameter("publish_viz", True)

        device_str = self.get_parameter("device").get_parameter_value().string_value
        self.clip_threshold = (
            self.get_parameter("clip_threshold").get_parameter_value().double_value
        )
        self.smolvlm_interval = (
            self.get_parameter("smolvlm_interval_sec").get_parameter_value().double_value
        )
        smolvlm_id = (
            self.get_parameter("smolvlm_model_id").get_parameter_value().string_value
        )
        self._pub_viz_enabled = (
            self.get_parameter("publish_viz").get_parameter_value().bool_value
        )

        self.device = (
            "cuda" if device_str == "cuda" and torch.cuda.is_available() else "cpu"
        )
        self.get_logger().info(
            f"Cascade pipeline  |  device={self.device}  "
            f"CLIP threshold={self.clip_threshold}  "
            f"SmolVLM interval={self.smolvlm_interval}s"
        )

        # ── Load CLIP ───────────────────────────────────────────────────
        self.get_logger().info("Loading CLIP ViT-B/32...")
        self._clip_model, self._clip_preprocess = clip.load("ViT-B/32", device=self.device)
        self._clip_model.eval()
        text_tokens = clip.tokenize(CLIP_PROMPTS).to(self.device)
        with torch.no_grad():
            self._txt_feat = self._clip_model.encode_text(text_tokens)
            self._txt_feat /= self._txt_feat.norm(dim=-1, keepdim=True)
        self.get_logger().info("CLIP loaded")

        # ── Load SmolVLM ────────────────────────────────────────────────
        self.get_logger().info(f"Loading SmolVLM ({smolvlm_id})...")
        t0 = time.perf_counter()
        self._smolvlm_proc = AutoProcessor.from_pretrained(smolvlm_id)
        self._smolvlm_model = SmolVLMForConditionalGeneration.from_pretrained(
            smolvlm_id,
            torch_dtype=torch.float32,
            device_map={"": self.device},
        )
        self._smolvlm_model.eval()
        self.get_logger().info(f"SmolVLM loaded in {time.perf_counter()-t0:.1f}s")

        # ── State ───────────────────────────────────────────────────────
        self._latest_img: PILImage.Image = None
        self._latest_rgb: np.ndarray = None
        self._smolvlm_busy = False
        self._last_trav = "unknown"
        self._last_smolvlm_terrain = "unknown"
        self._clip_stats = {"n": 0, "total_ms": 0.0, "triggered_slow": 0}

        # ── ROS2 infrastructure ─────────────────────────────────────────
        self._sub = self.create_subscription(
            Image, "/camera/image_raw", self._image_cb, 1
        )
        self._pub_clip  = self.create_publisher(String, "/terrain_classification", 10)
        self._pub_smolv = self.create_publisher(String, "/smolvlm_terrain", 10)
        self._pub_desc  = self.create_publisher(String, "/scene_description", 10)
        self._pub_trav  = self.create_publisher(String, "/traversability", 10)
        if self._pub_viz_enabled:
            self._pub_viz = self.create_publisher(Image, "/terrain_viz", 1)
        else:
            self._pub_viz = None

        # Safety watchdog
        self._last_img_time = self.get_clock().now()
        self.create_timer(5.0, self._watchdog_cb)

        # SmolVLM periodic timer (slow path)
        self.create_timer(self.smolvlm_interval, self._trigger_slow_path)

        self.get_logger().info("Cascade pipeline ready — waiting for /camera/image_raw")

    # ── Image callback (CLIP fast path) ───────────────────────────────────────

    def _image_cb(self, msg: Image) -> None:
        self._last_img_time = self.get_clock().now()

        # Decode ROS Image → PIL (no cv_bridge)
        np_arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(
            (msg.height, msg.width, -1)
        )
        if msg.encoding in ("bgr8", "bgra8"):
            rgb = np_arr[:, :, :3][:, :, ::-1].copy()
        else:
            rgb = np_arr[:, :, :3].copy()
        pil_img = PILImage.fromarray(rgb, "RGB")
        self._latest_img = pil_img
        self._latest_rgb = rgb

        # ── CLIP inference (fast path) ──────────────────────────────────
        t0 = time.perf_counter()
        label, conf, probs = self._clip_infer(pil_img)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        self._clip_stats["n"] += 1
        self._clip_stats["total_ms"] += elapsed_ms

        # Publish CLIP result
        clip_msg = String()
        clip_msg.data = f"{label}:{conf:.3f}"
        self._pub_clip.publish(clip_msg)

        # Fast-path traversability (when CLIP is confident)
        if conf >= self.clip_threshold:
            trav = TRAVERSABILITY_RULES.get(label, "caution")
            self._last_trav = trav
        else:
            # CLIP uncertain → publish caution + trigger SmolVLM immediately
            self._last_trav = "caution"
            self._clip_stats["triggered_slow"] += 1
            self.get_logger().info(
                f"CLIP uncertain ({conf:.2f}) — triggering SmolVLM slow path"
            )
            self._trigger_slow_path()

        # Publish traversability (may be updated by SmolVLM later)
        trav_msg = String()
        trav_msg.data = self._last_trav
        self._pub_trav.publish(trav_msg)

        # Debug viz
        if self._pub_viz is not None:
            self._publish_viz(rgb, label, conf, probs)

        # Log every 20 frames
        if self._clip_stats["n"] % 20 == 0:
            avg = self._clip_stats["total_ms"] / self._clip_stats["n"]
            slow_pct = 100 * self._clip_stats["triggered_slow"] / self._clip_stats["n"]
            self.get_logger().info(
                f"[{self._clip_stats['n']} frames] CLIP avg {avg:.0f}ms  "
                f"slow-path triggered {slow_pct:.1f}%  "
                f"last trav={self._last_trav}"
            )

    # ── CLIP inference ────────────────────────────────────────────────────────

    def _clip_infer(self, pil_img: PILImage.Image):
        img_t = self._clip_preprocess(pil_img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            img_feat = self._clip_model.encode_image(img_t)
            img_feat /= img_feat.norm(dim=-1, keepdim=True)
            probs = (100.0 * img_feat @ self._txt_feat.T).softmax(dim=-1)[0]
        idx = int(probs.argmax())
        return CLIP_CLASSES[idx], float(probs[idx]), probs.cpu().numpy()

    # ── SmolVLM slow path ─────────────────────────────────────────────────────

    def _trigger_slow_path(self) -> None:
        """Run SmolVLM in a background thread (non-blocking)."""
        if self._latest_img is None or self._smolvlm_busy:
            return
        t = threading.Thread(target=self._smolvlm_infer, daemon=True)
        t.start()

    def _smolvlm_infer(self) -> None:
        self._smolvlm_busy = True
        t0 = time.perf_counter()
        image = self._latest_img  # snapshot

        try:
            terrain_ans = self._smolvlm_ask(image, Q_TERRAIN)
            safety_ans  = self._smolvlm_ask(image, Q_SAFETY)
            elapsed_ms  = (time.perf_counter() - t0) * 1000

            terrain_name = _parse_smolvlm_terrain(terrain_ans)
            traversability = _parse_smolvlm_safety(safety_ans)
            self._last_trav = traversability
            self._last_smolvlm_terrain = terrain_name

            # Publish SmolVLM terrain
            smolv_msg = String()
            smolv_msg.data = f"{terrain_name}:smolvlm"
            self._pub_smolv.publish(smolv_msg)

            # Publish scene description
            desc_msg = String()
            desc_msg.data = f"terrain={terrain_ans} | safety={safety_ans}"
            self._pub_desc.publish(desc_msg)

            # Publish updated traversability
            trav_msg = String()
            trav_msg.data = traversability
            self._pub_trav.publish(trav_msg)

            self.get_logger().info(
                f"[SmolVLM {elapsed_ms:.0f}ms] "
                f"terrain='{terrain_ans}'({terrain_name}) | "
                f"safety='{safety_ans}' → {traversability}"
            )

        except Exception as e:
            self.get_logger().error(f"SmolVLM error: {e}")
        finally:
            self._smolvlm_busy = False

    def _smolvlm_ask(self, image: PILImage.Image, question: str) -> str:
        messages = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": question}
            ]}
        ]
        prompt = self._smolvlm_proc.apply_chat_template(
            messages, add_generation_prompt=True
        )
        inputs = self._smolvlm_proc(text=prompt, images=[image], return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            out_ids = self._smolvlm_model.generate(
                **inputs, max_new_tokens=20, do_sample=False
            )
        n_in = inputs["input_ids"].shape[1]
        return self._smolvlm_proc.tokenizer.decode(
            out_ids[0][n_in:], skip_special_tokens=True
        ).strip()

    # ── Viz ───────────────────────────────────────────────────────────────────

    def _publish_viz(self, rgb: np.ndarray, label: str,
                     conf: float, probs: np.ndarray) -> None:
        from PIL import ImageDraw

        COLOUR = {
            "soil": (139, 90, 43), "bedrock": (160, 150, 140),
            "sand": (210, 190, 120), "big_rock": (80, 80, 80),
        }
        TRAV_COLOUR = {"safe": (50, 180, 50), "caution": (220, 160, 0), "hazard": (200, 50, 50)}

        colour = COLOUR.get(label, (120, 120, 120))
        trav_colour = TRAV_COLOUR.get(self._last_trav, (150, 150, 150))

        pil = PILImage.fromarray(rgb.copy(), "RGB")
        draw = ImageDraw.Draw(pil)
        h, w = rgb.shape[:2]

        # Top bar: CLIP result
        draw.rectangle([(0, 0), (w, 40)], fill=colour)
        draw.text((8, 10), f"CLIP: {label} ({conf:.2f})", fill=(255, 255, 255))

        # Bottom bar: traversability
        draw.rectangle([(0, h - 30), (w, h)], fill=trav_colour)
        smv = self._last_smolvlm_terrain
        draw.text((8, h - 22), f"TRAV: {self._last_trav}  SmolVLM: {smv}", fill=(255, 255, 255))

        out_arr = np.asarray(pil)
        viz_msg = Image()
        viz_msg.height = out_arr.shape[0]
        viz_msg.width  = out_arr.shape[1]
        viz_msg.encoding = "rgb8"
        viz_msg.is_bigendian = False
        viz_msg.step = out_arr.shape[1] * 3
        viz_msg.data = out_arr.tobytes()
        self._pub_viz.publish(viz_msg)

    # ── Watchdog ──────────────────────────────────────────────────────────────

    def _watchdog_cb(self) -> None:
        elapsed = (self.get_clock().now() - self._last_img_time).nanoseconds / 1e9
        if elapsed > 5.0:
            for pub, val in [
                (self._pub_clip,  "unknown:0.000"),
                (self._pub_trav,  "unknown"),
            ]:
                msg = String()
                msg.data = val
                pub.publish(msg)
            self.get_logger().warn(f"No image for {elapsed:.1f}s — publishing unknown")


def main(args=None):
    rclpy.init(args=args)
    node = CascadePipelineNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
