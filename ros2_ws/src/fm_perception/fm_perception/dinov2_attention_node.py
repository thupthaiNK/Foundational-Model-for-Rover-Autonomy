#!/usr/bin/env python3
"""
Purpose: ROS2 node that subscribes to /camera/image_raw and publishes a DINOv2
         self-attention heatmap overlay on /terrain_attention. Shows WHERE in the
         camera frame DINOv2 focuses when processing terrain — strong attention on
         texture patches (sandy grains, bedrock fractures) vs. uniform zones provides
         interpretability evidence for Ch4 thesis discussion.
         Uses last-layer CLS-to-patch attention averaged over all 6 heads, matching
         experiments/dinov2_attention_maps.py. Runs as a separate process from
         dinov2_terrain_node.py to avoid adding ~50ms to the classification loop.
Inputs:  /camera/image_raw (sensor_msgs/Image)
         Model: facebook/dinov2-with-registers-small (frozen, same as terrain node)
Outputs: /terrain_attention (sensor_msgs/Image) — inferno heatmap blended with camera feed
         Publishes every skip_frames-th frame (default every 3rd ≈ 3fps at 10fps camera)
How to run:
    source /opt/ros/humble/setup.bash
    source ros2_ws/install/setup.bash
    ros2 run fm_perception dinov2_attention_node.py
    # View output:
    ros2 run rqt_image_view rqt_image_view /terrain_attention
Parameters:
    device        (string, default cpu):  cpu or cuda
    skip_frames   (int,    default 3):    publish every Nth frame to limit CPU load
    alpha         (float,  default 0.5):  heatmap blend opacity (0=original, 1=heatmap)
Project: Onboard Visual Foundation Models for Mars Terrain Perception and Rover Navigation
"""

import time

import matplotlib
matplotlib.use("Agg")
import numpy as np
import rclpy
import torch
from PIL import Image as PILImage
from rclpy.node import Node
from sensor_msgs.msg import Image
from transformers import AutoImageProcessor, AutoModel

DINOV2_MODEL_ID = "facebook/dinov2-with-registers-small"

# Inferno colormap lookup — compatible across matplotlib versions
try:
    _INFERNO = matplotlib.colormaps["inferno"]
except AttributeError:
    import matplotlib.cm as cm
    _INFERNO = cm.get_cmap("inferno")


class DINOv2AttentionNode(Node):

    def __init__(self):
        super().__init__("dinov2_attention_node")

        # ── Parameters ──────────────────────────────────────────────────
        self.declare_parameter("device",      "cpu")
        self.declare_parameter("skip_frames", 3)
        self.declare_parameter("alpha",       0.5)

        device_str  = self.get_parameter("device").get_parameter_value().string_value
        self._skip  = self.get_parameter("skip_frames").get_parameter_value().integer_value
        self._alpha = self.get_parameter("alpha").get_parameter_value().double_value

        self.device = "cuda" if (device_str == "cuda" and torch.cuda.is_available()) else "cpu"
        self.get_logger().info(f"Device: {self.device}")

        # ── Load DINOv2 with attention outputs enabled ───────────────────
        self.get_logger().info(f"Loading {DINOV2_MODEL_ID} (attention mode)...")
        t0 = time.perf_counter()
        self.processor = AutoImageProcessor.from_pretrained(DINOV2_MODEL_ID)
        self.encoder   = AutoModel.from_pretrained(
            DINOV2_MODEL_ID, output_attentions=True
        ).to(self.device).eval()
        self.get_logger().info(f"Model loaded in {time.perf_counter()-t0:.1f}s")

        # Pre-compute patch grid (constant for 224×224 input)
        crop_h = self.processor.crop_size.get("height", 224)
        crop_w = self.processor.crop_size.get("width",  224)
        patch  = self.encoder.config.patch_size          # 14 for ViT-S/14
        self._n_patches = (crop_h // patch) * (crop_w // patch)  # 256
        self._grid_h    = crop_h // patch                         # 16
        self._grid_w    = crop_w // patch                         # 16
        # Number of prefix tokens: 1 CLS + 4 register tokens
        # (computed at runtime from actual sequence length)

        # Warm-up pass to JIT-compile and avoid first-call spike
        dummy = PILImage.fromarray(np.zeros((224, 224, 3), dtype=np.uint8), "RGB")
        self._extract_attention(dummy)
        self.get_logger().info("Warm-up done")

        # ── ROS2 pub / sub ───────────────────────────────────────────────
        self.sub = self.create_subscription(
            Image, "/camera/image_raw", self._image_callback, 1
        )
        self.pub = self.create_publisher(Image, "/terrain_attention", 1)

        self._frame_count  = 0
        self._pub_count    = 0
        self._total_ms     = 0.0
        self.get_logger().info(
            f"dinov2_attention_node ready | skip={self._skip} | alpha={self._alpha:.1f}\n"
            "  /camera/image_raw → /terrain_attention (inferno heatmap overlay)"
        )

    # ── Attention extraction ────────────────────────────────────────────

    def _extract_attention(self, pil_img: PILImage.Image) -> np.ndarray:
        """Returns normalised float32 [grid_h, grid_w] attention map in [0, 1]."""
        inputs = self.processor(images=pil_img, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.encoder(**inputs, output_attentions=True)

        # Last-layer attention: [1, n_heads, seq_len, seq_len]
        last_attn = outputs.attentions[-1][0]   # [n_heads, seq_len, seq_len]
        cls_row   = last_attn[:, 0, :]          # [n_heads, seq_len]

        # Skip prefix tokens (CLS + register tokens), keep only patch tokens
        n_prefix   = cls_row.shape[1] - self._n_patches
        patch_attn = cls_row[:, n_prefix:].cpu().numpy()  # [n_heads, n_patches]

        avg = patch_attn.mean(axis=0)                     # [n_patches]  avg over heads
        avg = (avg - avg.min()) / (avg.max() - avg.min() + 1e-8)
        return avg.reshape(self._grid_h, self._grid_w).astype(np.float32)

    def _make_overlay(self, rgb_arr: np.ndarray, attn_map: np.ndarray) -> np.ndarray:
        """Blend original RGB frame with inferno attention heatmap."""
        h, w = rgb_arr.shape[:2]

        # Upsample 16×16 attention grid to full image resolution
        attn_pil = PILImage.fromarray((attn_map * 255).astype(np.uint8)).resize(
            (w, h), PILImage.BILINEAR
        )
        attn_np  = np.array(attn_pil) / 255.0            # [H, W] float [0, 1]

        # Apply inferno colormap → RGB [H, W, 3] uint8
        heatmap  = (_INFERNO(attn_np)[:, :, :3] * 255).astype(np.uint8)

        # Blend: alpha controls how much heatmap shows over original
        overlay = (self._alpha * heatmap + (1.0 - self._alpha) * rgb_arr).astype(np.uint8)
        return overlay

    # ── ROS2 callback ─────────────────────────────────────────────────

    def _image_callback(self, msg: Image) -> None:
        self._frame_count += 1
        if self._frame_count % self._skip != 0:
            return

        # ROS Image → numpy RGB (no cv_bridge — avoids NumPy 2.x incompatibility)
        np_arr = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, -1))
        if msg.encoding in ("bgr8", "bgra8"):
            np_arr = np_arr[:, :, :3][:, :, ::-1].copy()
        else:
            np_arr = np_arr[:, :, :3].copy()

        t0 = time.perf_counter()
        pil_img  = PILImage.fromarray(np_arr, "RGB")
        attn_map = self._extract_attention(pil_img)
        overlay  = self._make_overlay(np_arr, attn_map)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        out_msg              = Image()
        out_msg.header       = msg.header
        out_msg.height       = overlay.shape[0]
        out_msg.width        = overlay.shape[1]
        out_msg.encoding     = "rgb8"
        out_msg.is_bigendian = False
        out_msg.step         = overlay.shape[1] * 3
        out_msg.data         = overlay.tobytes()
        self.pub.publish(out_msg)

        self._pub_count += 1
        self._total_ms  += elapsed_ms
        if self._pub_count % 10 == 0:
            avg_ms = self._total_ms / self._pub_count
            self.get_logger().info(
                f"[pub {self._pub_count}] attention | {elapsed_ms:.0f}ms avg {avg_ms:.0f}ms"
            )


def main(args=None):
    rclpy.init(args=args)
    node = DINOv2AttentionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
