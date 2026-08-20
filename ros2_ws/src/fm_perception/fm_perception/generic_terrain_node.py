#!/usr/bin/env python3
"""
Purpose: Generic frozen-encoder + LogReg terrain classification ROS2 node.
         Loads any HuggingFace vision encoder specified by model_id and a
         pre-extracted feature cache (.npy files). Used for multi-model
         Gazebo traversability comparison (DINOv3 ViT-S, DINOv2 ViT-L, etc.).
         Publishes the same /terrain_classification topic as dinov2_terrain_node.py
         so that terrain_controller_node.py and the experiment scripts work unchanged.
Inputs:  /camera/image_raw (sensor_msgs/Image)
         feat_npy:  path to train features  (.npy, shape [N, D])
         label_npy: path to train labels    (.npy, shape [N,])
Outputs: /terrain_classification (std_msgs/String)  "label:confidence"
         /terrain_viz (sensor_msgs/Image)            annotated frame
How to run:
    # Via launch file (recommended) — see dinov3_vits_controller_test.launch.py
    # Direct:
    ros2 run fm_perception generic_terrain_node.py \
        --ros-args -p model_id:=facebook/dinov3-vits16-pretrain-lvd1689m \
                   -p feat_npy:=/path/to/dinov3_train_1000_feats.npy \
                   -p label_npy:=/path/to/dinov3_train_1000_labels.npy
Project: Onboard Visual Foundation Models for Mars Terrain Perception and Rover Navigation
"""

import os
import time

import numpy as np
import rclpy
import torch
from PIL import Image as PILImage
from rclpy.node import Node
from sensor_msgs.msg import Image
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize
from std_msgs.msg import String
from transformers import AutoImageProcessor, AutoModel

CLASS_NAMES = ["soil", "bedrock", "sand", "big_rock"]
LOGR_C      = 0.316   # matches offline experiment protocol


class GenericTerrainNode(Node):

    def __init__(self):
        super().__init__("generic_terrain_node")

        # ── Parameters ─────────────────────────────────────────────────
        self.declare_parameter("model_id",             "facebook/dinov3-vits16-pretrain-lvd1689m")
        self.declare_parameter("feat_npy",             "")
        self.declare_parameter("label_npy",            "")
        self.declare_parameter("n_shot",               1000)
        self.declare_parameter("confidence_threshold", 0.40)
        self.declare_parameter("device",               "cpu")
        self.declare_parameter("publish_viz",          True)
        self.declare_parameter("display_name",         "")   # short label for viz overlay

        model_id    = self.get_parameter("model_id").get_parameter_value().string_value
        feat_npy    = self.get_parameter("feat_npy").get_parameter_value().string_value
        label_npy   = self.get_parameter("label_npy").get_parameter_value().string_value
        n_shot      = self.get_parameter("n_shot").get_parameter_value().integer_value
        self.threshold  = self.get_parameter("confidence_threshold").get_parameter_value().double_value
        device_str  = self.get_parameter("device").get_parameter_value().string_value
        publish_viz = self.get_parameter("publish_viz").get_parameter_value().bool_value
        display_name = self.get_parameter("display_name").get_parameter_value().string_value
        self.display_name = display_name if display_name else model_id.split("/")[-1]

        self.device = "cuda" if (device_str == "cuda" and torch.cuda.is_available()) else "cpu"
        self.get_logger().info(f"Device: {self.device}")

        # ── Load encoder ───────────────────────────────────────────────
        self.get_logger().info(f"Loading {model_id} (frozen)...")
        t0 = time.perf_counter()
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.encoder   = AutoModel.from_pretrained(model_id).to(self.device).eval()
        n_params = sum(p.numel() for p in self.encoder.parameters()) / 1e6
        self.get_logger().info(
            f"Loaded in {time.perf_counter()-t0:.1f}s | {n_params:.1f}M params"
        )

        # ── Train LogReg from .npy cache ───────────────────────────────
        self.clf = self._train_logreg(feat_npy, label_npy, n_shot)

        # ── Warm-up ────────────────────────────────────────────────────
        self._classify(PILImage.fromarray(np.zeros((224, 224, 3), dtype=np.uint8), "RGB"))
        self.get_logger().info("Warm-up done")

        # ── ROS2 pub / sub ─────────────────────────────────────────────
        self.sub     = self.create_subscription(Image, "/camera/image_raw", self._img_cb, 1)
        self.pub     = self.create_publisher(String, "/terrain_classification", 10)
        self.pub_viz = self.create_publisher(Image, "/terrain_viz", 1) if publish_viz else None

        self.last_img_time = self.get_clock().now()
        self.create_timer(5.0, self._watchdog_cb)

        self.n_processed = 0
        self.total_ms    = 0.0

        self.get_logger().info(
            f"generic_terrain_node ready | model={self.display_name} "
            f"| threshold={self.threshold:.2f} | subscribed /camera/image_raw"
        )

    # ── LogReg training ────────────────────────────────────────────────

    def _train_logreg(self, feat_npy: str, label_npy: str, n_shot: int) -> LogisticRegression:
        if not feat_npy or not os.path.exists(feat_npy):
            raise FileNotFoundError(
                f"Feature cache not found: '{feat_npy}'\n"
                "Set feat_npy parameter to the .npy file path."
            )
        if not label_npy or not os.path.exists(label_npy):
            raise FileNotFoundError(
                f"Label cache not found: '{label_npy}'\n"
                "Set label_npy parameter to the .npy file path."
            )

        feats  = np.load(feat_npy)    # (N, D)
        labels = np.load(label_npy)   # (N,)

        idx = []
        for c in np.unique(labels):
            c_idx  = np.where(labels == c)[0]
            chosen = c_idx[:n_shot] if len(c_idx) >= n_shot else c_idx
            idx.extend(chosen.tolist())
        idx = np.array(idx)

        X = normalize(feats[idx], norm="l2")
        y = labels[idx]

        t0  = time.perf_counter()
        clf = LogisticRegression(C=LOGR_C, max_iter=1000, solver="lbfgs", random_state=42)
        clf.fit(X, y)
        self.get_logger().info(
            f"LogReg trained on {len(idx)} samples in {time.perf_counter()-t0:.1f}s"
        )
        return clf

    # ── Inference ──────────────────────────────────────────────────────

    def _classify(self, pil_img: PILImage.Image):
        inputs = self.processor(images=pil_img, return_tensors="pt").to(self.device)
        with torch.no_grad():
            feat = self.encoder(**inputs).last_hidden_state[:, 0, :]  # CLS token
        feat_np = normalize(feat.cpu().numpy(), norm="l2")

        probs     = self.clf.predict_proba(feat_np)[0]
        full_probs = np.zeros(len(CLASS_NAMES), dtype=np.float32)
        for i, cls in enumerate(self.clf.classes_):
            full_probs[cls] = probs[i]

        pred_idx   = int(full_probs.argmax())
        confidence = float(full_probs[pred_idx])
        return CLASS_NAMES[pred_idx], confidence, full_probs

    # ── ROS2 callbacks ─────────────────────────────────────────────────

    def _img_cb(self, msg: Image) -> None:
        self.last_img_time = self.get_clock().now()

        np_arr = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, -1))
        if msg.encoding in ("bgr8", "bgra8"):
            np_arr = np_arr[:, :, :3][:, :, ::-1].copy()
        else:
            np_arr = np_arr[:, :, :3].copy()
        pil_img = PILImage.fromarray(np_arr, "RGB")

        t0 = time.perf_counter()
        label, confidence, probs = self._classify(pil_img)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        self.n_processed += 1
        self.total_ms    += elapsed_ms

        out      = String()
        out.data = (
            f"{label}:{confidence:.3f}"
            if confidence >= self.threshold
            else f"uncertain:{confidence:.3f}"
        )
        self.pub.publish(out)

        if self.n_processed % 10 == 0:
            self.get_logger().info(
                f"[{self.n_processed}] {out.data} | "
                f"{elapsed_ms:.0f}ms avg {self.total_ms/self.n_processed:.0f}ms"
            )

        if self.pub_viz:
            self._publish_viz(np_arr, label, confidence, probs)

    def _publish_viz(self, rgb_arr, label, confidence, probs):
        from PIL import ImageDraw
        colour_map = {
            "soil":      (139, 90,  43),
            "bedrock":   (160, 150, 140),
            "sand":      (210, 190, 120),
            "big_rock":  (80,  80,  80),
            "uncertain": (200, 50,  50),
        }
        colour  = colour_map.get(label, (200, 50, 50))
        pil_viz = PILImage.fromarray(rgb_arr.copy(), "RGB")
        draw    = ImageDraw.Draw(pil_viz)
        h, w    = rgb_arr.shape[:2]

        draw.rectangle([(0, 0), (w, 60)], fill=colour)
        draw.text((10, 10), self.display_name, fill=(200, 200, 200))
        draw.text((10, 30), f"{label}  ({confidence:.2f})", fill=(255, 255, 255))

        bar_w = max(1, w // len(CLASS_NAMES) - 5)
        for i, (name, prob) in enumerate(zip(CLASS_NAMES, probs)):
            x      = i * (bar_w + 5) + 5
            y      = h - 25
            filled = int(float(prob) * bar_w)
            draw.rectangle([(x, y), (x + bar_w, y + 15)], fill=(80, 80, 80))
            draw.rectangle([(x, y), (x + filled, y + 15)],
                           fill=colour_map.get(name, (150, 150, 150)))
            draw.text((x, h - 12), name[:3], fill=(220, 220, 220))

        out_arr          = np.asarray(pil_viz)
        viz_msg          = Image()
        viz_msg.height   = out_arr.shape[0]
        viz_msg.width    = out_arr.shape[1]
        viz_msg.encoding = "rgb8"
        viz_msg.step     = out_arr.shape[1] * 3
        viz_msg.data     = out_arr.tobytes()
        self.pub_viz.publish(viz_msg)

    def _watchdog_cb(self) -> None:
        elapsed = (self.get_clock().now() - self.last_img_time).nanoseconds / 1e9
        if elapsed > 5.0:
            msg      = String()
            msg.data = "unknown:0.000"
            self.pub.publish(msg)
            self.get_logger().warn(f"No image for {elapsed:.1f}s — publishing unknown")


def main(args=None):
    rclpy.init(args=args)
    node = GenericTerrainNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
