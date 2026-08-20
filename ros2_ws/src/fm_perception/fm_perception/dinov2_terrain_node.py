#!/usr/bin/env python3
"""
Purpose: ROS2 node that subscribes to /camera/image_raw, classifies terrain using
         a frozen DINOv2 ViT-S/14 encoder + LogReg probe trained on 1000-shot
         AI4Mars features. Replaces clip_terrain_node.py with the best RPi-viable
         model (90.24% accuracy, ~181ms/img on CPU vs CLIP 87.8%).
         LogReg is trained at node startup from the pre-extracted feature cache
         (same cache, same LogReg, regardless of encoder backend below -- the
         probe itself is unaffected by INT8 quantization, only feature
         extraction is).
         Two interchangeable feature-extraction backends, selected by the
         use_int8_model parameter:
           - FP32 torch (default): loads facebook/dinov2-with-registers-small
             via HuggingFace transformers. Used by every Gazebo experiment
             this thesis already reports -- unaffected by the addition below.
           - INT8 ONNX (opt-in, use_int8_model:=true): loads a pre-quantized
             ONNX Runtime session instead of the torch model (B1 result,
             2026-06-30: 570ms->386ms latency, 220MB->163MB RAM, -0.5pp
             accuracy on a 200-image sample -- see experiments/int8_quantization_rpi.py
             and experiments/int8_benchmark_rpi.py). Set as the default for
             real_hardware_deployment.launch.py specifically, since the RPi's
             CPU is shared across a growing number of concurrent nodes.
Inputs:  /camera/image_raw (sensor_msgs/Image)
         Feature cache: experiments/results/feature_cache/dinov2_reg_small_train_1000shot.npz
Outputs: /terrain_classification (std_msgs/String)        "label:confidence"
         /terrain_class_probs  (std_msgs/Float32MultiArray) [P_soil, P_bedrock, P_sand, P_big_rock]
         /traversability_score (std_msgs/Float64)           continuous risk score, 0.0=safe .. 1.0=stop
         /inference_latency_ms (std_msgs/Float64)           DINOv2+LogReg wall-clock ms per frame
         /terrain_viz (sensor_msgs/Image)                   annotated frame for debugging
How to run:
    source /opt/ros/humble/setup.bash
    cd ros2_ws && colcon build --packages-select fm_perception
    source install/setup.bash
    ros2 run fm_perception dinov2_terrain_node
    ros2 run fm_perception dinov2_terrain_node --ros-args -p cache_path:=/path/to/dinov2_reg_small_train_1000shot.npz
    ros2 run fm_perception dinov2_terrain_node --ros-args -p use_int8_model:=true
Project: Onboard Visual Foundation Models for Mars Terrain Perception and Rover Navigation
"""

import hashlib
import io
import json
import os
import time

import numpy as np
import rclpy
from PIL import Image as PILImage
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, Image
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize
from std_msgs.msg import Bool, Float32MultiArray, Float64, String
# transformers is imported lazily and ONLY on the FP32 torch path. Traced
# 2026-07-27: `from transformers import AutoImageProcessor` alone pulls in
# transformers/generation/logits_process.py, which does a top-level
# `import torch` -- so even the "torch-free" INT8 ONNX backend could not load
# in the Pi's ROS2 container, which has no torch. dinov2_preprocess
# reproduces the same preprocessing in pure PIL/numpy (pinned to
# AutoImageProcessor by test/test_dinov2_preprocess.py, including a
# feature-level equivalence check through the real ONNX encoder).
from fm_perception.asset_paths import describe_missing_asset, find_repo_asset
from fm_perception.dinov2_preprocess import preprocess_dinov2
from fm_perception.frame_quality import (
    DEFAULT_MIN_DETAIL, frame_detail, frame_is_informative,
)

CLASS_NAMES          = ["soil", "bedrock", "sand", "big_rock"]
CONFIDENCE_THRESHOLD = 0.40   # calibrated threshold matching offline experiments (T*=0.461 applied)
T_STAR               = 0.461  # temperature scaling (§4.7.27): ECE 0.1695→0.0325, 80.8% reduction
DINOV2_MODEL_ID      = "facebook/dinov2-with-registers-small"   # 384-d, 21M params, 90.24% AI4Mars
LOGR_C               = 0.316                      # matches offline experiment protocol

# Per-class hazard cost for the continuous traversability score (H3f).
# Normalized inverse of terrain_controller_node.py's existing per-class speed
# policy (soil 0.10, sand 0.05, bedrock 0.03, big_rock 0.00 m/s over max 0.10 m/s)
# so the continuous score stays consistent with the already-validated discrete policy.
CLASS_RISK = {"soil": 0.0, "bedrock": 0.7, "sand": 0.5, "big_rock": 1.0}


def logreg_cache_path(feature_cache_path: str) -> str:
    """Where the fitted probe is kept: beside the features it came from."""
    return feature_cache_path + ".logreg.npz"


def _logreg_fingerprint(feature_cache_path: str, n_shot: int,
                        class_weight_balanced: bool) -> str:
    """Everything that shapes the fit, so a stale probe can be spotted.

    The feature file is hashed rather than described by size and mtime.
    Regenerating it to the same length inside the same second is not a
    far-fetched case -- it is what a re-run of the experiment that produces it
    looks like -- and a stale probe is exactly the failure worth paying for.
    Hashing the 9 MB npz costs tens of milliseconds against the 10.3 s the
    cache saves.
    """
    digest = hashlib.sha256()
    with open(feature_cache_path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return json.dumps([
        digest.hexdigest(),
        int(n_shot),
        bool(class_weight_balanced),
        float(LOGR_C),
    ])


def fit_or_load_logreg(feature_cache_path: str, n_shot: int,
                       class_weight_balanced: bool):
    """Return (probe, was_fitted), reusing a cached fit when it still applies.

    Refitting the probe from the feature cache took 10.3 s of every launch on
    the Pi, measured 2026-07-29, to reproduce exactly the same coefficients --
    a fifth of the whole startup budget spent on a deterministic computation.

    Correctness comes first: anything that would change the fit invalidates
    the cache, and any problem reading or writing it falls back to fitting
    rather than to guessing. A stale probe would classify terrain silently and
    wrongly, which is much worse than a slow boot.

    Stored as plain arrays rather than a pickled estimator. A fitted
    LogisticRegression is entirely described by coef_, intercept_ and
    classes_, so nothing is lost, and it avoids both unpickling a file as code
    and the version coupling that a pickled estimator carries.
    """
    if not feature_cache_path or not os.path.exists(feature_cache_path):
        raise FileNotFoundError(
            (describe_missing_asset(_REL_CACHE, "cache_path")
             if not feature_cache_path
             else f"Feature cache not found: {feature_cache_path}")
            + "\nRun experiments/dinov2_terrain_test.py to generate it."
        )

    fingerprint = _logreg_fingerprint(
        feature_cache_path, n_shot, class_weight_balanced
    )
    cached_at = logreg_cache_path(feature_cache_path)
    if os.path.exists(cached_at):
        try:
            with np.load(cached_at) as payload:
                if str(payload["fingerprint"]) == fingerprint:
                    clf = LogisticRegression(
                        C=LOGR_C, max_iter=1000, solver="lbfgs",
                        random_state=42,
                        class_weight="balanced" if class_weight_balanced else None,
                    )
                    clf.coef_ = payload["coef"]
                    clf.intercept_ = payload["intercept"]
                    clf.classes_ = payload["classes"]
                    clf.n_features_in_ = clf.coef_.shape[1]
                    return clf, False
        except Exception:
            # Truncated, half-written, or not an npz at all.
            pass

    clf = _fit_logreg(feature_cache_path, n_shot, class_weight_balanced)
    try:
        np.savez(cached_at, fingerprint=fingerprint, coef=clf.coef_,
                 intercept=clf.intercept_, classes=clf.classes_)
    except Exception:
        # The workspace may be read-only. Losing the optimisation is fine;
        # refusing to boot over it is not.
        pass
    return clf, True


def _fit_logreg(feature_cache_path: str, n_shot: int,
                class_weight_balanced: bool) -> LogisticRegression:
    data = np.load(feature_cache_path)
    feats = data["feats"]     # (N, 384)
    labels = data["labels"]   # (N,)

    # n_shot per class, taken in file order: deterministic, and the same
    # protocol as the offline experiments, so the deployed probe is the one
    # the reported accuracy figures describe.
    idx = []
    for c in np.unique(labels):
        c_idx = np.where(labels == c)[0]
        chosen = c_idx[:n_shot] if len(c_idx) >= n_shot else c_idx
        idx.extend(chosen.tolist())
    idx = np.array(idx)

    X = normalize(feats[idx], norm="l2")
    y = labels[idx]

    clf = LogisticRegression(
        C=LOGR_C,
        max_iter=1000,
        solver="lbfgs",
        random_state=42,
        class_weight="balanced" if class_weight_balanced else None,
    )
    clf.fit(X, y)
    return clf


def encode_jpeg(rgb_arr: np.ndarray, quality: int = 80) -> bytes:
    """JPEG-encode one RGB frame for recording.

    The bag used to record /camera/image_raw, every frame, uncompressed: 11 GB
    for a 573 s run on 2026-07-29, nearly all of it identical views of a rover
    that was not moving. What the thesis needs from a run is the frame behind
    each DINOv2 verdict, and there is one of those per inference rather than
    per camera frame. Encoding costs a few ms against a 1800 ms inference.
    """
    buf = io.BytesIO()
    PILImage.fromarray(rgb_arr, "RGB").save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def to_grayscale_3ch(rgb_arr: np.ndarray) -> np.ndarray:
    """Convert an HxWx3 RGB array to grayscale, replicated across 3 channels.

    Uses ITU-R BT.601 luma weights (0.299R + 0.587G + 0.114B), matching PIL's
    default Image.convert("L"). Output stays 3-channel (R=G=B) so it remains
    compatible with DINOv2's RGB-pretrained normalisation stats, matching how
    the AI4Mars NAVCAM grayscale training textures are represented (see
    project_gazebo_quality memory: "texture PNGs are grayscale, R=G=B").
    Opt-in ablation (H3f wait-time plan item 4) targeting the Exp 5b root
    cause: NAVCAM grayscale training vs RPi RGB inference, luminance mismatch.
    """
    gray = (
        rgb_arr[..., 0].astype(np.float32) * 0.299
        + rgb_arr[..., 1].astype(np.float32) * 0.587
        + rgb_arr[..., 2].astype(np.float32) * 0.114
    ).astype(rgb_arr.dtype)
    return np.stack([gray, gray, gray], axis=-1)


def traversability_score(probs, confidence: float, threshold: float = CONFIDENCE_THRESHOLD) -> float:
    """Continuous risk score in [0, 1] (0.0=safe, 1.0=stop).

    Expected hazard cost under the full class-probability vector, so a mixed
    prediction (e.g. soil/big_rock near 50/50) yields a mid-range score rather
    than snapping to one class. Confidence below `threshold` forces score=1.0,
    matching the existing calibrated uncertain→STOP safety rule.
    """
    if confidence < threshold:
        return 1.0
    return float(sum(p * CLASS_RISK[name] for p, name in zip(probs, CLASS_NAMES)))

# Default asset paths — override via ROS2 parameter, or FM_PERCEPTION_ASSET_ROOT.
# These used to be built from a fixed ".." count up from this file. The count was
# wrong in both source and install space, so the node raised FileNotFoundError
# from _train_logreg on startup with default parameters, on every machine. No
# launch file overrides either parameter, so real_hardware_deployment.launch.py
# inherited the broken defaults. find_repo_asset searches upward until the file
# is really there, which does not care how deep the module sits.
_REL_CACHE = os.path.join(
    "experiments", "results", "feature_cache", "dinov2_reg_small_train_1000shot.npz"
)
_REL_ONNX_ENCODER = os.path.join(
    "experiments", "results", "dinov2_reg_small_encoder_int8.onnx"
)
# May be "" when the assets are not on this machine (the Pi container, until they
# are copied across). Declaring the parameter empty rather than as a wrong path
# lets the load site raise an error that names how to fix it.
_DEFAULT_CACHE = find_repo_asset(_REL_CACHE) or ""
_DEFAULT_ONNX_ENCODER = find_repo_asset(_REL_ONNX_ENCODER) or ""


class DINOv2TerrainNode(Node):

    def __init__(self):
        super().__init__("dinov2_terrain_node")

        # ── Parameters ─────────────────────────────────────────────────
        self.declare_parameter("device", "cpu")
        self.declare_parameter("confidence_threshold", CONFIDENCE_THRESHOLD)
        self.declare_parameter("publish_viz", True)
        self.declare_parameter("cache_path", _DEFAULT_CACHE)
        self.declare_parameter("n_shot", 1000)
        self.declare_parameter("class_weight_balanced", False)
        self.declare_parameter("enable_obstacle_gate", False)
        self.declare_parameter("obstacle_gate_label", "uncertain")
        self.declare_parameter("obstacle_gate_min_upper_edge", 4.6)
        self.declare_parameter("obstacle_gate_max_p90", 88.0)
        self.declare_parameter("obstacle_gate_max_mean", 66.0)
        self.declare_parameter("obstacle_gate_min_darkness", 40.0)  # mean <= this also triggers gate
        self.declare_parameter("grayscale_preprocessing", False)  # opt-in ablation, see to_grayscale_3ch
        # Frames below this absolute intensity spread are published as uncertain without
        # being classified. See frame_quality.py for why a confidence threshold
        # cannot do this job. Set to 0.0 to disable the gate entirely.
        self.declare_parameter("min_frame_detail", DEFAULT_MIN_DETAIL)
        self.declare_parameter("use_int8_model", False)
        self.declare_parameter("onnx_encoder_path", _DEFAULT_ONNX_ENCODER)

        device_str     = self.get_parameter("device").get_parameter_value().string_value
        self.threshold = self.get_parameter("confidence_threshold").get_parameter_value().double_value
        publish_viz    = self.get_parameter("publish_viz").get_parameter_value().bool_value
        cache_path     = self.get_parameter("cache_path").get_parameter_value().string_value
        n_shot         = self.get_parameter("n_shot").get_parameter_value().integer_value
        class_weight_balanced = (
            self.get_parameter("class_weight_balanced").get_parameter_value().bool_value
        )
        self.enable_obstacle_gate = (
            self.get_parameter("enable_obstacle_gate").get_parameter_value().bool_value
        )
        self.obstacle_gate_label = (
            self.get_parameter("obstacle_gate_label").get_parameter_value().string_value
        )
        self.obstacle_gate_min_upper_edge = (
            self.get_parameter("obstacle_gate_min_upper_edge").get_parameter_value().double_value
        )
        self.obstacle_gate_max_p90 = (
            self.get_parameter("obstacle_gate_max_p90").get_parameter_value().double_value
        )
        self.obstacle_gate_max_mean = (
            self.get_parameter("obstacle_gate_max_mean").get_parameter_value().double_value
        )
        self.obstacle_gate_min_darkness = (
            self.get_parameter("obstacle_gate_min_darkness").get_parameter_value().double_value
        )
        self.grayscale_preprocessing = (
            self.get_parameter("grayscale_preprocessing").get_parameter_value().bool_value
        )
        self.min_frame_detail = (
            self.get_parameter("min_frame_detail").get_parameter_value().double_value
        )
        self.use_int8_model = self.get_parameter("use_int8_model").get_parameter_value().bool_value
        onnx_encoder_path = (
            self.get_parameter("onnx_encoder_path").get_parameter_value().string_value
        )

        # ── Device ─────────────────────────────────────────────────────
        # torch is imported lazily throughout this node (2026-07-27): the
        # INT8 ONNX backend below is entirely torch-free (numpy tensors via
        # return_tensors="np", onnxruntime for inference, sklearn to
        # normalise), and torch is not installable in the Pi's ROS2 container.
        # A module-level `import torch` therefore crashed this node on every
        # single real-hardware launch, which silently disabled terrain
        # classification -- and with it the whole foundation-model layer this
        # thesis is about -- even when use_int8_model:=true was requested.
        if device_str == "cuda":
            import torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = "cpu"
        self.get_logger().info(f"Device: {self.device}")

        # ── Load DINOv2 ViT-S encoder ──────────────────────────────────
        # The FP32 path keeps HuggingFace's own processor (it needs torch
        # tensors anyway); the ONNX path uses the pinned pure-numpy
        # equivalent so it can run where transformers/torch do not exist.
        self.processor = None
        if not self.use_int8_model:
            from transformers import AutoImageProcessor
            self.processor = AutoImageProcessor.from_pretrained(DINOV2_MODEL_ID)

        if self.use_int8_model:
            if not onnx_encoder_path or not os.path.exists(onnx_encoder_path):
                raise FileNotFoundError(
                    (describe_missing_asset(_REL_ONNX_ENCODER, "onnx_encoder_path")
                     if not onnx_encoder_path
                     else f"INT8 ONNX encoder not found: {onnx_encoder_path}")
                    + "\nRun experiments/int8_quantization_rpi.py to produce it, or set "
                      "use_int8_model:=false to fall back to the FP32 torch model."
                )
            self.get_logger().info(f"Loading INT8 ONNX encoder: {onnx_encoder_path}")
            t0 = time.perf_counter()
            import onnxruntime as ort
            self.onnx_session = ort.InferenceSession(
                onnx_encoder_path, providers=["CPUExecutionProvider"]
            )
            load_s = time.perf_counter() - t0
            size_mb = os.path.getsize(onnx_encoder_path) / 1e6
            self.get_logger().info(f"INT8 ONNX encoder loaded in {load_s:.1f}s | {size_mb:.1f}MB")
            self._extract_features = self._extract_features_onnx
        else:
            self.get_logger().info(f"Loading {DINOV2_MODEL_ID} (frozen, FP32 torch)...")
            t0 = time.perf_counter()
            from transformers import AutoModel
            self.encoder = AutoModel.from_pretrained(DINOV2_MODEL_ID).to(self.device).eval()
            load_s = time.perf_counter() - t0
            n_params = sum(p.numel() for p in self.encoder.parameters()) / 1e6
            self.get_logger().info(f"DINOv2 loaded in {load_s:.1f}s | {n_params:.1f}M params")
            self._extract_features = self._extract_features_torch

        # ── Train LogReg from feature cache ────────────────────────────
        self.get_logger().info(f"Training LogReg from cache: {cache_path}")
        self.clf = self._train_logreg(cache_path, n_shot, class_weight_balanced)

        # ── Warm-up inference (avoid first-call latency spike) ─────────
        dummy = PILImage.fromarray(np.zeros((224, 224, 3), dtype=np.uint8), "RGB")
        self._classify(dummy)
        self.get_logger().info("Warm-up done")

        # ── ROS2 pub / sub ─────────────────────────────────────────────
        self.sub = self.create_subscription(Image, "/camera/image_raw", self._image_callback, 1)
        self.pub = self.create_publisher(String, "/terrain_classification", 10)
        self.pub_probs = self.create_publisher(Float32MultiArray, "/terrain_class_probs", 10)
        self.pub_score = self.create_publisher(Float64, "/traversability_score", 10)
        self.pub_latency = self.create_publisher(Float64, "/inference_latency_ms", 10)
        self.pub_viz = self.create_publisher(Image, "/terrain_viz", 1) if publish_viz else None
        # The camera frame behind each verdict, JPEG-compressed, one per
        # inference rather than one per camera frame. This is what the rosbag
        # records in place of /camera/image_raw: the same evidence, sharing a
        # stamp with the classification it produced, at a fraction of the size.
        # Note JPEG is lossy, so re-running the encoder on a recorded frame
        # will not reproduce the recorded verdict bit for bit -- raise quality
        # if a run is meant to feed a replay-vs-live comparison.
        self.pub_classified_image = self.create_publisher(
            CompressedImage, "/terrain_classified_image", 1
        )
        # Whether the frame behind the latest verdict carried any information.
        # Published explicitly because /traversability_score cannot express it:
        # a frame the blank-frame gate rejected scores 1.0, the same as an
        # impassable rock, so anything downstream reading only the score treats
        # "I cannot see" as "the ground is bad". On 2026-07-29 that made
        # reactive_explorer refuse a heading it had never measured, because a
        # featureless wall 0.28 m away reads as no information: frame_detail is
        # the standard deviation of intensity, and a flat surface filling the
        # frame has almost none.
        self.pub_frame_informative = self.create_publisher(
            Bool, "/terrain_frame_informative", 10
        )

        # ── Safety watchdog ────────────────────────────────────────────
        self.last_img_time = self.get_clock().now()
        self.create_timer(5.0, self._watchdog_callback)

        self.n_processed = 0
        self.n_uninformative = 0
        self.total_ms    = 0.0

        self.get_logger().info(
            f"dinov2_terrain_node ready | backend={'INT8 ONNX' if self.use_int8_model else 'FP32 torch'} | "
            f"threshold={self.threshold:.2f} | "
            f"T*={T_STAR} (calibrated, ECE 0.1695→0.0325) | "
            f"obstacle_gate={self.enable_obstacle_gate} | "
            f"grayscale_preprocessing={self.grayscale_preprocessing} | "
            "subscribed /camera/image_raw → publishing /terrain_classification"
        )

    # ── LogReg training ────────────────────────────────────────────────

    def _train_logreg(
        self, cache_path: str, n_shot: int, class_weight_balanced: bool
    ) -> LogisticRegression:
        clf, fitted = fit_or_load_logreg(
            cache_path, n_shot, class_weight_balanced
        )
        if fitted:
            self.get_logger().info(
                f"LogReg fitted and cached to {logreg_cache_path(cache_path)}"
            )
        else:
            self.get_logger().info(
                f"LogReg loaded from {logreg_cache_path(cache_path)} "
                "(no refit needed)"
            )
        return clf

    # ── Inference ──────────────────────────────────────────────────────

    def _extract_features_torch(self, pil_img: PILImage.Image) -> np.ndarray:
        """FP32 torch backend: HuggingFace DINOv2 CLS token, L2-normalised. [1, 384]"""
        import torch
        inputs = self.processor(images=pil_img, return_tensors="pt").to(self.device)
        with torch.no_grad():
            feat = self.encoder(**inputs).last_hidden_state[:, 0, :]  # CLS token [1, 384]
        return normalize(feat.cpu().numpy(), norm="l2")

    def _extract_features_onnx(self, pil_img: PILImage.Image) -> np.ndarray:
        """INT8 ONNX backend: same processor output, fed through onnxruntime instead
        of torch. The exported graph already L2-normalises internally (see
        _DINOv2RegEncoder in experiments/int8_quantization_rpi.py) -- normalize()
        here is a no-op safety net, not a second real normalisation. [1, 384]"""
        inputs = {"pixel_values": preprocess_dinov2(pil_img)}
        pixel_values = inputs["pixel_values"].astype(np.float32)
        feat = self.onnx_session.run(None, {"pixel_values": pixel_values})[0]
        return normalize(feat, norm="l2")

    def _classify(self, pil_img: PILImage.Image):
        """Extract DINOv2 CLS features → LogReg predict with T*=0.461 calibration. Returns (label, confidence, probs)."""
        feat_np = self._extract_features(pil_img)                      # [1, 384]

        # Temperature-scaled probabilities (§4.7.27): logits/T* then softmax
        logits = self.clf.decision_function(feat_np)[0]               # (n_classes,) raw scores
        scaled = logits / T_STAR
        scaled -= scaled.max()                                         # numerical stability
        exp_s  = np.exp(scaled)
        probs  = exp_s / exp_s.sum()                                  # calibrated (n_classes,)

        # Map clf.classes_ indices to full 4-class array
        full_probs = np.zeros(len(CLASS_NAMES), dtype=np.float32)
        for i, cls in enumerate(self.clf.classes_):
            full_probs[cls] = probs[i]

        pred_idx   = int(full_probs.argmax())
        confidence = float(full_probs[pred_idx])
        return CLASS_NAMES[pred_idx], confidence, full_probs

    def _obstacle_gate(self, rgb_arr: np.ndarray, label: str) -> tuple[bool, dict]:
        """Conservative image gate for dark, edge-rich foreground rock hazards."""
        if not self.enable_obstacle_gate or label != "sand":
            return False, {}

        gray = rgb_arr.astype(np.float32).mean(axis=2)
        h = gray.shape[0]

        gx = np.zeros_like(gray)
        gy = np.zeros_like(gray)
        gx[:, 1:-1] = gray[:, 2:] - gray[:, :-2]
        gy[1:-1, :] = gray[2:, :] - gray[:-2, :]
        mag = np.hypot(gx, gy)

        upper_edge = float(mag[: int(h * 0.65), :].mean())
        p90 = float(np.percentile(gray, 90))
        mean = float(gray.mean())
        # Two trigger paths:
        # Path A — close/large rocks: high edge density in upper frame (rock_cluster)
        # Path B — distant/dark rocks: very dark frame overall, rover body + rocks (boulder_zone)
        edge_trigger     = upper_edge >= self.obstacle_gate_min_upper_edge
        darkness_trigger = mean <= self.obstacle_gate_min_darkness
        triggered = (
            (edge_trigger or darkness_trigger)
            and p90 <= self.obstacle_gate_max_p90
            and mean <= self.obstacle_gate_max_mean
        )

        return triggered, {
            "upper_edge": upper_edge, "p90": p90, "mean": mean,
            "edge_trigger": edge_trigger, "darkness_trigger": darkness_trigger,
        }

    # ── ROS2 callbacks ─────────────────────────────────────────────────

    def _publish_uninformative(self, detail: float) -> None:
        """Publish an explicit uncertain reading for a frame not worth classifying.

        Every topic the normal path publishes is published here too, so a
        subscriber sees a fresh uncertain reading rather than a silence it
        would have to distinguish from a dead node. Traversability is set to
        1.0, maximum risk, matching what traversability_score() returns for any
        sub-threshold confidence.
        """
        self.n_uninformative += 1

        informative_msg = Bool()
        informative_msg.data = False
        self.pub_frame_informative.publish(informative_msg)

        out_msg = String()
        out_msg.data = "uncertain:0.000"
        self.pub.publish(out_msg)

        probs_msg = Float32MultiArray()
        probs_msg.data = [0.0] * len(CLASS_NAMES)
        self.pub_probs.publish(probs_msg)

        score_msg = Float64()
        score_msg.data = 1.0
        self.pub_score.publish(score_msg)

        # Warn on the first frame and then rarely, so a covered lens is visible
        # in the log without burying every other message at camera frame rate.
        if self.n_uninformative == 1 or self.n_uninformative % 50 == 0:
            self.get_logger().warn(
                f"frame carries no information (detail {detail:.3f} < "
                f"{self.min_frame_detail}); publishing uncertain without "
                f"classifying. Camera disconnected, lens covered, or exposure "
                f"blown out? [{self.n_uninformative} such frames]"
            )

    def _image_callback(self, msg: Image) -> None:
        self.last_img_time = self.get_clock().now()

        # ROS Image → PIL (no cv_bridge — avoids NumPy 2.x incompatibility)
        np_arr = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, -1))
        if msg.encoding in ("bgr8", "bgra8"):
            np_arr = np_arr[:, :, :3][:, :, ::-1].copy()
        else:
            np_arr = np_arr[:, :, :3].copy()

        # Reject frames with no information before classifying them. Measured
        # 2026-07-28 on this exact INT8 path: pure black scores soil 0.913,
        # white 0.699, grey 0.778, flat red 0.510 -- all above self.threshold,
        # so publishing them would report safe ground for a camera that is
        # disconnected, covered, or blown out, and nothing downstream would
        # stop the rover. The model is not hesitant about these images, it is
        # confidently wrong, so no confidence threshold can catch them.
        if not frame_is_informative(np_arr, self.min_frame_detail):
            self._publish_uninformative(frame_detail(np_arr))
            return

        classify_arr = to_grayscale_3ch(np_arr) if self.grayscale_preprocessing else np_arr
        pil_img = PILImage.fromarray(classify_arr, "RGB")

        t0 = time.perf_counter()
        label, confidence, probs = self._classify(pil_img)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        gate_triggered, gate_stats = self._obstacle_gate(np_arr, label)
        if gate_triggered:
            label = self.obstacle_gate_label
            confidence = max(confidence, self.threshold)

        self.n_processed += 1
        self.total_ms    += elapsed_ms

        out_msg      = String()
        out_msg.data = (
            f"{label}:{confidence:.3f}"
            if confidence >= self.threshold
            else f"uncertain:{confidence:.3f}"
        )
        self.pub.publish(out_msg)

        probs_msg = Float32MultiArray()
        probs_msg.data = [float(p) for p in probs]
        self.pub_probs.publish(probs_msg)

        score_msg = Float64()
        score_msg.data = 1.0 if gate_triggered else traversability_score(probs, confidence, self.threshold)
        self.pub_score.publish(score_msg)

        lat_msg = Float64()
        lat_msg.data = elapsed_ms
        self.pub_latency.publish(lat_msg)

        if self.n_processed % 10 == 0:
            avg_ms = self.total_ms / self.n_processed
            if gate_triggered:
                gate_info = (
                    " | GATE edge={:.2f}({}) p90={:.1f} mean={:.1f}({})".format(
                        gate_stats["upper_edge"],
                        "E" if gate_stats["edge_trigger"] else "-",
                        gate_stats["p90"],
                        gate_stats["mean"],
                        "D" if gate_stats["darkness_trigger"] else "-",
                    )
                )
            elif gate_stats:
                gate_info = (
                    " | gate-miss edge={:.2f} p90={:.1f} mean={:.1f}".format(
                        gate_stats["upper_edge"], gate_stats["p90"], gate_stats["mean"]
                    )
                )
            else:
                gate_info = ""
            self.get_logger().info(
                f"[{self.n_processed}] {out_msg.data} | {elapsed_ms:.0f}ms avg {avg_ms:.0f}ms"
                f"{gate_info}"
            )

        # The run's visual record, published after the verdict so the two share
        # a stamp and can be paired against /terrain_classification.
        #
        # This is the camera's RGB frame, NOT the array handed to the encoder:
        # with grayscale_preprocessing on (the real-hardware default) the model
        # sees to_grayscale_3ch() of this. Recording the colour original keeps
        # more for later inspection, and the grey version is one function call
        # away from it.
        informative_msg = Bool()
        informative_msg.data = True
        self.pub_frame_informative.publish(informative_msg)

        frame_msg = CompressedImage()
        frame_msg.header.stamp = self.get_clock().now().to_msg()
        frame_msg.header.frame_id = msg.header.frame_id
        frame_msg.format = "jpeg"
        frame_msg.data = encode_jpeg(np_arr)
        self.pub_classified_image.publish(frame_msg)

        if self.pub_viz is not None:
            self._publish_viz(np_arr, label, confidence, probs)

    def _publish_viz(self, rgb_arr: np.ndarray, label: str,
                     confidence: float, probs: np.ndarray) -> None:
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
        draw.text((10, 10), "DINOv2 ViT-S  90.24%", fill=(200, 200, 200))
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

        out_arr       = np.asarray(pil_viz)
        viz_msg       = Image()
        viz_msg.height      = out_arr.shape[0]
        viz_msg.width       = out_arr.shape[1]
        viz_msg.encoding    = "rgb8"
        viz_msg.is_bigendian = False
        viz_msg.step        = out_arr.shape[1] * 3
        viz_msg.data        = out_arr.tobytes()
        self.pub_viz.publish(viz_msg)

    def _watchdog_callback(self) -> None:
        elapsed = (self.get_clock().now() - self.last_img_time).nanoseconds / 1e9
        if elapsed > 5.0:
            msg      = String()
            msg.data = "unknown:0.000"
            self.pub.publish(msg)
            self.get_logger().warn(f"No image for {elapsed:.1f}s — publishing unknown")


def main(args=None):
    rclpy.init(args=args)
    node = DINOv2TerrainNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
