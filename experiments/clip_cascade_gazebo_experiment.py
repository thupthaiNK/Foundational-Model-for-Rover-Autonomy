"""
Purpose: X5 CLIP Cascade Gazebo Experiment — compare CLIP zero-shot vs DINOv2 probe
         on 5 Gazebo terrain zones, then evaluate a confidence-gated cascade.
         Cascade logic: if CLIP top-1 confidence > CLIP_THRESH → use CLIP prediction;
         else fall back to DINOv2 LogReg probe prediction.
         Lighting note: experiment runs under Gazebo default lighting (noon-equivalent).
         CLIP text prompts are lighting-invariant by design; DINOv2 uses visual features.
Inputs:  /terrain_classification  (std_msgs/String)  DINOv2+LogReg running in background
         /exomy/camera/image_raw  (sensor_msgs/Image) Gazebo camera
         /delete_entity, /spawn_entity               Gazebo teleport services
Outputs: experiments/results/clip_cascade_gazebo.csv
         experiments/results/figures/clip_cascade_gazebo.png
How to run:
    # Terminal 1: Gazebo + DINOv2 already running (run_exp_launch.sh)
    # Terminal 2:
    source /opt/ros/humble/setup.bash && source ros2_ws/install/setup.bash
    python3 experiments/clip_cascade_gazebo_experiment.py
Project: Onboard Visual Foundation Models for Mars Terrain Perception and Rover Navigation
"""

import csv, os, time
import numpy as np
import torch
import clip
from PIL import Image

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image as RosImage
from gazebo_msgs.srv import SpawnEntity, DeleteEntity
import xacro

# ── Config ─────────────────────────────────────────────────────────────────────
CLIP_MODEL      = "ViT-B/32"
CLIP_THRESH     = 0.50      # cascade: use CLIP if top-1 prob > this; else DINOv2
DINOV2_THRESH   = 0.40      # uncertain if DINOv2 conf < this (matches deployment)
N_READINGS      = 8         # DINOv2 readings per zone
SETTLE_S        = 3.5       # seconds after respawn

CLIP_LABELS  = ["soil", "bedrock", "sand", "big_rock"]
CLIP_PROMPTS = [
    "fine-grained loose soil on Mars surface",
    "flat smooth bedrock rock surface on Mars",
    "bright sandy terrain or sand dunes on Mars",
    "large boulders or big rocks on Mars terrain",
]

POLICY = {"soil": "SAFE", "bedrock": "HAZARD", "sand": "CAUTION",
          "big_rock": "STOP", "uncertain": "STOP"}
SAFE_LABELS = {"soil", "sand"}   # SAFE or CAUTION → passes safety

ZONE_POSES = [
    {"name": "soil_zone",    "x": -7.5, "y":  6.0, "z": 0.15, "gt": "soil"},
    {"name": "bedrock_zone", "x":  7.5, "y":  6.0, "z": 0.15, "gt": "bedrock"},
    {"name": "sand_zone",    "x": -7.5, "y": -6.0, "z": 0.15, "gt": "sand"},
    {"name": "rock_cluster", "x":  2.0, "y": -4.0, "z": 0.15, "gt": "big_rock"},
    {"name": "boulder_zone", "x":  2.0, "y": -8.0, "z": 0.15, "gt": "big_rock"},
]

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

SIM_DIR   = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "simulation"))
URDF_PATH = os.path.join(SIM_DIR, "urdf", "exomy.urdf.xacro")


# ── ROS2 node ──────────────────────────────────────────────────────────────────
class CLIPCascadeNode(Node):
    def __init__(self):
        super().__init__("clip_cascade_gazebo")
        self._dinov2_terrain = "unknown"
        self._dinov2_conf    = 0.0
        self._latest_image   = None
        self._urdf_xml       = xacro.process_file(URDF_PATH).toxml()

        self.create_subscription(String,   "/terrain_classification", self._terrain_cb, 10)
        self.create_subscription(RosImage, "/exomy/camera/image_raw", self._image_cb,  10)

        self._delete_client = self.create_client(DeleteEntity, "/delete_entity")
        self._spawn_client  = self.create_client(SpawnEntity,  "/spawn_entity")

    def _terrain_cb(self, msg):
        parts = msg.data.split(":")
        self._dinov2_terrain = parts[0].strip().lower() if parts else "uncertain"
        try:
            self._dinov2_conf = float(parts[1]) if len(parts) > 1 else 0.0
        except ValueError:
            self._dinov2_conf = 0.0

    def _image_cb(self, msg):
        try:
            # Convert sensor_msgs/Image to numpy then PIL
            enc = msg.encoding.lower()
            arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
            if enc in ("bgr8", "bgra8"):
                arr = arr[:, :, :3][:, :, ::-1]   # BGR→RGB
            elif enc in ("rgb8", "rgba8"):
                arr = arr[:, :, :3]
            elif enc == "mono8":
                arr = np.stack([arr[:, :, 0]] * 3, axis=2)
            self._latest_image = Image.fromarray(arr.astype(np.uint8))
        except Exception as e:
            self.get_logger().warn(f"Image conversion failed: {e}")

    def teleport(self, x, y, z) -> bool:
        if not self._delete_client.wait_for_service(timeout_sec=10.0):
            return False
        del_req = DeleteEntity.Request()
        del_req.name = "exomy"
        for attempt in range(2):
            future = self._delete_client.call_async(del_req)
            rclpy.spin_until_future_complete(self, future, timeout_sec=15.0)
            if future.done() and future.result().success:
                break
            if attempt == 0:
                time.sleep(2.0)
        time.sleep(1.5)

        if not self._spawn_client.wait_for_service(timeout_sec=10.0):
            return False
        req = SpawnEntity.Request()
        req.name = "exomy"
        req.xml  = self._urdf_xml
        req.initial_pose.position.x    = float(x)
        req.initial_pose.position.y    = float(y)
        req.initial_pose.position.z    = float(z)
        req.initial_pose.orientation.w = 1.0
        future = self._spawn_client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=10.0)
        return future.done() and future.result().success

    def collect_dinov2(self, n: int) -> tuple[str, float]:
        from collections import Counter
        readings = []
        confs    = []
        deadline = time.time() + n * 1.2 + 5.0
        while len(readings) < n and time.time() < deadline:
            rclpy.spin_once(self, timeout_sec=0.5)
            if self._dinov2_terrain not in ("unknown",):
                readings.append(self._dinov2_terrain)
                confs.append(self._dinov2_conf)
        if not readings:
            return "uncertain", 0.0
        mode = Counter(readings).most_common(1)[0][0]
        mode_confs = [confs[i] for i, l in enumerate(readings) if l == mode]
        return mode, round(sum(mode_confs) / len(mode_confs), 3)

    def capture_frame(self) -> Image.Image | None:
        self._latest_image = None
        deadline = time.time() + 5.0
        while self._latest_image is None and time.time() < deadline:
            rclpy.spin_once(self, timeout_sec=0.2)
        return self._latest_image


# ── CLIP inference ─────────────────────────────────────────────────────────────
def clip_classify(model, preprocess, text_features, image: Image.Image, device) -> tuple[str, float, list[float]]:
    img_tensor  = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        img_features = model.encode_image(img_tensor)
        img_features = img_features / img_features.norm(dim=-1, keepdim=True)
        logits = (100.0 * img_features @ text_features.T).softmax(dim=-1)
    probs = logits[0].cpu().numpy()
    top_i = int(np.argmax(probs))
    return CLIP_LABELS[top_i], float(probs[top_i]), [float(p) for p in probs]


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    rclpy.init()
    node = CLIPCascadeNode()

    device = "cpu"   # CLIP float16 conv2d fails on this CUDA build; CPU is sufficient
    node.get_logger().info(f"Loading CLIP {CLIP_MODEL} on {device}...")
    clip_model, clip_preprocess = clip.load(CLIP_MODEL, device=device)
    clip_model = clip_model.float()
    clip_model.eval()

    with torch.no_grad():
        text_tokens  = clip.tokenize(CLIP_PROMPTS).to(device)
        text_features = clip_model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    node.get_logger().info("CLIP loaded. Waiting for DINOv2 terrain_classification...")
    deadline = time.time() + 30.0
    while node._dinov2_terrain == "unknown" and time.time() < deadline:
        rclpy.spin_once(node, timeout_sec=0.5)

    results = []
    print(f"\nX5 CLIP Cascade — 5 Gazebo zones (CLIP_THRESH={CLIP_THRESH})")
    print("=" * 70)

    for zone in ZONE_POSES:
        print(f"\n[Zone] {zone['name']}  gt={zone['gt']}  pos=({zone['x']},{zone['y']})")

        ok = node.teleport(zone["x"], zone["y"], zone["z"])
        if not ok:
            node.get_logger().warn("  Teleport failed")

        time.sleep(SETTLE_S)

        # DINOv2 prediction (from running node on /terrain_classification)
        dinov2_label, dinov2_conf = node.collect_dinov2(N_READINGS)
        if dinov2_conf < DINOV2_THRESH:
            dinov2_label = "uncertain"

        # CLIP zero-shot on captured frame
        frame = node.capture_frame()
        if frame is not None:
            clip_label, clip_conf, clip_probs = clip_classify(
                clip_model, clip_preprocess, text_features, frame, device
            )
        else:
            clip_label, clip_conf, clip_probs = "uncertain", 0.0, [0.0] * 4
            node.get_logger().warn("  No camera frame received")

        # Cascade decision
        cascade_label = clip_label if clip_conf >= CLIP_THRESH else dinov2_label
        cascade_source = "CLIP" if clip_conf >= CLIP_THRESH else "DINOv2"

        gt = zone["gt"]
        row = {
            "zone":              zone["name"],
            "ground_truth":      gt,
            # DINOv2
            "dinov2_pred":       dinov2_label,
            "dinov2_conf":       dinov2_conf,
            "dinov2_correct":    dinov2_label == gt,
            "dinov2_safe":       dinov2_label in SAFE_LABELS or (gt in ("big_rock",) and dinov2_label not in SAFE_LABELS),
            # CLIP
            "clip_pred":         clip_label,
            "clip_conf":         round(clip_conf, 3),
            "clip_correct":      clip_label == gt,
            "clip_prob_soil":    round(clip_probs[0], 3),
            "clip_prob_bedrock": round(clip_probs[1], 3),
            "clip_prob_sand":    round(clip_probs[2], 3),
            "clip_prob_bigrock": round(clip_probs[3], 3),
            # Cascade
            "cascade_pred":      cascade_label,
            "cascade_source":    cascade_source,
            "cascade_correct":   cascade_label == gt,
        }
        results.append(row)

        print(f"  DINOv2:  {dinov2_label:12s} conf={dinov2_conf:.3f}  correct={row['dinov2_correct']}")
        print(f"  CLIP:    {clip_label:12s} conf={clip_conf:.3f}  correct={row['clip_correct']}")
        print(f"           probs: soil={clip_probs[0]:.2f} bed={clip_probs[1]:.2f} sand={clip_probs[2]:.2f} rock={clip_probs[3]:.2f}")
        print(f"  Cascade: {cascade_label:12s} (used {cascade_source})  correct={row['cascade_correct']}")

    # Summary
    n = len(results)
    if n:
        d_acc = sum(r["dinov2_correct"]  for r in results)
        c_acc = sum(r["clip_correct"]    for r in results)
        x_acc = sum(r["cascade_correct"] for r in results)
        print(f"\n{'='*70}")
        print(f"RESULTS SUMMARY ({n} zones)")
        print(f"  DINOv2 terrain accuracy:  {d_acc}/{n}")
        print(f"  CLIP   terrain accuracy:  {c_acc}/{n}")
        print(f"  Cascade terrain accuracy: {x_acc}/{n}")
        print(f"  Cascade used CLIP:  {sum(1 for r in results if r['cascade_source']=='CLIP')}/{n}")
        print(f"  Cascade used DINOv2:{sum(1 for r in results if r['cascade_source']=='DINOv2')}/{n}")

    # Save CSV
    csv_path = os.path.join(RESULTS_DIR, "clip_cascade_gazebo.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)
    node.get_logger().info(f"Saved → {csv_path}")

    # Plot
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        zone_names = [r["zone"].replace("_", "\n") for r in results]
        x = np.arange(len(results))
        width = 0.28

        # Accuracy per zone
        d_c = [1 if r["dinov2_correct"]  else 0 for r in results]
        c_c = [1 if r["clip_correct"]    else 0 for r in results]
        x_c = [1 if r["cascade_correct"] else 0 for r in results]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle("X5 CLIP Cascade vs DINOv2 — Gazebo 5-Zone Comparison", fontsize=13)

        # Bar chart: correct/incorrect per zone
        ax1.bar(x - width, d_c, width, label="DINOv2",  color="#3498db", alpha=0.85, edgecolor="k")
        ax1.bar(x,          c_c, width, label="CLIP",    color="#e67e22", alpha=0.85, edgecolor="k")
        ax1.bar(x + width,  x_c, width, label="Cascade", color="#2ecc71", alpha=0.85, edgecolor="k")
        ax1.set_xticks(x)
        ax1.set_xticklabels(zone_names, fontsize=8)
        ax1.set_yticks([0, 1])
        ax1.set_yticklabels(["Wrong", "Correct"])
        ax1.set_title("Terrain classification correct (1) / wrong (0)")
        ax1.legend(fontsize=9)
        ax1.grid(True, axis="y", alpha=0.4)

        # Confidence comparison
        d_confs = [r["dinov2_conf"] for r in results]
        c_confs = [r["clip_conf"]   for r in results]
        ax2.bar(x - width/2, d_confs, width, label="DINOv2 conf", color="#3498db", alpha=0.85, edgecolor="k")
        ax2.bar(x + width/2, c_confs, width, label="CLIP conf",   color="#e67e22", alpha=0.85, edgecolor="k")
        ax2.axhline(CLIP_THRESH,    color="#e67e22", linestyle="--", linewidth=1.2,
                    label=f"CLIP cascade thresh ({CLIP_THRESH})")
        ax2.axhline(DINOV2_THRESH,  color="#3498db", linestyle="--", linewidth=1.2,
                    label=f"DINOv2 uncertain thresh ({DINOV2_THRESH})")
        ax2.set_xticks(x)
        ax2.set_xticklabels(zone_names, fontsize=8)
        ax2.set_ylim(0, 1.05)
        ax2.set_ylabel("Confidence")
        ax2.set_title("Confidence per zone")
        ax2.legend(fontsize=8)
        ax2.grid(True, axis="y", alpha=0.4)

        plt.tight_layout()
        fig_path = os.path.join(FIGURES_DIR, "clip_cascade_gazebo.png")
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        node.get_logger().info(f"Figure → {fig_path}")
        plt.close()
    except Exception as e:
        node.get_logger().warn(f"Plot failed: {e}")

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
