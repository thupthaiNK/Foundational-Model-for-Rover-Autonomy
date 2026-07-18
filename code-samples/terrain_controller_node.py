#!/usr/bin/env python3
"""
Purpose: Reactive terrain controller — subscribes to /terrain_classification from
         dinov2_terrain_node (or clip_terrain_node) and publishes velocity commands
         to /exomy/cmd_vel. Implements a rule-based traversability policy:
           soil    (conf ≥ threshold) → forward 0.10 m/s  [safe]
           sand    (conf ≥ threshold) → forward 0.05 m/s  [caution]
           bedrock (conf ≥ threshold) → forward 0.03 m/s  [hazard — very slow]
           uncertain / low-confidence → stop               [unknown terrain]
           big_rock / unknown         → stop               [obstacle or no data]
         This is Layer 3 (Traversability) of the 5-layer autonomy stack — reactive
         reflexes with no map, no planning, no localization.
         Also publishes /terrain_controller/stopped (true whenever current_speed==0
         due to a hazard label/latch/resume-confirm, independent of the measurement
         freeze) and cedes /exomy/cmd_vel to reactive_explorer_node whenever
         /reactive_explorer/active is true, so the two nodes never publish cmd_vel
         at the same time. If /reactive_explorer/active is never published (e.g. the
         older dinov2_controller_test.launch.py), it defaults false and behavior is
         unchanged from before this handoff was added.
         Also cedes cmd_vel to stuck_detection_node.py whenever
         /stuck_detection/active is true (highest priority of the three --
         wheels spinning uselessly in sand can happen regardless of which
         node is driving, so its handshake is checked in addition to, not
         instead of, /reactive_explorer/active).
         Also cedes to l5_lite_planner_node.py (backlog "L5-lite", scoped
         via grill-thesis 2026-07-17) whenever /l5_lite_planner/active is
         true -- lowest priority of the four driving nodes; l5_lite_planner
         itself cedes to reactive_explorer_node/stuck_detection_node first,
         so this handshake only matters once neither of those is active.
         Optional continuous-speed mode (use_continuous_score, default False --
         every previously reported Gazebo result in this thesis used the
         discrete POLICY table above and is unaffected): when enabled, speed
         is instead computed from dinov2_terrain_node.py's /traversability_score
         (thesis Ch5 §5.6.2) as v = v_max * (1 - T_score), smoothed over a
         short sliding window to damp single-frame noise, so a near-boundary
         reading gets a proportionally reduced speed instead of snapping to
         one discrete class's fixed speed or the nearest hazard STOP. The
         discrete label pathway still runs unchanged for logging/telemetry
         and for /terrain_controller/stopped.
         score_source (default "dinov2") selects which continuous-score
         topic is read when use_continuous_score is enabled: "dinov2"
         (default, /traversability_score, DINOv2-only) or "fused"
         (/traversability_score_fused, from traversability_score_fusion_node.py
         -- adds LiDAR range + IMU tilt risk terms, backlog item 8, scoped
         via grill-thesis 2026-07-17). Same pattern as pose_source in the
         SLAM live-costmap work: opt-in, default preserves prior behavior.
Inputs:  /terrain_classification (std_msgs/String)  "label:confidence"
         /reactive_explorer/active (std_msgs/Bool)   true while explorer owns cmd_vel
         /traversability_score or /traversability_score_fused (std_msgs/Float64)
             only read if use_continuous_score; topic selected by score_source
Outputs: /exomy/cmd_vel (geometry_msgs/Twist)       forward velocity commands
         /terrain_controller/stopped (std_msgs/Bool) true while hazard-stopped
How to run:
    source /opt/ros/humble/setup.bash
    source ros2_ws/install/setup.bash
    ros2 run fm_perception terrain_controller_node.py
    # Change cmd_vel topic for real ExoMy hardware:
    ros2 run fm_perception terrain_controller_node.py --ros-args -p cmd_vel_topic:=/cmd_vel
Project: Foundational Model for Rover Autonomy
"""

from geometry_msgs.msg import Twist
from std_msgs.msg import Bool, Float64, String
import rclpy
from rclpy.node import Node

# Traversability policy — speed in m/s
POLICY = {
    "soil":     0.10,   # safe loose regolith — full speed
    "sand":     0.05,   # caution — loose granular, wheel slip risk
    "bedrock":  0.03,   # hazard — hard uneven surface, minimal speed
    "big_rock": 0.00,   # obstacle — full stop
    "uncertain":0.00,   # low confidence — full stop (safety-first)
    "unknown":  0.00,   # no data — full stop (watchdog triggered)
}

TRAVERSABILITY_LABEL = {
    "soil":     "SAFE",
    "sand":     "CAUTION",
    "bedrock":  "HAZARD",
    "big_rock": "STOP — obstacle",
    "uncertain":"STOP — uncertain",
    "unknown":  "STOP — no data",
}


def score_topic_for(score_source: str) -> str:
    """Map the score_source param ("dinov2" default | "fused") to the topic
    subscribed to for continuous-score mode. Unknown values fall back to
    "dinov2" (the default, zero-risk topic) rather than raising, so a typo
    degrades to existing behavior instead of silently subscribing to
    nothing. See traversability_score_fusion_node.py (backlog item 8,
    scoped via grill-thesis 2026-07-17) for what "fused" adds: LiDAR range
    + IMU tilt risk terms, max-combined with the DINOv2-only score."""
    return "/traversability_score_fused" if score_source == "fused" else "/traversability_score"


def continuous_speed(t_score: float, v_max: float) -> float:
    """v = v_max * (1 - T_score), clamped to [0, v_max].

    T_score is bounded in [0,1] by construction in dinov2_terrain_node.py,
    but clamp defensively regardless of the caller. At the same risk weights
    dinov2_terrain_node.py uses (soil=0.0, sand=0.5, bedrock=0.7,
    big_rock=1.0), a confident single-class score reproduces the discrete
    POLICY speed for that class exactly; a mixed/ambiguous score gets a
    genuinely intermediate speed instead of snapping to one class.
    """
    t_score = max(0.0, min(1.0, t_score))
    return v_max * (1.0 - t_score)


class ScoreSmoother:
    """Simple moving average over the last `window` values.

    Damps single-frame noise in /traversability_score before it reaches
    continuous_speed(), per thesis Ch5 §5.6.2 ("temporal smoothing over a
    3-5 frame sliding window").
    """

    def __init__(self, window: int = 4):
        self.window = window
        self._values = []

    def update(self, value: float) -> float:
        self._values.append(value)
        if len(self._values) > self.window:
            self._values.pop(0)
        return sum(self._values) / len(self._values)


class TerrainControllerNode(Node):

    def __init__(self):
        super().__init__("terrain_controller_node")

        # ── Parameters ─────────────────────────────────────────────────
        self.declare_parameter("cmd_vel_topic",        "/exomy/cmd_vel")
        self.declare_parameter("terrain_topic",        "/terrain_classification")
        self.declare_parameter("confidence_threshold", 0.60)
        self.declare_parameter("speed_soil",           0.10)
        self.declare_parameter("speed_sand",           0.05)
        self.declare_parameter("speed_bedrock",        0.03)
        self.declare_parameter("max_uncertain_count",  5)    # consecutive stops before E-stop log
        self.declare_parameter("publish_rate_hz",      10.0) # /cmd_vel publish rate
        self.declare_parameter("stale_timeout_s",      3.0)  # seconds before watchdog stops rover
        self.declare_parameter("stop_latch_s",         1.0)  # hold STOP briefly after hazardous flicker
        self.declare_parameter("resume_confirm_count", 2)    # consecutive go labels before releasing STOP
        self.declare_parameter("use_continuous_score", False)  # opt-in; default preserves all prior results
        self.declare_parameter("score_smoothing_window", 4)    # frames averaged, per thesis Ch5 §5.6.2
        self.declare_parameter("score_source", "dinov2")       # "dinov2" (default) or "fused" -- backlog item 8

        cmd_vel_topic    = self.get_parameter("cmd_vel_topic").get_parameter_value().string_value
        terrain_topic    = self.get_parameter("terrain_topic").get_parameter_value().string_value
        self.threshold   = self.get_parameter("confidence_threshold").get_parameter_value().double_value
        self.max_unc     = self.get_parameter("max_uncertain_count").get_parameter_value().integer_value
        publish_hz       = self.get_parameter("publish_rate_hz").get_parameter_value().double_value
        self.stale_s     = self.get_parameter("stale_timeout_s").get_parameter_value().double_value
        self.stop_latch_s = self.get_parameter("stop_latch_s").get_parameter_value().double_value
        self.resume_confirm_count = (
            self.get_parameter("resume_confirm_count").get_parameter_value().integer_value
        )
        self.use_continuous_score = self.get_parameter("use_continuous_score").value
        score_window = self.get_parameter("score_smoothing_window").value
        score_source = self.get_parameter("score_source").get_parameter_value().string_value
        score_topic = score_topic_for(score_source)

        # Override policy speeds from parameters
        POLICY["soil"]    = self.get_parameter("speed_soil").get_parameter_value().double_value
        POLICY["sand"]    = self.get_parameter("speed_sand").get_parameter_value().double_value
        POLICY["bedrock"] = self.get_parameter("speed_bedrock").get_parameter_value().double_value

        self._score_smoother = ScoreSmoother(window=score_window)

        # ── State ──────────────────────────────────────────────────────
        self.current_label      = "unknown"
        self.current_confidence = 0.0
        self.current_speed      = 0.0
        self.uncertain_count    = 0
        self.last_msg_time      = self.get_clock().now()
        self.prev_label         = None   # for change-detection logging
        self._frozen            = False  # set True by experiment to freeze rover during classification
        self._stop_latch_until_ns = 0
        self._resume_candidate_count = 0
        self._explorer_active = False
        self._stuck_detection_active = False
        self._l5_lite_active = False

        # ── ROS2 pub / sub ─────────────────────────────────────────────
        self.sub = self.create_subscription(
            String, terrain_topic, self._terrain_callback, 10
        )
        self.sub_freeze = self.create_subscription(
            Bool, "/measurement_mode", self._freeze_cb, 10
        )
        self.sub_explorer_active = self.create_subscription(
            Bool, "/reactive_explorer/active", self._explorer_active_cb, 10
        )
        self.sub_stuck_active = self.create_subscription(
            Bool, "/stuck_detection/active", self._stuck_detection_active_cb, 10
        )
        self.sub_l5_lite_active = self.create_subscription(
            Bool, "/l5_lite_planner/active", self._l5_lite_active_cb, 10
        )
        self.sub_score = self.create_subscription(
            Float64, score_topic, self._score_callback, 10
        )
        self.pub = self.create_publisher(Twist, cmd_vel_topic, 10)
        self.pub_stopped = self.create_publisher(Bool, "/terrain_controller/stopped", 10)

        # Publish cmd_vel at fixed rate (not just on each terrain msg)
        self.create_timer(1.0 / publish_hz, self._publish_cmd_vel)

        # Watchdog: stop if no terrain classification for stale_timeout_s
        self.create_timer(1.0, self._watchdog_callback)

        self.get_logger().info(
            f"terrain_controller_node ready\n"
            f"  Subscribed: {terrain_topic}\n"
            f"  Publishing: {cmd_vel_topic}\n"
            f"  Policy: soil {POLICY['soil']} m/s | sand {POLICY['sand']} m/s | "
            f"bedrock {POLICY['bedrock']} m/s | uncertain STOP\n"
            f"  Confidence threshold: {self.threshold:.2f}\n"
            f"  Stop latch: {self.stop_latch_s:.1f}s | "
            f"resume confirm: {self.resume_confirm_count}\n"
            f"  Continuous score mode: {self.use_continuous_score} "
            f"(window={score_window}, source={score_source} -> {score_topic})"
        )

    # ── Measurement freeze callback ────────────────────────────────────

    def _freeze_cb(self, msg: Bool) -> None:
        """Freeze rover at 0 m/s during experiment classification phase."""
        self._frozen = msg.data
        self.get_logger().info(
            "Controller FROZEN for measurement" if msg.data
            else "Controller UNFROZEN — resuming traversability policy"
        )

    def _score_callback(self, msg: Float64) -> None:
        smoothed = self._score_smoother.update(msg.data)
        if self.use_continuous_score:
            # POLICY["soil"] is the v_max reference (see continuous_speed()
            # docstring: a confident single-class score reproduces the
            # matching discrete POLICY speed exactly at this v_max choice).
            self.current_speed = continuous_speed(smoothed, POLICY["soil"])

    def _explorer_active_cb(self, msg: Bool) -> None:
        if msg.data != self._explorer_active:
            self.get_logger().info(
                "Ceding /exomy/cmd_vel to reactive_explorer_node" if msg.data
                else "reactive_explorer_node released cmd_vel — resuming policy"
            )
        self._explorer_active = msg.data

    def _stuck_detection_active_cb(self, msg: Bool) -> None:
        if msg.data != self._stuck_detection_active:
            self.get_logger().info(
                "Ceding /exomy/cmd_vel to stuck_detection_node" if msg.data
                else "stuck_detection_node released cmd_vel — resuming policy"
            )
        self._stuck_detection_active = msg.data

    def _l5_lite_active_cb(self, msg: Bool) -> None:
        if msg.data != self._l5_lite_active:
            self.get_logger().info(
                "Ceding /exomy/cmd_vel to l5_lite_planner_node" if msg.data
                else "l5_lite_planner_node released cmd_vel — resuming policy"
            )
        self._l5_lite_active = msg.data

    # ── Terrain classification callback ────────────────────────────────

    def _terrain_callback(self, msg: String) -> None:
        now = self.get_clock().now()
        now_ns = now.nanoseconds
        self.last_msg_time = now

        # Parse "label:confidence" format
        parts = msg.data.split(":")
        if len(parts) != 2:
            self.get_logger().warn(f"Unexpected message format: '{msg.data}'")
            return

        label = parts[0].strip()
        try:
            confidence = float(parts[1])
        except ValueError:
            self.get_logger().warn(f"Cannot parse confidence in: '{msg.data}'")
            return

        # Apply confidence threshold — low-confidence → uncertain
        if confidence < self.threshold:
            label = "uncertain"

        requested_speed = POLICY.get(label, 0.00)

        # Track consecutive uncertain/stop events
        if requested_speed == 0.0:
            self.uncertain_count += 1
            self._resume_candidate_count = 0
            self._stop_latch_until_ns = now_ns + int(self.stop_latch_s * 1e9)
        else:
            self.uncertain_count = 0
            self._resume_candidate_count += 1

        latch_active = now_ns < self._stop_latch_until_ns
        resume_confirmed = self._resume_candidate_count >= self.resume_confirm_count
        speed = requested_speed if (requested_speed > 0.0 and not latch_active and resume_confirmed) else 0.0

        self.current_label      = label
        self.current_confidence = confidence
        if not self.use_continuous_score:
            self.current_speed = speed
        # else: current_speed is driven by _score_callback() instead --
        # label/confidence above are still updated for logging/telemetry
        # and for /terrain_controller/stopped's underlying current_speed,
        # which _score_callback() sets directly in that mode.

        # Log only on label change (avoids log spam at 10 Hz)
        if label != self.prev_label:
            trav = TRAVERSABILITY_LABEL.get(label, "UNKNOWN")
            self.get_logger().info(
                f"Terrain: {label} (conf={confidence:.2f})  →  "
                f"{trav}  speed={speed:.2f} m/s"
            )
            self.prev_label = label

        # Log E-stop trigger
        if self.uncertain_count == self.max_unc:
            self.get_logger().warn(
                f"E-STOP: {self.uncertain_count} consecutive uncertain/stop decisions — "
                "rover halted until confident classification resumes"
            )

    # ── Periodic cmd_vel publisher ─────────────────────────────────────

    def _publish_cmd_vel(self) -> None:
        stopped = Bool()
        stopped.data = (self.current_speed == 0.0)
        self.pub_stopped.publish(stopped)

        if self._explorer_active or self._stuck_detection_active or self._l5_lite_active:
            return  # ceded to reactive_explorer_node, stuck_detection_node, or l5_lite_planner_node

        twist = Twist()
        twist.linear.x  = 0.0 if self._frozen else self.current_speed
        twist.linear.y  = 0.0
        twist.linear.z  = 0.0
        twist.angular.x = 0.0
        twist.angular.y = 0.0
        twist.angular.z = 0.0
        self.pub.publish(twist)

    # ── Watchdog: stop if terrain data goes stale ──────────────────────

    def _watchdog_callback(self) -> None:
        elapsed = (self.get_clock().now() - self.last_msg_time).nanoseconds / 1e9
        if elapsed > self.stale_s:
            if self.current_label != "unknown":
                self.get_logger().warn(
                    f"No terrain classification for {elapsed:.1f}s — "
                    "forcing STOP (unknown terrain)"
                )
            self.current_label      = "unknown"
            self.current_confidence = 0.0
            self.current_speed      = 0.0
            self.prev_label         = "unknown"
            self._resume_candidate_count = 0
            self._stop_latch_until_ns = self.get_clock().now().nanoseconds + int(self.stop_latch_s * 1e9)


def main(args=None):
    rclpy.init(args=args)
    node = TerrainControllerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        # Publish one final stop command on shutdown
        stop = Twist()
        node.pub.publish(stop)
        node.get_logger().info("Shutdown — final STOP command sent")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
