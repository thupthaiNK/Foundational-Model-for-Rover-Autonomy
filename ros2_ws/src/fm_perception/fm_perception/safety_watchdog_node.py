#!/usr/bin/env python3
"""
Purpose: Safety watchdog node that monitors terrain classification quality and
         triggers a latched E-stop when the fail rate exceeds a threshold or
         when no classification messages arrive for too long.
         Complements terrain_controller_node.py (Layer 3 traversability):
           - terrain_controller_node: per-message stop/slow/go policy
           - safety_watchdog_node:    aggregate fail-rate monitoring + latched E-stop
         E-stop state is latched — once triggered it persists until explicitly
         reset via the /e_stop_reset service (std_srvs/Trigger).
Inputs:  /terrain_classification (std_msgs/String)  "label:confidence"
Outputs: /e_stop       (std_msgs/Bool)      True = E-stop active
         /e_stop_reason (std_msgs/String)   human-readable reason
         /exomy/cmd_vel (geometry_msgs/Twist) zero velocity when E-stop active
Services provided:
         /e_stop_reset  (std_srvs/Trigger)  clears latched E-stop
How to run:
    source /opt/ros/humble/setup.bash
    source ros2_ws/install/setup.bash
    ros2 run fm_perception safety_watchdog_node.py
    # Reset E-stop after manual inspection:
    ros2 service call /e_stop_reset std_srvs/srv/Trigger
Parameters:
    confidence_threshold (float, default 0.40): below this the message is
        treated as uncertain for fail-rate accounting. Keep aligned with the
        terrain controller threshold to avoid false-positive E-stops.
    fail_rate_threshold (float, default 0.5):  fraction of msgs that must be
        uncertain/stop in the sliding window to trigger E-stop (0.5 = 50%)
    window_s       (float, default 10.0):  sliding window duration in seconds
    stale_timeout_s (float, default 3.0):  seconds without any msg before E-stop
    startup_grace_s (float, default 15.0): ignore fail/stale checks during
        initial classifier warm-up and topic stabilisation
    disable_during_measurement (bool, default True): pause fail-rate/stale
        checks while the experiment freezes the controller with /measurement_mode
    cmd_vel_topic  (string, default /exomy/cmd_vel): topic for stop commands
    terrain_topic  (string, default /terrain_classification): input topic
    publish_rate_hz (float, default 5.0): /e_stop publish rate
Project: Onboard Visual Foundation Models for Mars Terrain Perception and Rover Navigation
"""

import time
from collections import deque

from geometry_msgs.msg import Twist
from std_msgs.msg import Bool, String
from std_srvs.srv import Trigger
import rclpy
from rclpy.node import Node

STOP_LABELS = {"uncertain", "unknown", "big_rock"}


class SafetyWatchdogNode(Node):

    def __init__(self):
        super().__init__("safety_watchdog_node")

        # ── Parameters ──────────────────────────────────────────────────
        self.declare_parameter("confidence_threshold", 0.40)
        self.declare_parameter("fail_rate_threshold", 0.5)
        self.declare_parameter("window_s",            10.0)
        self.declare_parameter("stale_timeout_s",     3.0)
        self.declare_parameter("startup_grace_s",     15.0)
        self.declare_parameter("disable_during_measurement", True)
        self.declare_parameter("cmd_vel_topic",       "/exomy/cmd_vel")
        self.declare_parameter("terrain_topic",       "/terrain_classification")
        self.declare_parameter("publish_rate_hz",     5.0)

        self.conf_threshold = (
            self.get_parameter("confidence_threshold").get_parameter_value().double_value
        )
        self.threshold  = self.get_parameter("fail_rate_threshold").get_parameter_value().double_value
        self.window_s   = self.get_parameter("window_s").get_parameter_value().double_value
        self.stale_s    = self.get_parameter("stale_timeout_s").get_parameter_value().double_value
        self.startup_grace_s = (
            self.get_parameter("startup_grace_s").get_parameter_value().double_value
        )
        self.disable_during_measurement = (
            self.get_parameter("disable_during_measurement").get_parameter_value().bool_value
        )
        cmd_vel_topic   = self.get_parameter("cmd_vel_topic").get_parameter_value().string_value
        terrain_topic   = self.get_parameter("terrain_topic").get_parameter_value().string_value
        publish_hz      = self.get_parameter("publish_rate_hz").get_parameter_value().double_value

        # ── State ────────────────────────────────────────────────────────
        # Sliding window: deque of (timestamp_ns, is_fail: bool)
        self._window: deque[tuple[int, bool]] = deque()
        self._last_msg_ns: int | None = None
        self._startup_ns: int | None = None
        self._e_stop_active: bool = False
        self._e_stop_reason: str  = ""
        self._measurement_frozen: bool = False

        # ── Publishers ───────────────────────────────────────────────────
        self._pub_estop  = self.create_publisher(Bool,   "/e_stop",        10)
        self._pub_reason = self.create_publisher(String, "/e_stop_reason", 10)
        self._pub_cmd    = self.create_publisher(Twist,  cmd_vel_topic,    10)

        # ── Subscribers ──────────────────────────────────────────────────
        self.create_subscription(
            String, terrain_topic, self._terrain_callback, 10
        )
        self.create_subscription(
            Bool, "/measurement_mode", self._measurement_callback, 10
        )

        # ── Timers ───────────────────────────────────────────────────────
        self.create_timer(1.0 / publish_hz, self._publish_status)
        self.create_timer(1.0,              self._stale_watchdog)

        # ── Reset service ────────────────────────────────────────────────
        self.create_service(Trigger, "/e_stop_reset", self._reset_callback)

        self.get_logger().info(
            f"safety_watchdog_node ready\n"
            f"  Listening:  {terrain_topic}\n"
            f"  E-stop out: /e_stop, /e_stop_reason, {cmd_vel_topic}\n"
            f"  Confidence threshold: {self.conf_threshold:.2f}\n"
            f"  Fail-rate threshold: {self.threshold:.0%} over {self.window_s:.0f}s window\n"
            f"  Stale timeout: {self.stale_s:.1f}s\n"
            f"  Startup grace: {self.startup_grace_s:.1f}s\n"
            f"  Pause during measurement: {self.disable_during_measurement}\n"
            f"  Reset via: ros2 service call /e_stop_reset std_srvs/srv/Trigger"
        )

    def _measurement_callback(self, msg: Bool) -> None:
        previously_frozen = self._measurement_frozen
        self._measurement_frozen = msg.data

        if self.disable_during_measurement and msg.data and not previously_frozen:
            # Measurement mode intentionally freezes the controller and may hold
            # hazardous zones in view; do not let that latch the watchdog.
            self._window.clear()
            self._last_msg_ns = self.get_clock().now().nanoseconds

        if self.disable_during_measurement and previously_frozen and not msg.data:
            # Resume with a fresh window after the experiment unfreezes.
            self._window.clear()
            self._last_msg_ns = self.get_clock().now().nanoseconds

    def _guard_checks_suspended(self, now_ns: int) -> bool:
        if self._startup_ns is None:
            return True
        if (now_ns - self._startup_ns) / 1e9 < self.startup_grace_s:
            return True
        if self.disable_during_measurement and self._measurement_frozen:
            return True
        return False

    # ── Terrain classification callback ─────────────────────────────────

    def _terrain_callback(self, msg: String) -> None:
        now_ns = self.get_clock().now().nanoseconds
        parts = msg.data.split(":")
        if len(parts) != 2:
            return
        label = parts[0].strip()
        try:
            confidence = float(parts[1])
        except ValueError:
            return

        self._last_msg_ns = now_ns
        if self._startup_ns is None:
            # Start grace timing from the first valid terrain message, not from
            # node construction. With use_sim_time the node can start while the
            # simulated clock is still at 0, which otherwise makes stale checks
            # fire immediately once the clock jumps forward.
            self._startup_ns = now_ns

        # Low confidence counts as uncertain. Keep this aligned with the
        # controller threshold; otherwise watchdog can E-stop valid controller
        # states (e.g. soil 0.59 with controller threshold 0.40).
        if confidence < self.conf_threshold:
            label = "uncertain"

        if self._guard_checks_suspended(now_ns):
            return

        is_fail = label in STOP_LABELS
        self._window.append((now_ns, is_fail))
        self._prune_window(now_ns)
        self._check_fail_rate()

    # ── Sliding window helpers ───────────────────────────────────────────

    def _prune_window(self, now_ns: int) -> None:
        cutoff_ns = now_ns - int(self.window_s * 1e9)
        while self._window and self._window[0][0] < cutoff_ns:
            self._window.popleft()

    def _fail_rate(self) -> float:
        if not self._window:
            return 0.0
        return sum(1 for _, f in self._window if f) / len(self._window)

    def _check_fail_rate(self) -> None:
        if self._e_stop_active:
            return  # already latched — nothing to do
        rate = self._fail_rate()
        if rate >= self.threshold and len(self._window) >= 3:
            reason = (
                f"Fail rate {rate:.0%} >= threshold {self.threshold:.0%} "
                f"over last {self.window_s:.0f}s "
                f"({sum(1 for _,f in self._window if f)}/{len(self._window)} msgs)"
            )
            self._trigger_estop(reason)

    # ── Stale watchdog (no msgs for stale_timeout_s) ─────────────────────

    def _stale_watchdog(self) -> None:
        now_ns  = self.get_clock().now().nanoseconds
        if self._guard_checks_suspended(now_ns):
            return
        if self._last_msg_ns is None:
            return
        if not self._window:
            return
        elapsed = (now_ns - self._last_msg_ns) / 1e9
        if elapsed > self.stale_s and not self._e_stop_active:
            self._trigger_estop(
                f"No terrain classification for {elapsed:.1f}s "
                f"(timeout={self.stale_s:.1f}s)"
            )

    # ── E-stop trigger / reset ───────────────────────────────────────────

    def _trigger_estop(self, reason: str) -> None:
        self._e_stop_active = True
        self._e_stop_reason = reason
        self.get_logger().error(f"E-STOP TRIGGERED: {reason}")

    def _reset_callback(self, request, response) -> Trigger.Response:
        if self._e_stop_active:
            old_reason = self._e_stop_reason
            self._e_stop_active = False
            self._e_stop_reason = ""
            self._window.clear()
            self._last_msg_ns = self.get_clock().now().nanoseconds
            self._startup_ns = self._last_msg_ns
            response.success = True
            response.message = f"E-stop cleared (was: {old_reason})"
            self.get_logger().warn(f"E-stop RESET by operator. Was: {old_reason}")
        else:
            response.success = True
            response.message = "E-stop was not active — nothing to reset"
        return response

    # ── Periodic publish ─────────────────────────────────────────────────

    def _publish_status(self) -> None:
        estop_msg = Bool()
        estop_msg.data = self._e_stop_active
        self._pub_estop.publish(estop_msg)

        reason_msg = String()
        reason_msg.data = self._e_stop_reason if self._e_stop_active else ""
        self._pub_reason.publish(reason_msg)

        if self._e_stop_active:
            self._pub_cmd.publish(Twist())  # zero velocity


def main(args=None):
    rclpy.init(args=args)
    node = SafetyWatchdogNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutdown — final STOP command sent")
        node._pub_cmd.publish(Twist())
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
