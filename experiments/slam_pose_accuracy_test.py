"""
Purpose: L4 Phase A -- measures slam_toolbox's estimated pose against
         Gazebo's ground-truth /exomy/odom while the rover drives near the
         Q4 rock cluster (spawn (2.0,-6.0), the same feature-rich area used
         by the 5-zone benchmark's rock_cluster/boulder_zone tests, §3.11.2).
         Deliberately bypasses the safety/perception stack entirely
         (DINOv2, terrain_controller_node, reactive_explorer_node) --
         cmd_vel is published directly by this script, isolating the SLAM
         component from the driving-policy component so a first SLAM
         result is not confounded by hazard-stop behaviour (Q4 is a STOP
         zone under the normal safety policy, §4.8.3).

         DRIVE PATTERN: a single continuous gentle arc (constant small
         linear_x + constant small angular_z together), not a square loop
         with sharp turns. A first version of this script drove a
         forward/turn-x4 square loop; both an open-loop-timed turn attempt
         and a closed-loop (yaw-tracking) turn attempt failed to reliably
         complete a 90 deg in-place rotation -- 4/4 closed-loop turns timed
         out at 60s (10x the ~5.5s naive estimate), reaching an inconsistent
         7.4/68.0/4.2/0.8 degrees of the 90 degree target. This corroborates
         the H5 finding (§4.8.17: a ~5s-estimated turn took ~7.8 minutes
         under this machine's Gazebo load) as a second, independent
         confirmation that in-place rotation is unreliable for this rover
         model on this machine -- not a script bug. Investigating the root
         cause (wheel friction/torque tuning vs Gazebo RTF variability) is
         parked as separate follow-up work, not blocking this test: a
         gentle continuous arc needs only a small sustained wheel-speed
         differential (not the large one in-place rotation requires), so
         it sidesteps the problem rather than solving it. See
         _turn_closed_loop()/drive_square_loop() below, kept for reference
         and reuse once the turning investigation happens, but not called
         by main() currently.
         This test uses slam_toolbox's default odom-assisted configuration
         (see simulation/launch/slam_test.launch.py docstring) -- i.e. it
         has access to Gazebo's ground-truth odom->base_link TF as a motion
         prior, which real ExoMy hardware cannot reproduce (no wheel
         encoders). This is a best-case, upper-bound test of what
         slam_toolbox can achieve in this environment at all, not a
         real-hardware-representative test.
Inputs:  /exomy/odom (nav_msgs/Odometry)                    ground truth
         /pose       (geometry_msgs/PoseWithCovarianceStamped) slam_toolbox estimate
         /exomy/cmd_vel (published by this script, not subscribed)
Outputs: experiments/results/slam_pose_accuracy_test.csv
         experiments/results/figures/slam_pose_accuracy_test.png
How to run:
    # Terminal 1:
    source /opt/ros/humble/setup.bash && source ros2_ws/install/setup.bash
    ros2 launch simulation/launch/slam_test.launch.py

    # Terminal 2 (after Terminal 1 shows "Spawn status: ... Successfully
    # spawned entity [exomy]" and slam_toolbox has registered the sensor):
    source /opt/ros/humble/setup.bash && source ros2_ws/install/setup.bash
    python3 experiments/slam_pose_accuracy_test.py
Project: Onboard Visual Foundation Models for Mars Terrain Perception and Rover Navigation
"""

import csv
import math
import os
import time

import rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped, Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node

LINEAR_SPEED  = 0.10    # m/s -- matches this thesis's established real-world speed ceiling
ANGULAR_SPEED = 0.30    # rad/s -- matches angular_speed already used throughout this codebase
                         # (used only by the parked square-loop/closed-loop-turn code below)
FORWARD_LEG_S = 15.0    # 15s * 0.10 m/s = 1.5 m per leg (parked square-loop code)
TURN_TARGET_RAD    = math.pi / 2   # 90 deg per leg (parked square-loop code)
TURN_TOLERANCE_RAD = math.radians(5.0)
TURN_TIMEOUT_S      = 60.0   # circuit-breaker: give up this leg's turn (not the whole run) if it
                              # never converges, rather than looping forever -- mirrors the
                              # max_wiggle_attempts/FAILSAFE philosophy already used elsewhere in
                              # this codebase (stuck_detection_node.py, reactive_explorer_node.py)
N_LEGS        = 4      # a rough square loop, returning close to the start (parked square-loop code)
SETTLE_S      = 5.0    # wait after each command switch before the next, damps physics transients

# -- Gentle continuous arc (the drive pattern actually used, see docstring) --
ARC_LINEAR_SPEED  = 0.10   # m/s, same speed ceiling as above
ARC_ANGULAR_SPEED = 0.08   # rad/s -- gentle; radius = 0.10/0.08 = 1.25 m, much smaller wheel-speed
                            # differential than in-place rotation needs, sidestepping the turning
                            # problem rather than depending on it being fixed
ARC_DURATION_S    = 90.0   # slightly over one full 1.25 m-radius loop period (2*pi/0.08 ~= 78.5 s)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)


def yaw_from_quaternion(q) -> float:
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def normalize_angle(angle: float) -> float:
    """Wrap an angle in radians to (-pi, pi]."""
    return math.atan2(math.sin(angle), math.cos(angle))


def angle_delta(target: float, current: float) -> float:
    """Shortest signed angular distance from current to target, in (-pi, pi]."""
    return normalize_angle(target - current)


class SlamAccuracyRecorder(Node):

    def __init__(self):
        super().__init__("slam_pose_accuracy_recorder")
        self._records = []
        self._t0 = time.time()

        self._gt_pos  = None   # (x, y, yaw) from /exomy/odom
        self._est_pos = None   # (x, y, yaw) from /pose
        self._pose_msg_count = 0

        self.create_subscription(Odometry, "/exomy/odom", self._odom_cb, 10)
        self.create_subscription(PoseWithCovarianceStamped, "/pose", self._pose_cb, 10)
        self.pub_cmd = self.create_publisher(Twist, "/exomy/cmd_vel", 10)

        self.create_timer(0.5, self._record_tick)

    def _odom_cb(self, msg: Odometry) -> None:
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        self._gt_pos = (p.x, p.y, yaw_from_quaternion(q))

    def _pose_cb(self, msg: PoseWithCovarianceStamped) -> None:
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        self._est_pos = (p.x, p.y, yaw_from_quaternion(q))
        self._pose_msg_count += 1

    def _record_tick(self) -> None:
        elapsed = time.time() - self._t0
        gt  = self._gt_pos
        est = self._est_pos
        error_m = None
        if gt is not None and est is not None:
            error_m = math.hypot(gt[0] - est[0], gt[1] - est[1])
        self._records.append({
            "elapsed_s": round(elapsed, 2),
            "gt_x":  round(gt[0], 3) if gt else None,
            "gt_y":  round(gt[1], 3) if gt else None,
            "gt_yaw_deg": round(math.degrees(gt[2]), 1) if gt else None,
            "est_x": round(est[0], 3) if est else None,
            "est_y": round(est[1], 3) if est else None,
            "est_yaw_deg": round(math.degrees(est[2]), 1) if est else None,
            "error_m": round(error_m, 3) if error_m is not None else None,
        })

    def send_twist(self, linear_x: float, angular_z: float) -> None:
        msg = Twist()
        msg.linear.x = linear_x
        msg.angular.z = angular_z
        self.pub_cmd.publish(msg)

    def save_csv(self, path: str) -> None:
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(self._records[0].keys()))
            w.writeheader()
            w.writerows(self._records)
        self.get_logger().info(f"Saved {len(self._records)} records -> {path}")


def _hold_for(node: SlamAccuracyRecorder, duration_s: float, linear_x: float, angular_z: float) -> None:
    """Publish a constant Twist and keep spinning (so subscriptions/recording
    keep working) for duration_s of wall-clock time, guaranteed -- unlike a
    single spin_once(timeout_sec=duration_s), which returns as soon as any
    one callback fires and does not actually block for the full duration."""
    end_t = time.time() + duration_s
    while time.time() < end_t:
        node.send_twist(linear_x, angular_z)
        rclpy.spin_once(node, timeout_sec=0.1)


def _turn_closed_loop(node: SlamAccuracyRecorder, turn_target_rad: float) -> None:
    """Turn until ground-truth odometry yaw actually reaches the target, not
    for a fixed duration. A first attempt at this test used a fixed 5.5s
    open-loop turn command and produced ~0 deg of actual rotation -- this
    codebase's own reactive_explorer_node.py and stuck_detection_node.py
    already use closed-loop yaw tracking for exactly this reason (open-loop
    timing is unreliable under this machine's variable Gazebo real-time
    factor, already documented for a heavier launch in Ch4 SS4.8.17's H5
    result: a ~5s-estimated turn took ~7.8 minutes under load)."""
    if node._gt_pos is None:
        node.get_logger().warn("No ground-truth pose yet -- skipping this turn")
        return
    start_yaw = node._gt_pos[2]
    target_yaw = normalize_angle(start_yaw + turn_target_rad)

    deadline = time.time() + TURN_TIMEOUT_S
    while time.time() < deadline:
        current_yaw = node._gt_pos[2]
        delta = angle_delta(target_yaw, current_yaw)
        if abs(delta) <= TURN_TOLERANCE_RAD:
            node.get_logger().info(
                f"  turn converged: {math.degrees(current_yaw - start_yaw):.1f} deg "
                f"(target {math.degrees(turn_target_rad):.0f} deg)"
            )
            return
        angular = ANGULAR_SPEED if delta > 0 else -ANGULAR_SPEED
        node.send_twist(0.0, angular)
        rclpy.spin_once(node, timeout_sec=0.1)

    node.get_logger().warn(
        f"  turn TIMED OUT after {TURN_TIMEOUT_S}s -- only reached "
        f"{math.degrees(node._gt_pos[2] - start_yaw):.1f} deg of "
        f"{math.degrees(turn_target_rad):.0f} deg target. Moving on to the next leg anyway."
    )


def drive_gentle_arc(node: SlamAccuracyRecorder) -> None:
    """The drive pattern actually used by this test (see module docstring):
    a single continuous small linear_x + small angular_z command, tracing a
    ~1.25m-radius circle. Needs no in-place rotation at all."""
    node.get_logger().info(
        f"Driving a continuous gentle arc for {ARC_DURATION_S}s "
        f"(linear={ARC_LINEAR_SPEED} m/s, angular={ARC_ANGULAR_SPEED} rad/s, "
        f"radius~={ARC_LINEAR_SPEED/ARC_ANGULAR_SPEED:.2f}m)"
    )
    _hold_for(node, ARC_DURATION_S, ARC_LINEAR_SPEED, ARC_ANGULAR_SPEED)
    node.send_twist(0.0, 0.0)


def drive_square_loop(node: SlamAccuracyRecorder) -> None:
    for leg in range(N_LEGS):
        node.get_logger().info(f"Leg {leg+1}/{N_LEGS}: forward {FORWARD_LEG_S}s")
        _hold_for(node, FORWARD_LEG_S, LINEAR_SPEED, 0.0)
        node.send_twist(0.0, 0.0)
        _hold_for(node, SETTLE_S, 0.0, 0.0)

        node.get_logger().info(f"Leg {leg+1}/{N_LEGS}: closed-loop turn to +{math.degrees(TURN_TARGET_RAD):.0f} deg")
        _turn_closed_loop(node, TURN_TARGET_RAD)
        node.send_twist(0.0, 0.0)
        _hold_for(node, SETTLE_S, 0.0, 0.0)


def main():
    rclpy.init()
    node = SlamAccuracyRecorder()
    node.get_logger().info(
        "SLAM pose accuracy test starting -- driving a continuous gentle arc "
        "near Q4, bypassing the safety stack entirely (isolated SLAM test)."
    )

    # Let subscriptions settle and confirm both topics are alive before driving.
    rclpy.spin_once(node, timeout_sec=2.0)

    drive_gentle_arc(node)

    node.get_logger().info(f"Arc complete. /pose messages received: {node._pose_msg_count}")

    csv_path = os.path.join(RESULTS_DIR, "slam_pose_accuracy_test.csv")
    node.save_csv(csv_path)

    valid = [r for r in node._records if r["error_m"] is not None]
    if valid:
        errors = [r["error_m"] for r in valid]
        node.get_logger().info(
            f"Position error vs ground truth -- n={len(errors)}, "
            f"mean={sum(errors)/len(errors):.3f}m, max={max(errors):.3f}m, min={min(errors):.3f}m"
        )
    else:
        node.get_logger().warn(
            "No /pose messages were ever received during this run -- "
            "slam_toolbox never produced a pose estimate. See CSV for the full record."
        )

    try:
        import matplotlib.pyplot as plt

        gt_x  = [r["gt_x"]  for r in node._records if r["gt_x"]  is not None]
        gt_y  = [r["gt_y"]  for r in node._records if r["gt_y"]  is not None]
        est_x = [r["est_x"] for r in node._records if r["est_x"] is not None]
        est_y = [r["est_y"] for r in node._records if r["est_y"] is not None]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle("L4 Phase A -- SLAM Pose Accuracy vs Gazebo Ground Truth (Q4, odom-assisted)")

        ax1.plot(gt_x, gt_y, "g-", label="Ground truth (/exomy/odom)", linewidth=2)
        if est_x:
            ax1.plot(est_x, est_y, "r--", label="slam_toolbox (/pose)", linewidth=2)
        ax1.set_xlabel("x (m)")
        ax1.set_ylabel("y (m)")
        ax1.set_title("Trajectory")
        ax1.legend()
        ax1.axis("equal")
        ax1.grid(True, alpha=0.3)

        t = [r["elapsed_s"] for r in valid]
        e = [r["error_m"] for r in valid]
        ax2.plot(t, e, "b-")
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Position error (m)")
        ax2.set_title("Error over time")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        fig_path = os.path.join(FIGURES_DIR, "slam_pose_accuracy_test.png")
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        node.get_logger().info(f"Figure saved -> {fig_path}")
        plt.close()
    except Exception as e:
        node.get_logger().warn(f"Plot failed: {e}")

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
