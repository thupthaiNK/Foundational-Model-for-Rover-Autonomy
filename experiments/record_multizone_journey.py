"""
Purpose: Record a continuous rock-field traversal in Gazebo as two clips:
           1. Onboard rover camera (/exomy/camera/image_raw)
              -> journey_onboard.mp4
           2. Chase cam rigidly mounted on the rover, 1.2m behind + 1.0m
              above base_link (added to exomy.urdf.xacro 2026-08-17
              specifically for this clip -- a per-zone fixed camera like
              record_zone_videos.py uses can't follow the rover across the
              whole map) -> journey_thirdperson.mp4
         Route (v4, author spec 2026-08-17): spawn directly at
         rock_cluster's spawn point (2.0,-4.0) facing east, straight into
         the rock field, with REACTIVE obstacle avoidance running the whole
         clip until it stops (boxed in) or reaches near the east map edge
         (x=14.5). Four-round history on this avoidance:
           v1: known rock positions/radii from mars_terrain.world as a
               "virtual sensor", single fixed avoid angle relative to
               CURRENT heading. Bug: once deflected, the heading never
               recovered, so the rover cleared the whole rock field after
               just one avoid.
           v2: tried switching to the REAL fm_perception.avoidance_planner
               geometry against the simulated /scan (RPLIDAR-proxy) topic
               (author asked for real LiDAR-based detection). Measured
               directly (spawned the rover 0.66m from rock_1's centre,
               dumped raw LaserScan ranges): zero returns closer than 2m
               anywhere, in any direction. Root cause -- physical, not a
               code or mounting-position bug (also checked: not the front
               camera mast occluding it, since the null result holds in
               every direction, not just forward): the LiDAR is mounted at
               0.215m (ground_clearance + body_height + 0.030, matching real
               hardware) but every rock's collision sphere is buried/
               undersized enough that even the tallest (boulder_2) only
               reaches 0.187m -- the horizontal beam passes over every rock
               in this world. Confirmed with the author rather than
               silently faking sensor data or changing the shared LiDAR
               mount height (used by every other experiment in this repo).
               Author chose to go back to v1's approach with the bug fixed.
           v3: kept v1's hardcoded rock-position "virtual sensor", but the
               avoid heading is picked relative to WORLD EAST (the field's
               true progress direction), not the rover's current heading --
               fixes v1's "never recovers" bug.
           v4 (this version, author review of v3): the bedrock->soil->sand
               preamble v3 still had (27m of driving before nearing a rock)
               buried the avoidance behaviour and made each test iteration
               slow, and the 1.2m avoidance lookahead triggered before a
               rock was ever close enough to be visible on screen. Now:
               spawns directly in the rock field (no preamble), and the
               lookahead is shortened to 0.8m so the rock is clearly in
               frame when each avoid turn happens.
         The avoidance itself (all versions since v3): a small forward
         "corridor" (rectangle, not the real corridor-box geometry --
         half_width 0.35m, lookahead 0.8m) is checked at east + each
         candidate offset in increasing |offset| (10 deg steps, full
         circle), and the smallest-deviation-from-east clear heading is
         taken -- so the rover turns back toward progress as soon as it
         safely can. If NO offset anywhere in the full circle clears, it
         stops ("boxed in") -- an accepted ending per the author, not a
         failure.
         Uses odometry feedback throughout (not wall-clock timers) --
         record_zone_videos.py's v1-v5 iterations found Gazebo runs faster
         than real time in this headless setup, which made timer-based
         driving overshoot into collisions; distance/yaw-from-odometry
         segment completion is correct regardless of that speed ratio.
         This is video-demo motion only, NOT a representation of the real
         FSM's behavior -- do not use as evidence for report claims.
Inputs:  Running ROS2 stack:
           ros2 launch simulation/launch/dinov2_controller_test.launch.py
         with terrain_controller_node and safety_watchdog_node killed
         (this script drives directly via /exomy/cmd_vel).
Outputs: docs/figures/gazebo_zone_videos/journey_onboard.mp4
         docs/figures/gazebo_zone_videos/journey_thirdperson.mp4
How to run:
    # Terminal 1
    source /opt/ros/humble/setup.bash && source ros2_ws/install/setup.bash
    ros2 launch simulation/launch/dinov2_controller_test.launch.py
    # once it's up:
    #   pkill -f terrain_controller_node.py; pkill -f safety_watchdog_node.py

    # Terminal 2
    source /opt/ros/humble/setup.bash && source ros2_ws/install/setup.bash
    python3 experiments/record_multizone_journey.py
Project: Onboard Visual Foundation Models for Mars Terrain Perception and Rover Navigation
"""

import math
import os
import time

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from gazebo_msgs.srv import SpawnEntity, DeleteEntity

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
OUT_DIR = os.path.join(_REPO_ROOT, "docs", "figures", "gazebo_zone_videos")
os.makedirs(OUT_DIR, exist_ok=True)

WAIT_STABLE_S = 3.0
FPS = 15
CMD_RATE_HZ = 30.0
MAX_WALL_S = 1500.0  # generous safety cap (author: clip length doesn't matter,
                      # just needs to finish); v1 needed ~470s to cover just
                      # bedrock->soil->sand->first rock, so budget generously

DRIVE_MS = 0.15
PIVOT_RATE_RS = 0.785398  # rad/s

# v4 (author review of v3): the bedrock->soil->sand preamble (27m of empty
# driving before ever nearing a rock) made the eventual rock encounters easy
# to miss on a casual watch, and each test iteration cost ~10min of sim time
# just to reach the interesting part. Per author instruction, this version
# spawns directly in the rock field (rock_cluster's spawn point) facing east,
# so the whole clip is the avoidance behaviour.
SPAWN_X, SPAWN_Y, SPAWN_Z = 2.0, -4.0, 0.15  # rock_cluster spawn
SPAWN_YAW_DEG = 0.0  # facing +x (east), straight into the rock field

ROCK_FIELD_STOP_X = 14.5  # near the east map edge (~x=15) -- "drive until
                          # stopped or off the map", not a fixed distance

# Hardcoded-rock-position "virtual sensor" avoidance (v3 -- see module
# docstring for why real LiDAR isn't physically possible in this world).
# Ground truth from mars_terrain.world's <collision><sphere> entries.
ROCKS = [
    (4.5, -3.5, 0.273),   # boulder_1
    (4.5, -8.0, 0.293),   # boulder_2
    (9.0, -5.5, 0.247),   # boulder_3
    (3.5, -4.0, 0.143),   # rock_1
    (6.0, -3.0, 0.117),   # rock_2
    (5.0, -5.0, 0.163),   # rock_3
    (7.5, -4.0, 0.098),   # rock_4
    (8.5, -3.5, 0.130),   # rock_5
    (3.5, -7.5, 0.130),   # rock_6
    (6.0, -8.5, 0.143),   # rock_7
    (5.5, -9.5, 0.117),   # rock_8
    (7.5, -6.5, 0.111),   # rock_9
    (8.5, -8.5, 0.163),   # rock_10
    (10.0, -6.0, 0.104),  # rock_11
    (11.0, -4.5, 0.091),  # rock_12
    (12.0, -3.0, 0.078),  # rock_13
    (13.0, -7.0, 0.085),  # rock_14
    (12.5, -10.5, 0.072), # rock_15
    (10.0, -11.0, 0.065), # rock_16
    (3.0, -11.0, 0.059),  # rock_17
    (7.0, -11.5, 0.052),  # rock_18
]

CORRIDOR_HALF_WIDTH_M = 0.35   # rover half-width + margin
# v4: shortened from 1.2m -- with the longer lookahead the rover turned away
# before a rock was ever close enough to be visible on screen, so the avoids
# were invisible to a casual watch even though they were happening. 0.8m
# matches the closer stand-off record_zone_videos.py's per-zone clips used,
# where the rock is clearly in frame when the turn starts.
CORRIDOR_LOOKAHEAD_M = 0.8      # how far ahead the corridor must be clear
EAST_HEADING_RAD = 0.0         # the field's true progress direction
AVOID_SEARCH_STEP_DEG = 10.0   # candidate offsets from east, in this step


def _yaw_from_quat(z: float, w: float) -> float:
    return 2.0 * math.atan2(z, w)


def _angle_wrap(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


class DeadReckoner:
    """Tracks believed world pose (x, y, heading) purely from RELATIVE local
    odometry deltas (distance travelled / yaw turned since the current
    segment started) -- never from an absolute odom-to-world transform.

    Earlier version of this script tried to convert /exomy/odom directly to
    world coordinates with a fixed rotation formula; that assumption was
    wrong (verified: the rover ran off the map, over 30m past soil_zone,
    because the waypoint-arrival check against believed world coordinates
    never fired). Local-frame relative distance/yaw deltas, by contrast,
    are frame-invariant (Euclidean distance and a relative angle don't care
    what the reference frame's absolute orientation is) and were already
    validated repeatedly in record_zone_videos.py. This class rebuilds world
    pose by hand from those same reliable relative measurements: each
    straight segment advances (x, y) along the CURRENT believed heading by
    the local distance travelled; each pivot segment advances heading by
    the local yaw turned (signed by the commanded direction). Since the
    commanded direction is what determines the turn (not odom's own yaw
    sign convention), heading updates use the segment's own direction/sign,
    not the raw local yaw value at completion.
    """

    def __init__(self, x0: float, y0: float, heading0: float):
        self.pos = (x0, y0)
        self.heading = heading0
        self._local0 = None  # (lx, ly, lyaw) at current segment start
        self._kind = None    # "straight" or "pivot"

    def begin_straight(self, local_now):
        self._local0 = local_now
        self._kind = "straight"

    def begin_pivot(self, local_now):
        self._local0 = local_now
        self._kind = "pivot"

    def current_pose(self, local_now):
        """Believed (x, y, heading) at this instant, mid-segment."""
        if self._local0 is None or self._kind == "pivot":
            return self.pos[0], self.pos[1], self.heading
        lx0, ly0, _ = self._local0
        lx, ly, _ = local_now
        dist = math.hypot(lx - lx0, ly - ly0)
        cx = self.pos[0] + dist * math.cos(self.heading)
        cy = self.pos[1] + dist * math.sin(self.heading)
        return cx, cy, self.heading

    def straight_progress_m(self, local_now) -> float:
        lx0, ly0, _ = self._local0
        lx, ly, _ = local_now
        return math.hypot(lx - lx0, ly - ly0)

    def pivot_progress_rad(self, local_now) -> float:
        _, _, lyaw0 = self._local0
        _, _, lyaw = local_now
        return abs(_angle_wrap(lyaw - lyaw0))

    def complete_straight(self, distance_m: float):
        self.pos = (self.pos[0] + distance_m * math.cos(self.heading),
                    self.pos[1] + distance_m * math.sin(self.heading))

    def complete_pivot(self, signed_angle_rad: float):
        self.heading = _angle_wrap(self.heading + signed_angle_rad)


class JourneyRecorder(Node):
    def __init__(self):
        super().__init__("journey_recorder")
        self._urdf_xml = None
        latched_qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(
            __import__("std_msgs.msg", fromlist=["String"]).String,
            "/robot_description", self._urdf_cb, latched_qos,
        )
        self._delete_client = self.create_client(DeleteEntity, "/delete_entity")
        self._spawn_client = self.create_client(SpawnEntity, "/spawn_entity")
        self._cmd_vel_pub = self.create_publisher(Twist, "/exomy/cmd_vel", 10)

        self._onboard_frame = None
        self._chase_frame = None
        self.create_subscription(Image, "/exomy/camera/image_raw", self._onboard_cb, 1)
        self.create_subscription(Image, "/exomy/chase_cam/image_raw", self._chase_cb, 1)

        self._odom_local = None  # (x, y, yaw) in the odom-plugin's own zeroed frame
        self.create_subscription(Odometry, "/exomy/odom", self._odom_cb, 1)

    def _odom_cb(self, msg):
        p = msg.pose.pose.position
        o = msg.pose.pose.orientation
        self._odom_local = (p.x, p.y, _yaw_from_quat(o.z, o.w))

    @staticmethod
    def corridor_clear(pos, heading: float) -> bool:
        """Rectangle corridor check (half_width x lookahead ahead of pos, at
        heading) against the known rock list -- see module docstring for why
        this replaced real LiDAR sensing here."""
        hx, hy = math.cos(heading), math.sin(heading)
        px, py = -hy, hx  # perpendicular (left) unit vector
        for rx, ry, rad in ROCKS:
            dx, dy = rx - pos[0], ry - pos[1]
            fwd = dx * hx + dy * hy
            lat = dx * px + dy * py
            if 0.0 <= fwd <= CORRIDOR_LOOKAHEAD_M and abs(lat) <= CORRIDOR_HALF_WIDTH_M + rad:
                return False
        return True

    def pick_avoid_heading(self, pos):
        """Smallest-|deviation|-from-EAST absolute heading (radians) whose
        corridor is clear, searched over the full circle in
        AVOID_SEARCH_STEP_DEG steps. None if nothing clears anywhere
        ("boxed in")."""
        step = math.radians(AVOID_SEARCH_STEP_DEG)
        offsets = [0.0]
        k = 1
        while k * step <= math.pi + 1e-9:
            offsets.append(k * step)
            offsets.append(-k * step)
            k += 1
        for offset in offsets:
            heading = EAST_HEADING_RAD + offset
            if self.corridor_clear(pos, heading):
                return heading
        return None

    def _urdf_cb(self, msg):
        self._urdf_xml = msg.data

    def _onboard_cb(self, msg):
        self._onboard_frame = self._to_bgr(msg)

    def _chase_cb(self, msg):
        self._chase_frame = self._to_bgr(msg)

    @staticmethod
    def _to_bgr(msg: Image) -> np.ndarray:
        arr = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width, -1)[:, :, :3]
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


    def teleport(self, x: float, y: float, z: float, yaw_deg: float = 0.0) -> bool:
        if self._urdf_xml is None:
            deadline = time.time() + 8
            while self._urdf_xml is None and time.time() < deadline:
                rclpy.spin_once(self, timeout_sec=0.2)
        if self._urdf_xml is None:
            self.get_logger().warn("No URDF -- cannot spawn rover")
            return False

        if not self._delete_client.wait_for_service(timeout_sec=10.0):
            self.get_logger().warn("/delete_entity not available")
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
            self.get_logger().warn("/spawn_entity not available")
            return False
        yaw = math.radians(yaw_deg)
        self._odom_local = None
        spawn_req = SpawnEntity.Request()
        spawn_req.name = "exomy"
        spawn_req.xml = self._urdf_xml
        spawn_req.initial_pose.position.x = float(x)
        spawn_req.initial_pose.position.y = float(y)
        spawn_req.initial_pose.position.z = float(z)
        spawn_req.initial_pose.orientation.z = math.sin(yaw / 2.0)
        spawn_req.initial_pose.orientation.w = math.cos(yaw / 2.0)
        future = self._spawn_client.call_async(spawn_req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=10.0)
        if future.done() and future.result().success:
            self.get_logger().info(f"Spawned exomy at ({x}, {y}, {z}), yaw={yaw_deg} deg")
            return True
        self.get_logger().warn(f"Spawn failed at ({x}, {y}, {z})")
        return False

    def stop(self):
        msg = Twist()
        for _ in range(3):
            self._cmd_vel_pub.publish(msg)
            rclpy.spin_once(self, timeout_sec=0.05)

    def run(self):
        if not self.teleport(SPAWN_X, SPAWN_Y, SPAWN_Z, SPAWN_YAW_DEG):
            return {"ok": False}

        deadline = time.time() + WAIT_STABLE_S
        while self._odom_local is None and time.time() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)

        onboard_path = os.path.join(OUT_DIR, "journey_onboard.mp4")
        chase_path = os.path.join(OUT_DIR, "journey_thirdperson.mp4")
        onboard_writer = None
        chase_writer = None
        n_onboard = 0
        n_chase = 0

        # ── Route state machine ──────────────────────────────────────────
        # v4: spawns directly in the rock field (see module docstring), so
        # there's just the one phase now. Driving/turning uses local-frame
        # relative odometry (frame-invariant) via DeadReckoner, never an
        # absolute odom->world transform (see DeadReckoner docstring for why
        # that broke an earlier version of this script).
        dr = DeadReckoner(SPAWN_X, SPAWN_Y, math.radians(SPAWN_YAW_DEG))
        phase = "rock_field"
        dr.begin_straight(self._odom_local)
        avoid_offset = None  # signed radians, current in-progress avoid pivot target

        print("Recording rock-field journey (odometry-driven, spawns in "
              "rock_cluster facing east)...", end="", flush=True)

        wall_deadline = time.time() + MAX_WALL_S
        last_cmd_t = 0.0
        last_write_t = 0.0
        last_progress_print = time.time()

        while time.time() < wall_deadline:
            now = time.time()
            if self._odom_local is None:
                rclpy.spin_once(self, timeout_sec=1.0 / FPS)
                continue
            local_now = self._odom_local
            wx, wy, wyaw = dr.current_pose(local_now)

            if now - last_cmd_t >= 1.0 / CMD_RATE_HZ:
                cmd = Twist()

                if phase == "rock_field":
                    if wx >= ROCK_FIELD_STOP_X:
                        phase = "done"
                    elif avoid_offset is not None:
                        direction = 1.0 if avoid_offset >= 0 else -1.0
                        if dr.pivot_progress_rad(local_now) >= abs(avoid_offset):
                            dr.complete_pivot(avoid_offset)
                            avoid_offset = None
                            dr.begin_straight(local_now)
                            cmd.linear.x = DRIVE_MS
                        else:
                            cmd.angular.z = PIVOT_RATE_RS * direction
                    elif not self.corridor_clear((wx, wy), wyaw):
                        # Full-360deg search for the smallest-deviation-from-
                        # EAST clear heading (not from current heading -- see
                        # module docstring, this is the v1->v3 fix). None
                        # means nothing clears anywhere -- "boxed in", same
                        # accepted ending as the real FSM's failsafe.
                        target_heading = self.pick_avoid_heading((wx, wy))
                        if target_heading is None:
                            phase = "boxed_in"
                        else:
                            dr.complete_straight(dr.straight_progress_m(local_now))
                            dr.begin_pivot(local_now)
                            avoid_offset = _angle_wrap(target_heading - wyaw)
                    else:
                        cmd.linear.x = DRIVE_MS

                if phase in ("done", "boxed_in"):
                    break

                self._cmd_vel_pub.publish(cmd)
                last_cmd_t = now

            if now - last_progress_print >= 10.0:
                print(f"\n  [{now - (wall_deadline - MAX_WALL_S):.0f}s] phase={phase} "
                      f"world=({wx:.2f},{wy:.2f}) yaw={math.degrees(wyaw):.0f}deg", end="", flush=True)
                last_progress_print = now

            rclpy.spin_once(self, timeout_sec=1.0 / FPS)

            if now - last_write_t < 1.0 / FPS:
                continue
            last_write_t = now

            if self._onboard_frame is not None:
                if onboard_writer is None:
                    h, w = self._onboard_frame.shape[:2]
                    onboard_writer = cv2.VideoWriter(
                        onboard_path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (w, h))
                onboard_writer.write(self._onboard_frame)
                n_onboard += 1

            if self._chase_frame is not None:
                if chase_writer is None:
                    h, w = self._chase_frame.shape[:2]
                    chase_writer = cv2.VideoWriter(
                        chase_path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (w, h))
                chase_writer.write(self._chase_frame)
                n_chase += 1

        # "boxed_in" (LiDAR found no clear heading anywhere) is a real,
        # accepted ending -- not a failure -- per the author: stopping is
        # fine if there genuinely isn't room to turn.
        ok = phase in ("done", "boxed_in")

        if onboard_writer is not None:
            onboard_writer.release()
        if chase_writer is not None:
            chase_writer.release()
        self.stop()

        print(f"\n done. phase={phase} ok={ok} "
              f"onboard_frames={n_onboard} chase_frames={n_chase}")

        return {
            "ok": ok and n_onboard > 0 and n_chase > 0,
            "phase": phase,
            "n_onboard": n_onboard,
            "n_chase": n_chase,
            "onboard_path": onboard_path if n_onboard else None,
            "chase_path": chase_path if n_chase else None,
        }


def main():
    rclpy.init()
    node = JourneyRecorder()
    try:
        result = node.run()
    finally:
        node.destroy_node()
        rclpy.shutdown()

    print("\n=== Summary ===")
    print(f"  journey: {'OK' if result.get('ok') else 'FAILED/INCOMPLETE'} "
          f"(ended in phase: {result.get('phase')})")
    if result.get("ok"):
        print(f"  {result['onboard_path']}")
        print(f"  {result['chase_path']}")


if __name__ == "__main__":
    main()
