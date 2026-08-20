"""
Purpose: Record two demo video clips per Gazebo traversability zone:
           1. Third-person "chase cam" (fixed camera per zone, added to
              mars_terrain.world 2026-08-17) -> <zone>_thirdperson.mp4
           2. Onboard rover camera (/exomy/camera/image_raw)            -> <zone>_onboard.mp4
         2026-08-17 v6 (author review, three rounds): the real
         terrain_controller FSM only drives straight or stops, so an
         earlier scripted continuous-weave fix (v1) looked "dizzy"/random,
         and two follow-up versions (v2, v3) scripted clean straight/pivot
         plans -- but timed each segment by WALL-CLOCK seconds
         (time.time()), assuming 1 real second ~= 1 simulated second. In
         this headless, no-GUI-render setup Gazebo actually runs faster
         than real time (measured: a "16s" plan produced 900 camera frames
         at the sensor's declared 15 Hz -- 60s of *simulated* driving, not
         16s), so the rover travelled ~3-4x further than intended and
         rammed straight into rocks it was supposed to stop short of
         (confirmed stuck: two frames 6-8s apart at pixel-identical rover
         position). v6 fixes this at the root: every segment is now
         completed by ODOMETRY FEEDBACK (/exomy/odom distance travelled /
         yaw turned), not a timer, so it is correct regardless of the
         sim/wall-clock speed ratio. Two motion plans, both published
         directly to /exomy/cmd_vel (terrain_controller_node and
         safety_watchdog_node must be killed first -- see "How to run" --
         so they don't fight this script for the same topic):
           - soil/sand/bedrock ("s_turn"): straight 0.6m -> pivot-in-place
             left 90 deg -> straight 0.6m -> pivot-in-place right 90 deg
             (back to original heading) -> straight 0.6m -> end.
           - rock_cluster/boulder_zone ("avoid"): straight toward the
             nearest rock until close, pivot-in-place away from it once,
             then straight in the new heading for the rest of the clip.
             Stand-off distances, turn direction and continue-distance were
             computed from the actual rock collision-sphere positions/radii
             in mars_terrain.world (both zones now swerve RIGHT/south --
             the safe/open direction for BOTH is south here, since
             rock_cluster's north side has a denser boulder_1+rock_2
             cluster and boulder_zone's north side has rock_6 sitting
             right next to the approach path). The two clips still look
             clearly different because their chase cameras view from
             different sides (rock_cluster: south, facing north;
             boulder_zone: east, facing west).
         Initial heading is also zone-specific: soil_zone and sand_zone
         (west side of the map) face -x (deeper into their own quadrant,
         away from the map centre / rock quadrant) instead of the old
         universal +x, which is what made the sand_zone clip bleed into
         rock_cluster's territory. bedrock_zone/rock_cluster/boulder_zone
         (east side) keep +x, which already points away from the centre.
         This is video-demo motion only, NOT a representation of the FSM's
         real STOP/speed behavior -- do not use these clips as evidence for
         report speed/stop claims.
Inputs:  Running ROS2 stack:
           ros2 launch simulation/launch/dinov2_controller_test.launch.py
         with terrain_controller_node and safety_watchdog_node killed
         (this script drives directly).
Outputs: docs/figures/gazebo_zone_videos/<zone>_thirdperson.mp4
         docs/figures/gazebo_zone_videos/<zone>_onboard.mp4
How to run:
    # Terminal 1
    source /opt/ros/humble/setup.bash && source ros2_ws/install/setup.bash
    ros2 launch simulation/launch/dinov2_controller_test.launch.py
    # once it's up, kill the two nodes that would otherwise fight this script:
    #   pkill -f terrain_controller_node.py; pkill -f safety_watchdog_node.py

    # Terminal 2
    source /opt/ros/humble/setup.bash && source ros2_ws/install/setup.bash
    python3 experiments/record_zone_videos.py
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

WAIT_STABLE_S = 3.0  # settle time after respawn before recording starts (wall clock, no motion)
FPS = 15             # matches <update_rate>15</update_rate> on both cameras
CMD_RATE_HZ = 30.0  # bumped from 10Hz: at ~3.7x sim speed, 10Hz commands left
                     # ~0.37 sim-seconds between refreshes, plausibly enough to
                     # trip the diff_drive plugin's cmd_vel timeout and cause
                     # stop-and-go motion (which inflated recording duration)
MAX_ZONE_WALL_S = 90.0  # safety cap per zone regardless of sim speed (plan should finish well before)

DRIVE_MS = 0.08       # forward speed for the s_turn (soil/sand/bedrock) plans
DRIVE_MS_ROCK = 0.20  # forward speed for the avoid (rock) plans (per review)
PIVOT_RATE_RS = 0.785398  # rad/s -- 90 deg pivot takes ~2.0s of *simulated* driving


def _straight(distance_m, speed=DRIVE_MS):
    return {"type": "straight", "distance": distance_m, "speed": speed}


def _pivot(angle_deg, direction):
    """direction: +1 = left (CCW), -1 = right (CW)"""
    return {"type": "pivot", "angle": math.radians(angle_deg), "direction": direction}


ZONE_POSES = [
    {
        "name": "soil_zone", "x": -7.5, "y": 6.0, "z": 0.15,
        "yaw_deg": 180.0,  # face -x: deeper into own (west) quadrant, away from centre
        "plan": [_straight(0.6), _pivot(90, +1),
                 _straight(0.6), _pivot(90, -1),
                 _straight(0.6)],
    },
    {
        "name": "bedrock_zone", "x": 7.5, "y": 6.0, "z": 0.15,
        "yaw_deg": 0.0,  # face +x: already away from centre (east quadrant)
        "plan": [_straight(0.6), _pivot(90, +1),
                 _straight(0.6), _pivot(90, -1),
                 _straight(0.6)],
    },
    {
        "name": "sand_zone", "x": -7.5, "y": -6.0, "z": 0.15,
        "yaw_deg": 180.0,  # face -x: deeper into own (west) quadrant, away from centre
        "plan": [_straight(0.6), _pivot(90, +1),
                 _straight(0.6), _pivot(90, -1),
                 _straight(0.6)],
    },
    {
        # v6: segment completion is now odometry-distance-based (see module
        # docstring) instead of wall-clock-timed, which is what let the
        # rover overshoot into rock_1 in v2/v3. rock_1 sphere r=0.143m at
        # (3.5,-4.0), 1.5m ahead of spawn (2.0,-4.0); stop 0.7m in, leaving
        # 0.8m clearance to its centre. Swerve RIGHT/south -- away from the
        # denser boulder_1(4.5,-3.5)+rock_2(6.0,-3.0) cluster to the north,
        # and toward the rock_cluster camera (sits south facing north, so a
        # south swerve also stays framed). Continue-straight checked clear
        # of rock_3(5.0,-5.0,r=0.163) and rock_6(3.5,-7.5,r=0.130) for the
        # full 1.2m at this heading.
        "name": "rock_cluster", "x": 2.0, "y": -4.0, "z": 0.15,
        "yaw_deg": 0.0,
        "plan": [_straight(0.7, speed=DRIVE_MS_ROCK),
                 _pivot(70, -1),
                 _straight(1.2, speed=DRIVE_MS_ROCK)],
    },
    {
        # v6: same odometry-feedback fix. The actual v2/v3 collision here
        # was with rock_6(3.5,-7.5,r=0.130) -- only 0.5m off the straight
        # approach path at x=3.5, not with boulder_2 itself. Stop point at
        # 0.8m travel (x=2.8) keeps ~0.86m clearance to rock_6's centre.
        # Swerves RIGHT/south (like rock_cluster -- a north swerve runs
        # straight at rock_6) at only 60 deg so it also stays toward the
        # boulder_zone camera (sits east facing west) instead of exiting
        # sideways. Continue-straight checked clear of boulder_2, rock_7,
        # rock_8 for the full 1.0m.
        "name": "boulder_zone", "x": 2.0, "y": -8.0, "z": 0.15,
        "yaw_deg": 0.0,
        "plan": [_straight(0.8, speed=DRIVE_MS_ROCK),
                 _pivot(60, -1),
                 _straight(1.0, speed=DRIVE_MS_ROCK)],
    },
]


def _yaw_from_quat(z: float, w: float) -> float:
    """Yaw from a planar (roll=pitch=0) quaternion's z,w components."""
    return 2.0 * math.atan2(z, w)


def _angle_wrap(a: float) -> float:
    """Wrap an angle to [-pi, pi]."""
    return (a + math.pi) % (2.0 * math.pi) - math.pi


class PlanRunner:
    """Advances through a drive plan using odometry feedback (distance
    travelled / yaw turned since the current segment started), not wall-clock
    time -- correct regardless of how fast Gazebo's sim clock runs relative
    to real time (measured faster-than-real-time in this headless setup)."""

    def __init__(self, plan):
        self._plan = plan
        self._idx = 0
        self._seg_start_xy = None
        self._seg_start_yaw = None

    @property
    def done(self) -> bool:
        return self._idx >= len(self._plan)

    def cmd(self, x: float, y: float, yaw: float) -> Twist:
        msg = Twist()
        if self.done:
            return msg

        seg = self._plan[self._idx]
        if self._seg_start_xy is None:
            self._seg_start_xy = (x, y)
            self._seg_start_yaw = yaw

        if seg["type"] == "straight":
            travelled = math.hypot(x - self._seg_start_xy[0], y - self._seg_start_xy[1])
            if travelled >= seg["distance"]:
                self._advance()
                return self.cmd(x, y, yaw)
            msg.linear.x = seg["speed"]
        else:  # pivot
            turned = abs(_angle_wrap(yaw - self._seg_start_yaw))
            if turned >= seg["angle"]:
                self._advance()
                return self.cmd(x, y, yaw)
            msg.angular.z = PIVOT_RATE_RS * seg["direction"]
        return msg

    def _advance(self):
        self._idx += 1
        self._seg_start_xy = None
        self._seg_start_yaw = None


class ZoneVideoRecorder(Node):
    def __init__(self):
        super().__init__("zone_video_recorder")
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
        self._thirdperson_frame = None
        self.create_subscription(Image, "/exomy/camera/image_raw", self._onboard_cb, 1)
        self._tp_sub = None  # (re)subscribed per zone below

        self._odom_pose = None  # (x, y, yaw), ground truth for driving + verification
        self.create_subscription(Odometry, "/exomy/odom", self._odom_cb, 1)

    def _odom_cb(self, msg):
        p = msg.pose.pose.position
        o = msg.pose.pose.orientation
        self._odom_pose = (p.x, p.y, _yaw_from_quat(o.z, o.w))

    def _urdf_cb(self, msg):
        self._urdf_xml = msg.data

    def _onboard_cb(self, msg):
        self._onboard_frame = self._to_bgr(msg)

    def _thirdperson_cb(self, msg):
        self._thirdperson_frame = self._to_bgr(msg)

    @staticmethod
    def _to_bgr(msg: Image) -> np.ndarray:
        arr = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width, -1)[:, :, :3]
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

    def subscribe_thirdperson(self, zone_name: str):
        if self._tp_sub is not None:
            self.destroy_subscription(self._tp_sub)
        self._thirdperson_frame = None
        topic = f"/gazebo_cam/{zone_name}/cam/image_raw"
        self._tp_sub = self.create_subscription(Image, topic, self._thirdperson_cb, 1)
        return topic

    def teleport(self, x: float, y: float, z: float, yaw_deg: float = 0.0) -> bool:
        if self._urdf_xml is None:
            deadline = time.time() + 8
            while self._urdf_xml is None and time.time() < deadline:
                rclpy.spin_once(self, timeout_sec=0.2)
        if self._urdf_xml is None:
            self.get_logger().warn("No URDF -- cannot respawn rover")
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
        self._odom_pose = None
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
            self.get_logger().info(f"Respawned exomy at ({x}, {y}, {z}), yaw={yaw_deg} deg")
            return True
        self.get_logger().warn(f"Respawn failed at ({x}, {y}, {z})")
        return False

    def stop(self):
        """Publish zero cmd_vel a few times (settle / end of clip)."""
        msg = Twist()
        for _ in range(3):
            self._cmd_vel_pub.publish(msg)
            rclpy.spin_once(self, timeout_sec=0.05)

    def record_zone(self, zone: dict) -> dict:
        name = zone["name"]
        print(f"\n=== {name} ===")
        tp_topic = self.subscribe_thirdperson(name)

        if not self.teleport(zone["x"], zone["y"], zone["z"], zone.get("yaw_deg", 0.0)):
            print(f"  SKIPPED (respawn failed)")
            return {"zone": name, "ok": False}

        # wait for odometry to publish post-respawn before driving/recording
        deadline = time.time() + WAIT_STABLE_S
        while self._odom_pose is None and time.time() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)

        onboard_path = os.path.join(OUT_DIR, f"{name}_onboard.mp4")
        thirdperson_path = os.path.join(OUT_DIR, f"{name}_thirdperson.mp4")
        onboard_writer = None
        thirdperson_writer = None

        print(f"  Recording (odometry-driven plan; onboard topic "
              f"/exomy/camera/image_raw, 3rd-person topic {tp_topic})...",
              end="", flush=True)
        runner = PlanRunner(zone["plan"])
        odom_start = self._odom_pose
        wall_deadline = time.time() + MAX_ZONE_WALL_S
        n_onboard = 0
        n_thirdperson = 0
        last_cmd_t = 0.0
        # Gazebo runs faster than real time in this headless setup (measured
        # ~3.7x): the cameras' declared 15 Hz is in SIM time, so frames can
        # arrive far faster than 15/s in WALL-CLOCK time. Writing every
        # arrival at a 15fps tag stretched a real ~30-50s recording into a
        # 70-180s video. Throttle writes to real wall-clock time instead, so
        # the output video's length matches how long the drive actually took.
        last_write_t = 0.0
        while not runner.done and time.time() < wall_deadline:
            now = time.time()
            if now - last_cmd_t >= 1.0 / CMD_RATE_HZ and self._odom_pose is not None:
                x, y, yaw = self._odom_pose
                self._cmd_vel_pub.publish(runner.cmd(x, y, yaw))
                last_cmd_t = now

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

            if self._thirdperson_frame is not None:
                if thirdperson_writer is None:
                    h, w = self._thirdperson_frame.shape[:2]
                    thirdperson_writer = cv2.VideoWriter(
                        thirdperson_path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (w, h))
                thirdperson_writer.write(self._thirdperson_frame)
                n_thirdperson += 1

        timed_out = not runner.done

        if onboard_writer is not None:
            onboard_writer.release()
        if thirdperson_writer is not None:
            thirdperson_writer.release()

        self.stop()
        odom_end = self._odom_pose
        dist_m = None
        if odom_start is not None and odom_end is not None:
            dist_m = math.hypot(odom_end[0] - odom_start[0], odom_end[1] - odom_start[1])
        status = " [PLAN DID NOT FINISH -- possible stuck/collision]" if timed_out else ""
        print(f" done ({n_onboard} onboard frames, {n_thirdperson} thirdperson frames, "
              f"net odom displacement {dist_m:.2f}m){status}"
              if dist_m is not None else
              f" done ({n_onboard} onboard frames, {n_thirdperson} thirdperson frames, "
              f"odom unavailable){status}")

        return {
            "zone": name,
            "ok": n_onboard > 0 and n_thirdperson > 0 and not timed_out,
            "n_onboard": n_onboard,
            "n_thirdperson": n_thirdperson,
            "onboard_path": onboard_path if n_onboard else None,
            "thirdperson_path": thirdperson_path if n_thirdperson else None,
        }


def main():
    rclpy.init()
    node = ZoneVideoRecorder()
    results = []
    try:
        for zone in ZONE_POSES:
            results.append(node.record_zone(zone))
    finally:
        node.destroy_node()
        rclpy.shutdown()

    print("\n=== Summary ===")
    ok = 0
    for r in results:
        status = "OK" if r.get("ok") else "FAILED/EMPTY/STUCK"
        print(f"  {r['zone']}: {status}")
        if r.get("ok"):
            ok += 1
    print(f"\n{ok}/{len(results)} zones recorded successfully.")
    print(f"Videos in: {OUT_DIR}")


if __name__ == "__main__":
    main()
