#!/usr/bin/env python3
"""
Purpose: Publishes the Condition B (live, no prior knowledge) traversability
         costmap. Starts entirely free and is updated in real time from
         streaming DINOv2 classifications + a pose source, painting a small
         patch of cells at a fixed lookahead distance ahead of the rover.
         This is a deliberate simplification — DINOv2 classifies the whole
         forward camera frame as one label, there is no per-pixel
         segmentation — and is stated as a limitation in the thesis
         write-up (Ch4/Ch6).
         Pose source is selectable via the pose_source parameter (2026-07-17
         addition, L4 Phase A2): "odom" (default, unchanged from the
         original D1 experiment, §4.8.13 — Gazebo's ground-truth
         /exomy/odom) or "slam" (slam_toolbox's /pose topic,
         geometry_msgs/PoseWithCovarianceStamped). This is the piece D1
         explicitly could not close (§4.8.13, §5.6.5): its live costmap
         used ground-truth odometry as a stand-in for real localisation.
         PoseWithCovarianceStamped has the identical nested
         pose.pose.position/pose.pose.orientation field layout as
         Odometry's pose field, so the same extraction logic
         (yaw_from_quaternion, position access) applies unchanged to either
         source — only the subscription's message type and topic differ.
         confidence_aware_painting (2026-07-19, opt-in, default false): root
         cause fix for the start-cell-hazard deadlock item 1 of the L1-L6
         further-work plan already patched around symptomatically. See
         OccupancyGridBuilder.paint_lookahead()'s docstring
         (traversability_grid.py) for the mechanism.
Inputs:  /terrain_classification (std_msgs/String, "label:confidence")
         /exomy/odom (nav_msgs/Odometry) — if pose_source=="odom" (default)
         /pose (geometry_msgs/PoseWithCovarianceStamped) — if pose_source=="slam"
Outputs: /traversability_costmap (nav_msgs/OccupancyGrid), republished on
         every update so Nav2's static_layer (subscribe_to_updates: true)
         rebuilds its costmap.
How to run:
    source /opt/ros/humble/setup.bash
    source ros2_ws/install/setup.bash
    ros2 run fm_perception live_traversability_costmap_node.py
    ros2 run fm_perception live_traversability_costmap_node.py --ros-args -p pose_source:=slam
Project: Onboard Visual Foundation Models for Mars Terrain Perception and Rover Navigation
"""
import math

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from nav_msgs.msg import OccupancyGrid, Odometry
from geometry_msgs.msg import Pose, PoseWithCovarianceStamped
from std_msgs.msg import String

from fm_perception.traversability_grid import (
    OccupancyGridBuilder, RESOLUTION_M, ORIGIN_X_M, ORIGIN_Y_M,
    WIDTH_CELLS, HEIGHT_CELLS,
)


def yaw_from_quaternion(q) -> float:
    """Extract yaw (rad) from a geometry_msgs/Quaternion."""
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


class LiveTraversabilityCostmapNode(Node):

    def __init__(self):
        super().__init__("live_traversability_costmap_node")
        self.declare_parameter("lookahead_m", 0.6)
        self.declare_parameter("patch_radius_m", 0.3)
        self.declare_parameter("pose_source", "odom")  # "odom" or "slam"
        # Opt-in for frontier exploration-lite: initialise every cell to -1
        # (OccupancyGrid's standard "unknown") instead of COST_SOIL=0, so
        # "never assessed by perception" is distinguishable from "assessed
        # as soil". Default false -- zero behaviour change for every other
        # launch/result that uses this node.
        self.declare_parameter("init_unknown", False)
        # Root-cause fix (2026-07-19, item 6 of the L1-L6 further-work plan)
        # for the start-cell-hazard deadlock this session already patched
        # around symptomatically (frontier_explorer.py's
        # grid_with_start_freed): opt-in, default False -- every existing
        # official result used unconditional overwrite and must remain
        # reproducible. Only meaningful together with init_unknown=true; see
        # OccupancyGridBuilder.paint_lookahead()'s docstring.
        self.declare_parameter("confidence_aware_painting", False)
        # A2 epistemic uncertainty map (2026-07-19, opt-in, default false):
        # keep a per-cell latest-write confidence store alongside the cost
        # grid and publish it on /traversability_confidence (OccupancyGrid,
        # confidence scaled 0-100, -1 = never observed) for the planner's
        # re-observation mode and the run recorder. The confidence value was
        # always present in the /terrain_classification message; until this,
        # it was parsed off and discarded.
        self.declare_parameter("track_confidence", False)
        # SuperMap-inspired log-odds label fusion (2026-07-19, opt-in,
        # default false, only meaningful with track_confidence=true):
        # replaces the confidence store's latest-write update with log-odds
        # accumulation across observations that agree on the same label.
        # See OccupancyGridBuilder's docstring (traversability_grid.py) for
        # the mechanism. Every existing official result (A2, §4.8.30) used
        # latest-write and must remain reproducible with this left False.
        self.declare_parameter("bayesian_fusion", False)
        lookahead_m = self.get_parameter("lookahead_m").get_parameter_value().double_value
        patch_radius_m = self.get_parameter("patch_radius_m").get_parameter_value().double_value
        pose_source = self.get_parameter("pose_source").get_parameter_value().string_value
        init_unknown = self.get_parameter("init_unknown").get_parameter_value().bool_value
        confidence_aware_painting = self.get_parameter(
            "confidence_aware_painting").get_parameter_value().bool_value
        self.track_confidence = self.get_parameter(
            "track_confidence").get_parameter_value().bool_value
        bayesian_fusion = self.get_parameter(
            "bayesian_fusion").get_parameter_value().bool_value

        builder_kwargs = dict(lookahead_m=lookahead_m, patch_radius_m=patch_radius_m,
                               confidence_aware_painting=confidence_aware_painting,
                               track_confidence=self.track_confidence,
                               bayesian_fusion=bayesian_fusion)
        if init_unknown:
            builder_kwargs["init_cost"] = -1
        self.builder = OccupancyGridBuilder(**builder_kwargs)
        self.latest_pose = None  # (x, y, yaw_rad)

        qos = QoSProfile(depth=1)
        qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        qos.reliability = ReliabilityPolicy.RELIABLE
        self.pub = self.create_publisher(OccupancyGrid, "/traversability_costmap", qos)
        self.pub_confidence = (
            self.create_publisher(OccupancyGrid, "/traversability_confidence", qos)
            if self.track_confidence else None
        )

        if pose_source == "slam":
            self.create_subscription(PoseWithCovarianceStamped, "/pose", self._on_slam_pose, 10)
        else:
            self.create_subscription(Odometry, "/exomy/odom", self._on_odom, 10)
        self.create_subscription(String, "/terrain_classification", self._on_terrain, 10)

        self._publish_grid()  # publish the initial all-free grid immediately
        self.get_logger().info(
            f"Live traversability costmap node started (Condition B). pose_source={pose_source}"
        )

    def _on_odom(self, msg: Odometry):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        yaw = yaw_from_quaternion(msg.pose.pose.orientation)
        self.latest_pose = (x, y, yaw)

    def _on_slam_pose(self, msg: PoseWithCovarianceStamped):
        # Identical extraction to _on_odom above -- PoseWithCovarianceStamped
        # has the same nested pose.pose.position/pose.pose.orientation
        # layout as Odometry's pose field.
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        yaw = yaw_from_quaternion(msg.pose.pose.orientation)
        self.latest_pose = (x, y, yaw)

    def _on_terrain(self, msg: String):
        if self.latest_pose is None:
            return  # no odom yet — nothing to anchor the classification to
        parts = msg.data.split(":")
        label = parts[0]
        confidence = None
        if self.track_confidence and len(parts) > 1:
            try:
                confidence = float(parts[1])
            except ValueError:
                pass  # malformed message -- paint the cost, skip confidence
        x, y, yaw = self.latest_pose
        self.builder.paint_lookahead(x, y, yaw, label, confidence=confidence)
        self._publish_grid()

    def _grid_msg(self, data) -> OccupancyGrid:
        msg = OccupancyGrid()
        msg.header.frame_id = "map"
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.info.resolution = RESOLUTION_M
        msg.info.width = WIDTH_CELLS
        msg.info.height = HEIGHT_CELLS
        origin = Pose()
        origin.position.x = ORIGIN_X_M
        origin.position.y = ORIGIN_Y_M
        msg.info.origin = origin
        msg.data = data
        return msg

    def _publish_grid(self):
        self.pub.publish(self._grid_msg(self.builder.grid))
        if self.pub_confidence is not None:
            # Confidence scaled to OccupancyGrid's int8 range: 0-100, with
            # -1 preserved as "never observed" (same sentinel as the cost
            # grid's own unknown).
            self.pub_confidence.publish(self._grid_msg(
                [-1 if c < 0 else int(round(c * 100)) for c in self.builder.confidence]
            ))


def main(args=None):
    rclpy.init(args=args)
    node = LiveTraversabilityCostmapNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
