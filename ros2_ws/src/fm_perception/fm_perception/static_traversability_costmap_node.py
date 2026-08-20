#!/usr/bin/env python3
"""
Purpose: Publishes the Condition A (known-hazard) traversability costmap
         once, built from the known 5-zone ground-truth layout via
         traversability_grid.build_static_grid(). Never updates — represents
         "navigation with the hazard zones already known in advance."
Inputs:  None (grid is built from the compile-time ZONES geometry).
Outputs: /traversability_costmap (nav_msgs/OccupancyGrid), transient_local
         QoS so a late-joining subscriber (Nav2's static_layer) still gets it.
How to run:
    source /opt/ros/humble/setup.bash
    source ros2_ws/install/setup.bash
    ros2 run fm_perception static_traversability_costmap_node.py
Project: Onboard Visual Foundation Models for Mars Terrain Perception and Rover Navigation
"""
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Pose

from fm_perception.traversability_grid import (
    build_static_grid, RESOLUTION_M, ORIGIN_X_M, ORIGIN_Y_M,
    WIDTH_CELLS, HEIGHT_CELLS,
)


class StaticTraversabilityCostmapNode(Node):

    def __init__(self):
        super().__init__("static_traversability_costmap_node")
        qos = QoSProfile(depth=1)
        qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        qos.reliability = ReliabilityPolicy.RELIABLE
        self.pub = self.create_publisher(OccupancyGrid, "/traversability_costmap", qos)
        self._publish_grid()
        self.get_logger().info("Static traversability costmap published (Condition A).")

    def _publish_grid(self):
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
        msg.data = build_static_grid()
        self.pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = StaticTraversabilityCostmapNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
