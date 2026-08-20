"""
Purpose: Diagnostic-only helper for l5_lite_minimal_diag.launch.py -- publishes
         traversability_grid.py's ground-truth static grid (build_static_grid(),
         Condition A style) once to /traversability_costmap with the same
         TRANSIENT_LOCAL QoS live_traversability_costmap_node.py uses, so
         l5_lite_planner_node.py (a late-joining subscriber) still receives
         it without needing dinov2_terrain_node running at all. Isolates
         compute load (DINOv2 measured at ~170% CPU) from L5-lite's own
         plan-and-follow logic when diagnosing why the rover wasn't moving
         in the full pipeline (systematic-debugging skill, 2026-07-17).
         NOT a reported thesis result on its own -- purely diagnostic.
Inputs:  None.
Outputs: /traversability_costmap (nav_msgs/OccupancyGrid), published once.
How to run: Launched automatically by l5_lite_minimal_diag.launch.py.
Project: Onboard Visual Foundation Models for Mars Terrain Perception and Rover Navigation
"""
import rclpy
from geometry_msgs.msg import Pose
from nav_msgs.msg import OccupancyGrid
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy

from fm_perception.traversability_grid import (
    build_static_grid, HEIGHT_CELLS, ORIGIN_X_M, ORIGIN_Y_M, RESOLUTION_M, WIDTH_CELLS,
)


def main():
    rclpy.init()
    node = Node("l5_lite_static_costmap_publisher")

    qos = QoSProfile(depth=1)
    qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
    qos.reliability = ReliabilityPolicy.RELIABLE
    pub = node.create_publisher(OccupancyGrid, "/traversability_costmap", qos)

    msg = OccupancyGrid()
    msg.header.frame_id = "map"
    msg.header.stamp = node.get_clock().now().to_msg()
    msg.info.resolution = RESOLUTION_M
    msg.info.width = WIDTH_CELLS
    msg.info.height = HEIGHT_CELLS
    origin = Pose()
    origin.position.x = ORIGIN_X_M
    origin.position.y = ORIGIN_Y_M
    msg.info.origin = origin
    msg.data = build_static_grid()

    # Publish repeatedly for a while rather than once-then-exit: a
    # TRANSIENT_LOCAL publisher's history cache is tied to the publisher
    # entity's lifetime in the default RMW (FastDDS) -- destroying the node
    # immediately after publish() can drop the message before a
    # late-joining subscriber (l5_lite_planner_node, starting 2s later)
    # ever sees it. The real pipeline doesn't have this problem because
    # live_traversability_costmap_node.py is a persistent node that never
    # exits; this script is diagnostic-only and needs to stay alive.
    import time
    end_t = time.time() + 30.0
    while time.time() < end_t:
        msg.header.stamp = node.get_clock().now().to_msg()
        pub.publish(msg)
        time.sleep(1.0)
    node.get_logger().info(
        f"Published static ground-truth costmap repeatedly for 30s: "
        f"{WIDTH_CELLS}x{HEIGHT_CELLS} cells"
    )

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
