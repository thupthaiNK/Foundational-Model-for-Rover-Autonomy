"""
Purpose: L4 Phase A2 -- verifies live_traversability_costmap_node.py's new
         pose_source=slam wiring actually works end-to-end: drive a gentle
         arc near Q4 (same pattern as Phase A's slam_pose_accuracy_test.py)
         while DINOv2 classifies terrain and slam_toolbox estimates pose,
         then check that /traversability_costmap cells got painted, and
         that the painted cells' world coordinates are consistent with
         where the rover actually was (cross-checked against Gazebo's
         ground-truth /exomy/odom -- used here only for validating this
         test, not fed into the costmap node itself, which uses
         slam_toolbox's /pose exclusively via pose_source:=slam).
         This is a plumbing/integration check, not a terrain-classification
         accuracy experiment -- classification accuracy in this exact
         Q4 area is already established (§4.8.3-§4.8.11).
Inputs:  /exomy/odom (nav_msgs/Odometry)                       ground truth, validation only
         /traversability_costmap (nav_msgs/OccupancyGrid)       what this test checks
Outputs: experiments/results/slam_costmap_test_summary.txt
How to run:
    # Terminal 1:
    source /opt/ros/humble/setup.bash && source ros2_ws/install/setup.bash
    ros2 launch simulation/launch/slam_costmap_test.launch.py

    # Terminal 2 (after Terminal 1 shows slam_toolbox + DINOv2 + costmap ready):
    source /opt/ros/humble/setup.bash && source ros2_ws/install/setup.bash
    python3 experiments/slam_costmap_test.py
Project: Onboard Visual Foundation Models for Mars Terrain Perception and Rover Navigation
"""

import math
import os
import time

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import OccupancyGrid, Odometry
from rclpy.node import Node

LINEAR_SPEED  = 0.10
ANGULAR_SPEED = 0.08
ARC_DURATION_S = 90.0

# Must match ros2_ws/src/fm_perception/fm_perception/traversability_grid.py
RESOLUTION_M = 0.1
ORIGIN_X_M = -15.0
ORIGIN_Y_M = -12.0
WIDTH_CELLS = 300
HEIGHT_CELLS = 240

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


class SlamCostmapCheck(Node):

    def __init__(self):
        super().__init__("slam_costmap_check")
        self._gt_positions = []  # list of (x, y)
        self._latest_grid = None

        self.create_subscription(Odometry, "/exomy/odom", self._odom_cb, 10)
        self.create_subscription(OccupancyGrid, "/traversability_costmap", self._grid_cb, 10)
        self.pub_cmd = self.create_publisher(Twist, "/exomy/cmd_vel", 10)

    def _odom_cb(self, msg: Odometry) -> None:
        self._gt_positions.append((msg.pose.pose.position.x, msg.pose.pose.position.y))

    def _grid_cb(self, msg: OccupancyGrid) -> None:
        self._latest_grid = msg

    def send_twist(self, linear_x: float, angular_z: float) -> None:
        msg = Twist()
        msg.linear.x = linear_x
        msg.angular.z = angular_z
        self.pub_cmd.publish(msg)


def cell_to_world(row: int, col: int) -> tuple:
    x = ORIGIN_X_M + (col + 0.5) * RESOLUTION_M
    y = ORIGIN_Y_M + (row + 0.5) * RESOLUTION_M
    return x, y


def main():
    rclpy.init()
    node = SlamCostmapCheck()
    node.get_logger().info(
        "Phase A2 costmap check starting -- driving a gentle arc near Q4 "
        "while DINOv2 + slam_toolbox(pose) feed the live costmap."
    )
    rclpy.spin_once(node, timeout_sec=2.0)

    end_t = time.time() + ARC_DURATION_S
    while time.time() < end_t:
        node.send_twist(LINEAR_SPEED, ANGULAR_SPEED)
        rclpy.spin_once(node, timeout_sec=0.1)
    node.send_twist(0.0, 0.0)
    rclpy.spin_once(node, timeout_sec=2.0)

    lines = []
    lines.append(f"Ground-truth odom samples received: {len(node._gt_positions)}")
    if node._gt_positions:
        xs = [p[0] for p in node._gt_positions]
        ys = [p[1] for p in node._gt_positions]
        lines.append(f"Ground-truth driven range: x=[{min(xs):.2f},{max(xs):.2f}] y=[{min(ys):.2f},{max(ys):.2f}]")

    if node._latest_grid is None:
        lines.append("FAIL: /traversability_costmap was never received at all.")
    else:
        grid = node._latest_grid
        data = grid.data
        # OccupancyGridBuilder's grid initialises every cell to COST_SOIL (0),
        # not -1/unknown (see traversability_grid.py) -- "painted" therefore
        # means "not equal to the COST_SOIL default", not "not equal to -1".
        COST_SOIL_DEFAULT = 0
        painted = [(i, v) for i, v in enumerate(data) if v != COST_SOIL_DEFAULT]
        lines.append(f"Costmap cells total: {len(data)}, painted (non-default) cells: {len(painted)}")
        if painted:
            painted_world = []
            for idx, v in painted:
                row = idx // WIDTH_CELLS
                col = idx % WIDTH_CELLS
                wx, wy = cell_to_world(row, col)
                painted_world.append((wx, wy, v))
            pxs = [p[0] for p in painted_world]
            pys = [p[1] for p in painted_world]
            lines.append(f"Painted-cell world range: x=[{min(pxs):.2f},{max(pxs):.2f}] y=[{min(pys):.2f},{max(pys):.2f}]")
            values_seen = sorted(set(v for _, _, v in painted_world))
            lines.append(f"Cost values seen in painted cells: {values_seen}")
            lines.append("Sample painted cells (up to 10): " + str(painted_world[:10]))
        else:
            lines.append("FAIL: costmap received but zero cells were ever painted (all -1).")

    summary = "\n".join(lines)
    print("\n" + "=" * 70)
    print(summary)
    print("=" * 70)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "slam_costmap_test_summary.txt"), "w") as f:
        f.write(summary + "\n")

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
