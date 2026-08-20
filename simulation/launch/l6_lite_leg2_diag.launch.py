"""
Purpose: Minimal diagnostic launch for "L6-lite" leg 2 (systematic-debugging,
         2026-07-18). The full round-trip test showed leg 1 (to GOAL_POSE)
         succeeding and the autonomous waypoint switch firing correctly, but
         leg 2 (the return to START_POSE) making essentially zero net
         ground-truth progress over 550s -- a real, different failure mode
         from L5-lite's earlier "just needed more time" case. Reproduces
         leg 2's exact starting condition directly (spawn near GOAL_POSE,
         waypoint list identical to the full test) instead of re-running the
         full ~350s leg 1 first, and swaps DINOv2 + the live costmap for a
         static all-free costmap (l5_lite_static_costmap_publisher.py) to
         remove perception/hazard-classification as a variable, isolating
         the planner+controller behaviour specifically -- same isolation
         technique as l5_lite_minimal_diag.launch.py. NOT a reported thesis
         result on its own -- purely a diagnostic step.
Inputs:  None.
Outputs: /exomy/cmd_vel (from l5_lite_planner_node.py)
How to run:
    source /opt/ros/humble/setup.bash
    source ros2_ws/install/setup.bash
    ros2 launch simulation/launch/l6_lite_leg2_diag.launch.py
Project: Onboard Visual Foundation Models for Mars Terrain Perception and Rover Navigation
"""
import os

import xacro
from launch import LaunchDescription
from launch.actions import ExecuteProcess, TimerAction
from launch_ros.actions import Node


def generate_launch_description():
    sim_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    urdf_path = os.path.join(sim_dir, "urdf", "exomy.urdf.xacro")
    world_path = os.path.join(sim_dir, "worlds", "mars_terrain.world")

    robot_description = xacro.process_file(urdf_path).toxml()

    models_dir = os.path.join(sim_dir, "models")
    existing_model_path = os.environ.get("GAZEBO_MODEL_PATH", "")
    gazebo_model_path = f"{models_dir}:{existing_model_path}" if existing_model_path else models_dir

    gazebo_server = ExecuteProcess(
        cmd=[
            "gzserver", "--verbose", world_path,
            "-s", "libgazebo_ros_init.so",
            "-s", "libgazebo_ros_factory.so",
        ],
        output="screen",
        additional_env={"GAZEBO_MODEL_PATH": gazebo_model_path},
    )

    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="screen",
        parameters=[{"robot_description": robot_description, "use_sim_time": True}],
    )

    joint_state_publisher = Node(
        package="joint_state_publisher",
        executable="joint_state_publisher",
        name="joint_state_publisher",
        output="screen",
        parameters=[{"use_sim_time": True}],
    )

    # Spawn near GOAL_POSE (-7.5, -9.0) -- leg 2's exact starting condition --
    # instead of START_POSE (7.5, 1.0), skipping leg 1's ~350s entirely.
    spawn_robot = TimerAction(
        period=35.0,
        actions=[
            Node(
                package="gazebo_ros",
                executable="spawn_entity.py",
                name="spawn_exomy",
                output="screen",
                arguments=[
                    "-topic", "robot_description",
                    "-entity", "exomy",
                    "-x", "-7.4", "-y", "-8.9", "-z", "0.15",
                ],
            )
        ],
    )

    slam_toolbox_node = TimerAction(
        period=42.0,
        actions=[
            Node(
                package="slam_toolbox",
                executable="async_slam_toolbox_node",
                name="slam_toolbox",
                output="screen",
                parameters=[{
                    "odom_frame": "odom",
                    "map_frame": "map",
                    "base_frame": "base_link",
                    "scan_topic": "/scan",
                    "mode": "mapping",
                    "use_sim_time": True,
                    "transform_publish_period": 0.02,
                    "map_update_interval": 2.0,
                    "resolution": 0.05,
                    "max_laser_range": 12.0,
                    "minimum_travel_distance": 0.2,
                    "minimum_travel_heading": 0.2,
                    "transform_timeout": 0.5,
                    "scan_queue_size": 50,
                }],
            )
        ],
    )

    static_costmap_publisher = TimerAction(
        period=44.0,
        actions=[
            ExecuteProcess(
                cmd=["python3", os.path.join(
                    os.path.dirname(sim_dir), "experiments", "l5_lite_static_costmap_publisher.py"
                )],
                output="screen",
            )
        ],
    )

    # Same waypoint list as the full round-trip test: GOAL_POSE (default
    # goal_x/goal_y) then START_POSE -- spawning near GOAL_POSE means the
    # switch to waypoint 1 should fire almost immediately.
    l5_lite_node = TimerAction(
        period=46.0,
        actions=[
            Node(
                package="fm_perception",
                executable="l5_lite_planner_node.py",
                name="l5_lite_planner_node",
                output="screen",
                parameters=[{
                    "use_sim_time": True,
                    "waypoint_xs": [7.5],
                    "waypoint_ys": [1.0],
                }],
            )
        ],
    )

    return LaunchDescription([
        gazebo_server,
        robot_state_publisher,
        joint_state_publisher,
        spawn_robot,
        slam_toolbox_node,
        static_costmap_publisher,
        l5_lite_node,
    ])
