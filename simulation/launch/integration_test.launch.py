"""
Purpose: Integration test — Gazebo (headless) + ExoMy robot + CLIP terrain node.
         Tests that the full pipeline works: camera image → CLIP → terrain label.
         Uses gzserver only (no GUI) for WSL2 reliability.
Inputs:  None (launches everything automatically)
Outputs: /terrain_classification  — terrain label published by CLIP node
         /terrain_viz             — annotated image (optional)
How to run:
    # Terminal 1 — build and source first
    cd ros2_ws && colcon build --packages-select fm_perception
    source /opt/ros/humble/setup.bash
    source ros2_ws/install/setup.bash

    # Terminal 2 — launch integration test
    source /opt/ros/humble/setup.bash
    source ros2_ws/install/setup.bash
    ros2 launch simulation/launch/integration_test.launch.py

    # Terminal 3 — verify output (in a new terminal, sourced)
    ros2 topic echo /terrain_classification
    ros2 topic hz  /terrain_classification
    ros2 topic list

Project: Onboard Visual Foundation Models for Mars Terrain Perception and Rover Navigation
"""

import os
import xacro
from launch import LaunchDescription
from launch.actions import ExecuteProcess, TimerAction, DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():

    # ── Paths ──────────────────────────────────────────────────────────────
    sim_dir    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    urdf_path  = os.path.join(sim_dir, "urdf", "exomy.urdf.xacro")
    world_path = os.path.join(sim_dir, "worlds", "mars_terrain.world")

    robot_description = xacro.process_file(urdf_path).toxml()

    # ── 1. Gazebo server — headless (no GUI) ───────────────────────────────
    # Use gzserver only for WSL2 reliability. Add gzclient manually if needed.
    gazebo_server = ExecuteProcess(
        cmd=[
            "gzserver", "--verbose", world_path,
            "-s", "libgazebo_ros_init.so",
            "-s", "libgazebo_ros_factory.so",
            # Note: libgazebo_ros_state.so not compatible with Gazebo 11.10.2
        ],
        output="screen",
    )

    # ── 2. Robot state publisher ───────────────────────────────────────────
    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="screen",
        parameters=[{
            "robot_description": robot_description,
            "use_sim_time": True,
        }],
    )

    # ── Spawn position args (for multi-zone traversability experiment) ────
    spawn_x = DeclareLaunchArgument("spawn_x", default_value="0.0",
                                     description="Rover spawn X position")
    spawn_y = DeclareLaunchArgument("spawn_y", default_value="0.0",
                                     description="Rover spawn Y position")

    # ── 3. Spawn ExoMy — delayed 3s to let Gazebo initialise ──────────────
    spawn_robot = TimerAction(
        period=3.0,
        actions=[
            Node(
                package="gazebo_ros",
                executable="spawn_entity.py",
                name="spawn_exomy",
                output="screen",
                arguments=[
                    "-topic", "robot_description",
                    "-entity", "exomy",
                    "-x", LaunchConfiguration("spawn_x"),
                    "-y", LaunchConfiguration("spawn_y"),
                    "-z", "0.15",
                ],
            )
        ],
    )

    # ── 4. CLIP terrain node — delayed 5s to let robot spawn + camera start
    clip_node = TimerAction(
        period=5.0,
        actions=[
            Node(
                package="fm_perception",
                executable="clip_terrain_node.py",
                name="clip_terrain_node",
                output="screen",
                parameters=[{
                    "device":               "cpu",
                    "confidence_threshold": 0.40,
                    "publish_viz":          False,  # keep output clean for testing
                }],
                remappings=[
                    ("/camera/image_raw", "/exomy/camera/image_raw"),
                ],
            )
        ],
    )

    return LaunchDescription([
        spawn_x, spawn_y,
        gazebo_server,
        robot_state_publisher,
        spawn_robot,
        clip_node,
    ])
