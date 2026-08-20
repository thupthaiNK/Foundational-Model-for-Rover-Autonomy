"""
Purpose: Interactive teleoperation demo — Gazebo WITH visual GUI + DINOv2 real-time
         terrain classification. User drives rover manually with keyboard teleop and
         watches terrain labels change in real-time as rover crosses zone boundaries.
         Unlike Exp 3b (static zone teleport), this tests continuous-motion classification
         demonstrating real-time inference during active manual navigation.
Inputs:  None (Gazebo GUI opens automatically via WSLg / X11)
Outputs: /terrain_classification  (std_msgs/String)  "label:confidence" — terminal echo
         /terrain_viz             (sensor_msgs/Image) annotated camera feed (rqt_image_view)
How to run:
    # Terminal 1 — build (once)
    cd ros2_ws && colcon build --packages-select fm_perception
    source /opt/ros/humble/setup.bash && source ros2_ws/install/setup.bash

    # Terminal 2 — launch Gazebo GUI + DINOv2  ← THIS FILE
    source /opt/ros/humble/setup.bash && source ros2_ws/install/setup.bash
    ros2 launch simulation/launch/interactive_demo.launch.py

    # Terminal 3 — drive rover with keyboard (WASD / arrow keys)
    source /opt/ros/humble/setup.bash
    ros2 run teleop_twist_keyboard teleop_twist_keyboard \
        --ros-args --remap cmd_vel:=/exomy/cmd_vel

    # Terminal 4 — watch terrain classification live
    ros2 topic echo /terrain_classification

    # Optional Terminal 5 — visualise annotated camera feed
    ros2 run rqt_image_view rqt_image_view /terrain_viz

Zone map (spawn_x positions for reference):
    soil    x ≈  0.0,  y = 7.0   (default spawn — safe, 0.10 m/s)
    bedrock x ≈  4.5,  y = 0.0   (hazard, 0.03 m/s)
    sand    x ≈ -4.5,  y = 0.0   (caution, 0.05 m/s)
Project: Onboard Visual Foundation Models for Mars Terrain Perception and Rover Navigation
"""

import os

import xacro
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    TimerAction,
)
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():

    # ── Paths ──────────────────────────────────────────────────────────────────
    sim_dir    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    urdf_path  = os.path.join(sim_dir, "urdf", "exomy.urdf.xacro")
    world_path = os.path.join(sim_dir, "worlds", "mars_terrain.world")
    repo_root  = os.path.normpath(os.path.join(sim_dir, ".."))
    cache_path = os.path.join(
        repo_root, "experiments", "results", "feature_cache",
        "dinov2_reg_small_train_1000shot.npz"
    )

    robot_description = xacro.process_file(urdf_path).toxml()

    models_dir          = os.path.join(sim_dir, "models")
    existing_model_path = os.environ.get("GAZEBO_MODEL_PATH", "")
    gazebo_model_path   = (
        f"{models_dir}:{existing_model_path}" if existing_model_path else models_dir
    )

    # ── Launch args ────────────────────────────────────────────────────────────
    spawn_x = DeclareLaunchArgument(
        "spawn_x", default_value="0.0",
        description="Rover spawn X  (0.0=soil, 4.5=bedrock, -4.5=sand)"
    )
    spawn_y = DeclareLaunchArgument(
        "spawn_y", default_value="7.0",
        description="Rover spawn Y"
    )

    # ── 1. Gazebo server ───────────────────────────────────────────────────────
    gazebo_server = ExecuteProcess(
        cmd=[
            "gzserver", "--verbose", world_path,
            "-s", "libgazebo_ros_init.so",
            "-s", "libgazebo_ros_factory.so",
            "-s", "libgazebo_ros_state.so",
        ],
        output="screen",
        additional_env={"GAZEBO_MODEL_PATH": gazebo_model_path},
    )

    # ── 2. Gazebo client (GUI) — delayed 2s to let server start ───────────────
    gazebo_client = TimerAction(
        period=2.0,
        actions=[
            ExecuteProcess(
                cmd=["gzclient", "--verbose"],
                output="screen",
                additional_env={"GAZEBO_MODEL_PATH": gazebo_model_path},
            )
        ],
    )

    # ── 3. Robot state publisher ───────────────────────────────────────────────
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

    # ── 4. Spawn ExoMy — delayed 3s ───────────────────────────────────────────
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

    # ── 5. DINOv2 terrain node — delayed 5s ───────────────────────────────────
    # publish_viz=True → /terrain_viz topic with label+confidence overlay
    # No terrain_controller here — user drives manually with teleop
    dinov2_node = TimerAction(
        period=5.0,
        actions=[
            Node(
                package="fm_perception",
                executable="dinov2_terrain_node.py",
                name="dinov2_terrain_node",
                output="screen",
                parameters=[{
                    "device":               "cpu",
                    "confidence_threshold": 0.40,
                    "publish_viz":          True,
                    "cache_path":           cache_path,
                    "n_shot":               1000,
                }],
                remappings=[
                    ("/camera/image_raw", "/exomy/camera/image_raw"),
                ],
            )
        ],
    )

    return LaunchDescription([
        spawn_x,
        spawn_y,
        gazebo_server,
        gazebo_client,
        robot_state_publisher,
        spawn_robot,
        dinov2_node,
    ])
