"""
Purpose: Gazebo launch for the HiRISE qualitative terrain demo (Step 7c).
         Loads the real Mars topography mesh (Nili Patera HiRISE DTM converted
         to DAE) and spawns ExoMy with DINOv2 terrain classification running.
         This is a VISUAL DEMO — no quantitative 5-zone benchmark zones.
         Use for thesis figure: rover placed in realistic Mars topography,
         DINOv2 labelling the visible terrain in real-time.
Inputs:  simulation/models/hirise_terrain/meshes/*.dae  (must be downloaded first)
         See: simulation/models/hirise_terrain/meshes/README.md
Outputs: /terrain_classification — DINOv2 terrain label
         /terrain_viz            — annotated camera image
How to run:
    # Download HiRISE mesh first (~100MB) — see README.md in meshes/ dir.
    # Then:
    source /opt/ros/humble/setup.bash && source ros2_ws/install/setup.bash
    ros2 launch simulation/launch/dinov2_hirise_qualitative.launch.py

Project: Onboard Visual Foundation Models for Mars Terrain Perception and Rover Navigation
"""

import os
import xacro
from launch import LaunchDescription
from launch.actions import ExecuteProcess, TimerAction
from launch_ros.actions import Node


def generate_launch_description():

    sim_dir    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    urdf_path  = os.path.join(sim_dir, "urdf", "exomy.urdf.xacro")
    world_path = os.path.join(sim_dir, "worlds", "mars_hirise_qualitative.world")

    robot_description = xacro.process_file(urdf_path).toxml()

    models_dir          = os.path.join(sim_dir, "models")
    existing_model_path = os.environ.get("GAZEBO_MODEL_PATH", "")
    gazebo_model_path   = (
        f"{models_dir}:{existing_model_path}" if existing_model_path else models_dir
    )

    gazebo_server = ExecuteProcess(
        cmd=[
            "gzserver", "--verbose", world_path,
            "-s", "libgazebo_ros_init.so",
            "-s", "libgazebo_ros_factory.so",
        ],
        output="screen",
        additional_env={"GAZEBO_MODEL_PATH": gazebo_model_path},
    )

    rsp = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="screen",
        parameters=[{"robot_description": robot_description, "use_sim_time": True}],
    )

    jsp = Node(
        package="joint_state_publisher",
        executable="joint_state_publisher",
        name="joint_state_publisher",
        output="screen",
        parameters=[{"use_sim_time": True}],
    )

    # HiRISE mesh loads slowly — delay spawn by 60s (mesh collision may take time)
    spawn = TimerAction(
        period=60.0,
        actions=[
            Node(
                package="gazebo_ros",
                executable="spawn_entity.py",
                name="spawn_exomy",
                output="screen",
                arguments=[
                    "-topic", "robot_description",
                    "-entity", "exomy",
                    "-x", "0.0", "-y", "0.0", "-z", "3.0",
                ],
            )
        ],
    )

    dinov2_node = TimerAction(
        period=65.0,
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
                    "cache_path":           "experiments/results/feature_cache/"
                                            "dinov2_reg_small_train_1000shot.npz",
                    "n_shot":               1000,
                    "use_sim_time":         True,
                }],
                remappings=[("/camera/image_raw", "/exomy/camera/image_raw")],
            )
        ],
    )

    return LaunchDescription([
        gazebo_server, rsp, jsp, spawn, dinov2_node,
    ])
