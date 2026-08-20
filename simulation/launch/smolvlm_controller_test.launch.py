"""
Purpose: Gazebo headless launch for the SmolVLM-256M-Instruct standalone 5-zone
         traversability experiment (§4.8 / Exp 3a comparison). Runs SmolVLM as
         the sole terrain classifier — no terrain_controller_node, because SmolVLM
         inference takes ~55s/image on CPU, which is too slow for reactive rover
         control (this is itself a thesis finding: VLM direct inference on CPU is
         not viable for real-time autonomous traversal).
         Compare results against DINOv2 (3/5 terrain) to quantify the accuracy vs.
         inference-speed trade-off between VLM and frozen-encoder+probe architectures.
Inputs:  None (all args optional)
Outputs: /smolvlm_terrain (std_msgs/String)  "label:smolvlm" — terrain classification
         /traversability  (std_msgs/String)  "safe"|"caution"|"hazard" — VQA safety
         /scene_description (std_msgs/String) — raw SmolVLM text answers
How to run:
    # Terminal 1 — build and source first
    cd ros2_ws && colcon build --packages-select fm_perception
    source /opt/ros/humble/setup.bash && source ros2_ws/install/setup.bash

    # Terminal 2 — launch (takes ~35s for mesh rocks + ~60s for SmolVLM to load)
    source /opt/ros/humble/setup.bash && source ros2_ws/install/setup.bash
    ros2 launch simulation/launch/smolvlm_controller_test.launch.py

    # Terminal 3 — wait until "smolvlm_terrain_node ready", then:
    python3 experiments/smolvlm_traversability_experiment.py

Project: Onboard Visual Foundation Models for Mars Terrain Perception and Rover Navigation
"""

import os
import xacro
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, TimerAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():

    # ── Paths ───────────────────────────────────────────────────────────────
    sim_dir    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    urdf_path  = os.path.join(sim_dir, "urdf", "exomy.urdf.xacro")
    world_path = os.path.join(sim_dir, "worlds", "mars_terrain.world")

    robot_description = xacro.process_file(urdf_path).toxml()

    models_dir          = os.path.join(sim_dir, "models")
    existing_model_path = os.environ.get("GAZEBO_MODEL_PATH", "")
    gazebo_model_path   = (
        f"{models_dir}:{existing_model_path}" if existing_model_path else models_dir
    )

    # ── Spawn position args ─────────────────────────────────────────────────
    spawn_x = DeclareLaunchArgument(
        "spawn_x", default_value="-7.5",
        description="Rover spawn X — default centre of soil zone Q2"
    )
    spawn_y = DeclareLaunchArgument(
        "spawn_y", default_value="6.0",
        description="Rover spawn Y — default centre of soil zone Q2"
    )
    interval_sec = DeclareLaunchArgument(
        "interval_sec", default_value="5.0",
        description="SmolVLM inference interval in seconds (timer-based slow path)"
    )

    # ── 1. Gazebo server (headless) ─────────────────────────────────────────
    # libgazebo_ros_factory.so required for delete_entity + spawn_entity services
    # used by smolvlm_traversability_experiment.py to teleport between zones.
    gazebo_server = ExecuteProcess(
        cmd=[
            "gzserver", "--verbose", world_path,
            "-s", "libgazebo_ros_init.so",
            "-s", "libgazebo_ros_factory.so",
        ],
        output="screen",
        additional_env={"GAZEBO_MODEL_PATH": gazebo_model_path},
    )

    # ── 2. Robot state publisher ────────────────────────────────────────────
    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="screen",
        parameters=[{"robot_description": robot_description, "use_sim_time": True}],
    )

    # ── 2b. Joint state publisher (required for RViz2 TF) ──────────────────
    joint_state_publisher = Node(
        package="joint_state_publisher",
        executable="joint_state_publisher",
        name="joint_state_publisher",
        output="screen",
        parameters=[{"use_sim_time": True}],
    )

    # ── 3. Spawn ExoMy — delayed 35s for 21 mesh rocks to load ────────────
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
                    "-x", LaunchConfiguration("spawn_x"),
                    "-y", LaunchConfiguration("spawn_y"),
                    "-z", "0.15",
                ],
            )
        ],
    )

    # ── 4. SmolVLM terrain node — delayed 42s ──────────────────────────────
    # Loads SmolVLM-256M-Instruct (~2GB RAM, ~60s load time on first run).
    # No CLIP running alongside → only timer-based inference (every interval_sec).
    # Subscribes to /camera/image_raw (remapped to /exomy/camera/image_raw).
    # Publishes /smolvlm_terrain ("label:smolvlm"), /traversability, /scene_description.
    #
    # NOTE: terrain_controller_node is intentionally ABSENT. SmolVLM inference
    # at ~55s/image is too slow for reactive closed-loop control. This is a
    # documented thesis finding (§4.8 / §5.X): VLM direct inference on CPU is
    # not viable for real-time autonomous traversal without hardware acceleration.
    smolvlm_node = TimerAction(
        period=42.0,
        actions=[
            Node(
                package="fm_perception",
                executable="smolvlm_terrain_node.py",
                name="smolvlm_terrain_node",
                output="screen",
                parameters=[{
                    "device":       "cpu",
                    "interval_sec": LaunchConfiguration("interval_sec"),
                    "model_id":     "HuggingFaceTB/SmolVLM-256M-Instruct",
                }],
                remappings=[
                    ("/camera/image_raw", "/exomy/camera/image_raw"),
                ],
            )
        ],
    )

    return LaunchDescription([
        spawn_x, spawn_y, interval_sec,
        gazebo_server,
        robot_state_publisher,
        joint_state_publisher,
        spawn_robot,
        smolvlm_node,
    ])
