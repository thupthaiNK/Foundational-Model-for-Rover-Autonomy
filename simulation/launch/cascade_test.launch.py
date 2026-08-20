"""
Purpose: Gazebo headless launch for the CASCADE pipeline 5-zone traversability
         experiment (§4.8 / Exp 3b). Runs the full cascade pipeline:
           - CLIP fast path (every frame, ~136ms) → /terrain_classification
           - SmolVLM slow path (every 5s, ~55s/inference) → /smolvlm_terrain
           - Combined /traversability decision ("safe"|"caution"|"hazard")
         terrain_controller_node subscribes to /terrain_classification (CLIP)
         and drives the rover in real-time (CLIP is fast enough; SmolVLM refines
         the traversability assessment asynchronously in the background).
         Compare result against DINOv2-only (3/5 terrain, 4/5 traversability).
Inputs:  None (launches everything automatically)
Outputs: /terrain_classification — CLIP label per frame ("label:confidence")
         /smolvlm_terrain        — SmolVLM VQA label every ~5s ("label:smolvlm")
         /traversability         — "safe" | "caution" | "hazard"
         /scene_description      — SmolVLM text answer
         /exomy/cmd_vel          — controller velocity (CLIP-driven)
         /measurement_mode       — Bool freeze signal (used by experiment)
How to run:
    # Build and source (once):
    cd ros2_ws && colcon build --packages-select fm_perception
    source /opt/ros/humble/setup.bash && source ros2_ws/install/setup.bash

    # Launch (Terminal 2):
    ros2 launch simulation/launch/cascade_test.launch.py

    # Wait ~45s for models to load, then run experiment (Terminal 3):
    python3 experiments/cascade_traversability_experiment.py

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

    # ── 1. Gazebo server (headless) ─────────────────────────────────────────
    # libgazebo_ros_factory.so: delete_entity + spawn_entity (teleport)
    # libgazebo_ros_state.so: /gazebo/model_states
    gazebo = ExecuteProcess(
        cmd=[
            "gzserver", "--verbose", world_path,
            "-s", "libgazebo_ros_init.so",
            "-s", "libgazebo_ros_factory.so",
        ],
        output="screen",
        additional_env={"GAZEBO_MODEL_PATH": gazebo_model_path},
    )

    # ── 2. Robot state publisher ────────────────────────────────────────────
    rsp = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="screen",
        parameters=[{"robot_description": robot_description, "use_sim_time": True}],
    )

    # ── 2b. Joint state publisher ───────────────────────────────────────────
    joint_state_publisher = Node(
        package="joint_state_publisher",
        executable="joint_state_publisher",
        name="joint_state_publisher",
        output="screen",
        parameters=[{"use_sim_time": True}],
    )

    # ── 3. Spawn ExoMy — delayed 35s for 21 mesh rocks to load ────────────
    spawn = TimerAction(
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

    # ── 4. Cascade pipeline — starts at 42s ────────────────────────────────
    # CLIP fast path loads quickly (~2s). SmolVLM slow path takes ~60s to load
    # on first run (model download + tokenizer). After loading, CLIP publishes
    # every frame immediately; SmolVLM runs in background thread every 5s.
    cascade = TimerAction(
        period=42.0,
        actions=[
            Node(
                package="fm_perception",
                executable="cascade_pipeline.py",
                name="cascade_pipeline",
                output="screen",
                remappings=[("/camera/image_raw", "/exomy/camera/image_raw")],
                parameters=[{
                    "device":               "cpu",
                    "clip_threshold":       0.40,
                    "smolvlm_interval_sec": 5.0,
                    "smolvlm_model_id":     "HuggingFaceTB/SmolVLM-256M-Instruct",
                    "publish_viz":          True,
                }],
            )
        ],
    )

    # ── 5. Terrain controller — starts at 45s ──────────────────────────────
    # Subscribes to /terrain_classification (CLIP output, "label:confidence").
    # Publishes /exomy/cmd_vel based on terrain label:
    #   soil     → 0.10 m/s  (SAFE)
    #   sand     → 0.05 m/s  (CAUTION)
    #   bedrock  → 0.03 m/s  (HAZARD — conservative)
    #   uncertain/big_rock → 0.00 m/s  (STOP)
    # Respects /measurement_mode freeze signal from experiment script.
    terrain_controller = TimerAction(
        period=45.0,
        actions=[
            Node(
                package="fm_perception",
                executable="terrain_controller_node.py",
                name="terrain_controller_node",
                output="screen",
                parameters=[{
                    "use_sim_time":        True,
                    "confidence_threshold": 0.35,
                    "safe_speed":          0.10,
                    "caution_speed":       0.05,
                    "hazard_speed":        0.03,
                    "stop_speed":          0.00,
                }],
            )
        ],
    )

    return LaunchDescription([
        spawn_x, spawn_y,
        gazebo,
        rsp,
        joint_state_publisher,
        spawn,
        cascade,
        terrain_controller,
    ])
