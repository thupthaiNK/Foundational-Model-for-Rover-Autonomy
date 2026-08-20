"""
Purpose: Full DINOv2 ViT-L + terrain controller integration launch — Gazebo (headless)
         + ExoMy robot + generic terrain node loaded with DINOv2 ViT-L encoder.
         Used for multi-model Gazebo comparison (Experiment 9 §4.8):
           camera image → DINOv2 ViT-L (frozen, 1024-d CLS) → LogReg probe →
           terrain label → controller → /exomy/cmd_vel
         ViT-L is the largest DINOv2 variant (307M params); expect ~5× slower
         inference than ViT-S on CPU. Same world/zones/controller as other models.
Inputs:  None
Outputs: /terrain_classification  — DINOv2 ViT-L terrain label ("label:confidence")
         /exomy/cmd_vel           — velocity commands from terrain controller
         /terrain_viz             — annotated image
How to run:
    source /opt/ros/humble/setup.bash
    source ros2_ws/install/setup.bash
    ros2 launch simulation/launch/dinov2_vitl_controller_test.launch.py

    # Terminal 2 (after node shows "generic_terrain_node ready")
    python3 experiments/multimodel_traversability_experiment.py --model dinov2_vitl

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

    repo_root  = os.path.normpath(os.path.join(sim_dir, ".."))
    cache_dir  = os.path.join(repo_root, "experiments", "results", "feature_cache")
    feat_npy   = os.path.join(cache_dir, "dinov2_vitl_train_1000_feats.npy")
    label_npy  = os.path.join(cache_dir, "dinov2_vitl_train_1000_labels.npy")

    robot_description = xacro.process_file(urdf_path).toxml()

    spawn_x = DeclareLaunchArgument(
        "spawn_x", default_value="0.0",
        description="Rover spawn X"
    )
    spawn_y = DeclareLaunchArgument(
        "spawn_y", default_value="7.0",
        description="Rover spawn Y"
    )

    models_dir = os.path.join(sim_dir, "models")
    existing_model_path = os.environ.get("GAZEBO_MODEL_PATH", "")
    gazebo_model_path   = f"{models_dir}:{existing_model_path}" if existing_model_path else models_dir

    # ── 1. Gazebo server (headless) ────────────────────────────────────────
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

    # ── 3. Spawn ExoMy ────────────────────────────────────────────────────
    spawn_robot = TimerAction(
        period=15.0,
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

    # ── 4. DINOv2 ViT-L terrain node ─────────────────────────────────────
    # Frozen encoder (1024-d CLS, 307M params) + LogReg probe.
    # Expect ~3–5s/frame on CPU (MX330). N_READINGS stabilisation period
    # accounts for this via WAIT_STABLE_S in the experiment script.
    terrain_node = TimerAction(
        period=30.0,   # extra 8s vs ViT-S — ViT-L model load is slower
        actions=[
            Node(
                package="fm_perception",
                executable="generic_terrain_node.py",
                name="generic_terrain_node",
                output="screen",
                parameters=[{
                    "model_id":             "facebook/dinov2-large",
                    "feat_npy":             feat_npy,
                    "label_npy":            label_npy,
                    "display_name":         "DINOv2-ViT-L",
                    "device":               "cpu",
                    "confidence_threshold": 0.40,
                    "publish_viz":          True,
                    "n_shot":               1000,
                }],
                remappings=[
                    ("/camera/image_raw", "/exomy/camera/image_raw"),
                ],
            )
        ],
    )

    # ── 5. Terrain controller ─────────────────────────────────────────────
    controller_node = TimerAction(
        period=33.0,   # after ViT-L node
        actions=[
            Node(
                package="fm_perception",
                executable="terrain_controller_node.py",
                name="terrain_controller_node",
                output="screen",
                parameters=[{
                    "cmd_vel_topic":        "/exomy/cmd_vel",
                    "terrain_topic":        "/terrain_classification",
                    "confidence_threshold": 0.40,
                    "speed_soil":           0.10,
                    "speed_sand":           0.05,
                    "speed_bedrock":        0.03,
                    "max_uncertain_count":  5,
                    "publish_rate_hz":      10.0,
                    "stale_timeout_s":      3.0,
                }],
            )
        ],
    )

    return LaunchDescription([
        spawn_x, spawn_y,
        gazebo_server,
        robot_state_publisher,
        spawn_robot,
        terrain_node,
        controller_node,
    ])
