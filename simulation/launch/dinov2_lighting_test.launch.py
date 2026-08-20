"""
Purpose: Gazebo headless launch for DINOv2 lighting robustness experiment (Step 7a).
         Loads one of three lighting variants of the 5-zone Mars world:
           - dawn: low sun from east, orange-red, long shadows westward
           - noon: overhead sun, bright white, minimal shadows
           - dusk: low sun from west, deep red, long shadows eastward
         Uses the same DINOv2 terrain node and terrain controller as the baseline
         dinov2_controller_test.launch.py. The only change is the world file.
Inputs:  variant:=dawn|noon|dusk  (default: noon)
Outputs: Same as dinov2_controller_test.launch.py:
           /terrain_classification, /terrain_class_probs, /inference_latency_ms,
           /terrain_viz, /exomy/cmd_vel
How to run:
    # Build and source:
    cd ros2_ws && colcon build --packages-select fm_perception
    source /opt/ros/humble/setup.bash && source ros2_ws/install/setup.bash

    # Launch (choose variant):
    ros2 launch simulation/launch/dinov2_lighting_test.launch.py variant:=dawn
    ros2 launch simulation/launch/dinov2_lighting_test.launch.py variant:=noon
    ros2 launch simulation/launch/dinov2_lighting_test.launch.py variant:=dusk

    # Run experiment in second terminal:
    python3 experiments/lighting_traversability_experiment.py --variant dawn
    python3 experiments/lighting_traversability_experiment.py --variant noon
    python3 experiments/lighting_traversability_experiment.py --variant dusk

Project: Onboard Visual Foundation Models for Mars Terrain Perception and Rover Navigation
"""

import os
import xacro
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, TimerAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


VALID_VARIANTS = ["dawn", "noon", "dusk"]


def generate_launch_description():

    # ── Paths ───────────────────────────────────────────────────────────────
    sim_dir   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    urdf_path = os.path.join(sim_dir, "urdf", "exomy.urdf.xacro")

    robot_description = xacro.process_file(urdf_path).toxml()

    models_dir          = os.path.join(sim_dir, "models")
    existing_model_path = os.environ.get("GAZEBO_MODEL_PATH", "")
    gazebo_model_path   = (
        f"{models_dir}:{existing_model_path}" if existing_model_path else models_dir
    )

    # Build world path map at launch-description-generation time (Python context)
    world_map = {
        v: os.path.join(sim_dir, "worlds", f"mars_terrain_{v}.world")
        for v in VALID_VARIANTS
    }

    # ── Launch argument ─────────────────────────────────────────────────────
    variant_arg = DeclareLaunchArgument(
        "variant",
        default_value="noon",
        description="Lighting variant: dawn | noon | dusk",
        choices=VALID_VARIANTS,
    )
    class_weight_balanced = DeclareLaunchArgument(
        "class_weight_balanced",
        default_value="false",
        description="Use class_weight='balanced' for the DINOv2 logistic-regression probe",
    )
    enable_obstacle_gate = DeclareLaunchArgument(
        "enable_obstacle_gate",
        default_value="true",
        description="Enable conservative image obstacle gate for dark edge-rich rock hazards",
    )
    obstacle_gate_label = DeclareLaunchArgument(
        "obstacle_gate_label",
        default_value="uncertain",
        description="Label published when obstacle gate triggers: uncertain or big_rock",
    )
    obstacle_gate_min_upper_edge = DeclareLaunchArgument(
        "obstacle_gate_min_upper_edge",
        default_value="4.6",
        description="Minimum upper-frame edge score for the obstacle gate",
    )
    obstacle_gate_max_p90 = DeclareLaunchArgument(
        "obstacle_gate_max_p90",
        default_value="88.0",
        description="Maximum grayscale p90 for the obstacle gate",
    )
    obstacle_gate_min_darkness = DeclareLaunchArgument(
        "obstacle_gate_min_darkness",
        default_value="40.0",
        description="Mean brightness <= this also triggers gate (catches dark boulder_zone)",
    )

    # Read variant at Python level for world file selection.
    # LaunchConfiguration substitutions can't be used directly as dict keys,
    # so we generate one Gazebo server per variant and use OpaqueFunction.
    # Simpler: just resolve at import time via an OpaqueFunction.
    from launch.actions import OpaqueFunction

    def _launch_with_variant(context, *args, **kwargs):
        variant    = context.launch_configurations.get("variant", "noon")
        world_path = world_map.get(variant, world_map["noon"])

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
                        "-x", "-7.5", "-y", "6.0", "-z", "0.15",
                    ],
                )
            ],
        )

        dinov2_node = TimerAction(
            period=42.0,
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
                        "class_weight_balanced": LaunchConfiguration("class_weight_balanced"),
                        "enable_obstacle_gate": LaunchConfiguration("enable_obstacle_gate"),
                        "obstacle_gate_label":  LaunchConfiguration("obstacle_gate_label"),
                        "obstacle_gate_min_upper_edge": LaunchConfiguration("obstacle_gate_min_upper_edge"),
                        "obstacle_gate_max_p90": LaunchConfiguration("obstacle_gate_max_p90"),
                        "obstacle_gate_max_mean": 66.0,
                        "obstacle_gate_min_darkness": LaunchConfiguration("obstacle_gate_min_darkness"),
                        "use_sim_time":         True,
                    }],
                    remappings=[("/camera/image_raw", "/exomy/camera/image_raw")],
                )
            ],
        )

        controller = TimerAction(
            period=45.0,
            actions=[
                Node(
                    package="fm_perception",
                    executable="terrain_controller_node.py",
                    name="terrain_controller_node",
                    output="screen",
                    parameters=[{
                        "use_sim_time":         True,
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

        return [gazebo_server, rsp, jsp, spawn, dinov2_node, controller]

    return LaunchDescription([
        variant_arg,
        class_weight_balanced,
        enable_obstacle_gate,
        obstacle_gate_label,
        obstacle_gate_min_upper_edge,
        obstacle_gate_max_p90,
        obstacle_gate_min_darkness,
        OpaqueFunction(function=_launch_with_variant),
    ])
