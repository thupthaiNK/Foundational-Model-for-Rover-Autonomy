"""
Purpose: Full visualisation launch — everything in dinov2_controller_test.launch.py
         PLUS Gazebo GUI, RViz2, and zone ground-truth markers. Designed for the
         §4.8 thesis figure showing the complete Layer 2–3 pipeline live:
           Gazebo world → ExoMy camera → DINOv2 terrain node → controller →
           /terrain_classification → RViz2 (robot model + zone slabs + camera feeds)
         Run the 5-zone experiment (run_exp.sh) from a second terminal while this
         launch is active to record the full session.
Inputs:  None (all args optional)
         Passes all gate args through to dinov2_controller_test.launch.py.
Outputs: Gazebo GUI window (gzclient)
         RViz2 window with:
           - ExoMy robot model + TF
           - /exomy/camera/image_raw  (raw camera)
           - /terrain_viz             (DINOv2 annotated)
           - /zone_markers            (colour-coded ground-truth zone slabs)
How to run:
    # Terminal 1 — launch everything (GUI + RViz2 + DINOv2 + controller)
    source /opt/ros/humble/setup.bash
    source ros2_ws/install/setup.bash
    ros2 launch simulation/launch/dinov2_rviz2.launch.py

    # Terminal 2 — run 5-zone experiment
    python3 experiments/dinov2_traversability_experiment.py

    # WSL2 note: requires WSLg (Windows 11) or an X11 server (VcXsrv) for GUI windows.
Project: Onboard Visual Foundation Models for Mars Terrain Perception and Rover Navigation
"""

import os

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    IncludeLaunchDescription,
    TimerAction,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():

    sim_dir   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    repo_root = os.path.normpath(os.path.join(sim_dir, ".."))
    rviz_cfg  = os.path.join(sim_dir, "rviz", "dinov2_traversability.rviz")

    models_dir          = os.path.join(sim_dir, "models")
    existing_model_path = os.environ.get("GAZEBO_MODEL_PATH", "")
    gazebo_model_path   = (
        f"{models_dir}:{existing_model_path}" if existing_model_path else models_dir
    )

    # ── Passthrough args → forwarded to dinov2_controller_test.launch.py ──────
    spawn_x = DeclareLaunchArgument("spawn_x", default_value="-7.5")
    spawn_y = DeclareLaunchArgument("spawn_y", default_value="6.0")
    enable_obstacle_gate = DeclareLaunchArgument(
        "enable_obstacle_gate", default_value="true"
    )
    obstacle_gate_label = DeclareLaunchArgument(
        "obstacle_gate_label", default_value="uncertain"
    )
    obstacle_gate_min_upper_edge = DeclareLaunchArgument(
        "obstacle_gate_min_upper_edge", default_value="4.8"
    )
    obstacle_gate_min_darkness = DeclareLaunchArgument(
        "obstacle_gate_min_darkness", default_value="40.0"
    )
    obstacle_gate_max_p90 = DeclareLaunchArgument(
        "obstacle_gate_max_p90", default_value="80.0"
    )

    # ── 1. Include base launch (Gazebo server + all nodes) ────────────────────
    base_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(sim_dir, "launch", "dinov2_controller_test.launch.py")
        ),
        launch_arguments={
            "spawn_x":                    LaunchConfiguration("spawn_x"),
            "spawn_y":                    LaunchConfiguration("spawn_y"),
            "enable_obstacle_gate":       LaunchConfiguration("enable_obstacle_gate"),
            "obstacle_gate_label":        LaunchConfiguration("obstacle_gate_label"),
            "obstacle_gate_min_upper_edge": LaunchConfiguration("obstacle_gate_min_upper_edge"),
            "obstacle_gate_min_darkness": LaunchConfiguration("obstacle_gate_min_darkness"),
            "obstacle_gate_max_p90":      LaunchConfiguration("obstacle_gate_max_p90"),
        }.items(),
    )

    # ── 2. Gazebo client (GUI) — delayed 3s to let server start ───────────────
    gazebo_client = TimerAction(
        period=3.0,
        actions=[
            ExecuteProcess(
                cmd=["gzclient", "--verbose"],
                output="screen",
                additional_env={"GAZEBO_MODEL_PATH": gazebo_model_path},
            )
        ],
    )

    # ── 3. Zone ground-truth markers — latched, starts immediately ────────────
    markers_script = os.path.join(repo_root, "experiments", "publish_zone_markers.py")
    zone_markers = ExecuteProcess(
        cmd=["python3", markers_script],
        output="screen",
    )

    # ── 4. RViz2 — delayed 50s until robot model + TF are available ───────────
    # (robot spawns at 35s, TF becomes consistent ~50s)
    rviz2 = TimerAction(
        period=50.0,
        actions=[
            Node(
                package="rviz2",
                executable="rviz2",
                name="rviz2",
                output="screen",
                arguments=["-d", rviz_cfg],
                parameters=[{"use_sim_time": True}],
            )
        ],
    )

    return LaunchDescription([
        spawn_x, spawn_y,
        enable_obstacle_gate,
        obstacle_gate_label,
        obstacle_gate_min_upper_edge,
        obstacle_gate_min_darkness,
        obstacle_gate_max_p90,
        base_launch,
        gazebo_client,
        zone_markers,
        rviz2,
    ])
