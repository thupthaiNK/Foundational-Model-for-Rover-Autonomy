"""
Purpose: Full DINOv2 + terrain controller integration launch — Gazebo (headless)
         + ExoMy robot + DINOv2 terrain node + terrain controller node.
         Tests the complete Layer 2–3 pipeline:
           camera image → DINOv2 ViT-S (90.24%) → terrain label →
           controller → /exomy/cmd_vel velocity commands
         Replaces integration_test.launch.py (CLIP) with DINOv2 + reactive control.
Inputs:  None (launches everything automatically)
         Optional args: spawn_x, spawn_y (rover start position for zone testing)
Outputs: /terrain_classification  — DINOv2 terrain label ("label:confidence")
         /exomy/cmd_vel           — velocity commands from terrain controller
         /terrain_viz             — annotated image (disabled by default)
How to run:
    # Terminal 1 — build and source first
    cd ros2_ws && colcon build --packages-select fm_perception
    source /opt/ros/humble/setup.bash
    source ros2_ws/install/setup.bash

    # Terminal 2 — launch DINOv2 + controller
    source /opt/ros/humble/setup.bash
    source ros2_ws/install/setup.bash
    ros2 launch simulation/launch/dinov2_controller_test.launch.py

    # With specific zone position:
    ros2 launch simulation/launch/dinov2_controller_test.launch.py spawn_x:=4.5 spawn_y:=0.0

    # Terminal 3 — monitor outputs
    ros2 topic echo /terrain_classification
    ros2 topic echo /exomy/cmd_vel
    ros2 topic hz /terrain_classification

    # Terminal 4 — run full zone traversal experiment
    python3 experiments/dinov2_traversability_experiment.py

Project: Onboard Visual Foundation Models for Mars Terrain Perception and Rover Navigation
"""

import datetime
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

    # DINOv2 feature cache — absolute path from repo root; overridable via cache_path arg
    repo_root        = os.path.normpath(os.path.join(sim_dir, ".."))
    _default_cache   = os.path.join(
        repo_root, "experiments", "results", "feature_cache",
        "dinov2_reg_small_train_1000shot.npz"
    )
    _configB_cache   = os.path.join(
        repo_root, "experiments", "results", "feature_cache",
        "dinov2_reg_small_configB_train.npz"
    )

    robot_description = xacro.process_file(urdf_path).toxml()

    # ── Spawn position args (for zone-specific testing) ───────────────────
    cache_path_arg = DeclareLaunchArgument(
        "cache_path", default_value=_default_cache,
        description="Path to LogReg feature cache .npz. Use _configB_cache for augmented big_rock probe."
    )
    spawn_x = DeclareLaunchArgument(
        "spawn_x", default_value="-7.5",
        description="Rover spawn X — default centre of soil zone Q2 (-7.5)"
    )
    spawn_y = DeclareLaunchArgument(
        "spawn_y", default_value="6.0",
        description="Rover spawn Y — default centre of soil zone Q2 (6.0)"
    )
    class_weight_balanced = DeclareLaunchArgument(
        "class_weight_balanced", default_value="false",
        description="Use class_weight='balanced' for the DINOv2 logistic-regression probe"
    )
    enable_obstacle_gate = DeclareLaunchArgument(
        "enable_obstacle_gate", default_value="false",
        description="Enable conservative image obstacle gate for dark edge-rich rock hazards"
    )
    obstacle_gate_label = DeclareLaunchArgument(
        "obstacle_gate_label", default_value="uncertain",
        description="Label published when obstacle gate triggers: uncertain or big_rock"
    )
    obstacle_gate_min_upper_edge = DeclareLaunchArgument(
        "obstacle_gate_min_upper_edge", default_value="4.6",
        description="Minimum upper-frame edge score for the obstacle gate"
    )
    obstacle_gate_max_p90 = DeclareLaunchArgument(
        "obstacle_gate_max_p90", default_value="88.0",
        description="Maximum grayscale p90 for the obstacle gate"
    )
    obstacle_gate_min_darkness = DeclareLaunchArgument(
        "obstacle_gate_min_darkness", default_value="40.0",
        description="Mean brightness <= this also triggers gate (catches dark boulder_zone)"
    )
    use_continuous_score = DeclareLaunchArgument(
        "use_continuous_score", default_value="false",
        description="Opt-in continuous-speed mode (thesis Ch5 SS5.6.2, Ch4 SS4.8.16): "
                     "v = v_max * (1 - T_score) instead of discrete POLICY speed steps. "
                     "Default false preserves every previously reported result using this "
                     "launch file (already unit-tested in terrain_controller_node.py's own "
                     "tests, but never yet exercised live in Gazebo -- this arg exists to "
                     "close that verification gap, §5.6.2)."
    )

    # GAZEBO_MODEL_PATH — include our simulation/models so mars_rock/rock_large are found
    models_dir = os.path.join(sim_dir, "models")
    existing_model_path = os.environ.get("GAZEBO_MODEL_PATH", "")
    gazebo_model_path   = f"{models_dir}:{existing_model_path}" if existing_model_path else models_dir

    # ── 1. Gazebo server — headless (no GUI, WSL2 compatible) ─────────────
    # libgazebo_ros_state.so is intentionally omitted here. This repo already
    # documents that it is not compatible with Gazebo Classic 11.10.2 in this
    # environment, and the experiment teleports via delete+respawn using the
    # factory plugin instead of the state plugin.
    gazebo_server = ExecuteProcess(
        cmd=[
            "gzserver", "--verbose", world_path,
            "-s", "libgazebo_ros_init.so",
            "-s", "libgazebo_ros_factory.so",
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

    # ── 2b. Joint state publisher — needed for fixed joints in RViz TF tree ──
    joint_state_publisher = Node(
        package="joint_state_publisher",
        executable="joint_state_publisher",
        name="joint_state_publisher",
        output="screen",
        parameters=[{"use_sim_time": True}],
    )

    # ── 3. Spawn ExoMy — delayed to let Gazebo load all mesh rocks ──────────
    # 35s needed: 21 mars_rock.dae meshes generate collision geometry serially.
    # With only 3 meshes (old world) 15s was enough; 21 meshes need ~35s.
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

    # ── 4. DINOv2 terrain node — delayed until after robot spawn starts ───
    # Uses frozen DINOv2 ViT-S/14 (384-d CLS) + LogReg probe (90.24% AI4Mars)
    # LogReg is trained at startup from feature cache (~1.2s, deterministic)
    dinov2_node = TimerAction(
        period=42.0,  # spawn at 35s + 7s model load buffer
        actions=[
            Node(
                package="fm_perception",
                executable="dinov2_terrain_node.py",
                name="dinov2_terrain_node",
                output="screen",
                parameters=[{
                    "device":               "cpu",
                    "confidence_threshold": 0.40,   # lower for Gazebo synthetic textures (domain gap)
                    "publish_viz":          True,    # annotated camera feed on /terrain_viz
                    "cache_path":           LaunchConfiguration("cache_path"),
                    "n_shot":               1000,
                    "class_weight_balanced": LaunchConfiguration("class_weight_balanced"),
                    "enable_obstacle_gate": LaunchConfiguration("enable_obstacle_gate"),
                    "obstacle_gate_label":  LaunchConfiguration("obstacle_gate_label"),
                    "obstacle_gate_min_upper_edge": LaunchConfiguration("obstacle_gate_min_upper_edge"),
                    "obstacle_gate_max_p90": LaunchConfiguration("obstacle_gate_max_p90"),
                    "obstacle_gate_max_mean": 66.0,
                    "obstacle_gate_min_darkness": LaunchConfiguration("obstacle_gate_min_darkness"),
                }],
                remappings=[
                    ("/camera/image_raw", "/exomy/camera/image_raw"),
                ],
            )
        ],
    )

    # ── 5. Terrain controller — delayed until DINOv2 has started loading ──
    # Subscribes /terrain_classification → publishes /exomy/cmd_vel
    # Policy: soil 0.10 m/s | sand 0.05 m/s | bedrock 0.03 m/s | uncertain STOP
    controller_node = TimerAction(
        period=45.0,  # after DINOv2 starts loading
        actions=[
            Node(
                package="fm_perception",
                executable="terrain_controller_node.py",
                name="terrain_controller_node",
                output="screen",
                parameters=[{
                    "cmd_vel_topic":        "/exomy/cmd_vel",
                    "terrain_topic":        "/terrain_classification",
                    "confidence_threshold": 0.40,   # match DINOv2 node threshold for Gazebo
                    "speed_soil":           0.10,
                    "speed_sand":           0.05,
                    "speed_bedrock":        0.03,
                    "max_uncertain_count":  5,
                    "publish_rate_hz":      10.0,
                    "stale_timeout_s":      3.0,
                    "use_continuous_score": LaunchConfiguration("use_continuous_score"),
                }],
            )
        ],
    )

    # ── 6. Safety watchdog — delayed until after controller is live ────────
    # Monitors /terrain_classification fail-rate; publishes /e_stop (Bool)
    # + /e_stop_reason (String) and sends zero cmd_vel when E-stop is latched.
    watchdog_node = TimerAction(
        period=48.0,
        actions=[
            Node(
                package="fm_perception",
                executable="safety_watchdog_node.py",
                name="safety_watchdog_node",
                output="screen",
                parameters=[{
                    "fail_rate_threshold": 0.5,
                    "window_s":            10.0,
                    "stale_timeout_s":     3.0,
                    "use_sim_time":        True,
                }],
            )
        ],
    )

    # ── 7. ROS2 bag recording — starts early to capture full experiment ────
    # Records all experiment-relevant topics for reproducibility evidence.
    bag_name = os.path.join(
        repo_root, "bags",
        f"exp8_dinov2_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    bag_record = TimerAction(
        period=5.0,
        actions=[
            ExecuteProcess(
                cmd=[
                    "ros2", "bag", "record",
                    "/exomy/camera/image_raw",
                    "/terrain_classification",
                    "/terrain_class_probs",
                    "/inference_latency_ms",
                    "/exomy/cmd_vel",
                    "/measurement_mode",
                    "/e_stop",
                    "/e_stop_reason",
                    "-o", bag_name,
                ],
                output="screen",
            )
        ],
    )

    return LaunchDescription([
        cache_path_arg,
        spawn_x, spawn_y,
        class_weight_balanced,
        enable_obstacle_gate,
        obstacle_gate_label,
        obstacle_gate_min_upper_edge,
        obstacle_gate_max_p90,
        obstacle_gate_min_darkness,
        use_continuous_score,
        gazebo_server,
        robot_state_publisher,
        joint_state_publisher,
        spawn_robot,
        dinov2_node,
        controller_node,
        watchdog_node,
        bag_record,
    ])
