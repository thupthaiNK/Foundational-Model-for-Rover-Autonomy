"""
Purpose: Reactive-exploration integration launch — Gazebo (headless) + ExoMy
         (with simulated LiDAR) + DINOv2 terrain node + terrain controller +
         LiDAR proximity guard + reactive_explorer_node (bug-algorithm
         hazard recovery) + safety watchdog, PLUS observability: a ros2 bag
         recording every relevant topic and (optionally) RViz2 / Foxglove
         for live viewing.
         This is a SEPARATE launch file from dinov2_controller_test.launch.py
         on purpose -- that file backs already-written-up results (D1,
         Exp5b, the 3/5-terrain/5/5-safety Gazebo baseline) and must not
         change behavior. reactive_explorer_node only exists here.
         Pipeline:
           camera -> DINOv2 terrain label -\
           /scan  -> LiDAR proximity stop  -+-> terrain_controller_node -> cmd_vel
                                             \-> reactive_explorer_node (takes
                                                 over cmd_vel via the
                                                 /reactive_explorer/active
                                                 handshake whenever hazard-stopped
                                                 longer than stuck_dwell_s)
Inputs:  None (launches everything automatically).
         Optional args: spawn_x, spawn_y, use_rviz, use_foxglove
Outputs: /terrain_classification, /exomy/cmd_vel, /terrain_controller/stopped,
         /lidar_proximity_stop, /scan, /reactive_explorer/active,
         /reactive_explorer/failsafe
         A ros2 bag under bags/reactive_exploration_<timestamp>/
How to run:
    # Terminal 1 — build and source first
    cd ros2_ws && colcon build --packages-select fm_perception
    source /opt/ros/humble/setup.bash
    source ros2_ws/install/setup.bash

    # Terminal 2 — launch everything
    source /opt/ros/humble/setup.bash
    source ros2_ws/install/setup.bash
    ros2 launch simulation/launch/reactive_exploration_test.launch.py

    # Disable Foxglove if ros-humble-foxglove-bridge isn't installed yet:
    ros2 launch simulation/launch/reactive_exploration_test.launch.py use_foxglove:=false

    # Foxglove Studio: connect to ws://localhost:8765
    # RViz2 fallback: opens automatically (use_rviz:=false to disable)
Project: Onboard Visual Foundation Models for Mars Terrain Perception and Rover Navigation
"""

import datetime
import os
import xacro
from launch import LaunchDescription
from launch.actions import ExecuteProcess, TimerAction, DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():

    # ── Paths ──────────────────────────────────────────────────────────────
    sim_dir    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    urdf_path  = os.path.join(sim_dir, "urdf", "exomy.urdf.xacro")
    world_path = os.path.join(sim_dir, "worlds", "mars_terrain.world")
    rviz_cfg   = os.path.join(sim_dir, "rviz", "reactive_exploration.rviz")

    repo_root      = os.path.normpath(os.path.join(sim_dir, ".."))
    _default_cache = os.path.join(
        repo_root, "experiments", "results", "feature_cache",
        "dinov2_reg_small_train_1000shot.npz"
    )

    robot_description = xacro.process_file(urdf_path).toxml()

    # ── Args ──────────────────────────────────────────────────────────────
    cache_path_arg = DeclareLaunchArgument(
        "cache_path", default_value=_default_cache,
        description="Path to LogReg feature cache .npz"
    )
    spawn_x = DeclareLaunchArgument(
        "spawn_x", default_value="-7.5",
        description="Rover spawn X — default centre of soil zone Q2 (-7.5)"
    )
    spawn_y = DeclareLaunchArgument(
        "spawn_y", default_value="6.0",
        description="Rover spawn Y — default centre of soil zone Q2 (6.0)"
    )
    use_rviz = DeclareLaunchArgument(
        "use_rviz", default_value="true",
        description="Open RViz2 with camera/terrain_viz/LiDAR/odom (guaranteed-working fallback viewer)"
    )
    use_foxglove = DeclareLaunchArgument(
        "use_foxglove", default_value="true",
        description="Start foxglove_bridge on port 8765 (requires ros-humble-foxglove-bridge installed)"
    )

    # GAZEBO_MODEL_PATH — include our simulation/models so mars_rock/rock_large are found
    models_dir = os.path.join(sim_dir, "models")
    existing_model_path = os.environ.get("GAZEBO_MODEL_PATH", "")
    gazebo_model_path   = f"{models_dir}:{existing_model_path}" if existing_model_path else models_dir

    # ── 1. Gazebo server — headless (no GUI, WSL2-compatible) ─────────────
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

    joint_state_publisher = Node(
        package="joint_state_publisher",
        executable="joint_state_publisher",
        name="joint_state_publisher",
        output="screen",
        parameters=[{"use_sim_time": True}],
    )

    # ── 3. Spawn ExoMy (now includes the simulated LiDAR) ──────────────────
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

    # ── 4. DINOv2 terrain node ──────────────────────────────────────────────
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
                    "cache_path":           LaunchConfiguration("cache_path"),
                    "n_shot":               1000,
                }],
                remappings=[
                    ("/camera/image_raw", "/exomy/camera/image_raw"),
                ],
            )
        ],
    )

    # ── 5. LiDAR proximity guard — real-time depth-based stop ──────────────
    # Independent of terrain classification: detects ANY obstacle (rock,
    # equipment, a person) within stop_distance_m, regardless of what it is.
    lidar_guard_node = TimerAction(
        period=44.0,
        actions=[
            Node(
                package="fm_perception",
                executable="lidar_proximity_guard_node.py",
                name="lidar_proximity_guard_node",
                output="screen",
                parameters=[{
                    "scan_topic":        "/scan",
                    "cmd_vel_topic":     "/exomy/cmd_vel",
                    "stop_distance_m":   0.4,
                    "resume_distance_m": 0.5,
                    "stale_timeout_s":   3.0,
                    "publish_rate_hz":   5.0,
                    "use_sim_time":      True,
                }],
            )
        ],
    )

    # ── 6. Terrain controller — now also publishes /terrain_controller/stopped
    #        and cedes cmd_vel to reactive_explorer_node while it's active ────
    controller_node = TimerAction(
        period=46.0,
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

    # ── 7. Reactive explorer — bug-algorithm hazard recovery ───────────────
    reactive_explorer_node = TimerAction(
        period=49.0,
        actions=[
            Node(
                package="fm_perception",
                executable="reactive_explorer_node.py",
                name="reactive_explorer_node",
                output="screen",
                parameters=[{
                    "stuck_dwell_s":         3.0,
                    "check_angle_deg":       90.0,
                    "angle_tolerance_deg":   3.0,
                    "resume_confirm_count":  2,
                    "angular_speed":         0.3,
                    "retreat_speed":         0.03,
                    "retreat_distance_m":    1.0,
                    "max_retreat_cycles":    3,
                    "publish_rate_hz":       10.0,
                    "cmd_vel_topic":         "/exomy/cmd_vel",
                    "odom_topic":            "/exomy/odom",
                    "use_sim_time":          True,
                }],
            )
        ],
    )

    # ── 7b. Stuck-in-sand detection — highest cmd_vel priority of the three ─
    stuck_detection_node = TimerAction(
        period=50.0,
        actions=[
            Node(
                package="fm_perception",
                executable="stuck_detection_node.py",
                name="stuck_detection_node",
                output="screen",
                parameters=[{
                    "stuck_window_s":              4.0,
                    "stuck_displacement_fraction": 0.2,
                    "min_commanded_speed":         0.01,
                    "boost_max_speed":              0.10,
                    "boost_duration_s":             4.0,
                    "wiggle_angle_deg":             20.0,
                    "angular_speed":                0.3,
                    "angle_tolerance_deg":          3.0,
                    "max_wiggle_attempts":          3,
                    "publish_rate_hz":              10.0,
                    "cmd_vel_topic":                "/exomy/cmd_vel",
                    "odom_topic":                   "/exomy/odom",
                    "use_sim_time":                 True,
                }],
            )
        ],
    )

    # ── 8. Safety watchdog ──────────────────────────────────────────────────
    watchdog_node = TimerAction(
        period=51.0,
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

    # ── 9. RViz2 — guaranteed-working fallback live viewer ─────────────────
    rviz2 = TimerAction(
        period=55.0,
        condition=IfCondition(LaunchConfiguration("use_rviz")),
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

    # ── 10. Foxglove bridge — live viewing in Foxglove Studio (ws://8765) ──
    # Requires ros-humble-foxglove-bridge; disable with use_foxglove:=false
    # if it isn't installed yet (RViz2 above works regardless).
    foxglove_bridge = TimerAction(
        period=5.0,
        condition=IfCondition(LaunchConfiguration("use_foxglove")),
        actions=[
            Node(
                package="foxglove_bridge",
                executable="foxglove_bridge",
                name="foxglove_bridge",
                output="screen",
                parameters=[{"port": 8765, "use_sim_time": True}],
            )
        ],
    )

    # ── 11. ROS2 bag recording — full session log for offline review ───────
    bag_name = os.path.join(
        repo_root, "bags",
        f"reactive_exploration_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    bag_record = TimerAction(
        period=5.0,
        actions=[
            ExecuteProcess(
                cmd=[
                    "ros2", "bag", "record",
                    "/exomy/camera/image_raw",
                    "/terrain_viz",
                    "/terrain_classification",
                    "/terrain_class_probs",
                    "/traversability_score",
                    "/terrain_controller/stopped",
                    "/scan",
                    "/lidar_proximity_stop",
                    "/reactive_explorer/active",
                    "/reactive_explorer/failsafe",
                    "/imu_slope_stop",
                    "/stuck_detection/active",
                    "/stuck_detection/failsafe",
                    "/exomy/cmd_vel",
                    "/exomy/odom",
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
        use_rviz, use_foxglove,
        gazebo_server,
        robot_state_publisher,
        joint_state_publisher,
        spawn_robot,
        dinov2_node,
        lidar_guard_node,
        controller_node,
        reactive_explorer_node,
        stuck_detection_node,
        watchdog_node,
        rviz2,
        foxglove_bridge,
        bag_record,
    ])
