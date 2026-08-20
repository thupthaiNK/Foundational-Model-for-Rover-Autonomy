"""
Purpose: Sensor-validation sanity check (2026-08-19, N=1, NOT a thesis-
         reportable result — see docs/gazebo_mixed_obstacles_sensor_
         validation_20260819.md) — same real pipeline as
         reactive_exploration_test.launch.py (Gazebo headless + ExoMy with
         simulated camera/LiDAR/IMU + DINOv2 terrain node + terrain
         controller + LiDAR proximity guard + reactive_explorer_node +
         safety watchdog + rosbag), but pointed at the new
         mars_terrain_mixed_obstacles.world instead of mars_terrain.world.
         That world adds 3 rocks to each of soil/bedrock/sand zone (only
         the rock quadrant had obstacles before) so the real FSM has to
         change heading in every zone, and is shrunk to 5x5 m/quadrant
         (10x10 m total, vs the original 15x12 m/30x24 m) so a run
         finishes within ~10 minutes.
         This is a SEPARATE launch file from reactive_exploration_test.
         launch.py on purpose, for the same reason that one is separate
         from dinov2_controller_test.launch.py — must not change the
         behavior of launch files backing already-written-up results.
         Pipeline (identical to reactive_exploration_test.launch.py):
           camera -> DINOv2 terrain label -\
           /scan  -> LiDAR proximity stop  -+-> terrain_controller_node -> cmd_vel
                                             \-> reactive_explorer_node (takes
                                                 over cmd_vel via the
                                                 /reactive_explorer/active
                                                 handshake whenever hazard-stopped
                                                 longer than stuck_dwell_s)
Inputs:  None (launches everything automatically).
         Optional args: spawn_x, spawn_y, use_rviz, use_foxglove
         Default spawn (0.2, 0.2) — just inside bedrock_zone, 0.2m off the
         exact map centre (2026-08-19 v3). v2 spawned at the literal
         (0,0) corner where all 4 zone planes meet; run 4 showed LiDAR
         reading 0.18m BLOCKED on the very first sweep heading before the
         rover had even turned, from the ~6mm z-discontinuity between
         adjacent zones' differing thicknesses at that exact seam --
         false "no_room_to_turn" failsafe, not a real obstacle. 0.2m
         inside one zone keeps the "near map centre, one rock in every
         cardinal direction" test intent while sitting on uniform ground.
         Was (0.0, 0.0) in v2, (-2.5, 2.5) in v1, (-7.5, 6.0) in
         mars_terrain.world.
Outputs: /terrain_classification, /exomy/cmd_vel, /terrain_controller/stopped,
         /lidar_proximity_stop, /scan, /exomy/imu_raw, /imu_slope_stop,
         /reactive_explorer/active, /reactive_explorer/failsafe,
         /exomy/chase_cam/image_raw (third-person, follows the rover),
         /exomy/camera/image_raw (onboard)
         A ros2 bag under bags/mixed_obstacles_sensor_validation_<timestamp>/
         — added /exomy/imu_raw and /exomy/chase_cam/image_raw to the
         topic list vs reactive_exploration_test.launch.py, since this run
         is specifically about checking IMU and the third-person view
         against ground truth.
How to run:
    # Terminal 1 — build and source first
    cd ros2_ws && colcon build --packages-select fm_perception
    source /opt/ros/humble/setup.bash
    source ros2_ws/install/setup.bash

    # Terminal 2 — launch everything
    source /opt/ros/humble/setup.bash
    source ros2_ws/install/setup.bash
    ros2 launch simulation/launch/mixed_obstacles_sensor_validation.launch.py

    # Disable Foxglove if ros-humble-foxglove-bridge isn't installed yet:
    ros2 launch simulation/launch/mixed_obstacles_sensor_validation.launch.py use_foxglove:=false
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
    world_path = os.path.join(sim_dir, "worlds", "mars_terrain_mixed_obstacles.world")
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
        "spawn_x", default_value="0.2",
        description="Rover spawn X — map centre (0,0), where all 4 quadrants meet (2026-08-19 v2)"
    )
    spawn_y = DeclareLaunchArgument(
        "spawn_y", default_value="0.2",
        description="Rover spawn Y — map centre (0,0), where all 4 quadrants meet (2026-08-19 v2)"
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

    # ── 3. Spawn ExoMy (camera + LiDAR + IMU + chase cam) ──────────────────
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

    # ── 6. Terrain controller ────────────────────────────────────────────
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
    # Period bumped 49.0 -> 90.0 vs reactive_exploration_test.launch.py, and
    # max_turn_duration_s bumped 20.0 -> 40.0 (2026-08-19, first run of this
    # launch file hit FAILSAFE(sweep_turn_timeout) within ~22s of entering
    # STARTUP_SWEEP). Root cause measured directly, not guessed: standalone
    # dinov2_terrain_node.py takes ~14s to reach "ready" with zero CPU
    # contention; in the full stack it only gets a 7s head start (dinov2 at
    # 42s, reactive_explorer at 49s) before Gazebo+7 other nodes are also
    # competing for this machine's CPU, so STARTUP_SWEEP's first heading
    # (20s deadline) elapsed with zero traversability_score ever received.
    # This is a pre-existing timing gap in reactive_exploration_test.
    # launch.py's periods, not something the map/obstacle changes caused —
    # fixed here only, since that file "must not change behavior".
    # terrain_confirm_timeout_s bumped 15.0 -> 40.0 for the same reason:
    # a 2nd run (with the fixes above) got past STARTUP_SWEEP and drove
    # for ~75s, then hit FAILSAFE(terrain_confirm_timeout) — same CPU-
    # contention pattern (Gazebo + CPU-only DINOv2 + full node stack
    # sharing this dev machine), not a new bug.
    # sweep_headings bumped 8 -> 4 (author request 2026-08-19): matches the
    # real-hardware field-test convention of checking 4 directions before
    # committing, and halves STARTUP_SWEEP's DINOv2 inference load.
    reactive_explorer_node = TimerAction(
        period=90.0,
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
                    "max_turn_duration_s":   40.0,
                    "terrain_confirm_timeout_s": 40.0,
                    "sweep_headings":        4,
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
        period=91.0,
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
        period=92.0,
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
        period=96.0,
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

    # ── 11. ROS2 bag recording — adds /exomy/imu_raw + chase cam vs the
    #        reactive_exploration_test.launch.py topic list, since this run
    #        is specifically for the IMU/LiDAR/DINOv2-vs-ground-truth check
    #        and the third-person video deliverable. ────────────────────────
    # Written to a native Linux path, NOT repo_root/bags (2026-08-19, after
    # run 5): repo_root lives under /mnt/c/... (a WSL2 cross-OS mount to the
    # Windows drive), which is far slower than native ext4. Under this run's
    # CPU contention (Gazebo + CPU-only DINOv2 + 7 nodes), the recorder's
    # write cache fell behind and the bag lost its last ~25s -- exactly the
    # drive-to-rock-and-stop window the run exists to capture -- even
    # before the process's own SIGSEGV on shutdown. Move the bag itself
    # after the run: `cp -r ~/bags_native/<name> bags/` from repo root.
    _native_bags_dir = os.path.join(os.path.expanduser("~"), "bags_native")
    bag_name = os.path.join(
        _native_bags_dir,
        f"mixed_obstacles_sensor_validation_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    bag_record = TimerAction(
        period=5.0,
        actions=[
            ExecuteProcess(
                cmd=[
                    "ros2", "bag", "record",
                    "/exomy/camera/image_raw",
                    "/exomy/chase_cam/image_raw",
                    "/terrain_viz",
                    "/terrain_classification",
                    "/terrain_class_probs",
                    "/traversability_score",
                    "/terrain_controller/stopped",
                    "/scan",
                    "/lidar_proximity_stop",
                    "/exomy/imu_raw",
                    "/imu_slope_stop",
                    "/reactive_explorer/active",
                    "/reactive_explorer/failsafe",
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
