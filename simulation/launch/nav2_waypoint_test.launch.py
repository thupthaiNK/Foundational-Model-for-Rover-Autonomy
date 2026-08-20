"""
Purpose: Full Nav2 + DINOv2 + live/static traversability-costmap bring-up —
         Gazebo (headless) + ExoMy + DINOv2 terrain node + a costmap producer
         node (Condition A or B) + the Nav2 navigation stack (no AMCL/SLAM —
         ground-truth odometry + identity map->odom transform). Tests the
         L1 (DINOv2 perception) -> L4 (Nav2 waypoint navigation) integration.
Inputs:  None (launches everything automatically).
         Launch args: costmap_mode (static|live, default static),
                      spawn_x, spawn_y (rover start position).
Outputs: /traversability_costmap (nav_msgs/OccupancyGrid)
         /terrain_classification (DINOv2 label:confidence)
         /cmd_vel (Nav2's velocity_smoother output) relayed to /exomy/cmd_vel
         by cmd_vel_relay_node (the reactive terrain_controller_node is NOT
         launched here — Nav2 drives the rover instead)
         /plan (nav_msgs/Path, used by experiments/nav2_waypoint_experiment.py)
How to run:
    cd ros2_ws && colcon build --packages-select fm_perception
    source /opt/ros/humble/setup.bash
    source ros2_ws/install/setup.bash
    ros2 launch simulation/launch/nav2_waypoint_test.launch.py costmap_mode:=static
    ros2 launch simulation/launch/nav2_waypoint_test.launch.py costmap_mode:=live
Project: Onboard Visual Foundation Models for Mars Terrain Perception and Rover Navigation
"""

import os
import xacro
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    ExecuteProcess, TimerAction, DeclareLaunchArgument, IncludeLaunchDescription,
)
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def generate_launch_description():

    sim_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    urdf_path = os.path.join(sim_dir, "urdf", "exomy.urdf.xacro")
    world_path = os.path.join(sim_dir, "worlds", "mars_terrain.world")
    nav2_params_path = os.path.join(sim_dir, "config", "nav2_params.yaml")

    repo_root = os.path.normpath(os.path.join(sim_dir, ".."))
    default_cache = os.path.join(
        repo_root, "experiments", "results", "feature_cache",
        "dinov2_reg_small_train_1000shot.npz"
    )

    robot_description = xacro.process_file(urdf_path).toxml()

    costmap_mode_arg = DeclareLaunchArgument(
        "costmap_mode", default_value="static",
        description="static (Condition A, known hazard map) or live (Condition B, DINOv2-driven)"
    )
    spawn_x_arg = DeclareLaunchArgument(
        "spawn_x", default_value="-7.5",
        description="Rover spawn X — default soil_zone centre. NOTE: this is "
                     "only where the launch file itself spawns the rover for "
                     "manual smoke tests; experiments/nav2_waypoint_experiment.py "
                     "teleports to the real mission's START_POSE (bedrock_zone, "
                     "(7.5, 1.0)) before every trial, overriding this default."
    )
    spawn_y_arg = DeclareLaunchArgument(
        "spawn_y", default_value="6.0",
        description="Rover spawn Y — default soil_zone centre (see spawn_x note)"
    )

    # is_live is the logical complement of is_static (not an independent
    # equality check) so the two are mutually exclusive AND exhaustive by
    # construction: costmap_mode == 'static' launches the static node,
    # literally anything else (including 'live' and any typo) launches the
    # live node. This avoids the failure mode where an invalid costmap_mode
    # value would silently launch neither costmap node.
    is_static = IfCondition(PythonExpression(["'", LaunchConfiguration("costmap_mode"), "' == 'static'"]))
    is_live = UnlessCondition(PythonExpression(["'", LaunchConfiguration("costmap_mode"), "' == 'static'"]))

    models_dir = os.path.join(sim_dir, "models")
    existing_model_path = os.environ.get("GAZEBO_MODEL_PATH", "")
    gazebo_model_path = f"{models_dir}:{existing_model_path}" if existing_model_path else models_dir

    # ── 1. Gazebo server — headless. libgazebo_ros_state.so intentionally
    #       omitted (confirmed incompatible with this Gazebo 11.10.2 install).
    gazebo_server = ExecuteProcess(
        cmd=["gzserver", "--verbose", world_path,
             "-s", "libgazebo_ros_init.so",
             "-s", "libgazebo_ros_factory.so"],
        output="screen",
        additional_env={"GAZEBO_MODEL_PATH": gazebo_model_path},
    )

    # ── 2. Robot state publisher + joint state publisher ───────────────────
    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="screen",
        parameters=[{"robot_description": robot_description, "use_sim_time": True}],
    )
    joint_state_publisher = Node(
        package="joint_state_publisher",
        executable="joint_state_publisher",
        name="joint_state_publisher",
        output="screen",
        parameters=[{"use_sim_time": True}],
    )

    # ── 3. map -> odom identity static transform — ground-truth odometry
    #       from libgazebo_ros_diff_drive stands in for localization/SLAM,
    #       which is explicitly out of scope for this thesis.
    map_to_odom_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="map_to_odom_tf",
        output="screen",
        arguments=["--x", "0", "--y", "0", "--z", "0",
                   "--qx", "0", "--qy", "0", "--qz", "0", "--qw", "1",
                   "--frame-id", "map", "--child-frame-id", "odom"],
        parameters=[{"use_sim_time": True}],
    )

    # ── 4. Spawn ExoMy — 35s delay to let Gazebo load the 21 rock meshes
    #       (same timing as simulation/launch/dinov2_controller_test.launch.py).
    spawn_robot = TimerAction(
        period=35.0,
        actions=[
            Node(
                package="gazebo_ros",
                executable="spawn_entity.py",
                name="spawn_exomy",
                output="screen",
                arguments=["-topic", "robot_description", "-entity", "exomy",
                           "-x", LaunchConfiguration("spawn_x"),
                           "-y", LaunchConfiguration("spawn_y"),
                           "-z", "0.15"],
                parameters=[{"use_sim_time": True}],
            )
        ],
    )

    # ── 5. DINOv2 terrain node ──────────────────────────────────────────────
    # Condition A (static) never reads /terrain_classification, so this node is
    # skipped entirely for it to free CPU for Gazebo/Nav2 under this machine's
    # RTF constraint (see docs/d1_nav2_waypoint_experiment_log.md §7-8).
    dinov2_node = TimerAction(
        period=42.0,
        condition=is_live,
        actions=[
            Node(
                package="fm_perception",
                executable="dinov2_terrain_node.py",
                name="dinov2_terrain_node",
                output="screen",
                parameters=[{
                    "device": "cpu",
                    "confidence_threshold": 0.40,
                    "publish_viz": False,
                    "cache_path": default_cache,
                    "n_shot": 1000,
                    "class_weight_balanced": False,
                    "enable_obstacle_gate": False,
                }],
                remappings=[("/camera/image_raw", "/exomy/camera/image_raw")],
            )
        ],
    )

    # ── 6. Costmap producer — exactly one of these runs, by costmap_mode ───
    static_costmap_node = TimerAction(
        period=44.0,
        condition=is_static,
        actions=[
            Node(
                package="fm_perception",
                executable="static_traversability_costmap_node.py",
                name="static_traversability_costmap_node",
                output="screen",
                parameters=[{"use_sim_time": True}],
            )
        ],
    )
    live_costmap_node = TimerAction(
        period=44.0,
        condition=is_live,
        actions=[
            Node(
                package="fm_perception",
                executable="live_traversability_costmap_node.py",
                name="live_traversability_costmap_node",
                output="screen",
                parameters=[{"use_sim_time": True, "lookahead_m": 0.6, "patch_radius_m": 0.3}],
            )
        ],
    )

    # ── 7. Nav2 navigation stack (no AMCL/map_server — see params header).
    #       No namespace, so its final cmd_vel output is the global /cmd_vel
    #       topic — bridged to /exomy/cmd_vel by the relay node below.
    nav2_navigation = TimerAction(
        period=46.0,
        actions=[
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    os.path.join(get_package_share_directory("nav2_bringup"),
                                 "launch", "navigation_launch.py")
                ),
                launch_arguments={
                    "use_sim_time": "true",
                    "params_file": nav2_params_path,
                    "autostart": "true",
                }.items(),
            )
        ],
    )

    # ── 8. /cmd_vel -> /exomy/cmd_vel relay (see Task 5 rationale above) ────
    cmd_vel_relay = TimerAction(
        period=46.0,
        actions=[
            Node(
                package="fm_perception",
                executable="cmd_vel_relay_node.py",
                name="cmd_vel_relay_node",
                output="screen",
                parameters=[{"use_sim_time": True}],
            )
        ],
    )

    return LaunchDescription([
        costmap_mode_arg, spawn_x_arg, spawn_y_arg,
        gazebo_server, robot_state_publisher, joint_state_publisher,
        map_to_odom_tf, spawn_robot, dinov2_node,
        static_costmap_node, live_costmap_node,
        nav2_navigation, cmd_vel_relay,
    ])
