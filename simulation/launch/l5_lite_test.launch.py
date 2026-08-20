"""
Purpose: Live Gazebo verification of "L5-lite" (backlog item, scoped via
         grill-thesis 2026-07-17) -- a lightweight A*/pure-pursuit path
         planner built as an alternative to the full Nav2 stack that D1
         (§4.8.13) found alone drove load average to 9.33 on this 4-core
         development machine. Reuses D1's exact START_POSE (7.5, 1.0,
         bedrock_zone near the hazard boundary -- NOT the launch file's
         generic spawn_x/spawn_y default of -7.5,6.0, which D1 itself
         overrides via delete+respawn) and GOAL_POSE (-7.5, -9.0, sand_zone
         opposite side, the l5_lite_planner_node.py default) for direct
         comparability with D1's own null result (0/20 success, §4.8.13).
         The direct route between these two points crosses rock_cluster/
         boulder_zone (hazard zones) -- the only safe path is a detour
         north through soil_zone, a genuine test of hazard avoidance, not
         just an empty-corridor straight line. Combines the
         working odom-assisted SLAM configuration (slam_test.launch.py,
         §4.8.22) with DINOv2 terrain classification,
         live_traversability_costmap_node.py's pose_source="slam" mode
         (§4.8.24, Phase A2), and l5_lite_planner_node.py -- mirrors
         slam_costmap_test.launch.py exactly, only changing the spawn
         point and adding the new planner node. Deliberately excludes
         terrain_controller_node.py/reactive_explorer_node.py/
         stuck_detection_node.py: this isolates the planner+follower
         pipeline itself, matching every prior isolation precedent in this
         thesis (Phase A/A2, D1 Condition A/B).
Inputs:  None (launches everything automatically).
Outputs: /exomy/cmd_vel (from l5_lite_planner_node.py)
         /l5_lite_plan (nav_msgs/Path)
         /traversability_costmap (nav_msgs/OccupancyGrid)
How to run:
    source /opt/ros/humble/setup.bash
    source ros2_ws/install/setup.bash
    ros2 launch simulation/launch/l5_lite_test.launch.py
Project: Onboard Visual Foundation Models for Mars Terrain Perception and Rover Navigation
"""

import os

import xacro
from launch import LaunchDescription
from launch.actions import ExecuteProcess, TimerAction
from launch_ros.actions import Node


def generate_launch_description():

    sim_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    urdf_path = os.path.join(sim_dir, "urdf", "exomy.urdf.xacro")
    world_path = os.path.join(sim_dir, "worlds", "mars_terrain.world")
    repo_root = os.path.normpath(os.path.join(sim_dir, ".."))
    default_cache = os.path.join(
        repo_root, "experiments", "results", "feature_cache",
        "dinov2_reg_small_train_1000shot.npz"
    )

    robot_description = xacro.process_file(urdf_path).toxml()

    models_dir = os.path.join(sim_dir, "models")
    existing_model_path = os.environ.get("GAZEBO_MODEL_PATH", "")
    gazebo_model_path = f"{models_dir}:{existing_model_path}" if existing_model_path else models_dir

    gazebo_server = ExecuteProcess(
        cmd=[
            "gzserver", "--verbose", world_path,
            "-s", "libgazebo_ros_init.so",
            "-s", "libgazebo_ros_factory.so",
        ],
        output="screen",
        additional_env={"GAZEBO_MODEL_PATH": gazebo_model_path},
    )

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

    # D1's exact START_POSE (§4.8.13): bedrock_zone near the hazard boundary.
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
                    "-x", "7.5", "-y", "1.0", "-z", "0.15",
                ],
            )
        ],
    )

    # -- slam_toolbox (async, mapping mode), odom-assisted -- the config
    # that worked in Phase A (Run 3, ~0.18m mean error), §4.8.22.
    slam_toolbox_node = TimerAction(
        period=42.0,
        actions=[
            Node(
                package="slam_toolbox",
                executable="async_slam_toolbox_node",
                name="slam_toolbox",
                output="screen",
                parameters=[{
                    "odom_frame": "odom",
                    "map_frame": "map",
                    "base_frame": "base_link",
                    "scan_topic": "/scan",
                    "mode": "mapping",
                    "use_sim_time": True,
                    "transform_publish_period": 0.02,
                    "map_update_interval": 2.0,
                    "resolution": 0.05,
                    "max_laser_range": 12.0,
                    "minimum_travel_distance": 0.2,
                    "minimum_travel_heading": 0.2,
                    "transform_timeout": 0.5,
                    "scan_queue_size": 50,
                }],
            )
        ],
    )

    dinov2_node = TimerAction(
        period=44.0,
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
                }],
                remappings=[
                    ("/camera/image_raw", "/exomy/camera/image_raw"),
                ],
            )
        ],
    )

    costmap_node = TimerAction(
        period=46.0,
        actions=[
            Node(
                package="fm_perception",
                executable="live_traversability_costmap_node.py",
                name="live_traversability_costmap_node",
                output="screen",
                parameters=[{
                    "pose_source": "slam",
                }],
            )
        ],
    )

    # -- L5-lite: A* + pure pursuit over the live costmap above, using
    # slam_toolbox's /pose. Default goal (-7.5, -9.0) matches D1 exactly.
    l5_lite_node = TimerAction(
        period=48.0,
        actions=[
            Node(
                package="fm_perception",
                executable="l5_lite_planner_node.py",
                name="l5_lite_planner_node",
                output="screen",
                parameters=[{
                    "use_sim_time": True,
                }],
            )
        ],
    )

    return LaunchDescription([
        gazebo_server,
        robot_state_publisher,
        joint_state_publisher,
        spawn_robot,
        slam_toolbox_node,
        dinov2_node,
        costmap_node,
        l5_lite_node,
    ])
