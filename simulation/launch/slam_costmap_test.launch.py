"""
Purpose: L4 Phase A2 -- combines the working odom-assisted SLAM
         configuration (simulation/launch/slam_test.launch.py) with DINOv2
         terrain classification and live_traversability_costmap_node.py in
         its new pose_source=slam mode, to verify the wiring closes the gap
         D1 (§4.8.13) explicitly left open: a live costmap driven by a real
         pose estimate instead of Gazebo's ground-truth odometry cheat.
         Deliberately excludes Nav2 and reactive_explorer_node/
         terrain_controller_node -- this tests the perception+SLAM+costmap
         chain in isolation, not full navigation (that step, and its much
         higher compute risk from re-adding Nav2, is Phase C, parked
         separately).
Inputs:  None (launches everything automatically).
Outputs: /traversability_costmap (nav_msgs/OccupancyGrid), driven by
         slam_toolbox's /pose instead of /exomy/odom.
How to run:
    source /opt/ros/humble/setup.bash
    source ros2_ws/install/setup.bash
    ros2 launch simulation/launch/slam_costmap_test.launch.py
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
                    "-x", "2.0", "-y", "-6.0", "-z", "0.15",
                ],
            )
        ],
    )

    # -- slam_toolbox (async, mapping mode), odom-assisted -- the config
    # that worked in Phase A (Run 3, ~0.18m mean error).
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
                }],
            )
        ],
    )

    # -- DINOv2 terrain classification --
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

    # -- Live traversability costmap, driven by slam_toolbox's /pose (the
    # piece D1 could not close, §4.8.13/§5.6.5) instead of ground-truth odom.
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

    return LaunchDescription([
        gazebo_server,
        robot_state_publisher,
        joint_state_publisher,
        spawn_robot,
        slam_toolbox_node,
        dinov2_node,
        costmap_node,
    ])
