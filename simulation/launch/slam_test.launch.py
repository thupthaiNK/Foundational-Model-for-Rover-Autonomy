"""
Purpose: Phase A of the L4 (Localisation) feasibility investigation -- an
         ISOLATED test of slam_toolbox against the existing simulated LiDAR,
         deliberately kept separate from the perception/safety stack
         (DINOv2, terrain_controller_node, reactive_explorer_node, Nav2) so
         a first SLAM result is not confounded by any of that. This launch
         file starts Gazebo (the same mars_terrain.world every other Gazebo
         result in this thesis uses, unmodified) + ExoMy (with its existing
         simulated LiDAR, §3.11.5) + async_slam_toolbox_node.
         base_frame is set to "base_link" to match the diff_drive Gazebo
         plugin's actual robot_base_frame (the slam_toolbox package default
         is "base_footprint", which ExoMy's URDF does not publish).
         odom_frame is left at the slam_toolbox default ("odom"), which
         means this first test uses Gazebo's ground-truth /exomy/odom (via
         the diff_drive plugin's own odom->base_link TF, publish_odom_tf via
         URDF) as slam_toolbox's motion prior -- this is a BEST-CASE
         configuration that real ExoMy hardware cannot reproduce (no wheel
         encoders). This is intentional: the first question is "can this
         tool produce a sane map/pose at all in this environment," before
         asking the harder, more real-hardware-relevant question of how
         much accuracy is lost without wheel odometry.
Inputs:  None (launches everything automatically).
Outputs: /map (nav_msgs/OccupancyGrid), /pose (geometry_msgs/PoseWithCovarianceStamped)
         from slam_toolbox; a map->base_link TF.
How to run:
    source /opt/ros/humble/setup.bash
    source ros2_ws/install/setup.bash
    ros2 launch simulation/launch/slam_test.launch.py
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

    # -- slam_toolbox (async, mapping mode) -- started after the robot has
    # spawned and settled so /scan is already publishing real data.
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

    return LaunchDescription([
        gazebo_server,
        robot_state_publisher,
        joint_state_publisher,
        spawn_robot,
        slam_toolbox_node,
    ])
