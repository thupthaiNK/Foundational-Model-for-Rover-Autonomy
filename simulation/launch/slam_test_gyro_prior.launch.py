"""
Purpose: L4 Phase A, gyro-prior variant -- a third attempt at odom-free
         SLAM, after odom_frame==base_frame (Run 4: slam_toolbox never
         registered the sensor) and a fully-static identity transform
         (Run 5: registered but dropped every scan, message-filter queue
         full) both failed to produce any pose output. This version
         replaces the static identity transform with
         experiments/gyro_odom_publisher.py, which integrates Gazebo's
         simulated IMU gyro (/exomy/imu_raw) into a real, continuously-
         updating gyro_odom->base_link TF -- correct rotation, but always
         zero translation (a gyro physically cannot measure translation).
         The IMU sensor added here (Gazebo's simulated one, already present
         in simulation/urdf/exomy.urdf.xacro for the future real-hardware
         slope-detection use case) publishes the same /exomy/imu_raw topic
         and message type that icm20948_driver_node.py will publish on real
         hardware once it exists there -- this test's architecture
         transfers directly, only the sensor source changes.
Inputs:  None (launches everything automatically).
         Optional arg: transform_timeout (default 0.5, matching the value
         that failed 3x before -- §4.8.23 follow-up tests a much more
         generous value, e.g. transform_timeout:=5.0, as the most targeted
         untried lever, since all 3 prior odom-free attempts failed with
         the tf2 message-filter "queue is full" symptom regardless of what
         content fed odom_frame).
Outputs: /map (nav_msgs/OccupancyGrid), /pose (geometry_msgs/PoseWithCovarianceStamped)
         from slam_toolbox; a map->gyro_odom->base_link TF chain.
How to run:
    source /opt/ros/humble/setup.bash
    source ros2_ws/install/setup.bash
    ros2 launch simulation/launch/slam_test_gyro_prior.launch.py
    ros2 launch simulation/launch/slam_test_gyro_prior.launch.py transform_timeout:=5.0
Project: Onboard Visual Foundation Models for Mars Terrain Perception and Rover Navigation
"""

import os

import xacro
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, TimerAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():

    transform_timeout_arg = DeclareLaunchArgument(
        "transform_timeout", default_value="0.5",
        description="slam_toolbox's transform_timeout parameter (seconds). Default matches the "
                     "value that failed in all 3 prior odom-free attempts."
    )

    sim_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    urdf_path = os.path.join(sim_dir, "urdf", "exomy.urdf.xacro")
    world_path = os.path.join(sim_dir, "worlds", "mars_terrain.world")
    repo_root = os.path.normpath(os.path.join(sim_dir, ".."))
    gyro_script = os.path.join(repo_root, "experiments", "gyro_odom_publisher.py")

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

    # -- Gyro-only motion-prior publisher (real translation-blind, TF-valid
    # alternative to the static identity transform tried in Run 5).
    gyro_odom_node = TimerAction(
        period=38.0,
        actions=[
            ExecuteProcess(
                cmd=["python3", gyro_script, "--ros-args", "-p", "use_sim_time:=true"],
                output="screen",
            )
        ],
    )

    # -- slam_toolbox (async, mapping mode), gyro-prior: odom_frame points
    # at gyro_odom_publisher's output, not Gazebo's real "odom" frame and
    # not a static identity transform.
    slam_toolbox_node = TimerAction(
        period=42.0,
        actions=[
            Node(
                package="slam_toolbox",
                executable="async_slam_toolbox_node",
                name="slam_toolbox",
                output="screen",
                parameters=[{
                    "odom_frame": "gyro_odom",
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
                    "transform_timeout": LaunchConfiguration("transform_timeout"),
                }],
            )
        ],
    )

    return LaunchDescription([
        transform_timeout_arg,
        gazebo_server,
        robot_state_publisher,
        joint_state_publisher,
        spawn_robot,
        gyro_odom_node,
        slam_toolbox_node,
    ])
