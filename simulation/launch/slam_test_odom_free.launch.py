"""
Purpose: L4 Phase A, odom-free variant -- the real-hardware-relevant test
         (ExoMy has no wheel encoders, so no real odometry source exists).
         A first attempt at this used odom_frame == base_frame ("base_link"
         for both), the technique commonly suggested for odometry-less
         platforms -- but empirically, slam_toolbox never even reached its
         "Registering sensor" log line in that configuration (checked via
         the launch log, not assumed), meaning it never successfully
         processed a single scan. This version uses the standard,
         better-supported pattern instead: a `static_transform_publisher`
         publishing a fixed IDENTITY transform from a new frame ("slam_odom",
         which nothing else in this simulation publishes to or reads from --
         Gazebo's diff_drive plugin publishes its own separate real,
         moving "odom"->"base_link" TF that this does not touch) to
         "base_link". Because this transform never updates, slam_toolbox's
         usual "where do I expect to be based on odometry" prior is always
         "exactly where I started" between scans -- it must then rely
         entirely on scan-to-map matching (via its own map->slam_odom
         correction) to account for any actual motion. This still provides
         slam_toolbox a continuously-valid TF chain to look up (unlike the
         odom_frame==base_frame attempt), which is the piece that was
         missing before. rf2o_laser_odometry and laser_scan_matcher (the
         other common "real" laser-odometry ROS2 packages) were checked and
         are not installed on this machine and not available via apt --
         building either from source was judged out of scope for this
         diagnostic test.
Inputs:  None (launches everything automatically).
Outputs: /map (nav_msgs/OccupancyGrid), /pose (geometry_msgs/PoseWithCovarianceStamped)
         from slam_toolbox; a map->slam_odom->base_link TF chain.
How to run:
    source /opt/ros/humble/setup.bash
    source ros2_ws/install/setup.bash
    ros2 launch simulation/launch/slam_test_odom_free.launch.py
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

    # -- Static identity "zero motion prior" transform, slam_odom -> base_link.
    # Started alongside Gazebo/robot_state_publisher (not gated on spawn) --
    # a static transform has no dependency on the robot existing yet.
    zero_odom_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="slam_odom_identity_tf",
        output="screen",
        arguments=["--frame-id", "slam_odom", "--child-frame-id", "base_link"],
    )

    # -- slam_toolbox (async, mapping mode), ODOM-FREE: odom_frame points at
    # the static identity transform above, not Gazebo's real "odom" frame.
    slam_toolbox_node = TimerAction(
        period=42.0,
        actions=[
            Node(
                package="slam_toolbox",
                executable="async_slam_toolbox_node",
                name="slam_toolbox",
                output="screen",
                parameters=[{
                    "odom_frame": "slam_odom",
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
        zero_odom_tf,
        slam_toolbox_node,
    ])
