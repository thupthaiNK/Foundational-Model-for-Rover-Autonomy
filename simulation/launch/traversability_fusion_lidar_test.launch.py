"""
Purpose: Isolated Gazebo live-verify launch for traversability_score_fusion_node.py's
         LiDAR term (backlog item 8, scoped via grill-thesis 2026-07-17). Gazebo +
         ExoMy + dinov2_terrain_node.py (for /traversability_score) + the new
         fusion node -- deliberately excludes terrain_controller_node.py,
         safety_watchdog_node.py, and reactive_explorer_node.py, mirroring the
         L4 Phase A isolation precedent ("bypasses the safety/perception stack
         entirely -- cmd_vel is published directly by [the test] script"):
         driving toward Q4 near a known STOP-policy hazard would otherwise
         cause terrain_controller_node to fight the test script for
         /exomy/cmd_vel. Default spawn (2.0, -4.0) is the Q4 rock_cluster
         point already used and verified in this thesis (§3.11.4: driving
         straight into it at 0.10 m/s for 25s produces measurable collision
         within the run), giving the LiDAR term real obstacles to react to.
Inputs:  None (launches everything automatically).
         Optional args: spawn_x, spawn_y (default 2.0, -4.0 -- Q4 rock_cluster)
Outputs: /traversability_score (std_msgs/Float64) -- DINOv2-only, from dinov2_terrain_node.py
         /traversability_score_fused (std_msgs/Float64) -- from the new fusion node
         /scan (sensor_msgs/LaserScan) -- Gazebo's simulated LiDAR
How to run:
    source /opt/ros/humble/setup.bash
    source ros2_ws/install/setup.bash
    ros2 launch simulation/launch/traversability_fusion_lidar_test.launch.py
    # Terminal 2, after Gazebo/DINOv2/fusion node are all up:
    python3 experiments/traversability_fusion_lidar_live_test.py
Project: Onboard Visual Foundation Models for Mars Terrain Perception and Rover Navigation
"""
import os

import xacro
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, TimerAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    sim_dir    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    urdf_path  = os.path.join(sim_dir, "urdf", "exomy.urdf.xacro")
    world_path = os.path.join(sim_dir, "worlds", "mars_terrain.world")

    repo_root = os.path.normpath(os.path.join(sim_dir, ".."))
    cache_path = os.path.join(
        repo_root, "experiments", "results", "feature_cache",
        "dinov2_reg_small_train_1000shot.npz"
    )

    robot_description = xacro.process_file(urdf_path).toxml()

    spawn_x = DeclareLaunchArgument(
        "spawn_x", default_value="2.0",
        description="Rover spawn X -- default Q4 rock_cluster (2.0, -4.0), §3.11.4"
    )
    spawn_y = DeclareLaunchArgument(
        "spawn_y", default_value="-4.0",
        description="Rover spawn Y -- default Q4 rock_cluster (2.0, -4.0), §3.11.4"
    )

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
                    "-x", LaunchConfiguration("spawn_x"),
                    "-y", LaunchConfiguration("spawn_y"),
                    "-z", "0.15",
                ],
            )
        ],
    )

    dinov2_node = TimerAction(
        period=42.0,
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
                    "cache_path": cache_path,
                    "n_shot": 1000,
                }],
                remappings=[
                    ("/camera/image_raw", "/exomy/camera/image_raw"),
                ],
            )
        ],
    )

    fusion_node = TimerAction(
        period=45.0,
        actions=[
            Node(
                package="fm_imu_fusion",
                executable="traversability_score_fusion_node.py",
                name="traversability_score_fusion_node",
                output="screen",
                parameters=[{"use_sim_time": True}],
            )
        ],
    )

    return LaunchDescription([
        spawn_x, spawn_y,
        gazebo_server,
        robot_state_publisher,
        joint_state_publisher,
        spawn_robot,
        dinov2_node,
        fusion_node,
    ])
