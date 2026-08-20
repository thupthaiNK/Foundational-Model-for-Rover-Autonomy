"""
Purpose: Launch ExoMy rover in Gazebo with Mars terrain world.
         Spawns robot_state_publisher, Gazebo, and the ExoMy URDF model.
How to run:
    source /opt/ros/humble/setup.bash
    cd "Thesis code"
    ros2 launch simulation/launch/exomy_gazebo.launch.py
Project: Onboard Visual Foundation Models for Mars Terrain Perception and Rover Navigation
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import ExecuteProcess, DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
import xacro


def generate_launch_description():

    # ── Paths ─────────────────────────────────────────────────────────
    pkg_dir   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    urdf_path = os.path.join(pkg_dir, "urdf", "exomy.urdf.xacro")
    world_path = os.path.join(pkg_dir, "worlds", "mars_terrain.world")

    # Process xacro → URDF string
    robot_description = xacro.process_file(urdf_path).toxml()

    # ── Arguments ─────────────────────────────────────────────────────
    use_sim_time = DeclareLaunchArgument(
        "use_sim_time", default_value="true",
        description="Use simulation time"
    )

    # ── Nodes ──────────────────────────────────────────────────────────

    # Robot state publisher (publishes /tf from URDF)
    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="screen",
        parameters=[{
            "robot_description": robot_description,
            "use_sim_time": LaunchConfiguration("use_sim_time"),
        }]
    )

    # Gazebo server
    gazebo_server = ExecuteProcess(
        cmd=["gzserver", "--verbose", world_path,
             "-s", "libgazebo_ros_init.so",
             "-s", "libgazebo_ros_factory.so"],
        output="screen",
    )

    # Gazebo client (GUI)
    gazebo_client = ExecuteProcess(
        cmd=["gzclient", "--verbose"],
        output="screen",
    )

    # Spawn ExoMy robot
    spawn_robot = Node(
        package="gazebo_ros",
        executable="spawn_entity.py",
        name="spawn_exomy",
        output="screen",
        arguments=[
            "-topic", "robot_description",
            "-entity", "exomy",
            "-x", "0.0", "-y", "0.0", "-z", "0.15",
            "-R", "0.0", "-P", "0.0", "-Y", "0.0",
        ]
    )

    return LaunchDescription([
        use_sim_time,
        robot_state_publisher,
        gazebo_server,
        gazebo_client,
        spawn_robot,
    ])
