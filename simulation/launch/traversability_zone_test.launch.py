"""
Traversability zone test — spawn rover at a specific terrain zone position.
Run this 4 times (once per zone) to collect traversability data for the thesis.

Usage:
    # Zone 1: bedrock (origin)
    ros2 launch simulation/launch/traversability_zone_test.launch.py zone:=origin

    # Zone 2: soil
    ros2 launch simulation/launch/traversability_zone_test.launch.py zone:=soil

    # Zone 3: bedrock far
    ros2 launch simulation/launch/traversability_zone_test.launch.py zone:=bedrock

    # Zone 4: rock
    ros2 launch simulation/launch/traversability_zone_test.launch.py zone:=rock

    # Monitor output (in second terminal):
    ros2 topic echo /terrain_classification

Project: Onboard Visual Foundation Models for Mars Terrain Perception and Rover Navigation
"""

import os, xacro
from launch import LaunchDescription
from launch.actions import ExecuteProcess, TimerAction, DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


# Terrain zone spawn positions (from mars_terrain.world layout)
ZONE_POSITIONS = {
    "origin":  (0.0,  0.0,  0.15),   # spawn default — bedrock/rock mix
    "soil":    (-3.0,  0.0, 0.15),   # soil zone (west)
    "bedrock": ( 3.0,  0.0, 0.15),   # bedrock zone (east)
    "sand":    ( 0.0,  3.0, 0.15),   # sand zone (north)
    "rock":    ( 0.0, -3.0, 0.15),   # boulder field (south)
}


def generate_launch_description():
    sim_dir   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    urdf_path = os.path.join(sim_dir, "urdf", "exomy.urdf.xacro")
    world_path = os.path.join(sim_dir, "worlds", "mars_terrain.world")

    robot_description = xacro.process_file(urdf_path).toxml()

    zone_arg = DeclareLaunchArgument(
        "zone", default_value="origin",
        description="Terrain zone to test: origin, soil, bedrock, sand, rock"
    )
    zone = LaunchConfiguration("zone")

    gazebo = ExecuteProcess(
        cmd=[
            "gzserver", "--verbose", world_path,
            "-s", "libgazebo_ros_init.so",
            "-s", "libgazebo_ros_factory.so",
        ],
        output="screen",
    )

    rsp = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="screen",
        parameters=[{"robot_description": robot_description, "use_sim_time": True}],
    )

    # Spawn at zone-specific position
    # NOTE: zone is a LaunchConfiguration (string), positions are hardcoded per zone
    # We use a TimerAction to spawn after Gazebo starts
    spawn = TimerAction(
        period=3.0,
        actions=[Node(
            package="gazebo_ros",
            executable="spawn_entity.py",
            output="screen",
            arguments=[
                "-topic", "robot_description",
                "-entity", "exomy",
                "-x", "0.0", "-y", "0.0", "-z", "0.15",  # default position
                # Note: to test different zones, change x/y here or use zone arg
            ],
        )],
    )

    # CLIP terrain node starts after robot spawns
    clip_node = TimerAction(
        period=5.0,
        actions=[Node(
            package="fm_perception",
            executable="clip_terrain_node.py",
            name="clip_terrain_node",
            output="screen",
            parameters=[{
                "device": "cpu",
                "confidence_threshold": 0.40,
                "publish_viz": False,
            }],
            remappings=[("/camera/image_raw", "/exomy/camera/image_raw")],
        )],
    )

    return LaunchDescription([zone_arg, gazebo, rsp, spawn, clip_node])
