"""
Purpose: Launch the complete FM perception pipeline:
         clip_terrain_node (every frame) + blip2_scene_node (every 5s)
How to run:
    source /opt/ros/humble/setup.bash
    cd ros2_ws && colcon build --packages-select fm_perception
    source install/setup.bash
    ros2 launch fm_perception fm_perception.launch.py
Project: Onboard Visual Foundation Models for Mars Terrain Perception and Rover Navigation
"""

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    clip_node = Node(
        package="fm_perception",
        executable="clip_terrain_node.py",
        name="clip_terrain_node",
        output="screen",
        parameters=[{
            "device":               "cpu",
            "confidence_threshold": 0.40,
            "publish_viz":          True,
        }],
        remappings=[
            ("/camera/image_raw", "/exomy/camera/image_raw"),
        ],
    )

    blip2_node = Node(
        package="fm_perception",
        executable="blip2_scene_node.py",
        name="blip2_scene_node",
        output="screen",
        parameters=[{
            "device":                       "cpu",
            "interval_sec":                 5.0,
            "clip_uncertainty_threshold":   0.40,
            "use_int8":                     True,
        }],
        remappings=[
            ("/camera/image_raw", "/exomy/camera/image_raw"),
        ],
    )

    return LaunchDescription([clip_node, blip2_node])
