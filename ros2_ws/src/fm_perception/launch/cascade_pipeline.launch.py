"""
Launch file for the dual-process cascade perception pipeline.
Starts cascade_pipeline node (CLIP fast path + SmolVLM slow path).

Usage:
    # With Gazebo running (integration test):
    ros2 launch fm_perception cascade_pipeline.launch.py

    # Standalone (with a live camera or bag file):
    ros2 launch fm_perception cascade_pipeline.launch.py

    # Adjust SmolVLM interval:
    ros2 launch fm_perception cascade_pipeline.launch.py smolvlm_interval:=10.0
"""
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            "device", default_value="cpu",
            description="Device for inference: cpu or cuda"
        ),
        DeclareLaunchArgument(
            "clip_threshold", default_value="0.40",
            description="CLIP confidence threshold below which SmolVLM is triggered"
        ),
        DeclareLaunchArgument(
            "smolvlm_interval", default_value="5.0",
            description="Seconds between periodic SmolVLM slow-path runs"
        ),

        Node(
            package="fm_perception",
            executable="cascade_pipeline.py",
            name="cascade_pipeline",
            output="screen",
            remappings=[
                ("/camera/image_raw", "/exomy/camera/image_raw"),
            ],
            parameters=[{
                "device":               LaunchConfiguration("device"),
                "clip_threshold":       LaunchConfiguration("clip_threshold"),
                "smolvlm_interval_sec": LaunchConfiguration("smolvlm_interval"),
                "smolvlm_model_id":     "HuggingFaceTB/SmolVLM-256M-Instruct",
                "publish_viz":          True,
            }],
        ),
    ])
