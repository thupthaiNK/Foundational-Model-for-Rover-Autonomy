"""
Purpose: Launch the IMU-terrain fusion node alongside the DINOv2 terrain node.
         Provides slope-aware traversability decisions by combining DINOv2 texture
         classification with IMU tilt overrides for inclined terrain.
How to run:
    source /opt/ros/humble/setup.bash
    cd ros2_ws && colcon build --packages-select fm_imu_fusion
    source install/setup.bash
    ros2 launch fm_imu_fusion imu_fusion.launch.py
    # With custom thresholds:
    ros2 launch fm_imu_fusion imu_fusion.launch.py slope_stop_deg:=15.0
    # Verify:
    ros2 topic echo /traversability_fused
Project: Onboard Visual Foundation Models for Mars Terrain Perception and Rover Navigation
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    slope_caution = DeclareLaunchArgument("slope_caution_deg", default_value="10.0",
                                          description="Tilt (degrees) above which SAFE downgrades to CAUTION")
    slope_stop    = DeclareLaunchArgument("slope_stop_deg",    default_value="20.0",
                                          description="Tilt (degrees) above which any terrain becomes STOP")
    conf_thresh   = DeclareLaunchArgument("conf_threshold",    default_value="0.40",
                                          description="DINOv2 confidence below which terrain is uncertain")
    pub_hz        = DeclareLaunchArgument("publish_hz",        default_value="2.0",
                                          description="Output topic publish rate in Hz")

    imu_fusion_node = Node(
        package="fm_imu_fusion",
        executable="imu_slope_fusion_node.py",
        name="imu_slope_fusion_node",
        output="screen",
        parameters=[{
            "slope_caution_deg": LaunchConfiguration("slope_caution_deg"),
            "slope_stop_deg":    LaunchConfiguration("slope_stop_deg"),
            "conf_threshold":    LaunchConfiguration("conf_threshold"),
            "publish_hz":        LaunchConfiguration("publish_hz"),
        }],
    )

    return LaunchDescription([
        slope_caution,
        slope_stop,
        conf_thresh,
        pub_hz,
        imu_fusion_node,
    ])
