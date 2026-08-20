"""
Launch file for ExoMy ROS2 rover control nodes.
Launches: robot_node (kinematics) + motor_node (hardware, RPi only).
"""
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    dry_run = DeclareLaunchArgument(
        "dry_run", default_value="true",
        description="true = log motor commands only (no hardware); false = drive real motors"
    )

    robot_node = Node(
        package="exomy_ros2",
        executable="robot_node.py",
        name="robot_node",
        output="screen",
    )

    motor_node = Node(
        package="exomy_ros2",
        executable="motor_node.py",
        name="motor_node",
        output="screen",
        parameters=[{
            # PWM pin assignments (Adafruit PCA9685 channels)
            # Adjust these to match your physical wiring
            # As-built PCA9685 channels, verified live 2026-07-23 one channel
            # at a time. Must stay in sync with MotorNode.DEFAULT_PINS.
            "pin_drive_fl": 14, "pin_steer_fl": 8,
            "pin_drive_fr": 2,  "pin_steer_fr": 0,
            "pin_drive_cl": 6,  "pin_steer_cl": 7,
            "pin_drive_cr": 4,  "pin_steer_cr": 5,
            "pin_drive_rl": 10, "pin_steer_rl": 9,
            "pin_drive_rr": 15, "pin_steer_rr": 1,
            # PWM calibration (tune per hardware)
            "steer_pwm_neutral_fl": 307,
            "steer_pwm_neutral_fr": 307,
            "steer_pwm_neutral_cl": 307,
            "steer_pwm_neutral_cr": 307,
            "steer_pwm_neutral_rl": 307,
            "steer_pwm_neutral_rr": 307,
            "steer_pwm_range":  100,
            "drive_pwm_neutral": 307,
            "drive_pwm_range":   200,
        }],
    )

    return LaunchDescription([dry_run, robot_node, motor_node])
