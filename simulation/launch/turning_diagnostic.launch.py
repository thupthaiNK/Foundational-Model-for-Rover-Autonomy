"""
Purpose: Minimal Gazebo + ExoMy launch for the turning-unreliability
         diagnostic (§4.8.23) -- no slam_toolbox, no DINOv2, no LiDAR
         consumer, to get the cleanest possible real-time-factor signal
         with minimal confounding process overhead. Distinguishes two
         candidate root causes for the confirmed (3x independent) turning
         problem without touching the shared URDF's physics parameters
         (mu1/mu2, max_wheel_torque, max_wheel_acceleration), which are
         used by every already-validated Gazebo result in this thesis and
         must not be changed casually: (a) Gazebo real-time-factor
         throttling under load -- testable by comparing simulated-time
         elapsed (/clock) against real wall-clock elapsed during a turn
         vs during forward driving, with no physics changes at all; (b) a
         genuine skid-steer torque/friction limitation -- if RTF is
         similar in both cases but rotation still lags, that points away
         from (a) and toward (b).
Inputs:  None (launches everything automatically).
         Optional arg: urdf_file (default exomy.urdf.xacro, the shared file
         every other Gazebo result depends on -- pass
         urdf_file:=exomy_turning_test.urdf.xacro for the higher-torque/
         lower-friction diagnostic variant, §4.8.23 follow-up).
Outputs: /clock (rosgraph_msgs/Clock), /exomy/odom (nav_msgs/Odometry)
How to run:
    source /opt/ros/humble/setup.bash
    source ros2_ws/install/setup.bash
    ros2 launch simulation/launch/turning_diagnostic.launch.py
    ros2 launch simulation/launch/turning_diagnostic.launch.py \
        urdf_file:=exomy_turning_test.urdf.xacro
Project: Onboard Visual Foundation Models for Mars Terrain Perception and Rover Navigation
"""

import os

import xacro
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction, TimerAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _make_nodes(context):
    sim_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    urdf_file = LaunchConfiguration("urdf_file").perform(context)
    urdf_path = os.path.join(sim_dir, "urdf", urdf_file)
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

    return [
        gazebo_server,
        robot_state_publisher,
        joint_state_publisher,
        spawn_robot,
    ]


def generate_launch_description():
    urdf_file_arg = DeclareLaunchArgument(
        "urdf_file", default_value="exomy.urdf.xacro",
        description="URDF file under simulation/urdf/ to load. Default is the shared file every "
                     "other Gazebo result in this thesis depends on -- pass "
                     "urdf_file:=exomy_turning_test.urdf.xacro for the higher-torque/lower-friction "
                     "diagnostic variant (§4.8.23 follow-up), which only this launch file uses."
    )
    return LaunchDescription([
        urdf_file_arg,
        OpaqueFunction(function=_make_nodes),
    ])
