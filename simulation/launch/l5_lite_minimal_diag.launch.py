"""
Purpose: Minimal diagnostic launch for "L5-lite" (backlog item, scoped via
         grill-thesis 2026-07-17). Isolates whether astar_planner.py +
         path_follower.py can actually drive the rover to D1's real goal
         under LOW compute load -- no dinov2_terrain_node (measured at
         ~170% CPU, the heaviest single process in the full pipeline).
         Publishes a single, static all-free costmap once (matching this
         session's established precedent of removing one variable at a
         time when isolating a compute-load confound from a genuine logic
         bug, systematic-debugging skill). NOT a reported thesis result on
         its own -- purely a diagnostic step.
Inputs:  None.
Outputs: /exomy/cmd_vel (from l5_lite_planner_node.py)
How to run:
    source /opt/ros/humble/setup.bash
    source ros2_ws/install/setup.bash
    ros2 launch simulation/launch/l5_lite_minimal_diag.launch.py
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
                    "-x", "7.5", "-y", "1.0", "-z", "0.15",  # matches l5_lite_test.launch.py's
                    # D1 START_POSE, for consistency between the two launch files
                ],
            )
        ],
    )

    slam_toolbox_node = TimerAction(
        period=42.0,
        actions=[
            Node(
                package="slam_toolbox",
                executable="async_slam_toolbox_node",
                name="slam_toolbox",
                output="screen",
                parameters=[{
                    "odom_frame": "odom",
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
                    "scan_queue_size": 50,
                }],
            )
        ],
    )

    # Static all-free costmap, published once, TRANSIENT_LOCAL so the
    # planner (a late-joining subscriber) still gets it -- no DINOv2.
    static_costmap_publisher = TimerAction(
        period=44.0,
        actions=[
            ExecuteProcess(
                cmd=["python3", os.path.join(
                    os.path.dirname(sim_dir), "experiments", "l5_lite_static_costmap_publisher.py"
                )],
                output="screen",
            )
        ],
    )

    l5_lite_node = TimerAction(
        period=46.0,
        actions=[
            Node(
                package="fm_perception",
                executable="l5_lite_planner_node.py",
                name="l5_lite_planner_node",
                output="screen",
                parameters=[{"use_sim_time": True}],
            )
        ],
    )

    return LaunchDescription([
        gazebo_server,
        robot_state_publisher,
        joint_state_publisher,
        spawn_robot,
        slam_toolbox_node,
        static_costmap_publisher,
        l5_lite_node,
    ])
