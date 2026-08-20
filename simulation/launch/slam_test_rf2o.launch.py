"""
Purpose: L4 Phase A, odom-free variant -- 4th attempt (Ch4 SS4.8.23 follow-up,
         item 4/4 of the "try every remaining lever" list). Prior odom-free
         attempts all fed slam_toolbox an odom_frame that was either fake
         (self-referential base_frame trick), frozen (static identity TF),
         or rotation-only (gyro integration) -- all 3 hit the same
         "Message Filter dropping message ... queue is full" symptom. This
         attempt uses rf2o_laser_odometry, a genuine scan-matching laser
         odometry package (built from source into ros2_ws/src/, since it is
         not available via apt on this machine -- confirmed again before
         building), which produces a real, continuously-updating
         rf2o_odom->base_link TF derived purely from consecutive /scan
         matches, no encoders/gyro/ground-truth required.
         init_pose_from_topic is explicitly set to "" (empty string) --
         confirmed by reading CLaserOdometry2DNode.h that this fully
         disables the optional ground-truth-seeded startup pose
         (GT_pose_initialized is forced true immediately), so this test
         does not cheat with Gazebo's ground truth at any point, unlike
         Phase A/A2's odom-assisted configuration.
Inputs:  None (launches everything automatically).
         Optional arg: urdf_file (default exomy.urdf.xacro, the shared file
         every other Gazebo result depends on -- pass
         urdf_file:=exomy_odom_free_test.urdf.xacro to disable Gazebo's
         competing odom->base_link TF broadcast, testing a new hypothesis:
         base_link had two concurrently-published parent frames -- Gazebo's
         real odom AND rf2o_odom -- in every prior odom-free attempt, which
         is invalid TF tree topology and may be the actual "queue is full"
         cause, independent of motion-prior content or timing).
Outputs: /map (nav_msgs/OccupancyGrid), /pose (geometry_msgs/PoseWithCovarianceStamped)
         from slam_toolbox; /odom_rf2o (nav_msgs/Odometry) from rf2o; a
         map->rf2o_odom->base_link TF chain.
How to run:
    source /opt/ros/humble/setup.bash
    source ros2_ws/install/setup.bash
    ros2 launch simulation/launch/slam_test_rf2o.launch.py
    ros2 launch simulation/launch/slam_test_rf2o.launch.py \
        urdf_file:=exomy_odom_free_test.urdf.xacro
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
    transform_timeout = LaunchConfiguration("transform_timeout").perform(context)
    scan_queue_size = LaunchConfiguration("scan_queue_size").perform(context)
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

    # -- rf2o_laser_odometry: real scan-matching laser odometry, no ground
    # truth, no encoders, no gyro. Publishes rf2o_odom->base_link TF and
    # /odom_rf2o. init_pose_from_topic="" disables the optional ground-truth
    # seed entirely (confirmed via source read, not assumed).
    rf2o_node = TimerAction(
        period=38.0,
        actions=[
            Node(
                package="rf2o_laser_odometry",
                executable="rf2o_laser_odometry_node",
                name="rf2o_laser_odometry",
                output="screen",
                parameters=[{
                    "laser_scan_topic": "/scan",
                    "odom_topic": "/odom_rf2o",
                    "base_frame_id": "base_link",
                    "odom_frame_id": "rf2o_odom",
                    "publish_tf": True,
                    "init_pose_from_topic": "",
                    "freq": 10.0,
                    "use_sim_time": True,
                }],
            )
        ],
    )

    # -- slam_toolbox (async, mapping mode), rf2o-prior: odom_frame points at
    # rf2o_laser_odometry's output, not Gazebo's real "odom" frame.
    slam_toolbox_node = TimerAction(
        period=42.0,
        actions=[
            Node(
                package="slam_toolbox",
                executable="async_slam_toolbox_node",
                name="slam_toolbox",
                output="screen",
                parameters=[{
                    "odom_frame": "rf2o_odom",
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
                    "transform_timeout": float(transform_timeout),
                    "scan_queue_size": int(scan_queue_size),
                }],
            )
        ],
    )

    return [
        gazebo_server,
        robot_state_publisher,
        joint_state_publisher,
        spawn_robot,
        rf2o_node,
        slam_toolbox_node,
    ]


def generate_launch_description():
    urdf_file_arg = DeclareLaunchArgument(
        "urdf_file", default_value="exomy.urdf.xacro",
        description="URDF file under simulation/urdf/ to load. Default is the shared file "
                     "every other Gazebo result in this thesis depends on -- pass "
                     "urdf_file:=exomy_odom_free_test.urdf.xacro to disable Gazebo's "
                     "competing odom->base_link TF broadcast (publish_odom_tf=false), "
                     "which only this launch file's odom-free tests use."
    )
    transform_timeout_arg = DeclareLaunchArgument(
        "transform_timeout", default_value="0.5",
        description="slam_toolbox's transform_timeout parameter (seconds). Default matches "
                     "the value used in every prior odom-free attempt."
    )
    scan_queue_size_arg = DeclareLaunchArgument(
        "scan_queue_size", default_value="1",
        description="slam_toolbox's scan_queue_size parameter (tf2_ros::MessageFilter queue "
                     "depth for /scan). Default (1) matches the value compiled into the "
                     "apt-installed 2.6.10 binary -- confirmed via declare_parameter() in "
                     "slam_toolbox_common.cpp, contrary to an earlier session's conclusion "
                     "that this was compile-time-only. Pass a larger value (e.g. 50) to test "
                     "whether it resolves the persistent 'Message Filter queue is full' drops."
    )
    return LaunchDescription([
        urdf_file_arg,
        transform_timeout_arg,
        scan_queue_size_arg,
        OpaqueFunction(function=_make_nodes),
    ])
