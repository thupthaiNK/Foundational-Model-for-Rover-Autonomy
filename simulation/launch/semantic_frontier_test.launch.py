"""
Purpose: Live Gazebo verification of "semantic-biased frontier selection"
         (scoped via grill-thesis 2026-07-18 night) -- the third sliver of
         L6 (Mission Autonomy): frontier exploration-lite's autonomous goal
         selection, but with lexicographic bedrock-adjacency priority
         (science-interest bias, Candela & Wettergreen 2022 framing), so
         the L1 foundation-model classification directly drives an L6
         mission decision. Identical to frontier_exploration_test.launch.py
         except: the exploration box straddles the soil/bedrock boundary
         (x in [-3,3], y in [3,9] -- the be39399 box shifted 6 m east), the
         rover spawns at the new box centre (0, 6) on that boundary, and
         the planner runs with semantic_frontier:=true. The box stays
         entirely in y>0 so no hazard zone is involved (isolates the
         science bias from the safety logic).
Inputs:  None (launches everything automatically).
Outputs: /exomy/cmd_vel (from l5_lite_planner_node.py)
         /l5_lite_plan (nav_msgs/Path)
         /l5_lite_frontier_goal (geometry_msgs/PointStamped)
         /traversability_costmap (nav_msgs/OccupancyGrid, -1 = unexplored)
How to run:
    source /opt/ros/humble/setup.bash
    source ros2_ws/install/setup.bash
    ros2 launch simulation/launch/semantic_frontier_test.launch.py
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
    repo_root = os.path.normpath(os.path.join(sim_dir, ".."))
    default_cache = os.path.join(
        repo_root, "experiments", "results", "feature_cache",
        "dinov2_reg_small_train_1000shot.npz"
    )

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

    # Spawn at the exploration box centre (0, 6): on the soil/bedrock zone
    # boundary (soil_zone x<0, bedrock_zone x>0, both y in [0,12]), open
    # quadrants, far from Q4's hazard-classified rocks. This gives the
    # semantic bias a genuine choice: plain-soil frontiers to the west,
    # bedrock-adjacent frontiers to the east, at comparable distances.
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
                    "-x", "0.0", "-y", "6.0", "-z", "0.15",
                ],
            )
        ],
    )

    # -- slam_toolbox (async, mapping mode), odom-assisted -- the config
    # that worked in Phase A (Run 3, ~0.18m mean error), §4.8.22.
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

    dinov2_node = TimerAction(
        period=44.0,
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
                    "cache_path": default_cache,
                    "n_shot": 1000,
                }],
                remappings=[
                    ("/camera/image_raw", "/exomy/camera/image_raw"),
                ],
            )
        ],
    )

    # init_unknown:=true -- cells start at -1 ("never assessed") so the
    # frontier explorer can distinguish unexplored from explored-as-soil
    # (both would otherwise be 0). Opt-in; every other launch is untouched.
    costmap_node = TimerAction(
        period=46.0,
        actions=[
            Node(
                package="fm_perception",
                executable="live_traversability_costmap_node.py",
                name="live_traversability_costmap_node",
                output="screen",
                parameters=[{
                    "pose_source": "slam",
                    "init_unknown": True,
                }],
            )
        ],
    )

    # -- frontier mode + semantic bias: goals selected autonomously
    # (bedrock-adjacent frontiers first, nearest within the pool, plain
    # nearest as fallback), no goal_x/goal_y/waypoints.
    l5_lite_node = TimerAction(
        period=48.0,
        actions=[
            Node(
                package="fm_perception",
                executable="l5_lite_planner_node.py",
                name="l5_lite_planner_node",
                output="screen",
                parameters=[{
                    "use_sim_time": True,
                    "frontier_mode": True,
                    "semantic_frontier": True,
                    "frontier_box_x_min": -3.0,
                    "frontier_box_x_max": 3.0,
                    "frontier_box_y_min": 3.0,
                    "frontier_box_y_max": 9.0,
                }],
            )
        ],
    )

    return LaunchDescription([
        gazebo_server,
        robot_state_publisher,
        joint_state_publisher,
        spawn_robot,
        slam_toolbox_node,
        dinov2_node,
        costmap_node,
        l5_lite_node,
    ])
