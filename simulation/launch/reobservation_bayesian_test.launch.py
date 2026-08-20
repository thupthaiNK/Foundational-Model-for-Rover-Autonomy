"""
Purpose: Live Gazebo verification of "A2 re-observation mode" WITH
         bayesian_fusion enabled (SuperMap-inspired log-odds label fusion,
         scoped via grill-thesis 2026-07-19) -- an honest, direct comparison
         against reobservation_test.launch.py's latest-write baseline
         (§4.8.30) on the identical arena, box, and mission, changing only
         one costmap parameter (bayesian_fusion:=true instead of the
         baseline's implicit false). The costmap now accumulates log-odds
         evidence across observations that agree with a cell's last-painted
         label instead of trusting only the newest one; once frontier
         exploration of the box is exhausted, the planner autonomously
         transitions to revisiting the known non-hazard cell with the
         LOWEST recorded confidence, one cell at a time, with no operator
         command at any transition -- identical selection logic to the
         baseline, so any difference in the confidence-delta outcome is
         attributable to the fusion rule alone, not to a different mission.
         Arena: a NEW 3 m x 3 m box straddling the soil/bedrock zone
         boundary (x in [-1.5, 1.5], y in [4.5, 7.5]), spawn at its centre
         (0, 6) -- the same boundary-straddling location as the
         semantic-frontier arena, chosen because borderline (low-
         confidence) classifications concentrate at the class boundary, so
         re-observation has genuinely low-confidence cells to target; a
         uniform-terrain box would exercise the mechanism against a flat
         confidence field and show nothing. Box size follows
         explore_return_home_test.launch.py's proven 3x3 sizing (961 cells
         inclusive) so exhaustion is reachable in one session. All other
         opt-in features are deliberately OFF (semantic_frontier,
         return_home, confidence_aware_painting) -- one new feature per
         official run; the start-cell-hazard risk this arena is known for
         is already covered by grid_with_start_freed, which is
         unconditional in frontier mode and held through both semantic
         official runs.
Inputs:  None (launches everything automatically).
Outputs: /exomy/cmd_vel (from l5_lite_planner_node.py)
         /l5_lite_plan (nav_msgs/Path)
         /l5_lite_frontier_goal (geometry_msgs/PointStamped) -- frontier
             selections only
         /l5_lite_reobserve_goal (geometry_msgs/PointStamped) --
             re-observation selections only
         /traversability_costmap (nav_msgs/OccupancyGrid, -1 = unexplored)
         /traversability_confidence (nav_msgs/OccupancyGrid, 0-100,
             -1 = never observed)
How to run:
    source /opt/ros/humble/setup.bash
    source ros2_ws/install/setup.bash
    ros2 launch simulation/launch/reobservation_bayesian_test.launch.py
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

    # Spawn at the box centre (0, 6), on the soil/bedrock zone boundary
    # (soil_zone x<0, bedrock_zone x>0, both y in [0,12]) -- borderline
    # classifications, and therefore low recorded confidences, concentrate
    # here, giving re-observation genuine targets.
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

    # init_unknown + track_confidence: cells start at -1 ("never assessed")
    # for the frontier explorer, and every paint also records the
    # classifier's confidence (latest-write) for the re-observation mode.
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
                    "track_confidence": True,
                    "bayesian_fusion": True,
                }],
            )
        ],
    )

    # -- frontier mode + re-observation: explore the box until no frontiers
    # remain, then autonomously revisit lowest-confidence cells. No
    # goal_x/goal_y/waypoints; semantic_frontier / return_home /
    # confidence_aware_painting all deliberately left at their false
    # defaults (one new feature per official run).
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
                    "reobserve_mode": True,
                    "frontier_box_x_min": -1.5,
                    "frontier_box_x_max": 1.5,
                    "frontier_box_y_min": 4.5,
                    "frontier_box_y_max": 7.5,
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
