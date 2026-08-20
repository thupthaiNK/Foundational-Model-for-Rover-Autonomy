"""
Purpose: Unit tests for the pure-logic helper functions in
         nav2_waypoint_experiment.py (path length, replan counting, hazard
         detection, CSV row construction) — no ROS2 or Gazebo required.
Inputs:  None.
Outputs: pytest results.
How to run:
    source /opt/ros/humble/setup.bash && source ros2_ws/install/setup.bash
    python3 -m pytest experiments/test_nav2_waypoint_experiment.py -v
Project: Onboard Visual Foundation Models for Mars Terrain Perception and Rover Navigation
"""
from nav2_waypoint_experiment import (
    path_length_m, count_replans, path_entered_hazard_zone, trial_result_row,
)


def test_path_length_m_straight_line():
    poses = [(0.0, 0.0), (3.0, 0.0), (3.0, 4.0)]
    assert path_length_m(poses) == 7.0


def test_path_length_m_single_point_is_zero():
    assert path_length_m([(1.0, 1.0)]) == 0.0


def test_path_length_m_empty_is_zero():
    assert path_length_m([]) == 0.0


def test_count_replans_identical_plans_is_zero():
    plan = [(0.0, 0.0), (1.0, 1.0)]
    assert count_replans([plan, plan, plan]) == 0


def test_count_replans_counts_each_change():
    plan_a = [(0.0, 0.0), (1.0, 1.0)]
    plan_b = [(0.0, 0.0), (2.0, 2.0)]
    plan_c = [(0.0, 0.0), (3.0, 3.0)]
    assert count_replans([plan_a, plan_b, plan_c]) == 2


def test_count_replans_within_tolerance_not_counted():
    plan_a = [(0.0, 0.0), (1.0, 1.0)]
    plan_b = [(0.0, 0.0), (1.01, 1.0)]  # 0.01m shift, within 0.05m tolerance
    assert count_replans([plan_a, plan_b], position_tol_m=0.05) == 0


def test_count_replans_single_plan_is_zero():
    assert count_replans([[(0.0, 0.0)]]) == 0


def test_count_replans_empty_is_zero():
    assert count_replans([]) == 0


def test_path_entered_hazard_zone_true_for_rock_cluster_point():
    poses = [(7.5, 1.0), (2.5, -3.5), (-7.5, -9.0)]  # passes through rock_cluster centre
    assert path_entered_hazard_zone(poses) is True


def test_path_entered_hazard_zone_false_for_safe_only_path():
    poses = [(-7.5, 6.0), (-7.5, -6.0)]  # soil_zone -> sand_zone, both safe
    assert path_entered_hazard_zone(poses) is False


def test_trial_result_row_shape():
    row = trial_result_row(
        condition="static", trial=1, success=True,
        path_length_m=12.345, time_to_goal_s=34.5,
        replan_count=0, entered_hazard_zone=False,
    )
    assert row == {
        "condition": "static",
        "trial": 1,
        "success": True,
        "path_length_m": 12.345,
        "time_to_goal_s": 34.5,
        "replan_count": 0,
        "entered_hazard_zone": False,
    }
