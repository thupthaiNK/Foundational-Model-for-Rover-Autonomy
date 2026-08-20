"""
Purpose: Unit tests for the pure-Python risk-mapping/fusion functions used by
         traversability_score_fusion_node.py -- lidar_range_risk(),
         imu_tilt_risk(), and fuse_traversability_score(). Backlog item 8
         (multi-modal fusion), scoped via grill-thesis 2026-07-17: additive
         extension of H3f's continuous /traversability_score, existing
         independent Bool E-stop triggers (/terrain_controller/stopped,
         /lidar_proximity_stop, /imu_slope_stop) untouched.
Inputs:  None.
Outputs: pytest results.
How to run:
    cd ros2_ws && colcon build --packages-select fm_imu_fusion
    source /opt/ros/humble/setup.bash && source install/setup.bash
    python3 -m pytest src/fm_imu_fusion/test/test_traversability_score_fusion.py -v
Project: Onboard Visual Foundation Models for Mars Terrain Perception and Rover Navigation
"""
from fm_imu_fusion.traversability_score_fusion_node import (
    LIDAR_CLEAR_DISTANCE_M, LIDAR_STOP_DISTANCE_M,
    IMU_CAUTION_DEG, IMU_STOP_DEG,
    lidar_range_risk, imu_tilt_risk, fuse_traversability_score,
)


# -- lidar_range_risk() --------------------------------------------------

def test_lidar_risk_zero_at_clear_distance():
    assert lidar_range_risk(LIDAR_CLEAR_DISTANCE_M) == 0.0


def test_lidar_risk_zero_beyond_clear_distance():
    assert lidar_range_risk(5.0) == 0.0


def test_lidar_risk_max_at_stop_distance():
    assert lidar_range_risk(LIDAR_STOP_DISTANCE_M) == 1.0


def test_lidar_risk_max_inside_stop_distance():
    assert lidar_range_risk(0.1) == 1.0


def test_lidar_risk_midpoint_between_stop_and_clear():
    midpoint = (LIDAR_STOP_DISTANCE_M + LIDAR_CLEAR_DISTANCE_M) / 2.0
    assert lidar_range_risk(midpoint) == 0.5


# -- imu_tilt_risk() -------------------------------------------------------

def test_imu_risk_zero_at_caution_threshold():
    assert imu_tilt_risk(IMU_CAUTION_DEG) == 0.0


def test_imu_risk_zero_below_caution_threshold():
    assert imu_tilt_risk(2.0) == 0.0


def test_imu_risk_max_at_stop_threshold():
    assert imu_tilt_risk(IMU_STOP_DEG) == 1.0


def test_imu_risk_max_above_stop_threshold():
    assert imu_tilt_risk(30.0) == 1.0


def test_imu_risk_midpoint_between_caution_and_stop():
    midpoint = (IMU_CAUTION_DEG + IMU_STOP_DEG) / 2.0
    assert imu_tilt_risk(midpoint) == 0.5


# -- fuse_traversability_score() -------------------------------------------

def test_fuse_returns_dinov2_when_it_is_the_max():
    assert fuse_traversability_score(dinov2_score=0.8, lidar_risk=0.1, imu_risk=0.2) == 0.8


def test_fuse_lidar_dominates_when_higher_than_dinov2():
    assert fuse_traversability_score(dinov2_score=0.1, lidar_risk=0.9, imu_risk=0.0) == 0.9


def test_fuse_imu_dominates_when_higher_than_dinov2():
    assert fuse_traversability_score(dinov2_score=0.1, lidar_risk=0.0, imu_risk=0.7) == 0.7


def test_fuse_all_zero_is_zero():
    assert fuse_traversability_score(dinov2_score=0.0, lidar_risk=0.0, imu_risk=0.0) == 0.0


def test_fuse_bounded_in_unit_interval():
    score = fuse_traversability_score(dinov2_score=1.0, lidar_risk=1.0, imu_risk=1.0)
    assert 0.0 <= score <= 1.0
