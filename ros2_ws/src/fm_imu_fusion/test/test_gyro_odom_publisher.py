"""
Purpose: Unit tests for the pure yaw-to-quaternion helper shared by
         gyro_odom_publisher_node.py's TF broadcast and its
         nav_msgs/Odometry publish (added 2026-07-26 after a live
         hardware trial found nothing published /exomy/odom at all).
Inputs:  None.
Outputs: pytest results.
How to run:
    cd ros2_ws && colcon build --packages-select fm_imu_fusion
    source /opt/ros/humble/setup.bash && source install/setup.bash
    python3 -m pytest src/fm_imu_fusion/test/test_gyro_odom_publisher.py -v
Project: Onboard Visual Foundation Models for Mars Terrain Perception and Rover Navigation
"""
import math

from fm_imu_fusion.gyro_odom_publisher_node import yaw_to_quaternion


def test_zero_yaw_is_identity_quaternion():
    x, y, z, w = yaw_to_quaternion(0.0)
    assert x == 0.0 and y == 0.0
    assert math.isclose(z, 0.0, abs_tol=1e-9)
    assert math.isclose(w, 1.0, abs_tol=1e-9)


def test_90deg_yaw_quaternion():
    x, y, z, w = yaw_to_quaternion(math.pi / 2)
    assert x == 0.0 and y == 0.0
    assert math.isclose(z, math.sin(math.pi / 4), abs_tol=1e-9)
    assert math.isclose(w, math.cos(math.pi / 4), abs_tol=1e-9)


def test_180deg_yaw_quaternion():
    x, y, z, w = yaw_to_quaternion(math.pi)
    assert x == 0.0 and y == 0.0
    assert math.isclose(z, 1.0, abs_tol=1e-9)
    assert math.isclose(w, 0.0, abs_tol=1e-9)


def test_negative_yaw_quaternion():
    x, y, z, w = yaw_to_quaternion(-math.pi / 2)
    assert x == 0.0 and y == 0.0
    assert math.isclose(z, -math.sin(math.pi / 4), abs_tol=1e-9)
    assert math.isclose(w, math.cos(math.pi / 4), abs_tol=1e-9)


def test_quaternion_is_always_unit_norm():
    for yaw_deg in (0, 10, 45, 90, 137, 180, 270, 359):
        x, y, z, w = yaw_to_quaternion(math.radians(yaw_deg))
        norm = math.sqrt(x * x + y * y + z * z + w * w)
        assert math.isclose(norm, 1.0, abs_tol=1e-9)
