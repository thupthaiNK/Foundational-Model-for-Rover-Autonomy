"""
Purpose: Unit tests for real_stuck_detection_node.py's pure
         rover_command_to_commanded_linear_x() helper, added 2026-07-26
         after a live hardware trial found this node falsely triggered
         "stuck" during reactive_explorer_node.py's legitimate
         SCAN_AVOID_TURN (a rotation, not a stall).
Inputs:  None.
Outputs: pytest results.
How to run:
    cd ros2_ws && colcon build --packages-select fm_perception
    source /opt/ros/humble/setup.bash && source install/setup.bash
    python3 -m pytest src/fm_perception/test/test_real_stuck_detection_node.py -v
Project: Onboard Visual Foundation Models for Mars Terrain Perception and Rover Navigation
"""
from fm_perception.real_stuck_detection_node import (
    LOCOMOTION_FAKE_ACKERMANN,
    rover_command_to_commanded_linear_x,
)

LOCOMOTION_POINT_TURN = 2  # exomy_ros2.locomotion_modes.LocomotionMode.POINT_TURN.value


def test_fake_ackermann_vel_passes_through_as_linear_x():
    assert rover_command_to_commanded_linear_x(
        vel=25.0, locomotion_mode=LOCOMOTION_FAKE_ACKERMANN
    ) == 25.0


def test_point_turn_vel_is_not_linear_motion():
    # This is the exact live-hardware failure mode: reactive_explorer_node's
    # SCAN_AVOID_TURN produces vel=100 (full rotation speed) in POINT_TURN
    # mode, which must NOT be read as "commanded to drive forward at 100".
    assert rover_command_to_commanded_linear_x(
        vel=100.0, locomotion_mode=LOCOMOTION_POINT_TURN
    ) == 0.0


def test_negative_fake_ackermann_vel_passes_through():
    assert rover_command_to_commanded_linear_x(
        vel=-30.0, locomotion_mode=LOCOMOTION_FAKE_ACKERMANN
    ) == -30.0


def test_zero_vel_in_either_mode_is_zero():
    assert rover_command_to_commanded_linear_x(
        vel=0.0, locomotion_mode=LOCOMOTION_FAKE_ACKERMANN
    ) == 0.0
    assert rover_command_to_commanded_linear_x(
        vel=0.0, locomotion_mode=LOCOMOTION_POINT_TURN
    ) == 0.0
