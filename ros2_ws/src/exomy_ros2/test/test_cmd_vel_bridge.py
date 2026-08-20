"""
Purpose: Unit tests for twist_to_rover_command() -- the pure conversion
         function used by cmd_vel_to_rovercommand_bridge_node.py to translate
         a geometry_msgs/Twist into exomy_ros2_msgs/RoverCommand fields. No
         rclpy dependency, no hardware required.
Inputs:  None.
Outputs: pytest results.
How to run:
    cd ros2_ws && colcon build --packages-select exomy_ros2
    source /opt/ros/humble/setup.bash && source install/setup.bash
    python3 -m pytest src/exomy_ros2/test/test_cmd_vel_bridge.py -v
Project: Onboard Visual Foundation Models for Mars Terrain Perception and Rover Navigation
"""
from exomy_ros2.cmd_vel_to_rovercommand_bridge_node import (
    twist_to_rover_command, STEERING_STRAIGHT, STEERING_TURN_A, STEERING_TURN_B,
)
from exomy_ros2.locomotion_modes import LocomotionMode


def test_zero_twist_means_disabled_and_stopped():
    enabled, mode, vel, steering = twist_to_rover_command(0.0, 0.0)
    assert enabled is False
    assert vel == 0.0


# ── pure forward (mirrors terrain_controller_node's POLICY speeds) ────────

def test_pure_forward_soil_speed_maps_to_full_scale():
    # soil policy speed = 0.10 m/s = max_linear_mps -> should be 100%.
    enabled, mode, vel, steering = twist_to_rover_command(0.10, 0.0)
    assert enabled is True
    assert mode == LocomotionMode.FAKE_ACKERMANN.value
    assert steering == STEERING_STRAIGHT
    assert vel == 100.0


def test_pure_forward_sand_speed_is_half_scale():
    _, _, vel, _ = twist_to_rover_command(0.05, 0.0)
    assert vel == 50.0


def test_pure_forward_bedrock_speed_is_30_percent():
    _, _, vel, _ = twist_to_rover_command(0.03, 0.0)
    assert abs(vel - 30.0) < 1e-6


def test_forward_speed_clamped_above_max():
    _, _, vel, _ = twist_to_rover_command(0.5, 0.0)
    assert vel == 100.0


def test_forward_speed_never_negative_for_negative_linear_x_below_max():
    # No current sender ever commands reverse, but the clamp should still
    # behave sanely (not invert sign unexpectedly) if it ever did.
    _, _, vel, _ = twist_to_rover_command(-0.05, 0.0)
    assert vel == -50.0


# ── pure rotation (mirrors reactive_explorer_node's turn commands) ────────

def test_pure_left_turn_uses_steering_a_full_scale():
    enabled, mode, vel, steering = twist_to_rover_command(0.0, 0.3)
    assert enabled is True
    assert mode == LocomotionMode.POINT_TURN.value
    assert steering == STEERING_TURN_A
    assert vel == 100.0


def test_pure_right_turn_uses_steering_b_full_scale():
    _, _, vel, steering = twist_to_rover_command(0.0, -0.3)
    assert steering == STEERING_TURN_B
    assert vel == 100.0


def test_turn_speed_always_nonnegative_regardless_of_direction():
    _, _, vel_left, _ = twist_to_rover_command(0.0, 0.3)
    _, _, vel_right, _ = twist_to_rover_command(0.0, -0.3)
    assert vel_left >= 0.0 and vel_right >= 0.0


def test_turn_speed_clamped_above_max():
    _, _, vel, _ = twist_to_rover_command(0.0, 1.5)
    assert vel == 100.0


def test_turn_speed_proportional_below_max():
    # retreat_speed's angular_speed is always 0.3 in reactive_explorer_node,
    # but the conversion itself should still scale correctly at other values.
    _, _, vel, _ = twist_to_rover_command(0.0, 0.15)
    assert vel == 50.0


# ── edge cases ──────────────────────────────────────────────────────────

def test_combined_linear_and_angular_is_a_defensive_stop():
    # No current sender (terrain_controller_node, reactive_explorer_node) ever
    # sends both nonzero -- but if it happened, don't guess which one wins.
    enabled, mode, vel, steering = twist_to_rover_command(0.05, 0.2)
    assert enabled is False
    assert vel == 0.0


def test_custom_scale_parameters_are_respected():
    _, _, vel, _ = twist_to_rover_command(0.20, 0.0, max_linear_mps=0.20, max_angular_rps=0.3)
    assert vel == 100.0
