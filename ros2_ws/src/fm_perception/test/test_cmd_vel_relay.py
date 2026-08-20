"""
Purpose: Unit tests for the pure clamp/gate functions used by
         cmd_vel_relay_node.py (wait-time plan item 5). The relay node used
         to forward Nav2/teleop's /cmd_vel straight to /exomy/cmd_vel with
         no speed limiting and no awareness of the E-stop or LiDAR-proximity
         safety signals -- these functions close that gap.
Inputs:  None.
Outputs: pytest results.
How to run:
    cd ros2_ws && colcon build --packages-select fm_perception
    source /opt/ros/humble/setup.bash && source install/setup.bash
    python3 -m pytest src/fm_perception/test/test_cmd_vel_relay.py -v
Project: Onboard Visual Foundation Models for Mars Terrain Perception and Rover Navigation
"""
from fm_perception.cmd_vel_relay_node import clamp_twist_values, should_gate_stop


# ── clamp_twist_values ────────────────────────────────────────────────────

def test_within_limits_passes_through_unchanged():
    lin, ang = clamp_twist_values(0.05, 0.1, max_linear=0.10, max_angular=0.3)
    assert lin == 0.05
    assert ang == 0.1


def test_clamps_linear_velocity_above_max():
    lin, ang = clamp_twist_values(0.50, 0.0, max_linear=0.10, max_angular=0.3)
    assert lin == 0.10
    assert ang == 0.0


def test_clamps_negative_linear_velocity():
    lin, ang = clamp_twist_values(-0.50, 0.0, max_linear=0.10, max_angular=0.3)
    assert lin == -0.10


def test_clamps_angular_velocity_above_max():
    lin, ang = clamp_twist_values(0.0, 1.5, max_linear=0.10, max_angular=0.3)
    assert ang == 0.3


def test_clamps_negative_angular_velocity():
    lin, ang = clamp_twist_values(0.0, -1.5, max_linear=0.10, max_angular=0.3)
    assert ang == -0.3


def test_clamps_both_simultaneously():
    lin, ang = clamp_twist_values(2.0, -2.0, max_linear=0.10, max_angular=0.3)
    assert lin == 0.10
    assert ang == -0.3


# ── should_gate_stop ──────────────────────────────────────────────────────

def test_no_gate_when_both_clear():
    assert should_gate_stop(e_stop_active=False, lidar_stop_active=False) is False


def test_gates_on_e_stop_alone():
    assert should_gate_stop(e_stop_active=True, lidar_stop_active=False) is True


def test_gates_on_lidar_stop_alone():
    assert should_gate_stop(e_stop_active=False, lidar_stop_active=True) is True


def test_gates_when_both_active():
    assert should_gate_stop(e_stop_active=True, lidar_stop_active=True) is True
