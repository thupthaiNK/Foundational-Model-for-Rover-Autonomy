"""
Purpose: Unit tests for Rover.joystickToVelocity()'s proportional-speed
         scaling (FAKE_ACKERMANN and POINT_TURN modes), and for
         Rover.joystickToSteeringAngle()'s return type. Verifies two fixes:
         (1) motor_speeds now scales with the commanded driving_command
         magnitude instead of a hardcoded +/-50.0, so real-hardware speed
         differentiation (soil/sand/bedrock policy speeds) survives through
         to Motors.setDriving(), which already applies PWM proportionally
         and is unchanged/untouched by this fix; (2) motor_angles now
         returns Python floats in every locomotion mode, not ints --
         MotorCommands.msg declares motor_angles as float32[6], and rosidl's
         generated setter raises AssertionError on an int-containing list.
         This was a universal, previously-undiscovered bug: EVERY call to
         joystickToSteeringAngle(), across every locomotion mode and every
         input, returned an int-only list, meaning robot_node.py had never
         actually published a single /motor_commands message without
         crashing -- caught only now because Gazebo experiments never
         exercise this code path (they use /exomy/cmd_vel -> Gazebo's
         diff-drive plugin directly, bypassing robot_node.py entirely), and
         this is the first time the real-hardware chain has been run
         end-to-end (real_hardware_deployment.launch.py, Ch4 thesis notes).
Inputs:  None.
Outputs: pytest results.
How to run:
    cd ros2_ws && colcon build --packages-select exomy_ros2
    source /opt/ros/humble/setup.bash && source install/setup.bash
    python3 -m pytest src/exomy_ros2/test/test_rover_kinematics.py -v
Project: Onboard Visual Foundation Models for Mars Terrain Perception and Rover Navigation
"""
from exomy_ros2.rover import Rover
from exomy_ros2.locomotion_modes import LocomotionMode


def _rover(mode):
    r = Rover()
    r.locomotion_mode = mode
    return r


# ── FAKE_ACKERMANN ──────────────────────────────────────────────────────

def test_fake_ackermann_forward_speed_proportional_to_driving_command():
    r = _rover(LocomotionMode.FAKE_ACKERMANN)
    speeds = r.joystickToVelocity(driving_command=30.0, steering_command=90.0)
    assert speeds == [30.0] * 6


def test_fake_ackermann_full_speed_is_100():
    r = _rover(LocomotionMode.FAKE_ACKERMANN)
    speeds = r.joystickToVelocity(driving_command=100.0, steering_command=90.0)
    assert speeds == [100.0] * 6


def test_fake_ackermann_zero_driving_command_means_stop():
    r = _rover(LocomotionMode.FAKE_ACKERMANN)
    speeds = r.joystickToVelocity(driving_command=0.0, steering_command=90.0)
    assert speeds == [0.0] * 6


def test_fake_ackermann_negative_steering_flips_sign_but_stays_proportional():
    r = _rover(LocomotionMode.FAKE_ACKERMANN)
    speeds = r.joystickToVelocity(driving_command=40.0, steering_command=-10.0)
    assert speeds == [-40.0] * 6


def test_fake_ackermann_reverse_drives_straight_back():
    # Previously unsupported: any driving_command < 0 silently returned
    # [0.0]*6 (no motion at all), needed for reactive_explorer_node's
    # slope retreat (straight reverse, no rotation).
    r = _rover(LocomotionMode.FAKE_ACKERMANN)
    speeds = r.joystickToVelocity(driving_command=-30.0, steering_command=90.0)
    assert speeds == [-30.0] * 6


# ── POINT_TURN ────────────────────────────────────────────────────────────

def test_point_turn_speed_proportional_straight_steering():
    r = _rover(LocomotionMode.POINT_TURN)
    speeds = r.joystickToVelocity(driving_command=60.0, steering_command=0.0)
    assert speeds == [-60.0, 60.0, -60.0, 60.0, -60.0, 60.0]


def test_point_turn_speed_proportional_opposite_steering():
    r = _rover(LocomotionMode.POINT_TURN)
    speeds = r.joystickToVelocity(driving_command=60.0, steering_command=180.0)
    assert speeds == [60.0, -60.0, 60.0, -60.0, 60.0, -60.0]


def test_point_turn_uses_magnitude_regardless_of_driving_command_sign():
    r = _rover(LocomotionMode.POINT_TURN)
    speeds_pos = r.joystickToVelocity(driving_command=45.0, steering_command=0.0)
    speeds_neg = r.joystickToVelocity(driving_command=-45.0, steering_command=0.0)
    assert speeds_pos == speeds_neg == [-45.0, 45.0, -45.0, 45.0, -45.0, 45.0]


def test_point_turn_zero_driving_command_means_stop():
    r = _rover(LocomotionMode.POINT_TURN)
    speeds = r.joystickToVelocity(driving_command=0.0, steering_command=0.0)
    assert speeds == [0.0] * 6


def test_point_turn_ambiguous_steering_band_stays_zero():
    # 85 <= steering <= 95 is the original code's neutral dead-zone between
    # the two turn-direction bands -- must stay all-zero, not guess a direction.
    r = _rover(LocomotionMode.POINT_TURN)
    speeds = r.joystickToVelocity(driving_command=60.0, steering_command=90.0)
    assert speeds == [0.0] * 6


# ── joystickToSteeringAngle() must always return floats (float32[6] ROS2 field) ──

def _assert_all_floats(angles):
    assert len(angles) == 6
    for a in angles:
        assert isinstance(a, float), f"expected float, got {type(a).__name__}: {a!r}"


def test_fake_ackermann_zero_driving_returns_floats():
    r = _rover(LocomotionMode.FAKE_ACKERMANN)
    _assert_all_floats(r.joystickToSteeringAngle(0.0, 0.0))


def test_fake_ackermann_straight_returns_floats():
    r = _rover(LocomotionMode.FAKE_ACKERMANN)
    _assert_all_floats(r.joystickToSteeringAngle(50.0, 90.0))


def test_fake_ackermann_turn_returns_floats():
    r = _rover(LocomotionMode.FAKE_ACKERMANN)
    angles = r.joystickToSteeringAngle(50.0, 0.0)
    _assert_all_floats(angles)
    assert angles == [-45.0, -45.0, 0.0, 0.0, 45.0, 45.0]


def test_ackermann_zero_driving_returns_floats():
    r = _rover(LocomotionMode.ACKERMANN)
    _assert_all_floats(r.joystickToSteeringAngle(0.0, 0.0))


def test_ackermann_turn_returns_floats():
    r = _rover(LocomotionMode.ACKERMANN)
    _assert_all_floats(r.joystickToSteeringAngle(50.0, 0.0))


def test_point_turn_returns_floats():
    r = _rover(LocomotionMode.POINT_TURN)
    angles = r.joystickToSteeringAngle(0.0, 0.0)
    _assert_all_floats(angles)
    assert angles == [45.0, -45.0, 0.0, 0.0, -45.0, 45.0]


def test_crabbing_zero_driving_returns_floats():
    r = _rover(LocomotionMode.CRABBING)
    _assert_all_floats(r.joystickToSteeringAngle(0.0, 0.0))


def test_crabbing_driving_returns_floats():
    r = _rover(LocomotionMode.CRABBING)
    _assert_all_floats(r.joystickToSteeringAngle(50.0, 45.0))


def test_all_modes_all_inputs_return_floats():
    # Broad regression sweep -- this exact matrix crashed robot_node.py's
    # ROS2 message assignment before the fix, in every single case.
    for mode in LocomotionMode:
        r = _rover(mode)
        for dc, sc in [(0.0, 0.0), (50.0, 0.0), (50.0, 90.0), (50.0, 180.0), (-50.0, 0.0)]:
            _assert_all_floats(r.joystickToSteeringAngle(dc, sc))
