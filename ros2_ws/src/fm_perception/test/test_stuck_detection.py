"""
Purpose: Unit tests for the pure-Python is_stuck() helper and the
         StuckDetectionFSM state machine used by stuck_detection_node.py. No
         rclpy/ROS2 dependency, no hardware or Gazebo required -- the FSM is
         driven with synthetic yaw/odom/commanded-velocity readings and a
         caller-controlled clock, matching test_reactive_explorer.py's style.
Inputs:  None.
Outputs: pytest results.
How to run:
    cd ros2_ws && colcon build --packages-select fm_perception
    source /opt/ros/humble/setup.bash && source install/setup.bash
    python3 -m pytest src/fm_perception/test/test_stuck_detection.py -v
Project: Onboard Visual Foundation Models for Mars Terrain Perception and Rover Navigation
"""
import math

import pytest

from fm_perception.stuck_detection_node import (
    is_stuck, is_rotation_stuck, StuckDetectionFSM,
    STATE_MONITORING, STATE_BOOST_SPEED, STATE_WIGGLE_TURN,
    STATE_WIGGLE_RETRY, STATE_STUCK_FAILSAFE,
    STATE_BOOST_ANGULAR, STATE_ROTATION_RETRY,
    lidar_motion_signal, is_stuck_lidar,
    front_sector_min_range, is_stuck_lidar_front,
    RealStuckDetectionFSM, STATE_RETREAT,
)


# ── is_stuck() ────────────────────────────────────────────────────────────

def test_not_stuck_when_not_commanded_to_move():
    assert is_stuck(commanded_linear_x=0.0, actual_displacement_m=0.0,
                     window_s=4.0) is False


def test_not_stuck_when_displacement_matches_expected():
    # commanded 0.10 m/s for 4s -> expected 0.40m; actual 0.38m is close enough.
    assert is_stuck(commanded_linear_x=0.10, actual_displacement_m=0.38,
                     window_s=4.0) is False


def test_stuck_when_displacement_near_zero_despite_command():
    assert is_stuck(commanded_linear_x=0.10, actual_displacement_m=0.01,
                     window_s=4.0) is True


def test_not_stuck_below_min_commanded_speed():
    # Tiny commanded speed (e.g. residual noise) shouldn't trigger detection.
    assert is_stuck(commanded_linear_x=0.005, actual_displacement_m=0.0,
                     window_s=4.0, min_commanded_speed=0.01) is False


def test_stuck_check_uses_sign_agnostic_magnitude():
    # Reverse commands (negative linear_x) must be checked the same way.
    assert is_stuck(commanded_linear_x=-0.10, actual_displacement_m=0.01,
                     window_s=4.0) is True


# ── is_rotation_stuck() ─────────────────────────────────────────────────────

def test_not_rotation_stuck_when_not_commanded_to_turn():
    assert is_rotation_stuck(commanded_angular_z=0.0, actual_yaw_change_rad=0.0,
                              window_s=4.0) is False


def test_not_rotation_stuck_when_yaw_change_matches_expected():
    # commanded 0.2 rad/s for 4s -> expected 0.8 rad; actual 0.75 is close enough.
    assert is_rotation_stuck(commanded_angular_z=0.2, actual_yaw_change_rad=0.75,
                              window_s=4.0) is False


def test_rotation_stuck_when_yaw_barely_changes_despite_command():
    assert is_rotation_stuck(commanded_angular_z=0.2, actual_yaw_change_rad=0.05,
                              window_s=4.0) is True


def test_not_rotation_stuck_below_min_commanded_angular_speed():
    assert is_rotation_stuck(commanded_angular_z=0.01, actual_yaw_change_rad=0.0,
                              window_s=4.0, min_commanded_angular_speed=0.05) is False


def test_rotation_stuck_check_uses_sign_agnostic_magnitude():
    # Reverse (negative) commanded turns must be checked the same way.
    assert is_rotation_stuck(commanded_angular_z=-0.2, actual_yaw_change_rad=-0.05,
                              window_s=4.0) is True


# ── lidar_motion_signal() ─────────────────────────────────────────────────
# Real-hardware substitute for odom: no wheel encoders or gyro exist on the
# real rover, so "did it actually move" is judged from how much the raw
# /scan ranges changed between two points in time, not from a fitted pose
# (lidar_yaw_from_drive.py's wall-fit position estimate was shown this
# session to be too noisy -- sharpness stuck at 4-8% in every environment
# tried -- for anything needing precision; a coarse "changed a lot vs barely
# changed at all" signal only needs to survive that same noise floor, not
# beat it).

def test_lidar_motion_signal_zero_when_scans_identical():
    ranges_a = [1.0, 2.0, 3.0]
    ranges_b = [1.0, 2.0, 3.0]
    assert lidar_motion_signal(ranges_a, ranges_b) == 0.0


def test_lidar_motion_signal_measures_mean_absolute_difference():
    ranges_a = [1.0, 2.0, 3.0]
    ranges_b = [1.1, 2.2, 2.7]
    # |1.0-1.1| + |2.0-2.2| + |3.0-2.7| = 0.1 + 0.2 + 0.3 = 0.6 -> mean 0.2
    assert lidar_motion_signal(ranges_a, ranges_b) == pytest.approx(0.2)


def test_lidar_motion_signal_ignores_non_finite_pairs():
    ranges_a = [1.0, float("inf"), 3.0, float("nan")]
    ranges_b = [1.5, 2.0, float("inf"), 4.0]
    # Only index 0 (1.0 vs 1.5) is finite in both -> mean abs diff 0.5.
    assert lidar_motion_signal(ranges_a, ranges_b) == pytest.approx(0.5)


def test_lidar_motion_signal_none_when_no_common_valid_beams():
    ranges_a = [float("inf"), float("inf")]
    ranges_b = [1.0, 2.0]
    assert lidar_motion_signal(ranges_a, ranges_b) is None


# ── front_sector_min_range() ──────────────────────────────────────────────
# Live hardware test (2026-07-25) found a real flaw: lidar_motion_signal
# (whole-scan mean abs diff) cannot tell "drove forward" from "spun in place
# against an obstacle" -- rotation shifts every beam's bearing just as much
# as translation shifts ranges, so a rover skidding/yawing in place against
# a wall (never actually progressing) still reads as "moved" and gets
# released back to MONITORING. Tracking only the closest return within a
# narrow sector directly ahead is immune to this: if the rover isn't
# actually getting closer to whatever is blocking it, that value doesn't
# shrink, no matter how much the rover yaws.

import math as _math  # noqa: E402  (local alias, avoids shadowing the module-level `math` import above)


def test_front_sector_min_range_picks_the_closest_return_directly_ahead():
    # angle_min=-pi, angle_increment covers a full circle in 8 steps (45 deg
    # apart): index 4 sits at bearing 0 (straight ahead).
    ranges = [5.0, 5.0, 5.0, 5.0, 1.2, 5.0, 5.0, 5.0]
    angle_min = -_math.pi
    angle_increment = 2 * _math.pi / 8
    assert front_sector_min_range(ranges, angle_min, angle_increment,
                                  sector_half_width_deg=20.0) == pytest.approx(1.2)


def test_front_sector_min_range_ignores_beams_outside_the_sector():
    # A close return at bearing 90 deg (index 2) must not count as "ahead",
    # even though it is the closest return in the whole scan.
    ranges = [5.0, 5.0, 0.3, 5.0, 4.0, 5.0, 5.0, 5.0]
    angle_min = -_math.pi
    angle_increment = 2 * _math.pi / 8
    assert front_sector_min_range(ranges, angle_min, angle_increment,
                                  sector_half_width_deg=20.0) == pytest.approx(4.0)


def test_front_sector_min_range_ignores_non_finite_beams():
    ranges = [float("inf")] * 4 + [1.5] + [float("inf")] * 3
    angle_min = -_math.pi
    angle_increment = 2 * _math.pi / 8
    assert front_sector_min_range(ranges, angle_min, angle_increment,
                                  sector_half_width_deg=20.0) == pytest.approx(1.5)


def test_front_sector_min_range_none_when_sector_has_no_returns():
    ranges = [float("inf")] * 8
    angle_min = -_math.pi
    angle_increment = 2 * _math.pi / 8
    assert front_sector_min_range(ranges, angle_min, angle_increment,
                                  sector_half_width_deg=20.0) is None


# ── is_stuck_lidar_front() ─────────────────────────────────────────────────

def test_not_stuck_front_when_not_commanded_to_move():
    assert is_stuck_lidar_front(commanded_linear_x=0.0, front_range_start=1.0,
                                 front_range_end=1.0) is False


def test_not_stuck_front_when_range_start_or_end_is_none():
    # Nothing ahead within the sector at one end -- insufficient data, must
    # not false-trigger recovery.
    assert is_stuck_lidar_front(commanded_linear_x=25.0, front_range_start=None,
                                 front_range_end=1.0) is False
    assert is_stuck_lidar_front(commanded_linear_x=25.0, front_range_start=1.0,
                                 front_range_end=None) is False


def test_stuck_front_when_range_barely_closes_despite_forward_command():
    # Commanded forward but the obstacle ahead is still ~as far away as
    # before -- the classic "wheels spinning against a wall" case.
    assert is_stuck_lidar_front(commanded_linear_x=25.0, front_range_start=1.20,
                                 front_range_end=1.18,
                                 stuck_progress_threshold_m=0.05) is True


def test_not_stuck_front_when_range_closes_meaningfully():
    assert is_stuck_lidar_front(commanded_linear_x=25.0, front_range_start=1.20,
                                 front_range_end=0.90,
                                 stuck_progress_threshold_m=0.05) is False


def test_not_stuck_front_for_reverse_commands():
    # Reversing legitimately increases the front-sector range (backing away
    # from an obstacle), which "progress = start - end" would read as
    # strongly negative -- i.e. always "stuck". Reverse-stuck detection is
    # out of scope for the front-sector metric (would need the rear sector
    # instead), so this must never trigger, not even a false positive.
    assert is_stuck_lidar_front(commanded_linear_x=-25.0, front_range_start=1.0,
                                 front_range_end=1.5) is False


# ── is_stuck_lidar() ──────────────────────────────────────────────────────

def test_not_stuck_lidar_when_not_commanded_to_move():
    assert is_stuck_lidar(commanded_linear_x=0.0, motion_signal=0.0) is False


def test_not_stuck_lidar_below_min_commanded_speed():
    assert is_stuck_lidar(commanded_linear_x=0.005, motion_signal=0.0,
                           min_commanded_speed=0.01) is False


def test_stuck_lidar_when_motion_signal_below_threshold_despite_command():
    assert is_stuck_lidar(commanded_linear_x=25.0, motion_signal=0.01,
                           stuck_motion_threshold_m=0.05) is True


def test_not_stuck_lidar_when_motion_signal_above_threshold():
    assert is_stuck_lidar(commanded_linear_x=25.0, motion_signal=0.20,
                           stuck_motion_threshold_m=0.05) is False


def test_not_stuck_lidar_when_signal_is_none_insufficient_data():
    # No common valid beams to compare (e.g. scan glitch) -- must not
    # false-trigger recovery just because there was nothing to measure.
    assert is_stuck_lidar(commanded_linear_x=25.0, motion_signal=None) is False


def test_stuck_lidar_check_uses_sign_agnostic_magnitude():
    assert is_stuck_lidar(commanded_linear_x=-25.0, motion_signal=0.01,
                           stuck_motion_threshold_m=0.05) is True


# ── StuckDetectionFSM ─────────────────────────────────────────────────────

def _make_fsm(**overrides):
    params = dict(
        stuck_window_s=4.0,
        stuck_displacement_fraction=0.2,
        min_commanded_speed=0.01,
        boost_max_speed=0.10,
        boost_duration_s=4.0,
        wiggle_angle_deg=20.0,
        angular_speed=0.3,
        angle_tolerance_deg=3.0,
        max_wiggle_attempts=3,
        boost_max_angular_speed=0.3,
        min_commanded_angular_speed=0.05,
    )
    params.update(overrides)
    return StuckDetectionFSM(**params)


def test_stays_monitoring_when_not_commanded_to_move():
    fsm = _make_fsm()
    out = fsm.step(now_s=0.0, yaw=0.0, pos_x=0.0, pos_y=0.0, commanded_linear_x=0.0)
    assert out["state"] == STATE_MONITORING
    assert out["active"] is False


def test_stays_monitoring_before_window_elapses():
    fsm = _make_fsm(stuck_window_s=4.0)
    fsm.step(now_s=0.0, yaw=0.0, pos_x=0.0, pos_y=0.0, commanded_linear_x=0.10)
    out = fsm.step(now_s=2.0, yaw=0.0, pos_x=0.01, pos_y=0.0, commanded_linear_x=0.10)
    assert out["state"] == STATE_MONITORING
    assert out["active"] is False


def test_moving_normally_does_not_trigger_stuck():
    fsm = _make_fsm(stuck_window_s=4.0)
    fsm.step(now_s=0.0, yaw=0.0, pos_x=0.0, pos_y=0.0, commanded_linear_x=0.10)
    # 0.10 m/s * 4s = 0.40m expected; 0.38m actual is well within tolerance.
    out = fsm.step(now_s=4.0, yaw=0.0, pos_x=0.38, pos_y=0.0, commanded_linear_x=0.10)
    assert out["state"] == STATE_MONITORING
    assert out["active"] is False


def test_detects_stuck_and_enters_boost_speed():
    fsm = _make_fsm(stuck_window_s=4.0)
    fsm.step(now_s=0.0, yaw=0.0, pos_x=0.0, pos_y=0.0, commanded_linear_x=0.10)
    out = fsm.step(now_s=4.0, yaw=0.0, pos_x=0.01, pos_y=0.0, commanded_linear_x=0.10)
    assert out["state"] == STATE_BOOST_SPEED
    assert out["active"] is True


def test_boost_speed_commands_ceiling_speed_with_correct_sign():
    fsm = _make_fsm(boost_max_speed=0.10)
    fsm.step(now_s=0.0, yaw=0.0, pos_x=0.0, pos_y=0.0, commanded_linear_x=0.10)
    fsm.step(now_s=4.0, yaw=0.0, pos_x=0.01, pos_y=0.0, commanded_linear_x=0.10)
    assert fsm.state == STATE_BOOST_SPEED
    out = fsm.step(now_s=4.5, yaw=0.0, pos_x=0.02, pos_y=0.0, commanded_linear_x=0.10)
    assert out["linear_x"] == 0.10
    assert out["angular_z"] == 0.0


def test_boost_speed_resolves_when_displacement_resumes():
    fsm = _make_fsm(boost_duration_s=4.0)
    fsm.step(now_s=0.0, yaw=0.0, pos_x=0.0, pos_y=0.0, commanded_linear_x=0.10)
    fsm.step(now_s=4.0, yaw=0.0, pos_x=0.01, pos_y=0.0, commanded_linear_x=0.10)
    assert fsm.state == STATE_BOOST_SPEED
    # Boost worked -- rover freed itself and travelled close to the expected distance.
    out = fsm.step(now_s=8.0, yaw=0.0, pos_x=0.41, pos_y=0.0, commanded_linear_x=0.10)
    assert out["state"] == STATE_MONITORING
    assert out["active"] is False


def test_boost_speed_still_stuck_enters_wiggle_turn():
    fsm = _make_fsm(boost_duration_s=4.0, wiggle_angle_deg=20.0)
    fsm.step(now_s=0.0, yaw=0.0, pos_x=0.0, pos_y=0.0, commanded_linear_x=0.10)
    fsm.step(now_s=4.0, yaw=0.0, pos_x=0.01, pos_y=0.0, commanded_linear_x=0.10)
    assert fsm.state == STATE_BOOST_SPEED
    out = fsm.step(now_s=8.0, yaw=0.0, pos_x=0.02, pos_y=0.0, commanded_linear_x=0.10)
    assert out["state"] == STATE_WIGGLE_TURN


def test_wiggle_turn_commands_angular_until_target_then_moves_to_retry():
    fsm = _make_fsm(wiggle_angle_deg=20.0, angular_speed=0.3)
    fsm.step(now_s=0.0, yaw=0.0, pos_x=0.0, pos_y=0.0, commanded_linear_x=0.10)
    fsm.step(now_s=4.0, yaw=0.0, pos_x=0.01, pos_y=0.0, commanded_linear_x=0.10)
    fsm.step(now_s=8.0, yaw=0.0, pos_x=0.02, pos_y=0.0, commanded_linear_x=0.10)
    assert fsm.state == STATE_WIGGLE_TURN

    out = fsm.step(now_s=8.1, yaw=math.radians(5), pos_x=0.02, pos_y=0.0, commanded_linear_x=0.0)
    assert out["state"] == STATE_WIGGLE_TURN
    assert out["angular_z"] != 0.0
    assert out["linear_x"] == 0.0

    out = fsm.step(now_s=8.2, yaw=math.radians(19), pos_x=0.02, pos_y=0.0, commanded_linear_x=0.0)
    assert out["state"] == STATE_WIGGLE_RETRY


def _drive_to_wiggle_retry(fsm):
    fsm.step(now_s=0.0, yaw=0.0, pos_x=0.0, pos_y=0.0, commanded_linear_x=0.10)
    fsm.step(now_s=4.0, yaw=0.0, pos_x=0.01, pos_y=0.0, commanded_linear_x=0.10)
    fsm.step(now_s=8.0, yaw=0.0, pos_x=0.02, pos_y=0.0, commanded_linear_x=0.10)
    assert fsm.state == STATE_WIGGLE_TURN
    target = fsm._target_yaw
    out = fsm.step(now_s=8.5, yaw=target, pos_x=0.02, pos_y=0.0, commanded_linear_x=0.0)
    assert out["state"] == STATE_WIGGLE_RETRY
    return target


def test_wiggle_retry_resolves_when_displacement_resumes():
    fsm = _make_fsm()
    target_yaw = _drive_to_wiggle_retry(fsm)
    out = fsm.step(now_s=12.5, yaw=target_yaw, pos_x=0.41, pos_y=0.0, commanded_linear_x=0.0)
    assert out["state"] == STATE_MONITORING
    assert out["active"] is False


def test_wiggle_retry_still_stuck_alternates_direction_and_retries():
    fsm = _make_fsm()
    target_yaw = _drive_to_wiggle_retry(fsm)
    first_direction = fsm._wiggle_direction
    out = fsm.step(now_s=12.5, yaw=target_yaw, pos_x=0.02, pos_y=0.0, commanded_linear_x=0.0)
    assert out["state"] == STATE_WIGGLE_TURN
    assert fsm._wiggle_direction == -first_direction
    assert fsm._wiggle_attempt_count == 1


def test_stuck_cap_triggers_failsafe_immediately_when_cap_is_one():
    # max_wiggle_attempts=1 -- the first failed retry already hits the cap.
    fsm = _make_fsm(max_wiggle_attempts=1)
    target_yaw = _drive_to_wiggle_retry(fsm)
    out = fsm.step(now_s=12.5, yaw=target_yaw, pos_x=0.02, pos_y=0.0, commanded_linear_x=0.0)
    assert out["state"] == STATE_STUCK_FAILSAFE
    assert out["failsafe"] is True


def test_stuck_cap_allows_configured_number_of_attempts_before_failsafe():
    fsm = _make_fsm(max_wiggle_attempts=2)
    target_yaw = _drive_to_wiggle_retry(fsm)
    # Attempt 1 fails -> cap=2 not yet reached, retries with flipped direction.
    fsm.step(now_s=12.5, yaw=target_yaw, pos_x=0.02, pos_y=0.0, commanded_linear_x=0.0)
    assert fsm.state == STATE_WIGGLE_TURN
    target_yaw_2 = fsm._target_yaw
    fsm.step(now_s=13.0, yaw=target_yaw_2, pos_x=0.02, pos_y=0.0, commanded_linear_x=0.0)
    assert fsm.state == STATE_WIGGLE_RETRY
    # Attempt 2 also fails, now cap=2 reached -> FAILSAFE.
    out = fsm.step(now_s=17.0, yaw=target_yaw_2, pos_x=0.03, pos_y=0.0, commanded_linear_x=0.0)
    assert out["state"] == STATE_STUCK_FAILSAFE
    assert out["failsafe"] is True


def test_failsafe_is_terminal_ignores_further_input():
    fsm = _make_fsm(max_wiggle_attempts=0)
    _drive_to_wiggle_retry(fsm)
    out = fsm.step(now_s=12.5, yaw=0.0, pos_x=0.02, pos_y=0.0, commanded_linear_x=0.0)
    assert out["state"] == STATE_STUCK_FAILSAFE

    out = fsm.step(now_s=100.0, yaw=0.0, pos_x=5.0, pos_y=5.0, commanded_linear_x=0.10)
    assert out["state"] == STATE_STUCK_FAILSAFE
    assert out["linear_x"] == 0.0 and out["angular_z"] == 0.0


# ── Rotation-stuck FSM (pure in-place turns, e.g. reactive_explorer_node's own
#    check-left/check-right turns getting stuck -- linear_x stays at 0 the
#    whole time, so the linear-only checks above never even see this case) ──

def test_pure_rotation_command_starts_monitoring_window():
    fsm = _make_fsm()
    out = fsm.step(now_s=0.0, yaw=0.0, pos_x=0.0, pos_y=0.0,
                    commanded_linear_x=0.0, commanded_angular_z=0.3)
    assert out["state"] == STATE_MONITORING
    assert out["active"] is False


def test_rotation_moving_normally_does_not_trigger_stuck():
    fsm = _make_fsm()
    fsm.step(now_s=0.0, yaw=0.0, pos_x=0.0, pos_y=0.0,
              commanded_linear_x=0.0, commanded_angular_z=0.3)
    # 0.3 rad/s * 4s = 1.2 rad expected; 1.15 rad actual is well within tolerance.
    out = fsm.step(now_s=4.0, yaw=1.15, pos_x=0.0, pos_y=0.0,
                    commanded_linear_x=0.0, commanded_angular_z=0.3)
    assert out["state"] == STATE_MONITORING
    assert out["active"] is False


def test_detects_rotation_stuck_and_enters_boost_angular():
    fsm = _make_fsm()
    fsm.step(now_s=0.0, yaw=0.0, pos_x=0.0, pos_y=0.0,
              commanded_linear_x=0.0, commanded_angular_z=0.3)
    out = fsm.step(now_s=4.0, yaw=0.02, pos_x=0.0, pos_y=0.0,
                    commanded_linear_x=0.0, commanded_angular_z=0.3)
    assert out["state"] == STATE_BOOST_ANGULAR
    assert out["active"] is True


def test_boost_angular_commands_ceiling_speed_with_correct_sign():
    fsm = _make_fsm(boost_max_angular_speed=0.3)
    fsm.step(now_s=0.0, yaw=0.0, pos_x=0.0, pos_y=0.0,
              commanded_linear_x=0.0, commanded_angular_z=0.3)
    fsm.step(now_s=4.0, yaw=0.02, pos_x=0.0, pos_y=0.0,
              commanded_linear_x=0.0, commanded_angular_z=0.3)
    assert fsm.state == STATE_BOOST_ANGULAR
    out = fsm.step(now_s=4.5, yaw=0.02, pos_x=0.0, pos_y=0.0,
                    commanded_linear_x=0.0, commanded_angular_z=0.3)
    assert out["angular_z"] == 0.3
    assert out["linear_x"] == 0.0


def test_boost_angular_resolves_when_yaw_change_resumes():
    fsm = _make_fsm(boost_duration_s=4.0)
    fsm.step(now_s=0.0, yaw=0.0, pos_x=0.0, pos_y=0.0,
              commanded_linear_x=0.0, commanded_angular_z=0.3)
    fsm.step(now_s=4.0, yaw=0.02, pos_x=0.0, pos_y=0.0,
              commanded_linear_x=0.0, commanded_angular_z=0.3)
    assert fsm.state == STATE_BOOST_ANGULAR
    # Boost worked -- rover freed itself and rotated close to the expected amount.
    out = fsm.step(now_s=8.0, yaw=0.02 + 1.2, pos_x=0.0, pos_y=0.0,
                    commanded_linear_x=0.0, commanded_angular_z=0.0)
    assert out["state"] == STATE_MONITORING
    assert out["active"] is False


def test_boost_angular_still_stuck_enters_rotation_retry():
    fsm = _make_fsm(boost_duration_s=4.0)
    fsm.step(now_s=0.0, yaw=0.0, pos_x=0.0, pos_y=0.0,
              commanded_linear_x=0.0, commanded_angular_z=0.3)
    fsm.step(now_s=4.0, yaw=0.02, pos_x=0.0, pos_y=0.0,
              commanded_linear_x=0.0, commanded_angular_z=0.3)
    assert fsm.state == STATE_BOOST_ANGULAR
    out = fsm.step(now_s=8.0, yaw=0.04, pos_x=0.0, pos_y=0.0,
                    commanded_linear_x=0.0, commanded_angular_z=0.0)
    assert out["state"] == STATE_ROTATION_RETRY
    # First retry alternates away from the original (positive) boost direction.
    assert fsm._wiggle_direction == -1


def _drive_to_rotation_retry(fsm):
    fsm.step(now_s=0.0, yaw=0.0, pos_x=0.0, pos_y=0.0,
              commanded_linear_x=0.0, commanded_angular_z=0.3)
    fsm.step(now_s=4.0, yaw=0.02, pos_x=0.0, pos_y=0.0,
              commanded_linear_x=0.0, commanded_angular_z=0.3)
    assert fsm.state == STATE_BOOST_ANGULAR
    out = fsm.step(now_s=8.0, yaw=0.04, pos_x=0.0, pos_y=0.0,
                    commanded_linear_x=0.0, commanded_angular_z=0.0)
    assert out["state"] == STATE_ROTATION_RETRY
    return fsm._retry_start_yaw


def test_rotation_retry_commands_flipped_direction():
    fsm = _make_fsm(boost_max_angular_speed=0.3)
    _drive_to_rotation_retry(fsm)
    out = fsm.step(now_s=8.1, yaw=0.04, pos_x=0.0, pos_y=0.0,
                    commanded_linear_x=0.0, commanded_angular_z=0.0)
    assert out["state"] == STATE_ROTATION_RETRY
    assert out["angular_z"] == -0.3


def test_rotation_retry_resolves_when_yaw_change_resumes():
    fsm = _make_fsm()
    retry_start_yaw = _drive_to_rotation_retry(fsm)
    out = fsm.step(now_s=12.0, yaw=retry_start_yaw - 1.2, pos_x=0.0, pos_y=0.0,
                    commanded_linear_x=0.0, commanded_angular_z=0.0)
    assert out["state"] == STATE_MONITORING
    assert out["active"] is False


def test_rotation_retry_still_stuck_alternates_direction_and_retries():
    fsm = _make_fsm()
    retry_start_yaw = _drive_to_rotation_retry(fsm)
    out = fsm.step(now_s=12.0, yaw=retry_start_yaw + 0.01, pos_x=0.0, pos_y=0.0,
                    commanded_linear_x=0.0, commanded_angular_z=0.0)
    assert out["state"] == STATE_ROTATION_RETRY
    assert fsm._wiggle_direction == 1
    assert fsm._wiggle_attempt_count == 1


def test_rotation_stuck_cap_triggers_failsafe_immediately_when_cap_is_one():
    fsm = _make_fsm(max_wiggle_attempts=1)
    retry_start_yaw = _drive_to_rotation_retry(fsm)
    out = fsm.step(now_s=12.0, yaw=retry_start_yaw, pos_x=0.0, pos_y=0.0,
                    commanded_linear_x=0.0, commanded_angular_z=0.0)
    assert out["state"] == STATE_STUCK_FAILSAFE
    assert out["failsafe"] is True


def test_linear_stuck_takes_priority_over_rotation_when_both_commanded():
    # An arc command (both linear_x and angular_z nonzero) that fails to
    # displace is treated as a linear-stuck event, not a rotation-stuck one --
    # matches the existing recovery philosophy (translation is the primary
    # failure mode; BOOST_SPEED already existed and is checked first).
    fsm = _make_fsm()
    fsm.step(now_s=0.0, yaw=0.0, pos_x=0.0, pos_y=0.0,
              commanded_linear_x=0.10, commanded_angular_z=0.3)
    out = fsm.step(now_s=4.0, yaw=0.02, pos_x=0.01, pos_y=0.0,
                    commanded_linear_x=0.10, commanded_angular_z=0.3)
    assert out["state"] == STATE_BOOST_SPEED


# ── RealStuckDetectionFSM ─────────────────────────────────────────────────
# Real hardware has no wheel encoders and no gyro, so unlike StuckDetectionFSM
# above (Gazebo-verified, driven by odom pos_x/pos_y/yaw), this variant is
# driven by a single front_sector_min_range scalar (computed by the caller
# from the raw /scan) via is_stuck_lidar_front, not the whole-scan
# lidar_motion_signal/is_stuck_lidar pair -- live testing on the real rover
# (2026-07-25) found the whole-scan diff can't tell "drove forward" from
# "yawed in place against an obstacle". Only handles the linear-stuck,
# forward-command case -- there is no sensor to confirm a WIGGLE_TURN
# actually turned the rover, so that state runs for a fixed duration
# (angle / angular_speed) rather than checking a target yaw, and reverse
# commands are out of scope for the front-sector metric.

FRONT_RANGE = 1.2


def _make_real_fsm(**overrides):
    params = dict(
        stuck_window_s=4.0,
        stuck_progress_threshold_m=0.05,
        min_commanded_speed=1.0,
        boost_max_speed=25.0,
        boost_duration_s=4.0,
        wiggle_angle_deg=20.0,
        angular_speed=0.3,
        max_wiggle_attempts=4,
    )
    params.update(overrides)
    return RealStuckDetectionFSM(**params)


def test_real_fsm_default_max_wiggle_attempts_is_four():
    # 4 (not 3) gives an even left/right wiggle split: direction sequence is
    # +1,-1,+1,-1 across 4 attempts, vs 3's uneven +1,-1,+1 (one side tried
    # twice, the other once). Constructed with no override, matching how the
    # real node builds it from its own parameter default.
    fsm = RealStuckDetectionFSM()
    assert fsm.max_wiggle_attempts == 4


def test_real_fsm_stays_monitoring_when_not_commanded_to_move():
    fsm = _make_real_fsm()
    out = fsm.step(now_s=0.0, front_range=FRONT_RANGE, commanded_linear_x=0.0)
    assert out["state"] == STATE_MONITORING
    assert out["active"] is False


def test_real_fsm_stays_monitoring_before_window_elapses():
    fsm = _make_real_fsm()
    fsm.step(now_s=0.0, front_range=FRONT_RANGE, commanded_linear_x=25.0)
    out = fsm.step(now_s=2.0, front_range=FRONT_RANGE, commanded_linear_x=25.0)
    assert out["state"] == STATE_MONITORING
    assert out["active"] is False


def test_real_fsm_moving_normally_does_not_trigger_stuck():
    fsm = _make_real_fsm()
    fsm.step(now_s=0.0, front_range=FRONT_RANGE, commanded_linear_x=25.0)
    # Front range closed by 0.4m over 4s at 0.10 m/s -- consistent with real
    # forward motion toward whatever is ahead.
    out = fsm.step(now_s=4.0, front_range=FRONT_RANGE - 0.4,
                    commanded_linear_x=25.0)
    assert out["state"] == STATE_MONITORING
    assert out["active"] is False


def test_real_fsm_detects_stuck_and_enters_boost_speed():
    fsm = _make_real_fsm()
    fsm.step(now_s=0.0, front_range=FRONT_RANGE, commanded_linear_x=25.0)
    out = fsm.step(now_s=4.0, front_range=FRONT_RANGE, commanded_linear_x=25.0)
    assert out["state"] == STATE_BOOST_SPEED
    assert out["active"] is True


def test_real_fsm_boost_speed_commands_the_ceiling_speed():
    fsm = _make_real_fsm(boost_max_speed=25.0)
    fsm.step(now_s=0.0, front_range=FRONT_RANGE, commanded_linear_x=25.0)
    fsm.step(now_s=4.0, front_range=FRONT_RANGE, commanded_linear_x=25.0)
    out = fsm.step(now_s=4.5, front_range=FRONT_RANGE, commanded_linear_x=25.0)
    assert out["state"] == STATE_BOOST_SPEED
    assert out["linear_x"] == pytest.approx(25.0)


def test_real_fsm_boost_speed_resolves_when_front_range_closes():
    fsm = _make_real_fsm()
    fsm.step(now_s=0.0, front_range=FRONT_RANGE, commanded_linear_x=25.0)
    fsm.step(now_s=4.0, front_range=FRONT_RANGE, commanded_linear_x=25.0)
    out = fsm.step(now_s=8.0, front_range=FRONT_RANGE - 0.3,
                    commanded_linear_x=25.0)
    assert out["state"] == STATE_MONITORING
    assert out["active"] is False


def _drive_real_fsm_to_wiggle_retry(fsm, start=0.0):
    fsm.step(now_s=start, front_range=FRONT_RANGE, commanded_linear_x=25.0)
    fsm.step(now_s=start + 4.0, front_range=FRONT_RANGE, commanded_linear_x=25.0)
    fsm.step(now_s=start + 4.5, front_range=FRONT_RANGE, commanded_linear_x=25.0)
    # boost_duration_s elapses with the front range unchanged -> still stuck
    # -> WIGGLE_TURN.
    out = fsm.step(now_s=start + 8.5, front_range=FRONT_RANGE, commanded_linear_x=25.0)
    assert out["state"] == STATE_WIGGLE_TURN
    return start + 8.5


def test_real_fsm_boost_speed_still_stuck_enters_wiggle_turn():
    fsm = _make_real_fsm()
    _drive_real_fsm_to_wiggle_retry(fsm)


def test_real_fsm_wiggle_turn_runs_fixed_duration_then_moves_to_retry():
    fsm = _make_real_fsm(wiggle_angle_deg=20.0, angular_speed=0.3)
    t = _drive_real_fsm_to_wiggle_retry(fsm)
    # duration = radians(20)/0.3 ~= 1.16s -- not yet elapsed.
    out = fsm.step(now_s=t + 0.5, front_range=FRONT_RANGE, commanded_linear_x=25.0)
    assert out["state"] == STATE_WIGGLE_TURN
    assert out["angular_z"] != 0.0
    out = fsm.step(now_s=t + 2.0, front_range=FRONT_RANGE, commanded_linear_x=25.0)
    assert out["state"] == STATE_WIGGLE_RETRY


def test_real_fsm_wiggle_retry_resolves_when_front_range_closes():
    fsm = _make_real_fsm()
    t = _drive_real_fsm_to_wiggle_retry(fsm)
    fsm.step(now_s=t + 2.0, front_range=FRONT_RANGE, commanded_linear_x=25.0)
    out = fsm.step(now_s=t + 6.0, front_range=FRONT_RANGE - 0.3,
                    commanded_linear_x=25.0)
    assert out["state"] == STATE_MONITORING
    assert out["active"] is False


def test_real_fsm_wiggle_retry_still_stuck_alternates_and_retries():
    fsm = _make_real_fsm()
    t = _drive_real_fsm_to_wiggle_retry(fsm)
    fsm.step(now_s=t + 2.0, front_range=FRONT_RANGE, commanded_linear_x=25.0)
    out = fsm.step(now_s=t + 6.0, front_range=FRONT_RANGE, commanded_linear_x=25.0)
    assert out["state"] == STATE_WIGGLE_TURN


# ── RETREAT-before-FAILSAFE (2026-07-25) ───────────────────────────────────
# Added because live testing sometimes ended with the rover's front wheels
# stopped mid-climb on a wall or sand mound -- a resting pose that isn't a
# safe one to leave the rover in unattended. Recovery giving up should still
# back the rover a short distance away from whatever it was fighting before
# finally going terminal, not just cut power in whatever pose it was stuck.

def test_real_fsm_exhausting_attempts_enters_retreat_not_failsafe_directly():
    fsm = _make_real_fsm(max_wiggle_attempts=1)
    t = _drive_real_fsm_to_wiggle_retry(fsm)
    fsm.step(now_s=t + 2.0, front_range=FRONT_RANGE, commanded_linear_x=25.0)
    out = fsm.step(now_s=t + 6.0, front_range=FRONT_RANGE, commanded_linear_x=25.0)
    assert out["state"] == STATE_RETREAT
    assert out["failsafe"] is False


def test_real_fsm_retreat_commands_negative_speed():
    fsm = _make_real_fsm(max_wiggle_attempts=1, retreat_speed=20.0)
    t = _drive_real_fsm_to_wiggle_retry(fsm)
    fsm.step(now_s=t + 2.0, front_range=FRONT_RANGE, commanded_linear_x=25.0)
    fsm.step(now_s=t + 6.0, front_range=FRONT_RANGE, commanded_linear_x=25.0)
    out = fsm.step(now_s=t + 6.5, front_range=FRONT_RANGE, commanded_linear_x=25.0)
    assert out["state"] == STATE_RETREAT
    assert out["linear_x"] == pytest.approx(-20.0)
    assert out["angular_z"] == 0.0


def test_real_fsm_retreat_then_becomes_failsafe_and_stops():
    fsm = _make_real_fsm(max_wiggle_attempts=1, retreat_duration_s=2.0)
    t = _drive_real_fsm_to_wiggle_retry(fsm)
    fsm.step(now_s=t + 2.0, front_range=FRONT_RANGE, commanded_linear_x=25.0)
    fsm.step(now_s=t + 6.0, front_range=FRONT_RANGE, commanded_linear_x=25.0)
    out = fsm.step(now_s=t + 8.5, front_range=FRONT_RANGE, commanded_linear_x=25.0)
    assert out["state"] == STATE_STUCK_FAILSAFE
    assert out["failsafe"] is True
    assert out["linear_x"] == 0.0
    assert out["angular_z"] == 0.0


def test_real_fsm_failsafe_is_terminal():
    fsm = _make_real_fsm(max_wiggle_attempts=1, retreat_duration_s=2.0)
    t = _drive_real_fsm_to_wiggle_retry(fsm)
    fsm.step(now_s=t + 2.0, front_range=FRONT_RANGE, commanded_linear_x=25.0)
    fsm.step(now_s=t + 6.0, front_range=FRONT_RANGE, commanded_linear_x=25.0)
    fsm.step(now_s=t + 8.5, front_range=FRONT_RANGE, commanded_linear_x=25.0)
    out = fsm.step(now_s=t + 20.0, front_range=FRONT_RANGE - 5.0,
                    commanded_linear_x=25.0)
    assert out["state"] == STATE_STUCK_FAILSAFE
    assert out["linear_x"] == 0.0
    assert out["angular_z"] == 0.0


# ── retreat gated on rear-sector LiDAR clearance (2026-08-02) ──────────────
# Added after a pre-hardware-run audit found the retreat ran with zero
# sensing at all -- WIGGLE_TURN/WIGGLE_RETRY on real hardware drive the
# rover forward on a hard steering arc (FAKE_ACKERMANN turn bands), not a
# true in-place rotation, so the heading/position before a retreat cannot be
# assumed to be "back the way it came". These tests use the default
# retreat_min_clearance_m=0.30 unless overridden.

def test_real_fsm_no_rear_return_treated_as_clear_and_retreats():
    # rear_range=None means the sector had no finite return at all (nothing
    # within LiDAR range behind the rover) -- same "no return = clear"
    # convention used everywhere else this LiDAR is read in this stack.
    fsm = _make_real_fsm(max_wiggle_attempts=1)
    t = _drive_real_fsm_to_wiggle_retry(fsm)
    fsm.step(now_s=t + 2.0, front_range=FRONT_RANGE, commanded_linear_x=25.0)
    out = fsm.step(now_s=t + 6.0, front_range=FRONT_RANGE,
                    commanded_linear_x=25.0, rear_range=None)
    assert out["state"] == STATE_RETREAT
    assert out["linear_x"] == pytest.approx(-20.0)


def test_real_fsm_rear_clear_at_entry_retreats_normally():
    fsm = _make_real_fsm(max_wiggle_attempts=1, retreat_min_clearance_m=0.30)
    t = _drive_real_fsm_to_wiggle_retry(fsm)
    fsm.step(now_s=t + 2.0, front_range=FRONT_RANGE, commanded_linear_x=25.0)
    out = fsm.step(now_s=t + 6.0, front_range=FRONT_RANGE,
                    commanded_linear_x=25.0, rear_range=1.0)
    assert out["state"] == STATE_RETREAT
    assert out["linear_x"] == pytest.approx(-20.0)


def test_real_fsm_rear_blocked_at_entry_fails_safe_instead_of_reversing():
    # Every wiggle attempt failed AND the rear is blocked -- must not reverse
    # blind into it.
    fsm = _make_real_fsm(max_wiggle_attempts=1, retreat_min_clearance_m=0.30)
    t = _drive_real_fsm_to_wiggle_retry(fsm)
    fsm.step(now_s=t + 2.0, front_range=FRONT_RANGE, commanded_linear_x=25.0)
    out = fsm.step(now_s=t + 6.0, front_range=FRONT_RANGE,
                    commanded_linear_x=25.0, rear_range=0.15)
    assert out["state"] == STATE_STUCK_FAILSAFE
    assert out["failsafe"] is True
    assert out["linear_x"] == 0.0
    assert out["angular_z"] == 0.0


def test_real_fsm_rear_clears_exactly_at_threshold_retreats():
    fsm = _make_real_fsm(max_wiggle_attempts=1, retreat_min_clearance_m=0.30)
    t = _drive_real_fsm_to_wiggle_retry(fsm)
    fsm.step(now_s=t + 2.0, front_range=FRONT_RANGE, commanded_linear_x=25.0)
    out = fsm.step(now_s=t + 6.0, front_range=FRONT_RANGE,
                    commanded_linear_x=25.0, rear_range=0.30)
    assert out["state"] == STATE_RETREAT


def test_real_fsm_rear_obstacle_appears_mid_retreat_stops_early():
    # Rear was clear at entry but something closes in during the reverse
    # move (e.g. the rover itself backing toward it) -- must stop before
    # retreat_duration_s elapses, not run the fixed duration regardless.
    fsm = _make_real_fsm(max_wiggle_attempts=1, retreat_duration_s=2.0,
                          retreat_min_clearance_m=0.30)
    t = _drive_real_fsm_to_wiggle_retry(fsm)
    fsm.step(now_s=t + 2.0, front_range=FRONT_RANGE, commanded_linear_x=25.0)
    out = fsm.step(now_s=t + 6.0, front_range=FRONT_RANGE,
                    commanded_linear_x=25.0, rear_range=1.0)
    assert out["state"] == STATE_RETREAT
    # Mid-retreat tick, well before retreat_duration_s elapses, rear closes.
    out = fsm.step(now_s=t + 6.5, front_range=FRONT_RANGE,
                    commanded_linear_x=25.0, rear_range=0.10)
    assert out["state"] == STATE_STUCK_FAILSAFE
    assert out["linear_x"] == 0.0


def _scan_with_mast_return_behind(n_beams=36, background_m=1.0, mast_m=0.072):
    # n_beams=36 -> 10 deg apart, fine enough that the +/-20 deg rear sector
    # contains several beams besides the single self-return one, matching a
    # real LaserScan's angular resolution far better than a coarse 8-beam
    # synthetic scan would.
    increment = 2 * math.pi / n_beams
    ranges = [background_m] * n_beams
    mast_index = round(math.pi / increment) % n_beams  # bearing == pi
    ranges[mast_index] = mast_m
    return ranges, 0.0, increment


def test_front_sector_min_range_ignores_mast_self_return_behind():
    # Real measurement (sandpit, top plate v2): the centre-mounted C1 sees
    # its own mast at ~0.072m in a narrow sector -- direction not assumed,
    # could land in the rear sector on any given rover. Without min_ignore_m
    # this would permanently read "blocked" and defeat the rear-clearance
    # gate regardless of what's actually behind the rover.
    ranges, angle_min, increment = _scan_with_mast_return_behind()
    result = front_sector_min_range(
        ranges, angle_min=angle_min, angle_increment=increment,
        sector_half_width_deg=20.0, center_bearing_rad=math.pi,
        min_ignore_m=0.2)
    # The mast return is filtered out; the next-closest in-sector beam (the
    # 1.0m background) is what's left.
    assert result == pytest.approx(1.0)


def test_real_fsm_retreats_despite_mast_self_return_behind():
    # End-to-end: the node-level min_ignore_m filtering means a mast
    # self-return behind the rover must not block a genuinely clear retreat.
    ranges, angle_min, increment = _scan_with_mast_return_behind()
    rear_range = front_sector_min_range(
        ranges, angle_min=angle_min, angle_increment=increment,
        sector_half_width_deg=20.0, center_bearing_rad=math.pi,
        min_ignore_m=0.2)
    fsm = _make_real_fsm(max_wiggle_attempts=1, retreat_min_clearance_m=0.30)
    t = _drive_real_fsm_to_wiggle_retry(fsm)
    fsm.step(now_s=t + 2.0, front_range=FRONT_RANGE, commanded_linear_x=25.0)
    out = fsm.step(now_s=t + 6.0, front_range=FRONT_RANGE,
                    commanded_linear_x=25.0, rear_range=rear_range)
    assert out["state"] == STATE_RETREAT


def test_front_sector_min_range_center_bearing_reads_behind():
    ranges = [1.0] * 8
    # 8 beams over 2*pi -> pi/4 rad apart. Put a close return at index 4,
    # which sits at bearing pi (directly behind) for angle_min=0.
    ranges[4] = 0.2
    result = front_sector_min_range(
        ranges, angle_min=0.0, angle_increment=math.pi / 4,
        sector_half_width_deg=20.0, center_bearing_rad=math.pi)
    assert result == pytest.approx(0.2)
    # The same close return must NOT show up in the default forward-facing
    # (center_bearing_rad=0.0) read.
    front = front_sector_min_range(
        ranges, angle_min=0.0, angle_increment=math.pi / 4,
        sector_half_width_deg=20.0)
    assert front == pytest.approx(1.0)
