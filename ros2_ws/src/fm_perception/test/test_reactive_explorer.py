"""
Purpose: Unit tests for the pure-Python geometry helpers and the
         ReactiveExplorerFSM bug-algorithm state machine used by
         reactive_explorer_node.py. No rclpy/ROS2 dependency, no hardware
         or Gazebo required -- the FSM is driven with synthetic yaw/odom/
         lidar readings and a caller-controlled clock.
         REDESIGNED 2026-07-27 after a real wall collision: the rover no
         longer drives backward blind to avoid obstacles. STARTUP_CHECK (at
         boot and whenever re-triggered) and MONITORING both consult a
         single caller-supplied heading_offset (radians, or None if boxed
         in) instead of a fixed +/-90deg candidate list, and there is a
         single TURN_TO_HEADING state instead of
         SCAN_AVOID_TURN/SCAN_AVOID_CONFIRM/RETREAT_TURN/RETREAT_DRIVE.
         SLOPE_RETREAT (over-tilt recovery) is unchanged by this redesign.
Inputs:  None.
Outputs: pytest results.
How to run:
    cd ros2_ws && colcon build --packages-select fm_perception
    source /opt/ros/humble/setup.bash && source install/setup.bash
    python3 -m pytest src/fm_perception/test/test_reactive_explorer.py -v
Project: Onboard Visual Foundation Models for Mars Terrain Perception and Rover Navigation
"""
import math
import signal

from fm_perception.reactive_explorer_node import (
    normalize_angle, yaw_from_quaternion, angle_delta, is_drivable_label,
    distance_2d, SAFE_LABELS, ReactiveExplorerFSM,
    default_shutdown, should_shutdown_now, correction_min_offset_rad,
    STATE_STARTUP_CHECK, STATE_MONITORING, STATE_TURN_TO_HEADING,
    STATE_FAILSAFE, STATE_STARTUP_SWEEP,
)


# ── yaw_from_quaternion ─────────────────────────────────────────────────

def test_yaw_from_identity_quaternion_is_zero():
    assert yaw_from_quaternion(0.0, 0.0, 0.0, 1.0) == 0.0


def test_yaw_from_90deg_quaternion():
    # Quaternion for +90 deg rotation about Z: (0, 0, sin(45deg), cos(45deg))
    z = math.sin(math.pi / 4)
    w = math.cos(math.pi / 4)
    yaw = yaw_from_quaternion(0.0, 0.0, z, w)
    assert math.isclose(yaw, math.pi / 2, abs_tol=1e-9)


def test_yaw_from_180deg_quaternion():
    yaw = yaw_from_quaternion(0.0, 0.0, 1.0, 0.0)
    assert math.isclose(abs(yaw), math.pi, abs_tol=1e-9)


# ── normalize_angle ──────────────────────────────────────────────────────

def test_normalize_angle_identity_within_range():
    assert math.isclose(normalize_angle(0.5), 0.5, abs_tol=1e-9)


def test_normalize_angle_wraps_positive_overflow():
    # 3*pi/2 should wrap to -pi/2
    assert math.isclose(normalize_angle(3 * math.pi / 2), -math.pi / 2, abs_tol=1e-9)


def test_normalize_angle_wraps_negative_overflow():
    assert math.isclose(normalize_angle(-3 * math.pi / 2), math.pi / 2, abs_tol=1e-9)


def test_normalize_angle_large_multiple_of_tau():
    assert math.isclose(normalize_angle(4 * math.pi + 0.3), 0.3, abs_tol=1e-9)


# ── angle_delta ───────────────────────────────────────────────────────────

def test_angle_delta_shortest_path_forward():
    assert math.isclose(angle_delta(math.pi / 2, 0.0), math.pi / 2, abs_tol=1e-9)


def test_angle_delta_shortest_path_wraparound():
    # From 170deg to -170deg the shortest path is +20deg, not -340deg.
    target = math.radians(-170)
    current = math.radians(170)
    delta = angle_delta(target, current)
    assert math.isclose(delta, math.radians(20), abs_tol=1e-6)


# ── is_drivable_label ─────────────────────────────────────────────────────
# Currently unused by the FSM itself (camera-confirm is bypassed until
# torch is installed on the Pi -- see module docstring), kept for the
# planned re-add.

def test_is_drivable_label_safe_classes():
    for label in ("soil", "sand", "bedrock"):
        assert is_drivable_label(label) is True


def test_is_drivable_label_hazard_classes():
    for label in ("big_rock", "uncertain", "unknown"):
        assert is_drivable_label(label) is False


def test_is_drivable_label_unrecognized_string_defaults_false():
    assert is_drivable_label("banana") is False


def test_safe_labels_matches_terrain_controller_policy():
    # Mirrors POLICY in terrain_controller_node.py: only soil/sand/bedrock drive.
    assert SAFE_LABELS == frozenset({"soil", "sand", "bedrock"})


# ── distance_2d ───────────────────────────────────────────────────────────

def test_distance_2d_basic():
    assert math.isclose(distance_2d(0.0, 0.0, 3.0, 4.0), 5.0, abs_tol=1e-9)


# ── ReactiveExplorerFSM ─────────────────────────────────────────────────
# Fast params so tests run in a handful of ticks: small dwell/settle, generous tolerance.

def _make_fsm(**overrides):
    params = dict(
        stuck_dwell_s=1.0,
        angle_tolerance_rad=math.radians(3.0),
        angular_speed=0.3,
        settle_time_s=0.0,
        # Off by default HERE ONLY: the shipped default is 8. These tests are
        # about what the FSM does once it is running, and a real 8-stop sweep
        # in front of every one of them would test the sweep 50 times over
        # and nothing else. The sweep's own tests below opt back in.
        sweep_headings=0,
    )
    params.update(overrides)
    return ReactiveExplorerFSM(**params)


def _clear_startup(fsm, now_s=0.0, heading_offset=0.0):
    """Helper: push a freshly constructed FSM through STARTUP_CHECK into
    MONITORING (heading_offset defaults to 0.0 -- already facing a viable
    heading, no turn needed)."""
    out = fsm.step(now_s=now_s, yaw=0.0, pos_x=0.0, pos_y=0.0,
                    terrain_stopped=False, lidar_stopped=False,
                    heading_offset=heading_offset)
    assert out["state"] == STATE_MONITORING
    return out


# ── STARTUP_CHECK ────────────────────────────────────────────────────────

def test_initial_state_is_startup_check():
    fsm = _make_fsm()
    assert fsm.state == STATE_STARTUP_CHECK
    assert fsm.active is True


def test_startup_check_holds_still_during_settle_time():
    fsm = _make_fsm(settle_time_s=1.0)
    out = fsm.step(now_s=0.0, yaw=0.0, pos_x=0.0, pos_y=0.0,
                    terrain_stopped=False, lidar_stopped=False,
                    heading_offset=0.0)
    assert out["state"] == STATE_STARTUP_CHECK
    assert out["linear_x"] == 0.0 and out["angular_z"] == 0.0

    out = fsm.step(now_s=0.5, yaw=0.0, pos_x=0.0, pos_y=0.0,
                    terrain_stopped=False, lidar_stopped=False,
                    heading_offset=0.0)
    assert out["state"] == STATE_STARTUP_CHECK

    out = fsm.step(now_s=1.1, yaw=0.0, pos_x=0.0, pos_y=0.0,
                    terrain_stopped=False, lidar_stopped=False,
                    heading_offset=0.0)
    assert out["state"] == STATE_MONITORING


def test_startup_check_waits_for_odom_without_failing():
    fsm = _make_fsm()
    out = fsm.step(now_s=0.0, yaw=0.0, pos_x=0.0, pos_y=0.0,
                    terrain_stopped=False, lidar_stopped=False,
                    heading_offset=0.0, odom_fresh=False)
    assert out["state"] == STATE_STARTUP_CHECK
    assert out["failsafe"] is False


def test_startup_check_boxed_in_goes_to_failsafe_with_reason():
    fsm = _make_fsm()
    out = fsm.step(now_s=0.0, yaw=0.0, pos_x=0.0, pos_y=0.0,
                    terrain_stopped=False, lidar_stopped=False,
                    heading_offset=None)
    assert out["state"] == STATE_FAILSAFE
    assert out["failsafe"] is True
    assert "boxed_in" in out["failsafe_reason"]


def test_startup_check_over_tilted_goes_to_failsafe_with_reason():
    fsm = _make_fsm()
    out = fsm.step(now_s=0.0, yaw=0.0, pos_x=0.0, pos_y=0.0,
                    terrain_stopped=False, lidar_stopped=False,
                    heading_offset=0.0, imu_slope_stop=True)
    assert out["state"] == STATE_FAILSAFE
    assert "over_tilted" in out["failsafe_reason"]


def test_startup_check_imu_not_ready_goes_to_failsafe_with_reason():
    fsm = _make_fsm()
    out = fsm.step(now_s=0.0, yaw=0.0, pos_x=0.0, pos_y=0.0,
                    terrain_stopped=False, lidar_stopped=False,
                    heading_offset=0.0, imu_fresh=False)
    assert out["state"] == STATE_FAILSAFE
    assert "imu_not_ready" in out["failsafe_reason"]


def test_startup_check_reports_all_applicable_reasons_together():
    fsm = _make_fsm()
    out = fsm.step(now_s=0.0, yaw=0.0, pos_x=0.0, pos_y=0.0,
                    terrain_stopped=False, lidar_stopped=False,
                    heading_offset=None, imu_slope_stop=True, imu_fresh=False)
    assert out["state"] == STATE_FAILSAFE
    assert "boxed_in" in out["failsafe_reason"]
    assert "over_tilted" in out["failsafe_reason"]
    assert "imu_not_ready" in out["failsafe_reason"]


def test_startup_check_zero_offset_goes_straight_to_monitoring_no_turn():
    fsm = _make_fsm()
    out = fsm.step(now_s=0.0, yaw=0.0, pos_x=0.0, pos_y=0.0,
                    terrain_stopped=False, lidar_stopped=False,
                    heading_offset=0.0)
    assert out["state"] == STATE_MONITORING
    assert out["active"] is False
    assert out["linear_x"] == 0.0 and out["angular_z"] == 0.0


def test_startup_check_nonzero_offset_enters_turn_to_heading():
    fsm = _make_fsm()
    out = fsm.step(now_s=0.0, yaw=0.0, pos_x=0.0, pos_y=0.0,
                    terrain_stopped=False, lidar_stopped=False,
                    heading_offset=math.pi / 2)
    assert out["state"] == STATE_TURN_TO_HEADING
    assert out["active"] is True
    assert math.isclose(fsm._target_yaw, math.pi / 2, abs_tol=1e-9)


# ── /scan staleness -- fatal in every state ─────────────────────────────

def test_scan_stale_goes_to_failsafe_from_startup_check():
    # A scan DID arrive at some point and then went stale -- something broke,
    # so this must still fail safe even in STARTUP_CHECK. Contrast with
    # test_startup_check_waits_for_first_scan_instead_of_failsafe below.
    fsm = _make_fsm()
    out = fsm.step(now_s=0.0, yaw=0.0, pos_x=0.0, pos_y=0.0,
                    terrain_stopped=False, lidar_stopped=False,
                    heading_offset=0.0, scan_fresh=False,
                    scan_ever_received=True)
    assert out["state"] == STATE_FAILSAFE
    assert "lidar_stale" in out["failsafe_reason"]


def test_startup_check_waits_for_first_scan_instead_of_failsafe():
    # Regression, real hardware 2026-07-28: the first tick fires ~11ms after
    # the node starts, long before DDS discovery delivers the first /scan, so
    # treating "no scan yet" the same as "stale scan" sent the rover to
    # FAILSAFE on every single launch even with the LiDAR driver running fine.
    # Nothing has been committed to in STARTUP_CHECK, so wait -- exactly what
    # a not-yet-published /exomy/odom already does here.
    fsm = _make_fsm()
    out = fsm.step(now_s=0.0, yaw=0.0, pos_x=0.0, pos_y=0.0,
                    terrain_stopped=False, lidar_stopped=False,
                    heading_offset=None, scan_fresh=False,
                    scan_ever_received=False)
    assert out["state"] == STATE_STARTUP_CHECK
    assert out["failsafe"] is False
    assert out["linear_x"] == 0.0 and out["angular_z"] == 0.0


def test_scan_never_received_outside_startup_check_still_failsafes():
    # Defensive: "never received" should be unreachable once the FSM has left
    # STARTUP_CHECK (leaving it requires a heading picked from a real scan),
    # but if it ever happens, fail safe rather than wait.
    fsm = _make_fsm()
    _clear_startup(fsm)
    out = fsm.step(now_s=1.0, yaw=0.0, pos_x=0.0, pos_y=0.0,
                    terrain_stopped=False, lidar_stopped=False,
                    scan_fresh=False, scan_ever_received=False)
    assert out["state"] == STATE_FAILSAFE
    assert "lidar_stale" in out["failsafe_reason"]


def test_startup_check_proceeds_once_first_scan_arrives():
    # The wait must not latch: once /scan starts flowing, STARTUP_CHECK
    # resumes its normal checks on the very next tick.
    fsm = _make_fsm()
    for tick_s in (0.0, 0.1, 0.2):
        out = fsm.step(now_s=tick_s, yaw=0.0, pos_x=0.0, pos_y=0.0,
                        terrain_stopped=False, lidar_stopped=False,
                        heading_offset=None, scan_fresh=False,
                        scan_ever_received=False)
        assert out["state"] == STATE_STARTUP_CHECK

    out = fsm.step(now_s=0.3, yaw=0.0, pos_x=0.0, pos_y=0.0,
                    terrain_stopped=False, lidar_stopped=False,
                    heading_offset=0.0, scan_fresh=True,
                    scan_ever_received=True)
    assert out["state"] == STATE_MONITORING
    assert out["failsafe"] is False


def test_scan_stale_goes_to_failsafe_from_monitoring():
    fsm = _make_fsm()
    _clear_startup(fsm)
    out = fsm.step(now_s=1.0, yaw=0.0, pos_x=0.0, pos_y=0.0,
                    terrain_stopped=False, lidar_stopped=False,
                    scan_fresh=False)
    assert out["state"] == STATE_FAILSAFE
    assert "lidar_stale" in out["failsafe_reason"]


def test_scan_stale_aborts_turn_to_heading():
    fsm = _make_fsm()
    fsm.step(now_s=0.0, yaw=0.0, pos_x=0.0, pos_y=0.0,
              terrain_stopped=False, lidar_stopped=False,
              heading_offset=math.pi / 2)
    assert fsm.state == STATE_TURN_TO_HEADING

    out = fsm.step(now_s=1.0, yaw=math.radians(10), pos_x=0.0, pos_y=0.0,
                    terrain_stopped=False, lidar_stopped=False,
                    scan_fresh=False)
    assert out["state"] == STATE_FAILSAFE
    assert "lidar_stale" in out["failsafe_reason"]


# ── MONITORING ───────────────────────────────────────────────────────────

def test_monitoring_stays_idle_when_not_stopped():
    fsm = _make_fsm()
    _clear_startup(fsm)
    out = fsm.step(now_s=1.0, yaw=0.0, pos_x=0.0, pos_y=0.0,
                    terrain_stopped=False, lidar_stopped=False)
    assert out["state"] == STATE_MONITORING
    assert out["active"] is False
    assert out["linear_x"] == 0.0 and out["angular_z"] == 0.0


def test_monitoring_requires_dwell_before_escalating():
    fsm = _make_fsm(stuck_dwell_s=3.0)
    _clear_startup(fsm)
    out = fsm.step(now_s=1.0, yaw=0.0, pos_x=0.0, pos_y=0.0,
                    terrain_stopped=True, lidar_stopped=False,
                    heading_offset=math.pi / 2)
    assert out["state"] == STATE_MONITORING and out["active"] is False

    out = fsm.step(now_s=2.5, yaw=0.0, pos_x=0.0, pos_y=0.0,
                    terrain_stopped=True, lidar_stopped=False,
                    heading_offset=math.pi / 2)
    assert out["state"] == STATE_MONITORING and out["active"] is False

    out = fsm.step(now_s=4.1, yaw=0.0, pos_x=0.0, pos_y=0.0,
                    terrain_stopped=True, lidar_stopped=False,
                    heading_offset=math.pi / 2)
    assert out["state"] == STATE_TURN_TO_HEADING
    assert out["active"] is True


def test_monitoring_dwell_resets_if_stop_clears_briefly():
    fsm = _make_fsm(stuck_dwell_s=3.0)
    _clear_startup(fsm)
    fsm.step(now_s=1.0, yaw=0.0, pos_x=0.0, pos_y=0.0,
              terrain_stopped=True, lidar_stopped=False)
    fsm.step(now_s=3.0, yaw=0.0, pos_x=0.0, pos_y=0.0,
              terrain_stopped=False, lidar_stopped=False)
    fsm.step(now_s=3.1, yaw=0.0, pos_x=0.0, pos_y=0.0,
              terrain_stopped=True, lidar_stopped=False)
    out = fsm.step(now_s=4.5, yaw=0.0, pos_x=0.0, pos_y=0.0,
                    terrain_stopped=True, lidar_stopped=False)
    assert out["state"] == STATE_MONITORING


def test_lidar_stop_alone_triggers_same_as_terrain_stop():
    fsm = _make_fsm(stuck_dwell_s=1.0)
    _clear_startup(fsm)
    fsm.step(now_s=1.0, yaw=0.0, pos_x=0.0, pos_y=0.0,
              terrain_stopped=False, lidar_stopped=True)
    out = fsm.step(now_s=2.1, yaw=0.0, pos_x=0.0, pos_y=0.0,
                    terrain_stopped=False, lidar_stopped=True,
                    heading_offset=0.3)
    assert out["state"] == STATE_TURN_TO_HEADING


def test_forward_corridor_blocked_triggers_immediately_no_dwell():
    # The whole point of this feature: no need to wait stuck_dwell_s when the
    # LiDAR itself sees the forward corridor closing in, independent of the
    # terrain/lidar-proximity-stop flags.
    fsm = _make_fsm(stuck_dwell_s=3.0)
    _clear_startup(fsm)
    out = fsm.step(now_s=1.0, yaw=0.0, pos_x=0.0, pos_y=0.0,
                    terrain_stopped=False, lidar_stopped=False,
                    forward_corridor_blocked=True, heading_offset=0.3)
    assert out["state"] == STATE_TURN_TO_HEADING
    assert out["active"] is True


def test_monitoring_forward_blocked_but_boxed_in_goes_to_failsafe():
    fsm = _make_fsm(stuck_dwell_s=3.0)
    _clear_startup(fsm)
    out = fsm.step(now_s=1.0, yaw=0.0, pos_x=0.0, pos_y=0.0,
                    terrain_stopped=False, lidar_stopped=False,
                    forward_corridor_blocked=True, heading_offset=None)
    assert out["state"] == STATE_FAILSAFE
    assert "boxed_in" in out["failsafe_reason"]


def test_monitoring_forward_blocked_odom_stale_stays_put():
    fsm = _make_fsm(stuck_dwell_s=3.0)
    _clear_startup(fsm)
    out = fsm.step(now_s=1.0, yaw=0.0, pos_x=0.0, pos_y=0.0,
                    terrain_stopped=False, lidar_stopped=False,
                    forward_corridor_blocked=True, heading_offset=0.3,
                    odom_fresh=False)
    assert out["state"] == STATE_MONITORING
    assert out["active"] is False
    assert out["linear_x"] == 0.0 and out["angular_z"] == 0.0


# ── Terrain rejection memory ─────────────────────────────────────────────
# The heading picker is LiDAR-only: it knows what is physically passable,
# not what DINOv2 thinks of the ground there. So when the rover turns to a
# heading, drives, and terrain_controller_node then refuses to move on that
# terrain, re-scanning would pick the exact same heading again forever --
# the geometry has not changed. The FSM therefore remembers headings that
# were rejected on TERRAIN grounds and publishes them so the node can
# exclude them from the next pick. This is the mechanism that puts DINOv2
# back in charge of WHERE the rover goes, with LiDAR as the veto rather
# than the decider.

def test_terrain_stop_while_facing_a_clear_heading_records_it_as_rejected():
    fsm = _make_fsm(stuck_dwell_s=1.0)
    _clear_startup(fsm)
    # LiDAR is happy with straight ahead, but terrain says stop.
    fsm.step(now_s=1.0, yaw=0.0, pos_x=0.0, pos_y=0.0,
              terrain_stopped=True, lidar_stopped=False, heading_offset=0.0)
    out = fsm.step(now_s=2.1, yaw=0.0, pos_x=0.0, pos_y=0.0,
                    terrain_stopped=True, lidar_stopped=False, heading_offset=0.0)
    assert out["excluded_offsets"], "the heading just refused must be excluded"
    assert any(abs(o) <= math.radians(3.0) for o in out["excluded_offsets"])


def test_excluded_offsets_track_the_rovers_current_heading():
    # Exclusions are stored in the odom frame and reported relative to where
    # the rover is pointing NOW, so they stay attached to the same patch of
    # world after the rover turns away from it.
    fsm = _make_fsm(stuck_dwell_s=1.0)
    _clear_startup(fsm)
    fsm.step(now_s=1.0, yaw=0.0, pos_x=0.0, pos_y=0.0,
              terrain_stopped=True, lidar_stopped=False, heading_offset=0.0)
    fsm.step(now_s=2.1, yaw=0.0, pos_x=0.0, pos_y=0.0,
              terrain_stopped=True, lidar_stopped=False, heading_offset=0.0)

    # Same rejected patch, but the rover is now facing 90deg away from it.
    out = fsm.step(now_s=2.2, yaw=math.radians(90), pos_x=0.0, pos_y=0.0,
                    terrain_stopped=True, lidar_stopped=False, heading_offset=0.0)
    assert any(math.isclose(o, math.radians(-90), abs_tol=math.radians(2))
               for o in out["excluded_offsets"]), out["excluded_offsets"]


def test_driving_cleanly_clears_the_rejection_memory():
    # Once terrain and LiDAR both agree the rover is fine, whatever was
    # rejected belongs to a patch of ground it has left behind.
    fsm = _make_fsm(stuck_dwell_s=1.0)
    _clear_startup(fsm)
    fsm.step(now_s=1.0, yaw=0.0, pos_x=0.0, pos_y=0.0,
              terrain_stopped=True, lidar_stopped=False, heading_offset=0.0)
    fsm.step(now_s=2.1, yaw=0.0, pos_x=0.0, pos_y=0.0,
              terrain_stopped=True, lidar_stopped=False, heading_offset=0.0)
    assert fsm.excluded_offsets(0.0)

    out = fsm.step(now_s=2.5, yaw=0.0, pos_x=0.0, pos_y=0.0,
                    terrain_stopped=False, lidar_stopped=False)
    assert out["state"] == STATE_MONITORING
    assert out["excluded_offsets"] == []


def test_lidar_only_stop_does_not_poison_the_terrain_rejection_memory():
    # A LiDAR proximity stop is about an object in the way, not about the
    # ground being untraversable -- excluding the heading for that would
    # permanently blacklist directions the rover could use once a person or
    # a rock moves.
    fsm = _make_fsm(stuck_dwell_s=1.0)
    _clear_startup(fsm)
    fsm.step(now_s=1.0, yaw=0.0, pos_x=0.0, pos_y=0.0,
              terrain_stopped=False, lidar_stopped=True, heading_offset=0.0)
    out = fsm.step(now_s=2.1, yaw=0.0, pos_x=0.0, pos_y=0.0,
                    terrain_stopped=False, lidar_stopped=True, heading_offset=0.0)
    assert out["excluded_offsets"] == []


def test_a_heading_is_only_recorded_once_however_long_terrain_stays_bad():
    fsm = _make_fsm(stuck_dwell_s=1.0)
    _clear_startup(fsm)
    fsm.step(now_s=1.0, yaw=0.0, pos_x=0.0, pos_y=0.0,
              terrain_stopped=True, lidar_stopped=False, heading_offset=0.0)
    for t in (2.1, 3.1, 4.1, 5.1):
        out = fsm.step(now_s=t, yaw=0.0, pos_x=0.0, pos_y=0.0,
                        terrain_stopped=True, lidar_stopped=False,
                        heading_offset=0.0)
    assert len(out["excluded_offsets"]) == 1, out["excluded_offsets"]


def test_all_headings_rejected_eventually_reaches_boxed_in_failsafe():
    # With every direction excluded the node's picker returns None, which is
    # the same "nowhere to go" signal as being physically walled in.
    fsm = _make_fsm(stuck_dwell_s=1.0)
    _clear_startup(fsm)
    fsm.step(now_s=1.0, yaw=0.0, pos_x=0.0, pos_y=0.0,
              terrain_stopped=True, lidar_stopped=False, heading_offset=0.0)
    out = fsm.step(now_s=2.1, yaw=0.0, pos_x=0.0, pos_y=0.0,
                    terrain_stopped=True, lidar_stopped=False, heading_offset=None)
    assert out["state"] == STATE_FAILSAFE
    assert "boxed_in" in out["failsafe_reason"]


# ── TURN_TO_HEADING ──────────────────────────────────────────────────────

def test_completing_a_turn_returns_to_startup_check_to_re_settle_and_re_scan():
    # MAJOR 5 (2026-07-27): settle_time_s existed but STARTUP_CHECK was only
    # ever entered once, at boot, so the "let the IMU/LiDAR settle before you
    # trust them" guarantee did not apply after a turn -- which is exactly
    # when the readings are most disturbed by the motion that just stopped.
    # Completing a turn must re-enter STARTUP_CHECK, not jump to MONITORING.
    fsm = _make_fsm(settle_time_s=1.0)
    # Clear the boot settle window first, then commit to a turn.
    fsm.step(now_s=0.0, yaw=0.0, pos_x=0.0, pos_y=0.0,
              terrain_stopped=False, lidar_stopped=False, heading_offset=0.0)
    fsm.step(now_s=1.1, yaw=0.0, pos_x=0.0, pos_y=0.0,
              terrain_stopped=False, lidar_stopped=False,
              heading_offset=math.pi / 2)
    assert fsm.state == STATE_TURN_TO_HEADING

    out = fsm.step(now_s=2.0, yaw=math.radians(89), pos_x=0.0, pos_y=0.0,
                    terrain_stopped=False, lidar_stopped=False)
    assert out["state"] == STATE_STARTUP_CHECK

    # ...and the settle timer really does restart, rather than still holding
    # the boot-time value (which would make the re-check instantaneous).
    # The new settle window starts on the first tick after re-entry (2.5),
    # so it clears at 3.5 -- not at boot_time + settle_time.
    out = fsm.step(now_s=2.5, yaw=math.radians(89), pos_x=0.0, pos_y=0.0,
                    terrain_stopped=False, lidar_stopped=False,
                    heading_offset=0.0)
    assert out["state"] == STATE_STARTUP_CHECK
    out = fsm.step(now_s=3.4, yaw=math.radians(89), pos_x=0.0, pos_y=0.0,
                    terrain_stopped=False, lidar_stopped=False,
                    heading_offset=0.0)
    assert out["state"] == STATE_STARTUP_CHECK
    out = fsm.step(now_s=3.6, yaw=math.radians(89), pos_x=0.0, pos_y=0.0,
                    terrain_stopped=False, lidar_stopped=False,
                    heading_offset=0.0)
    assert out["state"] == STATE_MONITORING


def test_turn_to_heading_commands_angular_until_target_reached():
    fsm = _make_fsm()
    fsm.step(now_s=0.0, yaw=0.0, pos_x=0.0, pos_y=0.0,
              terrain_stopped=False, lidar_stopped=False,
              heading_offset=math.pi / 2)
    assert fsm.state == STATE_TURN_TO_HEADING

    out = fsm.step(now_s=1.0, yaw=math.radians(10), pos_x=0.0, pos_y=0.0,
                    terrain_stopped=False, lidar_stopped=False)
    assert out["state"] == STATE_TURN_TO_HEADING
    assert out["angular_z"] > 0.0
    assert out["linear_x"] == 0.0

    out = fsm.step(now_s=1.1, yaw=math.radians(89), pos_x=0.0, pos_y=0.0,
                    terrain_stopped=False, lidar_stopped=False)
    # A completed turn re-enters STARTUP_CHECK (re-settle + re-scan from the
    # new heading) rather than handing straight back to MONITORING -- see
    # test_completing_a_turn_returns_to_startup_check_to_re_settle_and_re_scan.
    assert out["state"] == STATE_STARTUP_CHECK
    assert out["angular_z"] == 0.0


def test_turn_to_heading_timeout_aborts_to_failsafe_even_with_fresh_odom():
    fsm = _make_fsm(max_turn_duration_s=5.0)
    fsm.step(now_s=0.0, yaw=0.0, pos_x=0.0, pos_y=0.0,
              terrain_stopped=False, lidar_stopped=False,
              heading_offset=math.pi / 2)
    assert fsm.state == STATE_TURN_TO_HEADING

    # yaw never approaches the target, but odom_fresh stays True throughout
    # -- only the wall-clock deadline should trigger this.
    out = fsm.step(now_s=2.0, yaw=0.0, pos_x=0.0, pos_y=0.0,
                    terrain_stopped=False, lidar_stopped=False, odom_fresh=True)
    assert out["state"] == STATE_TURN_TO_HEADING  # deadline not yet passed

    out = fsm.step(now_s=5.2, yaw=0.0, pos_x=0.0, pos_y=0.0,
                    terrain_stopped=False, lidar_stopped=False, odom_fresh=True)
    assert out["state"] == STATE_FAILSAFE
    # active stays True in FAILSAFE so the node keeps commanding the stop
    assert out["active"] is True
    assert out["failsafe"] is True
    assert "turn_timeout" in out["failsafe_reason"]
    assert out["linear_x"] == 0.0 and out["angular_z"] == 0.0


def test_odom_stale_aborts_in_progress_turn_to_failsafe():
    fsm = _make_fsm()
    fsm.step(now_s=0.0, yaw=0.0, pos_x=0.0, pos_y=0.0,
              terrain_stopped=False, lidar_stopped=False,
              heading_offset=math.pi / 2)
    assert fsm.state == STATE_TURN_TO_HEADING

    out = fsm.step(now_s=1.0, yaw=math.radians(10), pos_x=0.0, pos_y=0.0,
                    terrain_stopped=False, lidar_stopped=False,
                    odom_fresh=False)
    assert out["state"] == STATE_FAILSAFE
    # active stays True in FAILSAFE so the node keeps commanding the stop
    assert out["active"] is True
    assert out["failsafe"] is True
    assert "odom_stale" in out["failsafe_reason"]
    assert out["linear_x"] == 0.0 and out["angular_z"] == 0.0


# ── IMU over-tilt: stop and re-decide, never reverse ─────────────────────
# SLOPE_RETREAT was removed on 2026-07-29. It was the only state that drove
# the rover backwards, and the rover has no rear-facing sensing beyond the
# same LiDAR, so a blind reverse was the most dangerous manoeuvre in the
# stack -- it is what put the rover into a wall on 2026-07-27. It was also
# firing on tilt readings that were not real: Trial A logged up to 62.9 deg
# on a flat lab floor because the accelerometer-only estimate cannot separate
# gravity from acceleration (fixed separately in imu_slope_fusion_node).
#
# An over-tilt is now treated exactly like a blocked corridor: hold still,
# re-scan, and turn towards a heading that is clear. The IMU keeps its veto
# over where the rover goes; it just cannot command a manoeuvre any more.

def test_over_tilt_turns_towards_a_clear_heading_instead_of_reversing():
    fsm = _make_fsm(stuck_dwell_s=3.0)
    _clear_startup(fsm)
    out = fsm.step(now_s=1.0, yaw=0.0, pos_x=0.0, pos_y=0.0,
                    terrain_stopped=False, lidar_stopped=False,
                    imu_slope_stop=True, heading_offset=math.radians(40.0))
    assert out["state"] == STATE_TURN_TO_HEADING
    assert out["linear_x"] == 0.0


def test_over_tilt_never_commands_reverse_in_any_state():
    # The property that matters most: whatever the FSM decides, it must not
    # ask the wheels to go backwards.
    for dwell in (0.0, 3.0):
        fsm = _make_fsm(stuck_dwell_s=dwell)
        _clear_startup(fsm)
        for t in range(20):
            out = fsm.step(now_s=float(t), yaw=0.0, pos_x=0.0, pos_y=0.0,
                            terrain_stopped=False, lidar_stopped=False,
                            imu_slope_stop=True,
                            heading_offset=math.radians(40.0))
            assert out["linear_x"] >= 0.0, (
                f"reverse commanded at t={t} in state {out['state']}"
            )


def test_over_tilt_preempts_a_turn_already_in_progress():
    # The rover must re-decide from where it now is rather than finish a turn
    # planned before the slope was detected.
    fsm = _make_fsm(stuck_dwell_s=3.0)
    _clear_startup(fsm)
    fsm.step(now_s=1.0, yaw=0.0, pos_x=0.0, pos_y=0.0,
             terrain_stopped=True, lidar_stopped=False,
             forward_corridor_blocked=True, heading_offset=math.radians(90.0))
    assert fsm.state == STATE_TURN_TO_HEADING
    out = fsm.step(now_s=2.0, yaw=0.0, pos_x=0.0, pos_y=0.0,
                    terrain_stopped=False, lidar_stopped=False,
                    imu_slope_stop=True, heading_offset=math.radians(30.0))
    assert out["state"] == STATE_TURN_TO_HEADING
    assert math.isclose(fsm._target_yaw, math.radians(30.0), abs_tol=1e-9)


def test_over_tilt_with_nowhere_clear_fails_safe():
    fsm = _make_fsm(stuck_dwell_s=3.0)
    _clear_startup(fsm)
    out = fsm.step(now_s=1.0, yaw=0.0, pos_x=0.0, pos_y=0.0,
                    terrain_stopped=False, lidar_stopped=False,
                    imu_slope_stop=True, heading_offset=None)
    assert out["state"] == STATE_FAILSAFE
    assert "boxed_in" in out["failsafe_reason"]


def test_over_tilt_already_facing_a_clear_heading_just_holds():
    fsm = _make_fsm(stuck_dwell_s=3.0)
    _clear_startup(fsm)
    out = fsm.step(now_s=1.0, yaw=0.0, pos_x=0.0, pos_y=0.0,
                    terrain_stopped=False, lidar_stopped=False,
                    imu_slope_stop=True, heading_offset=0.0)
    assert out["linear_x"] == 0.0
    assert out["angular_z"] == 0.0


def test_over_tilt_during_startup_check_still_fails_safe():
    # Unchanged: an over-tilt before the rover has moved at all means "refuse
    # to start", not "drive somewhere else".
    fsm = _make_fsm(stuck_dwell_s=3.0, settle_time_s=0.0)
    out = fsm.step(now_s=1.0, yaw=0.0, pos_x=0.0, pos_y=0.0,
                    terrain_stopped=False, lidar_stopped=False,
                    imu_slope_stop=True, heading_offset=0.0,
                    scan_ever_received=True)
    assert out["state"] == STATE_FAILSAFE
    assert "over_tilted" in out["failsafe_reason"]


# ── FAILSAFE is terminal ─────────────────────────────────────────────────

def test_failsafe_is_terminal_ignores_further_input():
    fsm = _make_fsm()
    out = fsm.step(now_s=0.0, yaw=0.0, pos_x=0.0, pos_y=0.0,
                    terrain_stopped=False, lidar_stopped=False,
                    heading_offset=None)
    assert out["state"] == STATE_FAILSAFE

    out = fsm.step(now_s=100.0, yaw=1.0, pos_x=5.0, pos_y=5.0,
                    terrain_stopped=False, lidar_stopped=False,
                    heading_offset=0.0)
    assert out["state"] == STATE_FAILSAFE
    assert out["linear_x"] == 0.0 and out["angular_z"] == 0.0
    assert out["failsafe"] is True


def test_failsafe_keeps_owning_cmd_vel_so_the_stop_is_actually_commanded():
    # 2026-07-27: FAILSAFE used to clear active, which meant the node stopped
    # publishing cmd_vel entirely -- it never actually SENT a stop, it just
    # went quiet and let whoever else was publishing take over. Combined with
    # the failsafe shutdown that follows, that left a window where nothing was
    # commanding zero. FAILSAFE must keep ownership and keep commanding zero.
    fsm = _make_fsm()
    out = fsm.step(now_s=0.0, yaw=0.0, pos_x=0.0, pos_y=0.0,
                    terrain_stopped=False, lidar_stopped=False,
                    heading_offset=None)
    assert out["state"] == STATE_FAILSAFE
    assert out["active"] is True
    assert out["linear_x"] == 0.0 and out["angular_z"] == 0.0

    out = fsm.step(now_s=50.0, yaw=0.0, pos_x=0.0, pos_y=0.0,
                    terrain_stopped=False, lidar_stopped=False)
    assert out["active"] is True
    assert out["linear_x"] == 0.0 and out["angular_z"] == 0.0


# ── Failsafe shutdown sequencing ─────────────────────────────────────────
# The rover must be commanded to a stop FIRST and the launch torn down
# SECOND. Tearing the stack down immediately kills motor_node, which is the
# only thing that calls stopMotors() -- and if the signal is late or lost,
# the servo HAT simply holds the last PWM command (exactly the 2026-07-27
# wall-collision failure mode).

def test_shutdown_is_withheld_until_the_stop_has_been_commanded_for_the_grace_period():
    assert should_shutdown_now(failsafe=True, failsafe_since_s=10.0,
                                now_s=10.0, grace_s=2.0) is False
    assert should_shutdown_now(failsafe=True, failsafe_since_s=10.0,
                                now_s=11.9, grace_s=2.0) is False
    assert should_shutdown_now(failsafe=True, failsafe_since_s=10.0,
                                now_s=12.0, grace_s=2.0) is True
    assert should_shutdown_now(failsafe=True, failsafe_since_s=10.0,
                                now_s=99.0, grace_s=2.0) is True


def test_shutdown_never_requested_while_not_in_failsafe():
    assert should_shutdown_now(failsafe=False, failsafe_since_s=None,
                                now_s=99.0, grace_s=2.0) is False


def test_shutdown_signals_the_whole_process_group_not_just_the_leader():
    # os.kill(os.getpgid(0), SIGINT) targets the single process whose PID
    # happens to equal the group id -- verified experimentally 2026-07-27 to
    # leave sibling processes (i.e. every other node in the launch, including
    # motor_node) running. os.killpg is what actually signals the group.
    calls = []

    class _FakeOs:
        @staticmethod
        def getpgid(pid):
            return 4242

        @staticmethod
        def killpg(pgid, sig):
            calls.append(("killpg", pgid, sig))

        @staticmethod
        def kill(pid, sig):
            calls.append(("kill", pid, sig))

    default_shutdown(os_module=_FakeOs)
    assert calls == [("killpg", 4242, signal.SIGINT)], calls


# ── Startup 8-direction sweep: the foundation model chooses ──────────────
# Added 2026-07-29. Before this, the rover picked its opening heading from
# LiDAR geometry alone and DINOv2 only ever got a veto that never fired --
# the 214-frame study found `uncertain` firing 0/214 and a wall reading
# soil:0.937, so a threshold-based veto rubber-stamps everything.
#
# The rover now turns through 8 stops of 45 deg, which is what the camera
# needs: the imx219's ~62 deg field of view means a stationary rover has
# terrain data for one direction out of six. The LiDAR already returns the
# full circle in a single frame and gains nothing from the rotation.
#
# Selection is a rank, not a threshold: LiDAR eliminates headings the rover
# physically cannot fit through, and among what survives the LOWEST
# traversability score wins. Ranking only needs the model to order two
# surfaces correctly, where a threshold needs its absolute calibration to
# hold in a domain it was never trained on.

def _start_sweep(fsm):
    """Push a freshly constructed FSM out of STARTUP_CHECK and into the sweep."""
    out = fsm.step(now_s=0.0, yaw=0.0, pos_x=0.0, pos_y=0.0,
                    terrain_stopped=False, lidar_stopped=False,
                    heading_offset=0.0)
    assert out["state"] == STATE_STARTUP_SWEEP
    return out


def _drive_sweep(fsm, per_heading, settle_time_s=0.0, start_t=10.0):
    """Tick a sweep to completion.

    per_heading is one (score, corridor_blocked, clearance_m) triple per stop,
    consumed in the order the rover visits them. Yaw is snapped to whatever
    the FSM asks for, which is what a real point turn converges to.
    """
    t = start_t
    seq = 0
    yaw = 0.0
    out = None
    for _ in range(4000):
        idx = min(fsm._sweep_index, len(per_heading) - 1)
        score, blocked, clearance = per_heading[idx]
        if fsm._sweep_phase == "SAMPLE":
            seq += 1
        out = fsm.step(now_s=t, yaw=yaw, pos_x=0.0, pos_y=0.0,
                       terrain_stopped=False, lidar_stopped=False,
                       forward_corridor_blocked=blocked,
                       forward_clearance_m=clearance,
                       traversability_score=score, terrain_score_seq=seq,
                       heading_offset=0.0)
        if out["angular_z"] != 0.0:
            yaw = fsm._target_yaw          # the turn converges
        if fsm.state != STATE_STARTUP_SWEEP:
            return out, yaw
        t += max(settle_time_s, 0.1)
    raise AssertionError("sweep never finished")


def _uniform(n=8, score=0.2, blocked=False, clearance=5.0):
    return [(score, blocked, clearance)] * n


def test_sweep_visits_every_heading_before_deciding():
    fsm = _make_fsm(settle_time_s=0.0, sweep_headings=8)
    _start_sweep(fsm)
    _drive_sweep(fsm, _uniform())
    assert len(fsm.sweep_report) == 8
    visited = sorted(round(math.degrees(s["offset_rad"])) for s in fsm.sweep_report)
    assert visited == [-135, -90, -45, 0, 45, 90, 135, 180]


def test_sweep_picks_the_lowest_traversability_score():
    fsm = _make_fsm(settle_time_s=0.0, sweep_headings=8)
    _start_sweep(fsm)
    per_heading = _uniform()
    per_heading[2] = (0.05, False, 5.0)      # third stop is the best ground
    _drive_sweep(fsm, per_heading)
    assert math.isclose(fsm.sweep_choice["offset_rad"], math.radians(90.0),
                        abs_tol=1e-9)


def test_sweep_refuses_a_heading_the_rover_cannot_fit_through():
    # The LiDAR veto runs first and is absolute: the best-looking ground in
    # the room is worthless if the rover cannot physically get down it.
    fsm = _make_fsm(settle_time_s=0.0, sweep_headings=8)
    _start_sweep(fsm)
    per_heading = _uniform()
    per_heading[2] = (0.01, True, 0.1)       # gorgeous terrain, blocked
    per_heading[5] = (0.10, False, 5.0)      # next best, and passable
    _drive_sweep(fsm, per_heading)
    assert math.isclose(fsm.sweep_choice["offset_rad"], math.radians(-135.0),
                        abs_tol=1e-9)


def test_sweep_breaks_a_score_tie_on_lidar_clearance():
    fsm = _make_fsm(settle_time_s=0.0, sweep_headings=8)
    _start_sweep(fsm)
    per_heading = _uniform(score=0.2, clearance=1.0)
    per_heading[4] = (0.2, False, 6.0)       # same score, much more room
    _drive_sweep(fsm, per_heading)
    assert math.isclose(fsm.sweep_choice["offset_rad"], math.radians(180.0),
                        abs_tol=1e-9)


def test_sweep_fails_safe_when_every_heading_is_blocked():
    fsm = _make_fsm(settle_time_s=0.0, sweep_headings=8)
    _start_sweep(fsm)
    out, _ = _drive_sweep(fsm, _uniform(blocked=True, clearance=0.1))
    assert out["state"] == STATE_FAILSAFE
    assert "boxed_in" in out["failsafe_reason"]


def test_sweep_sample_ignores_an_uninformative_frame():
    # 2026-08-02, pre-Trial-A audit: an uninformative frame (camera still
    # converging exposure, or a genuinely blank/dark view) is published with
    # traversability_score=1.0, the same value an impassable rock gets. Before
    # this fix the sweep's SAMPLE phase had no terrain_frame_informative
    # gate at all (unlike the mid-drive confirm, which always had one), so a
    # sweep stop sampled mid-exposure-race could silently record the worst
    # possible score for a heading never actually seen.
    fsm = _make_fsm(settle_time_s=0.0, sweep_headings=1,
                    sweep_samples_per_heading=1)
    _start_sweep(fsm)
    # TURN -> already at the target yaw (single-heading sweep, offset 0) ->
    # arrives immediately.
    fsm.step(now_s=1.0, yaw=0.0, pos_x=0.0, pos_y=0.0,
             terrain_stopped=False, lidar_stopped=False,
             forward_corridor_blocked=False, forward_clearance_m=5.0,
             heading_offset=0.0)
    # SETTLE -> SAMPLE (settle_time_s=0.0).
    fsm.step(now_s=1.0, yaw=0.0, pos_x=0.0, pos_y=0.0,
             terrain_stopped=False, lidar_stopped=False,
             forward_corridor_blocked=False, forward_clearance_m=5.0,
             heading_offset=0.0)
    assert fsm._sweep_phase == "SAMPLE"

    # An uninformative-but-fresh result arrives first: must not be recorded.
    out = fsm.step(now_s=2.0, yaw=0.0, pos_x=0.0, pos_y=0.0,
                    terrain_stopped=False, lidar_stopped=False,
                    forward_corridor_blocked=False, forward_clearance_m=5.0,
                    traversability_score=1.0, terrain_score_seq=1,
                    terrain_frame_informative=False, heading_offset=0.0)
    assert out["state"] == STATE_STARTUP_SWEEP
    assert fsm._sweep_scores == []

    # Then a genuinely informative result -- this is the one that must win.
    fsm.step(now_s=3.0, yaw=0.0, pos_x=0.0, pos_y=0.0,
             terrain_stopped=False, lidar_stopped=False,
             forward_corridor_blocked=False, forward_clearance_m=5.0,
             traversability_score=0.05, terrain_score_seq=2,
             terrain_frame_informative=True, heading_offset=0.0)
    assert math.isclose(fsm.sweep_report[0]["score"], 0.05, abs_tol=1e-9)


def test_sweep_records_a_baseline_from_the_scores_it_saw():
    # The baseline is what makes the mid-drive terrain check relative to THIS
    # environment instead of to a threshold calibrated on another planet.
    # Changed 2026-07-29: this was the median of every score the sweep saw,
    # which refuses about half of any compass by construction and cost Trial A
    # its whole run. It is now the chosen heading's own score -- still
    # measured in THIS environment, but describing the direction actually
    # being driven rather than an average direction.
    fsm = _make_fsm(settle_time_s=0.0, sweep_headings=8)
    _start_sweep(fsm)
    per_heading = [(0.1 * i, False, 5.0) for i in range(8)]
    _drive_sweep(fsm, per_heading)
    assert fsm.terrain_baseline is not None
    assert math.isclose(fsm.terrain_baseline, 0.0, abs_tol=1e-9)


def test_sweep_turns_to_its_choice_then_monitors():
    fsm = _make_fsm(settle_time_s=0.0, sweep_headings=8)
    _start_sweep(fsm)
    per_heading = _uniform()
    per_heading[2] = (0.05, False, 5.0)
    out, yaw = _drive_sweep(fsm, per_heading)
    assert out["state"] == STATE_TURN_TO_HEADING
    assert math.isclose(fsm._target_yaw, math.radians(90.0), abs_tol=1e-9)


def test_sweep_commands_no_linear_motion_at_any_point():
    fsm = _make_fsm(settle_time_s=0.0, sweep_headings=8)
    _start_sweep(fsm)
    t, seq, yaw = 10.0, 0, 0.0
    for _ in range(400):
        seq += 1
        out = fsm.step(now_s=t, yaw=yaw, pos_x=0.0, pos_y=0.0,
                       terrain_stopped=False, lidar_stopped=False,
                       forward_clearance_m=5.0, traversability_score=0.2,
                       terrain_score_seq=seq, heading_offset=0.0)
        assert out["linear_x"] == 0.0
        if out["angular_z"] != 0.0:
            yaw = fsm._target_yaw
        if fsm.state != STATE_STARTUP_SWEEP:
            break
        t += 0.1


def test_sweep_runs_once_not_on_every_later_startup_check():
    # A turn that completes re-enters STARTUP_CHECK to settle and re-scan.
    # Re-running the whole 8-stop sweep there would cost ~85 s every time the
    # rover changed its mind.
    fsm = _make_fsm(settle_time_s=0.0, sweep_headings=8)
    _start_sweep(fsm)
    _drive_sweep(fsm, _uniform())
    assert fsm.sweep_choice is not None
    fsm._enter_startup_check()
    # A score is supplied because the terrain confirm now runs here too; the
    # point of this test is only that the 8-stop sweep does not run again.
    out = _feed_confirm(fsm, 0.2, now_s=500.0)
    assert out["state"] == STATE_MONITORING


# ── Mid-drive terrain confirm: DINOv2 keeps its veto after the sweep ─────
# The sweep only chooses the OPENING heading. Every later heading comes from
# the LiDAR picker, which sees geometry and nothing else, so without this the
# foundation model would have no say for the rest of the run.
#
# The test is relative, not absolute: a heading is refused when its score is
# worse than the baseline the sweep measured in THIS environment by more than
# terrain_reject_margin. An absolute cut-off cannot work here -- the same
# camera and model called a wall soil:0.937 and fired `uncertain` 0/214 times
# across a real capture, so any fixed threshold either passes everything or
# fails everything.

def _feed_confirm(fsm, score, now_s, heading_offset=0.0):
    """Deliver a terrain verdict to a waiting STARTUP_CHECK.

    Two scores, not one: the first result to arrive after a turn was captured
    while the rover was still moving (inference is slower than the settle) and
    is discarded by design, so a test that wants its score acted on has to
    supply the one after it too.
    """
    out = None
    for value in (score, score):
        fsm._last_seen_score_seq += 0      # readability: seq comes from below
        out = fsm.step(now_s=now_s, yaw=0.0, pos_x=0.0, pos_y=0.0,
                        terrain_stopped=False, lidar_stopped=False,
                        heading_offset=heading_offset,
                        traversability_score=value,
                        terrain_score_seq=fsm._last_seen_score_seq + 1)
        now_s += 1.0
    return out


def _confirmed_sweep(fsm, baseline=0.3):
    """Sweep, then stand at STARTUP_CHECK as if a LiDAR-picked turn just ended.

    The confirm deliberately does not apply to the heading the sweep itself
    chose -- DINOv2 ranked every direction and picked that one, so re-scoring
    it could only overrule the model. That grace is consumed once, and every
    test below is about the NEXT heading, which comes from the LiDAR picker and
    has never been looked at. Clearing the flag here is what "the rover already
    drove the model's choice and has since turned again" looks like.
    """
    _start_sweep(fsm)
    _drive_sweep(fsm, _uniform(score=baseline))
    assert fsm.terrain_baseline == baseline
    fsm._heading_chosen_by_model = False
    fsm._enter_startup_check()


def test_bad_terrain_after_a_turn_is_refused_and_the_heading_blacklisted():
    fsm = _make_fsm(settle_time_s=0.0, sweep_headings=8,
                    terrain_reject_margin=0.15)
    _confirmed_sweep(fsm)
    out = _feed_confirm(fsm, 0.9, now_s=100.0)
    assert out["state"] == STATE_STARTUP_CHECK      # refused, still deciding
    assert fsm.excluded_offsets(0.0) != []


def test_good_terrain_after_a_turn_is_accepted():
    fsm = _make_fsm(settle_time_s=0.0, sweep_headings=8,
                    terrain_reject_margin=0.15)
    _confirmed_sweep(fsm)
    out = _feed_confirm(fsm, 0.31, now_s=100.0)
    assert out["state"] == STATE_MONITORING
    assert fsm.excluded_offsets(0.0) == []


def test_terrain_only_slightly_worse_than_baseline_is_within_the_margin():
    # Ground is never uniform. Refusing anything at all above the baseline
    # would blacklist the whole world one heading at a time.
    fsm = _make_fsm(settle_time_s=0.0, sweep_headings=8,
                    terrain_reject_margin=0.15)
    _confirmed_sweep(fsm)
    out = _feed_confirm(fsm, 0.44, now_s=100.0)
    assert out["state"] == STATE_MONITORING


def test_confirm_waits_for_a_fresh_score_rather_than_guessing():
    # DINOv2 runs at ~0.5 Hz on the Pi against a 10 Hz FSM tick. Deciding on a
    # stale score would judge the new heading using a reading taken while the
    # rover still faced the old one.
    fsm = _make_fsm(settle_time_s=0.0, sweep_headings=8,
                    terrain_reject_margin=0.15)
    _confirmed_sweep(fsm)
    out = fsm.step(now_s=100.0, yaw=0.0, pos_x=0.0, pos_y=0.0,
                    terrain_stopped=False, lidar_stopped=False,
                    heading_offset=0.0,
                    traversability_score=None, terrain_score_seq=None)
    assert out["state"] == STATE_STARTUP_CHECK
    assert out["linear_x"] == 0.0 and out["angular_z"] == 0.0


def test_confirm_is_skipped_when_no_sweep_ever_ran():
    # Without a sweep there is no baseline, so there is nothing to compare
    # against and the rover must not stall waiting for one.
    fsm = _make_fsm(settle_time_s=0.0, sweep_headings=0)
    out = fsm.step(now_s=0.0, yaw=0.0, pos_x=0.0, pos_y=0.0,
                    terrain_stopped=False, lidar_stopped=False,
                    heading_offset=0.0)
    assert out["state"] == STATE_MONITORING


def test_every_heading_refused_by_terrain_ends_in_boxed_in():
    fsm = _make_fsm(settle_time_s=0.0, sweep_headings=8,
                    terrain_reject_margin=0.15)
    _confirmed_sweep(fsm)
    out = _feed_confirm(fsm, 0.9, now_s=100.0)
    assert out["state"] == STATE_STARTUP_CHECK
    # The picker now has nothing left to offer.
    out = fsm.step(now_s=105.0, yaw=0.0, pos_x=0.0, pos_y=0.0,
                    terrain_stopped=False, lidar_stopped=False,
                    heading_offset=None,
                    traversability_score=0.9,
                    terrain_score_seq=fsm._last_seen_score_seq + 1)
    assert out["state"] == STATE_FAILSAFE
    assert "boxed_in" in out["failsafe_reason"]


# ── Sample freshness: a score must post-date the manoeuvre it judges ─────
# Found in review, 2026-07-29. Resetting the "last seen" sequence to None on
# entering a sampling phase made ANY sequence count as fresh, including the
# inference already in flight when the rover was still turning. DINOv2 takes
# 2.2 s per frame against a 1.0 s settle, so that in-flight frame is almost
# always one captured mid-turn -- which is exactly what settling exists to
# avoid. The fix is to remember the sequence current at the moment sampling
# begins and require a strictly later one.

def test_confirm_ignores_a_score_that_predates_the_turn():
    fsm = _make_fsm(settle_time_s=0.0, sweep_headings=8,
                    terrain_reject_margin=0.15)
    _confirmed_sweep(fsm, baseline=0.3)
    # seq 7 was already in hand before the rover finished turning. Feeding it
    # back must not be accepted as a verdict on the new heading.
    stale_seq = fsm._last_seen_score_seq
    out = fsm.step(now_s=100.0, yaw=0.0, pos_x=0.0, pos_y=0.0,
                    terrain_stopped=False, lidar_stopped=False,
                    heading_offset=0.0,
                    traversability_score=0.9, terrain_score_seq=stale_seq)
    assert out["state"] == STATE_STARTUP_CHECK
    assert fsm.excluded_offsets(0.0) == []      # not judged, so not rejected


def test_confirm_accepts_the_next_score_after_the_turn():
    fsm = _make_fsm(settle_time_s=0.0, sweep_headings=8,
                    terrain_reject_margin=0.15)
    _confirmed_sweep(fsm, baseline=0.3)
    out = _feed_confirm(fsm, 0.31, now_s=100.0)
    assert out["state"] == STATE_MONITORING


def test_sweep_discards_the_inference_still_in_flight_from_the_last_turn():
    # Second, separate rule. A sequence number only says "this result is new",
    # not "the image behind it was taken after the rover stopped". DINOv2's
    # 2.2 s inference exceeds the 1.0 s settle, so the FIRST result to arrive
    # after settling was captured mid-turn however new its sequence is. It has
    # to be thrown away, and the count taken from the ones after it.
    fsm = _make_fsm(settle_time_s=0.0, sweep_headings=8,
                    sweep_samples_per_heading=1)
    _start_sweep(fsm)
    # 0.99 is the inference captured during the turn; 0.10 is the first one
    # that actually describes where the rover is now standing.
    scores = iter([0.99, 0.10])
    score = 0.99
    for _ in range(20):
        if fsm._sweep_phase == "SAMPLE":
            score = next(scores, 0.10)
        fsm.step(now_s=0.0, yaw=fsm._target_yaw, pos_x=0.0, pos_y=0.0,
                 terrain_stopped=False, lidar_stopped=False,
                 forward_clearance_m=5.0, traversability_score=score,
                 terrain_score_seq=fsm._last_seen_score_seq + 1,
                 heading_offset=0.0)
        if fsm.sweep_report:
            break
    assert len(fsm.sweep_report) == 1
    assert fsm.sweep_report[0]["score"] == 0.10


def test_confirm_discards_the_first_score_after_a_turn():
    fsm = _make_fsm(settle_time_s=0.0, sweep_headings=8,
                    terrain_reject_margin=0.15)
    _confirmed_sweep(fsm, baseline=0.3)
    seq = fsm._last_seen_score_seq
    # Captured mid-turn and terrible; must not blacklist the new heading.
    out = fsm.step(now_s=100.0, yaw=0.0, pos_x=0.0, pos_y=0.0,
                    terrain_stopped=False, lidar_stopped=False,
                    heading_offset=0.0,
                    traversability_score=0.99, terrain_score_seq=seq + 1)
    assert out["state"] == STATE_STARTUP_CHECK
    assert fsm.excluded_offsets(0.0) == []
    out = fsm.step(now_s=101.0, yaw=0.0, pos_x=0.0, pos_y=0.0,
                    terrain_stopped=False, lidar_stopped=False,
                    heading_offset=0.0,
                    traversability_score=0.31, terrain_score_seq=seq + 2)
    assert out["state"] == STATE_MONITORING


# ── The terrain confirm must be bounded ──────────────────────────────────
# Found in review, 2026-07-29. Every other state carries a wall-clock
# deadline; the confirm wait did not. If DINOv2 stops publishing -- the camera
# dies, the node crashes -- the rover held cmd_vel, commanded zero forever,
# never entered FAILSAFE and so never shut the stack down, leaving the rosbag
# recording indefinitely. Stopped, therefore safe, but neither recovering nor
# admitting it. An explicit failsafe is the honest outcome and matches the
# "camera dies, rover stops" chain the rest of the stack already demonstrates.

def test_confirm_fails_safe_when_no_score_ever_arrives():
    fsm = _make_fsm(settle_time_s=0.0, sweep_headings=8,
                    terrain_confirm_timeout_s=10.0)
    _confirmed_sweep(fsm)
    out = fsm.step(now_s=100.0, yaw=0.0, pos_x=0.0, pos_y=0.0,
                    terrain_stopped=False, lidar_stopped=False,
                    heading_offset=0.0)
    assert out["state"] == STATE_STARTUP_CHECK
    out = fsm.step(now_s=111.0, yaw=0.0, pos_x=0.0, pos_y=0.0,
                    terrain_stopped=False, lidar_stopped=False,
                    heading_offset=0.0)
    assert out["state"] == STATE_FAILSAFE
    assert "terrain_confirm_timeout" in out["failsafe_reason"]


def test_confirm_timeout_does_not_fire_once_a_score_arrives_in_time():
    fsm = _make_fsm(settle_time_s=0.0, sweep_headings=8,
                    terrain_confirm_timeout_s=10.0)
    _confirmed_sweep(fsm)
    out = _feed_confirm(fsm, 0.31, now_s=100.0)
    assert out["state"] == STATE_MONITORING


# ── Room to rotate: the box protects forward driving, not point turns ────
# Found in review, 2026-07-29, and introduced by the box itself. The forward
# corridor is a rectangle 0.20 m either side of the heading, which correctly
# lets the rover drive past a wall 0.25 m to its side. But a point turn sweeps
# a CIRCLE of the rover's corner radius -- hypot(0.20 front offset, 0.16 half
# width) = 0.256 m -- so that same wall is hit the moment the rover rotates.
# The old direction-blind 0.40 m guard covered this by accident; the box does
# not cover it at all, and the new design turns far more often than the old
# one did (eight times before it even starts driving).
#
# Geometry decides this, not preference: a rover whose corners reach 0.256 m
# genuinely cannot rotate in a 0.25 m gap. Refusing is the correct answer.

def test_turn_is_refused_when_the_corners_would_sweep_into_something():
    fsm = _make_fsm(settle_time_s=0.0, turn_radius_m=0.30)
    _clear_startup(fsm)
    out = fsm.step(now_s=1.0, yaw=0.0, pos_x=0.0, pos_y=0.0,
                    terrain_stopped=True, lidar_stopped=False,
                    forward_corridor_blocked=True,
                    heading_offset=math.radians(90.0),
                    turn_clearance_m=0.25)
    assert out["state"] == STATE_FAILSAFE
    assert "no_room_to_turn" in out["failsafe_reason"]


def test_turn_proceeds_when_the_swept_circle_is_clear():
    fsm = _make_fsm(settle_time_s=0.0, turn_radius_m=0.30)
    _clear_startup(fsm)
    out = fsm.step(now_s=1.0, yaw=0.0, pos_x=0.0, pos_y=0.0,
                    terrain_stopped=True, lidar_stopped=False,
                    forward_corridor_blocked=True,
                    heading_offset=math.radians(90.0),
                    turn_clearance_m=0.60)
    assert out["state"] == STATE_TURN_TO_HEADING


def test_no_turn_needed_means_tight_clearance_is_not_a_failure():
    # Standing still in a tight spot is fine; only rotating in one is not.
    fsm = _make_fsm(settle_time_s=0.0, turn_radius_m=0.30)
    _clear_startup(fsm)
    out = fsm.step(now_s=1.0, yaw=0.0, pos_x=0.0, pos_y=0.0,
                    terrain_stopped=True, lidar_stopped=False,
                    forward_corridor_blocked=True,
                    heading_offset=0.0,
                    turn_clearance_m=0.25)
    assert out["state"] == STATE_MONITORING


def test_sweep_refuses_to_start_without_room_to_rotate():
    # The sweep is eight consecutive point turns; if the rover cannot rotate
    # it must say so rather than grind against whatever is beside it.
    fsm = _make_fsm(settle_time_s=0.0, sweep_headings=8, turn_radius_m=0.30)
    _start_sweep(fsm)
    out = None
    # yaw stays at 0: the first stop is offset 0 and needs no turn, so the
    # refusal has to come when the sweep tries to move on to the second.
    for _ in range(40):
        out = fsm.step(now_s=0.0, yaw=0.0, pos_x=0.0, pos_y=0.0,
                       terrain_stopped=False, lidar_stopped=False,
                       forward_clearance_m=5.0, traversability_score=0.2,
                       terrain_score_seq=fsm._last_seen_score_seq + 1,
                       heading_offset=0.0, turn_clearance_m=0.25)
        if out["state"] == STATE_FAILSAFE:
            break
    assert out["state"] == STATE_FAILSAFE
    assert "no_room_to_turn" in out["failsafe_reason"]


def test_missing_turn_clearance_defaults_to_permissive():
    # Callers that do not supply it (every existing test, and any launch
    # profile without a LiDAR) must behave exactly as before.
    fsm = _make_fsm(settle_time_s=0.0, turn_radius_m=0.30)
    _clear_startup(fsm)
    out = fsm.step(now_s=1.0, yaw=0.0, pos_x=0.0, pos_y=0.0,
                    terrain_stopped=True, lidar_stopped=False,
                    forward_corridor_blocked=True,
                    heading_offset=math.radians(90.0))
    assert out["state"] == STATE_TURN_TO_HEADING


# ── Trial A, 2026-07-29 evening: the rover stalled after a perfect sweep ──
# The sweep ran, all eight headings were open, DINOv2 picked -45 at 0.004, the
# rover turned, and then it sat still for 573 s with /reactive_explorer/active
# stuck True and not one line of log. Read back from the bag, the cause was a
# chain: the rover overshot every commanded turn by about 20 deg (it turns at
# 40-90 deg/s against a commanded 8.6 deg/s, because RoverCommand.vel is not
# rad/s and the servos saturate), so the sweep labelled each sample with a
# heading the rover was not actually pointing at; the winner's recorded 0.004
# therefore belonged to a different direction than the one it turned to, which
# re-measured at 0.42; that failed the confirm, which rejected the heading and
# tried another, forever, with no deadline and no log.

def _drive_sweep_with_overshoot(fsm, per_heading, overshoot_rad,
                                settle_time_s=0.0, start_t=10.0, dt=0.1,
                                step_rad=math.radians(6.0)):
    """Like _drive_sweep, but the chassis behaves the way the real one does.

    The order matters and is the whole point. While the FSM commands rotation
    the yaw advances a fixed step per tick (the real rover turns at 40-90
    deg/s against a 10 Hz loop, so roughly 6 deg per tick). The FSM sees that
    yaw, decides it has arrived and stops commanding -- and THEN the chassis
    coasts the overshoot. Applying the overshoot while still commanding, which
    an earlier version of this helper did, makes the target unreachable and is
    not what happens.
    """
    t = start_t
    seq = 0
    yaw = 0.0
    was_turning = False
    out = None
    for _ in range(4000):
        idx = min(fsm._sweep_index, len(per_heading) - 1)
        score, blocked, clearance = per_heading[idx]
        if fsm._sweep_phase == "SAMPLE":
            seq += 1
        out = fsm.step(now_s=t, yaw=yaw, pos_x=0.0, pos_y=0.0,
                       terrain_stopped=False, lidar_stopped=False,
                       forward_corridor_blocked=blocked,
                       forward_clearance_m=clearance,
                       traversability_score=score, terrain_score_seq=seq,
                       heading_offset=0.0)
        turning = out["angular_z"] != 0.0
        if turning:
            direction = 1.0 if out["angular_z"] > 0 else -1.0
            remaining = angle_delta(fsm._target_yaw, yaw)
            advance = min(step_rad, abs(remaining)) * direction
            yaw = normalize_angle(yaw + advance)
        elif was_turning:
            direction = 1.0 if angle_delta(fsm._target_yaw, yaw) >= 0 else -1.0
            yaw = normalize_angle(yaw + direction * overshoot_rad)
        was_turning = turning
        if fsm.state != STATE_STARTUP_SWEEP:
            return out, yaw
        t += max(settle_time_s, dt)
    raise AssertionError("sweep never finished")


def test_sweep_records_the_yaw_the_rover_reached_not_the_one_it_asked_for():
    fsm = _make_fsm(settle_time_s=0.0, sweep_headings=4)
    _start_sweep(fsm)
    _drive_sweep_with_overshoot(fsm, _uniform(n=4), math.radians(20.0))
    # Every entry must carry the yaw the sample was actually taken at. Without
    # it the table is a list of scores attached to the wrong directions.
    for entry in fsm.sweep_report:
        assert "yaw" in entry, "each sweep sample must record its measured yaw"
    yaws = [round(math.degrees(e["yaw"])) for e in fsm.sweep_report]
    # Nominal stops are 0/90/180/270; a 20 deg overrun puts them elsewhere,
    # and the point is that the report says where the rover really was.
    assert yaws != [0, 90, 180, 270]


def test_sweep_turns_back_to_the_winners_measured_yaw():
    # The winner is the third stop. After the sweep the rover must aim at the
    # yaw that sample was taken at, not at the nominal offset, or it arrives
    # somewhere the foundation model never looked.
    fsm = _make_fsm(settle_time_s=0.0, sweep_headings=4)
    _start_sweep(fsm)
    per_heading = [(0.5, False, 5.0), (0.5, False, 5.0),
                   (0.01, False, 5.0), (0.5, False, 5.0)]
    _drive_sweep_with_overshoot(fsm, per_heading, math.radians(20.0))
    winner = fsm.sweep_choice
    assert math.isclose(winner["score"], 0.01, abs_tol=1e-9)
    assert math.isclose(normalize_angle(fsm._target_yaw - winner["yaw"]),
                        0.0, abs_tol=1e-6)


def test_terrain_baseline_is_the_chosen_headings_own_score():
    # Was the median of the whole sweep, which by construction rejects about
    # half of every compass the rover ever measures: with scores
    # 0.035/0.006/0.14/0.383/0.158/0.358/0.405/0.004 the median 0.149 plus a
    # 0.15 margin refuses three of the eight headings outright.
    fsm = _make_fsm(settle_time_s=0.0, sweep_headings=4)
    _start_sweep(fsm)
    per_heading = [(0.5, False, 5.0), (0.5, False, 5.0),
                   (0.01, False, 5.0), (0.5, False, 5.0)]
    _drive_sweep_with_overshoot(fsm, per_heading, math.radians(20.0))
    assert math.isclose(fsm.terrain_baseline, 0.01, abs_tol=1e-9)


def test_a_run_of_terrain_rejections_fails_safe_rather_than_looping():
    # The confirm's existing timeout only covers "no score is arriving". A
    # score that arrives and is refused resets the clock, so the rover
    # rejected, re-picked, rejected again, silently, for as long as it was
    # left running. Every other hold in this FSM has a deadline; this one did
    # not.
    fsm = _make_fsm(settle_time_s=0.0, sweep_headings=4,
                    max_terrain_rejections=3)
    _start_sweep(fsm)
    _drive_sweep(fsm, _uniform(n=4, score=0.01))
    # Carry on from where the sweep ended: the turn deadline was set during it,
    # so jumping the clock forward would trip turn_timeout instead.
    t = fsm._now_s + 0.1
    seq = 1000
    yaw = 0.0
    for _ in range(2000):
        seq += 1
        out = fsm.step(now_s=t, yaw=yaw, pos_x=0.0, pos_y=0.0,
                       terrain_stopped=False, lidar_stopped=False,
                       heading_offset=math.radians(30.0),
                       traversability_score=0.9, terrain_score_seq=seq)
        if out["angular_z"] != 0.0:
            yaw = fsm._target_yaw          # let each turn converge
        if out["state"] == STATE_FAILSAFE:
            assert "terrain" in fsm.failsafe_reason, fsm.failsafe_reason
            assert len(fsm.rejection_log) == 3
            return
        t += 0.1
    raise AssertionError("the rover rejected headings forever without failing safe")


def _grind_rejections(fsm, ticks=2000, dt=0.1, score=0.9, stop_after=None):
    """Feed refused terrain verdicts until FAILSAFE, or until `stop_after`
    rejections have been logged. Returns the last step output.

    Mirrors test_a_run_of_terrain_rejections_fails_safe_rather_than_looping's
    loop: carry the clock on from where the sweep left it (the turn deadline is
    already running) and snap yaw to the target so each turn converges in a
    tick instead of being simulated degree by degree.
    """
    t = fsm._now_s + 0.1
    seq = 1000
    yaw = 0.0
    out = None
    for _ in range(ticks):
        seq += 1
        out = fsm.step(now_s=t, yaw=yaw, pos_x=0.0, pos_y=0.0,
                       terrain_stopped=False, lidar_stopped=False,
                       heading_offset=math.radians(30.0),
                       traversability_score=score, terrain_score_seq=seq)
        if out["angular_z"] != 0.0:
            yaw = fsm._target_yaw
        if out["state"] == STATE_FAILSAFE:
            return out
        if stop_after is not None and len(fsm.rejection_log) >= stop_after:
            return out
        t += dt
    return out


def test_terrain_reject_floor_passes_sand_once_the_baseline_has_collapsed():
    # The baseline follows the ground the rover is standing on and has a
    # ceiling but no floor, so a stretch of soil (CLASS_RISK 0.0) drags it to
    # ~0 and the threshold becomes an absolute baseline+0.15. Every one of the
    # four terrain classes except soil scores above that, so from then on the
    # rover refuses sand -- the surface it had been driving on all afternoon.
    # Measured on 2026-08-04: baselines of 0.003/0.034/0.134 produced
    # thresholds of 0.153/0.184/0.284, and a 0.525 sand reading was refused at
    # the 0.284 one, which is the refusal that ended that run.
    fsm = _make_fsm(settle_time_s=0.0, sweep_headings=8,
                    terrain_reject_margin=0.15, terrain_reject_floor=0.6)
    _confirmed_sweep(fsm, baseline=0.01)
    out = _feed_confirm(fsm, 0.525, now_s=100.0)
    assert out["state"] == STATE_MONITORING, fsm.rejection_log
    assert fsm.rejection_log == []


def test_terrain_reject_floor_still_refuses_bedrock():
    # The floor sits between sand (CLASS_RISK 0.5) and bedrock (0.7) on
    # purpose. Loosening it far enough to also admit bedrock would undo the
    # behaviour H5 was run to demonstrate -- DINOv2 weighting bedrock's hazard
    # above soil's and steering away from it even when it is the more open
    # direction -- so bedrock and big_rock/uncertain (1.0) must still refuse.
    fsm = _make_fsm(settle_time_s=0.0, sweep_headings=8,
                    terrain_reject_margin=0.15, terrain_reject_floor=0.6)
    _confirmed_sweep(fsm, baseline=0.01)
    out = _feed_confirm(fsm, 0.7, now_s=100.0)
    assert out["state"] == STATE_STARTUP_CHECK
    assert len(fsm.rejection_log) == 1


def test_terrain_reject_floor_never_tightens_a_higher_threshold():
    # A floor, not a replacement. On ground that is itself risky the adaptive
    # baseline is the stricter of the two and has to keep winning, otherwise
    # this would quietly cap how much risk the rover tolerates relative to
    # where it already is.
    fsm = _make_fsm(settle_time_s=0.0, sweep_headings=8,
                    terrain_reject_margin=0.15, terrain_reject_floor=0.6)
    _confirmed_sweep(fsm, baseline=0.7)          # threshold 0.85, not 0.6
    out = _feed_confirm(fsm, 0.8, now_s=100.0)
    assert out["state"] == STATE_MONITORING, fsm.rejection_log
    assert fsm.rejection_log == []


def test_terrain_search_does_not_give_up_after_three_rejections():
    # Measured on real hardware 2026-08-04: six sandpit runs ended in FAILSAFE
    # after one or two terrain refusals, with whole arcs of the compass never
    # looked at. The picker searches the full 360 deg at 5 deg steps and each
    # refusal only blacklists +/-20 deg, so roughly nine refusals are needed
    # before every direction has genuinely been ruled out. Stopping at three
    # ended runs that still had somewhere to go.
    fsm = _make_fsm(settle_time_s=0.0, sweep_headings=4)
    _start_sweep(fsm)
    _drive_sweep(fsm, _uniform(n=4, score=0.01))
    out = _grind_rejections(fsm, stop_after=6)
    assert len(fsm.rejection_log) >= 6, fsm.rejection_log
    assert out["state"] != STATE_FAILSAFE, fsm.failsafe_reason


def test_terrain_search_fails_safe_on_its_own_deadline():
    # The count is no longer what bounds the search, so the search needs a
    # bound of its own -- otherwise this reopens exactly the silent-forever
    # loop that max_terrain_rejections was added to close. Wall clock, not
    # count: "keep looking until you have proved there is nowhere to go, but
    # not for longer than this".
    fsm = _make_fsm(settle_time_s=0.0, sweep_headings=4,
                    terrain_search_timeout_s=2.0)
    _start_sweep(fsm)
    _drive_sweep(fsm, _uniform(n=4, score=0.01))
    out = _grind_rejections(fsm)
    assert out["state"] == STATE_FAILSAFE
    assert "terrain_search_timeout" in fsm.failsafe_reason, fsm.failsafe_reason


def test_terrain_search_deadline_restarts_when_a_heading_is_accepted():
    # Same reasoning as the rejection counter resetting on acceptance: a rover
    # that refuses a patch, drives on, and later refuses another is working,
    # not stuck, and must not inherit the earlier run's clock.
    fsm = _make_fsm(settle_time_s=0.0, sweep_headings=4,
                    terrain_search_timeout_s=2.0)
    _start_sweep(fsm)
    _drive_sweep(fsm, _uniform(n=4, score=0.01))
    _grind_rejections(fsm, stop_after=1)
    assert fsm._rejection_search_start_s is not None
    fsm._heading_chosen_by_model = False
    fsm._enter_startup_check()
    _feed_confirm(fsm, 0.01, now_s=fsm._now_s + 1.0)
    assert fsm._rejection_search_start_s is None


def test_turn_stops_early_by_the_measured_yaw_rate():
    # The rover cannot be told to turn slowly: RoverCommand.vel saturates, and
    # 20/25/50 were all measured at the same speed. The only lever is stopping
    # sooner, so the turn ends when the angle still to go is smaller than what
    # the chassis covers during one actuation lag at its current rate.
    fsm = _make_fsm(settle_time_s=0.0, angle_tolerance_rad=math.radians(3.0),
                    turn_stop_lead_s=0.35)
    _clear_startup(fsm, now_s=0.0)
    fsm._origin_yaw = 0.0
    fsm._enter_turn_to_heading(math.radians(90.0))
    # Two ticks 0.1 s apart, 6 deg of yaw each: 60 deg/s. One 0.35 s lag at
    # that rate is 21 deg, so a turn with 15 deg to go must already be done.
    fsm.step(now_s=10.0, yaw=math.radians(69.0), pos_x=0.0, pos_y=0.0,
             terrain_stopped=False, lidar_stopped=False, heading_offset=0.0)
    out = fsm.step(now_s=10.1, yaw=math.radians(75.0), pos_x=0.0, pos_y=0.0,
                   terrain_stopped=False, lidar_stopped=False, heading_offset=0.0)
    assert out["angular_z"] == 0.0, (
        "15 deg to go at 60 deg/s must count as arrived once the 0.35 s "
        "actuation lag is allowed for"
    )


# ── The sweep's own winner must not be re-judged ──────────────────────────
# Raised by the user, 2026-07-29, and correct: the sweep IS DINOv2 choosing.
# It ranks every direction the camera can see and commits to the best one, and
# the rover then turned to it and put that same heading through a numeric
# threshold that could refuse it. A threshold is exactly what this stack does
# not use -- the 214-frame study found the model calling a wall soil:0.937 --
# and DINOv2 already holds a continuous veto over forward motion anyway:
# terrain_controller_node owns cmd_vel while the explorer is idle and applies
# soil 0.10 / sand 0.05 / bedrock 0.03 / big_rock 0.00 / uncertain 0.00 every
# frame. The confirm earns its place only on a heading the LIDAR picked, which
# sees geometry and no terrain at all.

def _sweep_then_settle(fsm, per_heading, score_after, start_t=10.0):
    """Run a sweep, converge the turn to its winner, then tick in
    STARTUP_CHECK feeding `score_after`. Returns the last output."""
    _start_sweep(fsm)
    out, yaw = _drive_sweep_with_overshoot(fsm, per_heading, math.radians(20.0),
                                          start_t=start_t)
    t = fsm._now_s + 0.1
    seq = 5000
    for _ in range(600):
        seq += 1
        out = fsm.step(now_s=t, yaw=yaw, pos_x=0.0, pos_y=0.0,
                       terrain_stopped=False, lidar_stopped=False,
                       heading_offset=0.0,
                       traversability_score=score_after, terrain_score_seq=seq)
        if out["angular_z"] != 0.0:
            yaw = fsm._target_yaw
        if out["state"] in (STATE_MONITORING, STATE_FAILSAFE):
            return out, yaw
        t += 0.1
    raise AssertionError(f"never settled; stuck in {fsm.state}")


def _winner_third_of_four():
    return [(0.5, False, 5.0), (0.5, False, 5.0),
            (0.01, False, 5.0), (0.5, False, 5.0)]


def test_the_heading_the_sweep_chose_is_driven_not_re_judged():
    fsm = _make_fsm(settle_time_s=0.0, sweep_headings=4)
    # 0.9 is far above anything the sweep saw, so the old code refused the
    # model's own choice and, before the rejection bound existed, forever.
    out, _ = _sweep_then_settle(fsm, _winner_third_of_four(), score_after=0.9)
    assert out["state"] == STATE_MONITORING, fsm.failsafe_reason
    assert fsm.rejection_log == [], (
        "the sweep already ranked every direction and picked this one"
    )


def test_a_lidar_picked_heading_is_still_judged_by_dinov2():
    # Mid-drive the picker is geometry-only, so without this check the model
    # would have no say in WHERE the rover goes for the rest of the run -- it
    # could only refuse to move, leaving the rover stopped instead of turning
    # towards better ground.
    fsm = _make_fsm(settle_time_s=0.0, sweep_headings=4)
    _sweep_then_settle(fsm, _winner_third_of_four(), score_after=0.02)
    assert fsm.state == STATE_MONITORING

    # The corridor blocks, so the picker offers a new heading. That one has
    # not been seen by DINOv2 and must be checked.
    t = fsm._now_s + 0.1
    seq = 9000
    yaw = fsm._last_yaw
    saw_rejection = False
    for _ in range(900):
        seq += 1
        blocked = fsm.state == STATE_MONITORING
        out = fsm.step(now_s=t, yaw=yaw, pos_x=0.0, pos_y=0.0,
                       terrain_stopped=False, lidar_stopped=False,
                       forward_corridor_blocked=blocked,
                       heading_offset=math.radians(40.0),
                       traversability_score=0.9, terrain_score_seq=seq)
        if out["angular_z"] != 0.0:
            yaw = fsm._target_yaw
        if fsm.rejection_log:
            saw_rejection = True
            break
        if out["state"] == STATE_FAILSAFE:
            break
        t += 0.1
    assert saw_rejection, (
        f"a LiDAR-picked heading scoring 0.9 was accepted; state={fsm.state} "
        f"reason={fsm.failsafe_reason}"
    )


def test_the_baseline_follows_the_ground_the_rover_is_on():
    # Frozen at the opening winner's score, the baseline judged every later
    # heading against one reading taken somewhere else: the lab floor measured
    # 0.004 to 0.405 across directions in one spot, so winner+0.15 refused most
    # of the compass for the whole run.
    fsm = _make_fsm(settle_time_s=0.0, sweep_headings=4)
    _sweep_then_settle(fsm, _winner_third_of_four(), score_after=0.02)
    assert math.isclose(fsm.terrain_baseline, 0.01, abs_tol=1e-9), (
        "straight after the sweep the baseline is still the winner's score"
    )

    # Accept a LiDAR-picked heading measuring 0.12. The baseline should now
    # describe that ground, so a later 0.25 is a small step rather than a
    # 0.24 leap above a stale 0.01.
    t = fsm._now_s + 0.1
    seq = 9000
    yaw = fsm._last_yaw
    for _ in range(900):
        seq += 1
        blocked = fsm.state == STATE_MONITORING and fsm.terrain_baseline == 0.01
        out = fsm.step(now_s=t, yaw=yaw, pos_x=0.0, pos_y=0.0,
                       terrain_stopped=False, lidar_stopped=False,
                       forward_corridor_blocked=blocked,
                       heading_offset=math.radians(40.0),
                       traversability_score=0.12, terrain_score_seq=seq)
        if out["angular_z"] != 0.0:
            yaw = fsm._target_yaw
        if fsm.terrain_baseline != 0.01:
            break
        t += 0.1
    assert math.isclose(fsm.terrain_baseline, 0.12, abs_tol=1e-9), (
        f"baseline did not track the accepted ground: {fsm.terrain_baseline}"
    )


def test_the_baseline_cannot_ratchet_above_what_the_sweep_measured():
    # Tracking must not become "accept anything": each acceptance raising the
    # bar would let the rover walk its own threshold up indefinitely. The
    # sweep measured what this environment actually offers, so the worst
    # passable direction it found is the ceiling.
    fsm = _make_fsm(settle_time_s=0.0, sweep_headings=4)
    _sweep_then_settle(fsm, _winner_third_of_four(), score_after=0.02)
    assert math.isclose(fsm.terrain_baseline_ceiling, 0.5, abs_tol=1e-9)


# ── "I cannot see" is not "the ground is bad" ─────────────────────────────
# Trial A, 2026-07-29 22:10. The rover drove, avoided a cupboard, then faced a
# featureless wall 0.28 m away. frame_detail is the standard deviation of image
# intensity, so a flat wall filling the frame read 1.708 against a threshold of
# 2.0 and the blank-frame gate fired -- the gate cannot tell a dead camera from
# a surface with no texture. dinov2 then published traversability 1.0, the same
# value it publishes for an impassable rock, and the confirm refused the
# heading three times in 0.42 s (uninformative frames publish at camera rate,
# not at the 1.5 s inference rate) and shut the run down. Refusing ground the
# rover never measured is the wrong call, and it burned the whole rejection
# budget on one unchanged situation.

def _blind_confirm(fsm, ticks, score=1.0, informative=False, start_t=None):
    t = fsm._now_s + 0.1 if start_t is None else start_t
    seq = 7000
    out = None
    for _ in range(ticks):
        seq += 1
        out = fsm.step(now_s=t, yaw=0.0, pos_x=0.0, pos_y=0.0,
                       terrain_stopped=False, lidar_stopped=False,
                       heading_offset=0.0,
                       traversability_score=score, terrain_score_seq=seq,
                       terrain_frame_informative=informative)
        if out["state"] == STATE_FAILSAFE:
            break
        t += 0.1
    return out


def test_an_uninformative_frame_is_not_counted_as_bad_terrain():
    fsm = _make_fsm(settle_time_s=0.0, sweep_headings=4,
                    terrain_confirm_timeout_s=15.0)
    _confirmed_sweep(fsm, baseline=0.3)
    out = _blind_confirm(fsm, ticks=60)          # 6 s of blindness
    assert fsm.rejection_log == [], (
        "a frame carrying no information says nothing about the ground"
    )
    assert out["state"] == STATE_STARTUP_CHECK, "it should still be waiting"


def test_lasting_blindness_times_out_with_its_own_reason():
    # It must still fail safe -- a rover that cannot see must not sit forever.
    # But the reason has to say what happened, and "rejected everywhere" would
    # claim the model judged terrain it never saw.
    fsm = _make_fsm(settle_time_s=0.0, sweep_headings=4,
                    terrain_confirm_timeout_s=2.0)
    _confirmed_sweep(fsm, baseline=0.3)
    out = _blind_confirm(fsm, ticks=400)
    assert out["state"] == STATE_FAILSAFE
    assert fsm.failsafe_reason == "terrain_confirm_timeout", fsm.failsafe_reason


def test_an_informative_frame_showing_bad_ground_is_still_refused():
    # The discrimination must not become a way to ignore real hazards.
    fsm = _make_fsm(settle_time_s=0.0, sweep_headings=4)
    _confirmed_sweep(fsm, baseline=0.3)
    _blind_confirm(fsm, ticks=40, score=0.9, informative=True)
    assert fsm.rejection_log != []


def test_a_refused_heading_makes_the_rover_turn_not_re_measure():
    # Trial A run 3, 2026-07-29 22:27. The rover refused the ground ahead three
    # times at yaw -129, -130, -130 -- it never moved between them, so the
    # rejection budget bought three readings of one unchanged view instead of
    # three candidate directions. _reject_current_heading blacklists the yaw and
    # the picker does then offer something else, but the FSM never reached the
    # code that acts on it: the confirm returns a hold, and on the next tick the
    # confirm runs again and measures the same view.
    fsm = _make_fsm(settle_time_s=0.0, sweep_headings=4,
                    max_terrain_rejections=3)
    _confirmed_sweep(fsm, baseline=0.1)
    t = fsm._now_s + 0.1
    seq = 8000
    turned = False
    for _ in range(400):
        seq += 1
        out = fsm.step(now_s=t, yaw=0.0, pos_x=0.0, pos_y=0.0,
                       terrain_stopped=False, lidar_stopped=False,
                       heading_offset=math.radians(50.0),
                       traversability_score=0.9, terrain_score_seq=seq)
        if fsm.rejection_log and out["angular_z"] != 0.0:
            turned = True
            break
        if out["state"] == STATE_FAILSAFE:
            break
        t += 0.1
    assert turned, (
        "after refusing the ground ahead the rover must turn to the next "
        f"candidate; it spent {len(fsm.rejection_log)} rejections standing "
        f"still and ended in {fsm.state} ({fsm.failsafe_reason})"
    )


# ── A heading needs somewhere to go, not just room to start ────────────────
# Observed by the user across several runs in both the lab and the sandpit, and
# it is in the logged sweep tables: the rover kept choosing directions with very
# little room to travel. The sand run of 2026-07-30 picked +0 at 0.60 m of
# clearance over +180 at 2.18 m, because the LiDAR's only question was whether
# the rover fitted through the first 0.40 m and every heading passed that, so
# clearance survived as nothing more than a tie-break.
#
# Fixed by making the LiDAR's veto stricter rather than by weighting clearance
# against the DINOv2 score. A weight would need an exchange rate between metres
# and a traversability score, and would let geometry overrule the model. This
# keeps geometry eliminating and the model choosing.

def test_a_heading_with_no_usable_run_is_not_chosen():
    fsm = _make_fsm(settle_time_s=0.0, sweep_headings=4, min_run_m=1.0)
    _start_sweep(fsm)
    # Best score has 0.6 m to travel; second best has 2.0 m. The sand run.
    per_heading = [(0.06, False, 0.60), (0.14, False, 2.00),
                   (0.30, False, 1.80), (0.50, False, 1.50)]
    _drive_sweep(fsm, per_heading)
    assert math.isclose(fsm.sweep_choice["clearance_m"], 2.00, abs_tol=1e-9), (
        "chose the shortest run available: "
        f"{fsm.sweep_choice['clearance_m']} m"
    )
    assert math.isclose(fsm.sweep_choice["score"], 0.14, abs_tol=1e-9)


def test_the_model_still_ranks_among_headings_with_room():
    # Among directions that all have somewhere to go, DINOv2 decides and
    # clearance does not. Geometry eliminates, the model chooses.
    fsm = _make_fsm(settle_time_s=0.0, sweep_headings=4, min_run_m=1.0)
    _start_sweep(fsm)
    per_heading = [(0.40, False, 3.00), (0.05, False, 1.20),
                   (0.30, False, 2.50), (0.50, False, 2.00)]
    _drive_sweep(fsm, per_heading)
    assert math.isclose(fsm.sweep_choice["score"], 0.05, abs_tol=1e-9), (
        "the longest run won instead of the best terrain"
    )


def test_a_tight_space_falls_back_rather_than_giving_up():
    # In a corner every heading may be short. Refusing them all would fail safe
    # on a rover that could still move, so the requirement relaxes rather than
    # eliminating the last option.
    fsm = _make_fsm(settle_time_s=0.0, sweep_headings=4, min_run_m=1.0)
    _start_sweep(fsm)
    per_heading = [(0.50, False, 0.55), (0.05, False, 0.60),
                   (0.30, False, 0.50), (0.40, False, 0.45)]
    _drive_sweep(fsm, per_heading)
    assert fsm.state != STATE_FAILSAFE, fsm.failsafe_reason
    assert math.isclose(fsm.sweep_choice["score"], 0.05, abs_tol=1e-9)


def test_min_run_defaults_to_off_so_gazebo_results_stand():
    fsm = _make_fsm(settle_time_s=0.0, sweep_headings=4)
    assert fsm.min_run_m == 0.0
    _start_sweep(fsm)
    per_heading = [(0.06, False, 0.60), (0.14, False, 2.00),
                   (0.30, False, 1.80), (0.50, False, 1.50)]
    _drive_sweep(fsm, per_heading)
    assert math.isclose(fsm.sweep_choice["score"], 0.06, abs_tol=1e-9)


# ── Openness weighted against the terrain score ───────────────────────────
# Chosen from the five sweep tables actually recorded on hardware, not guessed.
# Cost is score + openness_weight * (1 - min(clearance, horizon) / horizon), so
# both terms are dimensionless and the weight is the exchange rate.
#
# At weight 0 the model decided all five sweeps and picked a direction the user
# judged wrong in three of them, including the 0.60 m heading in the sandpit
# that ended in the pit wall. Any weight from 0.4 to 1.0 fixes all three while
# the score still changes the outcome in two of the five. Above 1.5 the model's
# influence collapses to one and then none. 0.5 sits in the middle of that
# plateau rather than fitted to its edge.
#
# n=5 and the "right" heading is the user's judgement by eye, not a measured
# outcome. That is enough to choose a parameter and not enough to call it
# optimal, and Chapter 5 has to say so.

def test_openness_weight_rejects_the_sandpit_choice():
    # The real numbers from the 2026-07-30 sand run.
    fsm = _make_fsm(settle_time_s=0.0, sweep_headings=4, openness_weight=0.5,
                    clearance_horizon_m=3.0)
    _start_sweep(fsm)
    per_heading = [(0.060, False, 0.60), (0.155, False, 0.96),
                   (0.138, False, 2.18), (0.311, False, 1.83)]
    _drive_sweep(fsm, per_heading)
    assert math.isclose(fsm.sweep_choice["clearance_m"], 2.18, abs_tol=1e-9), (
        f"chose {fsm.sweep_choice['clearance_m']} m again"
    )


def test_openness_weight_leaves_the_model_deciding_when_room_is_similar():
    # Lab run 1: clearances 1.71/1.35/1.27/1.33 are close, so the score should
    # still carry the decision.
    fsm = _make_fsm(settle_time_s=0.0, sweep_headings=4, openness_weight=0.5,
                    clearance_horizon_m=3.0)
    _start_sweep(fsm)
    per_heading = [(0.306, False, 1.71), (0.015, False, 1.35),
                   (0.095, False, 1.27), (0.420, False, 1.33)]
    _drive_sweep(fsm, per_heading)
    assert math.isclose(fsm.sweep_choice["score"], 0.015, abs_tol=1e-9), (
        "geometry overruled the model where the room was comparable"
    )


def test_unbounded_clearance_is_not_infinitely_attractive():
    # An 'inf' return means "nothing within the horizon", not "infinitely
    # good". Without clamping, one inf reading would win every sweep it
    # appeared in whatever the terrain looked like.
    fsm = _make_fsm(settle_time_s=0.0, sweep_headings=4, openness_weight=0.5,
                    clearance_horizon_m=3.0)
    _start_sweep(fsm)
    per_heading = [(0.90, False, float("inf")), (0.02, False, 2.90),
                   (0.50, False, 2.80), (0.60, False, 2.70)]
    _drive_sweep(fsm, per_heading)
    assert math.isclose(fsm.sweep_choice["score"], 0.02, abs_tol=1e-9), (
        "an inf clearance beat terrain that was 45x better"
    )


def test_the_sweep_records_what_geometry_alone_would_have_chosen():
    # So the log can state how often the foundation model changed the outcome,
    # rather than the thesis asserting that it did.
    fsm = _make_fsm(settle_time_s=0.0, sweep_headings=4, openness_weight=0.5,
                    clearance_horizon_m=3.0)
    _start_sweep(fsm)
    per_heading = [(0.306, False, 1.71), (0.015, False, 1.35),
                   (0.095, False, 1.27), (0.420, False, 1.33)]
    _drive_sweep(fsm, per_heading)
    assert fsm.sweep_geometry_choice is not None
    # Geometry alone takes the roomiest, 1.71 m; the model took 1.35 m.
    assert math.isclose(fsm.sweep_geometry_choice["clearance_m"], 1.71,
                        abs_tol=1e-9)
    assert fsm.sweep_model_changed_the_choice is True


def test_no_claimed_influence_when_the_model_agrees_with_geometry():
    fsm = _make_fsm(settle_time_s=0.0, sweep_headings=4, openness_weight=0.5,
                    clearance_horizon_m=3.0)
    _start_sweep(fsm)
    per_heading = [(0.02, False, 2.90), (0.50, False, 1.00),
                   (0.60, False, 1.10), (0.70, False, 1.20)]
    _drive_sweep(fsm, per_heading)
    assert fsm.sweep_model_changed_the_choice is False


def test_openness_weight_defaults_to_off():
    fsm = _make_fsm(settle_time_s=0.0, sweep_headings=4)
    assert fsm.openness_weight == 0.0


# ── correction_min_offset_rad (2026-08-04, hardware round 6 regression) ──
# min_correction_turn_deg (round 4/5's fix) was wired to bias EVERY heading
# pick, including STARTUP_CHECK's routine "is straight ahead still fine"
# recheck at the end of every turn. With a fully open circle nearly
# everywhere clears >=30deg away, so that recheck could no longer see 0.0 as
# an acceptable answer even when the corridor was not blocked -- the rover
# looped STARTUP_CHECK <-> TURN_TO_HEADING for 76+ seconds on round 6's
# sandpit run and never drove forward once. The minimum must only apply when
# there is an actual reason to turn (the forward corridor is blocked);
# never when the current heading is already fine.

def test_correction_min_offset_is_applied_when_the_corridor_is_blocked():
    assert correction_min_offset_rad(
        forward_corridor_blocked=True,
        min_correction_turn_rad=math.radians(30.0),
    ) == math.radians(30.0)


def test_correction_min_offset_is_zero_when_the_corridor_is_clear():
    # This is the exact condition STARTUP_CHECK's post-turn recheck runs
    # under when nothing is wrong -- it must get the unrestricted picker
    # answer or a good heading can never be re-confirmed.
    assert correction_min_offset_rad(
        forward_corridor_blocked=False,
        min_correction_turn_rad=math.radians(30.0),
    ) == 0.0
