"""
Purpose: Unit tests for is_slope_stop() -- the pure-Python boolean condition
         used by imu_slope_fusion_node.py to publish /imu_slope_stop, mirroring
         the IMU_STOP source decision already used in /traversability_fused.
Inputs:  None.
Outputs: pytest results.
How to run:
    cd ros2_ws && colcon build --packages-select fm_imu_fusion
    source /opt/ros/humble/setup.bash && source install/setup.bash
    python3 -m pytest src/fm_imu_fusion/test/test_imu_slope_fusion.py -v
Project: Onboard Visual Foundation Models for Mars Terrain Perception and Rover Navigation
"""
import math

from fm_imu_fusion.imu_slope_fusion_node import (
    apply_slope_override, is_slope_stop, gate_tilt, SlopeStopLatch,
)


def _artifact_accel_magnitude_g(apparent_tilt_deg: float) -> float:
    """Accel magnitude a flat-ground rover reads when horizontal acceleration
    alone produces `apparent_tilt_deg` of apparent tilt.

    Gravity contributes 1 g down; a horizontal component a_h adds in
    quadrature, so tan(theta) = a_h and |a| = 1/cos(theta). A rover genuinely
    parked on a slope of the same angle reads |a| = 1.000 g, and that gap is
    the whole basis of the accel-magnitude gate.
    """
    return 1.0 / math.cos(math.radians(apparent_tilt_deg))


def test_tilt_below_threshold_is_not_stop():
    assert is_slope_stop(tilt_deg=5.0, stop_deg=20.0) is False


def test_tilt_above_threshold_is_stop():
    assert is_slope_stop(tilt_deg=25.0, stop_deg=20.0) is True


def test_tilt_exactly_at_threshold_is_not_stop():
    # Strict greater-than, matching the existing IMU_STOP condition in _publish().
    assert is_slope_stop(tilt_deg=20.0, stop_deg=20.0) is False


# ── gate_tilt -- rejects accel-tilt spikes caused by in-place rotation ──────
# Root cause of the 2026-07-27 wall collision: the accelerometer-only tilt
# estimate reads spurious 30-40deg during a real POINT_TURN because angular
# acceleration corrupts the gravity-vector estimate. Gate on gyro yaw-rate
# instead: while the rover is actually rotating, hold the last tilt reading
# from before rotation started rather than trusting the fresh accel sample.

def test_tilt_held_when_rotating_fast():
    assert gate_tilt(new_tilt_deg=35.0, held_tilt_deg=2.0,
                      gyro_z_rad_s=0.15, gate_rad_s=0.05) == 2.0


def test_tilt_passes_through_when_not_rotating():
    assert gate_tilt(new_tilt_deg=25.0, held_tilt_deg=2.0,
                      gyro_z_rad_s=0.01, gate_rad_s=0.05) == 25.0


def test_tilt_gate_uses_absolute_gyro_rate():
    # A leftward turn (negative gyro_z) must gate just as a rightward one does.
    assert gate_tilt(new_tilt_deg=35.0, held_tilt_deg=2.0,
                      gyro_z_rad_s=-0.15, gate_rad_s=0.05) == 2.0


def test_tilt_gate_boundary_is_not_gated():
    # Strict greater-than, matching is_slope_stop's existing convention.
    assert gate_tilt(new_tilt_deg=25.0, held_tilt_deg=2.0,
                      gyro_z_rad_s=0.05, gate_rad_s=0.05) == 25.0


# ── gate_tilt -- also reject spikes from LINEAR accel and vibration ─────────
# The gyro gate above only covers rotation. Trial A on a flat lab floor
# (2026-07-29) logged apparent tilts of 15.8, 20.5, 25.6, 31.7, 33.5 and
# 62.9 deg while the rover was driving STRAIGHT in FAKE_ACKERMANN, i.e. with
# gyro_z near zero, so every one of those passed the gyro gate untouched and
# drove the rover into a reverse loop. Gravity is fixed at 1 g, so any sample
# whose accel magnitude departs from 1 g carries a non-gravitational force and
# its tilt angle cannot be trusted.

def test_tilt_held_when_accel_magnitude_departs_from_one_g():
    # 62.9 deg was the worst reading of Trial A; as a pure artifact it implies
    # 2.19 g, which no real slope can produce.
    mag = _artifact_accel_magnitude_g(62.9)
    assert gate_tilt(new_tilt_deg=62.9, held_tilt_deg=3.0,
                     gyro_z_rad_s=0.0, gate_rad_s=0.05,
                     accel_magnitude_g=mag, accel_gate_g=0.05) == 3.0


def test_tilt_held_for_artifact_right_at_the_stop_threshold():
    # The gate must still catch an artifact that only just reaches the 20 deg
    # STOP threshold (1.064 g) -- that is the weakest spike that can still
    # trigger a false stop, so it sets the ceiling on accel_gate_g.
    mag = _artifact_accel_magnitude_g(20.5)
    assert gate_tilt(new_tilt_deg=20.5, held_tilt_deg=3.0,
                     gyro_z_rad_s=0.0, gate_rad_s=0.05,
                     accel_magnitude_g=mag, accel_gate_g=0.05) == 3.0


def test_real_slope_at_one_g_is_not_gated():
    # A rover genuinely sitting on a 25 deg slope reads exactly 1 g. This is
    # the case the gate must NOT suppress, or the IMU safety layer is dead.
    assert gate_tilt(new_tilt_deg=25.0, held_tilt_deg=3.0,
                     gyro_z_rad_s=0.0, gate_rad_s=0.05,
                     accel_magnitude_g=1.0, accel_gate_g=0.05) == 25.0


def test_small_accel_noise_does_not_gate():
    # Sensor noise of a few hundredths of a g must not block every update, or
    # the held value would never refresh.
    assert gate_tilt(new_tilt_deg=12.0, held_tilt_deg=3.0,
                     gyro_z_rad_s=0.0, gate_rad_s=0.05,
                     accel_magnitude_g=1.02, accel_gate_g=0.05) == 12.0


def test_accel_gate_catches_free_fall_direction_too():
    # A wheel dropping off a lip unloads the IMU below 1 g; that is just as
    # non-gravitational as an over-reading and must gate the same way.
    assert gate_tilt(new_tilt_deg=30.0, held_tilt_deg=3.0,
                     gyro_z_rad_s=0.0, gate_rad_s=0.05,
                     accel_magnitude_g=0.90, accel_gate_g=0.05) == 3.0


def test_accel_gate_defaults_keep_the_old_two_argument_behaviour():
    # Existing callers that pass no accel data must behave exactly as before.
    assert gate_tilt(new_tilt_deg=25.0, held_tilt_deg=2.0,
                     gyro_z_rad_s=0.01, gate_rad_s=0.05) == 25.0


# ── SlopeStopLatch -- an over-tilt must persist before it is believed ───────
# Second layer behind the accel gate. A real slope does not vanish in a
# second; a wheel hitting a floor joint does. Requiring the over-tilt to hold
# for a dwell before it is acted on removes the single-sample spikes that
# survive both gates.

def test_latch_does_not_fire_on_a_single_spike():
    latch = SlopeStopLatch(stop_deg=20.0, hold_s=1.5)
    assert latch.update(tilt_deg=35.0, now_s=0.0) is False
    assert latch.update(tilt_deg=3.0, now_s=0.1) is False


def test_latch_fires_once_over_tilt_is_sustained():
    latch = SlopeStopLatch(stop_deg=20.0, hold_s=1.5)
    assert latch.update(tilt_deg=25.0, now_s=0.0) is False
    assert latch.update(tilt_deg=25.0, now_s=1.0) is False
    assert latch.update(tilt_deg=25.0, now_s=1.6) is True


def test_latch_clears_when_tilt_returns_below_threshold():
    latch = SlopeStopLatch(stop_deg=20.0, hold_s=1.5)
    latch.update(tilt_deg=25.0, now_s=0.0)
    latch.update(tilt_deg=25.0, now_s=2.0)
    assert latch.update(tilt_deg=5.0, now_s=2.1) is False


def test_latch_restarts_the_dwell_after_a_dip():
    # A dip below the threshold must reset the clock, otherwise an
    # intermittent spike train accumulates into a false stop.
    latch = SlopeStopLatch(stop_deg=20.0, hold_s=1.5)
    latch.update(tilt_deg=25.0, now_s=0.0)
    latch.update(tilt_deg=5.0, now_s=1.0)
    latch.update(tilt_deg=25.0, now_s=1.1)
    assert latch.update(tilt_deg=25.0, now_s=2.0) is False
    assert latch.update(tilt_deg=25.0, now_s=2.7) is True


def test_latch_uses_the_same_strict_threshold_as_is_slope_stop():
    latch = SlopeStopLatch(stop_deg=20.0, hold_s=0.0)
    assert latch.update(tilt_deg=20.0, now_s=0.0) is False
    assert latch.update(tilt_deg=20.1, now_s=0.1) is True


# ── The slope override can be switched off, and is off on real hardware ────
# Measured from the bag of Trial A run 2 (2026-07-29 22:24): while the rover
# drove on a flat lab floor, /exomy/imu_raw was live at 50 Hz and every message
# differed from the last -- nothing was frozen -- but the accelerometer was
# swamped by chassis vibration. |a| swung between 0.521 g and 2.306 g against a
# true 1.000 g, and the apparent tilt ranged over 10-74 deg.
#
# accel_gate_g cannot filter that out, because it gates MAGNITUDE and not
# DIRECTION: the recorded sample tilt=53.04 deg with |a|=0.979 g sits inside a
# 1.000 +/- 0.05 g gate while pointing nowhere near down. A vibration vector
# that happens to have near-unit magnitude can point anywhere.
#
# Accelerometer-only tilt is therefore unusable on this rover while it moves.
# Separating gravity from vibration needs gyro fusion, which is out of scope.
# The node keeps computing and publishing tilt so the limitation stays
# evidenced, but on real hardware nothing acts on it.

def test_slope_override_stops_the_rover_when_enabled():
    policy, speed, source = apply_slope_override(
        "SAFE", 0.10, tilt_deg=25.0, caution_deg=10.0, stop_deg=20.0,
        enabled=True,
    )
    assert (policy, speed, source) == ("STOP", 0.00, "IMU_STOP")


def test_slope_override_cautions_the_rover_when_enabled():
    policy, speed, source = apply_slope_override(
        "SAFE", 0.10, tilt_deg=15.0, caution_deg=10.0, stop_deg=20.0,
        enabled=True,
    )
    assert (policy, speed, source) == ("CAUTION", 0.05, "IMU_CAUTION")


def test_a_disabled_override_passes_the_terrain_decision_through():
    # 36.8 deg was recorded on a flat floor. With the override off the rover
    # keeps driving on what DINOv2 says, which is the whole point.
    policy, speed, source = apply_slope_override(
        "SAFE", 0.10, tilt_deg=36.8, caution_deg=10.0, stop_deg=20.0,
        enabled=False,
    )
    assert (policy, speed, source) == ("SAFE", 0.10, "DINOV2")


def test_a_disabled_override_does_not_rescue_bad_terrain():
    # Switching the IMU out must not let the rover drive on terrain DINOv2
    # refused. The terrain decision is passed through, not overridden upward.
    policy, speed, source = apply_slope_override(
        "STOP", 0.00, tilt_deg=0.0, caution_deg=10.0, stop_deg=20.0,
        enabled=False,
    )
    assert (policy, speed, source) == ("STOP", 0.00, "DINOV2")


def test_the_override_defaults_to_enabled():
    # Every Gazebo result this thesis reports ran with the IMU override live,
    # so the default must not change underneath them. It is turned off in
    # real_hardware_deployment.launch.py only.
    import inspect
    sig = inspect.signature(apply_slope_override)
    assert sig.parameters["enabled"].default is True
