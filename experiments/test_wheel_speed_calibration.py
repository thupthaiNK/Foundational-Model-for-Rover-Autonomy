#!/usr/bin/env python3
"""
Purpose: Unit tests for the pure calculation functions in
         wheel_speed_calibration_test.py (speed/duration math only --
         no ROS2, no hardware).
Inputs:  None (pytest).
Outputs: Pass/fail via pytest.
How to run:
    pytest experiments/test_wheel_speed_calibration.py -v
Project: Onboard Visual Foundation Models for Mars Terrain Perception and Rover Navigation
"""
import pytest

from wheel_speed_calibration_test import (
    compute_speed_m_s,
    compute_official_duration_s,
)


def test_compute_speed_m_s_basic():
    assert compute_speed_m_s(distance_m=1.5, duration_s=3.0) == pytest.approx(0.5)


def test_compute_speed_m_s_zero_distance():
    assert compute_speed_m_s(distance_m=0.0, duration_s=1.0) == 0.0


def test_compute_speed_m_s_rejects_nonpositive_duration():
    with pytest.raises(ValueError):
        compute_speed_m_s(distance_m=1.0, duration_s=0.0)


def test_official_duration_uses_max_when_probe_stationary():
    # probe measured no movement -- safe to use the full requested duration.
    assert compute_official_duration_s(probe_speed_m_s=0.0) == 6.0


def test_official_duration_shrinks_for_fast_probe():
    # 0.6 m/s * 6s would travel 3.6m, past the 2.2m safety target, so the
    # official duration must shrink to keep the travel distance at 2.2m.
    d = compute_official_duration_s(probe_speed_m_s=0.6)
    assert d == pytest.approx(2.2 / 0.6)
    assert d < 6.0


def test_official_duration_capped_at_max_for_slow_probe():
    # 0.1 m/s would only need 22s to hit 2.2m -- still capped at max_duration_s.
    d = compute_official_duration_s(probe_speed_m_s=0.1)
    assert d == 6.0


def test_official_duration_respects_custom_target_and_cap():
    d = compute_official_duration_s(
        probe_speed_m_s=1.0, target_distance_m=2.0, max_duration_s=4.0
    )
    assert d == 2.0


def test_official_duration_rejects_negative_probe_speed():
    with pytest.raises(ValueError):
        compute_official_duration_s(probe_speed_m_s=-0.1)
