"""
Purpose: Unit tests for the pure-Python range-filtering and hysteresis functions
         used by lidar_proximity_guard_node.py (H4 wait-time prep, item 2).
Inputs:  None.
Outputs: pytest results.
How to run:
    cd ros2_ws && colcon build --packages-select fm_perception
    source /opt/ros/humble/setup.bash && source install/setup.bash
    python3 -m pytest src/fm_perception/test/test_lidar_proximity_guard.py -v
Project: Onboard Visual Foundation Models for Mars Terrain Perception and Rover Navigation
"""
import math

import pytest

from fm_perception.lidar_proximity_guard_node import (
    apply_angle_mask,
    angle_is_blocked,
    masked_fraction,
    min_valid_range,
    nearest_forward_obstacle_m,
    normalise_angle,
    parse_blocked_sectors,
    should_stop,
)


# ── min_valid_range ──────────────────────────────────────────────────────

def test_all_inf_means_clear_path():
    ranges = [float("inf")] * 10
    assert min_valid_range(ranges, range_min=0.1, range_max=12.0) == float("inf")


def test_returns_closest_finite_in_range_reading():
    ranges = [float("inf"), 3.0, 0.35, float("inf"), 5.2]
    assert min_valid_range(ranges, range_min=0.1, range_max=12.0) == 0.35


def test_ignores_nan_readings():
    ranges = [float("nan"), 0.2, float("nan")]
    assert min_valid_range(ranges, range_min=0.1, range_max=12.0) == 0.2


def test_ignores_readings_below_range_min():
    # A spurious 0.02m reading with range_min=0.1 is sensor noise, not a real hit.
    ranges = [0.02, 2.0]
    assert min_valid_range(ranges, range_min=0.1, range_max=12.0) == 2.0


def test_ignores_readings_above_range_max():
    ranges = [15.0, 4.0]
    assert min_valid_range(ranges, range_min=0.1, range_max=12.0) == 4.0


def test_empty_scan_means_clear_path():
    assert min_valid_range([], range_min=0.1, range_max=12.0) == float("inf")


# ── min_ignore (rover self-return rejection) ──────────────────────────────

def test_min_ignore_rejects_rover_footprint_returns():
    # ExoMy's centre-mounted C1 sees its own legs/wheels at ~0.08-0.17m while
    # the real obstacle is at 1.5m. With min_ignore=0.2 the near self-returns
    # must be discarded and the 1.5m obstacle reported.
    ranges = [0.08, 0.15, 0.17, 1.5]
    assert min_valid_range(ranges, range_min=0.05, range_max=16.0,
                           min_ignore=0.2) == 1.5


def test_min_ignore_all_self_returns_means_clear_path():
    # Nothing but the rover itself in view -> path is clear, not a STOP.
    ranges = [0.08, 0.12, 0.16]
    assert min_valid_range(ranges, range_min=0.05, range_max=16.0,
                           min_ignore=0.2) == float("inf")


def test_min_ignore_still_reports_real_obstacle_past_the_floor():
    # A genuine hazard at 0.3m (beyond the 0.2 footprint, inside stop_distance)
    # must still be reported so the guard can STOP for it.
    ranges = [0.10, 0.30, 2.0]
    assert min_valid_range(ranges, range_min=0.05, range_max=16.0,
                           min_ignore=0.2) == 0.30


def test_min_ignore_defaults_to_disabled():
    # Default 0.0 keeps the original behaviour for callers that don't set it.
    ranges = [0.08, 1.5]
    assert min_valid_range(ranges, range_min=0.05, range_max=16.0) == 0.08


def test_all_readings_invalid_means_clear_path():
    ranges = [float("nan"), 0.01, 20.0]
    assert min_valid_range(ranges, range_min=0.1, range_max=12.0) == float("inf")


# ── should_stop (hysteresis) ──────────────────────────────────────────────

def test_clear_path_never_stops():
    assert should_stop(float("inf"), currently_stopped=False,
                        stop_distance=0.4, resume_distance=0.5) is False
    assert should_stop(float("inf"), currently_stopped=True,
                        stop_distance=0.4, resume_distance=0.5) is False


def test_triggers_stop_when_closer_than_stop_distance():
    assert should_stop(0.30, currently_stopped=False,
                        stop_distance=0.4, resume_distance=0.5) is True


def test_does_not_trigger_stop_when_farther_than_stop_distance():
    assert should_stop(0.60, currently_stopped=False,
                        stop_distance=0.4, resume_distance=0.5) is False


def test_stays_stopped_inside_hysteresis_band():
    # 0.45m is farther than stop_distance but closer than resume_distance —
    # once stopped, must not resume yet (avoids flicker right at the edge).
    assert should_stop(0.45, currently_stopped=True,
                        stop_distance=0.4, resume_distance=0.5) is True


def test_resumes_once_past_resume_distance():
    assert should_stop(0.55, currently_stopped=True,
                        stop_distance=0.4, resume_distance=0.5) is False


def test_boundary_exactly_at_stop_distance_does_not_trigger():
    # Strict less-than: exactly at the threshold is not yet a violation.
    assert should_stop(0.4, currently_stopped=False,
                        stop_distance=0.4, resume_distance=0.5) is False


def test_boundary_exactly_at_resume_distance_resumes():
    assert should_stop(0.5, currently_stopped=True,
                        stop_distance=0.4, resume_distance=0.5) is False


# ── normalise_angle ───────────────────────────────────────────────────────

def test_normalise_angle_leaves_in_range_angles_alone():
    assert normalise_angle(0.0) == 0.0
    assert math.isclose(normalise_angle(1.0), 1.0)


def test_normalise_angle_wraps_above_pi():
    # 190 deg is the same bearing as -170 deg.
    assert math.isclose(normalise_angle(math.radians(190.0)),
                        math.radians(-170.0), abs_tol=1e-9)


def test_normalise_angle_wraps_below_minus_pi():
    assert math.isclose(normalise_angle(math.radians(-190.0)),
                        math.radians(170.0), abs_tol=1e-9)


def test_normalise_angle_pi_maps_to_minus_pi():
    # Half-open [-pi, pi) so exactly one representation of the rear bearing.
    assert math.isclose(normalise_angle(math.pi), -math.pi, abs_tol=1e-9)


# ── parse_blocked_sectors ─────────────────────────────────────────────────

def test_parse_blocked_sectors_empty_is_no_mask():
    assert parse_blocked_sectors([]) == []


def test_parse_blocked_sectors_converts_degrees_to_radians():
    sectors = parse_blocked_sectors([90.0, 120.0])
    assert len(sectors) == 1
    start, end = sectors[0]
    assert math.isclose(start, math.radians(90.0), abs_tol=1e-9)
    assert math.isclose(end, math.radians(120.0), abs_tol=1e-9)


def test_parse_blocked_sectors_accepts_multiple_pairs():
    assert len(parse_blocked_sectors([10.0, 20.0, 100.0, 130.0, -50.0, -40.0])) == 3


def test_parse_blocked_sectors_rejects_odd_length():
    # A dangling bound is a config typo; failing loudly beats silently
    # dropping half a mask the operator believed was applied.
    with pytest.raises(ValueError):
        parse_blocked_sectors([10.0, 20.0, 30.0])


def test_parse_blocked_sectors_rejects_out_of_range_degrees():
    with pytest.raises(ValueError):
        parse_blocked_sectors([0.0, 400.0])


# ── angle_is_blocked ──────────────────────────────────────────────────────

def test_angle_is_blocked_inside_simple_sector():
    sectors = parse_blocked_sectors([90.0, 120.0])
    assert angle_is_blocked(math.radians(100.0), sectors) is True


def test_angle_is_blocked_outside_simple_sector():
    sectors = parse_blocked_sectors([90.0, 120.0])
    assert angle_is_blocked(math.radians(0.0), sectors) is False
    assert angle_is_blocked(math.radians(-100.0), sectors) is False


def test_angle_is_blocked_handles_wraparound_sector():
    # The rear sector runs 170 deg -> -170 deg through +/-180.
    sectors = parse_blocked_sectors([170.0, -170.0])
    assert angle_is_blocked(math.radians(175.0), sectors) is True
    assert angle_is_blocked(math.radians(-175.0), sectors) is True
    assert angle_is_blocked(math.radians(0.0), sectors) is False
    assert angle_is_blocked(math.radians(160.0), sectors) is False


def test_angle_is_blocked_includes_both_sector_bounds():
    # Inclusive bounds: an operator writing [90, 120] means those readings
    # are structure, so do not leave a one-bin sliver of the mast unmasked.
    sectors = parse_blocked_sectors([90.0, 120.0])
    assert angle_is_blocked(math.radians(90.0), sectors) is True
    assert angle_is_blocked(math.radians(120.0), sectors) is True


def test_angle_is_blocked_normalises_the_query_angle():
    sectors = parse_blocked_sectors([170.0, -170.0])
    assert angle_is_blocked(math.radians(185.0), sectors) is True


def test_angle_is_blocked_with_no_sectors_is_always_false():
    assert angle_is_blocked(0.0, []) is False


# ── apply_angle_mask ──────────────────────────────────────────────────────

def test_apply_angle_mask_with_no_sectors_returns_ranges_unchanged():
    ranges = [1.0, 2.0, 3.0, 4.0]
    out = apply_angle_mask(ranges, angle_min=0.0,
                           angle_increment=math.radians(90.0), sectors=[])
    assert out == ranges


def test_apply_angle_mask_replaces_blocked_bearings_with_inf():
    # 4 beams at 0, 90, 180, 270 deg (270 normalises to -90).
    ranges = [5.0, 0.22, 5.0, 0.22]
    sectors = parse_blocked_sectors([80.0, 100.0])
    out = apply_angle_mask(ranges, angle_min=0.0,
                           angle_increment=math.radians(90.0), sectors=sectors)
    assert out[0] == 5.0
    assert math.isinf(out[1])       # 90 deg is masked
    assert out[2] == 5.0
    assert out[3] == 0.22           # -90 deg is not in the mask


def test_apply_angle_mask_is_the_fix_for_the_self_view_stop():
    # Live 2026-07-23/24: the mast and servo arms return 0.20-0.23m, inside
    # stop_distance 0.4 and beyond min_ignore 0.2, so the guard STOPped
    # forever. Masking their bearings must leave the real 1.5m obstacle as
    # the closest reading.
    # Beams at 0, 90, 180, -90 deg. Masking everything except the forward
    # +/-45 arc is expressed as the single wrapping sector 45 -> -45.
    ranges = [1.5, 0.21, 0.23, 0.22]
    out = apply_angle_mask(ranges, angle_min=0.0,
                           angle_increment=math.radians(90.0),
                           sectors=parse_blocked_sectors([45.0, -45.0]))
    assert min_valid_range(out, range_min=0.05, range_max=16.0,
                           min_ignore=0.2) == 1.5


def test_apply_angle_mask_does_not_hide_an_obstacle_in_a_clear_bearing():
    ranges = [0.30, 0.21, 5.0, 5.0]
    out = apply_angle_mask(ranges, angle_min=0.0,
                           angle_increment=math.radians(90.0),
                           sectors=parse_blocked_sectors([80.0, 100.0]))
    # The 0.30m hazard straight ahead survives the mask.
    assert min_valid_range(out, range_min=0.05, range_max=16.0,
                           min_ignore=0.2) == 0.30


def test_apply_angle_mask_preserves_nan_and_inf_entries():
    ranges = [float("nan"), float("inf"), 2.0, 3.0]
    out = apply_angle_mask(ranges, angle_min=0.0,
                           angle_increment=math.radians(90.0),
                           sectors=parse_blocked_sectors([170.0, -170.0]))
    assert math.isnan(out[0])
    assert math.isinf(out[1])
    assert math.isinf(out[2])       # 180 deg is masked by the wrapping sector
    assert out[3] == 3.0


# ── masked_fraction (safety reporting) ────────────────────────────────────

def test_masked_fraction_of_empty_mask_is_zero():
    assert masked_fraction([]) == 0.0


def test_masked_fraction_of_a_quarter_turn_is_one_quarter():
    assert math.isclose(masked_fraction(parse_blocked_sectors([0.0, 90.0])),
                        0.25, abs_tol=1e-6)


def test_masked_fraction_of_a_wrapping_sector():
    # 170 -> -170 is a 20 deg sector.
    assert math.isclose(masked_fraction(parse_blocked_sectors([170.0, -170.0])),
                        20.0 / 360.0, abs_tol=1e-6)


def test_masked_fraction_sums_multiple_sectors():
    frac = masked_fraction(parse_blocked_sectors([0.0, 90.0, 180.0, -90.0]))
    assert math.isclose(frac, 0.5, abs_tol=1e-6)


def test_no_object_permanently_stuck_stopped():
    # Regression guard: a NaN-only scan must read as "clear", not as
    # "no valid data -> stay stopped forever".
    ranges = [float("nan")] * 5
    r = min_valid_range(ranges, range_min=0.1, range_max=12.0)
    assert should_stop(r, currently_stopped=True,
                        stop_distance=0.4, resume_distance=0.5) is False
    assert math.isinf(r)


# ── nearest_forward_obstacle_m: the guard measures the box, not the circle ──
# Changed 2026-07-29. min_valid_range() answers "is anything close in any
# direction", which is not the question a forward-driving rover needs. A wall
# 0.25 m to the SIDE of a 0.32 m wide rover that fits past it latched a
# permanent STOP, which is why the lab runs kept halting with clear ground
# ahead; meanwhile an obstacle diagonally off the front-left corner at 0.39 m
# read as safely distant right up until a wheel hit it. Both are the same
# mistake: distance without direction. The guard now measures down the same
# rectangle the planner uses, so the two layers agree on what "in the way"
# means and differ only in how far ahead they look.

def _scan_with_point(forward_m, lateral_m, n_beams=720):
    ranges = [float("inf")] * n_beams
    increment = 2.0 * math.pi / n_beams
    angle_min = -math.pi
    bearing = math.atan2(lateral_m, forward_m)
    idx = int(round((bearing - angle_min) / increment)) % n_beams
    ranges[idx] = math.hypot(forward_m, lateral_m)
    return ranges, angle_min, increment


def test_obstacle_dead_ahead_is_measured_at_its_forward_distance():
    ranges, angle_min, inc = _scan_with_point(0.50, 0.0)
    d = nearest_forward_obstacle_m(ranges, angle_min, inc, 0.05, 16.0,
                                    half_width_m=0.20, heading_offset_rad=0.0)
    assert math.isclose(d, 0.50, abs_tol=1e-6)


def test_wall_beside_the_rover_does_not_stop_it():
    # 0.25 m to the side, outside a 0.20 m half-width: the rover fits past.
    ranges, angle_min, inc = _scan_with_point(0.0, 0.25)
    d = nearest_forward_obstacle_m(ranges, angle_min, inc, 0.05, 16.0,
                                    half_width_m=0.20, heading_offset_rad=0.0)
    assert d == float("inf")


def test_obstacle_off_the_front_corner_is_measured_by_forward_distance():
    # The cupboard case: 0.30 m ahead and 0.15 m left is a 0.335 m straight-
    # line range, which the old circular measure compared against
    # stop_distance directly. What matters is that it is 0.30 m AHEAD.
    ranges, angle_min, inc = _scan_with_point(0.30, 0.15)
    d = nearest_forward_obstacle_m(ranges, angle_min, inc, 0.05, 16.0,
                                    half_width_m=0.20, heading_offset_rad=0.0)
    # 1 mm, not 1 um: the point has to be snapped to the nearest of 720 beams,
    # so an off-axis obstacle lands up to a quarter degree from where it was
    # placed. Only obstacles dead ahead fall exactly on a beam.
    assert math.isclose(d, 0.30, abs_tol=1e-3)


def test_obstacle_behind_the_rover_is_ignored():
    ranges, angle_min, inc = _scan_with_point(-0.30, 0.0)
    d = nearest_forward_obstacle_m(ranges, angle_min, inc, 0.05, 16.0,
                                    half_width_m=0.20, heading_offset_rad=0.0)
    assert d == float("inf")


def test_heading_offset_rotates_what_counts_as_forward():
    # The rover's physical front is not bearing 0 in its own scans; the guard
    # must apply the same measured mounting offset the planner does or it
    # watches the wrong direction.
    ranges, angle_min, inc = _scan_with_point(0.0, 0.40)   # 90 deg to the left
    straight = nearest_forward_obstacle_m(ranges, angle_min, inc, 0.05, 16.0,
                                           half_width_m=0.20,
                                           heading_offset_rad=0.0)
    rotated = nearest_forward_obstacle_m(ranges, angle_min, inc, 0.05, 16.0,
                                          half_width_m=0.20,
                                          heading_offset_rad=math.pi / 2)
    assert straight == float("inf")
    assert math.isclose(rotated, 0.40, abs_tol=1e-6)


def test_returns_inside_the_footprint_are_still_ignored():
    # min_ignore's job is unchanged: the rover seeing its own structure must
    # not read as an obstacle.
    ranges, angle_min, inc = _scan_with_point(0.07, 0.0)
    d = nearest_forward_obstacle_m(ranges, angle_min, inc, 0.20, 16.0,
                                    half_width_m=0.20, heading_offset_rad=0.0)
    assert d == float("inf")


def test_nearest_of_several_obstacles_wins():
    ranges = [float("inf")] * 720
    inc = 2.0 * math.pi / 720
    for forward in (1.2, 0.45, 0.8):
        idx = int(round((0.0 + math.pi) / inc)) % 720
        ranges[idx] = min(ranges[idx], forward)
    d = nearest_forward_obstacle_m(ranges, -math.pi, inc, 0.05, 16.0,
                                    half_width_m=0.20, heading_offset_rad=0.0)
    assert math.isclose(d, 0.45, abs_tol=1e-6)
