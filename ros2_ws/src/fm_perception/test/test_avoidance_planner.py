"""
Purpose: Unit tests for the pure-Python corridor/heading-selection geometry used
         by reactive_explorer_node.py's LiDAR-based proactive obstacle avoidance.
Inputs:  None.
Outputs: pytest results.
How to run:
    cd ros2_ws && colcon build --packages-select fm_perception
    source /opt/ros/humble/setup.bash && source install/setup.bash
    python3 -m pytest src/fm_perception/test/test_avoidance_planner.py -v
Project: Onboard Visual Foundation Models for Mars Terrain Perception and Rover Navigation
"""
import math

import pytest

from fm_perception.avoidance_planner import (
    find_safe_heading,
    is_corridor_clear,
    pick_heading_tiered,
)
from fm_perception.reactive_explorer_node import ReactiveExplorerFSM


# ── is_corridor_clear (box geometry, replaced the wedge on 2026-07-29) ─────
# half_width_m 0.16 is the rover's measured half-width and far_m is measured
# from the LiDAR, so it already covers the 0.14-0.20 m from the LiDAR to the
# front wheels.

def _flat_scan(n_beams, value):
    return [value] * n_beams


def test_corridor_clear_when_all_beams_are_infinite():
    ranges = _flat_scan(360, float("inf"))
    assert is_corridor_clear(ranges, angle_min=-math.pi, angle_increment=math.radians(1.0),
                              range_min=0.05, range_max=16.0, heading_offset_rad=0.0,
                              half_width_m=0.16, far_m=1.5) is True


def test_corridor_blocked_by_a_close_beam_inside_the_box():
    ranges = _flat_scan(360, float("inf"))
    # Beam index for bearing 0deg with angle_min=-180deg, 1deg increment: index 180.
    ranges[180] = 0.5  # dead ahead, closer than the 1.5m far edge
    assert is_corridor_clear(ranges, angle_min=-math.pi, angle_increment=math.radians(1.0),
                              range_min=0.05, range_max=16.0, heading_offset_rad=0.0,
                              half_width_m=0.16, far_m=1.5) is False


def test_corridor_ignores_a_close_beam_beside_the_box():
    ranges = _flat_scan(360, float("inf"))
    ranges[270] = 0.5  # bearing -90deg: forward = 0, so beside the rover
    assert is_corridor_clear(ranges, angle_min=-math.pi, angle_increment=math.radians(1.0),
                              range_min=0.05, range_max=16.0, heading_offset_rad=0.0,
                              half_width_m=0.16, far_m=1.5) is True


def test_corridor_ignores_a_beam_behind_the_rover():
    ranges = _flat_scan(360, float("inf"))
    ranges[0] = 0.3  # bearing -180deg: directly behind
    assert is_corridor_clear(ranges, angle_min=-math.pi, angle_increment=math.radians(1.0),
                              range_min=0.05, range_max=16.0, heading_offset_rad=0.0,
                              half_width_m=0.16, far_m=1.5) is True


def test_corridor_treats_nan_as_no_detection():
    # Changed 2026-07-29 with the box geometry. The wedge only ever inspected a
    # ~9 deg window, so failing the candidate on a NaN there was cheap. The box
    # inspects the whole forward half-plane, where one NaN would veto every
    # heading at once and drop the rover straight into a boxed_in FAILSAFE.
    ranges = _flat_scan(360, float("inf"))
    ranges[180] = float("nan")
    assert is_corridor_clear(ranges, angle_min=-math.pi, angle_increment=math.radians(1.0),
                              range_min=0.05, range_max=16.0, heading_offset_rad=0.0,
                              half_width_m=0.16, far_m=1.5) is True


def test_corridor_width_does_not_narrow_at_close_range():
    # The defect that caused the 2026-07-29 cupboard collision, pinned directly:
    # an obstacle 0.15 m to the side must block at every distance along the box,
    # not only near its far edge. Under the old wedge the same obstacle read
    # clear at 0.30 m and blocked at 1.50 m.
    for forward in (0.20, 0.30, 0.60, 1.00, 1.40):
        ranges = _flat_scan(720, float("inf"))
        increment = 2.0 * math.pi / 720
        bearing = math.atan2(0.15, forward)
        idx = int(round((bearing + math.pi) / increment)) % 720
        ranges[idx] = math.hypot(forward, 0.15)
        assert is_corridor_clear(
            ranges, angle_min=-math.pi, angle_increment=increment,
            range_min=0.05, range_max=16.0, heading_offset_rad=0.0,
            half_width_m=0.16, far_m=1.5,
        ) is False, f"missed an obstacle beside the rover at {forward} m ahead"


def test_corridor_masked_self_view_sector_does_not_block_a_real_clear_heading():
    from fm_perception.lidar_proximity_guard_node import parse_blocked_sectors
    ranges = _flat_scan(360, float("inf"))
    ranges[180] = 0.075  # the rover's own mast, directly ahead
    sectors = parse_blocked_sectors([-5.0, 5.0])
    assert is_corridor_clear(ranges, angle_min=-math.pi, angle_increment=math.radians(1.0),
                              range_min=0.05, range_max=16.0, heading_offset_rad=0.0,
                              half_width_m=0.16, far_m=1.5,
                              blocked_sectors=sectors) is True


def test_corridor_out_of_range_reading_counts_as_no_obstacle():
    ranges = _flat_scan(360, float("inf"))
    ranges[180] = 20.0  # beyond range_max=16.0 -> "no detection", not a hazard
    assert is_corridor_clear(ranges, angle_min=-math.pi, angle_increment=math.radians(1.0),
                              range_min=0.05, range_max=16.0, heading_offset_rad=0.0,
                              half_width_m=0.16, far_m=1.5) is True


def test_lateral_margin_widens_the_box():
    # An obstacle just outside the chassis blocks once a margin is demanded.
    ranges = _flat_scan(720, float("inf"))
    increment = 2.0 * math.pi / 720
    forward, lateral = 0.30, 0.18
    bearing = math.atan2(lateral, forward)
    idx = int(round((bearing + math.pi) / increment)) % 720
    ranges[idx] = math.hypot(forward, lateral)
    common = dict(angle_min=-math.pi, angle_increment=increment,
                  range_min=0.05, range_max=16.0, heading_offset_rad=0.0,
                  far_m=0.40)
    assert is_corridor_clear(ranges, half_width_m=0.16, **common) is True
    assert is_corridor_clear(ranges, half_width_m=0.20, **common) is False


# ── find_safe_heading ──────────────────────────────────────────────────────

def test_find_safe_heading_picks_smallest_deviation_when_forward_is_blocked():
    ranges = _flat_scan(360, float("inf"))
    ranges[180] = 0.5  # dead ahead blocked
    result = find_safe_heading(ranges, angle_min=-math.pi, angle_increment=math.radians(1.0),
                                range_min=0.05, range_max=16.0, rover_width_m=0.32,
                                lookahead_m=1.5, search_half_range_rad=math.pi / 2,
                                angle_step_rad=math.radians(5.0))
    assert len(result) > 0
    assert min(abs(r) for r in result) == abs(result[0])
    assert result[0] != 0.0


def test_find_safe_heading_returns_empty_when_fully_blocked():
    ranges = _flat_scan(360, 0.3)  # everything close, all around
    result = find_safe_heading(ranges, angle_min=-math.pi, angle_increment=math.radians(1.0),
                                range_min=0.05, range_max=16.0, rover_width_m=0.32,
                                lookahead_m=1.5, search_half_range_rad=math.pi / 2,
                                angle_step_rad=math.radians(5.0))
    assert result == []


def test_find_safe_heading_rejects_a_gap_too_narrow_for_the_rover():
    # A narrow slot: only beams within +/-2deg of +30deg are clear, everything
    # else close. corridor_half_angle at 1.5m for a 0.32m rover is ~6deg, wider
    # than the 4deg-wide gap, so this heading must NOT be selected.
    ranges = _flat_scan(360, 0.3)
    for deg in range(28, 33):
        ranges[180 + deg] = float("inf")
    result = find_safe_heading(ranges, angle_min=-math.pi, angle_increment=math.radians(1.0),
                                range_min=0.05, range_max=16.0, rover_width_m=0.32,
                                lookahead_m=1.5, search_half_range_rad=math.pi / 2,
                                angle_step_rad=math.radians(5.0))
    assert math.radians(30.0) not in [round(r, 6) for r in result]
    for r in result:
        assert not math.isclose(r, math.radians(30.0), abs_tol=1e-6)


def test_find_safe_heading_forward_is_first_candidate_when_clear():
    ranges = _flat_scan(360, float("inf"))
    result = find_safe_heading(ranges, angle_min=-math.pi, angle_increment=math.radians(1.0),
                                range_min=0.05, range_max=16.0, rover_width_m=0.32,
                                lookahead_m=1.5, search_half_range_rad=math.pi / 2,
                                angle_step_rad=math.radians(5.0))
    assert result[0] == 0.0


# ── lidar_yaw_offset_rad ─────────────────────────────────────────────────
# Measured live 2026-07-26 (project_lidar_avoidance_hardware_trial_20260726):
# bearing 0 in a real LaserScan message was NOT the rover's physical front
# (measured offset ~22deg on that rover). These tests use a synthetic scan
# where the true obstacle sits at LiDAR bearing +30deg, not bearing 0, to
# prove the offset correctly redirects which physical-frame candidate gets
# rejected/accepted.

def test_find_safe_heading_offset_shifts_which_lidar_bearing_is_checked():
    # Obstacle occupies LiDAR bearings [25, 35]deg (everything else clear).
    # With lidar_yaw_offset_rad=30deg, physical offset 0.0 (straight ahead)
    # maps to LiDAR bearing 30deg -- inside the obstacle -- so it must be
    # rejected, even though bearing 0 itself (physical offset -30deg) is
    # actually clear.
    ranges = _flat_scan(360, float("inf"))
    for deg in range(25, 36):
        ranges[180 + deg] = 0.3
    result = find_safe_heading(ranges, angle_min=-math.pi, angle_increment=math.radians(1.0),
                                range_min=0.05, range_max=16.0, rover_width_m=0.32,
                                lookahead_m=1.5, search_half_range_rad=math.pi / 2,
                                angle_step_rad=math.radians(5.0),
                                lidar_yaw_offset_rad=math.radians(30.0))
    assert 0.0 not in result


def test_find_safe_heading_offset_of_zero_matches_no_offset_behaviour():
    ranges = _flat_scan(360, float("inf"))
    for deg in range(25, 36):
        ranges[180 + deg] = 0.3
    with_default = find_safe_heading(
        ranges, angle_min=-math.pi, angle_increment=math.radians(1.0),
        range_min=0.05, range_max=16.0, rover_width_m=0.32,
        lookahead_m=1.5, search_half_range_rad=math.pi / 2,
        angle_step_rad=math.radians(5.0))
    with_explicit_zero = find_safe_heading(
        ranges, angle_min=-math.pi, angle_increment=math.radians(1.0),
        range_min=0.05, range_max=16.0, rover_width_m=0.32,
        lookahead_m=1.5, search_half_range_rad=math.pi / 2,
        angle_step_rad=math.radians(5.0), lidar_yaw_offset_rad=0.0)
    assert with_default == with_explicit_zero


def test_find_safe_heading_offset_returns_physical_frame_not_lidar_frame():
    # With the obstacle at LiDAR bearing 0 and lidar_yaw_offset_rad=30deg,
    # physical offset 0.0 maps to LiDAR bearing 30deg -- clear -- so it must
    # be selected and returned AS 0.0 (physical frame), not as 30deg.
    ranges = _flat_scan(360, float("inf"))
    # Far enough out that at 30deg to the side its lateral offset (0.5 m)
    # clears the rover's half-width -- otherwise the box blocks physical 0.0
    # on geometry alone and this test could not isolate the frame conversion.
    ranges[180] = 1.0  # obstacle squarely at LiDAR bearing 0
    result = find_safe_heading(ranges, angle_min=-math.pi, angle_increment=math.radians(1.0),
                                range_min=0.05, range_max=16.0, rover_width_m=0.32,
                                lookahead_m=1.5, search_half_range_rad=math.pi / 2,
                                angle_step_rad=math.radians(5.0),
                                lidar_yaw_offset_rad=math.radians(30.0))
    assert result[0] == 0.0


# ── pick_heading_tiered ────────────────────────────────────────────────────
# 2026-07-27 redesign: the rover never drives backward blind anymore -- it
# always scans the full 360deg and rotates to whatever heading is open,
# trying a generous lookahead tier first and only relaxing to a tighter one
# if nothing at all qualifies. Never returns a heading from a looser tier
# when a stricter one already found something.

def test_pick_heading_tiered_prefers_first_tier_when_available():
    ranges = _flat_scan(360, float("inf"))
    result = pick_heading_tiered(
        ranges, angle_min=-math.pi, angle_increment=math.radians(1.0),
        range_min=0.05, range_max=16.0, rover_width_m=0.32,
        lookahead_tiers=(1.0, 0.3), angle_step_rad=math.radians(5.0))
    assert result == 0.0


def test_pick_heading_tiered_falls_back_to_second_tier():
    # Every beam reads 0.5m: fails the 1.0m tier everywhere, but qualifies
    # for the relaxed 0.3m tier everywhere, so tier 2 must find heading 0.0.
    ranges = _flat_scan(360, 0.5)
    result = pick_heading_tiered(
        ranges, angle_min=-math.pi, angle_increment=math.radians(1.0),
        range_min=0.05, range_max=16.0, rover_width_m=0.32,
        lookahead_tiers=(1.0, 0.3), angle_step_rad=math.radians(5.0))
    assert result == 0.0


def test_pick_heading_tiered_returns_none_when_boxed_in():
    # Every beam reads 0.1m -- closer than even the loosest tier (0.3m) can
    # tolerate anywhere -- so no heading at any tier qualifies.
    ranges = _flat_scan(360, 0.1)
    result = pick_heading_tiered(
        ranges, angle_min=-math.pi, angle_increment=math.radians(1.0),
        range_min=0.05, range_max=16.0, rover_width_m=0.32,
        lookahead_tiers=(1.0, 0.3), angle_step_rad=math.radians(5.0))
    assert result is None


def _starve_the_picker(ranges, tolerance_rad, limit=60):
    """Reject whatever the picker offers until it has nothing left.

    Returns the list of headings it offered. Feeding the picker's own choices
    back as exclusions is what reactive_explorer_node actually does during a
    run of terrain refusals, so this is the real loop rather than a hand-built
    set of exclusions.
    """
    rejected = []
    for _ in range(limit):
        offset = pick_heading_tiered(
            ranges, angle_min=-math.pi, angle_increment=math.radians(1.0),
            range_min=0.05, range_max=16.0, rover_width_m=0.32,
            lookahead_tiers=(1.0, 0.3), angle_step_rad=math.radians(5.0),
            exclude_offsets=rejected, exclude_tolerance_rad=tolerance_rad)
        if offset is None:
            return rejected
        rejected.append(offset)
    raise AssertionError(f"picker never starved within {limit} rejections")


def test_pick_heading_tiered_starves_once_its_own_offers_are_all_excluded():
    # The property the terrain search now relies on to terminate. Since
    # 2026-08-04 a run of terrain refusals is bounded by exhaustion (surfacing
    # as boxed_in) rather than by a fixed count, so "exclusions eventually
    # starve the picker" has to be true, and true on the worst case: a
    # completely clear circle, where geometry rules nothing out.
    offered = _starve_the_picker(_flat_scan(360, float("inf")),
                                 math.radians(20.0))
    # 15, not the 9 that +/-20 deg exclusions tiling a 360 deg circle would
    # suggest. The picker returns the SMALLEST-deviation heading that is not
    # excluded, so each refusal only pushes the frontier out by
    # exclude_tolerance + angle_step = 25 deg, not by the full 40 deg an
    # exclusion covers. Measured, because max_terrain_rejections has to sit
    # above it or the count fires first and exhaustion never gets to happen.
    assert len(offered) == 15, [round(math.degrees(o), 1) for o in offered]
    assert offered[:3] == pytest.approx(
        [0.0, math.radians(25.0), math.radians(-25.0)])


def test_pick_heading_tiered_starvation_stays_below_the_rejection_backstop():
    # Guards the ordering the redesign depends on: exhaustion must be reachable
    # before max_terrain_rejections trips, otherwise the count silently becomes
    # the bound again and the rover goes back to giving up with directions it
    # never looked at. Compares against the shipped value rather than a literal
    # so that lowering the parameter breaks this test rather than the rover.
    offered = _starve_the_picker(_flat_scan(360, float("inf")),
                                 math.radians(20.0))
    fsm = ReactiveExplorerFSM(stuck_dwell_s=1.0)
    assert len(offered) < fsm.max_terrain_rejections


def test_pick_heading_tiered_applies_lidar_yaw_offset():
    ranges = _flat_scan(360, float("inf"))
    # See the note in the find_safe_heading equivalent above: 0.5 m puts the
    # obstacle 0.25 m to the side of a straight-ahead rover, outside its
    # 0.16 m half-width, so only the frame conversion decides this test.
    ranges[180] = 0.5  # obstacle squarely at LiDAR bearing 0
    result = pick_heading_tiered(
        ranges, angle_min=-math.pi, angle_increment=math.radians(1.0),
        range_min=0.05, range_max=16.0, rover_width_m=0.32,
        lookahead_tiers=(1.0, 0.3), angle_step_rad=math.radians(5.0),
        lidar_yaw_offset_rad=math.radians(30.0))
    # Physical-frame 0.0 maps to LiDAR bearing 30deg, which is clear, so it
    # must still be selected even though LiDAR bearing 0 (physical -30deg)
    # is blocked.
    assert result == 0.0


# ── pick_heading_tiered: min_offset_rad (2026-08-04) ─────────────────────
# Found on hardware round 4/5 of the H5 follow-up confirmation testing: the
# picker's "smallest deviation that clears the lookahead" preference let the
# rover nudge a few degrees at a time while approaching a wall almost head
# on, drifting into it at a shallow angle instead of turning away properly.
# min_offset_rad biases the search toward a heading that is ALSO at least
# this far from straight ahead, without ever making the rover less safe than
# before: if nothing beyond the minimum clears, it falls back to the
# unrestricted search rather than reporting boxed-in.

def test_pick_heading_tiered_prefers_a_heading_at_or_beyond_minimum_offset():
    # Fully open circle: every offset clears every tier, so without
    # min_offset_rad the answer would be 0.0 (see
    # test_pick_heading_tiered_prefers_first_tier_when_available). With a
    # 30deg minimum and a 5deg step, the smallest heading that both clears
    # and is >= 30deg away is exactly +30deg (offsets are swept
    # 0, +5, -5, +10, -10, ... so +30 is reached before -30).
    ranges = _flat_scan(360, float("inf"))
    result = pick_heading_tiered(
        ranges, angle_min=-math.pi, angle_increment=math.radians(1.0),
        range_min=0.05, range_max=16.0, rover_width_m=0.32,
        lookahead_tiers=(1.0, 0.3), angle_step_rad=math.radians(5.0),
        min_offset_rad=math.radians(30.0))
    assert result == pytest.approx(math.radians(30.0))


def test_pick_heading_tiered_falls_back_below_minimum_when_nothing_else_clears():
    # Blocked everywhere except a +/-40deg opening dead ahead (0.3m all
    # around otherwise -- close enough to intrude on the 0.4m lookahead box
    # at any wider heading): only 0, +5, -5deg clear, none of them >= the
    # 30deg minimum, so a hard minimum would report boxed-in even though a
    # real, safe (if small) correction exists. The picker must fall back to
    # its ordinary unrestricted answer (0.0) rather than refuse to move.
    ranges = _flat_scan(360, 0.3)
    for deg in range(-40, 41):
        ranges[180 + deg] = float("inf")
    result = pick_heading_tiered(
        ranges, angle_min=-math.pi, angle_increment=math.radians(1.0),
        range_min=0.05, range_max=16.0, rover_width_m=0.32,
        lookahead_tiers=(0.4,), angle_step_rad=math.radians(5.0),
        min_offset_rad=math.radians(30.0))
    assert result == 0.0


def test_pick_heading_tiered_min_offset_rad_defaults_to_no_restriction():
    # Omitting min_offset_rad must reproduce the pre-2026-08-04 behaviour
    # exactly -- existing callers (and every test above) rely on this.
    ranges = _flat_scan(360, float("inf"))
    result = pick_heading_tiered(
        ranges, angle_min=-math.pi, angle_increment=math.radians(1.0),
        range_min=0.05, range_max=16.0, rover_width_m=0.32,
        lookahead_tiers=(1.0, 0.3), angle_step_rad=math.radians(5.0))
    assert result == 0.0


# ── Performance optimisations (2026-07-27) ───────────────────────────────
# Measured before optimisation on a dev PC: 74ms best case / 93ms worst for
# one pick_heading_tiered() call against a 720-beam C1 scan, against a 100ms
# tick budget at 10Hz -- and a Pi 4 runs pure-Python loops several times
# slower still. Two changes, both behaviour-preserving:
#   1. find_safe_heading(stop_at_first=True): pick_heading_tiered only ever
#      reads safe[0], so evaluating the remaining ~71 candidates is wasted.
#   2. is_corridor_clear(): skip beams that cannot matter, instead of running
#      an atan2 on all 720.
# The equivalence tests below are what license both. The shortcut in (2)
# changed on 2026-07-29 with the move from wedge to box: the old version
# skipped beams by BEARING (only those near the candidate heading), which the
# box cannot do because its angular extent runs to +/-90 deg at close range.
# The new version skips them by RANGE instead -- nothing beyond the box's far
# corner can be inside it for any heading -- so the oracle below is a naive
# box test over every beam.

def _reference_is_corridor_clear(ranges, angle_min, angle_increment, range_min,
                                  range_max, heading_offset_rad, half_width_m,
                                  far_m, blocked_sectors=None):
    """Naive box test over every beam, with no pre-filtering. Deliberately
    slow and obvious: this is the oracle, not the implementation."""
    from fm_perception.lidar_proximity_guard_node import angle_is_blocked
    blocked_sectors = blocked_sectors or []
    for i, r in enumerate(ranges):
        if math.isnan(r) or math.isinf(r):
            continue
        if r < range_min or r > range_max:
            continue
        bearing = angle_min + i * angle_increment
        if angle_is_blocked(bearing, blocked_sectors):
            continue
        beam_offset = math.atan2(math.sin(bearing - heading_offset_rad),
                                 math.cos(bearing - heading_offset_rad))
        forward = r * math.cos(beam_offset)
        lateral = r * math.sin(beam_offset)
        if 0.0 < forward <= far_m and abs(lateral) <= half_width_m:
            return False
    return True


def test_is_corridor_clear_matches_reference_on_randomised_scans():
    # Fuzz the optimised implementation against the pre-optimisation one over
    # full-circle scans containing every awkward value the real C1 produces:
    # finite ranges, +Inf (no return), NaN (bad reading), and out-of-spec.
    import random
    rng = random.Random(20260727)
    n = 720
    inc = 2 * math.pi / n
    for trial in range(200):
        ranges = []
        for _ in range(n):
            roll = rng.random()
            if roll < 0.10:
                ranges.append(float("inf"))
            elif roll < 0.13:
                ranges.append(float("nan"))
            elif roll < 0.16:
                ranges.append(0.01)          # below range_min
            else:
                ranges.append(rng.uniform(0.05, 3.0))
        heading = rng.uniform(-math.pi, math.pi)
        half = rng.uniform(0.05, 0.40)
        lookahead = rng.uniform(0.2, 2.0)
        sectors = [(math.radians(-100.0), math.radians(-80.0))] if trial % 3 == 0 else []
        expected = _reference_is_corridor_clear(
            ranges, -math.pi, inc, 0.05, 16.0, heading, half, lookahead, sectors)
        actual = is_corridor_clear(
            ranges, -math.pi, inc, 0.05, 16.0, heading, half, lookahead, sectors)
        assert actual == expected, (
            f"trial {trial}: heading={math.degrees(heading):.1f}deg "
            f"half_width={half:.2f}m far={lookahead:.2f}")


def test_is_corridor_clear_matches_reference_on_partial_arc_scan():
    # Not every caller supplies a full 360deg scan (synthetic/Gazebo scans in
    # this repo's own tests do not), so the index-window shortcut must fall
    # back correctly rather than wrap around a circle that isn't there.
    ranges = [1.0] * 180                      # 180 beams x 1deg = a 180deg arc
    inc = math.radians(1.0)
    for heading_deg in (-170, -90, 0, 45, 90, 170):
        heading = math.radians(heading_deg)
        expected = _reference_is_corridor_clear(
            ranges, -math.pi / 2, inc, 0.05, 16.0, heading, 0.16, 0.5)
        actual = is_corridor_clear(
            ranges, -math.pi / 2, inc, 0.05, 16.0, heading, 0.16, 0.5)
        assert actual == expected, f"heading={heading_deg}deg"


def test_find_safe_heading_stop_at_first_returns_only_the_best_candidate():
    ranges = _flat_scan(360, float("inf"))
    full = find_safe_heading(
        ranges, angle_min=-math.pi, angle_increment=math.radians(1.0),
        range_min=0.05, range_max=16.0, rover_width_m=0.32,
        lookahead_m=1.0, search_half_range_rad=math.pi,
        angle_step_rad=math.radians(5.0))
    first = find_safe_heading(
        ranges, angle_min=-math.pi, angle_increment=math.radians(1.0),
        range_min=0.05, range_max=16.0, rover_width_m=0.32,
        lookahead_m=1.0, search_half_range_rad=math.pi,
        angle_step_rad=math.radians(5.0), stop_at_first=True)
    assert len(full) > 1          # the unoptimised call really does find many
    assert first == [full[0]]     # and the fast path agrees on the winner


def test_find_safe_heading_stop_at_first_still_empty_when_fully_blocked():
    ranges = _flat_scan(360, 0.1)
    assert find_safe_heading(
        ranges, angle_min=-math.pi, angle_increment=math.radians(1.0),
        range_min=0.05, range_max=16.0, rover_width_m=0.32,
        lookahead_m=1.0, search_half_range_rad=math.pi,
        angle_step_rad=math.radians(5.0), stop_at_first=True) == []


def test_pick_heading_tiered_is_fast_enough_for_a_10hz_tick():
    # Regression guard: a 720-beam C1 scan must be evaluated well inside the
    # 100ms tick budget even in the worst case -- boxed in at 0.1 m on every
    # bearing, so the range pre-filter discards nothing and every candidate is
    # tested. Uses the shipped 2 deg step, which is 2.5x the candidates the
    # old 5 deg step produced. Threshold is deliberately loose so this is a
    # bug-catcher, not a flaky microbenchmark. History on this machine: 93ms
    # before the 2026-07-27 optimisations, 20.4ms when the box geometry first
    # landed with per-point trigonometry, 3.3ms once corridor_box_clear
    # switched to rotating the frame instead of the points.
    import time
    ranges = _flat_scan(720, 0.1)
    t0 = time.perf_counter()
    pick_heading_tiered(
        ranges, angle_min=-math.pi, angle_increment=math.radians(0.5),
        range_min=0.05, range_max=16.0, rover_width_m=0.32,
        lookahead_tiers=(0.40,), angle_step_rad=math.radians(2.0))
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    assert elapsed_ms < 20.0, f"worst-case pick took {elapsed_ms:.1f}ms"


# ── The corridor is a WEDGE, not a band -- root cause of the 2026-07-29
# ── cupboard collision ─────────────────────────────────────────────────────
# is_corridor_clear() bounds the far end with a flat line (required_range =
# lookahead / cos(beam_offset)) but bounds the sides with a fixed half-angle
# taken from atan2(rover_width/2, lookahead). That angle makes
# the checked region exactly one rover wide AT the lookahead distance and
# linearly narrower everywhere closer, converging to a point at the LiDAR. So
# the region swept by the rover's own body -- a constant 0.32 m wide all the
# way along -- sticks out of the region that is actually checked, and an
# obstacle beside the rover at close range is invisible to the planner while
# being squarely in the path of a front wheel.
#
# Trial A (2026-07-29) ran with lookahead 1.0 m, giving a half-angle of 9.1 deg.
# At 0.30 m ahead that wedge is only 0.096 m wide, less than a third of the
# rover, which is how the rover drove its front-left wheel into a flat 2 m
# cupboard that the LiDAR could see perfectly well.

_LIDAR_TO_FRONT_WHEEL_M = 0.20   # measured on the rover, 2026-07-29
_ROVER_HALF_WIDTH_M = 0.16       # measured on the rover, 2026-07-29


def _scan_with_point_obstacle(forward_m, lateral_m, n_beams=720):
    """A 360 deg scan that is empty except for one return at (forward, lateral),
    x forward and y to the left, in metres from the LiDAR."""
    ranges = [float("inf")] * n_beams
    increment = 2.0 * math.pi / n_beams
    angle_min = -math.pi
    bearing = math.atan2(lateral_m, forward_m)
    distance = math.hypot(forward_m, lateral_m)
    idx = int(round((bearing - angle_min) / increment)) % n_beams
    ranges[idx] = distance
    return ranges, angle_min, increment


def test_obstacle_beside_the_front_wheel_blocks_the_corridor():
    # An obstacle 0.30 m ahead and 0.15 m to the left is inside the rover's
    # 0.16 m half-width, i.e. the front-left wheel drives straight into it.
    ranges, angle_min, increment = _scan_with_point_obstacle(0.30, 0.15)
    assert is_corridor_clear(
        ranges, angle_min=angle_min, angle_increment=increment,
        range_min=0.05, range_max=16.0, heading_offset_rad=0.0,
        half_width_m=_ROVER_HALF_WIDTH_M, far_m=1.0,
    ) is False


def test_obstacle_just_outside_the_rover_width_leaves_the_corridor_clear():
    # The complement of the test above: 0.30 m ahead but 0.40 m to the left is
    # well clear of a 0.32 m wide rover and must NOT block it, or the rover can
    # never drive down anything narrower than a corridor.
    ranges, angle_min, increment = _scan_with_point_obstacle(0.30, 0.40)
    assert is_corridor_clear(
        ranges, angle_min=angle_min, angle_increment=increment,
        range_min=0.05, range_max=16.0, heading_offset_rad=0.0,
        half_width_m=_ROVER_HALF_WIDTH_M, far_m=1.0,
    ) is True
