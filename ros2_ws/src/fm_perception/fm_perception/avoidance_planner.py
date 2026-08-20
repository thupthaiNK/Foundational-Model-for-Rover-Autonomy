#!/usr/bin/env python3
"""
Purpose: Pure-Python (no rclpy) geometry for reactive_explorer_node.py's
         corridor-aware proactive obstacle avoidance. Computes how far ahead the
         rover must see clear ground before it can safely commit to a turn
         (based on measured real speed and turn-time physics), how wide an
         how wide a corridor the rover's real footprint needs,
         and ranks candidate avoidance headings by smallest deviation from
         straight ahead among those that are actually wide and clear enough to
         drive through -- not just "the beam happens to read far".
         Clearance is decided against a BOX, defined once in
         corridor_geometry.py and shared with lidar_proximity_guard_node --
         see that module for what the angular wedge it replaced got wrong.
Inputs:  None (pure functions, driven by caller-supplied LaserScan arrays).
Outputs: None.
How to run:
    cd ros2_ws && colcon build --packages-select fm_perception
    source /opt/ros/humble/setup.bash && source install/setup.bash
    python3 -m pytest src/fm_perception/test/test_avoidance_planner.py -v
Project: Onboard Visual Foundation Models for Mars Terrain Perception and Rover Navigation
"""
import math

from fm_perception.lidar_proximity_guard_node import apply_angle_mask
# The box geometry lives in its own module so lidar_proximity_guard_node can
# use the same definition without an import cycle -- see its docstring for why
# both callers must share one definition.
from fm_perception.corridor_geometry import (      # noqa: F401 (re-exported)
    corridor_box_clear, corridor_box_radius_m, corridor_clearance_m,
    near_returns as _near_returns_unmasked,
)


def near_returns(ranges, angle_min: float, angle_increment: float,
                 range_min: float, range_max: float, max_radius_m: float,
                 blocked_sectors=None) -> list:
    """corridor_geometry.near_returns() with this repo's self-view sector
    masking applied first, which is the form every caller in fm_perception
    wants. The shared module deliberately knows nothing about sectors."""
    if blocked_sectors:
        ranges = apply_angle_mask(ranges, angle_min, angle_increment,
                                  blocked_sectors)
    return _near_returns_unmasked(ranges, angle_min, angle_increment,
                                  range_min, range_max, max_radius_m)


def is_corridor_clear(ranges, angle_min: float, angle_increment: float,
                      range_min: float, range_max: float,
                      heading_offset_rad: float, half_width_m: float,
                      far_m: float, blocked_sectors=None) -> bool:
    """Convenience wrapper: near_returns() then corridor_box_clear(), for
    callers holding a raw scan and checking a single heading.

    `half_width_m` is half the width of ground the rover sweeps, including
    whatever margin the caller wants over its physical half-width, and `far_m`
    is how far ahead of the LiDAR that ground must be clear -- so it already
    includes the offset from the LiDAR to the front wheels, which sit 0.14-0.20 m
    ahead of it on this rover.
    """
    near = near_returns(ranges, angle_min, angle_increment, range_min,
                        range_max, corridor_box_radius_m(half_width_m, far_m),
                        blocked_sectors)
    return corridor_box_clear(near, heading_offset_rad, half_width_m, far_m)


def find_safe_heading(ranges, angle_min: float, angle_increment: float,
                      range_min: float, range_max: float,
                      rover_width_m: float, lookahead_m: float,
                      search_half_range_rad: float, angle_step_rad: float,
                      blocked_sectors=None, lidar_yaw_offset_rad: float = 0.0,
                      stop_at_first: bool = False,
                      lateral_margin_m: float = 0.0) -> list:
    """Ranked candidate avoidance headings (radians, offset from straight
    ahead in the rover's PHYSICAL/odom frame -- this is what the FSM adds to
    its current odom yaw to compute a turn target), smallest |deviation|
    first, among those wide and clear enough per is_corridor_clear(). Empty
    list means no safe heading exists within +/- search_half_range_rad.

    Candidates are swept 0, +step, -step, +2*step, -2*step, ... so the first
    entry that passes is always the smallest-deviation safe heading, matching
    the "turn as little as possible" preference decided for this feature.

    lidar_yaw_offset_rad corrects for the LiDAR's mounting yaw relative to
    the rover's physical front (measured live 2026-07-26, see
    experiments/measure_lidar_yaw_offset_direct.py and
    project_lidar_avoidance_hardware_trial_20260726 -- bearing 0 in a
    LaserScan message is NOT guaranteed to be the physical front).

    SIGN CONVENTION (do not guess this when re-measuring): defined as the
    LiDAR-frame bearing at which the rover's PHYSICAL FRONT appears, i.e.
    lidar_bearing = physical_offset + lidar_yaw_offset_rad. This matches
    what experiments/measure_lidar_yaw_offset_direct.py prints directly --
    plug its output straight into lidar_yaw_offset_deg with no sign flip.

    Each physical-frame candidate offset is shifted by this amount before
    being checked against the LiDAR-frame scan data in is_corridor_clear(),
    but the values returned are still in the physical frame -- callers (and
    the FSM) never need to know the offset exists.
    """
    # lateral_margin_m is clearance the rover does not physically need but
    # should still demand: the LiDAR's own yaw offset is only known to a few
    # degrees, the point turn overshoots, and the wheels slip. Without it a
    # heading whose obstacle sits 1 mm outside the chassis counts as clear.
    half_width = rover_width_m / 2.0 + lateral_margin_m
    # One reduction of the scan, shared by every candidate below. Under the old
    # wedge geometry each candidate walked its own narrow index window; the box
    # spans the whole forward half-plane, so the saving has to come from
    # discarding far returns instead of far bearings.
    near = near_returns(ranges, angle_min, angle_increment, range_min,
                        range_max, corridor_box_radius_m(half_width, lookahead_m),
                        blocked_sectors)

    offsets = [0.0]
    k = 1
    while k * angle_step_rad <= search_half_range_rad + 1e-9:
        offsets.append(k * angle_step_rad)
        offsets.append(-k * angle_step_rad)
        k += 1

    safe = []
    for offset in offsets:
        if corridor_box_clear(near, offset + lidar_yaw_offset_rad,
                              half_width, lookahead_m):
            safe.append(offset)
            if stop_at_first:
                # offsets are already ordered by increasing |deviation|, so
                # the first hit is the answer -- callers that only read
                # safe[0] (pick_heading_tiered) skip ~71 wasted corridor
                # walks per tier this way.
                return safe
    return safe


def pick_heading_tiered(ranges, angle_min: float, angle_increment: float,
                        range_min: float, range_max: float,
                        rover_width_m: float, lookahead_tiers,
                        angle_step_rad: float, blocked_sectors=None,
                        lidar_yaw_offset_rad: float = 0.0,
                        exclude_offsets=None,
                        exclude_tolerance_rad: float = 0.0,
                        lateral_margin_m: float = 0.0,
                        min_offset_rad: float = 0.0):
    """Full-360deg replacement for the old fixed +/-90deg/retreat candidate
    list (2026-07-27 redesign, after a rear-facing blind reverse caused a
    real wall collision): the rover never drives backward blind anymore, so
    instead of a short candidate list it scans the entire circle
    (search_half_range_rad=pi) via find_safe_heading() and returns the
    smallest-deviation-from-forward heading (radians, physical frame) that
    clears the first lookahead distance in lookahead_tiers.

    lookahead_tiers is tried in order -- e.g. (1.0, 0.3) tries a generous
    1.0m clearance requirement everywhere first, and only relaxes to the
    tighter 0.3m requirement if NOTHING in the full circle clears 1.0m. A
    looser tier is never consulted once a stricter one already found a
    heading, so the rover always prefers the most clearance it can get.

    exclude_offsets are headings (physical frame, radians) the CALLER has
    already ruled out on grounds this function cannot see -- in practice,
    ground DINOv2 classified as untraversable. Geometry alone would keep
    re-proposing them, so they are skipped within exclude_tolerance_rad.
    Once exclusions rule out every clear heading this returns None, which
    the caller reads the same way as being physically walled in.

    min_offset_rad (2026-08-04, hardware round 4/5 of H5 follow-up
    confirmation testing): the "smallest deviation that clears" preference
    let the rover nudge a few degrees at a time while approaching a wall
    nearly head on, drifting into it at a shallow angle instead of turning
    away. When set, headings closer to straight ahead than this are skipped
    on a first pass over every tier; only if NOTHING anywhere clears the
    minimum does the search fall back to the ordinary unrestricted answer,
    so this can only make the rover turn further, never refuse a correction
    that was otherwise available.

    Returns None if no heading clears even the loosest (last) tier anywhere
    in the full circle -- the caller should treat this as "boxed in".
    """
    exclude_offsets = exclude_offsets or []

    def _excluded(offset):
        for bad in exclude_offsets:
            delta = math.atan2(math.sin(offset - bad), math.cos(offset - bad))
            if abs(delta) <= exclude_tolerance_rad:
                return True
        return False

    def _search(require_min_offset: bool):
        # min_offset_rad (like exclude_offsets) can rule out the very first
        # ranked candidate, so the stop_at_first shortcut -- which assumes
        # the first candidate is always the answer -- must be disabled
        # whenever either filter is active.
        stop_at_first = not exclude_offsets and not (
            require_min_offset and min_offset_rad > 0.0
        )
        for lookahead_m in lookahead_tiers:
            safe = find_safe_heading(
                ranges, angle_min, angle_increment, range_min, range_max,
                rover_width_m=rover_width_m, lookahead_m=lookahead_m,
                search_half_range_rad=math.pi, angle_step_rad=angle_step_rad,
                blocked_sectors=blocked_sectors,
                lidar_yaw_offset_rad=lidar_yaw_offset_rad,
                stop_at_first=stop_at_first,
                lateral_margin_m=lateral_margin_m,
            )
            for offset in safe:
                if _excluded(offset):
                    continue
                if require_min_offset and abs(offset) < min_offset_rad:
                    continue
                return offset
        return None

    if min_offset_rad > 0.0:
        result = _search(require_min_offset=True)
        if result is not None:
            return result
        # Nothing clears the minimum anywhere -- fall back rather than
        # report boxed-in when a smaller, still-safe correction exists.
    return _search(require_min_offset=False)
