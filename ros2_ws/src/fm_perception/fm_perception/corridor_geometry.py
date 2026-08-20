#!/usr/bin/env python3
"""
Purpose: The one definition of "is the ground the rover is about to cover
         clear", shared by the planner that chooses headings
         (avoidance_planner.py) and the independent hard-stop guard
         (lidar_proximity_guard_node.py). Both used to answer that question
         their own way -- the planner with an angular wedge, the guard with a
         360-degree nearest-return -- and both were wrong in the same
         direction on 2026-07-29, when the rover drove its front-left wheel
         into a lab cupboard that the LiDAR could see perfectly well.

         The rover sweeps a RECTANGLE: a constant width set by its chassis
         plus a margin, extending forward as far as the caller cares about.
         The wedge the planner used was one rover wide only AT the lookahead
         distance and linearly narrower closer in, converging to a point at
         the LiDAR, so at 0.30 m ahead it checked 0.096 m of a 0.32 m wide
         rover and an obstacle beside a front wheel read as clear. The
         guard's nearest-return had the opposite failure: direction-blind, so
         a wall 0.25 m to the SIDE of a rover that fits through the gap
         stopped it dead.

         Lives in its own module, with no dependency on either caller, so the
         two cannot drift apart again and neither import direction creates a
         cycle.
Inputs:  None (pure functions, driven by caller-supplied LaserScan arrays).
Outputs: None.
How to run:
    cd ros2_ws && colcon build --packages-select fm_perception
    source /opt/ros/humble/setup.bash && source install/setup.bash
    python3 -m pytest src/fm_perception/test/test_avoidance_planner.py -v
Project: Onboard Visual Foundation Models for Mars Terrain Perception and Rover Navigation
"""
import math


def corridor_box_radius_m(half_width_m: float, far_m: float) -> float:
    """Range beyond which a return cannot be inside the corridor box for ANY
    heading -- the box's own corner distance.

    This is what keeps a full +/-90 degree sweep affordable on the Pi. The
    angular window a rectangle subtends grows towards +/-90 degrees as the
    obstacle gets closer, so the old "only walk beams near the candidate
    bearing" shortcut cannot be used any more. Instead, note that the box is a
    bounded region: nothing farther away than its far corner can be inside it,
    whichever way the rover is pointing. With half_width 0.20 m and far 0.40 m
    that is 0.447 m, so all but a handful of a 720-beam scan are rejected by a
    single float comparison before any trigonometry runs.
    """
    return math.hypot(half_width_m, far_m)


def near_returns(ranges, angle_min: float, angle_increment: float,
                 range_min: float, range_max: float,
                 max_radius_m: float) -> list:
    """Reduce a scan to the returns that could possibly sit inside a corridor
    box of corner distance max_radius_m, for any heading.

    Computed once per scan and reused across every candidate heading, which is
    what makes a full-circle search affordable: a 720-beam scan in an open room
    collapses to a handful of entries, and each candidate then walks that short
    list instead of the whole scan. It replaces the pre-2026-07-29 shortcut of
    only walking beams NEAR the candidate bearing, which the box cannot use --
    a rectangle's angular extent runs out towards +/-90 degrees as an obstacle
    gets closer, and close obstacles are exactly the ones that matter.

    `ranges` must already have any self-view sectors masked out (see
    lidar_proximity_guard_node.apply_angle_mask, which sets them to +Inf).
    Keeping that out of here is what lets both the planner and the guard
    import this module without an import cycle.

    NaN is dropped rather than treated as fatal. Under the old wedge geometry
    a NaN was only ever inspected inside a ~9 degree window straight ahead, so
    failing the candidate on it was cheap insurance. The box looks across the
    entire forward half-plane, where one NaN beam would veto every heading at
    once and drop the rover into a boxed_in FAILSAFE.

    Points come back as Cartesian (x, y) in the LiDAR frame, x forward and y
    to the left, so that the tests below can rotate a candidate heading
    without any per-point trigonometry.
    """
    out = []
    for i, r in enumerate(ranges):
        if math.isnan(r) or math.isinf(r):
            continue
        if r < range_min or r > range_max or r > max_radius_m:
            continue
        bearing = angle_min + i * angle_increment
        out.append((r * math.cos(bearing), r * math.sin(bearing)))
    return out


def corridor_box_clear(near, heading_offset_rad: float,
                       half_width_m: float, far_m: float) -> bool:
    """True if nothing in `near` lies in the box the rover would sweep driving
    along heading_offset_rad.

    The box is the ground the rover actually covers: |lateral| <= half_width_m
    and 0 < forward <= far_m. Constant width the whole way along.

    Returns behind the rover (forward <= 0) never block: reverse motion was
    removed from the FSM on 2026-07-29, so no manoeuvre can drive into them.

    `near` holds Cartesian points in the LiDAR frame, so testing a heading is
    a rotation of the frame rather than of the points: with the heading's
    cosine and sine hoisted out of the loop, each point costs four multiplies
    and two adds and no trigonometry at all. That matters because the box has
    to consider the whole forward half-plane for every candidate heading,
    where the old wedge only looked at a narrow arc. Measured worst case on a
    fully boxed-in 720-beam scan: 93 ms before the 2026-07-27 optimisations,
    20.4 ms when the box first landed with per-point trigonometry, 3.3 ms once
    it rotated the frame instead.
    """
    ch = math.cos(heading_offset_rad)
    sh = math.sin(heading_offset_rad)
    for x, y in near:
        forward = x * ch + y * sh
        if forward <= 0.0 or forward > far_m:
            continue
        lateral = y * ch - x * sh
        if -half_width_m <= lateral <= half_width_m:
            return False
    return True


def corridor_clearance_m(near, heading_offset_rad: float,
                         half_width_m: float) -> float:
    """How far the rover could drive along heading_offset_rad before the first
    obstacle inside its swept width, or +inf if nothing is in the way.

    Same box as corridor_box_clear() but unbounded ahead, because this answers
    "how much room is there" rather than "is the next far_m clear".
    """
    ch = math.cos(heading_offset_rad)
    sh = math.sin(heading_offset_rad)
    best = float("inf")
    for x, y in near:
        forward = x * ch + y * sh
        if forward <= 0.0 or forward >= best:
            continue
        lateral = y * ch - x * sh
        if -half_width_m <= lateral <= half_width_m:
            best = forward
    return best
