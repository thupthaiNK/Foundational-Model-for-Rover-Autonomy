"""
Purpose: Pure-Python pure-pursuit path following for l5_lite_planner_node.py
         (backlog "L5-lite", scoped via grill-thesis 2026-07-17). No ROS2/
         Gazebo dependency, fully unit-testable in isolation. Deliberately
         simple ("lite"): a fixed-lookahead point search along the ordered
         cell-centre waypoints astar_planner.py produces, not a full
         closest-point path projection -- adequate for following an 8-
         connected grid path, not a general-purpose controller.
Inputs:  None directly; consumed by l5_lite_planner_node.py.
Outputs: find_lookahead_point(), pure_pursuit_command(), goal_reached().
How to run: Imported by l5_lite_planner_node.py. Tested via
         ros2_ws/src/fm_perception/test/test_path_follower.py
Project: Onboard Visual Foundation Models for Mars Terrain Perception and Rover Navigation
"""
import math
from typing import List, Optional, Tuple


def find_lookahead_point(path: List[Tuple[float, float]], robot_x: float, robot_y: float,
                          lookahead_m: float) -> Optional[Tuple[float, float]]:
    """First waypoint in `path` (in order) at or beyond lookahead_m from the
    robot's current position, or the final waypoint if none is far enough
    (path is shorter than the lookahead distance). None for an empty path."""
    if not path:
        return None
    for x, y in path:
        if math.hypot(x - robot_x, y - robot_y) >= lookahead_m:
            return (x, y)
    return path[-1]


def _normalize_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def pure_pursuit_command(robot_x: float, robot_y: float, robot_yaw: float,
                          target_x: float, target_y: float, linear_speed: float,
                          angular_gain: float = 2.0, max_angular_z: float = 0.3
                          ) -> Tuple[float, float]:
    """angular_z is proportional to the heading error between the robot's
    current yaw and the bearing to (target_x, target_y), clamped to
    +-max_angular_z (default 0.3 rad/s, matching the angular speed ceiling
    already established elsewhere in this codebase, e.g.
    reactive_explorer_node.py -- this platform cannot reliably achieve
    large rotations in one control tick, confirmed by the num_wheel_pairs=3
    turning investigation, §4.8.23: even after that fix, large rotations
    only reach 70-78% of commanded). linear_x is de-rated by
    cos(heading_error) -- full speed when aligned with the target, zero
    (not negative) once the target is 90+ degrees off to either side, so
    the rover rotates toward the target before committing to forward
    speed rather than fighting both at once (root cause of an earlier bug,
    2026-07-17: unclamped angular_z + constant linear_x meant a large
    initial heading error produced an unrealistic angular command while
    still commanding full forward speed, and the rover never made net
    progress). Matches the closed-loop yaw-tracking pattern already used
    elsewhere in this codebase (stuck_detection_node.py,
    reactive_explorer_node.py) -- heading error in world frame, not raw
    target-relative-to-origin angle."""
    heading_to_target = math.atan2(target_y - robot_y, target_x - robot_x)
    heading_error = _normalize_angle(heading_to_target - robot_yaw)
    angular_z = max(-max_angular_z, min(max_angular_z, angular_gain * heading_error))
    linear_x = linear_speed * max(0.0, math.cos(heading_error))
    return linear_x, angular_z


def goal_reached(robot_x: float, robot_y: float, goal_x: float, goal_y: float,
                  tolerance_m: float) -> bool:
    """True once the robot is within tolerance_m of the goal position."""
    return math.hypot(goal_x - robot_x, goal_y - robot_y) <= tolerance_m


def hybrid_rotate_command(cycle_time_s: float, rotate_phase_s: float, creep_phase_s: float,
                           angular_z_command: float, creep_speed: float) -> Tuple[float, float]:
    """Alternates between a pure-rotation phase (angular_z_command, linear_x=0)
    and a pure-creep phase (linear_x=creep_speed, angular_z=0), cycling on
    cycle_time_s modulo (rotate_phase_s + creep_phase_s). Used once a heading
    error is too large for pure_pursuit_command()'s cos()-based forward
    component to contribute anything, instead of committing to continuous
    pure rotation for the whole turn: found via systematic debugging
    (2026-07-17) that continuous pure rotation -- zero translation the whole
    time -- stops slam_toolbox from producing further /pose updates, even
    with a larger scan_queue_size, a slower rotation rate, or feature-rich
    scan-matchable geometry nearby (all three tested live in Gazebo, none
    fixed it). Periodic creep bursts give the scan-matcher translation to
    anchor a new estimate on."""
    total_period = rotate_phase_s + creep_phase_s
    phase_time = cycle_time_s % total_period
    if phase_time < rotate_phase_s:
        return 0.0, angular_z_command
    return creep_speed, 0.0


def should_use_hybrid_rotation(linear_x: float, creep_speed: float) -> bool:
    """True when pure_pursuit_command()'s forward component is slower than
    hybrid_rotate_command()'s creep bursts, meaning the hybrid pattern should
    take over. The original trigger (`linear_x == 0.0`, i.e. heading error of
    exactly 90+ degrees) left a dead band: found live 2026-07-18 (L6
    round-trip, fresh-boot machine, so compute pressure ruled out), a stale
    SLAM pose put the heading error at ~88 degrees, giving lin=0.004 with
    ang=-0.3 -- a 1.3cm-radius spin with effectively zero translation, the
    exact regime already known to stop slam_toolbox re-publishing /pose. With
    /pose frozen the heading error could never change, so the rover spun in
    place for 850s (3.01m of path length, zero displacement). Any commanded
    translation below creep speed feeds the scan-matcher strictly less than
    creeping would, so hybrid must engage over that whole band, not only at
    exactly zero."""
    return linear_x < creep_speed


def advance_waypoint_index(current_index: int, num_waypoints: int) -> Optional[int]:
    """"L6-lite" (scoped via grill-thesis 2026-07-18): the smallest possible
    sliver of L6 (Mission Autonomy) built on top of L5-lite -- once the
    current waypoint is reached, autonomously advance to the next one in a
    short list with no operator command in between, instead of stopping.
    Returns the next index, or None once current_index was the last
    waypoint (mission complete). A single-waypoint list (num_waypoints=1)
    always returns None immediately -- identical to every pre-L6-lite
    L5-lite result, which used exactly one goal."""
    next_index = current_index + 1
    if next_index >= num_waypoints:
        return None
    return next_index


def should_transition_to_return_home(frontier_selection_is_none: bool, return_home_enabled: bool,
                                      already_returning_home: bool) -> bool:
    """"Explore-then-return-home" (scoped via grill-thesis 2026-07-19): the
    frontier selector returning None means "nothing left to explore" -- by
    default (return_home_enabled=False) that ends the mission immediately,
    exactly as every pre-existing frontier-exploration-lite result. When
    return_home is enabled, that same signal should instead trigger one
    transition to driving back to the rover's recorded start pose, not end
    the mission there. already_returning_home guards against re-triggering
    the transition on every subsequent replan tick once the return leg is
    already under way (the frontier selector is not called again during the
    return leg, but the guard keeps this decision correct as a pure function
    regardless of the caller's own bookkeeping)."""
    return frontier_selection_is_none and return_home_enabled and not already_returning_home


def should_transition_to_reobservation(frontier_selection_is_none: bool,
                                        reobserve_enabled: bool,
                                        already_reobserving: bool,
                                        confidence_grid_available: bool) -> bool:
    """A2 re-observation mode (grill-scoped 2026-07-19), same shape as
    should_transition_to_return_home() above: frontier exhaustion is the
    trigger, the mode is opt-in, and the transition fires exactly once.
    confidence_grid_available fails loud-and-safe when the costmap node was
    launched without track_confidence -- with no confidence data there is
    nothing to rank, so the mission ends normally instead of entering a
    mode that could never select a target (the caller logs the
    misconfiguration; this predicate just refuses)."""
    return (frontier_selection_is_none and reobserve_enabled
            and not already_reobserving and confidence_grid_available)


def should_abort_to_home(reactive_failsafe: bool, abort_to_home_enabled: bool,
                          already_aborting: bool, frontier_mode_enabled: bool) -> bool:
    """Mission-level failsafe reaction (item 12, grill-scoped 2026-07-20): a
    reactive_explorer terminal FAILSAFE means it has given up on a specific
    hazard it cannot get around, but the rover is still mobile -- unlike
    stuck_detection's FAILSAFE (wheels commanded but the rover is not
    physically displacing), for which "drive home" would be physically
    meaningless. This predicate therefore takes no stuck_detection signal at
    all, by construction, not merely by convention. Gated to frontier_mode --
    "abort the exploration" only has meaning when a mission actually is
    exploring. already_aborting is a one-way latch, same shape as
    should_transition_to_return_home()/should_transition_to_reobservation()
    above: once fired, the caller keeps it set for the rest of the mission,
    so this predicate reports only the single upward transition and takes
    precedence over every other mode by construction of the caller's own
    if/elif ordering, not anything this pure function decides itself."""
    return (reactive_failsafe and abort_to_home_enabled
            and not already_aborting and frontier_mode_enabled)


def bootstrap_crawl_command(pose_received: bool, linear_speed: float) -> Optional[Tuple[float, float]]:
    """A stationary rover's odom->base_link TF never changes, so odom-assisted
    slam_toolbox's minimum_travel_distance/heading gate is never crossed and
    it never publishes even a first /pose -- but the planner needs /pose to
    plan, and the rover needs a plan to move: a genuine chicken-and-egg
    deadlock (found via systematic debugging, 2026-07-17, confirmed by log
    evidence: /pose stayed missing for 40+s while the costmap arrived in
    under 100ms). Returns a straight-ahead crawl command to seed
    scan-matching until the first /pose arrives, or None once it has (the
    caller should fall through to normal plan-and-follow in that case)."""
    if pose_received:
        return None
    return linear_speed, 0.0
