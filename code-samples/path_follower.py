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
Project: Foundational Model for Rover Autonomy
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
