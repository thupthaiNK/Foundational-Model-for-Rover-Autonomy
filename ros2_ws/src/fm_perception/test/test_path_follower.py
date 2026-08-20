"""
Purpose: Unit tests for path_follower.py -- pure-Python pure-pursuit path
         following used by l5_lite_planner_node.py (backlog "L5-lite",
         scoped via grill-thesis 2026-07-17). No ROS2/Gazebo dependency.
Inputs:  None.
Outputs: pytest results.
How to run:
    cd ros2_ws && colcon build --packages-select fm_perception
    source /opt/ros/humble/setup.bash && source install/setup.bash
    python3 -m pytest src/fm_perception/test/test_path_follower.py -v
Project: Onboard Visual Foundation Models for Mars Terrain Perception and Rover Navigation
"""
import math

from fm_perception.path_follower import (
    find_lookahead_point, pure_pursuit_command, goal_reached, bootstrap_crawl_command,
    hybrid_rotate_command, advance_waypoint_index, should_use_hybrid_rotation,
    should_transition_to_return_home, should_transition_to_reobservation,
    should_abort_to_home,
)


# -- find_lookahead_point() --------------------------------------------------

def test_lookahead_returns_first_point_beyond_distance():
    path = [(0.0, 0.0), (0.3, 0.0), (0.6, 0.0), (0.9, 0.0), (1.2, 0.0)]
    point = find_lookahead_point(path, robot_x=0.0, robot_y=0.0, lookahead_m=0.6)
    assert point == (0.6, 0.0)


def test_lookahead_returns_last_point_if_path_too_short():
    path = [(0.0, 0.0), (0.1, 0.0)]
    point = find_lookahead_point(path, robot_x=0.0, robot_y=0.0, lookahead_m=5.0)
    assert point == (0.1, 0.0)


def test_lookahead_empty_path_returns_none():
    assert find_lookahead_point([], robot_x=0.0, robot_y=0.0, lookahead_m=0.6) is None


# -- pure_pursuit_command() --------------------------------------------------

def test_pure_pursuit_zero_angular_when_target_straight_ahead():
    linear_x, angular_z = pure_pursuit_command(
        robot_x=0.0, robot_y=0.0, robot_yaw=0.0,
        target_x=1.0, target_y=0.0, linear_speed=0.10,
    )
    assert angular_z == 0.0
    assert linear_x == 0.10


def test_pure_pursuit_turns_left_for_target_to_the_left():
    _linear_x, angular_z = pure_pursuit_command(
        robot_x=0.0, robot_y=0.0, robot_yaw=0.0,
        target_x=0.0, target_y=1.0, linear_speed=0.10,
    )
    assert angular_z > 0.0


def test_pure_pursuit_turns_right_for_target_to_the_right():
    _linear_x, angular_z = pure_pursuit_command(
        robot_x=0.0, robot_y=0.0, robot_yaw=0.0,
        target_x=0.0, target_y=-1.0, linear_speed=0.10,
    )
    assert angular_z < 0.0


def test_pure_pursuit_accounts_for_robot_yaw_not_just_world_frame():
    # Target is straight ahead of a robot already facing +90deg (yaw=pi/2)
    # only if the target is "up" in world Y, not world X.
    linear_x, angular_z = pure_pursuit_command(
        robot_x=0.0, robot_y=0.0, robot_yaw=math.pi / 2,
        target_x=0.0, target_y=1.0, linear_speed=0.10,
    )
    assert abs(angular_z) < 1e-9
    assert linear_x == 0.10


# Root cause (2026-07-17, systematic-debugging): angular_z was unclamped
# and linear_x was constant regardless of heading error. With the rover's
# spawn heading pointed away from D1's real goal, the first lookahead
# target produced a near-180deg heading error -> angular_gain*error far
# exceeded any angular speed this platform can reliably achieve in one
# control tick (§4.8.23: even after the num_wheel_pairs=3 fix, large
# rotations only reach 70-78% of commanded), while linear_x kept
# commanding full forward speed regardless -- the rover never made net
# progress. Compared against the working bootstrap crawl (linear-only,
# zero angular, confirmed to move the rover) -- angular_z magnitude was
# the one meaningful difference. Fixed with the standard pure-pursuit
# technique: clamp angular_z to a safe ceiling, and de-rate linear_x by
# cos(heading_error) so the rover rotates toward the target before
# committing to forward speed, rather than fighting both at once.

def test_pure_pursuit_clamps_large_angular_z_to_max():
    # Target far to the side (90deg away) -> raw angular_gain*error would
    # be well above a safe ceiling; must clamp, not pass through raw.
    _linear_x, angular_z = pure_pursuit_command(
        robot_x=0.0, robot_y=0.0, robot_yaw=0.0,
        target_x=0.0, target_y=1.0, linear_speed=0.10,
        angular_gain=2.0, max_angular_z=0.3,
    )
    assert angular_z == 0.3


def test_pure_pursuit_stops_translating_when_target_is_directly_behind():
    # Target directly behind (180deg heading error) -- rotating in place
    # is correct; committing to forward speed while doing so is not.
    linear_x, _angular_z = pure_pursuit_command(
        robot_x=0.0, robot_y=0.0, robot_yaw=0.0,
        target_x=-1.0, target_y=0.0, linear_speed=0.10,
    )
    assert linear_x == 0.0


def test_pure_pursuit_reduces_speed_for_a_perpendicular_target():
    linear_x, _angular_z = pure_pursuit_command(
        robot_x=0.0, robot_y=0.0, robot_yaw=0.0,
        target_x=0.0, target_y=1.0, linear_speed=0.10,
    )
    assert linear_x < 1e-9  # cos(90deg) == 0, up to floating-point precision


def test_pure_pursuit_full_speed_when_well_aligned():
    linear_x, _angular_z = pure_pursuit_command(
        robot_x=0.0, robot_y=0.0, robot_yaw=0.0,
        target_x=1.0, target_y=0.0, linear_speed=0.10,
    )
    assert linear_x == 0.10


# -- goal_reached() -----------------------------------------------------------

def test_goal_reached_true_within_tolerance():
    assert goal_reached(0.05, 0.0, 0.0, 0.0, tolerance_m=0.2) is True


def test_goal_reached_false_outside_tolerance():
    assert goal_reached(1.0, 0.0, 0.0, 0.0, tolerance_m=0.2) is False


def test_goal_reached_true_exactly_at_tolerance_boundary():
    assert goal_reached(0.2, 0.0, 0.0, 0.0, tolerance_m=0.2) is True


# -- bootstrap_crawl_command() -----------------------------------------------
# Root cause (2026-07-17, systematic-debugging): a stationary rover's
# odom->base_link TF never changes, so slam_toolbox's minimum_travel_distance
# gate is never crossed and it never publishes even a first /pose -- but the
# planner needs /pose to plan, and the rover needs a plan to move. Chicken-
# and-egg deadlock, confirmed via direct log evidence (100% "pose=MISSING"
# over 40+ seconds while the costmap arrived in <100ms). This crawl command
# seeds scan-matching with real motion until the first /pose arrives.

def test_bootstrap_crawl_drives_forward_when_pose_never_received():
    command = bootstrap_crawl_command(pose_received=False, linear_speed=0.10)
    assert command == (0.10, 0.0)


def test_bootstrap_crawl_returns_none_once_pose_received():
    assert bootstrap_crawl_command(pose_received=True, linear_speed=0.10) is None


# -- hybrid_rotate_command() --------------------------------------------------
# Root cause (2026-07-17, systematic-debugging, reopened session): three
# hypotheses (scan_queue_size, slower rotation, feature-rich spawn geometry)
# were tested live in Gazebo and none fixed slam_toolbox never producing a
# second /pose during CONTINUOUS pure rotation (ground truth confirmed the
# rover really was rotating throughout). This suggests continuous pure
# rotation itself -- zero translation the whole time -- is what breaks
# scan-matching, not queue capacity, speed, or geometry. Alternating short
# rotate-only bursts with short creep-only bursts gives the scan-matcher
# periodic translation to anchor a new pose estimate on.

def test_hybrid_rotate_during_rotate_phase():
    linear_x, angular_z = hybrid_rotate_command(
        cycle_time_s=1.0, rotate_phase_s=3.0, creep_phase_s=2.0,
        angular_z_command=-0.3, creep_speed=0.05,
    )
    assert linear_x == 0.0
    assert angular_z == -0.3


def test_hybrid_rotate_during_creep_phase():
    linear_x, angular_z = hybrid_rotate_command(
        cycle_time_s=4.0, rotate_phase_s=3.0, creep_phase_s=2.0,
        angular_z_command=-0.3, creep_speed=0.05,
    )
    assert linear_x == 0.05
    assert angular_z == 0.0


def test_hybrid_rotate_cycles_correctly_across_multiple_periods():
    # total period = 5.0s; cycle_time_s=11.0 -> 11.0 % 5.0 = 1.0 -> rotate phase
    linear_x, angular_z = hybrid_rotate_command(
        cycle_time_s=11.0, rotate_phase_s=3.0, creep_phase_s=2.0,
        angular_z_command=-0.3, creep_speed=0.05,
    )
    assert linear_x == 0.0
    assert angular_z == -0.3


# -- should_use_hybrid_rotation() ---------------------------------------------
# Dead-band deadlock found live 2026-07-18 (L6 round-trip run 3, fresh-boot
# machine): the old trigger `linear_x == 0.0` only engages hybrid mode at
# |heading error| >= 90 deg exactly, but a stale SLAM pose left the error at
# ~88 deg -> pure pursuit commanded lin=0.004/ang=-0.3 (a 1.3cm-radius spin,
# effectively zero translation) -> slam_toolbox never re-published /pose ->
# the error could never change -> a stable deadlock for 850s. Hybrid must
# instead engage whenever pure pursuit's forward component is slower than the
# creep burst itself, since any translation below creep speed is strictly
# worse than creeping at feeding the scan-matcher.

def test_hybrid_needed_in_the_dead_band_just_under_90_degrees():
    # lin=0.004 is the exact value observed during the live deadlock.
    assert should_use_hybrid_rotation(linear_x=0.004, creep_speed=0.05) is True


def test_hybrid_needed_at_exactly_zero_forward_component():
    # Preserves the original trigger condition (|error| >= 90 deg).
    assert should_use_hybrid_rotation(linear_x=0.0, creep_speed=0.05) is True


def test_hybrid_not_needed_when_pure_pursuit_outpaces_creep():
    # 45 deg heading error at 0.10 m/s -> lin ~= 0.071 > creep 0.05.
    assert should_use_hybrid_rotation(linear_x=0.071, creep_speed=0.05) is False


def test_hybrid_not_needed_at_exactly_creep_speed():
    # At lin == creep_speed pure pursuit translates as fast as the creep
    # burst while also steering -- no reason to give up steering for it.
    assert should_use_hybrid_rotation(linear_x=0.05, creep_speed=0.05) is False


# -- advance_waypoint_index() -------------------------------------------------
# "L6-lite" (scoped via grill-thesis 2026-07-18): the smallest possible sliver
# of L6 (Mission Autonomy) -- autonomously advancing to the next waypoint in a
# short list when the current one is reached, with no operator command in
# between. Not intelligent goal selection, just sequencing -- kept deliberately
# minimal and honestly scoped as such in the thesis.

def test_advance_waypoint_index_moves_to_next_when_more_remain():
    assert advance_waypoint_index(current_index=0, num_waypoints=2) == 1


def test_advance_waypoint_index_returns_none_when_mission_complete():
    assert advance_waypoint_index(current_index=1, num_waypoints=2) is None


def test_advance_waypoint_index_single_waypoint_completes_immediately():
    # Matches pre-L6-lite behaviour: a single-waypoint mission has nothing to
    # advance to -- backward compatible with every existing L5-lite result.
    assert advance_waypoint_index(current_index=0, num_waypoints=1) is None


def test_advance_waypoint_index_advances_through_a_longer_list():
    assert advance_waypoint_index(current_index=0, num_waypoints=3) == 1
    assert advance_waypoint_index(current_index=1, num_waypoints=3) == 2
    assert advance_waypoint_index(current_index=2, num_waypoints=3) is None


# -- should_transition_to_return_home() ---------------------------------------
# Explore-then-return-home (scoped via grill-thesis 2026-07-19): once frontier
# exploration reports nothing left to select, a return_home-enabled mission
# should switch to driving back to the rover's own recorded start pose instead
# of ending immediately, exactly once -- not re-triggering on every subsequent
# replan tick once the return leg is already under way.

def test_transitions_when_exploration_complete_and_return_home_enabled():
    assert should_transition_to_return_home(
        frontier_selection_is_none=True, return_home_enabled=True,
        already_returning_home=False) is True


def test_no_transition_when_return_home_disabled():
    # Matches every pre-existing frontier-exploration-lite result: mission
    # simply ends once exploration completes.
    assert should_transition_to_return_home(
        frontier_selection_is_none=True, return_home_enabled=False,
        already_returning_home=False) is False


def test_no_transition_while_exploration_still_has_frontiers_left():
    assert should_transition_to_return_home(
        frontier_selection_is_none=False, return_home_enabled=True,
        already_returning_home=False) is False


def test_no_repeat_transition_once_already_returning_home():
    assert should_transition_to_return_home(
        frontier_selection_is_none=True, return_home_enabled=True,
        already_returning_home=True) is False


# -- should_transition_to_reobservation() --------------------------------------
# A2 re-observation mode (item 7, grill-scoped 2026-07-19): same shape as
# should_transition_to_return_home() above -- frontier exhaustion is the
# trigger, the mode is opt-in, and the transition fires exactly once. The
# extra guard (confidence_grid_available) fails loud-and-safe when the
# costmap node was launched without track_confidence: with no confidence
# data there is nothing to rank, so the mission ends normally instead of
# entering a mode that could never select anything.

def test_transitions_to_reobservation_when_exploration_complete():
    assert should_transition_to_reobservation(
        frontier_selection_is_none=True, reobserve_enabled=True,
        already_reobserving=False, confidence_grid_available=True) is True


def test_no_reobservation_when_disabled():
    assert should_transition_to_reobservation(
        frontier_selection_is_none=True, reobserve_enabled=False,
        already_reobserving=False, confidence_grid_available=True) is False


def test_no_reobservation_while_frontiers_remain():
    assert should_transition_to_reobservation(
        frontier_selection_is_none=False, reobserve_enabled=True,
        already_reobserving=False, confidence_grid_available=True) is False


def test_no_repeat_transition_once_already_reobserving():
    assert should_transition_to_reobservation(
        frontier_selection_is_none=True, reobserve_enabled=True,
        already_reobserving=True, confidence_grid_available=True) is False


def test_no_reobservation_without_confidence_data():
    assert should_transition_to_reobservation(
        frontier_selection_is_none=True, reobserve_enabled=True,
        already_reobserving=False, confidence_grid_available=False) is False


# -- should_abort_to_home() -----------------------------------------------------
# Mission-level failsafe reaction (item 12, grill-scoped 2026-07-20): a
# reactive_explorer terminal FAILSAFE means it gave up on a specific hazard it
# could not get around, but the rover is still mobile -- unlike
# stuck_detection's FAILSAFE (wheels commanded but not physically displacing),
# for which "drive home" would be physically meaningless, so this predicate
# deliberately takes no stuck_detection signal at all. Gated to frontier_mode
# only ("abort the exploration" requires an exploration to be in progress).
# already_aborting is a one-way latch, same shape as should_transition_to_
# return_home()/should_transition_to_reobservation() above.

def test_aborts_to_home_when_reactive_failsafe_fires_during_frontier_mode():
    assert should_abort_to_home(
        reactive_failsafe=True, abort_to_home_enabled=True,
        already_aborting=False, frontier_mode_enabled=True) is True


def test_no_abort_when_disabled():
    assert should_abort_to_home(
        reactive_failsafe=True, abort_to_home_enabled=False,
        already_aborting=False, frontier_mode_enabled=True) is False


def test_no_abort_without_reactive_failsafe():
    assert should_abort_to_home(
        reactive_failsafe=False, abort_to_home_enabled=True,
        already_aborting=False, frontier_mode_enabled=True) is False


def test_no_abort_outside_frontier_mode():
    # "Abort the exploration" only has meaning when a mission actually is
    # exploring -- a single-waypoint or waypoint-sequencing mission has no
    # exploration to abort.
    assert should_abort_to_home(
        reactive_failsafe=True, abort_to_home_enabled=True,
        already_aborting=False, frontier_mode_enabled=False) is False


def test_no_repeat_abort_once_already_aborting():
    assert should_abort_to_home(
        reactive_failsafe=True, abort_to_home_enabled=True,
        already_aborting=True, frontier_mode_enabled=True) is False


def test_abort_ignores_stuck_detection_by_construction():
    # There is no stuck_detection parameter at all -- the predicate cannot be
    # made to fire from a stuck_detection failsafe signal, by construction,
    # not merely by convention. This test documents that omission is
    # deliberate (a rover that cannot physically move cannot "drive home").
    import inspect
    params = list(inspect.signature(should_abort_to_home).parameters)
    assert not any("stuck" in p for p in params)
