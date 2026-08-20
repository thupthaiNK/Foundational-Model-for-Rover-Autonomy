"""
Purpose: Property-based tests (hypothesis) for the boundary/latch-heavy
         predicates in path_follower.py -- backlog item 24, scoped via
         grill-thesis 2026-07-20. test_path_follower.py already covers
         example cases for all of path_follower.py; this file sweeps the
         numeric/boolean input space specifically for the three predicates
         picked as highest property-testing value (pure functions, boundary-
         numeric or one-way-latch logic, not already exhaustively swept by
         example tests): should_use_hybrid_rotation, should_abort_to_home,
         should_transition_to_return_home.
Inputs:  None.
Outputs: pytest results.
How to run:
    pip install hypothesis
    cd ros2_ws && colcon build --packages-select fm_perception
    source /opt/ros/humble/setup.bash && source install/setup.bash
    python3 -m pytest src/fm_perception/test/test_path_follower_properties.py -v
Project: Onboard Visual Foundation Models for Mars Terrain Perception and Rover Navigation
"""
import math

from hypothesis import given, strategies as st

from fm_perception.path_follower import (
    pure_pursuit_command, should_use_hybrid_rotation, should_abort_to_home,
    should_transition_to_return_home,
)

finite_floats = st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6)
speeds = st.floats(allow_nan=False, allow_infinity=False, min_value=0.0, max_value=10.0)


# -- should_use_hybrid_rotation() --------------------------------------------
# Regression target: the 2026-07-18 dead-band bug (a stale SLAM pose gave
# linear_x=0.004 against the default creep_speed, which the original
# `linear_x == 0.0` trigger missed entirely -- see path_follower.py's own
# docstring and Ch4 H15 for the incident). These sweep the input space the
# original example test could only sample a few points of.

@given(linear_x=finite_floats, creep_speed=speeds)
def test_hybrid_rotation_is_exactly_the_dead_band(linear_x, creep_speed):
    assert should_use_hybrid_rotation(linear_x, creep_speed) == (linear_x < creep_speed)


@given(robot_x=finite_floats, robot_y=finite_floats,
       robot_yaw=st.floats(min_value=-math.pi, max_value=math.pi),
       target_x=finite_floats, target_y=finite_floats,
       linear_speed=st.floats(min_value=0.01, max_value=1.0), creep_speed=speeds)
def test_hybrid_rotation_engages_whenever_pure_pursuit_is_slower_than_creep(
        robot_x, robot_y, robot_yaw, target_x, target_y, linear_speed, creep_speed):
    """Integration property: for ANY pose/target pair -- not just the one
    heading error (~88 degrees) logged during the incident -- the pairing of
    pure_pursuit_command's own linear_x output with should_use_hybrid_rotation
    must never leave a dead band. This is the actual invariant the fix
    establishes; the unit test above only checks the predicate in isolation."""
    if robot_x == target_x and robot_y == target_y:
        return  # heading to target is undefined at zero separation; not a real planner state
    linear_x, _ = pure_pursuit_command(robot_x, robot_y, robot_yaw, target_x, target_y, linear_speed)
    assert should_use_hybrid_rotation(linear_x, creep_speed) == (linear_x < creep_speed)


# -- should_abort_to_home() latch and gating invariants ----------------------

@given(reactive_failsafe=st.booleans(), abort_to_home_enabled=st.booleans(),
       frontier_mode_enabled=st.booleans())
def test_abort_never_refires_once_already_aborting(reactive_failsafe, abort_to_home_enabled,
                                                     frontier_mode_enabled):
    assert should_abort_to_home(reactive_failsafe, abort_to_home_enabled,
                                 already_aborting=True,
                                 frontier_mode_enabled=frontier_mode_enabled) is False


@given(reactive_failsafe=st.booleans(), abort_to_home_enabled=st.booleans(),
       already_aborting=st.booleans())
def test_abort_never_fires_outside_frontier_mode(reactive_failsafe, abort_to_home_enabled, already_aborting):
    assert should_abort_to_home(reactive_failsafe, abort_to_home_enabled,
                                 already_aborting, frontier_mode_enabled=False) is False


@given(reactive_failsafe=st.booleans(), abort_to_home_enabled=st.booleans(),
       already_aborting=st.booleans(), frontier_mode_enabled=st.booleans())
def test_abort_matches_conjunction_of_its_four_conditions(reactive_failsafe, abort_to_home_enabled,
                                                            already_aborting, frontier_mode_enabled):
    expected = (reactive_failsafe and abort_to_home_enabled
                and not already_aborting and frontier_mode_enabled)
    assert should_abort_to_home(reactive_failsafe, abort_to_home_enabled,
                                 already_aborting, frontier_mode_enabled) == expected


# -- should_transition_to_return_home() latch invariant ----------------------

@given(frontier_selection_is_none=st.booleans(), return_home_enabled=st.booleans())
def test_return_home_never_refires_once_already_returning(frontier_selection_is_none, return_home_enabled):
    assert should_transition_to_return_home(
        frontier_selection_is_none, return_home_enabled, already_returning_home=True) is False


@given(return_home_enabled=st.booleans(), already_returning_home=st.booleans())
def test_return_home_never_fires_while_still_exploring(return_home_enabled, already_returning_home):
    assert should_transition_to_return_home(
        frontier_selection_is_none=False, return_home_enabled=return_home_enabled,
        already_returning_home=already_returning_home) is False
