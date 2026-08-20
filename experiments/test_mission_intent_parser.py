"""
Purpose: Unit tests for mission_intent_parser.py (backlog item 17,
         "ROSA-lite", scoped down 2026-07-20). No ROS2/Gazebo dependency.
Inputs:  None.
Outputs: pytest results.
How to run:
    python3 -m pytest experiments/test_mission_intent_parser.py -v
Project: Onboard Visual Foundation Models for Mars Terrain Perception and Rover Navigation
"""
from mission_intent_parser import parse_mission_command, mission_script


def test_recon_and_return_recognised():
    assert parse_mission_command("do a recon and return") == "recon_and_return"


def test_explore_return_home_recognised():
    assert parse_mission_command("Explore the area then return home.") == "explore_return_home"


def test_frontier_exploration_recognised():
    assert parse_mission_command("map the area") == "frontier_exploration"


def test_reobservation_recognised():
    assert parse_mission_command("go check the uncertain areas") == "reobservation"


def test_unresolved_command_returns_none_not_a_default_mission():
    assert parse_mission_command("bake me a cake") is None


def test_case_insensitive_matching():
    assert parse_mission_command("EXPLORE THE AREA") == "frontier_exploration"


def test_longer_more_specific_phrase_wins_over_shorter_contained_one():
    # "explore the area" is a substring of "explore the area then return home";
    # the more specific explore_return_home match must win, not frontier_exploration.
    assert parse_mission_command("explore the area then return home") == "explore_return_home"


def test_mission_script_returns_the_backing_script_path():
    assert mission_script("recon_and_return") == "experiments/l6_lite_roundtrip_test.py"


def test_mission_script_returns_none_for_unknown_mission():
    assert mission_script("not_a_real_mission") is None
