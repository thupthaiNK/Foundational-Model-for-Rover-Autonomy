"""
Purpose: Load a mission's waypoint list from a YAML spec file instead of
         hardcoded Python constants -- backlog item 27, scoped 2026-07-20.
         Replaces the START_POSE/GOAL_POSE tuple constants previously
         hardcoded in experiments/l6_lite_roundtrip_test.py (and equivalent
         in other mission-driving scripts) with a per-mission YAML file
         under experiments/missions/, so a new mission (different
         waypoints) is a config edit, not a code change. l5_lite_planner_node
         itself already takes waypoints as ROS2 launch parameters
         (waypoint_xs/waypoint_ys); this only replaces where those values
         originate for the offline driving/recorder scripts and for
         building the matching `ros2 launch` command line.
Inputs:  A YAML file matching missions/recon_and_return.yaml's shape:
         start_pose: {x, y}, goal_tolerance_m, waypoints: [{x, y}, ...]
Outputs: MissionSpec namedtuple; launch_args() string for `ros2 launch`.
How to run: Imported by mission-driving scripts, e.g.
    from mission_spec import load_mission
    mission = load_mission("experiments/missions/recon_and_return.yaml")
Project: Onboard Visual Foundation Models for Mars Terrain Perception and Rover Navigation
"""
from typing import List, NamedTuple, Tuple

import yaml


class MissionSpec(NamedTuple):
    name: str
    start_pose: Tuple[float, float]
    goal_tolerance_m: float
    waypoints: List[Tuple[float, float]]

    def launch_args(self) -> str:
        """Builds the waypoint_xs/waypoint_ys ROS2 launch argument string for
        every waypoint after the first (matching l5_lite_planner_node.py's
        own convention: goal_x/goal_y is waypoint 0, waypoint_xs/ys are the
        rest)."""
        rest = self.waypoints[1:]
        xs = [x for x, _ in rest]
        ys = [y for _, y in rest]
        return f'waypoint_xs:="{xs}" waypoint_ys:="{ys}"'


def load_mission(yaml_path: str) -> MissionSpec:
    with open(yaml_path) as f:
        raw = yaml.safe_load(f)
    start = raw["start_pose"]
    waypoints = [(wp["x"], wp["y"]) for wp in raw["waypoints"]]
    return MissionSpec(
        name=raw["name"],
        start_pose=(start["x"], start["y"]),
        goal_tolerance_m=raw["goal_tolerance_m"],
        waypoints=waypoints,
    )
