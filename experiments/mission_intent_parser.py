"""
Purpose: Minimal natural-language-to-mission mapper -- backlog item 17
         ("ROSA-lite"), scoped down 2026-07-20 (grill-thesis + user
         decision) to a own-built keyword intent parser rather than a full
         ROSA (github nasa-jpl/rosa) + local-Ollama LangChain ReAct agent,
         for compute safety on this dev machine (a local LLM's latency
         would risk fighting Gazebo for CPU the same way SmolVLM's ~80s/img
         measured cost ruled out a VLM terrain-complexity advisor, see
         "Explicitly rejected" list in project_l1l6_further_work_plan
         memory). Deterministic, offline, no model inference: maps a
         command string to one of this thesis's existing, already-built
         missions (recon-and-return L6-lite, frontier exploration-lite,
         explore-then-return-home, A2 re-observation) by keyword match.
         Closes the thesis narrative loop (language -> mission -> the
         built L5/L6 stack) at the scope this thesis can actually support;
         does not attempt general open-ended command understanding.
Inputs:  A natural-language command string.
Outputs: parse_mission_command() -> mission name (str) or None if
         unresolved -- an unresolved command must NOT silently default to
         any mission, matching this codebase's fail-safe convention
         (should_transition_to_reobservation()'s "refuses loud-and-safe"
         precedent in path_follower.py).
How to run: Imported by any future mission-launch script, or:
    python3 experiments/mission_intent_parser.py "explore the area then return home"
Project: Onboard Visual Foundation Models for Mars Terrain Perception and Rover Navigation
"""
import sys
from typing import Optional

# Each mission maps to the existing script/launch file that actually runs
# it -- this parser only selects which one, it does not launch anything
# itself (keeping ROS2/Gazebo dependencies out of this module, matching
# the pure/ROS2-node split used throughout this codebase).
MISSION_KEYWORDS = {
    "recon_and_return": {
        "keywords": ["recon", "reconnaissance", "go to the goal and come back",
                     "go there and return", "round trip"],
        "script": "experiments/l6_lite_roundtrip_test.py",
    },
    "explore_return_home": {
        "keywords": ["explore the area then return home", "explore then return",
                     "explore and come back", "explore the box then go home"],
        "script": "experiments/explore_return_home_test.py",
    },
    "frontier_exploration": {
        "keywords": ["explore the area", "map the area", "cover the area",
                     "explore the box"],
        "script": "experiments/frontier_exploration_test.py",
    },
    "reobservation": {
        "keywords": ["check the uncertain areas", "re-check low confidence",
                     "revisit uncertain cells", "verify uncertain areas"],
        "script": "experiments/reobservation_test.py",
    },
}


def parse_mission_command(text: str) -> Optional[str]:
    """Returns the best-matching mission name, or None if no keyword set
    matches -- an unresolved command intentionally does not fall back to
    any default mission. Matching is substring-based over a lowercased
    command and the longest matching keyword wins ties, so a more specific
    phrase (e.g. "explore the area then return home") is preferred over a
    shorter one it contains ("explore the area")."""
    lowered = text.lower()
    best_mission, best_len = None, 0
    for mission, spec in MISSION_KEYWORDS.items():
        for kw in spec["keywords"]:
            if kw in lowered and len(kw) > best_len:
                best_mission, best_len = mission, len(kw)
    return best_mission


def mission_script(mission_name: str) -> Optional[str]:
    spec = MISSION_KEYWORDS.get(mission_name)
    return spec["script"] if spec else None


if __name__ == "__main__":
    command = " ".join(sys.argv[1:]) or "explore the area then return home"
    mission = parse_mission_command(command)
    if mission is None:
        print(f"Command not understood: {command!r} -- no mission selected (fail-safe).")
    else:
        print(f"Command: {command!r}")
        print(f"Mission: {mission}")
        print(f"Run:     python3 {mission_script(mission)}")
