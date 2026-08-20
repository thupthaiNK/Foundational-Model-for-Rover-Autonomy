#!/usr/bin/env python3
"""
Purpose: Locate the on-disk model assets dinov2_terrain_node loads at startup:
         the INT8 ONNX encoder and the 1000-shot LogReg feature cache.
         Why this exists: both defaults used to be built by walking a fixed
         count of ".." levels up from the module file. The count was wrong, and
         wrong by different amounts depending on whether the module was imported
         from source space or from the colcon install space, so the node raised
         FileNotFoundError out of _train_logreg with default parameters on every
         machine. Nothing caught it because no launch file overrides either
         parameter, and the only launch file that starts this node is the
         real-hardware one -- so the failure surfaced as "DINOv2 does not work on
         the rover" rather than as a path bug.
         Walking up until the asset is actually found removes the dependency on
         layout depth entirely. FM_PERCEPTION_ASSET_ROOT covers the case the
         search cannot: on the Pi the workspace is bind mounted at /ws and the
         experiments/ directory is not inside it at any depth.
Inputs:  A repo-relative asset path, e.g. "experiments/results/foo.onnx".
Outputs: The absolute path if it exists, otherwise None.
How to run:
    cd ros2_ws && colcon build --packages-select fm_perception
    source /opt/ros/humble/setup.bash && source install/setup.bash
    python3 -m pytest src/fm_perception/test/test_asset_paths.py -v
Project: Onboard Visual Foundation Models for Mars Terrain Perception and Rover Navigation
"""
import os

ASSET_ROOT_ENV = "FM_PERCEPTION_ASSET_ROOT"


def find_repo_asset(relative_path: str, start_dir: str = None):
    """Absolute path to `relative_path`, or None if it is nowhere to be found.

    Resolution order:
      1. $FM_PERCEPTION_ASSET_ROOT/<relative_path>, if that file exists. A stale
         env var deliberately does not shadow assets that are really present, so
         a leftover export cannot silently disable the node.
      2. <ancestor>/<relative_path> for each ancestor of `start_dir`, nearest
         first. `start_dir` defaults to this module's directory, which works
         from source space and install space alike because the search stops at
         whatever depth the asset actually sits, rather than assuming one.
    """
    root_env = os.environ.get(ASSET_ROOT_ENV)
    if root_env:
        candidate = os.path.join(root_env, relative_path)
        if os.path.exists(candidate):
            return os.path.abspath(candidate)

    current = os.path.abspath(start_dir or os.path.dirname(os.path.abspath(__file__)))
    while True:
        candidate = os.path.join(current, relative_path)
        if os.path.exists(candidate):
            return candidate
        parent = os.path.dirname(current)
        if parent == current:            # reached the filesystem root
            return None
        current = parent


def describe_missing_asset(relative_path: str, parameter_name: str) -> str:
    """Error text for an asset that did not resolve, naming the way out."""
    return (
        f"Could not find {relative_path} by searching upward from "
        f"{os.path.dirname(os.path.abspath(__file__))}.\n"
        f"Either set the ROS2 parameter {parameter_name} to its absolute path, "
        f"or export {ASSET_ROOT_ENV} to the directory that contains "
        f"experiments/results (needed on the Pi, where the workspace is bind "
        f"mounted at /ws and experiments/ lives outside it)."
    )
