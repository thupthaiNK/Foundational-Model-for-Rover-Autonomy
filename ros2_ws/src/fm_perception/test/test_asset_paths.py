"""
Purpose: Tests for fm_perception.asset_paths.find_repo_asset(), which resolves
         the on-disk model assets dinov2_terrain_node needs (the INT8 ONNX
         encoder and the 1000-shot LogReg feature cache).
         Why it exists: those two defaults were built by walking a fixed number
         of ".." levels up from the module file. That count was wrong, and wrong
         by different amounts in source space and install space, so
         dinov2_terrain_node could not start with default parameters on any
         machine -- it raised FileNotFoundError from _train_logreg before it ever
         reached the camera. No launch file overrides either parameter, so the
         real-hardware launch inherited the broken defaults. These tests pin the
         resolver to real layouts instead of a hardcoded depth.
Inputs:  None (synthetic directory trees in tmp_path).
Outputs: pytest results.
How to run:
    cd ros2_ws && colcon build --packages-select fm_perception
    source /opt/ros/humble/setup.bash && source install/setup.bash
    python3 -m pytest src/fm_perception/test/test_asset_paths.py -v
Project: Onboard Visual Foundation Models for Mars Terrain Perception and Rover Navigation
"""
import os

import pytest

from fm_perception.asset_paths import find_repo_asset

ASSET = os.path.join("experiments", "results", "encoder.onnx")


def _make_asset(root):
    path = root / "experiments" / "results" / "encoder.onnx"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("x")
    return path


def test_finds_asset_from_source_space_layout(tmp_path):
    """repo/ros2_ws/src/fm_perception/fm_perception/ -> repo/experiments/results."""
    asset = _make_asset(tmp_path)
    start = tmp_path / "ros2_ws" / "src" / "fm_perception" / "fm_perception"
    start.mkdir(parents=True)
    assert find_repo_asset(ASSET, start_dir=str(start)) == str(asset)


def test_finds_asset_from_install_space_layout(tmp_path):
    """The installed copy sits far deeper, at a different depth from the repo root."""
    asset = _make_asset(tmp_path)
    start = (tmp_path / "ros2_ws" / "install" / "fm_perception" / "local" / "lib"
             / "python3.10" / "dist-packages" / "fm_perception")
    start.mkdir(parents=True)
    assert find_repo_asset(ASSET, start_dir=str(start)) == str(asset)


def test_returns_none_when_asset_is_absent(tmp_path):
    """No assets anywhere: report absence rather than a path that does not exist,
    so the caller can raise an error naming the parameter to set."""
    start = tmp_path / "ros2_ws" / "src" / "fm_perception" / "fm_perception"
    start.mkdir(parents=True)
    assert find_repo_asset(ASSET, start_dir=str(start)) is None


def test_nearest_ancestor_wins(tmp_path):
    """With assets at two depths, the closest one to the module is chosen."""
    _make_asset(tmp_path)
    near_root = tmp_path / "ros2_ws"
    near = _make_asset(near_root)
    start = near_root / "src" / "fm_perception" / "fm_perception"
    start.mkdir(parents=True)
    assert find_repo_asset(ASSET, start_dir=str(start)) == str(near)


def test_environment_variable_overrides_the_search(tmp_path, monkeypatch):
    """On the Pi the workspace is bind mounted at /ws and experiments/ is not
    under it at all, so no amount of walking up finds the assets. An explicit
    root must win over the search."""
    other = tmp_path / "elsewhere"
    asset = _make_asset(other)
    start = tmp_path / "ros2_ws" / "src" / "fm_perception" / "fm_perception"
    start.mkdir(parents=True)
    monkeypatch.setenv("FM_PERCEPTION_ASSET_ROOT", str(other))
    assert find_repo_asset(ASSET, start_dir=str(start)) == str(asset)


def test_environment_variable_pointing_nowhere_falls_back_to_search(tmp_path, monkeypatch):
    """A stale env var must not shadow assets that are actually present."""
    asset = _make_asset(tmp_path)
    start = tmp_path / "ros2_ws" / "src" / "fm_perception" / "fm_perception"
    start.mkdir(parents=True)
    monkeypatch.setenv("FM_PERCEPTION_ASSET_ROOT", str(tmp_path / "does_not_exist"))
    assert find_repo_asset(ASSET, start_dir=str(start)) == str(asset)


def test_real_repo_assets_resolve():
    """The two assets dinov2_terrain_node actually loads must resolve in this
    checkout. This is the regression the fixed defaults exist for."""
    for rel in (
        os.path.join("experiments", "results", "dinov2_reg_small_encoder_int8.onnx"),
        os.path.join("experiments", "results", "feature_cache",
                     "dinov2_reg_small_train_1000shot.npz"),
    ):
        found = find_repo_asset(rel)
        assert found is not None, f"{rel} did not resolve from the fm_perception module"
        assert os.path.exists(found)
