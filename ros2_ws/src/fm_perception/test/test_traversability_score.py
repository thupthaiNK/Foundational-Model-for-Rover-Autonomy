"""
Purpose: Unit tests for traversability_score() — the pure-Python continuous
         risk-score function used by dinov2_terrain_node.py (H3f).
Inputs:  None.
Outputs: pytest results.
How to run:
    cd ros2_ws && colcon build --packages-select fm_perception
    source /opt/ros/humble/setup.bash && source install/setup.bash
    python3 -m pytest src/fm_perception/test/test_traversability_score.py -v
Project: Onboard Visual Foundation Models for Mars Terrain Perception and Rover Navigation
"""
from fm_perception.dinov2_terrain_node import (
    traversability_score, CLASS_NAMES, CLASS_RISK, CONFIDENCE_THRESHOLD,
)


def _probs_for(label: str, p: float) -> list:
    """Build a probs vector with `label` at prob `p`, remainder split evenly."""
    remainder = (1.0 - p) / (len(CLASS_NAMES) - 1)
    return [p if name == label else remainder for name in CLASS_NAMES]


def test_confident_soil_is_low_risk():
    probs = _probs_for("soil", 0.95)
    assert traversability_score(probs, confidence=0.95) < 0.1


def test_confident_big_rock_is_max_risk():
    probs = _probs_for("big_rock", 0.95)
    score = traversability_score(probs, confidence=0.95)
    assert score > 0.9


def test_below_threshold_forces_max_risk_regardless_of_label():
    probs = _probs_for("soil", 0.30)  # confident-looking probs, but...
    score = traversability_score(probs, confidence=0.30, threshold=CONFIDENCE_THRESHOLD)
    assert score == 1.0


def test_mixed_soil_big_rock_gives_mid_range_score():
    probs = [0.5, 0.0, 0.0, 0.5]  # soil/big_rock roughly 50/50
    score = traversability_score(probs, confidence=0.5)
    assert 0.4 < score < 0.6


def test_score_bounded_in_unit_interval():
    for label in CLASS_NAMES:
        probs = _probs_for(label, 1.0)
        score = traversability_score(probs, confidence=1.0)
        assert 0.0 <= score <= 1.0


def test_class_risk_matches_controller_speed_ordering():
    # soil < sand < bedrock < big_rock in risk, mirroring terrain_controller_node's
    # soil(0.10) > sand(0.05) > bedrock(0.03) > big_rock(0.00) m/s speed ordering.
    assert CLASS_RISK["soil"] < CLASS_RISK["sand"] < CLASS_RISK["bedrock"] < CLASS_RISK["big_rock"]
