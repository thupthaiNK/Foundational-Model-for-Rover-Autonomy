"""
Purpose: Unit tests for the pure-Python continuous-speed mapping and score
         smoothing functions used by terrain_controller_node.py's opt-in
         continuous-traversability-score mode (thesis Ch5 §5.6.2: v = v_max *
         (1 - T_score), with a 3-5 frame smoothing window). No rclpy
         dependency, no hardware required.
Inputs:  None.
Outputs: pytest results.
How to run:
    cd ros2_ws && colcon build --packages-select fm_perception
    source /opt/ros/humble/setup.bash && source install/setup.bash
    python3 -m pytest src/fm_perception/test/test_terrain_controller.py -v
Project: Onboard Visual Foundation Models for Mars Terrain Perception and Rover Navigation
"""
from fm_perception.terrain_controller_node import (
    continuous_speed, ScoreSmoother, score_topic_for, two_stage_vote_verdict,
    should_creep_to_verify, creep_verify_speed,
)


# ── should_creep_to_verify() / creep_verify_speed() (item 23, grill-scoped
#    2026-07-20: approach-to-verify creep near uncertain objects) ──────────
# Turns X3's finding (§4.8.10: rocks only reliably detected within ~2-3m of
# the rover) into a live behaviour: on an uncertain label, creep forward at
# reduced speed -- instead of A4's stop-in-place vote -- up to a hard
# distance cap, so the closer viewpoint has a chance to resolve the
# classification. Opt-in, and capped: an unresolved uncertain classification
# still ends in STOP once the cap is reached, exactly like every prior
# uncertain-handling result in this thesis -- this only delays, and can only
# ever delay, the STOP decision, never remove it.

def test_creep_to_verify_true_while_uncertain_and_under_cap():
    assert should_creep_to_verify("uncertain", creep_distance_m=0.5, max_creep_distance_m=1.5) is True


def test_creep_to_verify_false_once_cap_reached():
    assert should_creep_to_verify("uncertain", creep_distance_m=1.5, max_creep_distance_m=1.5) is False


def test_creep_to_verify_false_past_cap():
    assert should_creep_to_verify("uncertain", creep_distance_m=2.0, max_creep_distance_m=1.5) is False


def test_creep_to_verify_false_for_a_resolved_label():
    assert should_creep_to_verify("soil", creep_distance_m=0.0, max_creep_distance_m=1.5) is False
    assert should_creep_to_verify("big_rock", creep_distance_m=0.0, max_creep_distance_m=1.5) is False


def test_creep_verify_speed_is_slower_than_every_policy_speed():
    from fm_perception.terrain_controller_node import POLICY
    for label, speed in POLICY.items():
        assert creep_verify_speed() <= speed or speed == 0.0


def test_creep_verify_speed_is_positive():
    assert creep_verify_speed() > 0.0


# ── continuous_speed() ──────────────────────────────────────────────────

def test_confident_soil_score_gives_v_max():
    # T_score=0.0 (confident soil, risk=0.0) -> full speed, matching the
    # discrete policy's soil speed exactly.
    assert continuous_speed(t_score=0.0, v_max=0.10) == 0.10


def test_confident_big_rock_score_gives_zero():
    # T_score=1.0 (confident big_rock, risk=1.0) -> full stop.
    assert continuous_speed(t_score=1.0, v_max=0.10) == 0.0


def test_score_matches_discrete_sand_speed_at_sand_risk_weight():
    # sand risk weight = 0.5 (dinov2_terrain_node.py CLASS_RISK) -> confident
    # sand should reproduce the discrete policy's sand speed exactly.
    assert abs(continuous_speed(t_score=0.5, v_max=0.10) - 0.05) < 1e-9


def test_score_matches_discrete_bedrock_speed_at_bedrock_risk_weight():
    # bedrock risk weight = 0.7 -> should reproduce the discrete bedrock speed.
    assert abs(continuous_speed(t_score=0.7, v_max=0.10) - 0.03) < 1e-9


def test_mixed_score_gives_intermediate_speed():
    # A near-even soil/big_rock split (T_score~0.5) should NOT snap to a
    # discrete class -- it gets a genuinely intermediate speed, unlike the
    # discrete policy which would pick one label and one fixed speed.
    v = continuous_speed(t_score=0.5, v_max=0.10)
    assert 0.0 < v < 0.10


def test_speed_clamped_for_out_of_range_score_below_zero():
    assert continuous_speed(t_score=-0.2, v_max=0.10) == 0.10


def test_speed_clamped_for_out_of_range_score_above_one():
    assert continuous_speed(t_score=1.5, v_max=0.10) == 0.0


def test_speed_scales_with_v_max():
    assert continuous_speed(t_score=0.0, v_max=0.20) == 0.20
    assert continuous_speed(t_score=1.0, v_max=0.20) == 0.0


# ── ScoreSmoother ────────────────────────────────────────────────────────

def test_smoother_returns_the_single_value_on_first_update():
    s = ScoreSmoother(window=4)
    assert s.update(0.8) == 0.8


def test_smoother_averages_within_window():
    s = ScoreSmoother(window=4)
    s.update(0.0)
    s.update(1.0)
    avg = s.update(1.0)
    assert abs(avg - (0.0 + 1.0 + 1.0) / 3) < 1e-9


def test_smoother_drops_oldest_value_past_window():
    s = ScoreSmoother(window=2)
    s.update(0.0)   # dropped once the 3rd value arrives
    s.update(1.0)
    avg = s.update(1.0)
    # window=2 -> only the last two values (1.0, 1.0) are averaged.
    assert abs(avg - 1.0) < 1e-9


def test_smoother_smooths_a_single_noisy_spike():
    s = ScoreSmoother(window=4)
    for _ in range(4):
        s.update(0.0)
    spiked = s.update(1.0)
    # One noisy frame shouldn't swing the smoothed value all the way to 1.0.
    assert spiked < 0.5


# ── score_topic_for() ────────────────────────────────────────────────────

def test_score_topic_defaults_to_dinov2_only_score():
    # Backward-compatible default -- matches every previously reported result.
    assert score_topic_for("dinov2") == "/traversability_score"


def test_score_topic_switches_to_fused_when_requested():
    assert score_topic_for("fused") == "/traversability_score_fused"


def test_score_topic_falls_back_to_dinov2_for_unknown_value():
    assert score_topic_for("not_a_real_option") == "/traversability_score"


# ── two_stage_vote_verdict() ─────────────────────────────────────────────
# A4 two-stage uncertain policy (grill-scoped 2026-07-19): the rover STOPs
# immediately on an uncertain classification (safety unchanged), then votes
# over the next `window` classifications *while stopped*. If a strict
# majority are confident-traversable, the initial STOP was a single-frame
# false alarm and can be released early; otherwise the STOP is confirmed.
# "traversable" = the label carried a non-zero policy speed (soil/sand/
# bedrock), i.e. not uncertain/big_rock/unknown.

def test_verdict_gathering_until_window_filled():
    # Fewer than `window` votes: no decision yet.
    assert two_stage_vote_verdict(["soil"], window=3) == "gathering"
    assert two_stage_vote_verdict(["soil", "sand"], window=3) == "gathering"


def test_verdict_false_stop_when_majority_traversable():
    # First uncertain frame was noise: 2 of 3 votes are traversable.
    assert two_stage_vote_verdict(["soil", "uncertain", "sand"], window=3) == "false_stop"


def test_verdict_confirmed_when_majority_stop_labels():
    assert two_stage_vote_verdict(
        ["uncertain", "uncertain", "soil"], window=3) == "confirmed_stop"


def test_verdict_confirmed_on_genuine_obstacle_votes():
    # big_rock is a stop-label too, not traversable.
    assert two_stage_vote_verdict(
        ["big_rock", "big_rock", "soil"], window=3) == "confirmed_stop"


def test_verdict_requires_strict_majority_ties_stay_stopped():
    # Even split -> safety-first: no strict traversable majority -> confirmed.
    assert two_stage_vote_verdict(
        ["soil", "sand", "uncertain", "big_rock"], window=4) == "confirmed_stop"


def test_verdict_uses_only_the_most_recent_window_votes():
    # Older votes beyond the window must not count.
    votes = ["uncertain", "uncertain", "soil", "soil", "sand"]
    assert two_stage_vote_verdict(votes, window=3) == "false_stop"


def test_verdict_window_one_resumes_on_single_traversable():
    # Degenerate window=1: the very next frame decides.
    assert two_stage_vote_verdict(["soil"], window=1) == "false_stop"
    assert two_stage_vote_verdict(["uncertain"], window=1) == "confirmed_stop"
