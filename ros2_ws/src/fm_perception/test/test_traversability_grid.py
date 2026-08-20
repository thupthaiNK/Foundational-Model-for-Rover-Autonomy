"""
Purpose: Unit tests for traversability_grid.py — pure-Python grid geometry
         and cost logic shared by the static and live costmap ROS2 nodes.
Inputs:  None.
Outputs: pytest results.
How to run:
    cd ros2_ws && colcon build --packages-select fm_perception
    source /opt/ros/humble/setup.bash && source install/setup.bash
    python3 -m pytest src/fm_perception/test/test_traversability_grid.py -v
Project: Onboard Visual Foundation Models for Mars Terrain Perception and Rover Navigation
"""
from fm_perception.traversability_grid import (
    cost_for_label, zone_for_point, world_to_cell, build_static_grid,
    OccupancyGridBuilder, WIDTH_CELLS, HEIGHT_CELLS,
    COST_SOIL, COST_SAND, COST_BEDROCK, COST_HAZARD,
)


def test_cost_for_label_known_labels():
    assert cost_for_label("soil") == COST_SOIL
    assert cost_for_label("sand") == COST_SAND
    assert cost_for_label("bedrock") == COST_BEDROCK
    assert cost_for_label("big_rock") == COST_HAZARD
    assert cost_for_label("uncertain") == COST_HAZARD


def test_cost_for_label_unknown_defaults_to_hazard():
    assert cost_for_label("not_a_real_label") == COST_HAZARD


def test_zone_for_point_soil_zone_center():
    assert zone_for_point(-7.5, 6.0) == "soil_zone"


def test_zone_for_point_boulder_zone_center():
    assert zone_for_point(2.5, -9.0) == "boulder_zone"


def test_zone_for_point_rock_cluster_center():
    assert zone_for_point(2.5, -3.5) == "rock_cluster"


def test_zone_for_point_outside_arena():
    assert zone_for_point(100.0, 100.0) == "outside_arena"


def test_world_to_cell_origin():
    assert world_to_cell(-15.0, -12.0) == (0, 0)


def test_world_to_cell_arbitrary_point():
    # x=-7.5 -> col=(-7.5-(-15.0))/0.1=75 ; y=6.0 -> row=(6.0-(-12.0))/0.1=180
    assert world_to_cell(-7.5, 6.0) == (75, 180)


def test_build_static_grid_has_correct_length():
    grid = build_static_grid()
    assert len(grid) == WIDTH_CELLS * HEIGHT_CELLS


def test_build_static_grid_soil_zone_is_free():
    grid = build_static_grid()
    col, row = world_to_cell(-7.5, 6.0)
    assert grid[row * WIDTH_CELLS + col] == COST_SOIL


def test_build_static_grid_boulder_zone_is_hazard():
    grid = build_static_grid()
    col, row = world_to_cell(2.5, -9.0)
    assert grid[row * WIDTH_CELLS + col] == COST_HAZARD


def test_occupancy_grid_builder_starts_all_free():
    builder = OccupancyGridBuilder()
    assert all(c == COST_SOIL for c in builder.grid)


def test_occupancy_grid_builder_paints_lookahead_cell():
    builder = OccupancyGridBuilder(lookahead_m=0.6, patch_radius_m=0.3)
    # robot at origin facing +x (heading=0) perceives "big_rock" ahead
    builder.paint_lookahead(robot_x=0.0, robot_y=0.0, heading_rad=0.0, label="big_rock")
    col, row = world_to_cell(0.6, 0.0)
    assert builder.grid[row * WIDTH_CELLS + col] == COST_HAZARD


def test_occupancy_grid_builder_does_not_paint_outside_patch_radius():
    builder = OccupancyGridBuilder(lookahead_m=0.6, patch_radius_m=0.3)
    builder.paint_lookahead(robot_x=0.0, robot_y=0.0, heading_rad=0.0, label="big_rock")
    col, row = world_to_cell(-10.0, -10.0)
    assert builder.grid[row * WIDTH_CELLS + col] == COST_SOIL


# -- init_cost (frontier exploration-lite, 2026-07-18) ------------------------
# The frontier explorer needs "never assessed by perception" to be
# distinguishable from "assessed as soil", but COST_SOIL == 0 is also the
# grid's default fill. Opt-in init_cost=-1 (OccupancyGrid's standard
# "unknown") fixes that; the default stays COST_SOIL so every pre-existing
# costmap result is untouched.

def test_occupancy_grid_builder_default_init_cost_is_soil():
    builder = OccupancyGridBuilder()
    assert all(c == COST_SOIL for c in builder.grid)


def test_occupancy_grid_builder_init_cost_unknown_fills_grid_with_unknown():
    builder = OccupancyGridBuilder(init_cost=-1)
    assert all(c == -1 for c in builder.grid)


def test_painting_on_unknown_grid_paints_costs_and_leaves_rest_unknown():
    builder = OccupancyGridBuilder(lookahead_m=0.6, patch_radius_m=0.3, init_cost=-1)
    builder.paint_lookahead(robot_x=0.0, robot_y=0.0, heading_rad=0.0, label="sand")
    col, row = world_to_cell(0.6, 0.0)
    assert builder.grid[row * WIDTH_CELLS + col] == COST_SAND
    far_col, far_row = world_to_cell(-10.0, -10.0)
    assert builder.grid[far_row * WIDTH_CELLS + far_col] == -1


# -- confidence_aware_painting (root-cause fix for the 2026-07-19 start-cell-
# hazard deadlock, item 6 of the L1-L6 further-work plan) -------------------
# Root cause: live_traversability_costmap_node.py discards the confidence
# value dinov2_terrain_node.py attaches to every classification ("label:
# confidence") and paint_lookahead() unconditionally overwrites every cell in
# its disc with the new label's cost. "uncertain" (confidence below the
# deployed 0.40 threshold, by construction from the classifier -- checking
# the label string is equivalent to checking the confidence directly, no new
# data needs to flow through the message pipeline) maps to the same
# COST_HAZARD as a confidently-detected big_rock, so a single borderline
# frame can instantly overwrite a cell many prior confident observations had
# painted safe -- the exact mechanism behind grid_with_start_freed()'s
# symptom (frontier_explorer.py). Opt-in (default False): every existing
# official result used the unconditional-overwrite behaviour and must remain
# byte-for-byte reproducible.
#
# The rule only protects a cell whose current cost is a genuine prior
# OBSERVATION, not an ambiguous default -- COST_SOIL=0 is indistinguishable
# from "never painted" unless init_cost=-1 (frontier exploration-lite's own
# convention), so the protection is scoped to init_cost=-1 grids only;
# elsewhere "uncertain" continues to overwrite unconditionally, identical to
# confidence_aware_painting=False.

def test_confidence_aware_painting_disabled_by_default_uncertain_overwrites():
    builder = OccupancyGridBuilder(lookahead_m=0.6, patch_radius_m=0.3, init_cost=-1)
    builder.paint_lookahead(robot_x=0.0, robot_y=0.0, heading_rad=0.0, label="sand")
    builder.paint_lookahead(robot_x=0.0, robot_y=0.0, heading_rad=0.0, label="uncertain")
    col, row = world_to_cell(0.6, 0.0)
    assert builder.grid[row * WIDTH_CELLS + col] == COST_HAZARD


def test_confidence_aware_painting_uncertain_does_not_downgrade_known_safe_cell():
    builder = OccupancyGridBuilder(lookahead_m=0.6, patch_radius_m=0.3, init_cost=-1,
                                    confidence_aware_painting=True)
    builder.paint_lookahead(robot_x=0.0, robot_y=0.0, heading_rad=0.0, label="sand")
    builder.paint_lookahead(robot_x=0.0, robot_y=0.0, heading_rad=0.0, label="uncertain")
    col, row = world_to_cell(0.6, 0.0)
    assert builder.grid[row * WIDTH_CELLS + col] == COST_SAND


def test_confidence_aware_painting_protects_bedrock_too_not_just_soil():
    builder = OccupancyGridBuilder(lookahead_m=0.6, patch_radius_m=0.3, init_cost=-1,
                                    confidence_aware_painting=True)
    builder.paint_lookahead(robot_x=0.0, robot_y=0.0, heading_rad=0.0, label="bedrock")
    builder.paint_lookahead(robot_x=0.0, robot_y=0.0, heading_rad=0.0, label="uncertain")
    col, row = world_to_cell(0.6, 0.0)
    assert builder.grid[row * WIDTH_CELLS + col] == COST_BEDROCK


def test_confidence_aware_painting_uncertain_still_paints_over_never_assessed_cell():
    # Genuinely fresh terrain (still -1, never painted) has no prior
    # observation to protect -- uncertain must still default to cautious.
    builder = OccupancyGridBuilder(lookahead_m=0.6, patch_radius_m=0.3, init_cost=-1,
                                    confidence_aware_painting=True)
    builder.paint_lookahead(robot_x=0.0, robot_y=0.0, heading_rad=0.0, label="uncertain")
    col, row = world_to_cell(0.6, 0.0)
    assert builder.grid[row * WIDTH_CELLS + col] == COST_HAZARD


def test_confidence_aware_painting_real_label_still_overwrites_freely():
    # A genuine confident detection (big_rock, confidence >= threshold) is
    # real evidence, not an admission of ignorance -- it must still be able
    # to overwrite a previously-painted safe cell, unlike "uncertain".
    builder = OccupancyGridBuilder(lookahead_m=0.6, patch_radius_m=0.3, init_cost=-1,
                                    confidence_aware_painting=True)
    builder.paint_lookahead(robot_x=0.0, robot_y=0.0, heading_rad=0.0, label="soil")
    builder.paint_lookahead(robot_x=0.0, robot_y=0.0, heading_rad=0.0, label="big_rock")
    col, row = world_to_cell(0.6, 0.0)
    assert builder.grid[row * WIDTH_CELLS + col] == COST_HAZARD


def test_confidence_aware_painting_uncertain_overwriting_hazard_is_a_harmless_noop():
    builder = OccupancyGridBuilder(lookahead_m=0.6, patch_radius_m=0.3, init_cost=-1,
                                    confidence_aware_painting=True)
    builder.paint_lookahead(robot_x=0.0, robot_y=0.0, heading_rad=0.0, label="big_rock")
    builder.paint_lookahead(robot_x=0.0, robot_y=0.0, heading_rad=0.0, label="uncertain")
    col, row = world_to_cell(0.6, 0.0)
    assert builder.grid[row * WIDTH_CELLS + col] == COST_HAZARD


def test_confidence_aware_painting_has_no_effect_without_init_unknown():
    # COST_SOIL=0 is ambiguous between "observed as soil" and "never
    # painted" outside init_cost=-1 mode, so the protection cannot be
    # meaningfully applied there -- falls back to unconditional overwrite,
    # identical to confidence_aware_painting=False.
    builder = OccupancyGridBuilder(lookahead_m=0.6, patch_radius_m=0.3,
                                    confidence_aware_painting=True)  # init_cost default (COST_SOIL)
    builder.paint_lookahead(robot_x=0.0, robot_y=0.0, heading_rad=0.0, label="sand")
    builder.paint_lookahead(robot_x=0.0, robot_y=0.0, heading_rad=0.0, label="uncertain")
    col, row = world_to_cell(0.6, 0.0)
    assert builder.grid[row * WIDTH_CELLS + col] == COST_HAZARD


# -- track_confidence (A2 epistemic uncertainty map, item 7, 2026-07-19) ------
# Per-cell latest-write confidence store, parallel to the cost grid: the
# classifier already attaches a confidence to every classification (the
# "label:confidence" message), but the costmap discarded it, so nothing
# downstream could distinguish a barely-above-threshold observation (0.41)
# from a highly confident one (0.95). Opt-in (default False), same
# convention as init_cost/confidence_aware_painting: zero behaviour change
# and zero extra memory unless enabled. -1.0 = never observed, mirroring
# the cost grid's own -1 "unknown". Latest-write, matching the cost grid's
# own last-write semantics (a deliberate grill decision: no EMA smoothing
# constant to justify, and re-observation must be able to move the value).

def test_track_confidence_disabled_by_default_no_confidence_store():
    builder = OccupancyGridBuilder()
    assert builder.confidence == []


def test_track_confidence_initialises_all_cells_to_never_observed():
    builder = OccupancyGridBuilder(track_confidence=True)
    assert len(builder.confidence) == WIDTH_CELLS * HEIGHT_CELLS
    assert all(c == -1.0 for c in builder.confidence)


def test_painting_with_confidence_records_it_for_painted_cells_only():
    builder = OccupancyGridBuilder(lookahead_m=0.6, patch_radius_m=0.3,
                                    track_confidence=True)
    builder.paint_lookahead(robot_x=0.0, robot_y=0.0, heading_rad=0.0,
                             label="sand", confidence=0.72)
    col, row = world_to_cell(0.6, 0.0)
    assert builder.confidence[row * WIDTH_CELLS + col] == 0.72
    far_col, far_row = world_to_cell(-10.0, -10.0)
    assert builder.confidence[far_row * WIDTH_CELLS + far_col] == -1.0


def test_confidence_is_latest_write_second_observation_overwrites():
    builder = OccupancyGridBuilder(lookahead_m=0.6, patch_radius_m=0.3,
                                    track_confidence=True)
    builder.paint_lookahead(robot_x=0.0, robot_y=0.0, heading_rad=0.0,
                             label="sand", confidence=0.45)
    builder.paint_lookahead(robot_x=0.0, robot_y=0.0, heading_rad=0.0,
                             label="sand", confidence=0.88)
    col, row = world_to_cell(0.6, 0.0)
    assert builder.confidence[row * WIDTH_CELLS + col] == 0.88


def test_latest_write_can_also_lower_a_previously_high_confidence():
    # Deliberately NOT monotone -- latest-write records what the newest
    # observation actually said, honest even when it is worse.
    builder = OccupancyGridBuilder(lookahead_m=0.6, patch_radius_m=0.3,
                                    track_confidence=True)
    builder.paint_lookahead(robot_x=0.0, robot_y=0.0, heading_rad=0.0,
                             label="sand", confidence=0.88)
    builder.paint_lookahead(robot_x=0.0, robot_y=0.0, heading_rad=0.0,
                             label="sand", confidence=0.45)
    col, row = world_to_cell(0.6, 0.0)
    assert builder.confidence[row * WIDTH_CELLS + col] == 0.45


def test_painting_without_confidence_leaves_confidence_store_untouched():
    # Callers that never learned about confidence (or a message that failed
    # to parse) keep working: cost is painted, confidence stays as it was.
    builder = OccupancyGridBuilder(lookahead_m=0.6, patch_radius_m=0.3,
                                    track_confidence=True)
    builder.paint_lookahead(robot_x=0.0, robot_y=0.0, heading_rad=0.0,
                             label="sand", confidence=0.72)
    builder.paint_lookahead(robot_x=0.0, robot_y=0.0, heading_rad=0.0,
                             label="sand")
    col, row = world_to_cell(0.6, 0.0)
    assert builder.grid[row * WIDTH_CELLS + col] == COST_SAND
    assert builder.confidence[row * WIDTH_CELLS + col] == 0.72


def test_confidence_write_respects_confidence_aware_painting_protection():
    # When confidence_aware_painting skips a cell (uncertain must not
    # downgrade a known-safe cell), the confidence store must skip it too --
    # otherwise the cost would say "sand from the good observation" while
    # the confidence says "the bad frame's value", a mismatched pair.
    builder = OccupancyGridBuilder(lookahead_m=0.6, patch_radius_m=0.3,
                                    init_cost=-1, confidence_aware_painting=True,
                                    track_confidence=True)
    builder.paint_lookahead(robot_x=0.0, robot_y=0.0, heading_rad=0.0,
                             label="sand", confidence=0.72)
    builder.paint_lookahead(robot_x=0.0, robot_y=0.0, heading_rad=0.0,
                             label="uncertain", confidence=0.31)
    col, row = world_to_cell(0.6, 0.0)
    assert builder.grid[row * WIDTH_CELLS + col] == COST_SAND
    assert builder.confidence[row * WIDTH_CELLS + col] == 0.72


# -- bayesian_fusion (SuperMap-inspired log-odds label fusion, 2026-07-19) ----
# track_confidence's latest-write semantics (above) discard evidence: a cell
# revisited many times only ever remembers its single newest observation.
# SuperMap (Zhao et al., 2026) fuses per-object label belief across repeated
# observations via a log-odds update (their eq. 10) instead of overwriting.
# bayesian_fusion applies the same idea per grid cell: accumulate log-odds
# across observations that agree on the SAME label (reinforcement); reset to
# the new observation alone when the label disagrees (a different hypothesis
# should not inherit the old one's accumulated evidence). Opt-in (default
# False, requires track_confidence=True to have any effect) -- every existing
# official result (A2, §4.8.30) used latest-write and must remain
# byte-for-byte reproducible. Confidence is clamped to [0.01, 0.99] both on
# read and on write so accumulation can never saturate to a value that no
# future disagreeing observation could move away from.

def test_bayesian_fusion_off_by_default_is_latest_write():
    builder = OccupancyGridBuilder(lookahead_m=0.6, patch_radius_m=0.3,
                                    track_confidence=True)
    builder.paint_lookahead(robot_x=0.0, robot_y=0.0, heading_rad=0.0,
                             label="sand", confidence=0.6)
    builder.paint_lookahead(robot_x=0.0, robot_y=0.0, heading_rad=0.0,
                             label="sand", confidence=0.6)
    col, row = world_to_cell(0.6, 0.0)
    assert builder.confidence[row * WIDTH_CELLS + col] == 0.6


def test_bayesian_fusion_first_observation_sets_confidence_directly():
    builder = OccupancyGridBuilder(lookahead_m=0.6, patch_radius_m=0.3,
                                    track_confidence=True, bayesian_fusion=True)
    builder.paint_lookahead(robot_x=0.0, robot_y=0.0, heading_rad=0.0,
                             label="sand", confidence=0.6)
    col, row = world_to_cell(0.6, 0.0)
    assert builder.confidence[row * WIDTH_CELLS + col] == 0.6


def test_bayesian_fusion_reinforces_matching_label_above_either_single_observation():
    builder = OccupancyGridBuilder(lookahead_m=0.6, patch_radius_m=0.3,
                                    track_confidence=True, bayesian_fusion=True)
    builder.paint_lookahead(robot_x=0.0, robot_y=0.0, heading_rad=0.0,
                             label="sand", confidence=0.7)
    builder.paint_lookahead(robot_x=0.0, robot_y=0.0, heading_rad=0.0,
                             label="sand", confidence=0.7)
    col, row = world_to_cell(0.6, 0.0)
    assert builder.confidence[row * WIDTH_CELLS + col] > 0.7


def test_bayesian_fusion_resets_on_label_change_instead_of_blending():
    builder = OccupancyGridBuilder(lookahead_m=0.6, patch_radius_m=0.3,
                                    track_confidence=True, bayesian_fusion=True)
    builder.paint_lookahead(robot_x=0.0, robot_y=0.0, heading_rad=0.0,
                             label="sand", confidence=0.95)
    builder.paint_lookahead(robot_x=0.0, robot_y=0.0, heading_rad=0.0,
                             label="bedrock", confidence=0.6)
    col, row = world_to_cell(0.6, 0.0)
    # Must equal the new observation alone (clamped), not a fusion with the
    # old "sand" evidence -- a label change starts a fresh belief.
    assert builder.confidence[row * WIDTH_CELLS + col] == 0.6
    assert builder.last_label[row * WIDTH_CELLS + col] == "bedrock"


def test_bayesian_fusion_stays_bounded_after_many_reinforcing_observations():
    builder = OccupancyGridBuilder(lookahead_m=0.6, patch_radius_m=0.3,
                                    track_confidence=True, bayesian_fusion=True)
    for _ in range(20):
        builder.paint_lookahead(robot_x=0.0, robot_y=0.0, heading_rad=0.0,
                                 label="sand", confidence=0.99)
    col, row = world_to_cell(0.6, 0.0)
    fused = builder.confidence[row * WIDTH_CELLS + col]
    assert 0.01 <= fused <= 0.99


def test_bayesian_fusion_clamps_extreme_confidence_inputs():
    builder = OccupancyGridBuilder(lookahead_m=0.6, patch_radius_m=0.3,
                                    track_confidence=True, bayesian_fusion=True)
    builder.paint_lookahead(robot_x=0.0, robot_y=0.0, heading_rad=0.0,
                             label="sand", confidence=1.0)
    col, row = world_to_cell(0.6, 0.0)
    assert builder.confidence[row * WIDTH_CELLS + col] == 0.99


def test_bayesian_fusion_disagreeing_evidence_after_reset_can_still_reduce_confidence():
    # After a label change resets the belief, a further weak observation of
    # the SAME (new) label still fuses normally (reinforcement is symmetric),
    # confirming the reset is one-shot, not a permanent lock to last-write.
    builder = OccupancyGridBuilder(lookahead_m=0.6, patch_radius_m=0.3,
                                    track_confidence=True, bayesian_fusion=True)
    builder.paint_lookahead(robot_x=0.0, robot_y=0.0, heading_rad=0.0,
                             label="sand", confidence=0.9)
    builder.paint_lookahead(robot_x=0.0, robot_y=0.0, heading_rad=0.0,
                             label="bedrock", confidence=0.55)
    builder.paint_lookahead(robot_x=0.0, robot_y=0.0, heading_rad=0.0,
                             label="bedrock", confidence=0.55)
    col, row = world_to_cell(0.6, 0.0)
    assert builder.confidence[row * WIDTH_CELLS + col] > 0.55


def test_bayesian_fusion_requires_track_confidence_to_have_any_effect():
    # bayesian_fusion=True with track_confidence=False must be a no-op
    # (mirrors the existing confidence_aware_painting/init_cost pattern):
    # the cost grid still paints normally, no confidence store exists at all.
    builder = OccupancyGridBuilder(lookahead_m=0.6, patch_radius_m=0.3,
                                    bayesian_fusion=True)
    builder.paint_lookahead(robot_x=0.0, robot_y=0.0, heading_rad=0.0,
                             label="sand", confidence=0.7)
    col, row = world_to_cell(0.6, 0.0)
    assert builder.grid[row * WIDTH_CELLS + col] == COST_SAND
    assert builder.confidence == []
