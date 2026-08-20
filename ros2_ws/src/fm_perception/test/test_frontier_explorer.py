"""
Purpose: Unit tests for frontier_explorer.py -- pure frontier-detection and
         selection functions for "frontier exploration-lite" (scoped via
         grill-thesis 2026-07-18): the rover autonomously picks the nearest
         unexplored-boundary cell inside a bounded exploration box as its
         next goal, using the live traversability costmap (init_unknown
         mode, cells = -1 until DINOv2 paints them) as the map of what has
         been explored. No ROS2/Gazebo dependency.
Inputs:  None.
Outputs: pytest results.
How to run:
    cd ros2_ws && colcon build --packages-select fm_perception
    source /opt/ros/humble/setup.bash && source install/setup.bash
    python3 -m pytest src/fm_perception/test/test_frontier_explorer.py -v -p no:anyio
Project: Onboard Visual Foundation Models for Mars Terrain Perception and Rover Navigation
"""
from fm_perception.frontier_explorer import (
    UNKNOWN_COST, find_frontier_cells, is_bedrock_adjacent,
    select_nearest_frontier, grid_for_planning, count_known_cells_in_box,
    grid_with_start_freed, select_lowest_confidence_cell,
)


# Small synthetic grids: width=5, height=4, row-major (index = row*width+col),
# same layout convention as OccupancyGridBuilder/astar_planner. -1 = unknown,
# 0..99 = known traversable cost, 100 = known hazard.

W, H = 5, 4
U = UNKNOWN_COST  # -1


def make_grid(rows):
    """rows = list of H lists of W ints, row 0 first."""
    assert len(rows) == H and all(len(r) == W for r in rows)
    return [c for row in rows for c in row]


FULL_BOX = (0, 0, W - 1, H - 1)  # (col_min, row_min, col_max, row_max), inclusive


# -- find_frontier_cells() ----------------------------------------------------

def test_unknown_cell_adjacent_to_known_free_cell_is_a_frontier():
    grid = make_grid([
        [0, U, U, U, U],
        [U, U, U, U, U],
        [U, U, U, U, U],
        [U, U, U, U, U],
    ])
    frontiers = find_frontier_cells(grid, W, H, FULL_BOX)
    # The three unknown 8-neighbours of the single known-free cell (0,0).
    assert set(frontiers) == {(1, 0), (0, 1), (1, 1)}


def test_unknown_cell_adjacent_only_to_hazard_is_not_a_frontier():
    grid = make_grid([
        [100, U, U, U, U],
        [U,   U, U, U, U],
        [U,   U, U, U, U],
        [U,   U, U, U, U],
    ])
    assert find_frontier_cells(grid, W, H, FULL_BOX) == []


def test_known_cells_are_never_frontiers():
    grid = make_grid([
        [0, 35, U, U, U],
        [0, 65, U, U, U],
        [U, U,  U, U, U],
        [U, U,  U, U, U],
    ])
    frontiers = find_frontier_cells(grid, W, H, FULL_BOX)
    for cell in [(0, 0), (1, 0), (0, 1), (1, 1)]:
        assert cell not in frontiers


def test_unknown_cell_with_no_known_neighbour_is_not_a_frontier():
    grid = make_grid([
        [0, U, U, U, U],
        [U, U, U, U, U],
        [U, U, U, U, U],
        [U, U, U, U, U],
    ])
    frontiers = find_frontier_cells(grid, W, H, FULL_BOX)
    assert (4, 3) not in frontiers  # far corner, all neighbours unknown


def test_cells_outside_the_box_are_excluded():
    grid = make_grid([
        [0, U, U, U, U],
        [U, U, U, U, U],
        [U, U, U, U, U],
        [U, U, U, U, U],
    ])
    # Box covering only the rightmost two columns -- none of (0,0)'s
    # unknown neighbours are inside it.
    box = (3, 0, 4, 3)
    assert find_frontier_cells(grid, W, H, box) == []


# -- select_nearest_frontier() ------------------------------------------------

def test_selects_the_euclidean_nearest_frontier():
    frontiers = [(4, 3), (1, 0), (2, 2)]
    assert select_nearest_frontier(frontiers, robot_cell=(0, 0)) == (1, 0)


def test_blacklisted_frontiers_are_skipped():
    frontiers = [(1, 0), (2, 2)]
    assert select_nearest_frontier(
        frontiers, robot_cell=(0, 0), blacklist={(1, 0)}) == (2, 2)


def test_returns_none_when_no_frontiers_remain():
    assert select_nearest_frontier([], robot_cell=(0, 0)) is None
    assert select_nearest_frontier(
        [(1, 0)], robot_cell=(0, 0), blacklist={(1, 0)}) is None


def test_frontiers_closer_than_min_distance_are_skipped():
    # Found live in the first smoke test (2026-07-18): the very first
    # painted patch's boundary includes cells right next to the rover, so
    # without a minimum selection distance the nearest frontier can already
    # be inside goal_tolerance -- goal_reached() fires on the next control
    # tick and the node re-selects the same cell at 10 Hz forever. The
    # minimum must sit above goal_tolerance (0.2 m = 2 cells at 0.1 m
    # resolution) but below the first patch's outer frontier ring (~0.9 m =
    # 9 cells), or no first frontier could ever be selected.
    frontiers = [(2, 0), (12, 0)]
    assert select_nearest_frontier(
        frontiers, robot_cell=(0, 0), min_distance_cells=10.0) == (12, 0)


def test_returns_none_when_all_frontiers_are_too_close():
    assert select_nearest_frontier(
        [(1, 0), (0, 2)], robot_cell=(0, 0), min_distance_cells=10.0) is None


# -- is_bedrock_adjacent() ----------------------------------------------------
# Semantic-biased frontier selection (scoped via grill-thesis 2026-07-18,
# night): a frontier counts as "science-interesting" when at least one of its
# 8-neighbours is known bedrock (COST_BEDROCK=65, exposed outcrop -- the
# science-target framing of Candela & Wettergreen 2022). Same 8-neighbour
# convention as find_frontier_cells().

def test_frontier_with_bedrock_8_neighbour_is_bedrock_adjacent():
    grid = make_grid([
        [65, U, U, U, U],
        [U,  U, U, U, U],
        [U,  U, U, U, U],
        [U,  U, U, U, U],
    ])
    assert is_bedrock_adjacent(grid, W, (1, 0))       # orthogonal neighbour
    assert is_bedrock_adjacent(grid, W, (1, 1))       # diagonal neighbour


def test_frontier_with_only_non_bedrock_neighbours_is_not_bedrock_adjacent():
    grid = make_grid([
        [0, 35, U, U, U],
        [U, U,  U, U, U],
        [U, U,  U, U, U],
        [U, U,  U, U, U],
    ])
    assert not is_bedrock_adjacent(grid, W, (2, 0))   # soil+sand neighbours only


def test_hazard_neighbour_does_not_count_as_bedrock():
    grid = make_grid([
        [100, U, U, U, U],
        [U,   U, U, U, U],
        [U,   U, U, U, U],
        [U,   U, U, U, U],
    ])
    assert not is_bedrock_adjacent(grid, W, (1, 0))


def test_bedrock_adjacency_at_grid_edge_ignores_out_of_bounds():
    grid = make_grid([
        [U, U, U, U, U],
        [U, U, U, U, U],
        [U, U, U, U, U],
        [U, U, U, U, 65],
    ])
    # Corner cell: only in-bounds neighbours are checked, no IndexError.
    assert is_bedrock_adjacent(grid, W, (3, 3))
    assert not is_bedrock_adjacent(grid, W, (0, 0))


# -- select_nearest_frontier() semantic bias ----------------------------------
# Lexicographic priority (grill decision, no tuned weight): when grid+width
# are supplied and at least one candidate (after blacklist/min-distance
# filtering) is bedrock-adjacent, the selection pool is restricted to the
# bedrock-adjacent candidates; nearest-by-distance decides within the pool.
# With no bedrock-adjacent candidate -- or no grid supplied -- behaviour is
# exactly the pre-existing nearest-frontier rule.

BEDROCK_EAST_GRID = make_grid([
    [0, U, U, U, 65],
    [U, U, U, U, U],
    [U, U, U, U, U],
    [U, U, U, U, U],
])  # (1,0) is a plain frontier near the robot; (3,0) touches bedrock at (4,0)


def test_bedrock_adjacent_frontier_beats_nearer_plain_frontier():
    frontiers = [(1, 0), (3, 0)]
    assert select_nearest_frontier(
        frontiers, robot_cell=(0, 0), grid=BEDROCK_EAST_GRID, width=W) == (3, 0)


def test_nearest_within_bedrock_group_wins():
    grid = make_grid([
        [U, U, U, U, 65],
        [U, U, U, U, U],
        [U, U, U, U, 65],
        [U, U, U, U, U],
    ])
    frontiers = [(3, 0), (3, 2)]  # both bedrock-adjacent
    assert select_nearest_frontier(
        frontiers, robot_cell=(0, 3), grid=grid, width=W) == (3, 2)


def test_falls_back_to_plain_nearest_when_no_bedrock_adjacent_candidate():
    grid = make_grid([
        [0, U, U, U, U],
        [U, U, U, U, U],
        [U, U, U, U, U],
        [U, U, U, U, U],
    ])
    frontiers = [(1, 0), (3, 0)]
    assert select_nearest_frontier(
        frontiers, robot_cell=(0, 0), grid=grid, width=W) == (1, 0)


def test_blacklisted_bedrock_frontier_is_not_selected():
    frontiers = [(1, 0), (3, 0)]
    # The only bedrock-adjacent candidate is blacklisted -> pool is empty
    # after filtering -> plain nearest among the rest.
    assert select_nearest_frontier(
        frontiers, robot_cell=(0, 0), blacklist={(3, 0)},
        grid=BEDROCK_EAST_GRID, width=W) == (1, 0)


def test_min_distance_still_applies_to_bedrock_frontiers():
    grid = make_grid([
        [65, U, U, U, U],
        [U,  U, U, U, U],
        [U,  U, U, U, U],
        [U,  U, U, U, U],
    ])
    frontiers = [(1, 0), (4, 3)]  # (1,0) bedrock-adjacent but too close
    assert select_nearest_frontier(
        frontiers, robot_cell=(0, 0), min_distance_cells=3.0,
        grid=grid, width=W) == (4, 3)


def test_without_grid_behaviour_is_unchanged_nearest_frontier():
    # Backward compatibility: no grid supplied -> plain nearest, even though
    # (3,0) would be bedrock-adjacent on BEDROCK_EAST_GRID.
    frontiers = [(1, 0), (3, 0)]
    assert select_nearest_frontier(frontiers, robot_cell=(0, 0)) == (1, 0)


# -- grid_with_start_freed() --------------------------------------------------
# Fix for the 2026-07-19 start-cell-hazard deadlock (root-caused via smoke
# test, see project_semantic_frontier_inprogress_20260719): pure-pursuit
# deviation can carry the rover onto a cell its own paint disc mis-labelled
# hazard, and astar_path() then refuses every goal because the start cell
# itself is blocked. Proprioception beats a mispainted disc: free hazard
# cells within the escape radius (3 cells -- OccupancyGridBuilder's default
# patch_radius_m=0.3 / RESOLUTION_M=0.1, so a single self-paint disc can
# always be escaped) around the start cell, same circular-disc shape as
# paint_lookahead() (dc**2 + dr**2 <= radius**2), leaving distant real
# hazard walls untouched.

SQ = 9  # 9x9 square grid, room to test cells at several distances from centre


def make_square_grid(fill=0):
    return [fill] * (SQ * SQ)


def test_start_on_hazard_frees_hazard_cells_within_escape_radius():
    grid = make_square_grid(0)
    start = (4, 4)
    grid[4 * SQ + 4] = 100   # start cell itself: hazard
    grid[6 * SQ + 6] = 100   # dc=2,dr=2 -> dist^2=8  <= 9: inside radius
    grid[4 * SQ + 7] = 100   # dc=3,dr=0 -> dist^2=9  <= 9: exactly on boundary
    grid[7 * SQ + 6] = 100   # dc=2,dr=3 -> dist^2=13 > 9: outside radius
    grid[7 * SQ + 7] = 100   # dc=3,dr=3 -> dist^2=18 > 9: outside radius

    result = grid_with_start_freed(grid, SQ, start)

    assert result[4 * SQ + 4] == 0
    assert result[6 * SQ + 6] == 0
    assert result[4 * SQ + 7] == 0
    assert result[7 * SQ + 6] == 100
    assert result[7 * SQ + 7] == 100


def test_start_not_on_hazard_returns_grid_unchanged():
    grid = make_square_grid(0)
    grid[4 * SQ + 4] = 35            # start cell: known, non-hazard
    grid[4 * SQ + 5] = 100           # a nearby hazard cell, well within r=3
    result = grid_with_start_freed(grid, SQ, (4, 4))
    assert result == grid


def test_grid_with_start_freed_does_not_mutate_the_input():
    grid = make_square_grid(0)
    grid[4 * SQ + 4] = 100
    original = list(grid)
    grid_with_start_freed(grid, SQ, (4, 4))
    assert grid == original


# -- select_lowest_confidence_cell() ------------------------------------------
# A2 re-observation mode (item 7, grill-scoped 2026-07-19): once frontier
# exploration is exhausted, the rover revisits the known non-hazard cell
# whose recorded confidence is LOWEST -- uncertainty-driven re-observation,
# no tuned threshold (lowest-first was a deliberate grill decision over a
# "revisit everything below C" pool, which would need a magic constant).
# Confidence values here are ints 0-100, matching what the planner actually
# receives from the /traversability_confidence OccupancyGrid topic; -1 =
# never observed. Nearest-by-distance breaks ties among equal minima.
# The P2 invariant the official run checks is defined on the CONFIDENCE
# VALUE (selected cell's confidence == pool minimum), not the cell identity,
# precisely because ties are legal and broken by distance.

CONF_UNSEEN = -1


def test_selects_the_lowest_confidence_known_cell():
    cost = make_grid([
        [0,  0,  0,  U, U],
        [0,  0,  0,  U, U],
        [U,  U,  U,  U, U],
        [U,  U,  U,  U, U],
    ])
    conf = make_grid([
        [80, 45, 90, CONF_UNSEEN, CONF_UNSEEN],
        [70, 85, 60, CONF_UNSEEN, CONF_UNSEEN],
        [CONF_UNSEEN] * 5,
        [CONF_UNSEEN] * 5,
    ])
    assert select_lowest_confidence_cell(
        cost, conf, W, FULL_BOX, robot_cell=(4, 3),
        visited=set(), min_distance_cells=0.0) == (1, 0)


def test_nearest_breaks_ties_among_equal_minimum_confidence():
    cost = make_grid([
        [0,  0,  0,  0, U],
        [U,  U,  U,  U, U],
        [U,  U,  U,  U, U],
        [U,  U,  U,  U, U],
    ])
    conf = make_grid([
        [45, 90, 90, 45, CONF_UNSEEN],
        [CONF_UNSEEN] * 5,
        [CONF_UNSEEN] * 5,
        [CONF_UNSEEN] * 5,
    ])
    # (0,0) and (3,0) share the minimum 45; robot sits nearer (3,0).
    assert select_lowest_confidence_cell(
        cost, conf, W, FULL_BOX, robot_cell=(4, 0),
        visited=set(), min_distance_cells=0.0) == (3, 0)


def test_visited_cells_are_never_reselected():
    cost = make_grid([
        [0,  0,  U, U, U],
        [U,  U,  U, U, U],
        [U,  U,  U, U, U],
        [U,  U,  U, U, U],
    ])
    conf = make_grid([
        [45, 60, CONF_UNSEEN, CONF_UNSEEN, CONF_UNSEEN],
        [CONF_UNSEEN] * 5,
        [CONF_UNSEEN] * 5,
        [CONF_UNSEEN] * 5,
    ])
    assert select_lowest_confidence_cell(
        cost, conf, W, FULL_BOX, robot_cell=(4, 3),
        visited={(0, 0)}, min_distance_cells=0.0) == (1, 0)


def test_cells_closer_than_min_distance_are_skipped():
    # Same guard as frontier selection, same reason: a target already
    # inside goal tolerance re-selects itself at the control rate.
    cost = make_grid([
        [0, U, U, U, 0],
        [U, U, U, U, U],
        [U, U, U, U, U],
        [U, U, U, U, U],
    ])
    conf = make_grid([
        [45, CONF_UNSEEN, CONF_UNSEEN, CONF_UNSEEN, 90],
        [CONF_UNSEEN] * 5,
        [CONF_UNSEEN] * 5,
        [CONF_UNSEEN] * 5,
    ])
    # (0,0) has the lower confidence but sits 1 cell from the robot.
    assert select_lowest_confidence_cell(
        cost, conf, W, FULL_BOX, robot_cell=(1, 0),
        visited=set(), min_distance_cells=2.0) == (4, 0)


def test_hazard_cells_are_never_reobservation_targets():
    # A hazard cell is not somewhere the rover should drive, no matter how
    # unconfident the observation that painted it was.
    cost = make_grid([
        [100, 0,  U, U, U],
        [U,   U,  U, U, U],
        [U,   U,  U, U, U],
        [U,   U,  U, U, U],
    ])
    conf = make_grid([
        [41, 85, CONF_UNSEEN, CONF_UNSEEN, CONF_UNSEEN],
        [CONF_UNSEEN] * 5,
        [CONF_UNSEEN] * 5,
        [CONF_UNSEEN] * 5,
    ])
    assert select_lowest_confidence_cell(
        cost, conf, W, FULL_BOX, robot_cell=(4, 3),
        visited=set(), min_distance_cells=0.0) == (1, 0)


def test_never_observed_cells_are_not_reobservation_targets():
    # confidence -1 means "no observation exists to be unconfident about"
    # -- those are frontier territory, not re-observation territory.
    cost = make_grid([
        [0, 0, U, U, U],
        [U, U, U, U, U],
        [U, U, U, U, U],
        [U, U, U, U, U],
    ])
    conf = make_grid([
        [CONF_UNSEEN, 85, CONF_UNSEEN, CONF_UNSEEN, CONF_UNSEEN],
        [CONF_UNSEEN] * 5,
        [CONF_UNSEEN] * 5,
        [CONF_UNSEEN] * 5,
    ])
    assert select_lowest_confidence_cell(
        cost, conf, W, FULL_BOX, robot_cell=(4, 3),
        visited=set(), min_distance_cells=0.0) == (1, 0)


def test_cells_outside_the_box_are_not_reobservation_targets():
    cost = make_grid([
        [0,  0, U, U, U],
        [U,  U, U, U, U],
        [U,  U, U, U, U],
        [U,  U, U, U, U],
    ])
    conf = make_grid([
        [40, 85, CONF_UNSEEN, CONF_UNSEEN, CONF_UNSEEN],
        [CONF_UNSEEN] * 5,
        [CONF_UNSEEN] * 5,
        [CONF_UNSEEN] * 5,
    ])
    box = (1, 0, 4, 3)  # excludes column 0, where the lowest-conf cell sits
    assert select_lowest_confidence_cell(
        cost, conf, W, box, robot_cell=(4, 3),
        visited=set(), min_distance_cells=0.0) == (1, 0)


def test_returns_none_when_nothing_is_selectable():
    cost = make_grid([
        [0, U, U, U, U],
        [U, U, U, U, U],
        [U, U, U, U, U],
        [U, U, U, U, U],
    ])
    conf = make_grid([
        [45, CONF_UNSEEN, CONF_UNSEEN, CONF_UNSEEN, CONF_UNSEEN],
        [CONF_UNSEEN] * 5,
        [CONF_UNSEEN] * 5,
        [CONF_UNSEEN] * 5,
    ])
    assert select_lowest_confidence_cell(
        cost, conf, W, FULL_BOX, robot_cell=(4, 3),
        visited={(0, 0)}, min_distance_cells=0.0) is None


# -- grid_for_planning() ------------------------------------------------------

def test_planning_grid_maps_unknown_to_free_and_keeps_known_costs():
    # A* treats unknown as traversable (classic frontier exploration --
    # otherwise no path into unexplored space could ever be planned), but
    # step_cost() would misbehave on a negative cost, so -1 must become 0
    # while every known cost (including hazard=100) is preserved.
    grid = [U, 0, 35, 65, 100, U]
    assert grid_for_planning(grid) == [0, 0, 35, 65, 100, 0]


def test_planning_grid_does_not_mutate_the_input():
    grid = [U, 0]
    grid_for_planning(grid)
    assert grid == [U, 0]


# -- count_known_cells_in_box() -----------------------------------------------

def test_counts_only_known_cells_inside_the_box():
    grid = make_grid([
        [0, 35, U, U, U],
        [U, 100, U, U, U],
        [U, U,  U, U, U],
        [U, U,  U, U, U],
    ])
    # Full box: 3 known cells (0 at (0,0), 35 at (1,0), 100 at (1,1)).
    assert count_known_cells_in_box(grid, W, FULL_BOX) == 3
    # Box excluding column 0: only (1,0) and (1,1) remain.
    assert count_known_cells_in_box(grid, W, (1, 0, 4, 3)) == 2
