"""
Purpose: Unit tests for astar_planner.py -- pure-Python A* path planning
         over the existing traversability occupancy grid (backlog "L5-lite",
         scoped via grill-thesis 2026-07-17). No ROS2/Gazebo dependency.
Inputs:  None.
Outputs: pytest results.
How to run:
    cd ros2_ws && colcon build --packages-select fm_perception
    source /opt/ros/humble/setup.bash && source install/setup.bash
    python3 -m pytest src/fm_perception/test/test_astar_planner.py -v
Project: Onboard Visual Foundation Models for Mars Terrain Perception and Rover Navigation
"""
from fm_perception.astar_planner import (
    cell_blocked, step_cost, astar_path, path_to_world, inflate_hazards,
)
from fm_perception.traversability_grid import (
    COST_SOIL, COST_SAND, COST_BEDROCK, COST_HAZARD,
    ORIGIN_X_M, ORIGIN_Y_M, RESOLUTION_M,
)


def _grid(width, height, cost=COST_SOIL):
    return [cost] * (width * height)


# -- cell_blocked() / step_cost() ------------------------------------------

def test_cell_blocked_true_for_hazard():
    assert cell_blocked(COST_HAZARD) is True


def test_cell_blocked_false_for_non_hazard():
    assert cell_blocked(COST_SOIL) is False
    assert cell_blocked(COST_SAND) is False
    assert cell_blocked(COST_BEDROCK) is False


def test_step_cost_increases_with_terrain_cost():
    assert step_cost(COST_SOIL) < step_cost(COST_SAND) < step_cost(COST_BEDROCK)


def test_step_cost_minimum_is_one_for_free_terrain():
    assert step_cost(COST_SOIL) == 1.0


# -- step_cost() confidence channel (item 21, grill-scoped 2026-07-20) ------
# Extends step_cost() with an optional confidence penalty so A* can prefer
# well-observed cells over low-confidence ones of the same terrain type,
# using A2's per-cell confidence grid (§4.8.30 H18) as the source. Opt-in via
# an explicit confidence argument; every pre-existing call site that omits it
# must be bit-for-bit unaffected.

def test_step_cost_unaffected_when_confidence_omitted():
    assert step_cost(COST_SAND) == 1.0 + 1.0 * (COST_SAND / 100.0)


def test_step_cost_unaffected_at_full_confidence():
    assert step_cost(COST_SAND, confidence=1.0) == step_cost(COST_SAND)


def test_step_cost_increases_as_confidence_decreases():
    high = step_cost(COST_SAND, confidence=0.9)
    low = step_cost(COST_SAND, confidence=0.3)
    assert high < low


def test_step_cost_confidence_penalty_is_zero_at_full_confidence_nonzero_below():
    assert step_cost(COST_SOIL, confidence=1.0) == step_cost(COST_SOIL)
    assert step_cost(COST_SOIL, confidence=0.0) > step_cost(COST_SOIL)


# -- inflate_hazards() (item 13, grill-scoped 2026-07-20) -------------------
# Opt-in 1-2 cell safety margin around hazard cells: non-hazard neighbours of
# a hazard cell get a raised cost (so A* prefers to keep distance from a
# hazard, not just avoid entering it), but never as high as COST_HAZARD
# itself -- inflated cells stay traversable, matching this codebase's
# existing "penalise, don't block, for non-hazard cells" philosophy. Off by
# default (radius_cells=0 is a no-op): every existing astar_path() call site
# that doesn't opt in sees an identical grid.

def test_inflate_hazards_radius_zero_is_a_no_op():
    grid = _grid(5, 5)
    grid[2 * 5 + 2] = COST_HAZARD
    inflated = inflate_hazards(list(grid), 5, 5, radius_cells=0, inflated_cost=COST_BEDROCK)
    assert inflated == grid


def test_inflate_hazards_raises_cost_of_neighbours_only():
    width, height = 5, 5
    grid = _grid(width, height)
    grid[2 * width + 2] = COST_HAZARD  # hazard at (col=2, row=2)
    inflated = inflate_hazards(grid, width, height, radius_cells=1, inflated_cost=COST_BEDROCK)
    # 8-connected neighbours of (2,2) get raised
    assert inflated[1 * width + 1] == COST_BEDROCK
    assert inflated[1 * width + 2] == COST_BEDROCK
    assert inflated[2 * width + 1] == COST_BEDROCK
    # a cell two steps away is untouched
    assert inflated[0 * width + 0] == COST_SOIL


def test_inflate_hazards_never_overwrites_the_hazard_cell_itself():
    width, height = 5, 5
    grid = _grid(width, height)
    grid[2 * width + 2] = COST_HAZARD
    inflated = inflate_hazards(grid, width, height, radius_cells=1, inflated_cost=COST_BEDROCK)
    assert inflated[2 * width + 2] == COST_HAZARD


def test_inflate_hazards_never_exceeds_hazard_cost():
    width, height = 5, 5
    grid = _grid(width, height)
    grid[2 * width + 2] = COST_HAZARD
    inflated = inflate_hazards(grid, width, height, radius_cells=1, inflated_cost=COST_HAZARD + 50)
    for c in inflated:
        assert c <= COST_HAZARD


def test_inflate_hazards_does_not_lower_an_already_higher_cost_cell():
    width, height = 5, 5
    grid = _grid(width, height)
    grid[2 * width + 2] = COST_HAZARD
    grid[1 * width + 1] = COST_HAZARD  # a second, adjacent hazard cell
    inflated = inflate_hazards(grid, width, height, radius_cells=1, inflated_cost=COST_BEDROCK)
    assert inflated[1 * width + 1] == COST_HAZARD  # stays a hazard, not downgraded to inflated_cost


def test_astar_prefers_a_wider_detour_from_hazard_when_inflated():
    # Tall enough grid that a wide detour (avoiding the whole inflated
    # neighbourhood) is a real alternative to a tight detour (hugging the
    # hazard's immediate neighbour cells).
    width, height = 7, 5
    grid = _grid(width, height)
    grid[2 * width + 3] = COST_HAZARD  # hazard in the middle row
    baseline_path = astar_path(grid, width, height, (0, 2), (6, 2))
    inflated = inflate_hazards(list(grid), width, height, radius_cells=1, inflated_cost=COST_BEDROCK)
    inflated_path = astar_path(inflated, width, height, (0, 2), (6, 2))
    assert baseline_path is not None and inflated_path is not None
    # the inflated-grid path must visit fewer of the hazard's immediate
    # 8-neighbours than the uninflated baseline -- the margin measurably
    # pushes the route further away, not just around the single hazard cell.
    def n_neighbour_visits(path):
        return sum(1 for (col, row) in path if abs(col - 3) <= 1 and abs(row - 2) <= 1
                   and (col, row) != (3, 2))
    assert n_neighbour_visits(inflated_path) < n_neighbour_visits(baseline_path)


def test_step_cost_confidence_weight_scales_the_penalty():
    default_penalty = step_cost(COST_SOIL, confidence=0.0) - step_cost(COST_SOIL)
    doubled_penalty = step_cost(COST_SOIL, confidence=0.0, confidence_weight=1.0) - step_cost(COST_SOIL)
    assert doubled_penalty == default_penalty * 2


# -- astar_path() ------------------------------------------------------------

def test_astar_start_equals_goal():
    grid = _grid(5, 5)
    assert astar_path(grid, 5, 5, (2, 2), (2, 2)) == [(2, 2)]


def test_astar_straight_line_on_empty_grid():
    grid = _grid(10, 10)
    path = astar_path(grid, 10, 10, (0, 0), (5, 0))
    assert path is not None
    assert path[0] == (0, 0)
    assert path[-1] == (5, 0)
    assert len(path) == 6  # 8-connected, empty grid -> straight line is optimal


def test_astar_returns_none_when_goal_cell_is_blocked():
    grid = _grid(5, 5)
    grid[2 * 5 + 4] = COST_HAZARD
    assert astar_path(grid, 5, 5, (0, 0), (4, 2)) is None


def test_astar_returns_none_when_goal_walled_off():
    width, height = 5, 5
    grid = _grid(width, height)
    goal = (4, 4)
    for dc in (-1, 0, 1):
        for dr in (-1, 0, 1):
            if dc == 0 and dr == 0:
                continue
            c, r = goal[0] + dc, goal[1] + dr
            if 0 <= c < width and 0 <= r < height:
                grid[r * width + c] = COST_HAZARD
    assert astar_path(grid, width, height, (0, 0), goal) is None


def test_astar_detours_around_a_hazard_wall():
    width, height = 5, 5
    grid = _grid(width, height)
    for row in range(height - 1):
        grid[row * width + 2] = COST_HAZARD  # wall across column 2, gap at row 4
    path = astar_path(grid, width, height, (0, 0), (4, 0))
    assert path is not None
    assert (2, 4) in path       # must route through the only gap
    assert len(path) > 5        # longer than the blocked straight line


def test_astar_prefers_lower_total_cost_over_a_shorter_high_cost_route():
    # Middle row (row=1) is expensive (bedrock); rows 0 and 2 are free (soil).
    # The direct straight path along row 1 has fewer steps but higher total
    # cost than detouring up into row 0 -- A* must pick the detour.
    width, height = 5, 3
    grid = _grid(width, height, cost=COST_SOIL)
    for col in range(width):
        grid[1 * width + col] = COST_BEDROCK
    path = astar_path(grid, width, height, (0, 1), (4, 1))
    assert path is not None
    assert any(row != 1 for _col, row in path), (
        "expected the detour into a free row, not the direct expensive row"
    )


# -- path_to_world() ---------------------------------------------------------

def test_path_to_world_converts_cell_centres():
    world = path_to_world([(0, 0), (1, 0)])
    assert world[0] == (ORIGIN_X_M + 0.5 * RESOLUTION_M, ORIGIN_Y_M + 0.5 * RESOLUTION_M)
    assert world[1] == (ORIGIN_X_M + 1.5 * RESOLUTION_M, ORIGIN_Y_M + 0.5 * RESOLUTION_M)
