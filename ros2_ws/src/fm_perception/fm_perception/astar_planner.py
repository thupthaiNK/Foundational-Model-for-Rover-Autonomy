"""
Purpose: Pure-Python A* path planning over the existing traversability
         occupancy grid (traversability_grid.py) -- "L5-lite", scoped via
         grill-thesis 2026-07-17 as a lightweight alternative to the full
         Nav2 stack, which D1 (§4.8.13) found alone drove load average to
         9.33 on this 4-core development machine (confirmed independent of
         DINOv2 via the Condition A re-run). This module has no ROS2/Gazebo
         dependency, fully unit-testable in isolation, matching the pure/
         ROS2-node split already used for traversability_grid.py and
         traversability_score_fusion_node.py.
         Hazard cells (cost=COST_HAZARD=100: big_rock/uncertain/unknown)
         are treated as fully impassable, matching this codebase's existing
         safety philosophy of never routing through a confirmed hazard
         (the uncertain/hazard->STOP rule used throughout). All other cells
         are weighted by step_cost() so A* prefers lower-cost terrain but
         can still cross it if that is the only or cheaper option.
Inputs:  None directly; consumed by l5_lite_planner_node.py.
Outputs: cell_blocked(), step_cost(), astar_path(), path_to_world().
How to run: Imported by l5_lite_planner_node.py. Tested via
         ros2_ws/src/fm_perception/test/test_astar_planner.py
Project: Onboard Visual Foundation Models for Mars Terrain Perception and Rover Navigation
"""
import heapq
import math
from typing import List, Optional, Tuple

from fm_perception.traversability_grid import (
    COST_HAZARD, ORIGIN_X_M, ORIGIN_Y_M, RESOLUTION_M,
)

COST_WEIGHT = 1.0  # scales how much a non-hazard cell's cost penalises step_cost
CONFIDENCE_WEIGHT = 0.5  # scales the extra penalty step_cost() adds for a
                          # low-confidence cell (item 21, grill-scoped
                          # 2026-07-20: perception-aware step_cost, using
                          # A2's per-cell confidence grid, §4.8.30 H18, as
                          # an optional second cost channel alongside
                          # terrain cost -- A* prefers well-observed cells
                          # over low-confidence ones of the same terrain
                          # type, without ever treating low confidence as
                          # blocking, matching this codebase's existing
                          # "uncertain is a cost, not necessarily a wall"
                          # philosophy for non-hazard cells)


def cell_blocked(cost: int) -> bool:
    """True if a cell's cost makes it fully impassable. Only hazard cells
    (cost>=COST_HAZARD) are blocked -- soil/sand/bedrock are always
    traversable, just at different weighted cost."""
    return cost >= COST_HAZARD


def step_cost(cost: int, confidence: float = None, confidence_weight: float = CONFIDENCE_WEIGHT) -> float:
    """Traversal cost for entering a non-blocked cell: a base cost of 1.0
    (one grid step) plus a weighted penalty for higher-cost terrain, so A*
    prefers lower-cost cells without ever treating them as impassable.
    confidence (optional, 0.0-1.0) adds a second, independent penalty for
    low-confidence cells: zero extra cost at confidence=1.0, up to
    confidence_weight extra cost at confidence=0.0. Omitting confidence
    (the default, used by every pre-existing call site) reproduces the
    original terrain-only cost exactly -- this is purely additive."""
    terrain_component = COST_WEIGHT * (cost / 100.0)
    if confidence is None:
        confidence_component = 0.0
    else:
        confidence_component = confidence_weight * (1.0 - confidence)
    return 1.0 + terrain_component + confidence_component


def inflate_hazards(grid: List[int], width: int, height: int,
                     radius_cells: int = 0, inflated_cost: int = 65) -> List[int]:
    """Opt-in 1-2 cell safety margin around hazard cells (item 13,
    grill-scoped 2026-07-20): every non-hazard cell within radius_cells
    (8-connected/Chebyshev distance) of a hazard cell gets its cost raised
    to at least inflated_cost, so A* prefers to keep distance from a hazard
    rather than only avoiding entering it -- while inflated cells remain
    traversable (inflated_cost must stay below COST_HAZARD), matching this
    codebase's existing "penalise, don't block, for non-hazard cells"
    philosophy (step_cost() docstring). radius_cells=0 is a no-op, so every
    pre-existing astar_path() call site that does not opt in sees an
    identical grid. Never lowers a cell's cost (a hazard cell, or a cell
    already costlier than inflated_cost, is left unchanged)."""
    if radius_cells <= 0:
        return grid
    out = list(grid)
    hazard_cells = [(c, r) for r in range(height) for c in range(width)
                     if grid[r * width + c] >= COST_HAZARD]
    for hc, hr in hazard_cells:
        for dc in range(-radius_cells, radius_cells + 1):
            for dr in range(-radius_cells, radius_cells + 1):
                if dc == 0 and dr == 0:
                    continue
                c, r = hc + dc, hr + dr
                if not (0 <= c < width and 0 <= r < height):
                    continue
                idx = r * width + c
                if grid[idx] >= COST_HAZARD:
                    continue  # never downgrade an actual hazard cell
                out[idx] = max(out[idx], min(inflated_cost, COST_HAZARD - 1))
    return out


def _neighbors(col: int, row: int, width: int, height: int):
    for dc in (-1, 0, 1):
        for dr in (-1, 0, 1):
            if dc == 0 and dr == 0:
                continue
            nc, nr = col + dc, row + dr
            if 0 <= nc < width and 0 <= nr < height:
                yield nc, nr, math.hypot(dc, dr)


def _heuristic(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def astar_path(grid: List[int], width: int, height: int,
                start: Tuple[int, int], goal: Tuple[int, int]
                ) -> Optional[List[Tuple[int, int]]]:
    """8-connected A* over `grid` (row-major, length width*height, the same
    convention as traversability_grid.py). Returns a list of (col, row)
    cells from start to goal inclusive, or None if no path exists (start/
    goal out of bounds or blocked, or the goal is unreachable without
    crossing a hazard cell)."""

    def in_bounds(cell: Tuple[int, int]) -> bool:
        c, r = cell
        return 0 <= c < width and 0 <= r < height

    def cost_at(cell: Tuple[int, int]) -> int:
        c, r = cell
        return grid[r * width + c]

    if not in_bounds(start) or not in_bounds(goal):
        return None
    if cell_blocked(cost_at(start)) or cell_blocked(cost_at(goal)):
        return None
    if start == goal:
        return [start]

    open_heap = [(0.0, start)]
    came_from = {}
    g_score = {start: 0.0}

    while open_heap:
        _, current = heapq.heappop(open_heap)
        if current == goal:
            path = [current]
            while path[-1] != start:
                path.append(came_from[path[-1]])
            path.reverse()
            return path

        cc, cr = current
        for nc, nr, move_dist in _neighbors(cc, cr, width, height):
            neighbor = (nc, nr)
            n_cost = grid[nr * width + nc]
            if cell_blocked(n_cost):
                continue
            tentative_g = g_score[current] + move_dist * step_cost(n_cost)
            if tentative_g < g_score.get(neighbor, math.inf):
                g_score[neighbor] = tentative_g
                came_from[neighbor] = current
                f_score = tentative_g + _heuristic(neighbor, goal)
                heapq.heappush(open_heap, (f_score, neighbor))

    return None


def path_to_world(path: List[Tuple[int, int]]) -> List[Tuple[float, float]]:
    """Convert a cell path to world (x, y) waypoints at cell centres --
    inverse of traversability_grid.py's world_to_cell()."""
    return [
        (ORIGIN_X_M + (col + 0.5) * RESOLUTION_M, ORIGIN_Y_M + (row + 0.5) * RESOLUTION_M)
        for col, row in path
    ]
