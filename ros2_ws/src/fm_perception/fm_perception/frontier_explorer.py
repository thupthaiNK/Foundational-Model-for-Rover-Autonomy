"""
Purpose: Pure frontier-detection and selection functions for "frontier
         exploration-lite" (scoped via grill-thesis 2026-07-18): the rover
         autonomously selects the nearest unexplored-boundary cell inside a
         bounded exploration box as its next goal, repeating with no
         operator command until no frontiers remain -- the second sliver of
         L6 (Mission Autonomy) after L6-lite's waypoint sequencing,
         addressing the "cannot systematically cover an area" capability
         gap (Ch5 §5.6.4). "Explored" is defined perception-first: a cell
         counts as explored once DINOv2 (via the live traversability
         costmap running in init_unknown mode, cells -1 until painted) has
         assessed it -- deliberately NOT the LiDAR/SLAM map, because this
         Gazebo world's open quadrants give the LiDAR nothing to return, so
         slam_toolbox's map only grows near the Q4 rocks, while costmap
         painting follows the rover's own pose everywhere.
Inputs:  Costmap grids as row-major List[int] (same convention as
         OccupancyGridBuilder / astar_planner): -1 unknown, 0..99 known
         traversable cost, 100 known hazard.
Outputs: Frontier cell lists / selected goal cells as (col, row) tuples.
How to run:
    python3 -m pytest src/fm_perception/test/test_frontier_explorer.py -v
Project: Onboard Visual Foundation Models for Mars Terrain Perception and Rover Navigation
"""
import math
from typing import Iterable, List, Optional, Set, Tuple

from fm_perception.traversability_grid import COST_BEDROCK, COST_HAZARD, RESOLUTION_M

UNKNOWN_COST = -1
# Matches OccupancyGridBuilder's default patch_radius_m=0.3 (traversability_grid.py)
# -- the maximum extent of a single self-paint disc, so grid_with_start_freed()
# can always escape one while leaving distant real hazard walls intact.
ESCAPE_PATCH_RADIUS_M = 0.3
ESCAPE_RADIUS_CELLS = round(ESCAPE_PATCH_RADIUS_M / RESOLUTION_M)

Cell = Tuple[int, int]
Box = Tuple[int, int, int, int]  # (col_min, row_min, col_max, row_max), inclusive


def find_frontier_cells(grid: List[int], width: int, height: int,
                         box: Box) -> List[Cell]:
    """Classic frontier definition (Yamauchi 1997), restricted to a bounded
    exploration box: an unknown cell (UNKNOWN_COST) with at least one
    8-neighbour that is known AND non-hazard (0 <= cost < COST_HAZARD).
    Hazard-only neighbours do not qualify -- a frontier reachable only
    through known hazard is not a useful goal for a planner that treats
    hazard cells as impassable (astar_planner.cell_blocked)."""
    col_min, row_min, col_max, row_max = box
    frontiers: List[Cell] = []
    for row in range(max(0, row_min), min(height - 1, row_max) + 1):
        for col in range(max(0, col_min), min(width - 1, col_max) + 1):
            if grid[row * width + col] != UNKNOWN_COST:
                continue
            for dc in (-1, 0, 1):
                for dr in (-1, 0, 1):
                    if dc == 0 and dr == 0:
                        continue
                    nc, nr = col + dc, row + dr
                    if not (0 <= nc < width and 0 <= nr < height):
                        continue
                    neighbour = grid[nr * width + nc]
                    if 0 <= neighbour < COST_HAZARD:
                        frontiers.append((col, row))
                        break
                else:
                    continue
                break
    return frontiers


def is_bedrock_adjacent(grid: List[int], width: int, cell: Cell) -> bool:
    """True if any in-bounds 8-neighbour of `cell` is known bedrock
    (COST_BEDROCK). Semantic-biased frontier selection (scoped via
    grill-thesis 2026-07-18 night): bedrock = exposed outcrop, the
    science-interest signal in the Candela & Wettergreen (2022)
    science-aware exploration framing. Exact-cost match is safe because
    every known cost comes from LABEL_TO_COST's fixed values. Same
    8-neighbour convention as find_frontier_cells()."""
    height = len(grid) // width
    col, row = cell
    for dc in (-1, 0, 1):
        for dr in (-1, 0, 1):
            if dc == 0 and dr == 0:
                continue
            nc, nr = col + dc, row + dr
            if not (0 <= nc < width and 0 <= nr < height):
                continue
            if grid[nr * width + nc] == COST_BEDROCK:
                return True
    return False


def select_nearest_frontier(frontiers: Iterable[Cell], robot_cell: Cell,
                             blacklist: Optional[Set[Cell]] = None,
                             min_distance_cells: float = 0.0,
                             grid: Optional[List[int]] = None,
                             width: Optional[int] = None
                             ) -> Optional[Cell]:
    """Nearest-by-euclidean-distance frontier, skipping any the caller has
    blacklisted (e.g. frontiers A* already failed to reach) and any closer
    than min_distance_cells. The minimum distance exists because the first
    painted patch's boundary includes cells right next to the rover (found
    live in the first smoke test, 2026-07-18): without it, the nearest
    frontier can already sit inside goal_tolerance, so goal_reached() fires
    on the very next control tick and the same cell gets re-selected at
    10 Hz forever. It must stay above goal_tolerance (0.2 m) but below the
    first patch's outer frontier ring (~0.9 m from the rover at
    lookahead 0.6 m + patch radius 0.3 m), or no first frontier could ever
    be selected. Returns None once nothing selectable remains -- the
    mission-complete signal.

    Semantic bias (opt-in via grid+width): when supplied, candidates that
    survive the blacklist/min-distance filters are partitioned by
    is_bedrock_adjacent(); if any bedrock-adjacent candidate exists, the
    selection pool is restricted to those (lexicographic priority -- a
    deliberate no-tuned-weight design, so the preference is a checkable
    invariant rather than a magic number), with nearest-by-distance
    deciding within the pool. Without grid, behaviour is exactly the
    pre-existing nearest-frontier rule."""
    blacklist = blacklist or set()
    semantic = grid is not None and width is not None
    best: Optional[Cell] = None
    best_dist = math.inf
    best_bedrock: Optional[Cell] = None
    best_bedrock_dist = math.inf
    rc, rr = robot_cell
    for cell in frontiers:
        if cell in blacklist:
            continue
        dist = math.hypot(cell[0] - rc, cell[1] - rr)
        if dist < min_distance_cells:
            continue
        if dist < best_dist:
            best, best_dist = cell, dist
        if semantic and dist < best_bedrock_dist and \
                is_bedrock_adjacent(grid, width, cell):
            best_bedrock, best_bedrock_dist = cell, dist
    return best_bedrock if best_bedrock is not None else best


def grid_with_start_freed(grid: List[int], width: int, start_cell: Cell) -> List[int]:
    """If the start cell reads as hazard (>= COST_HAZARD), return a COPY with
    hazard cells within ESCAPE_RADIUS_CELLS of it freed to 0; otherwise
    return the grid unchanged. Root cause (2026-07-19 smoke test): pure-
    pursuit deviation can carry the rover onto a cell its own paint disc
    mis-labelled hazard, and astar_path() then refuses every goal because
    the start cell itself is blocked -- proprioception (the rover is
    physically standing there) beats a mispainted disc. Same circular-disc
    shape as OccupancyGridBuilder.paint_lookahead() so a single self-paint
    is always fully escaped. Never mutates the input."""
    height = len(grid) // width
    col, row = start_cell
    if grid[row * width + col] < COST_HAZARD:
        return grid
    freed = list(grid)
    r = ESCAPE_RADIUS_CELLS
    for dc in range(-r, r + 1):
        for dr in range(-r, r + 1):
            if dc * dc + dr * dr > r * r:
                continue
            nc, nr = col + dc, row + dr
            if not (0 <= nc < width and 0 <= nr < height):
                continue
            if freed[nr * width + nc] >= COST_HAZARD:
                freed[nr * width + nc] = 0
    return freed


def select_lowest_confidence_cell(cost_grid: List[int], confidence_grid: List[int],
                                   width: int, box: Box, robot_cell: Cell,
                                   visited: Set[Cell],
                                   min_distance_cells: float = 0.0) -> Optional[Cell]:
    """A2 re-observation target selection (grill-scoped 2026-07-19): the
    known non-hazard cell inside the box whose recorded confidence is
    lowest, nearest-by-distance breaking ties among equal minima.
    Lowest-first was a deliberate design choice over a "revisit everything
    below threshold C" pool -- no tuned constant to justify, and the
    officially-checked invariant (selected confidence == pool minimum) is a
    logical property of this rule. Eligible cells must be: inside the box;
    known non-hazard (0 <= cost < COST_HAZARD -- hazard is not somewhere to
    drive, however unconfident its observation; unknown has no observation
    to be unconfident about); actually observed (confidence >= 0, since -1
    means never painted with confidence data); not already visited this
    mission (each cell is re-observed at most once -- an ambiguous spot
    that stays low-confidence after revisiting must not be selected
    forever, the same role the frontier blacklist plays); and at least
    min_distance_cells away (same guard, same 10 Hz re-selection bug it
    prevents, as select_nearest_frontier). Returns None when nothing is
    selectable -- the re-observation-complete signal."""
    col_min, row_min, col_max, row_max = box
    height = len(cost_grid) // width
    rc, rr = robot_cell
    best: Optional[Cell] = None
    best_conf = None
    best_dist = math.inf
    for row in range(max(0, row_min), min(height - 1, row_max) + 1):
        for col in range(max(0, col_min), min(width - 1, col_max) + 1):
            idx = row * width + col
            if not (0 <= cost_grid[idx] < COST_HAZARD):
                continue
            conf = confidence_grid[idx]
            if conf < 0:
                continue
            cell = (col, row)
            if cell in visited:
                continue
            dist = math.hypot(col - rc, row - rr)
            if dist < min_distance_cells:
                continue
            if best_conf is None or conf < best_conf or \
                    (conf == best_conf and dist < best_dist):
                best, best_conf, best_dist = cell, conf, dist
    return best


def grid_for_planning(grid: List[int]) -> List[int]:
    """Copy of the grid with unknown cells mapped to cost 0 (free). A* must
    treat unexplored space as traversable -- otherwise no path into it could
    ever be planned (classic frontier-exploration convention) -- and
    astar_planner.step_cost() would misbehave on a negative cost. Known
    costs, including hazard, are preserved. Never mutates the input."""
    return [0 if c == UNKNOWN_COST else c for c in grid]


def count_known_cells_in_box(grid: List[int], width: int, box: Box) -> int:
    """Number of non-unknown cells inside the box -- the coverage metric
    reported (descriptively, not pass/fail) by the exploration recorder."""
    col_min, row_min, col_max, row_max = box
    count = 0
    for row in range(row_min, row_max + 1):
        for col in range(col_min, col_max + 1):
            if grid[row * width + col] != UNKNOWN_COST:
                count += 1
    return count
