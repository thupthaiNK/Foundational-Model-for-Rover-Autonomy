"""
Purpose: Pure-Python traversability occupancy-grid geometry and cost logic,
         shared by the static and live Nav2 costmap ROS2 nodes. No ROS2 or
         Gazebo dependency — fully unit-testable in isolation.
Inputs:  None directly; consumed by static_traversability_costmap_node.py
         and live_traversability_costmap_node.py.
Outputs: Grid geometry constants, cost_for_label(), zone_for_point(),
         world_to_cell(), build_static_grid(), OccupancyGridBuilder.
How to run: Imported by the costmap nodes. Tested via
         ros2_ws/src/fm_perception/test/test_traversability_grid.py
Project: Foundational Model for Rover Autonomy
"""
import math
from dataclasses import dataclass, field
from typing import List, Tuple

# Grid geometry — matches the 5-zone Gazebo arena (mars_terrain.world) extent
# and coordinate convention used by experiments/risk_score_map.py's ZONES table.
RESOLUTION_M = 0.1          # metres per cell
ORIGIN_X_M = -15.0          # world x of cell (col=0), metres
ORIGIN_Y_M = -12.0          # world y of cell (row=0), metres
WIDTH_CELLS = 300           # 30m / 0.1m
HEIGHT_CELLS = 240          # 24m / 0.1m

# Occupancy values — nav_msgs/OccupancyGrid convention (0=free .. 100=occupied,
# -1=unknown). NOT the internal Nav2 0-255 cost scale; static_layer converts.
COST_SOIL = 0
COST_SAND = 35
COST_BEDROCK = 65
COST_HAZARD = 100   # big_rock / uncertain / unknown

LABEL_TO_COST = {
    "soil": COST_SOIL,
    "sand": COST_SAND,
    "bedrock": COST_BEDROCK,
    "big_rock": COST_HAZARD,
    "uncertain": COST_HAZARD,
    "unknown": COST_HAZARD,
}

# Known zone bounding boxes — same geometry as experiments/risk_score_map.py
# ZONES table: (name, ground_truth_label, x_lo, x_hi, y_lo, y_hi).
ZONES = [
    ("soil_zone",    "soil",      -15.0,  0.0,   0.0,  12.0),
    ("bedrock_zone", "bedrock",     0.0, 15.0,   0.0,  12.0),
    ("sand_zone",    "sand",      -15.0,  0.0, -12.0,   0.0),
    ("rock_cluster", "big_rock",    0.0, 15.0,  -6.0,   0.0),
    ("boulder_zone", "big_rock",    0.0, 15.0, -12.0,  -6.0),
]


def cost_for_label(label: str) -> int:
    """Map a DINOv2 terrain label to an OccupancyGrid cost value (0-100)."""
    return LABEL_TO_COST.get(label, COST_HAZARD)


def zone_for_point(x: float, y: float) -> str:
    """Return the zone name containing world point (x, y), or 'outside_arena'."""
    for name, _gt_label, x_lo, x_hi, y_lo, y_hi in ZONES:
        if x_lo <= x < x_hi and y_lo <= y < y_hi:
            return name
    return "outside_arena"


def world_to_cell(x: float, y: float) -> Tuple[int, int]:
    """Convert world (x, y) metres to (col, row) grid cell indices."""
    col = int((x - ORIGIN_X_M) / RESOLUTION_M)
    row = int((y - ORIGIN_Y_M) / RESOLUTION_M)
    return col, row


def build_static_grid() -> List[int]:
    """Build the Condition A grid: every cell pre-filled from its known
    zone's ground-truth label. Row-major, length WIDTH_CELLS*HEIGHT_CELLS."""
    grid = [COST_HAZARD] * (WIDTH_CELLS * HEIGHT_CELLS)  # outside any known
                                                          # zone -> treat as hazard
    for col in range(WIDTH_CELLS):
        for row in range(HEIGHT_CELLS):
            x = ORIGIN_X_M + (col + 0.5) * RESOLUTION_M
            y = ORIGIN_Y_M + (row + 0.5) * RESOLUTION_M
            zone = zone_for_point(x, y)
            if zone == "outside_arena":
                continue
            gt_label = next(z[1] for z in ZONES if z[0] == zone)
            grid[row * WIDTH_CELLS + col] = cost_for_label(gt_label)
    return grid


@dataclass
class OccupancyGridBuilder:
    """Live costmap state for Condition B: starts entirely free (no prior
    knowledge), updated cell-by-cell as DINOv2 classifications stream in."""
    lookahead_m: float = 0.6
    patch_radius_m: float = 0.3
    grid: List[int] = field(default_factory=lambda: [COST_SOIL] * (WIDTH_CELLS * HEIGHT_CELLS))

    def paint_lookahead(self, robot_x: float, robot_y: float, heading_rad: float,
                         label: str) -> None:
        """Paint a disc of cells at lookahead_m ahead of (robot_x, robot_y)
        along heading_rad, with the cost for `label`."""
        target_x = robot_x + self.lookahead_m * math.cos(heading_rad)
        target_y = robot_y + self.lookahead_m * math.sin(heading_rad)
        cost = cost_for_label(label)
        radius_cells = max(1, int(round(self.patch_radius_m / RESOLUTION_M)))
        center_col, center_row = world_to_cell(target_x, target_y)
        for dc in range(-radius_cells, radius_cells + 1):
            for dr in range(-radius_cells, radius_cells + 1):
                if dc * dc + dr * dr > radius_cells * radius_cells:
                    continue
                col, row = center_col + dc, center_row + dr
                if 0 <= col < WIDTH_CELLS and 0 <= row < HEIGHT_CELLS:
                    self.grid[row * WIDTH_CELLS + col] = cost
