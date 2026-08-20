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
Project: Onboard Visual Foundation Models for Mars Terrain Perception and Rover Navigation
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

CONFIDENCE_EPS = 0.01  # log-odds clamp bound; keeps fused confidence in
                       # (0, 1) so a future disagreeing observation can
                       # always still move it (Thrun-style occupancy clamp).

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


def _clamp_confidence(p: float) -> float:
    """Clamp to (0, 1) open interval so logit() never sees 0 or 1."""
    return min(max(p, CONFIDENCE_EPS), 1.0 - CONFIDENCE_EPS)


def _logit(p: float) -> float:
    return math.log(p / (1.0 - p))


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


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
    knowledge), updated cell-by-cell as DINOv2 classifications stream in.
    init_cost=-1 (OccupancyGrid's standard "unknown") is an opt-in for
    frontier exploration-lite, where "never assessed by perception" must be
    distinguishable from "assessed as soil" (COST_SOIL is also 0); the
    default stays COST_SOIL so every pre-existing costmap result is
    untouched. confidence_aware_painting (opt-in, default False -- root
    cause of the 2026-07-19 start-cell-hazard deadlock this session
    already patched around symptomatically in frontier_explorer.py's
    grid_with_start_freed): when True, a single low-confidence "uncertain"
    classification can no longer overwrite a cell an earlier, real
    detection painted as known non-hazard -- only genuine confident
    detections (any label other than "uncertain") may still downgrade a
    known cell. Only meaningful with init_cost=-1: COST_SOIL=0 is
    otherwise indistinguishable from "never painted", so the protection
    is a no-op (falls back to unconditional overwrite) elsewhere.
    track_confidence (opt-in, default False -- A2 epistemic uncertainty
    map, item 7, 2026-07-19): keeps a parallel per-cell confidence store
    (-1.0 = never observed), latest-write to match the cost grid's own
    last-write semantics, fed by the confidence the classifier already
    attaches to every "label:confidence" message but the costmap
    previously discarded. Zero extra memory unless enabled.
    bayesian_fusion (opt-in, default False, requires track_confidence=True
    to have any effect -- SuperMap-inspired log-odds label fusion,
    2026-07-19): replaces track_confidence's latest-write update with a
    log-odds accumulation across observations that agree on the same
    label (SuperMap, Zhao et al. 2026, eq. 10), so a cell revisited many
    times remembers accumulated evidence, not just its newest observation.
    A label change resets the belief to the new observation alone rather
    than blending with the old label's evidence. Every existing official
    result (A2, §4.8.30) used latest-write and must remain byte-for-byte
    reproducible with this flag left at its default."""
    lookahead_m: float = 0.6
    patch_radius_m: float = 0.3
    init_cost: int = COST_SOIL
    confidence_aware_painting: bool = False
    track_confidence: bool = False
    bayesian_fusion: bool = False
    grid: List[int] = field(default_factory=list)
    confidence: List[float] = field(default_factory=list)
    last_label: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.grid:
            self.grid = [self.init_cost] * (WIDTH_CELLS * HEIGHT_CELLS)
        if self.track_confidence and not self.confidence:
            self.confidence = [-1.0] * (WIDTH_CELLS * HEIGHT_CELLS)
        if self.bayesian_fusion and not self.last_label:
            self.last_label = [""] * (WIDTH_CELLS * HEIGHT_CELLS)

    def _fuse_confidence(self, idx: int, label: str, confidence: float) -> float:
        """Log-odds fusion of `confidence` into cell `idx`'s belief about
        `label`, matching SuperMap's Bayesian label-fusion update (eq. 10)
        applied per grid cell instead of per tracked object. Resets to the
        new observation alone when `label` differs from what this cell was
        last painted with -- a disagreeing hypothesis must not inherit the
        old one's accumulated evidence."""
        observed = _clamp_confidence(confidence)
        prior = self.confidence[idx]
        same_label = self.last_label[idx] == label
        if prior < 0.0 or not same_label:
            return observed
        fused_logit = _logit(prior) + _logit(observed)
        return _clamp_confidence(_sigmoid(fused_logit))

    def paint_lookahead(self, robot_x: float, robot_y: float, heading_rad: float,
                         label: str, confidence: float = None) -> None:
        """Paint a disc of cells at lookahead_m ahead of (robot_x, robot_y)
        along heading_rad, with the cost for `label`. If
        confidence_aware_painting is on and the grid uses init_cost=-1, an
        "uncertain" label skips any cell that already holds a known
        non-hazard cost -- an admission of ignorance must not erase a real
        prior observation. With track_confidence on and a confidence value
        supplied, the parallel confidence store is latest-written for
        exactly the cells whose cost was written (a skipped cell keeps its
        old confidence too -- the two stores must describe the same
        observation, never a mismatched pair)."""
        target_x = robot_x + self.lookahead_m * math.cos(heading_rad)
        target_y = robot_y + self.lookahead_m * math.sin(heading_rad)
        cost = cost_for_label(label)
        protect_known_cells = (
            self.confidence_aware_painting and self.init_cost == -1 and label == "uncertain"
        )
        write_confidence = self.track_confidence and confidence is not None
        radius_cells = max(1, int(round(self.patch_radius_m / RESOLUTION_M)))
        center_col, center_row = world_to_cell(target_x, target_y)
        for dc in range(-radius_cells, radius_cells + 1):
            for dr in range(-radius_cells, radius_cells + 1):
                if dc * dc + dr * dr > radius_cells * radius_cells:
                    continue
                col, row = center_col + dc, center_row + dr
                if 0 <= col < WIDTH_CELLS and 0 <= row < HEIGHT_CELLS:
                    idx = row * WIDTH_CELLS + col
                    if protect_known_cells and 0 <= self.grid[idx] < COST_HAZARD:
                        continue
                    self.grid[idx] = cost
                    if write_confidence:
                        if self.bayesian_fusion:
                            self.confidence[idx] = self._fuse_confidence(idx, label, confidence)
                            self.last_label[idx] = label
                        else:
                            self.confidence[idx] = confidence
