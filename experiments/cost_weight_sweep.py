#!/usr/bin/env python3
"""
Purpose: Offline COST_WEIGHT sensitivity sweep for the L5-lite A* planner.
         astar_planner.step_cost() charges 1.0 + COST_WEIGHT*(cost/100) per
         step, so COST_WEIGHT sets how strongly the planner trades extra
         distance for cheaper terrain. This sweep quantifies that trade-off
         on the deterministic Condition A ground-truth grid (no Gazebo, no
         randomness): plan sand_zone -> bedrock_zone around the hazard
         quadrant for a range of weights, and report path length and
         terrain composition per weight. Purely descriptive ablation; the
         deployed default (1.0) is unchanged.
Inputs:  build_static_grid() from fm_perception.traversability_grid
         (Condition A: every cell pre-filled from its zone's ground truth).
Outputs: experiments/results/cost_weight_sweep.csv
         experiments/results/figures/cost_weight_sweep.png
How to run:
    source /opt/ros/humble/setup.bash && source ros2_ws/install/setup.bash
    python3 experiments/cost_weight_sweep.py
Project: Onboard Visual Foundation Models for Mars Terrain Perception and Rover Navigation
"""
import csv
import math
import os
import time
from typing import List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import fm_perception.astar_planner as astar_planner
from fm_perception.astar_planner import astar_path, path_to_world
from fm_perception.traversability_grid import (
    COST_BEDROCK, COST_HAZARD, COST_SAND, COST_SOIL, HEIGHT_CELLS,
    RESOLUTION_M, WIDTH_CELLS, build_static_grid, world_to_cell,
    zone_for_point,
)

# Start in sand_zone, goal in bedrock_zone: the only non-hazard route runs
# through the western half (rock_cluster/boulder_zone block x>0, y<0), and
# the crossing offers a genuine terrain choice -- shorter diagonals stay on
# sand (cost 35) longer, while soil (cost 0) rewards a northern detour.
# Bedrock (65) must be entered eventually to reach the goal.
START_WORLD = (-7.5, -6.0)
GOAL_WORLD = (7.5, 6.0)

WEIGHTS = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0]

COST_NAME = {COST_SOIL: "soil", COST_SAND: "sand",
             COST_BEDROCK: "bedrock", COST_HAZARD: "hazard"}

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")


def path_metrics(grid: List[int], cells: List[Tuple[int, int]]):
    """Length in metres (cell-centre polyline) + per-terrain cell counts."""
    world = path_to_world(cells)
    length = sum(math.hypot(x2 - x1, y2 - y1)
                 for (x1, y1), (x2, y2) in zip(world, world[1:]))
    composition = {name: 0 for name in COST_NAME.values()}
    for col, row in cells:
        composition[COST_NAME[grid[row * WIDTH_CELLS + col]]] += 1
    return length, composition


def main() -> None:
    os.makedirs(FIGURES_DIR, exist_ok=True)
    grid = build_static_grid()
    start = world_to_cell(*START_WORLD)
    goal = world_to_cell(*GOAL_WORLD)
    default_weight = astar_planner.COST_WEIGHT

    rows = []
    paths_world = {}
    for w in WEIGHTS:
        astar_planner.COST_WEIGHT = w  # step_cost() reads this at call time
        t0 = time.time()
        cells = astar_path(grid, WIDTH_CELLS, HEIGHT_CELLS, start, goal)
        plan_s = time.time() - t0
        assert cells, f"no path at COST_WEIGHT={w} (unexpected: route exists)"
        length, comp = path_metrics(grid, cells)
        assert comp["hazard"] == 0, "path crossed hazard -- must never happen"
        rows.append([w, f"{length:.2f}", len(cells), comp["soil"],
                     comp["sand"], comp["bedrock"], f"{plan_s:.3f}"])
        paths_world[w] = path_to_world(cells)
        print(f"COST_WEIGHT={w:>4}: length={length:6.2f} m, cells={len(cells)}"
              f" (soil={comp['soil']}, sand={comp['sand']},"
              f" bedrock={comp['bedrock']}), plan={plan_s:.3f}s")
    astar_planner.COST_WEIGHT = default_weight

    csv_path = os.path.join(RESULTS_DIR, "cost_weight_sweep.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["cost_weight", "length_m", "path_cells",
                         "soil_cells", "sand_cells", "bedrock_cells",
                         "plan_time_s"])
        writer.writerows(rows)

    # Figure: zone map with one path per weight.
    fig, ax = plt.subplots(figsize=(9, 7))
    import numpy as np
    img = np.array(grid, dtype=float).reshape(HEIGHT_CELLS, WIDTH_CELLS)
    ax.imshow(img, origin="lower", cmap="YlOrBr", alpha=0.5,
              extent=[-15.0, 15.0, -12.0, 12.0], aspect="equal")
    for w in WEIGHTS:
        xs = [p[0] for p in paths_world[w]]
        ys = [p[1] for p in paths_world[w]]
        ax.plot(xs, ys, linewidth=1.5, label=f"COST_WEIGHT={w}")
    ax.scatter(*START_WORLD, marker="s", s=60, color="black", zorder=3)
    ax.scatter(*GOAL_WORLD, marker="*", s=120, color="black", zorder=3)
    ax.annotate("start (sand)", START_WORLD, textcoords="offset points",
                xytext=(6, 6), fontsize=8)
    ax.annotate("goal (bedrock)", GOAL_WORLD, textcoords="offset points",
                xytext=(6, 6), fontsize=8)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("A* terrain-cost weight sweep (Condition A ground-truth grid)")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, "cost_weight_sweep.png")
    fig.savefig(fig_path, dpi=150)
    print(f"Saved: {csv_path}")
    print(f"Saved: {fig_path}")


if __name__ == "__main__":
    main()
