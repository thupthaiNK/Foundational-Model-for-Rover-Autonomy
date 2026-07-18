# Thupthai Chaiyadecha

MSc Astronautics and Space Engineering student at **Cranfield University**.

Thesis: **Foundational Model for Rover Autonomy** — adapting pretrained foundation and
vision-language models for terrain understanding and traversability on a low-cost,
Raspberry-Pi-class planetary rover, evaluated in ROS2/Gazebo simulation and on physical
ExoMy hardware.

This thesis is currently **in progress** (not yet submitted/defended). This repository
holds a public-facing summary, representative code, and selected results. The full
working repository (600+ files: all experiments, logs, and drafts) is private while the
thesis is being written.

---

## Project overview

**[Read the full project overview →](PROJECT_OVERVIEW.md)**

Research question: can a cascade of frozen, pretrained foundation models (no training
from scratch) provide reliable terrain classification and traversability-based safety on
resource-constrained rover hardware?

- **Terrain classification**: DINOv2 ViT-S/14 (+ registers) with a lightweight linear
  probe, 90.24% accuracy on AI4Mars, benchmarked against 21 other vision/
  vision-language models.
- **Reactive safety**: confidence-gated STOP/SLOW/GO policy, verified 5/5 in constructed
  Gazebo hazard scenarios.
- **Traversability estimation**: continuous cost score (not just discrete thresholds),
  live-verified in simulation.
- **Platform**: ExoMy rover (Raspberry Pi 4), ROS2 Humble, Gazebo, PyTorch/HuggingFace
  Transformers.

## Code samples

[`code-samples/`](code-samples/) — a few representative ROS2 nodes from the full
pipeline:

| File | What it does |
|---|---|
| [`dinov2_terrain_node.py`](code-samples/dinov2_terrain_node.py) | Frozen DINOv2 encoder + linear probe terrain classifier |
| [`terrain_controller_node.py`](code-samples/terrain_controller_node.py) | Reactive safety policy (confidence → STOP/SLOW/GO) |
| [`traversability_grid.py`](code-samples/traversability_grid.py) | Builds an occupancy/cost grid from terrain classifications |
| [`astar_planner.py`](code-samples/astar_planner.py) | Lightweight A* path planner over the traversability grid |
| [`path_follower.py`](code-samples/path_follower.py) | Pure-pursuit path following |

## Figures

[`figures/`](figures/) — selected results (model comparison, confusion matrix, live
hazard-avoidance behaviour in Gazebo/RViz).

---

## Tech stack

| Area | Tools |
|---|---|
| Foundation models | CLIP · DINOv2 · BLIP-2 · SAM/MobileSAM · SmolVLM |
| Robotics | ROS2 Humble · Gazebo · ExoMy |
| ML / vision | PyTorch · HuggingFace Transformers · OpenCV |
| Hardware | Raspberry Pi 4 (8GB) |
