# Foundational Model for Rover Autonomy — Project Overview

MSc thesis, Cranfield University (Astronautics and Space Engineering).
**Status: in progress, not yet submitted or defended.** This document summarises the
project for a public audience; it is a curated overview, not the full thesis text.

## Motivation

Planetary rovers with limited compute (e.g. Raspberry-Pi-class onboard hardware, no
cloud connectivity) still need to understand the terrain in front of them: is it safe
to drive over, or does it need to slow down or stop? Training a bespoke perception
model from scratch is expensive and needs large labelled datasets. This thesis instead
asks: **can pretrained, frozen foundation and vision-language models — used zero-/
few-shot, with only a lightweight adapter on top — do this job well enough for a real
rover, on real onboard-class hardware?**

The evaluation platform is [ExoMy](https://github.com/esa-prl/ExoMy), an open-source
Raspberry-Pi-based rover, driven in both ROS2/Gazebo simulation and on physical
hardware.

## System architecture: a six-layer autonomy stack

The full autonomy problem (perceive terrain → stay safe → estimate cost → know where
you are → plan a route → decide what to do next) is treated as six layers. The thesis
scopes its contribution honestly layer by layer rather than claiming full autonomy:

| Layer | Function | Status |
|---|---|---|
| L1 | Terrain semantics — classify terrain from camera | **Core contribution** — 90.24% accuracy on AI4Mars |
| L2 | Reactive safety — confidence → STOP / SLOW / GO | **Done** — 5/5 in constructed Gazebo hazard scenarios |
| L3 | Traversability estimation — continuous cost | **Done** — continuous score, live-verified in simulation |
| L4 | Localisation — where is the rover? | **Partial** — works in simulation with ground-truth-assisted SLAM; real-hardware localisation without that assist remains an open problem |
| L5 | Path planning — how to navigate to a goal? | **Partial** — a lightweight A* planner + path follower was built and demonstrated live in simulation (see Key Results); full Nav2-scale planning was scoped out as too heavy for the target hardware |
| L6 | Mission autonomy — what to do next? | Out of scope for this thesis (future work) |

## Key results

- **Terrain classification**: DINOv2 ViT-S/14 (+ registers) with a lightweight linear
  probe reaches **90.24% accuracy on AI4Mars**, benchmarked head-to-head against 21
  other vision and vision-language models (CLIP, BLIP-2, DINOv3, SmolVLM, and others).
- **On-device efficiency**: an INT8-quantised ONNX export of the same model cuts
  inference latency roughly 570ms → 386ms and RAM roughly 220MB → 163MB on
  Raspberry-Pi-class hardware, at a measured cost of about 0.5 percentage points of
  accuracy.
- **Reactive safety**: a confidence-gated STOP/SLOW/GO policy, tuned to an optimal
  decision threshold, achieved 5/5 successful outcomes across constructed Gazebo hazard
  scenarios (rover consistently stops rather than driving into unsafe terrain when
  perception confidence is low).
- **Lightweight path planning ("L5-lite")**: rather than running the full Nav2 stack
  (found to be too heavy for the 4-core development machine used for simulation), a
  custom, ~200-line A* planner plus a pure-pursuit path follower were built and tested
  end-to-end in Gazebo. Across a 300-second live run with the full perception pipeline
  active, the rover travelled 13.15 m (73% of the straight-line distance to a real,
  hazard-crossing goal), replanning live against the perception-derived costmap 58
  times.
- **Physical hardware**: the ExoMy platform has been assembled, and the full autonomous
  stack (camera-based terrain classification, LiDAR obstacle avoidance, IMU tilt
  sensing, reactive exploration) has been run and validated on physical hardware —
  including outdoor sand testing and a printed Mars-terrain backdrop test — driving
  itself, with no manual remote control, across dozens of real test runs.

## Honest limitations

This is reported explicitly rather than glossed over. Written so it's understandable
even without a robotics background:

- Simulation results do not transfer perfectly to the real world (a sim-to-real
  perception gap is present and measured, not assumed away).
- Real-hardware localisation without a ground-truth assist (the case the physical rover
  actually faces) remains unsolved; this bounds what the path-planning result above can
  claim for real deployment.
- The delivered system can drive itself safely through unfamiliar terrain reactively
  (classify, modulate speed, stop on hazard, recover from being stuck) — it is not yet
  a system that can be given a destination and reliably navigate there fully
  autonomously on real hardware.
- **The vision model struggles to recognise large rocks specifically** (it reliably
  tells apart flat rock, sand, and loose soil, but isolated boulders are much harder).
  This traces back to the training data having far fewer example images of large rocks
  than of the other terrain types — not a flaw in the AI model itself. As a safety
  measure, whenever the model is unsure it defaults to stopping the rover rather than
  guessing, so this weakness does not turn into an unsafe decision.
- **On everyday indoor surfaces the model doesn't know it's "out of its depth."**
  Trained only on real Mars/Mars-like imagery, it will still confidently label an
  office floor or wall as one of its known terrain types (usually "soil") rather than
  flagging "I don't recognise this." This is a known limitation of the approach and
  is discussed as a direction for future work rather than something the current system
  claims to solve.
- **The distance sensor (LiDAR) only "sees" objects at one fixed height.** It reliably
  detects obstacles at its own scan height, but can miss things that sit entirely
  above or below that line — for example, thin furniture legs, or an obstacle on the
  far side of a small rise in the ground that tips the sensor's view up and over it.
  Two such misses were observed directly during hardware testing (a low obstacle and a
  small rise). A rover intended for more cluttered, uneven real-world environments
  would need a sensor that scans more than one height (e.g. a 3D LiDAR or a stereo
  depth camera) to close this gap; that is out of scope for this project's hardware
  budget and is noted as future work.

## Tech stack

Python · ROS2 Humble · Gazebo · PyTorch · HuggingFace Transformers · OpenCV ·
scikit-learn.

## About this repository

This is a curated public summary. The full working repository — every experiment
script, log, and draft chapter accumulated over the course of the thesis — stays
private until the work is complete, both because it is still changing day to day and
because some of that material (in-progress drafts, raw logs) isn't meant for a general
audience. If you'd like more detail (e.g. as a supervisor or examiner), please get in
touch directly.
