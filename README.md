# Onboard Visual Foundation Models for Mars Terrain Perception and Rover Navigation

MSc thesis (Astronautics and Space Engineering), Cranfield University, August 2026.
Thupthai Chaiyadecha, supervised by Dr Saurabh Upadhyay.

This repository holds the working code behind the thesis: every experiment script,
the ROS2 packages deployed on the rover, the Gazebo simulation assets, and the result
CSVs and figures the report's numbers trace back to. It is described in full in the
report's Appendix A ("Code and repository") and Appendix B (the reproducibility map).

**[Read the thesis (PDF)](docs/files/thesis.pdf)** · **[Project website](https://thupthaink.github.io/Onboard-Visual-Foundation-Models-for-Mars-Terrain-Perception-and-Rover-Navigation/)** · **[Video playlist](https://www.youtube.com/playlist?list=PLWlI8ZIzh2Es)**

If you are new to this project, read this file top to bottom — it is written to
be followed without the thesis PDF open, though the PDF has the full derivation
of every number and design decision.

## The problem

A Mars rover cannot be joysticked from Earth. Light-speed delay to Mars is
4–24 minutes each way, so a command sent in response to what the rover currently
sees is already stale by the time it arrives, and the rover's very next move might
put it into a hazard the operator never saw. The rover therefore has to look at the
ground in front of it and decide, on its own and in real time, what is safe to
drive over. That decision has to run on the rover's own onboard compute — a
low-power, resource-constrained computer, not a data-centre GPU.

Two ways of building that "is this ground safe" judgement exist. One is to train
a model from scratch on Mars images, which needs enough labelled Mars terrain data
to teach a network from nothing — expensive to collect and slow to adapt to a new
mission's imagery. The other, which this thesis investigates, is to take a
**visual foundation model** — a large neural network already pretrained on
huge, general (mostly Earth) image datasets — and use it **frozen**, with no
retraining at all, adding only a small trained layer on top to map its general
visual understanding onto Mars terrain classes. Frozen encoders are far cheaper to
deploy and can, in principle, transfer visual concepts (edges, texture, geometry)
learned from millions of Earth images to a domain they never explicitly saw.

## The question this thesis answers

**Can pretrained visual foundation models, used frozen, classify Mars terrain
accurately enough to drive a rover safely, on hardware cheap and light enough to
actually fly or be built on a budget?**

The answer is investigated in three stages, each building on the last:

1. **Which foundation model, if any, works?** Twenty-two frozen encoders spanning
   eight different pretraining paradigms (self-supervised, contrastive
   language-image, masked-image-modelling, and others) are benchmarked on
   **AI4Mars**, a public, human-labelled dataset of real Curiosity/Perseverance
   rover images with per-pixel terrain classes (soil, bedrock, sand, big rock).
2. **Does the winner survive being deployed?** The best model is wrapped in a
   ROS2 perception pipeline, calibrated so its confidence scores are trustworthy,
   and given a reactive safety policy (STOP / SLOW / GO) driven by that confidence.
3. **Does it actually drive?** The pipeline is tested in Gazebo simulation and on
   a real, low-cost rover platform (ESA's open-source **ExoMy**, on a Raspberry Pi 4),
   both on synthetic Mars-like terrain and, separately, in a real sand pit.

## Key results

| Metric | Result | Why it matters |
| --- | --- | --- |
| Best single model on AI4Mars (DINOv2+reg ViT-S/14, 1000-shot linear probe) | **90.24%** | The core empirical answer: a frozen, general-purpose encoder can match specialised Mars models |
| Best result overall (Ensemble B of several frozen encoders) | **94.43%** | Ceiling reachable without any retraining, only 2.24 points below a fully supervised, end-to-end-trained baseline |
| Calibration (Expected Calibration Error), before → after temperature scaling | 0.1695 → 0.0325 (**80.8% reduction**, T\* = 0.461) | A model's raw confidence is not trustworthy out of the box; this makes "the model is unsure" mean what it says, which the safety policy depends on |
| Gazebo terrain classification | 3/5 zones correct | Simulation exposes a real sim-to-real domain gap even after the model does well on real Mars images |
| Gazebo reactive safety (uncertain → STOP) | **5/5 zones correct** | Even where terrain *classification* is imperfect, the safety layer built on top of model *confidence* still stops the rover before every hazard |
| Mars-Bench in-domain accuracy | 84.07% | Cross-checks the AI4Mars result against a second, independent Mars terrain benchmark |
| Mars-Bench zero-shot transfer (no fine-tuning at all) | 24.94% | Quantifies exactly how much the small trained linear layer is doing — raw zero-shot transfer alone is not enough |

Full experiment-by-experiment detail, every ablation, and the reasoning behind
each design choice are in the thesis PDF, chapters 3–6.

## How the system works (deployed pipeline)

```
Camera frame
   │
   ▼
Frozen foundation-model encoder (DINOv2+reg ViT-S/14)
   │  produces a general-purpose visual feature vector, no Mars-specific
   │  weights anywhere in this step
   ▼
Trained linear probe  →  terrain class + confidence score
   │  the ONE small component trained on Mars data (AI4Mars)
   ▼
Temperature-scaled calibration (T* = 0.461)
   │  turns the raw confidence into a number that is actually reliable
   ▼
Reactive safety policy (confidence threshold 0.40)
   │  low confidence  → STOP
   │  medium confidence → SLOW
   │  high confidence  → GO, at the classified terrain's safe speed
   ▼
IMU/LiDAR fusion veto (fm_imu_fusion)
   │  slope and obstacle checks can override the vision decision,
   │  vision stays primary — sensors are a safety veto, not a second vote
   ▼
cmd_vel → ExoMy hardware driver → wheels
```

This is what "onboard" means in the title: every box above runs on the rover's
own Raspberry Pi 4, nothing is offloaded to a remote GPU or the operator.

### The autonomy layers this thesis covers

The thesis frames rover autonomy as six layers, from "see the ground" up to
"decide the mission". This repository implements Layers 1–3 in full and
demonstrates working slivers of Layers 4–6 in Gazebo (not on real hardware,
and explicitly scoped as future work — see the thesis's own honest discussion
in Ch5–Ch6, not glossed over here):

| Layer | Question it answers | Status in this thesis |
| --- | --- | --- |
| L1 — Terrain semantics | What kind of ground is this? | **Core contribution** — the 22-model benchmark above |
| L2 — Reactive safety | Should I stop, slow, or go, right now? | **Implemented and validated** — 5/5 Gazebo safety zones |
| L3 — Traversability estimation | How costly is this ground to cross? | **Substantially implemented** — continuous cost score, live-verified in Gazebo |
| L4 — Localisation | Where am I? | Partial — investigated directly, real hardware limited by sensors (monocular camera, no stereo/LiDAR depth) |
| L5 — Path planning | How do I get to a goal? | Demonstrated in Gazebo simulation only, not attempted on real hardware |
| L6 — Mission autonomy | What should I do next, unsupervised? | Minimal Gazebo-only demonstrations (waypoint sequencing, small-area coverage) |

## Repository structure

| Path | Contents |
| --- | --- |
| `experiments/` | Every experiment script (145 scripts), their output CSVs and figures under `experiments/results/`, and the MATLAB figure-generation scripts used for the report |
| `ros2_ws/src/fm_perception/` | The perception package: CLIP, DINOv2, SmolVLM, and BLIP-2 ROS2 nodes, the traversability controller, the reactive-exploration and stuck-detection state machines, and this package's own unit tests |
| `ros2_ws/src/fm_imu_fusion/` | The IMU driver, slope-fusion logic, and the LiDAR/IMU/camera traversability fusion node |
| `ros2_ws/src/exomy_ros2/` | The ported ExoMy ROS2 hardware driver and the `cmd_vel`-to-`RoverCommand` bridge |
| `ros2_ws/src/exomy_ros2_msgs/` | Custom ROS2 message definitions shared across the packages above |
| `simulation/` | Gazebo world files, the ExoMy URDF/xacro model, and launch files |
| `figures/` | Standalone figures referenced from the thesis and website |
| `docs/` | The project website source (GitHub Pages) and the published PDFs (`docs/files/`) |

If you only want to read code, start with `ros2_ws/src/fm_perception/` for the
onboard perception/safety logic, or `experiments/` for how the headline numbers
above were produced.

## How to run

```bash
# Build the ROS2 workspace
source /opt/ros/humble/setup.bash
source ros2_ws/install/setup.bash
colcon build
source install/setup.bash

# Run a standalone classification experiment
python3 experiments/<script_name>.py

# Run the repository's unit tests
python3 -m pytest ros2_ws/src/fm_perception/test/ ros2_ws/src/exomy_ros2/test/ ros2_ws/src/fm_imu_fusion/test/
```

Full setup, dataset download instructions, and the exact command that produced
every headline number are in the thesis's Appendix A ("Code and repository") and
Appendix B (the reproducibility map, mapping each result to its script). Start
there rather than guessing script arguments from this README — the appendix is
kept in sync with this repository and is the authoritative reference.

## What is not in this repository

Real-hardware rosbag recordings and raw camera captures are not committed, since
they are large and include images of the laboratory. Trained linear-probe weights
and ONNX-exported encoders are regenerable from the AI4Mars dataset and the
scripts above rather than committed directly, to keep the repository a reasonable
size. Video recordings are on the
[playlist](https://www.youtube.com/playlist?list=PLWlI8ZIzh2Es) instead. This
policy is documented in full, including exactly what is regenerable and how, in
the thesis's Appendix A.

## Limitations, honestly stated

This is a 3-month MSc thesis on a monocular-camera, low-cost rover, not a flight
mission. The results above come with real constraints the thesis discusses
explicitly rather than hides:

- **Sim-to-real gap.** The model that scores 90%+ on real Mars images only
  classifies Gazebo terrain correctly 3/5 zones — simulation textures are not
  the same visual domain as real Mars imagery or real sand.
- **No stereo depth or LiDAR-based localisation on real hardware.** ExoMy here
  carries a monocular camera, an IMU, and a 2D LiDAR; SLAM-based navigation
  (Layer 4/5) is investigated but not usable on the real rover without depth
  sensing it does not have.
- **Reactive, not purposeful, autonomy on hardware.** The real ExoMy can drive
  itself safely through unfamiliar terrain — classify terrain, modulate speed,
  stop on hazards, recover when stuck — but cannot be given a destination,
  systematically cover an area, or select its own targets. Layers 5–6 are
  demonstrated only in Gazebo.

## Tech stack

Python 3.10+, ROS2 Humble, Gazebo Classic 11, PyTorch 2.5.1, HuggingFace Transformers,
ONNX Runtime, OpenCV, scikit-learn, pandas, MATLAB R2025b (for the report's figures).
Full dependency list in [`requirements.txt`](requirements.txt).

Models evaluated: CLIP, SigLIP2, DINOv2, DINOv3, EVA-02, AIMv2, RADIO, Franca, BLIP-2,
SmolVLM, SAM2. Deployed on: ESA ExoMy rover (Raspberry Pi 4, ICM-20948 IMU, 2D LiDAR).

## Citing this work

If this repository or the thesis is useful to your own work, please cite the thesis:

```
Chaiyadecha, T. (2026). Onboard Visual Foundation Models for Mars Terrain
Perception and Rover Navigation. MSc Thesis, Cranfield University.
```

## Questions

The full derivation of every number in the tables above, every design decision,
and every limitation is written out in the [thesis PDF](docs/files/thesis.pdf).
For anything not answered there, open an issue on this repository.
