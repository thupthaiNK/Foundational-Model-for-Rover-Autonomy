# Onboard Visual Foundation Models for Mars Terrain Perception and Rover Navigation

MSc thesis (Astronautics and Space Engineering), Cranfield University, August 2026.
Thupthai Chaiyadecha, supervised by Dr Saurabh Upadhyay.

This repository holds the working code behind the thesis: every experiment script,
the ROS2 packages deployed on the rover, the Gazebo simulation assets, and the result
CSVs and figures the report's numbers trace back to. It is described in full in the
report's Appendix A ("Code and repository").

**[Read the thesis (PDF)](docs/files/thesis.pdf)** · **[Project website](https://thupthaink.github.io/Onboard-Visual-Foundation-Models-for-Mars-Terrain-Perception-and-Rover-Navigation/)** · **[Video playlist](https://www.youtube.com/playlist?list=PLWlI8ZIzh2Es)**

## What this thesis does

Mars rovers cannot be controlled in real time from Earth, so they need to judge
terrain and drive safely on their own. This thesis tests whether pretrained visual
foundation models — frozen, with no Mars-specific training — can do that job on
low-cost edge hardware. Twenty-two frozen encoders across eight pretraining paradigms
were evaluated on AI4Mars; the best result reaches 94.43% accuracy, within 2.24 points
of a supervised baseline trained end-to-end. The chosen model, DINOv2+reg ViT-S/14,
was deployed in a ROS2 system on a Raspberry Pi 4 and tested in Gazebo and on a real
ESA ExoMy rover.

## Repository structure

| Path | Contents |
| --- | --- |
| `experiments/` | Every experiment script, their output CSVs and figures under `experiments/results/`, and the MATLAB figure-generation scripts used for the report |
| `ros2_ws/src/fm_perception/` | The perception package: CLIP, DINOv2, SmolVLM, and BLIP-2 ROS2 nodes, the traversability controller, the reactive-exploration and stuck-detection state machines |
| `ros2_ws/src/fm_imu_fusion/` | The IMU driver, slope-fusion logic, and the LiDAR/IMU/camera traversability fusion node |
| `ros2_ws/src/exomy_ros2/` | The ported ExoMy ROS2 hardware driver and the cmd_vel-to-RoverCommand bridge |
| `ros2_ws/src/exomy_ros2_msgs/` | Custom ROS2 message definitions |
| `simulation/` | Gazebo world files, the ExoMy URDF/xacro model, and launch files |

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

Full setup and every reproduction command is in the thesis's Appendix A and Appendix B
(the reproducibility map, mapping every headline number to the script that produced it).

## What is not in this repository

Real-hardware rosbag recordings and raw camera captures are not committed, since they
are large and include images of the laboratory. Trained linear-probe weights and
ONNX-exported encoders are regenerable from the AI4Mars dataset and the scripts above
rather than committed directly, to keep the repository a reasonable size. Video
recordings are on the [playlist](https://www.youtube.com/playlist?list=PLWlI8ZIzh2Es)
instead.

## Tech stack

Python 3.10+, ROS2 Humble, Gazebo Classic 11, PyTorch 2.5.1, HuggingFace Transformers,
ONNX Runtime, OpenCV, scikit-learn, pandas, MATLAB R2025b (for the report's figures).
Full dependency list in [`requirements.txt`](requirements.txt).

Models evaluated: CLIP, SigLIP2, DINOv2, DINOv3, EVA-02, AIMv2, RADIO, Franca, BLIP-2,
SmolVLM, SAM2. Deployed on: ESA ExoMy rover (Raspberry Pi 4).
