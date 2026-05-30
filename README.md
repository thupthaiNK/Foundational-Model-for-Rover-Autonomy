# Thupthai Chaiyadecha

MSc Robotics student at **Cranfield University**

Thesis: **Foundational Model for Rover Autonomy**
Building an ExoMy rover (Raspberry Pi 4) with a cascade FM perception pipeline for terrain understanding — no cloud, all on-device.

---

## Research

Adapting pretrained foundation / vision-language models for planetary rover autonomy:

- Zero-shot terrain classification (CLIP)
- Scene segmentation (SAM / SAM2)
- Scene description and reasoning (BLIP-2)
- Integrated via ROS2 Humble on Raspberry Pi 4

Research question: Can a frozen cascade pipeline (CLIP + SAM + BLIP-2) provide reliable terrain understanding and traversability assessment on RPi-class hardware?

---

## Tech Stack

| Area | Tools |
|------|-------|
| Foundation Models | CLIP · BLIP-2 · SAM · DINOv2 · LLaVA |
| Robotics | ROS2 Humble · Gazebo · ExoMy |
| ML / Vision | PyTorch · HuggingFace Transformers · OpenCV |
| Languages | Python · Bash |
| Hardware | Raspberry Pi 4 (8GB) |

---

## Key Literature Gaps This Thesis Addresses

1. No existing benchmark of FM inference on RPi-class hardware
2. No unified VLM + traversability pipeline for low-cost rovers
3. No ROS2-standardised FM perception node

---

*Currently assembling the ExoMy rover and setting up ROS2/Gazebo simulation — experiments begin June 2026.*
