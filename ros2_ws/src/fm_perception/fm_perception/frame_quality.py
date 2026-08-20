#!/usr/bin/env python3
"""
Purpose: Decide whether a camera frame carries enough information to be worth
         classifying at all, so dinov2_terrain_node cannot report a confident
         terrain class for an image that shows nothing.
         Why this exists: measured on real hardware 2026-07-28, the deployed
         INT8 encoder and 1000-shot probe return soil at 0.913 confidence for a
         pure black frame, 0.699 for white, 0.778 for mid grey and 0.510 for
         flat red. All four are above the node's 0.40 uncertain threshold, so
         the uncertain-then-STOP mechanism the thesis credits with 5/5 Gazebo
         safety fires for none of them. Two real captures that came out black
         (ExposureValue -8, Brightness -1) scored soil 0.926 and 0.913. On the
         rover this means a disconnected camera, a covered lens, or a frame
         blown out in sunlight all report ground that is safe to drive on, and
         since the 2026-07-27 redesign made DINOv2 the primary heading-decider,
         nothing downstream would stop it.
         Raising the confidence threshold cannot fix this. The model is not
         hesitant about these images, it is confidently wrong, so the frame has
         to be rejected before it reaches the classifier.
Inputs:  An HxWx3 or HxW uint8 image array.
Outputs: A detail score in absolute intensity levels, and a pass/fail.
How to run:
    cd ros2_ws && colcon build --packages-select fm_perception
    source /opt/ros/humble/setup.bash && source install/setup.bash
    python3 -m pytest src/fm_perception/test/test_frame_quality.py -v
Project: Onboard Visual Foundation Models for Mars Terrain Perception and Rover Navigation
"""
import numpy as np

# Standard deviation in absolute intensity levels, measured over the real
# captures of 2026-07-28 (640x480, IMX219):
#
#   Brightness:=-1.0, fully black                0.00   <- must be rejected
#   ExposureValue:=-8.0, effectively black       0.26   <- must be rejected
#   ExposureValue:=-4.0                          5.11
#   auto-exposure sand (worst usable frame)      5.88
#   lab floor                                    6.89
#   AI4Mars training images                     22.38
#   AeEnable:=false + AnalogueGain:=1.0         40.53
#
# 2.0 sits 7.7x above the worst frame that must be rejected and 2.6x below the
# worst frame that must be accepted.
#
# Absolute spread, deliberately, not the std/mean ratio this module first used.
# Scale invariance sounds like the right instinct and is wrong for this job: a
# near-black frame has a mean close to zero, so dividing by it inflates the
# score instead of shrinking it. The ExposureValue -8 capture, the very frame
# that scored soil 0.913, came out with the *highest* ratio of any image in the
# set at 1.38. What is actually missing from a broken frame is absolute
# variation in brightness, so that is what gets measured.
DEFAULT_MIN_DETAIL = 2.0


def frame_detail(image: np.ndarray) -> float:
    """Standard deviation of intensity, in absolute 0-255 levels."""
    arr = np.asarray(image, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr.mean(axis=2)
    return float(arr.std())


def frame_is_informative(image: np.ndarray,
                         min_detail: float = DEFAULT_MIN_DETAIL) -> bool:
    """True when the frame has enough structure to be worth classifying."""
    return frame_detail(image) >= min_detail
