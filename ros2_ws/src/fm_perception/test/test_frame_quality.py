"""
Purpose: Tests for fm_perception.frame_quality, the gate that stops
         dinov2_terrain_node reporting a confident terrain class for a frame
         that carries no information.
         Why it exists: measured on real hardware 2026-07-28, the deployed
         INT8 encoder plus 1000-shot probe returns soil at 0.913 confidence
         for a pure black image, 0.699 for white, 0.778 for mid grey and
         0.510 for flat red. Every one of those is above the node's 0.40
         uncertain threshold, so the uncertain-then-STOP mechanism the thesis
         credits with 5/5 Gazebo safety fires for none of them. Two real
         captures that came out black scored soil 0.926 and 0.913. On the
         rover that means a disconnected camera, a covered lens or a frame
         blown out in sunlight all report ground safe to drive on, and since
         the 2026-07-27 redesign made DINOv2 the primary heading-decider,
         nothing else would stop it. Raising the confidence threshold cannot
         help, because the model is genuinely confident; the frame has to be
         rejected before it is classified.
         The tests below pin measured values from the real captures rather
         than synthetic ideals. An earlier version of this file tested only
         perfectly flat synthetic images, passed, and shipped a metric that
         scored the real near-black capture higher than every good frame.
Inputs:  None (synthetic arrays, parametrised with measured statistics).
Outputs: pytest results.
How to run:
    cd ros2_ws && colcon build --packages-select fm_perception
    source /opt/ros/humble/setup.bash && source install/setup.bash
    python3 -m pytest src/fm_perception/test/test_frame_quality.py -v
Project: Onboard Visual Foundation Models for Mars Terrain Perception and Rover Navigation
"""
import numpy as np
import pytest

from fm_perception.frame_quality import (
    DEFAULT_MIN_DETAIL, frame_detail, frame_is_informative,
)


def _flat(value, shape=(480, 640, 3)):
    return np.full(shape, value, dtype=np.uint8)


def _noisy(mean, std, shape=(480, 640, 3), seed=0):
    rng = np.random.default_rng(seed)
    return np.clip(rng.normal(mean, std, shape), 0, 255).astype(np.uint8)


# ── The images that fooled the classifier ────────────────────────────────────

@pytest.mark.parametrize("value,name", [(0, "black"), (255, "white"), (128, "grey")])
def test_flat_frames_are_not_informative(value, name):
    assert not frame_is_informative(_flat(value)), f"flat {name} accepted"


def test_flat_colour_frame_is_not_informative():
    """Flat red scored soil 0.510 -- coloured, but no structure."""
    arr = np.zeros((480, 640, 3), dtype=np.uint8)
    arr[..., 0] = 255
    assert not frame_is_informative(arr)


def test_real_near_black_capture_is_rejected():
    """The ExposureValue:=-8.0 capture: mean 0.2, std 0.26, scored soil 0.926.

    This is the case that a std/mean ratio got backwards. Its mean is so close
    to zero that the ratio came out at 1.38, higher than any good frame in the
    set, so the ratio-based gate would have waved through the exact image it
    existed to catch.
    """
    assert not frame_is_informative(_noisy(0.2, 0.26))


def test_real_black_capture_is_rejected():
    """The Brightness:=-1.0 capture: mean 0.0, std 0.00, scored soil 0.913."""
    assert not frame_is_informative(_noisy(0.0, 0.0))


# ── Real frames must keep working ────────────────────────────────────────────

@pytest.mark.parametrize("mean,std,name", [
    (29.6, 5.11, "ExposureValue -4"),
    (204.9, 5.88, "auto-exposure sand, the worst usable frame recorded"),
    (203.7, 6.89, "lab floor"),
    (66.5, 22.38, "AI4Mars training image"),
    (163.2, 40.53, "AeEnable false + AnalogueGain 1.0, the best setting found"),
])
def test_real_frames_are_informative(mean, std, name):
    assert frame_is_informative(_noisy(mean, std)), f"rejected {name}"


def test_threshold_sits_between_the_two_populations():
    """7.7x above the worst frame that must be rejected, 2.6x below the worst
    that must be accepted. Stated as a test so narrowing the gap fails loudly."""
    assert 0.26 < DEFAULT_MIN_DETAIL < 5.11


def test_detail_is_measured_in_absolute_levels_not_a_ratio():
    """Halving brightness halves absolute spread, and that is intended: a frame
    with almost no absolute variation is unreadable however its variation
    compares to its own mean."""
    bright = _noisy(200, 40, seed=2)
    dim = (bright.astype(np.float32) / 2).astype(np.uint8)
    assert frame_detail(dim) == pytest.approx(frame_detail(bright) / 2, rel=0.05)


def test_threshold_is_configurable():
    arr = _noisy(120, 30, seed=3)
    assert frame_is_informative(arr, min_detail=1.0)
    assert not frame_is_informative(arr, min_detail=500.0)


def test_gate_can_be_disabled_with_zero():
    assert frame_is_informative(_flat(0), min_detail=0.0)


def test_grayscale_2d_input_supported():
    """The node may hand over a single-channel array; it must not raise."""
    assert frame_is_informative(_noisy(120, 30, shape=(480, 640), seed=4))
    assert not frame_is_informative(np.full((480, 640), 60, dtype=np.uint8))
