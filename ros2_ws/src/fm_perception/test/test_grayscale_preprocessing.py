"""
Purpose: Unit tests for to_grayscale_3ch() -- the pure-Python RGB->grayscale
         preprocessing function added to dinov2_terrain_node.py as an opt-in
         ablation (wait-time plan item 4) targeting the Exp 5b root cause
         (NAVCAM grayscale training vs RPi RGB inference, luminance mismatch).
Inputs:  None.
Outputs: pytest results.
How to run:
    cd ros2_ws && colcon build --packages-select fm_perception
    source /opt/ros/humble/setup.bash && source install/setup.bash
    python3 -m pytest src/fm_perception/test/test_grayscale_preprocessing.py -v
Project: Onboard Visual Foundation Models for Mars Terrain Perception and Rover Navigation
"""
import numpy as np

from fm_perception.dinov2_terrain_node import to_grayscale_3ch


def _solid(rgb, h=4, w=4):
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    arr[:, :] = rgb
    return arr


def test_output_shape_matches_input():
    arr = _solid((10, 20, 30))
    out = to_grayscale_3ch(arr)
    assert out.shape == arr.shape


def test_output_dtype_matches_input():
    arr = _solid((10, 20, 30))
    out = to_grayscale_3ch(arr)
    assert out.dtype == arr.dtype


def test_channels_are_equal_everywhere():
    arr = _solid((200, 50, 10))
    out = to_grayscale_3ch(arr)
    assert np.all(out[..., 0] == out[..., 1])
    assert np.all(out[..., 1] == out[..., 2])


def test_white_stays_white():
    arr = _solid((255, 255, 255))
    out = to_grayscale_3ch(arr)
    assert np.all(out == 255)


def test_black_stays_black():
    arr = _solid((0, 0, 0))
    out = to_grayscale_3ch(arr)
    assert np.all(out == 0)


def test_uses_itu_r_bt601_luma_weights():
    # Pure red at (255,0,0) -> 0.299*255 ~= 76 (matches PIL's Image.convert("L"))
    arr = _solid((255, 0, 0))
    out = to_grayscale_3ch(arr)
    assert abs(int(out[0, 0, 0]) - 76) <= 1


def test_gray_input_is_a_near_identity():
    arr = _solid((100, 100, 100))
    out = to_grayscale_3ch(arr)
    assert abs(int(out[0, 0, 0]) - 100) <= 1
