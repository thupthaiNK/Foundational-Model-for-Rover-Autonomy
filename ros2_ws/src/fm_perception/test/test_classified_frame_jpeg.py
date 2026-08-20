"""
Purpose: Pin the JPEG encoding behind /terrain_classified_image, the topic the
         rosbag records in place of the raw camera stream. Recording
         /camera/image_raw produced an 11 GB bag for a 573 s run on 2026-07-29,
         almost all of it frames of a rover that never moved, while what the
         thesis actually needs is the frame behind each DINOv2 verdict.
Inputs:  None; encodes synthetic arrays.
Outputs: pytest results.
How to run:
    cd ros2_ws && python3 -m pytest src/fm_perception/test/test_classified_frame_jpeg.py -v
Project: Onboard Visual Foundation Models for Mars Terrain Perception and Rover Navigation
"""
import io
import os
import sys

import numpy as np
from PIL import Image as PILImage

sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "fm_perception")
)

from dinov2_terrain_node import encode_jpeg


def _frame(width=640, height=480):
    # Structured rather than random: random noise is close to incompressible
    # and would make the size assertions meaningless.
    xs = np.linspace(0, 255, width, dtype=np.uint8)
    row = np.stack([xs, xs[::-1], np.full_like(xs, 128)], axis=-1)
    return np.repeat(row[None, :, :], height, axis=0)


def test_encodes_a_decodable_jpeg_of_the_same_size():
    data = encode_jpeg(_frame())
    decoded = PILImage.open(io.BytesIO(bytes(data)))
    assert decoded.format == "JPEG"
    assert decoded.size == (640, 480)


def test_a_frame_costs_far_less_than_the_raw_bytes():
    frame = _frame()
    raw = frame.nbytes                      # 640*480*3 = 921,600
    encoded = len(encode_jpeg(frame))
    assert encoded < raw / 10, (
        f"{encoded} bytes is not a worthwhile saving against {raw} raw"
    )


def test_quality_is_honoured():
    frame = _frame()
    assert len(encode_jpeg(frame, quality=95)) > len(
        encode_jpeg(frame, quality=40)
    )


def test_greyscale_input_is_accepted():
    # grayscale_preprocessing is on for real hardware, so the array handed to
    # the encoder can already be a 3-channel replicated grey image.
    grey = np.repeat(_frame()[:, :, :1], 3, axis=2)
    decoded = PILImage.open(io.BytesIO(bytes(encode_jpeg(grey))))
    assert decoded.size == (640, 480)
