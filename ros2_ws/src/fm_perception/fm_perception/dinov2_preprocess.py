#!/usr/bin/env python3
"""
Purpose: Torch-free, transformers-free image preprocessing for DINOv2
         (facebook/dinov2-with-registers-small), byte-identical to what
         transformers' AutoImageProcessor produces for the same image.
         Why this exists: the Pi's ROS2 container has no torch, and
         `transformers` hard-imports torch even to resolve
         AutoImageProcessor (traced 2026-07-27:
         transformers/generation/logits_process.py has a top-level
         `import torch`). That one dependency crashed dinov2_terrain_node on
         every real-hardware launch, which silently took the thesis's entire
         foundation-model layer offline -- the rover was navigating on LiDAR
         geometry alone. The INT8 ONNX inference path is otherwise already
         torch-free (numpy in, onnxruntime, sklearn), so replacing the
         processor is all that was needed to put DINOv2 back in the loop on
         real hardware.
         The constants below are not guesses: they are the preprocessor_config
         values published with the model, and
         test/test_dinov2_preprocess.py asserts this function agrees with
         AutoImageProcessor to 1e-4 on six real frame shapes. Do not tune
         them -- the LogReg probe was fitted on features extracted through
         the HuggingFace pipeline, so any drift here silently degrades
         classification accuracy instead of raising.
Inputs:  A PIL image (any mode/size).
Outputs: np.ndarray [1, 3, 224, 224] float32, NCHW, ImageNet-normalised.
How to run:
    cd ros2_ws && colcon build --packages-select fm_perception
    source /opt/ros/humble/setup.bash && source install/setup.bash
    python3 -m pytest src/fm_perception/test/test_dinov2_preprocess.py -v
Project: Onboard Visual Foundation Models for Mars Terrain Perception and Rover Navigation
"""
import numpy as np
from PIL import Image as PILImage

# facebook/dinov2-with-registers-small preprocessor_config.json
SHORTEST_EDGE = 256
CROP_SIZE = 224
RESCALE_FACTOR = 1.0 / 255.0
IMAGE_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGE_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
# resample=3 in the config is PIL's BICUBIC filter.
RESAMPLE = PILImage.BICUBIC


def resize_shortest_edge(width: int, height: int,
                         shortest_edge: int = SHORTEST_EDGE) -> tuple:
    """(new_width, new_height) scaling the shorter side to shortest_edge and
    preserving aspect ratio -- HuggingFace's get_resize_output_image_size for
    a `{"shortest_edge": N}` size dict."""
    short, long_ = (width, height) if width <= height else (height, width)
    new_long = int(shortest_edge * long_ / short)
    return (shortest_edge, new_long) if width <= height else (new_long, shortest_edge)


def center_crop_box(width: int, height: int, crop: int = CROP_SIZE) -> tuple:
    """(left, top, right, bottom) for a centred crop x crop box.

    Floor division, matching transformers.image_transforms.center_crop's
    `(orig - crop) // 2`. NOT round(): on an odd margin Python's round()
    does banker's rounding, so a 455px-wide resize (a 1280x720 frame) gave
    left=116 where HuggingFace gives 115 -- a one-pixel shift of the whole
    crop, which the shape-parametrised equivalence test caught as a 2.23
    max difference rather than the ~0.0175 seen elsewhere.
    """
    left = (width - crop) // 2
    top = (height - crop) // 2
    return left, top, left + crop, top + crop


def preprocess_dinov2(pil_img) -> np.ndarray:
    """PIL image -> [1, 3, 224, 224] float32 NCHW, exactly as
    AutoImageProcessor.from_pretrained(DINOV2_MODEL_ID) would produce it.

    Pipeline (all four steps are enabled in the model's own config):
    convert to RGB -> bicubic resize shortest edge to 256 -> centre crop 224
    -> rescale by 1/255 -> normalise by ImageNet mean/std -> HWC to CHW.
    """
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")

    new_w, new_h = resize_shortest_edge(pil_img.width, pil_img.height)
    if (new_w, new_h) != (pil_img.width, pil_img.height):
        pil_img = pil_img.resize((new_w, new_h), resample=RESAMPLE)

    pil_img = pil_img.crop(center_crop_box(pil_img.width, pil_img.height))

    arr = np.asarray(pil_img, dtype=np.float32) * RESCALE_FACTOR   # HWC, 0..1
    arr = (arr - IMAGE_MEAN) / IMAGE_STD
    return np.ascontiguousarray(arr.transpose(2, 0, 1)[None, ...], dtype=np.float32)
