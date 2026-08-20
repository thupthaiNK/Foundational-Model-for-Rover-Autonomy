"""
Purpose: Equivalence tests for preprocess_dinov2(), the torch-free
         replacement for transformers' AutoImageProcessor used by
         dinov2_terrain_node.py's INT8 ONNX backend.
         Why it exists: the Pi's ROS2 container has no torch, and
         `transformers` hard-imports torch even for AutoImageProcessor
         (traced 2026-07-27: transformers/generation/logits_process.py does
         a top-level `import torch`). That single dependency was enough to
         kill dinov2_terrain_node on every real-hardware launch, taking the
         thesis's entire foundation-model layer offline with it. These tests
         pin the hand-rolled preprocessing to the HuggingFace one on a
         machine where transformers DOES work, so the Pi can run the
         hand-rolled path with the same numbers -- and therefore the same
         probe accuracy the feature cache was built with.
Inputs:  None (synthetic images).
Outputs: pytest results.
How to run:
    cd ros2_ws && colcon build --packages-select fm_perception
    source /opt/ros/humble/setup.bash && source install/setup.bash
    python3 -m pytest src/fm_perception/test/test_dinov2_preprocess.py -v
Project: Onboard Visual Foundation Models for Mars Terrain Perception and Rover Navigation
"""
import numpy as np
import pytest
from PIL import Image as PILImage

from fm_perception.dinov2_preprocess import (
    IMAGE_MEAN, IMAGE_STD, RESAMPLE, center_crop_box, preprocess_dinov2,
    resize_shortest_edge,
)

DINOV2_MODEL_ID = "facebook/dinov2-with-registers-small"


def _hf_reference(pil_img):
    """HuggingFace AutoImageProcessor output for the same image, as [1,3,224,224]
    float32. Skips the test where transformers/torch are unavailable (i.e. on
    the Pi itself, which is exactly the environment this function exists to
    serve)."""
    transformers = pytest.importorskip("transformers")
    processor = transformers.AutoImageProcessor.from_pretrained(DINOV2_MODEL_ID)
    return processor(images=pil_img, return_tensors="np")["pixel_values"].astype(np.float32)


def _img(w, h, seed=0):
    rng = np.random.RandomState(seed)
    return PILImage.fromarray(rng.randint(0, 256, (h, w, 3), dtype=np.uint8))


@pytest.mark.parametrize("w,h", [
    (640, 480),    # the real ExoMy camera frame
    (480, 640),    # portrait, so the shortest edge is the width instead
    (256, 256),    # already square at the resize target
    (224, 224),    # already square at the crop target
    (1280, 720),   # larger than either target
    (300, 200),    # small and non-square
])
def test_matches_huggingface_processor_on_real_frame_shapes(w, h):
    img = _img(w, h, seed=w + h)
    expected = _hf_reference(img)
    actual = preprocess_dinov2(img)
    assert actual.shape == expected.shape == (1, 3, 224, 224)
    assert actual.dtype == np.float32
    # Resize output size and crop box are identical (verified separately);
    # what remains is a sub-LSB difference in HuggingFace's bicubic path,
    # which round-trips the image through numpy. 1/255 in 0..1 space becomes
    # 1/255/0.224 ~= 0.0175 after ImageNet normalisation, so a 1-LSB
    # tolerance is 0.02. Measured on these six shapes: max 0.0175, mean
    # 0.00013, 0.8% of pixels above 1e-3. The test that actually licenses
    # this substitution is the feature-level one below -- pixels are only a
    # smoke check.
    max_diff = np.abs(actual - expected).max()
    assert max_diff <= 0.02, f"max abs diff = {max_diff} (over 1 uint8 LSB)"


def test_each_stage_matches_huggingfaces_own_primitives_exactly():
    """Stage-by-stage equality against transformers' OWN resize/center_crop/
    rescale/normalize functions. This is the real correctness statement: the
    pipeline is byte-identical at every documented step.

    The residual ~1-LSB difference measured against the assembled
    `processor(...)` call is introduced inside HuggingFace's own wrapper (it
    reproduces against transformers' primitives too, i.e. the wrapper does
    not equal the sum of its documented parts on this version), not by this
    reimplementation. See test_measured_feature_delta_is_recorded below for
    the honest end-to-end number and the outstanding validation.
    """
    it = pytest.importorskip("transformers.image_transforms")
    img = _img(640, 480, seed=7)
    arr = np.asarray(img)

    hf_resized = it.resize(arr, size=(256, 341), resample=RESAMPLE,
                            input_data_format="channels_last")
    new_w, new_h = resize_shortest_edge(img.width, img.height)
    my_resized = np.asarray(img.resize((new_w, new_h), resample=RESAMPLE))
    assert np.array_equal(hf_resized, my_resized)

    hf_cropped = it.center_crop(hf_resized, size=(224, 224),
                                 input_data_format="channels_last")
    left, top, right, bottom = center_crop_box(new_w, new_h)
    my_cropped = my_resized[top:bottom, left:right]
    assert np.array_equal(hf_cropped, my_cropped)

    hf_norm = it.normalize(
        it.rescale(hf_cropped, scale=1 / 255.0, input_data_format="channels_last"),
        mean=list(IMAGE_MEAN), std=list(IMAGE_STD),
        input_data_format="channels_last")
    my_norm = (my_cropped.astype(np.float32) / 255.0 - IMAGE_MEAN) / IMAGE_STD
    assert np.allclose(hf_norm, my_norm, atol=1e-6)


def test_measured_feature_delta_is_recorded():
    """Pins the END-TO-END delta through the real INT8 ONNX encoder so a
    regression cannot widen it unnoticed.

    Measured 2026-07-27 on this machine: cosine similarity 0.984-0.992
    between features from HuggingFace's processor and this one, across white
    noise, smoothed noise and a synthetic regolith frame. That is NOT a proof
    of equivalence and is deliberately not presented as one.

    Context for why it is nonetheless acceptable to deploy: the INT8
    quantisation already in this path perturbs features far more than a
    1-LSB input change does, and the thesis measured that cost directly at
    -0.50 pp (B1: FP32 94.00% -> INT8 93.50% on a 200-image sample).

    OUTSTANDING (do before quoting any accuracy figure from the Pi): re-run
    the 287-image gold-standard evaluation through
    preprocess_dinov2 + the INT8 ONNX encoder + the cached probe and compare
    against the published 90.24%. The AI4Mars test images are not on this
    machine, so it could not be done here.
    """
    import os
    ort = pytest.importorskip("onnxruntime")
    repo_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
    )
    onnx_path = os.path.join(
        repo_root, "experiments", "results", "dinov2_reg_small_encoder_int8.onnx"
    )
    if not os.path.exists(onnx_path):
        pytest.skip(f"INT8 ONNX encoder not present at {onnx_path}")

    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    in_name = session.get_inputs()[0].name

    for (w, h) in [(640, 480), (480, 640), (300, 200)]:
        img = _img(w, h, seed=w * 7 + h)
        feat_hf = session.run(None, {in_name: _hf_reference(img)})[0].ravel()
        feat_mine = session.run(None, {in_name: preprocess_dinov2(img)})[0].ravel()
        cos = float(
            np.dot(feat_hf, feat_mine)
            / (np.linalg.norm(feat_hf) * np.linalg.norm(feat_mine))
        )
        assert cos > 0.98, f"{w}x{h}: cosine {cos:.6f} below the measured floor"


def test_grayscale_input_is_converted_to_three_channels():
    # AI4Mars is NAVCAM grayscale and the obstacle-gate ablation can feed
    # single-channel frames, so a mode-L image must not crash or produce a
    # [1,1,224,224] tensor the ONNX graph would reject.
    gray = PILImage.fromarray(np.full((480, 640), 128, dtype=np.uint8))
    out = preprocess_dinov2(gray)
    assert out.shape == (1, 3, 224, 224)
    # Each channel is uniform (the source was uniform), but the three channels
    # differ from each other because ImageNet mean/std are per-channel -- so
    # compare each channel against its own expected constant, not against the
    # other channels.
    for c in range(3):
        expected = (128 / 255.0 - IMAGE_MEAN[c]) / IMAGE_STD[c]
        assert np.allclose(out[0, c], expected, atol=1e-5)


def test_output_is_normalised_not_raw_pixels():
    # A mid-grey image must land near -0.x .. +0.x after ImageNet
    # normalisation, never in 0..255 -- catches a dropped rescale/normalise
    # step, which would silently wreck probe accuracy rather than error.
    grey = PILImage.fromarray(np.full((480, 640, 3), 128, dtype=np.uint8))
    out = preprocess_dinov2(grey)
    assert -3.0 < float(out.min()) and float(out.max()) < 3.0


def test_does_not_require_torch_or_transformers():
    # The whole point: this module must import and run in an environment
    # where neither package exists.
    import sys
    import importlib
    blocked = {"torch", "transformers"}

    class _Blocker:
        def find_module(self, name, path=None):
            root = name.split(".")[0]
            return self if root in blocked else None

        def load_module(self, name):
            raise ImportError(f"No module named '{name}'")

    sys.meta_path.insert(0, _Blocker())
    try:
        for mod in list(sys.modules):
            if mod.split(".")[0] in blocked:
                del sys.modules[mod]
        del sys.modules["fm_perception.dinov2_preprocess"]
        mod = importlib.import_module("fm_perception.dinov2_preprocess")
        out = mod.preprocess_dinov2(_img(640, 480))
        assert out.shape == (1, 3, 224, 224)
    finally:
        sys.meta_path.remove(_Blocker) if False else sys.meta_path.pop(0)
