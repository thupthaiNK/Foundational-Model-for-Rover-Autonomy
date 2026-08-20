"""
Purpose: Unit tests for the Earth+Mars clean-crop extractor (pure logic).
How to run: python3 -m pytest experiments/test_earth_mars_crop_extractor.py -q
Project: Onboard Visual Foundation Models for Mars Terrain Perception and Rover Navigation
"""

import random

import numpy as np

from earth_mars_crop_extractor import (
    AI4MARS_TO_SHARED, OTHER, crop_purity, parse_rugd_colormap, remap_labels,
    rugd_color_to_shared, sample_clean_crops,
)
from earth_mars_probe import harvest


def test_harvest_lazily_loads_and_releases_labels():
    """harvest() must call the lazy loader per-image, not require pre-materialised arrays."""
    calls = []

    def make_loader(value):
        def loader():
            calls.append(1)
            return np.full((20, 20), value, dtype=np.uint8)
        return loader

    pairs = [("imgA", make_loader(0)), ("imgB", make_loader(1)), ("imgC", make_loader(2))]
    rng = random.Random(0)
    crops = harvest(pairs, n_target=2, rng=rng, desc="test")
    assert len(calls) == 3  # each image's loader invoked exactly once
    assert len(crops) > 0


def test_remap_ai4mars_ids():
    raw = np.array([[0, 1, 2], [3, 255, 0]], dtype=np.uint8)
    out = remap_labels(raw, AI4MARS_TO_SHARED)
    # soil0->0, bedrock1->2, sand2->1, big_rock3->OTHER, 255->OTHER, soil0->0
    assert out.tolist() == [[0, 2, 1], [OTHER, OTHER, 0]]


def test_crop_purity_counts_other_against():
    crop = np.array([[0, 0], [0, OTHER]], dtype=np.uint8)
    assert crop_purity(crop, 0) == 0.75
    assert crop_purity(crop, 1) == 0.0


def test_sample_clean_crops_finds_pure_region():
    shared = np.full((20, 20), OTHER, dtype=np.uint8)
    shared[:10, :10] = 0  # a pure block of class 0
    rng = random.Random(0)
    boxes = sample_clean_crops(shared, target=0, crop_size=5,
                               purity_threshold=0.9, max_crops=5,
                               rng=rng, max_attempts=2000)
    assert len(boxes) == 5
    for top, left in boxes:
        assert crop_purity(shared[top:top + 5, left:left + 5], 0) >= 0.9


def test_sample_clean_crops_rejects_when_no_pure_region():
    # checkerboard: no 5x5 region is >=90% one class
    shared = np.indices((20, 20)).sum(axis=0) % 2
    shared = shared.astype(np.uint8)
    rng = random.Random(1)
    boxes = sample_clean_crops(shared, target=0, crop_size=5,
                               purity_threshold=0.9, max_crops=5,
                               rng=rng, max_attempts=500)
    assert boxes == []


def test_sample_clean_crops_deterministic():
    shared = np.full((30, 30), 0, dtype=np.uint8)
    b1 = sample_clean_crops(shared, 0, 5, 0.9, 3, random.Random(42), 100)
    b2 = sample_clean_crops(shared, 0, 5, 0.9, 3, random.Random(42), 100)
    assert b1 == b2


def test_crop_size_larger_than_image_returns_empty():
    shared = np.zeros((4, 4), dtype=np.uint8)
    assert sample_clean_crops(shared, 0, 8, 0.9, 5, random.Random(0), 100) == []


def test_parse_rugd_colormap():
    text = "0 void 0 0 0\n1 dirt 108 64 20\n2 sand 255 229 204\n"
    cm = parse_rugd_colormap(text)
    assert cm["dirt"] == (108, 64, 20)
    assert cm["sand"] == (255, 229, 204)


def test_rugd_color_to_shared():
    name_to_color = {"dirt": (108, 64, 20), "sand": (255, 229, 204),
                     "sky": (0, 0, 255)}
    rgb = np.array([
        [[108, 64, 20], [255, 229, 204]],
        [[0, 0, 255], [1, 2, 3]],
    ], dtype=np.uint8)
    out = rugd_color_to_shared(rgb, name_to_color)
    # dirt->0(smooth), sand->1(granular), sky->OTHER, unknown->OTHER
    assert out.tolist() == [[0, 1], [OTHER, OTHER]]
