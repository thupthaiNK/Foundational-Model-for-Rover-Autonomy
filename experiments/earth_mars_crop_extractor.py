"""
Purpose: Symmetric clean-crop extraction for the Earth+Mars cross-planet
         generalization probe. Turns AI4Mars (integer masks) and RUGD (colour
         masks) into single-class 224x224 terrain crops under one shared
         physical-surface taxonomy, so a frozen DINOv2+reg ViT-S linear probe
         can be compared across planets fairly. See
         docs/earth_mars_probe_preregistration.md for the fixed design.
Inputs:  AI4Mars msl labels/images; RUGD frames + colour annotations + colormap.
Outputs: Lists of clean crop boxes / cropped RGB tiles per shared class.
How to run:
    python3 -m pytest experiments/test_earth_mars_crop_extractor.py
    (extraction is driven by earth_mars_probe.py)
Project: Onboard Visual Foundation Models for Mars Terrain Perception and Rover Navigation
"""

import numpy as np

# ── Shared physical-surface taxonomy (FIXED in pre-registration) ──────────────
SHARED_CLASSES = ["smooth_fine", "granular", "rocky_solid"]
OTHER = 255  # discarded / non-comparable / ignore

# AI4Mars pixel id -> shared id.  soil=0, bedrock=1, sand=2, big_rock=3, 255=ignore.
AI4MARS_TO_SHARED = {0: 0, 2: 1, 1: 2}  # big_rock(3) and 255 fall through to OTHER

# RUGD class NAME -> shared id.  Mapped by physical surface, names discarded by omission.
RUGD_NAME_TO_SHARED = {
    "dirt": 0, "mulch": 0,          # smooth_fine
    "sand": 1,                       # granular
    "gravel": 2, "rock-bed": 2,      # rocky_solid
}


def remap_labels(raw: np.ndarray, mapping: dict) -> np.ndarray:
    """Remap a raw integer label array into shared-class ids (unmapped -> OTHER)."""
    out = np.full(raw.shape, OTHER, dtype=np.uint8)
    for src, dst in mapping.items():
        out[raw == src] = dst
    return out


def crop_purity(shared_crop: np.ndarray, target: int) -> float:
    """Fraction of ALL crop pixels equal to target shared class (OTHER counts against)."""
    if shared_crop.size == 0:
        return 0.0
    return float(np.count_nonzero(shared_crop == target)) / shared_crop.size


def sample_clean_crops(shared: np.ndarray, target: int, crop_size: int,
                       purity_threshold: float, max_crops: int,
                       rng, max_attempts: int) -> list:
    """Random square crop boxes whose purity for `target` >= threshold.

    Returns list of (top, left). Deterministic given rng (random.Random).
    """
    h, w = shared.shape
    boxes = []
    if h < crop_size or w < crop_size:
        return boxes
    attempts = 0
    while len(boxes) < max_crops and attempts < max_attempts:
        attempts += 1
        top = rng.randint(0, h - crop_size)
        left = rng.randint(0, w - crop_size)
        crop = shared[top:top + crop_size, left:left + crop_size]
        if crop_purity(crop, target) >= purity_threshold:
            boxes.append((top, left))
    return boxes


def parse_rugd_colormap(text: str) -> dict:
    """Parse RUGD colormap lines 'idx name R G B' -> {name: (R,G,B)}."""
    name_to_color = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            r, g, b = int(parts[-3]), int(parts[-2]), int(parts[-1])
        except ValueError:
            continue
        name = parts[1]
        name_to_color[name] = (r, g, b)
    return name_to_color


def rugd_color_to_shared(rgb: np.ndarray, name_to_color: dict,
                         name_to_shared: dict = RUGD_NAME_TO_SHARED) -> np.ndarray:
    """Convert an HxWx3 RUGD colour annotation to a shared-class id array."""
    out = np.full(rgb.shape[:2], OTHER, dtype=np.uint8)
    for name, shared in name_to_shared.items():
        color = name_to_color.get(name)
        if color is None:
            continue
        mask = np.all(rgb == np.array(color, dtype=rgb.dtype), axis=-1)
        out[mask] = shared
    return out
