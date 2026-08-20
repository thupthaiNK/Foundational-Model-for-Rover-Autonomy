"""
Purpose: Rebuild the poster's rover cut-out at print resolution. The
         background-removed PNG supplied by the author is 282 x 376 px, which
         at its 90 mm placement on the A1 poster is 80 dpi and would print
         visibly soft; every other figure on the poster is above 460 dpi.
         The mask itself is good, so this reuses it: the alpha channel is
         upscaled and applied to the full-resolution source photograph, giving
         a sharp rover with the author's own silhouette.
Inputs:  /mnt/c/Users/DELL/Downloads/Picture/Picture1.png   (author's cut-out)
         /mnt/c/Users/DELL/Downloads/Picture/IMG_1557.HEIC  (its source frame,
             identified by matching RGB over the opaque pixels: mean absolute
             difference 2.4 against 31 or more for every other candidate)
Outputs: experiments/results/figures/thesis/exomy_cutout.png  (RGBA, ~2200 px
         wide, about 620 dpi at the poster's 90 mm placement)
How to run: python3 experiments/upscale_rover_cutout.py
Project: Onboard Visual Foundation Models for Mars Terrain Perception and Rover Navigation
"""
import os

import numpy as np
import pillow_heif
from PIL import Image, ImageFilter

pillow_heif.register_heif_opener()

CUT = "/mnt/c/Users/DELL/Downloads/Picture/Picture1.png"
SRC = "/mnt/c/Users/DELL/Downloads/Picture/IMG_1557.HEIC"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "results", "figures", "thesis", "exomy_cutout.png")

# Placement width on the poster, used only to report the resulting dpi.
PLACED_MM = 90.0


def main():
    cut = Image.open(CUT).convert("RGBA")
    src = Image.open(SRC).convert("RGB")
    if abs(cut.width / cut.height - src.width / src.height) > 0.01:
        raise SystemExit("cut-out and source frame have different aspect "
                         "ratios; the mask is not a whole-frame mask")

    # Upscale the mask, then blur it by about one source pixel per mask pixel.
    # Nearest-neighbour edges from a 282 px mask would show as visible stair
    # steps at this magnification; a small blur turns them into an anti-aliased
    # edge, which is what the eye expects at a cut-out boundary anyway.
    alpha = cut.split()[-1].resize(src.size, Image.LANCZOS)
    alpha = alpha.filter(ImageFilter.GaussianBlur(radius=src.width / cut.width))

    rgba = Image.merge("RGBA", (*src.split(), alpha))

    a = np.array(alpha)
    ys, xs = np.where(a > 8)
    pad = 24
    box = (max(0, xs.min() - pad), max(0, ys.min() - pad),
           min(src.width, xs.max() + pad), min(src.height, ys.max() + pad))
    rgba = rgba.crop(box)

    # No need to ship 3000 px for a 90 mm placement; 2200 px is about 620 dpi
    # and keeps the poster file a sensible size.
    if rgba.width > 2200:
        rgba = rgba.resize((2200, round(2200 * rgba.height / rgba.width)),
                           Image.LANCZOS)

    rgba.save(OUT)
    dpi = rgba.width / (PLACED_MM / 25.4)
    print(f"wrote {OUT}")
    print(f"  {rgba.size[0]} x {rgba.size[1]} px, {dpi:.0f} dpi at {PLACED_MM:.0f} mm "
          f"(was 80 dpi)")


if __name__ == "__main__":
    main()
