"""
Purpose: Produce the single-image variants the project website uses in place of
         the thesis's multi-panel plates. The plates in Figure 3.1, 4.9 and the
         Gazebo world/zone figures carry (a)(b)(c)(d) panel letters, and one of
         them even cites "Section 4.10" inside the image. Those exist to be
         walked through by surrounding report text, which a web page does not
         have, so the site shows one frame and says what it is in its caption.
         The thesis plates are NOT touched: make_photo_figures.py still owns
         them and they stay exactly as submitted.
Inputs:  HEIC photographs in SRC (same captures as make_photo_figures.py),
         docs/figures/gazebo_world_views/world_oblique.png,
         docs/figures/gazebo_demo_latest/rock_cluster_view.png
Outputs: experiments/results/figures/thesis/sandpit_arena_single.png
         experiments/results/figures/thesis/gazebo_world_oblique.png
         experiments/results/figures/thesis/gazebo_rock_view.png
         (exomy_hero.png already exists and needs no rebuild)
How to run: python3 experiments/make_website_photos.py
Project: Onboard Visual Foundation Models for Mars Terrain Perception and Rover Navigation
"""
import os
import shutil

from PIL import Image
import pillow_heif

pillow_heif.register_heif_opener()

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SRC = "/mnt/c/Users/DELL/Downloads/Picture"
OUT = os.path.join(HERE, "results", "figures", "thesis")
FIGURES = os.path.join(ROOT, "docs", "figures")

# The build downscales again, so 1600 px is headroom rather than a final size.
LONG_EDGE = 1600


def from_heic(name, out_name, crop_top=0.0, crop_bottom=1.0):
    im = Image.open(f"{SRC}/{name}.HEIC").convert("RGB")
    if crop_top or crop_bottom != 1.0:
        w, h = im.size
        im = im.crop((0, int(h * crop_top), w, int(h * crop_bottom)))
    im.thumbnail((LONG_EDGE, LONG_EDGE))
    im.save(os.path.join(OUT, out_name), quality=92)
    print(out_name, im.size)


def copy_render(rel_path, out_name):
    """Gazebo captures are already PNG at a sensible size; copy them verbatim."""
    src = os.path.join(FIGURES, rel_path)
    dst = os.path.join(OUT, out_name)
    shutil.copyfile(src, dst)
    print(out_name, Image.open(dst).size)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)

    # Panel (b) of the thesis plate: the arena with an obstacle layout in it.
    # The empty arena reads as an empty sandbox with nothing to explain.
    # Same crop as the plate: the top fifth is dark ceiling carrying nothing.
    from_heic("IMG_1568", "sandpit_arena_single.png",
              crop_top=0.20, crop_bottom=0.97)

    # One oblique of the whole benchmark world, rather than the four-panel plate
    # whose first panel is a texture key that needs its own explanation.
    copy_render(os.path.join("gazebo_world_views", "world_oblique.png"),
                "gazebo_world_oblique.png")

    # One camera frame, at the rock position. That position is the one whose
    # confidence falls under the threshold, so it is the frame the surrounding
    # text is actually about.
    copy_render(os.path.join("gazebo_demo_latest", "rock_cluster_view.png"),
                "gazebo_rock_view.png")
