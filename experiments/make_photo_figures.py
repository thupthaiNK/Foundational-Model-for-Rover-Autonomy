"""
Purpose: Build the thesis's two photographic plates from the raw HEIC captures:
         a four-view plate of the assembled ExoMy rover (Figure 3.1) and a
         three-panel plate of the laboratory sandpit arena (Figure 4.9).
         Kept as a script so the plates are reproducible rather than pasted in
         by hand, in the same spirit as the MATLAB figure scripts.
Inputs:  HEIC photographs taken 2026-08-03/04, path set by SRC below.
Outputs: experiments/results/figures/thesis/exomy_platform.png
         experiments/results/figures/thesis/sandpit_arena.png
How to run: python3 experiments/make_photo_figures.py
Project: Onboard Visual Foundation Models for Mars Terrain Perception and Rover Navigation
"""
import os
from PIL import Image, ImageDraw, ImageFont
import pillow_heif

pillow_heif.register_heif_opener()

SRC = "/mnt/c/Users/DELL/Downloads/Picture"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "results", "figures", "thesis")


def _font(size):
    path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    return ImageFont.truetype(path, size) if os.path.exists(path) else ImageFont.load_default()


def _label(draw, x, y, tag, box, font):
    """Panel letter on a white chip, so it reads over any background."""
    draw.rectangle([x + 8, y + 8, x + 8 + box, y + 8 + box],
                   fill="white", outline="black", width=3)
    draw.text((x + 8 + box * 0.32, y + 8 + box * 0.18), tag, fill="black", font=font)


def plate(picks, cols, cell, out_name, crop_top=0.0, crop_bottom=1.0, pad=12):
    tiles = []
    for tag, name in picks:
        im = Image.open(f"{SRC}/{name}.HEIC").convert("RGB")
        if crop_top or crop_bottom != 1.0:
            w, h = im.size
            im = im.crop((0, int(h * crop_top), w, int(h * crop_bottom)))
        im.thumbnail((cell, cell))
        tiles.append((tag, im))
    rows = (len(tiles) + cols - 1) // cols
    tw = max(i.width for _, i in tiles)
    th = max(i.height for _, i in tiles)
    sheet = Image.new("RGB", (cols * tw + (cols + 1) * pad,
                              rows * th + (rows + 1) * pad), "white")
    draw = ImageDraw.Draw(sheet)
    font = _font(max(22, cell // 22))
    for k, (tag, im) in enumerate(tiles):
        x = pad + (k % cols) * (tw + pad) + (tw - im.width) // 2
        y = pad + (k // cols) * (th + pad) + (th - im.height) // 2
        sheet.paste(im, (x, y))
        _label(draw, x, y, tag, max(40, cell // 18), font)
    os.makedirs(OUT, exist_ok=True)
    sheet.save(os.path.join(OUT, out_name), quality=92)
    print(out_name, sheet.size)


if __name__ == "__main__":
    # The front view earns its panel: this thesis is about what that camera sees.
    plate([("a", "IMG_1557"), ("b", "IMG_1556"),
           ("c", "IMG_1550"), ("d", "IMG_1551")], 2, 900, "exomy_platform.png")
    # The top fifth of each arena frame is dark ceiling carrying no information,
    # and cropping it stops the sandpit shrinking when three panels share a line.
    plate([("a", "IMG_1564"), ("b", "IMG_1568"), ("c", "IMG_1572")], 3, 900,
          "sandpit_arena.png", crop_top=0.20, crop_bottom=0.97)
