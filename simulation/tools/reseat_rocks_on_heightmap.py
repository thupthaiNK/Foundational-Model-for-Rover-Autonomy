"""
Purpose: Re-seat every mars_rock mesh in the Gazebo world onto an undulating heightmap
         terrain so rocks neither float nor sink once flat ground is replaced by relief.
         For each rock it samples the heightmap PNG at the rock's (x,y), then sets
         pose_z = terrain_height - burial, keeping the same 15% partial burial used on flat ground.
Inputs:  --world  simulation/worlds/mars_terrain.world  (rocks = inline models using
                   model://mars_rock/meshes/model.dae, baked mesh height 1.5672 m, origin at base)
         --png    simulation/media/heightmaps/mars_relief.png  (square (2^n)+1 grayscale)
         --size-x --size-y --size-z   heightmap <size> in the world (default 20 20 0.4)
         --burial fraction of rock height buried (default 0.15)
         --apply  rewrite the world in place (default = dry-run, only prints the table)
Outputs: dry-run table of (name, x, y, scale, terrain_h, old_z -> new_z); with --apply, edits the world.
How to run:
    # after the <heightmap> ground is wired into the world:
    python3 simulation/tools/reseat_rocks_on_heightmap.py --apply
NOTE: Only models whose body references model://mars_rock/meshes/model.dae are touched (zones / edge
      patches are ignored). Verify the Gazebo heightmap convention once in-GUI (image row order / Z base);
      if rocks sit uniformly high/low, adjust --size-z or add a constant offset and re-run.
Project: Onboard Visual Foundation Models for Mars Terrain Perception and Rover Navigation
"""
import argparse, re, sys

MESH_H = 1.5672  # baked mars_rock mesh height (Z 0..1.5672, origin at base)
MODEL_RE = re.compile(r'<model name="([^"]+)">(.*?)</model>', re.S)
POSE_RE  = re.compile(r'<pose>([-0-9.]+) ([-0-9.]+) ([-0-9.]+) ([-0-9. ]+?)</pose>')
SCALE_RE = re.compile(r'<scale>([0-9.]+)')

def load_png_gray(path):
    try:
        from PIL import Image
        import numpy as np
        return np.asarray(Image.open(path).convert("L"), dtype=float)
    except Exception as e:
        sys.exit(f"cannot load {path}: {e}")

def sample(img, x, y, sx, sy):
    H, W = img.shape
    u = min(max((x + sx/2.0)/sx, 0.0), 1.0)
    v = min(max((y + sy/2.0)/sy, 0.0), 1.0)
    fx, fy = u*(W-1), v*(H-1)
    x0, y0 = int(fx), int(fy)
    x1, y1 = min(x0+1, W-1), min(y0+1, H-1)
    tx, ty = fx-x0, fy-y0
    top = img[y0, x0]*(1-tx) + img[y0, x1]*tx
    bot = img[y1, x0]*(1-tx) + img[y1, x1]*tx
    return (top*(1-ty) + bot*ty)/255.0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", default="simulation/worlds/mars_terrain.world")
    ap.add_argument("--png",   default="simulation/media/heightmaps/mars_relief.png")
    ap.add_argument("--size-x", type=float, default=20.0)
    ap.add_argument("--size-y", type=float, default=20.0)
    ap.add_argument("--size-z", type=float, default=0.4)
    ap.add_argument("--burial", type=float, default=0.15)
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()

    img = load_png_gray(a.png)
    txt = open(a.world).read()
    rows = []

    def repl(m):
        name, body = m.group(1), m.group(2)
        if "mars_rock/meshes/model.dae" not in body:
            return m.group(0)                      # zones / edge patches: leave untouched
        pm = POSE_RE.search(body); sm = SCALE_RE.search(body)
        if not pm or not sm:
            return m.group(0)
        x, y, z, rest, scale = float(pm.group(1)), float(pm.group(2)), float(pm.group(3)), pm.group(4), float(sm.group(1))
        h = sample(img, x, y, a.size_x, a.size_y) * a.size_z
        new_z = round(h - a.burial*MESH_H*scale, 4)
        rows.append((name, x, y, scale, round(h,4), z, new_z))
        new_body = body.replace(pm.group(0),
                                f"<pose>{pm.group(1)} {pm.group(2)} {new_z} {rest}</pose>", 1)
        return f'<model name="{name}">{new_body}</model>'

    new_txt = MODEL_RE.sub(repl, txt)

    print(f"{'name':18} {'x':>6} {'y':>6} {'scale':>6} {'terr_h':>7} {'old_z':>7} {'new_z':>7}")
    for r in rows:
        print(f"{r[0]:18} {r[1]:6.2f} {r[2]:6.2f} {r[3]:6.3f} {r[4]:7.3f} {r[5]:7.3f} {r[6]:7.3f}")
    print(f"\n{len(rows)} rocks matched.", "APPLIED." if a.apply else "(dry-run; pass --apply to write)")
    if a.apply:
        open(a.world, "w").write(new_txt)

if __name__ == "__main__":
    main()
