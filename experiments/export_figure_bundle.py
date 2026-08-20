"""
Purpose: Assemble one self-contained folder holding every figure that appears
         in the thesis, the MATLAB script that draws each one, and the CSV it
         reads, so the figures can be inspected or regenerated later without
         re-deriving which script made what. Writes the same bundle to two
         places: one inside the repository, one under Thesis/Picture/MATLAB.
Inputs:  experiments/*.m, experiments/results/figures/thesis/*, and the CSVs
         those scripts read from experiments/results/
         docs/word_transfer/*.md, for the figure-number to filename mapping
Outputs: docs/figure_bundle/  (in the repository)
         C:/Users/DELL/Desktop/Thesis/Picture/MATLAB/  (working copy)
How to run: python3 experiments/export_figure_bundle.py
Project: Onboard Visual Foundation Models for Mars Terrain Perception and Rover Navigation
"""
import os
import re
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
WT = os.path.join(ROOT, "docs", "word_transfer")
FIGDIR = os.path.join(HERE, "results", "figures", "thesis")
RESULTS = os.path.join(HERE, "results")

DESTS = [
    os.path.join(ROOT, "docs", "figure_bundle"),
    "/mnt/c/Users/DELL/Desktop/Thesis/Picture/MATLAB",
]

# Which script draws which output. Read off the exportgraphics calls in each
# script; the two that build their figures in a loop name their outputs
# through a variable, so those are listed here explicitly.
SCRIPT_OUTPUTS = {
    "make_thesis_figures.m": [
        "model_ranking_1000shot", "few_shot_accuracy_curve",
        "label_efficiency_curve", "coverage_risk_curve",
    ],
    "make_thesis_figures_2.m": [
        "confusion_matrix_dinov2_vitl", "tsne_dinov2_vitl",
    ],
    "make_deployed_confusion_figure.m": [
        "confusion_matrix_dinov2_reg_small",
    ],
    "make_thesis_figures_3.m": [
        "sim_to_real_gap", "accuracy_vs_speed", "backbone_scaling",
        "deployment_latency", "deployment_ram", "gazebo_zone_results",
        "paradigm_summary",
    ],
    "make_thesis_figures_4.m": ["real_camera_by_class"],
    "make_thesis_diagrams.m": ["fm_pipeline", "reactive_fsm"],
    "make_gazebo_plates.m": ["gazebo_world_plate", "gazebo_zone_views"],
}

# Poster-only variants. Not in the report, but they are generated artefacts of
# this project and the bundle exists so that nothing has to be re-derived later.
POSTER_FIGS = {
    "model_ranking_poster": "make_poster_figures.m",
    "accuracy_vs_speed_poster": "make_poster_figures.m",
    "reactive_fsm_poster": "make_thesis_diagrams.m",
    "gazebo_world_poster": "make_gazebo_plates.m",
    "exomy_cutout": "upscale_rover_cutout.py",
}

# Figures that are photographs or screenshots rather than MATLAB output.
NOT_MATLAB = {
    "exomy_platform": "photograph of the rover, not generated",
    "sandpit_arena": "photographs of the laboratory sandpit, not generated",
    # Website-only single-image variants, built by make_website_photos.py.
    # They are not thesis figures and are listed here only so a sweep over
    # the figure directory does not report them as unexplained.
    "sandpit_arena_single": "website variant of the arena photograph",
    "gazebo_world_oblique": "website variant, one Gazebo world render",
    "gazebo_rock_view": "website variant, one Gazebo camera frame",
    # Added 2026-08-20, placed by hand (converted from author-supplied HEIC
    # photos / cropped from existing docs/figures/gazebo_world_views/*.png),
    # not by make_website_photos.py.
    "exomy_gallery_1": "website teaser, ExoMy assembly photo 1 of 4",
    "exomy_gallery_2": "website teaser, ExoMy assembly photo 2 of 4",
    "exomy_gallery_3": "website teaser, ExoMy assembly photo 3 of 4",
    "exomy_gallery_4": "website teaser, ExoMy assembly photo 4 of 4",
    "gazebo_world_overhead": "website variant, unlabelled overhead Gazebo world render",
    "gazebo_rock_cluster": "website variant, close rover's-eye view of the rock quadrant",
    "gazebo_zone_boundary": "website variant, view across the sand/bedrock boundary",
    # A small matplotlib bar chart (not MATLAB), built inline for the
    # website only -- see docs/website/README.md or the build session notes
    # for 2026-08-20 for the generating snippet. Same 108-vs-1,000 numbers
    # as thesis Table/Figure discussion of the big-rock class imbalance.
    "big_rock_training_count": "website-only chart, usable training images per class",
}


def figure_numbers():
    """Map output stem -> the figure number it carries in the thesis, read
    from the [FIGURE: path | Figure N.M: ...] markers in the chapter files."""
    out = {}
    for name in sorted(os.listdir(WT)):
        if not name.endswith(".md") or name == "README.md":
            continue
        with open(os.path.join(WT, name)) as f:
            for m in re.finditer(r"\[FIGURE:\s*([^|]+)\|\s*Figure\s+([0-9.]+):", f.read()):
                stem = os.path.splitext(os.path.basename(m.group(1).strip()))[0]
                out[stem] = m.group(2).rstrip(".")
    return out


def script_inputs(script_path):
    """CSV files a script actually reads.

    Taken from the filenames in the script body, not from the Inputs: header.
    Two of the headers say "experiments/results/*.csv", and honouring that
    literally copies all 145 result files, 20 MB of them, most unrelated to
    any figure and all of them already tracked in experiments/results/."""
    with open(script_path) as f:
        body = f.read()
    return sorted({os.path.basename(m) for m in
                   re.findall(r"['\"]([\w./-]+\.csv)['\"]", body)})


def build(dest):
    figs = os.path.join(dest, "figures")
    scripts = os.path.join(dest, "scripts")
    data = os.path.join(dest, "data")
    for d in (figs, scripts, data):
        os.makedirs(d, exist_ok=True)

    numbers = figure_numbers()
    lines = [
        "# Thesis figure bundle",
        "",
        "Every figure in the thesis report, the MATLAB script that draws it, and",
        "the data it reads. Regenerate any figure by running its script from the",
        "repository root:",
        "",
        '    "/mnt/c/Program Files/MATLAB/R2025b/bin/matlab.exe" -batch "run(\'experiments/<script>\')"',
        "",
        "Figures are written as both .png (300 dpi, what the report embeds) and",
        ".pdf (vector, for reprinting at any size). Both are in figures/.",
        "",
        "| Figure | File | Drawn by | Reads |",
        "|---|---|---|---|",
    ]

    copied_scripts, copied_data = set(), set()
    rows = []
    for script, stems in SCRIPT_OUTPUTS.items():
        src = os.path.join(HERE, script)
        if not os.path.exists(src):
            continue
        shutil.copy2(src, os.path.join(scripts, script))
        copied_scripts.add(script)
        csvs = script_inputs(src)
        for pattern in csvs:
            base = os.path.basename(pattern)
            if "*" in base:
                prefix = base.split("*")[0]
                found = [f for f in os.listdir(RESULTS)
                         if f.startswith(prefix) and f.endswith(".csv")]
            else:
                found = [base] if os.path.exists(os.path.join(RESULTS, base)) else []
            for f in found:
                shutil.copy2(os.path.join(RESULTS, f), os.path.join(data, f))
                copied_data.add(f)
        reads = ", ".join(sorted({os.path.basename(c) for c in csvs})) or "see the script header"
        for stem in stems:
            num = numbers.get(stem)
            for ext in (".png", ".pdf"):
                p = os.path.join(FIGDIR, stem + ext)
                if os.path.exists(p):
                    shutil.copy2(p, os.path.join(figs, stem + ext))
            label = f"Figure {num}" if num else "not placed in the report"
            sort_key = tuple(int(x) for x in num.split(".")) if num else (99, 99)
            rows.append((sort_key, f"| {label} | `{stem}.png` | `{script}` | {reads} |"))

    for stem, why in NOT_MATLAB.items():
        num = numbers.get(stem)
        for ext in (".png", ".pdf"):
            p = os.path.join(FIGDIR, stem + ext)
            if os.path.exists(p):
                shutil.copy2(p, os.path.join(figs, stem + ext))
        sort_key = tuple(int(x) for x in num.split(".")) if num else (99, 99)
        rows.append((sort_key, f"| Figure {num} | `{stem}.png` | not applicable | {why} |"))

    lines += [r for _, r in sorted(rows)]

    # Poster-only variants, listed separately so the report mapping above
    # stays a clean one-to-one with the report's figure numbers.
    poster_dir = os.path.join(dest, "figures_poster")
    os.makedirs(poster_dir, exist_ok=True)
    lines += ["", "## Poster-only figure variants", "",
              "Built for the A1 poster, where the report versions stop being",
              "legible. The report keeps the full versions; see",
              "`docs/poster/README.md` for why each one differs.", "",
              "| File | Built by |", "|---|---|"]
    for stem, script in sorted(POSTER_FIGS.items()):
        for ext in (".png", ".pdf"):
            q = os.path.join(FIGDIR, stem + ext)
            if os.path.exists(q):
                shutil.copy2(q, os.path.join(poster_dir, stem + ext))
        src = os.path.join(HERE, script)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(scripts, script))
        lines.append(f"| `{stem}.png` | `{script}` |")

    lines += [
        "",
        "## Folders",
        "",
        "- `figures/` the rendered output, .png and .pdf for each",
        "- `scripts/` the MATLAB source, copied unchanged from `experiments/`",
        "- `data/` the CSVs those scripts read",
        "",
        "## Note on the two diagram scripts",
        "",
        "`make_thesis_diagrams.m` and `make_gazebo_plates.m` take no CSV input.",
        "The diagrams' layout is hand-specified in the script itself, and the",
        "Gazebo plates composite screenshots captured from the simulator. Their",
        "content is documented in the header comment of each script, including",
        "which source file each drawn element was derived from.",
        "",
    ]

    with open(os.path.join(dest, "README.md"), "w") as f:
        f.write("\n".join(lines))
    return len(copied_scripts), len(copied_data), len(os.listdir(figs))


if __name__ == "__main__":
    for dest in DESTS:
        os.makedirs(dest, exist_ok=True)
        s, d, fcount = build(dest)
        print(f"{dest}\n  {s} scripts, {d} data files, {fcount} report figures, "
              f"{len(os.listdir(os.path.join(dest, 'figures_poster')))} poster figures")
