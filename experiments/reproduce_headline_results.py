"""
Purpose: One-command reproduction package for this thesis's headline
         offline-derivable numbers/figures -- backlog item 19, scoped
         2026-07-20 (grill-thesis). Runs each listed script, which reads
         already-cached features/CSVs/JSON in experiments/results/ and
         regenerates its CSV/figure output; does not re-run any Gazebo/ROS2
         live experiment (those are stochastic, real-time-sim results that
         are reproducible only by re-running their own launch file + script
         per that script's own docstring, not by a single offline command --
         see LIVE_GAZEBO_RESULTS below for the explicit list of what this
         package deliberately does not attempt).
Inputs:  experiments/results/feature_cache/*.npy, *.npz, temperature_scaling.json,
         and other cached CSVs already checked into experiments/results/.
Outputs: Regenerates each listed script's own CSV/figure outputs in place.
How to run:
    python3 experiments/reproduce_headline_results.py --dry-run   # verify wiring only, no execution
    python3 experiments/reproduce_headline_results.py             # actually regenerate everything
    python3 experiments/reproduce_headline_results.py --only cost_weight_sweep,mcnemar_test
Project: Onboard Visual Foundation Models for Mars Terrain Perception and Rover Navigation
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

EXPERIMENTS_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EXPERIMENTS_DIR / "results"
FEATURE_CACHE = RESULTS_DIR / "feature_cache"

# cost_weight_sweep.py imports fm_perception (the built ROS2 package) directly,
# without ROS2/Gazebo running -- it only needs the package's Python modules on
# sys.path, normally provided by sourcing ros2_ws/install/setup.bash. Adding
# it to PYTHONPATH here keeps this a genuine one-command package instead of
# requiring the caller to source the ROS2 overlay first.
ROS2_INSTALL_PYTHONPATH = (EXPERIMENTS_DIR.parent / "ros2_ws" / "install"
                            / "fm_perception" / "local" / "lib" / "python3.10" / "dist-packages")


def subprocess_env() -> dict:
    env = os.environ.copy()
    if ROS2_INSTALL_PYTHONPATH.exists():
        env["PYTHONPATH"] = f"{ROS2_INSTALL_PYTHONPATH}{os.pathsep}{env.get('PYTHONPATH', '')}"
    return env

# Each entry: name, script (relative to experiments/), extra CLI args,
# required input files that must already exist (cached data, not re-derived
# here), and the thesis section the regenerated number/figure supports.
HEADLINE_SCRIPTS = [
    dict(
        name="cost_weight_sweep",
        script="cost_weight_sweep.py",
        args=[],
        requires=[],  # builds its own deterministic Condition A grid, no cache needed
        section="Ch4 S4.8.26 H14 (COST_WEIGHT sweep)",
    ),
    dict(
        name="confidence_intervals",
        script="confidence_intervals.py",
        args=[],
        requires=[],  # hard-coded (accuracy, n) pairs cited from other results
        section="Ch4 S4.10.4 (Wilson CIs on headline numbers)",
    ),
    dict(
        name="coverage_risk_curve",
        script="coverage_risk_curve.py",
        args=[],
        requires=[
            FEATURE_CACHE / "dinov2_reg_small_train_1000_feats.npy",
            FEATURE_CACHE / "dinov2_reg_small_test_287_feats.npy",
            RESULTS_DIR / "temperature_scaling.json",
        ],
        section="Ch4 (coverage-risk curve, T=0.40 operating point)",
    ),
    dict(
        name="boundary_band_uncertainty",
        script="boundary_band_uncertainty.py",
        args=[],
        requires=[RESULTS_DIR / "raw_logs"],
        section="Ch4 S4.8.28 addendum (A3 boundary-band null result)",
    ),
    dict(
        name="mcnemar_test",
        script="mcnemar_test.py",
        args=[],
        requires=[
            FEATURE_CACHE / "dinov2_vitl_train_1000_feats.npy",
            FEATURE_CACHE / "dinov2_vitb_train_1000_feats.npy",
        ],
        section="Ch4 (Ensemble B vs DINOv2 ViT-L significance test)",
    ),
    dict(
        name="per_class_f1_probe",
        script="per_class_f1_probe.py",
        args=[],
        requires=[
            FEATURE_CACHE / "dinov2_reg_small_train_1000_feats.npy",
            FEATURE_CACHE / "marsbench_train_feats.npy",
        ],
        section="Ch4 (Probe A vs Probe C per-class F1, big_rock)",
    ),
    dict(
        name="joint_domain_probe",
        script="joint_domain_probe.py",
        args=[],
        requires=[
            FEATURE_CACHE / "dinov2_reg_small_train_1000_feats.npy",
            FEATURE_CACHE / "marsbench_train_feats.npy",
        ],
        section="Ch4 (E1 joint-domain probe, 35.80 -> -5.56pp gap)",
    ),
    dict(
        name="failure_case_analysis",
        script="failure_case_analysis.py",
        args=[],
        requires=[
            FEATURE_CACHE / "dinov2_reg_small_train_1000_feats.npy",
            FEATURE_CACHE / "dinov2_reg_small_test_287_feats.npy",
        ],
        section="Ch4 / Ch5 (failure-mode grid and confusion breakdown)",
    ),
]

# Live Gazebo/ROS2 results this package deliberately does NOT attempt to
# regenerate -- each is only reproducible by re-running its own launch file
# + script per that script's own docstring, and involves real-time
# simulation variance this offline package cannot replay deterministically.
LIVE_GAZEBO_RESULTS = [
    ("frontier_exploration_test.py", "Ch4 S4.8.28 H16 (frontier exploration-lite)"),
    ("explore_return_home_test.py", "Ch4 S4.8.29 H17 (explore-then-return-home)"),
    ("reobservation_test.py / reobservation_bayesian_test.py", "Ch4 S4.8.30 H18 / S4.8.32 H20"),
    ("two_stage_uncertain_test.py", "Ch4 S4.8.31 H19 (A4 two-stage policy)"),
    ("abort_to_home_test.py", "Ch4 S4.8.33 H21 (mission-level failsafe)"),
    ("semantic_frontier_test.py", "Ch4 S4.8.28 (semantic frontier selection)"),
    ("l6_lite_roundtrip_test.py / l5_lite_live_test.py", "Ch4 S4.8.26-4.8.27 H14/H15"),
    ("earth_mars_probe.py", "Ch4 S4.8.34 H22 -- offline but re-embeds raw images "
                             "(AI4Mars + RUGD) rather than reading a feature cache; "
                             "excluded here because a full re-run costs real GPU/CPU "
                             "time, not because it touches Gazebo. Run directly per "
                             "its own docstring if regenerating this one specifically."),
]


def check_requirements(entry: dict) -> list:
    missing = [str(p) for p in entry["requires"] if not Path(p).exists()]
    return missing


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true",
                         help="verify each script and its cached inputs exist, run nothing")
    parser.add_argument("--only", type=str, default=None,
                         help="comma-separated subset of script names to run")
    args = parser.parse_args()

    selected = HEADLINE_SCRIPTS
    if args.only:
        wanted = set(args.only.split(","))
        selected = [e for e in HEADLINE_SCRIPTS if e["name"] in wanted]
        unknown = wanted - {e["name"] for e in HEADLINE_SCRIPTS}
        if unknown:
            print(f"Unknown script name(s): {sorted(unknown)}", file=sys.stderr)
            sys.exit(1)

    print(f"{'DRY-RUN: ' if args.dry_run else ''}Reproducing {len(selected)} headline result(s) "
          f"from cached data (no Gazebo/ROS2 involved).\n")

    failures = []
    for entry in selected:
        script_path = EXPERIMENTS_DIR / entry["script"]
        print(f"-- {entry['name']} ({entry['section']}) --")
        if not script_path.exists():
            print(f"   MISSING SCRIPT: {script_path}")
            failures.append(entry["name"])
            continue
        missing = check_requirements(entry)
        if missing:
            print(f"   MISSING CACHED INPUT(S): {missing}")
            failures.append(entry["name"])
            continue
        if args.dry_run:
            print("   OK (script + cached inputs present)")
            continue
        result = subprocess.run([sys.executable, str(script_path)] + entry["args"],
                                 cwd=str(EXPERIMENTS_DIR.parent), env=subprocess_env())
        if result.returncode != 0:
            print(f"   FAILED (exit {result.returncode})")
            failures.append(entry["name"])
        else:
            print("   done")

    print(f"\nNOT covered by this package (live Gazebo/ROS2 results -- see each script's own "
          f"docstring to reproduce):")
    for script, section in LIVE_GAZEBO_RESULTS:
        print(f"   {script}: {section}")

    if failures:
        print(f"\n{len(failures)} of {len(selected)} FAILED: {failures}")
        sys.exit(1)
    print(f"\nAll {len(selected)} requested headline result(s) OK.")


if __name__ == "__main__":
    main()
