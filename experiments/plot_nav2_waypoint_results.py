#!/usr/bin/env python3
"""
Purpose: Aggregate and plot the Nav2 waypoint experiment results (Task 10's
         CSV) — success rate and mean path length, static vs live costmap.
Inputs:  experiments/results/nav2_waypoint_experiment.csv
Outputs: experiments/results/figures/nav2_waypoint_comparison.png
How to run:
    python3 experiments/plot_nav2_waypoint_results.py
Project: Onboard Visual Foundation Models for Mars Terrain Perception and Rover Navigation
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

CSV_PATH = os.path.join(os.path.dirname(__file__), "results", "nav2_waypoint_experiment.csv")
FIG_PATH = os.path.join(os.path.dirname(__file__), "results", "figures", "nav2_waypoint_comparison.png")


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for condition, group in df.groupby("condition"):
        successful = group[group["success"] == True]  # noqa: E712
        rows.append({
            "condition": condition,
            "success_rate": len(successful) / len(group),
            "mean_path_length_m": successful["path_length_m"].mean() if len(successful) else float("nan"),
            "mean_replan_count": group["replan_count"].mean(),
        })
    return pd.DataFrame(rows)


def plot(summary: pd.DataFrame, fig_path: str = FIG_PATH) -> None:
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].bar(summary["condition"], summary["success_rate"] * 100, color=["#4CAF50", "#2196F3"])
    axes[0].set_ylabel("Success rate (%)")
    axes[0].set_ylim(0, 100)
    axes[0].set_title("Waypoint success rate")

    axes[1].bar(summary["condition"], summary["mean_path_length_m"], color=["#4CAF50", "#2196F3"])
    axes[1].set_ylabel("Mean path length (m), successful trials only")
    axes[1].set_title("Path length")

    fig.suptitle("Nav2 waypoint navigation: static known-hazard map vs live DINOv2 costmap")
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)


def main():
    df = pd.read_csv(CSV_PATH)
    summary = summarize(df)
    print(summary.to_string(index=False))
    plot(summary)
    print(f"Figure saved to {FIG_PATH}")


if __name__ == "__main__":
    main()
