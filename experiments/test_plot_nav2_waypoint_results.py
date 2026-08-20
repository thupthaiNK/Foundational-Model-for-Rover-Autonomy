"""
Purpose: Unit test for the summarize() aggregation function in
         plot_nav2_waypoint_results.py, using a small synthetic DataFrame —
         no dependency on the real experiment CSV.
Inputs:  None.
Outputs: pytest results.
How to run: python3 -m pytest experiments/test_plot_nav2_waypoint_results.py -v
Project: Onboard Visual Foundation Models for Mars Terrain Perception and Rover Navigation
"""
import pandas as pd
from plot_nav2_waypoint_results import summarize


def test_summarize_success_rate_and_mean_path_length():
    df = pd.DataFrame([
        {"condition": "static", "success": True, "path_length_m": 20.0, "replan_count": 0},
        {"condition": "static", "success": True, "path_length_m": 22.0, "replan_count": 0},
        {"condition": "static", "success": False, "path_length_m": 5.0, "replan_count": 1},
        {"condition": "live", "success": True, "path_length_m": 25.0, "replan_count": 2},
        {"condition": "live", "success": True, "path_length_m": 27.0, "replan_count": 3},
    ])
    summary = summarize(df)
    static_row = summary[summary["condition"] == "static"].iloc[0]
    live_row = summary[summary["condition"] == "live"].iloc[0]

    assert static_row["success_rate"] == 2 / 3
    assert static_row["mean_path_length_m"] == 21.0  # mean of the 2 successful trials only
    assert live_row["success_rate"] == 1.0
    assert live_row["mean_path_length_m"] == 26.0
    assert live_row["mean_replan_count"] == 2.5
