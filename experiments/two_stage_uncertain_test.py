#!/usr/bin/env python3
"""
Purpose: Live-capture + offline measurement of the A4 two-stage uncertain
         policy's false-STOP reduction (item 11 of the L1-L6 further-work
         plan, 2026-07-19). Records the FULL-RATE /terrain_classification
         stream from a live Gazebo drive in a flicker-prone zone (the raw
         DINOv2 node log is throttled to every 10th frame, which cannot
         resolve consecutive votes), then applies -- deterministically, on
         that one recorded stream, so the comparison is perfectly fair and
         free of the run-to-run nondeterminism seen elsewhere in this
         thesis -- both the baseline discrete policy (uncertain -> STOP) and
         the two_stage_vote_verdict() gate, and reports how many of the
         baseline's uncertain-STOP episodes a strict-majority vote over the
         next N frames would have released early as single-frame false
         alarms.
         Safety framing: the vote NEVER delays entering a STOP (that is
         unchanged from baseline); it only shortens how long an uncertain
         STOP that turns out to be a single-frame artefact is held. So the
         measured "false-STOPs prevented" is a comfort/throughput gain at
         identical safety, exactly the A4 scope.
Inputs:  live /terrain_classification (std_msgs/String "label:confidence")
             while this node runs (Terminal 2), OR
         --replay <csv> to recompute from a previously recorded stream.
Outputs: experiments/results/two_stage_uncertain_stream.csv (raw stream)
         experiments/results/two_stage_uncertain_summary.txt
How to run:
    # Terminal 1: a live drive through a flicker-prone zone (rock cluster)
    ros2 launch simulation/launch/dinov2_controller_test.launch.py \
        spawn_x:=2.5 spawn_y:=-3.5
    # Terminal 2:
    python3 experiments/two_stage_uncertain_test.py --duration 240
    # Or, purely offline from an existing capture:
    python3 experiments/two_stage_uncertain_test.py --replay \
        experiments/results/two_stage_uncertain_stream.csv
Project: Onboard Visual Foundation Models for Mars Terrain Perception and Rover Navigation
"""
import argparse
import csv
import os
import sys
import time
from typing import List, Tuple

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
CONFIDENCE_THRESHOLD = 0.40  # matches dinov2_controller_test.launch.py

# Traversable = non-zero discrete policy speed (soil/sand/bedrock), mirroring
# terrain_controller_node.POLICY. Kept local so this analysis has no ROS
# import dependency when run in --replay mode.
TRAVERSABLE = {"soil", "sand", "bedrock"}


def effective_label(raw_label: str, confidence: float) -> str:
    """Apply the controller's own threshold rule: below-threshold -> uncertain
    (terrain_controller_node._terrain_callback)."""
    return "uncertain" if confidence < CONFIDENCE_THRESHOLD else raw_label


def measure(stream: List[Tuple[float, str, float]], window: int):
    """stream = [(t_s, raw_label, confidence), ...] in arrival order.
    Returns (n_uncertain_stop_episodes, n_false_stops_prevented, details).
    An 'episode' begins when an effective-uncertain frame arrives while not
    already stopped-on-uncertain; the vote then looks at the next `window`
    effective labels (including the triggering frame) and, on a strict
    traversable majority, counts the episode as a prevented false-STOP."""
    labels = [effective_label(lbl, conf) for (_t, lbl, conf) in stream]
    episodes = 0
    prevented = 0
    i = 0
    n = len(labels)
    while i < n:
        if labels[i] != "uncertain":
            i += 1
            continue
        # An uncertain STOP episode begins at i.
        episodes += 1
        vote = labels[i:i + window]
        if len(vote) == window:
            traversable = sum(1 for v in vote if v in TRAVERSABLE)
            if traversable * 2 > window:
                prevented += 1
        # Skip to the end of this contiguous-uncertain run so back-to-back
        # uncertain frames count as ONE episode, not many.
        j = i
        while j < n and labels[j] == "uncertain":
            j += 1
        i = j
    return episodes, prevented, labels


def summarise(stream, windows=(3, 5)):
    total = len(stream)
    n_uncertain_frames = sum(
        1 for (_t, lbl, conf) in stream
        if effective_label(lbl, conf) == "uncertain")
    lines = []
    lines.append("=" * 66)
    lines.append("A4 TWO-STAGE UNCERTAIN POLICY -- FALSE-STOP MEASUREMENT")
    lines.append(f"Total classification frames (full rate): {total}")
    lines.append(f"Effective-uncertain frames (conf < {CONFIDENCE_THRESHOLD}): "
                 f"{n_uncertain_frames} ({100.0*n_uncertain_frames/max(total,1):.1f}%)")
    for w in windows:
        episodes, prevented, _ = measure(stream, window=w)
        pct = 100.0 * prevented / episodes if episodes else 0.0
        lines.append(
            f"window={w}: {episodes} uncertain-STOP episodes, "
            f"{prevented} released early as false-STOPs ({pct:.1f}% reduction)")
    lines.append("=" * 66)
    return "\n".join(lines)


def run_live(duration_s: float) -> List[Tuple[float, str, float]]:
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import String

    stream: List[Tuple[float, str, float]] = []

    class Rec(Node):
        def __init__(self):
            super().__init__("two_stage_uncertain_recorder")
            self._t0 = time.time()
            self.create_subscription(String, "/terrain_classification",
                                     self._cb, 50)
            self.get_logger().info(
                f"Recording /terrain_classification full-rate for {duration_s:.0f}s")

        def _cb(self, msg):
            parts = msg.data.split(":")
            if len(parts) != 2:
                return
            try:
                conf = float(parts[1])
            except ValueError:
                return
            stream.append((round(time.time() - self._t0, 3), parts[0].strip(), conf))

    rclpy.init()
    node = Rec()
    try:
        while rclpy.ok() and (time.time() - node._t0) < duration_s:
            rclpy.spin_once(node, timeout_sec=0.2)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()
    return stream


def load_replay(path: str) -> List[Tuple[float, str, float]]:
    stream = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            stream.append((float(row["t_s"]), row["raw_label"],
                           float(row["confidence"])))
    return stream


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=float, default=240.0)
    ap.add_argument("--replay", type=str, default=None)
    args = ap.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    if args.replay:
        stream = load_replay(args.replay)
        print(f"Replaying {len(stream)} frames from {args.replay}")
    else:
        stream = run_live(args.duration)
        stream_path = os.path.join(RESULTS_DIR, "two_stage_uncertain_stream.csv")
        with open(stream_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t_s", "raw_label", "confidence"])
            w.writerows(stream)
        print(f"Saved raw stream ({len(stream)} frames) -> {stream_path}")

    if not stream:
        print("No classification frames captured -- was the pipeline running?")
        sys.exit(1)

    summary = summarise(stream)
    print("\n" + summary)
    summary_path = os.path.join(RESULTS_DIR, "two_stage_uncertain_summary.txt")
    with open(summary_path, "w") as f:
        f.write(summary + "\n")
    print(f"\nSaved: {summary_path}")


if __name__ == "__main__":
    main()
