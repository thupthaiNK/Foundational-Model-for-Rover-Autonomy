"""
Purpose: Hold the latest reading from every sensor on the rover, know when one
         has gone quiet, and render the set either as text for the RViz2 overlay
         or as a CSV row for offline analysis.

         This is deliberately free of ROS imports. The logic most likely to be
         quietly wrong -- unit conversion, staleness, the LiDAR front sector --
         lives here so it can be tested with plain pytest, without starting a
         node, a bag or a simulator. sensor_overlay_node.py is a thin adapter
         over it, and experiments/bag_to_csv.py reuses the same rules offline so
         a graph cannot disagree with what was on screen in the field.

         The governing rule is that a stale field reads as absent, never as its
         last value. The IMU has dropped off I2C mid-run before; a frozen
         angle that still looks live is indistinguishable from a rover that is
         genuinely level, and in a CSV it becomes a flat line in a graph that
         nobody thinks to question.

Inputs:  values pushed in by whatever is reading the topics
Outputs: as_text() for a MarkerArray, as_csv_row() for a CSV writer
How to run: imported, not run. Tests:
    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest \
        ros2_ws/src/fm_perception/test/test_sensor_snapshot.py -q
Project: Onboard Visual Foundation Models for Mars Terrain Perception and Rover Navigation
"""

from __future__ import annotations

import math
from typing import Any

# Seconds after which a field stops being reported. Set per sensor because one
# shared timeout would either call a healthy DINOv2 stale or let a dead LiDAR
# look alive: inference takes 1.9-2.2 s under the full thirteen-node stack on
# the Pi, while a scan arrives at 10 Hz and an IMU sample far faster.
DEFAULT_TIMEOUTS_S: dict[str, float] = {
    # Terrain: 5.0 s is roughly two and a half missed inferences.
    "terrain_label": 5.0,
    "terrain_conf": 5.0,
    "terrain_probs": 5.0,
    "traversability_score": 5.0,
    "inference_latency_ms": 5.0,
    "frame_informative": 5.0,
    # LiDAR and IMU: fast sensors, so a 2 s gap already means something broke.
    "lidar_front_min_m": 2.0,
    "lidar_stop": 2.0,
    "lidar_reason": 2.0,
    "roll_deg": 2.0,
    "pitch_deg": 2.0,
    "yaw_deg": 2.0,
    "imu_slope_stop": 2.0,
    # Derived state and commands: published on change or at a moderate rate.
    "fused_verdict": 3.0,
    "cmd_vx": 3.0,
    "cmd_wz": 3.0,
    "explorer_active": 3.0,
    "e_stop": 3.0,
    "e_stop_reason": 3.0,
}

CSV_COLUMNS: list[str] = [
    "t_s",
    "terrain_label",
    "terrain_conf",
    "traversability_score",
    "inference_latency_ms",
    "frame_informative",
    "lidar_front_min_m",
    "lidar_status",
    "lidar_stop",
    "lidar_reason",
    "roll_deg",
    "pitch_deg",
    "yaw_deg",
    "imu_slope_stop",
    "fused_verdict",
    "cmd_vx",
    "cmd_wz",
    "explorer_active",
    "e_stop",
    "e_stop_reason",
]

# Columns computed from other fields rather than pushed in from a topic, so they
# have no timeout of their own.
DERIVED_COLUMNS: frozenset[str] = frozenset({"lidar_status"})

# Matches sector_half_width_deg already used by real_stuck_detection_node, so
# the overlay and the stuck detector are talking about the same patch of ground.
FRONT_SECTOR_HALF_WIDTH_DEG = 20.0

# Hard ceiling on a display line. RViz2's TEXT_VIEW_FACING lays text out in
# world units and centres each line on its own, so a long line does not wrap --
# it spreads across metres of the scene and stops being readable. Measured
# against the real thing on 2026-08-11 at a 120 px/m top-down view.
MAX_LINE_CHARS = 22


def parse_terrain_classification(text: str) -> tuple[str, float] | None:
    """Split a "label:confidence" verdict, or return None if it is malformed.

    Returning None rather than a default is the point. This feeds a safety
    display: rendering "soil 0.00" for something that could not be parsed would
    be worse than rendering nothing, because it looks like a real reading.
    """
    if not isinstance(text, str):
        return None
    parts = text.split(":")
    if len(parts) != 2:
        return None
    label = parts[0].strip()
    if not label:
        return None
    try:
        conf = float(parts[1].strip())
    except ValueError:
        return None
    return label, conf


def quaternion_to_euler_deg(x: float, y: float, z: float, w: float
                            ) -> tuple[float, float, float]:
    """Convert an orientation quaternion to roll, pitch, yaw in degrees.

    Standard ZYX (yaw-pitch-roll) convention, matching REP-103: roll about x,
    pitch about y, yaw about z, all right-handed.

    The quaternion is normalised first because an IMU driver's output drifts off
    unit length, and the pitch term is clamped before asin(): a near-vertical
    tilt can push the argument a hair past 1.0 through float error, and that is
    exactly the reading that matters most on a slope. Raising there would take
    the overlay node down at the moment it is most needed.
    """
    n = math.sqrt(x * x + y * y + z * z + w * w)
    if n == 0.0:
        return 0.0, 0.0, 0.0
    x, y, z, w = x / n, y / n, z / n, w / n

    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = max(-1.0, min(1.0, 2.0 * (w * y - z * x)))
    pitch = math.asin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return math.degrees(roll), math.degrees(pitch), math.degrees(yaw)


def front_sector_min(ranges, angle_min: float, angle_increment: float,
                     range_min: float, range_max: float,
                     half_width_deg: float = FRONT_SECTOR_HALF_WIDTH_DEG
                     ) -> float | None:
    """Closest valid return within +/- half_width_deg of straight ahead.

    Returns None when nothing in the sector is usable, which is a different
    state from "the LiDAR stopped" and is displayed differently.

    Two things this gets right that a naive version does not. Angles are wrapped
    into (-pi, pi] before the sector test, because the rover's sllidar publishes
    angle_min = 0, so forward sits at both ends of the array and an
    abs(angle) <= half_width test would see only half the beams it should.
    And returns outside [range_min, range_max], along with inf and NaN, are
    discarded: a 0.01 m artefact treated as an obstacle reads as an imminent
    collision while the rover sits in open ground.
    """
    if not len(ranges) or angle_increment == 0.0:
        return None

    half_width_rad = math.radians(half_width_deg)
    best: float | None = None

    for i, r in enumerate(ranges):
        if r is None:
            continue
        r = float(r)
        if math.isnan(r) or math.isinf(r):
            continue
        if r < range_min or r > range_max:
            continue

        angle = angle_min + i * angle_increment
        # wrap into (-pi, pi]
        angle = (angle + math.pi) % (2.0 * math.pi) - math.pi
        if abs(angle) > half_width_rad:
            continue

        if best is None or r < best:
            best = r

    return best


# ── topic to field mapping ──────────────────────────────────────────────────
#
# One definition, used by both the live overlay node and the offline bag
# converter. Kept here rather than in the node so a graph drawn afterwards
# cannot disagree with what was on the screen at the time -- if the two had
# their own copies, the first divergence would show up as an unexplained
# difference between a figure and a memory of the field test.
#
# These read message attributes and nothing else, so this module still imports
# no ROS. A recorded message and a live one are the same shape.

def _extract_terrain(msg) -> list[tuple[str, object]]:
    parsed = parse_terrain_classification(msg.data)
    if parsed is None:
        return []          # malformed reads as absent, never as a real verdict
    label, conf = parsed
    return [("terrain_label", label), ("terrain_conf", conf)]


def _extract_scan(msg) -> list[tuple[str, object]]:
    # None means alive but nothing in range, which lidar_status keeps distinct
    # from a sensor that stopped publishing.
    return [("lidar_front_min_m", front_sector_min(
        ranges=msg.ranges,
        angle_min=msg.angle_min,
        angle_increment=msg.angle_increment,
        range_min=msg.range_min,
        range_max=msg.range_max,
    ))]


def _extract_imu(msg) -> list[tuple[str, object]]:
    q = msg.orientation
    roll, pitch, yaw = quaternion_to_euler_deg(q.x, q.y, q.z, q.w)
    return [("roll_deg", roll), ("pitch_deg", pitch), ("yaw_deg", yaw)]


def _extract_twist(msg) -> list[tuple[str, object]]:
    return [("cmd_vx", msg.linear.x), ("cmd_wz", msg.angular.z)]


def _single(field: str):
    return lambda msg: [(field, msg.data)]


# topic -> (ROS message type name, extractor)
TOPIC_EXTRACTORS: dict[str, tuple[str, object]] = {
    "/terrain_classification": ("std_msgs/msg/String", _extract_terrain),
    # Carried in the snapshot but not on the display or in the CSV: four
    # probabilities do not fit a 22-character line, and a per-class column set
    # is better added to the converter when a graph actually needs it.
    "/terrain_class_probs": ("std_msgs/msg/Float32MultiArray",
                             lambda msg: [("terrain_probs", list(msg.data))]),
    "/traversability_score": ("std_msgs/msg/Float64",
                              _single("traversability_score")),
    "/inference_latency_ms": ("std_msgs/msg/Float64",
                              _single("inference_latency_ms")),
    "/terrain_frame_informative": ("std_msgs/msg/Bool",
                                   _single("frame_informative")),
    "/scan": ("sensor_msgs/msg/LaserScan", _extract_scan),
    "/lidar_proximity_stop": ("std_msgs/msg/Bool", _single("lidar_stop")),
    "/lidar_proximity_reason": ("std_msgs/msg/String", _single("lidar_reason")),
    "/exomy/imu_raw": ("sensor_msgs/msg/Imu", _extract_imu),
    "/imu_slope_stop": ("std_msgs/msg/Bool", _single("imu_slope_stop")),
    "/traversability_fused": ("std_msgs/msg/String", _single("fused_verdict")),
    "/exomy/cmd_vel": ("geometry_msgs/msg/Twist", _extract_twist),
    "/reactive_explorer/active": ("std_msgs/msg/Bool", _single("explorer_active")),
    "/e_stop": ("std_msgs/msg/Bool", _single("e_stop")),
    "/e_stop_reason": ("std_msgs/msg/String", _single("e_stop_reason")),
}


class SensorSnapshot:
    """The latest value of every sensor field, with per-field staleness."""

    def __init__(self, timeouts: dict[str, float] | None = None) -> None:
        self._timeouts = dict(DEFAULT_TIMEOUTS_S)
        if timeouts:
            unknown = set(timeouts) - set(self._timeouts)
            if unknown:
                raise KeyError(f"unknown sensor field(s): {sorted(unknown)}")
            self._timeouts.update(timeouts)
        self._values: dict[str, Any] = {}
        self._stamps: dict[str, float] = {}

    def apply(self, topic: str, msg: Any, t: float) -> bool:
        """Route one message through TOPIC_EXTRACTORS. False if unhandled.

        This is what keeps the live overlay and the offline converter honest:
        both call it, so neither can develop its own idea of what a topic means.
        """
        entry = TOPIC_EXTRACTORS.get(topic)
        if entry is None:
            return False
        _, extract = entry
        for name, value in extract(msg):
            self.update(name, value, t)
        return True

    def update(self, name: str, value: Any, t: float) -> None:
        """Record a value and the time it arrived. Unknown names raise.

        A typo would otherwise write a field nothing ever reads, and the display
        would go on showing the sensor as absent with no indication why.
        """
        if name not in self._timeouts:
            raise KeyError(f"unknown sensor field: {name!r}")
        self._values[name] = value
        self._stamps[name] = t

    def age(self, name: str, now: float) -> float | None:
        if name not in self._stamps:
            return None
        return now - self._stamps[name]

    def is_stale(self, name: str, now: float) -> bool:
        age = self.age(name, now)
        if age is None:
            return True
        return age > self._timeouts[name]

    def get(self, name: str, now: float) -> Any:
        """The value, or None if it was never written or has gone stale."""
        if self.is_stale(name, now):
            return None
        return self._values.get(name)

    def lidar_status(self, now: float) -> str:
        """Why the front range is or is not a number: ok, no_return, stale, absent.

        The range column alone cannot carry this. Open ground gives "nothing
        within 12 m" constantly and a dead sensor gives nothing either, and both
        write an empty cell -- so a gap in a field-test plot would be unreadable
        without somewhere to record which one it was. Outdoors, where most of
        Thursday's driving happens, no_return is the common case, not the alarm.
        """
        if "lidar_front_min_m" not in self._stamps:
            return "absent"
        if self.is_stale("lidar_front_min_m", now):
            return "stale"
        if self._values.get("lidar_front_min_m") is None:
            return "no_return"
        return "ok"

    # ── rendering ────────────────────────────────────────────────────────────

    def _field(self, name: str, now: float, fmt: str = "{}") -> str:
        """Render one field, or a visible marker of why it is missing."""
        if name not in self._stamps:
            return "--"
        if self.is_stale(name, now):
            return f"STALE {self.age(name, now):.1f}s"
        value = self._values.get(name)
        if value is None:
            # Written, fresh, and genuinely empty. Only the LiDAR does this, and
            # "no return" must not be confused with a dead sensor.
            return "no return"
        return fmt.format(value)

    @staticmethod
    def _flag(value) -> str:
        if value is None:
            return "--"
        if isinstance(value, bool):
            return "YES" if value else "no"
        return str(value)

    def as_text(self, now: float) -> str:
        """Short-line summary for the RViz2 text marker.

        Rendered on 2026-08-11 and rebuilt from what came back. RViz2's
        TEXT_VIEW_FACING centres every line independently and lays the text out
        in world units, so the first version -- wide lines padded with runs of
        spaces into neat columns -- came out as words scattered across several
        metres of the 3D view, completely illegible. What reads on a terminal is
        not what reads in a 3D scene.

        Hence MAX_LINE_CHARS and no run of two or more spaces anywhere, both
        pinned by tests. Lines of similar short length centre into something
        that still looks like a block.
        """
        label = self._field("terrain_label", now)
        conf = self._field("terrain_conf", now, "{:.2f}")
        score = self._field("traversability_score", now, "{:.2f}")
        latency = self._field("inference_latency_ms", now, "{:.0f}ms")

        front = self._field("lidar_front_min_m", now, "{:.2f}m")
        lidar_stop = self._flag(self.get("lidar_stop", now))

        # Roll and pitch keep a decimal: on a slope the difference between 8.4
        # and 8 degrees is the difference between two policies. Yaw gets its own
        # line because r-123.4 p-45.6 y359.9 together overflow the width, and
        # truncating that line would silently drop the heading.
        roll = self._field("roll_deg", now, "{:+.1f}")
        pitch = self._field("pitch_deg", now, "{:+.1f}")
        yaw = self._field("yaw_deg", now, "{:.1f}")

        vx = self._field("cmd_vx", now, "{:+.2f}")
        wz = self._field("cmd_wz", now, "{:+.2f}")

        lines = [
            f"TERRAIN {label} {conf}",
            f"score {score} {latency}",
            f"LIDAR {front} stop:{lidar_stop}",
            f"IMU r{roll} p{pitch}",
            f"yaw {yaw}",
            f"CMD {vx} {wz}",
        ]

        fused = self.get("fused_verdict", now)
        if fused is not None:
            lines.append(f"FUSED {fused}")

        # Only shown when something is actually asserting a stop. A permanent
        # "E-STOP no" line is one more thing to read past on a small display.
        if self.get("e_stop", now):
            reason = self.get("e_stop_reason", now) or "?"
            lines.append("E-STOP")
            lines.append(str(reason)[:MAX_LINE_CHARS])
        if self.get("imu_slope_stop", now):
            lines.append("SLOPE STOP")

        return "\n".join(line[:MAX_LINE_CHARS] for line in lines)

    def as_csv_row(self, now: float) -> list[Any]:
        """One row in CSV_COLUMNS order. Stale fields are written empty.

        Empty rather than carried forward, for the same reason the display
        blanks them: a converter that invents continuity across a dropout turns
        a gap into a flat line that reads as a real measurement.
        """
        row: list[Any] = [now]
        for col in CSV_COLUMNS[1:]:
            if col == "lidar_status":
                row.append(self.lidar_status(now))
                continue
            value = self.get(col, now)
            row.append("" if value is None else value)
        return row
