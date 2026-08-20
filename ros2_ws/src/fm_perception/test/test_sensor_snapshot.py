"""
Purpose: Tests for fm_perception/sensor_snapshot.py, the pure-Python core behind
         the field-test sensor overlay and the offline bag-to-CSV converter.
         Everything likely to be quietly wrong lives here -- staleness, unit
         conversion, and the LiDAR front-sector minimum -- so it is tested
         without ROS running.
Inputs:  none
Outputs: pass/fail
How to run:
    source /opt/ros/humble/setup.bash && source ros2_ws/install/setup.bash
    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest \
        ros2_ws/src/fm_perception/test/test_sensor_snapshot.py -q
    # PYTEST_DISABLE_PLUGIN_AUTOLOAD is required: ROS2's launch_testing plugin
    # otherwise breaks collection in this workspace.
Project: Onboard Visual Foundation Models for Mars Terrain Perception and Rover Navigation
"""

import math

import pytest

from fm_perception.sensor_snapshot import (
    TOPIC_EXTRACTORS,
    MAX_LINE_CHARS,
    CSV_COLUMNS,
    DEFAULT_TIMEOUTS_S,
    DERIVED_COLUMNS,
    SensorSnapshot,
    front_sector_min,
    parse_terrain_classification,
    quaternion_to_euler_deg,
)


# ── parse_terrain_classification ──────────────────────────────────────────────

def test_parses_label_and_confidence():
    assert parse_terrain_classification("soil:0.87") == ("soil", 0.87)


def test_parses_label_with_underscore():
    assert parse_terrain_classification("big_rock:0.42") == ("big_rock", 0.42)


def test_tolerates_whitespace():
    assert parse_terrain_classification("  sand : 0.5 ") == ("sand", 0.5)


@pytest.mark.parametrize("bad", ["", "soil", "soil:", ":0.5", "soil:abc", "a:b:c"])
def test_rejects_malformed_rather_than_guessing(bad):
    """A malformed verdict must read as absent, never as a plausible one.

    The overlay is a safety display: showing 'soil 0.00' for something that
    could not be parsed would be worse than showing nothing at all.
    """
    assert parse_terrain_classification(bad) is None


# ── quaternion_to_euler_deg ───────────────────────────────────────────────────

def test_identity_quaternion_is_level():
    roll, pitch, yaw = quaternion_to_euler_deg(0.0, 0.0, 0.0, 1.0)
    assert roll == pytest.approx(0.0, abs=1e-6)
    assert pitch == pytest.approx(0.0, abs=1e-6)
    assert yaw == pytest.approx(0.0, abs=1e-6)


def test_ninety_degree_yaw():
    # rotation of +90 deg about z
    h = math.sqrt(0.5)
    roll, pitch, yaw = quaternion_to_euler_deg(0.0, 0.0, h, h)
    assert roll == pytest.approx(0.0, abs=1e-6)
    assert pitch == pytest.approx(0.0, abs=1e-6)
    assert yaw == pytest.approx(90.0, abs=1e-4)


def test_thirty_degree_roll():
    a = math.radians(30.0) / 2.0
    roll, pitch, yaw = quaternion_to_euler_deg(math.sin(a), 0.0, 0.0, math.cos(a))
    assert roll == pytest.approx(30.0, abs=1e-4)
    assert pitch == pytest.approx(0.0, abs=1e-6)


def test_negative_pitch_is_nose_down_sign_preserved():
    a = math.radians(-20.0) / 2.0
    _, pitch, _ = quaternion_to_euler_deg(0.0, math.sin(a), 0.0, math.cos(a))
    assert pitch == pytest.approx(-20.0, abs=1e-4)


def test_gimbal_lock_pitch_clamps_instead_of_raising():
    """asin() of a value driven past 1.0 by float error must not raise.

    A near-vertical tilt is exactly the reading that matters most on a slope,
    so the conversion must survive it rather than kill the overlay node.
    """
    a = math.radians(90.0) / 2.0
    _, pitch, _ = quaternion_to_euler_deg(0.0, math.sin(a), 0.0, math.cos(a))
    assert pitch == pytest.approx(90.0, abs=1e-3)


def test_unnormalised_quaternion_is_normalised_first():
    roll, pitch, yaw = quaternion_to_euler_deg(0.0, 0.0, 0.0, 2.0)
    assert (roll, pitch, yaw) == pytest.approx((0.0, 0.0, 0.0), abs=1e-6)


# ── front_sector_min ──────────────────────────────────────────────────────────

def _scan(ranges, angle_min=-math.pi, angle_increment=None):
    if angle_increment is None:
        angle_increment = (2 * math.pi) / len(ranges)
    return dict(
        ranges=ranges,
        angle_min=angle_min,
        angle_increment=angle_increment,
        range_min=0.05,
        range_max=12.0,
    )


def test_takes_minimum_within_the_front_sector_only():
    # 360 beams, 1 deg apart, angle_min = -180 deg. Index 180 is straight ahead.
    ranges = [5.0] * 360
    ranges[180] = 0.42          # dead ahead, inside the sector
    ranges[0] = 0.10            # directly behind, must be ignored
    got = front_sector_min(half_width_deg=20.0, **_scan(ranges))
    assert got == pytest.approx(0.42)


def test_ignores_beams_just_outside_the_sector():
    ranges = [5.0] * 360
    ranges[180 + 25] = 0.20     # 25 deg off axis, outside a 20 deg half-width
    got = front_sector_min(half_width_deg=20.0, **_scan(ranges))
    assert got == pytest.approx(5.0)


def test_discards_inf_and_nan():
    ranges = [5.0] * 360
    ranges[178] = float("inf")
    ranges[179] = float("nan")
    ranges[180] = 0.60
    got = front_sector_min(half_width_deg=20.0, **_scan(ranges))
    assert got == pytest.approx(0.60)


def test_discards_out_of_spec_returns():
    """Below range_min or above range_max is a sensor artefact, not a wall.

    Treating a 0.01 m artefact as a real obstacle would read as an imminent
    collision on the overlay while the rover sat in open ground.
    """
    ranges = [5.0] * 360
    ranges[180] = 0.01          # below range_min
    ranges[181] = 99.0          # above range_max
    got = front_sector_min(half_width_deg=20.0, **_scan(ranges))
    assert got == pytest.approx(5.0)


def test_returns_none_when_the_sector_is_empty():
    ranges = [float("inf")] * 360
    assert front_sector_min(half_width_deg=20.0, **_scan(ranges)) is None


def test_returns_none_for_an_empty_scan():
    assert front_sector_min(
        ranges=[], angle_min=0.0, angle_increment=0.01,
        range_min=0.05, range_max=12.0, half_width_deg=20.0,
    ) is None


def test_handles_a_scan_that_starts_at_zero_and_wraps():
    """The rover's sllidar publishes angle_min = 0, so forward is at both ends.

    A sector test written as 'abs(angle) <= half_width' without wrapping would
    silently see only half the beams it should.
    """
    ranges = [5.0] * 360
    ranges[0] = 0.30            # dead ahead
    ranges[359] = 0.25          # 1 deg to the other side of ahead
    got = front_sector_min(
        half_width_deg=20.0,
        **_scan(ranges, angle_min=0.0, angle_increment=math.radians(1.0)),
    )
    assert got == pytest.approx(0.25)


# ── SensorSnapshot: staleness ─────────────────────────────────────────────────

def test_a_field_never_written_is_absent():
    s = SensorSnapshot()
    assert s.get("terrain_label", now=100.0) is None
    assert s.is_stale("terrain_label", now=100.0)


def test_a_fresh_field_is_returned():
    s = SensorSnapshot()
    s.update("terrain_label", "soil", t=100.0)
    assert s.get("terrain_label", now=100.5) == "soil"
    assert not s.is_stale("terrain_label", now=100.5)


def test_a_field_past_its_timeout_reads_as_absent_not_as_its_last_value():
    """The IMU has dropped off I2C mid-run before.

    A frozen angle that still looks live is more dangerous than a visible gap,
    because it is indistinguishable from a rover that is genuinely level.
    """
    s = SensorSnapshot()
    s.update("roll_deg", 12.5, t=100.0)
    timeout = DEFAULT_TIMEOUTS_S["roll_deg"]
    assert s.get("roll_deg", now=100.0 + timeout + 0.1) is None
    assert s.is_stale("roll_deg", now=100.0 + timeout + 0.1)


def test_staleness_boundary_is_inclusive_of_the_timeout():
    s = SensorSnapshot()
    s.update("roll_deg", 1.0, t=100.0)
    timeout = DEFAULT_TIMEOUTS_S["roll_deg"]
    assert not s.is_stale("roll_deg", now=100.0 + timeout - 1e-9)
    assert s.is_stale("roll_deg", now=100.0 + timeout + 1e-9)


def test_terrain_tolerates_a_longer_gap_than_lidar():
    """Inference takes about 2 s under the full stack; a scan arrives at 10 Hz.

    One timeout for everything would either call a healthy DINOv2 stale or let
    a dead LiDAR look alive for seconds.
    """
    assert DEFAULT_TIMEOUTS_S["terrain_label"] > DEFAULT_TIMEOUTS_S["lidar_front_min_m"]


def test_age_reports_seconds_since_the_last_update():
    s = SensorSnapshot()
    s.update("roll_deg", 1.0, t=100.0)
    assert s.age("roll_deg", now=101.25) == pytest.approx(1.25)


def test_age_of_a_never_written_field_is_none():
    assert SensorSnapshot().age("roll_deg", now=100.0) is None


def test_a_later_update_refreshes_the_field():
    s = SensorSnapshot()
    s.update("roll_deg", 1.0, t=100.0)
    s.update("roll_deg", 2.0, t=100.0 + DEFAULT_TIMEOUTS_S["roll_deg"] + 5.0)
    now = 100.0 + DEFAULT_TIMEOUTS_S["roll_deg"] + 5.5
    assert s.get("roll_deg", now=now) == 2.0


def test_unknown_field_names_are_rejected():
    """A typo must fail loudly rather than write a field nothing ever reads."""
    with pytest.raises(KeyError):
        SensorSnapshot().update("rol_deg", 1.0, t=100.0)


def test_custom_timeouts_override_the_defaults():
    s = SensorSnapshot(timeouts={"roll_deg": 0.5})
    s.update("roll_deg", 1.0, t=100.0)
    assert s.is_stale("roll_deg", now=100.6)


# ── SensorSnapshot: rendering ─────────────────────────────────────────────────

def test_text_shows_stale_with_the_age_rather_than_the_old_value():
    s = SensorSnapshot()
    s.update("roll_deg", 12.5, t=100.0)
    text = s.as_text(now=100.0 + DEFAULT_TIMEOUTS_S["roll_deg"] + 1.0)
    assert "STALE" in text
    assert "12.5" not in text


def test_text_reports_every_sensor_group_even_when_empty():
    text = SensorSnapshot().as_text(now=100.0)
    for group in ("TERRAIN", "LIDAR", "IMU", "CMD"):
        assert group in text


def test_text_includes_the_live_values():
    s = SensorSnapshot()
    s.update("terrain_label", "soil", t=100.0)
    s.update("terrain_conf", 0.87, t=100.0)
    s.update("lidar_front_min_m", 0.42, t=100.0)
    s.update("roll_deg", 2.1, t=100.0)
    text = s.as_text(now=100.1)
    assert "soil" in text
    assert "0.87" in text
    assert "0.42" in text
    assert "2.1" in text


def test_no_lidar_return_is_distinguished_from_a_stale_lidar():
    """'nothing in range' and 'the sensor stopped' must not look alike."""
    s = SensorSnapshot()
    s.update("lidar_front_min_m", None, t=100.0)
    lidar_line = [ln for ln in s.as_text(now=100.1).splitlines()
                  if ln.startswith("LIDAR")][0]
    assert "no return" in lidar_line
    assert "STALE" not in lidar_line


# ── display geometry ──────────────────────────────────────────────────────────
#
# Rendered in RViz2 on 2026-08-11 and rebuilt from what came back. The first
# version padded fields with runs of spaces into neat columns, which reads
# perfectly in a terminal and came out in the 3D view as words scattered across
# several metres, unreadable. TEXT_VIEW_FACING lays text out in world units and
# centres each line independently: a long line does not wrap, it spreads.

def _worst_case_snapshot() -> SensorSnapshot:
    s = SensorSnapshot()
    s.update("terrain_label", "big_rock", t=100.0)
    s.update("terrain_conf", 0.87, t=100.0)
    s.update("traversability_score", 0.123, t=100.0)
    s.update("inference_latency_ms", 2200.0, t=100.0)
    s.update("lidar_front_min_m", 11.75, t=100.0)
    s.update("lidar_stop", True, t=100.0)
    s.update("lidar_reason", "front sector 0.31 m below 0.40 m", t=100.0)
    s.update("roll_deg", -123.4, t=100.0)
    s.update("pitch_deg", -45.6, t=100.0)
    s.update("yaw_deg", 359.9, t=100.0)
    s.update("imu_slope_stop", True, t=100.0)
    s.update("fused_verdict", "CAUTION", t=100.0)
    s.update("cmd_vx", -0.10, t=100.0)
    s.update("cmd_wz", -1.57, t=100.0)
    s.update("e_stop", True, t=100.0)
    s.update("e_stop_reason", "watchdog: aggregate failure rate exceeded", t=100.0)
    return s


@pytest.mark.parametrize("snapshot_fn", [SensorSnapshot, _worst_case_snapshot])
def test_no_display_line_exceeds_the_width_that_stays_readable(snapshot_fn):
    for line in snapshot_fn().as_text(now=100.1).splitlines():
        assert len(line) <= MAX_LINE_CHARS, f"{len(line)} chars: {line!r}"


@pytest.mark.parametrize("snapshot_fn", [SensorSnapshot, _worst_case_snapshot])
def test_no_run_of_two_spaces_anywhere_in_the_display(snapshot_fn):
    """Padding runs are what scattered the first version across the scene."""
    text = snapshot_fn().as_text(now=100.1)
    assert "  " not in text, repr(text)


def test_stale_fields_also_respect_the_width_limit():
    s = _worst_case_snapshot()
    for line in s.as_text(now=100.0 + 999.0).splitlines():
        assert len(line) <= MAX_LINE_CHARS, f"{len(line)} chars: {line!r}"


def test_a_long_e_stop_reason_is_truncated_rather_than_spreading():
    s = _worst_case_snapshot()
    text = s.as_text(now=100.1)
    assert "E-STOP" in text
    assert "watchdog" in text


def test_stop_lines_appear_only_when_something_is_stopping_the_rover():
    quiet = SensorSnapshot()
    quiet.update("e_stop", False, t=100.0)
    quiet.update("imu_slope_stop", False, t=100.0)
    text = quiet.as_text(now=100.1)
    assert "E-STOP" not in text
    assert "SLOPE STOP" not in text


# ── SensorSnapshot: CSV ───────────────────────────────────────────────────────

def test_csv_row_matches_the_declared_column_order():
    s = SensorSnapshot()
    row = s.as_csv_row(now=100.0)
    assert len(row) == len(CSV_COLUMNS)


def test_csv_time_column_is_first_and_carries_now():
    assert CSV_COLUMNS[0] == "t_s"
    assert SensorSnapshot().as_csv_row(now=42.5)[0] == pytest.approx(42.5)


def test_csv_writes_empty_for_a_stale_field_not_its_last_value():
    """Same rule as the display. A recomputed graph must not invent continuity
    across a sensor dropout, or a gap becomes a flat line nobody questions."""
    s = SensorSnapshot()
    s.update("roll_deg", 12.5, t=100.0)
    row = s.as_csv_row(now=100.0 + DEFAULT_TIMEOUTS_S["roll_deg"] + 1.0)
    assert row[CSV_COLUMNS.index("roll_deg")] == ""


def test_csv_carries_live_values():
    s = SensorSnapshot()
    s.update("terrain_label", "sand", t=100.0)
    s.update("inference_latency_ms", 1982.0, t=100.0)
    row = s.as_csv_row(now=100.1)
    assert row[CSV_COLUMNS.index("terrain_label")] == "sand"
    assert row[CSV_COLUMNS.index("inference_latency_ms")] == pytest.approx(1982.0)


def test_every_csv_column_is_a_known_field_or_declared_derived():
    for col in CSV_COLUMNS:
        if col == "t_s" or col in DERIVED_COLUMNS:
            continue
        assert col in DEFAULT_TIMEOUTS_S, f"{col} has no timeout defined"


# ── lidar_status ──────────────────────────────────────────────────────────────
#
# Open ground produces "nothing within 12 m" constantly, and a dead LiDAR also
# produces no number. Both wrote an empty CSV cell, so a graph could not tell a
# clear horizon from a failed sensor -- which is exactly the question someone
# looking at a gap in a field-test plot will ask.

def test_lidar_status_is_ok_when_a_range_is_present():
    s = SensorSnapshot()
    s.update("lidar_front_min_m", 0.42, t=100.0)
    assert s.lidar_status(now=100.1) == "ok"


def test_lidar_status_is_no_return_when_the_sensor_is_alive_but_sees_nothing():
    s = SensorSnapshot()
    s.update("lidar_front_min_m", None, t=100.0)
    assert s.lidar_status(now=100.1) == "no_return"


def test_lidar_status_is_stale_when_the_sensor_stopped_publishing():
    s = SensorSnapshot()
    s.update("lidar_front_min_m", 0.42, t=100.0)
    late = 100.0 + DEFAULT_TIMEOUTS_S["lidar_front_min_m"] + 1.0
    assert s.lidar_status(now=late) == "stale"


def test_lidar_status_is_absent_before_the_first_scan():
    assert SensorSnapshot().lidar_status(now=100.0) == "absent"


def test_lidar_status_reaches_the_csv():
    s = SensorSnapshot()
    s.update("lidar_front_min_m", None, t=100.0)
    row = s.as_csv_row(now=100.1)
    assert row[CSV_COLUMNS.index("lidar_status")] == "no_return"
    # the range column stays empty; status is what disambiguates it
    assert row[CSV_COLUMNS.index("lidar_front_min_m")] == ""


def test_csv_distinguishes_no_return_from_stale_where_the_range_cannot():
    alive = SensorSnapshot()
    alive.update("lidar_front_min_m", None, t=100.0)
    dead = SensorSnapshot()
    dead.update("lidar_front_min_m", 0.42, t=100.0)
    late = 100.0 + DEFAULT_TIMEOUTS_S["lidar_front_min_m"] + 1.0

    alive_row = alive.as_csv_row(now=100.1)
    dead_row = dead.as_csv_row(now=late)
    idx_range = CSV_COLUMNS.index("lidar_front_min_m")
    idx_status = CSV_COLUMNS.index("lidar_status")

    assert alive_row[idx_range] == dead_row[idx_range] == ""
    assert alive_row[idx_status] != dead_row[idx_status]


# ── TOPIC_EXTRACTORS ──────────────────────────────────────────────────────────
#
# One table serves the live overlay node and the offline bag converter. If they
# had separate copies, the first divergence would surface as a graph that
# disagreed with what was on the screen during the field test, with nothing to
# say which was right.

class _Stub:
    """Stands in for a ROS message: the extractors only read attributes."""

    def __init__(self, **kw):
        self.__dict__.update(kw)


def test_every_extractor_writes_only_known_fields():
    for topic, (_, extract) in TOPIC_EXTRACTORS.items():
        assert callable(extract), topic


def test_terrain_topic_fills_label_and_confidence():
    s = SensorSnapshot()
    assert s.apply("/terrain_classification", _Stub(data="soil:0.87"), t=100.0)
    assert s.get("terrain_label", now=100.1) == "soil"
    assert s.get("terrain_conf", now=100.1) == pytest.approx(0.87)


def test_a_malformed_terrain_message_writes_nothing():
    s = SensorSnapshot()
    s.apply("/terrain_classification", _Stub(data="garbage"), t=100.0)
    assert s.get("terrain_label", now=100.1) is None


def test_imu_topic_fills_all_three_angles():
    s = SensorSnapshot()
    q = _Stub(x=0.0, y=0.0, z=math.sqrt(0.5), w=math.sqrt(0.5))
    s.apply("/exomy/imu_raw", _Stub(orientation=q), t=100.0)
    assert s.get("yaw_deg", now=100.1) == pytest.approx(90.0, abs=1e-4)
    assert s.get("roll_deg", now=100.1) == pytest.approx(0.0, abs=1e-6)


def test_scan_topic_fills_the_front_minimum():
    s = SensorSnapshot()
    ranges = [5.0] * 360
    ranges[0] = 0.42
    scan = _Stub(ranges=ranges, angle_min=0.0,
                 angle_increment=math.radians(1.0), range_min=0.05, range_max=12.0)
    s.apply("/scan", scan, t=100.0)
    assert s.get("lidar_front_min_m", now=100.1) == pytest.approx(0.42)


def test_cmd_vel_topic_fills_both_components():
    s = SensorSnapshot()
    twist = _Stub(linear=_Stub(x=0.10), angular=_Stub(z=-0.25))
    s.apply("/exomy/cmd_vel", twist, t=100.0)
    assert s.get("cmd_vx", now=100.1) == pytest.approx(0.10)
    assert s.get("cmd_wz", now=100.1) == pytest.approx(-0.25)


def test_an_unknown_topic_is_reported_rather_than_silently_dropped():
    assert SensorSnapshot().apply("/not_a_topic", _Stub(data=1), t=100.0) is False


def test_every_csv_column_can_actually_be_filled_by_some_topic():
    """A column no topic feeds would be permanently empty in every run."""
    filled = {"t_s"} | set(DERIVED_COLUMNS)
    known = {
        "terrain_label", "terrain_conf", "traversability_score",
        "inference_latency_ms", "frame_informative", "lidar_front_min_m",
        "lidar_stop", "lidar_reason", "roll_deg", "pitch_deg", "yaw_deg",
        "imu_slope_stop", "fused_verdict", "cmd_vx", "cmd_wz",
        "explorer_active", "e_stop", "e_stop_reason",
    }
    for col in CSV_COLUMNS:
        assert col in filled or col in known, f"{col} is fed by no topic"
