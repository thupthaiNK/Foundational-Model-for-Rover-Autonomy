"""
Purpose: Regression tests for real_hardware_deployment.launch.py's node
         wiring. Two things are pinned here:
         (1) Stuck detection. The real rover has no wheel encoders, so the
         odom-based stuck_detection_node.py/StuckDetectionFSM is a poor fit
         on real hardware -- real_stuck_detection_node.py (LiDAR front-
         sector-based, RealStuckDetectionFSM, max_wiggle_attempts default 4,
         includes the RETREAT safety state) is the real-hardware counterpart
         and must be the one this launch file starts.
         (2) Gyro odometry. /exomy/odom comes from gyro_odom_publisher_node,
         which was gated behind use_slam (off by default) until 2026-07-28.
         The reactive_explorer_node redesign made odom structurally required
         for TURN_TO_HEADING, so that node must start unconditionally.
         Parses the launch file's source as text rather than importing/
         executing it, since launch_ros/ament resolution isn't available
         outside a sourced ROS2 overlay.
Inputs:  None.
Outputs: pytest results.
How to run:
    cd ros2_ws && colcon build --packages-select exomy_ros2
    python3 -m pytest src/exomy_ros2/test/test_real_hardware_launch_wiring.py -v
Project: Onboard Visual Foundation Models for Mars Terrain Perception and Rover Navigation
"""
from pathlib import Path

LAUNCH_FILE = (
    Path(__file__).resolve().parents[1]
    / "launch" / "real_hardware_deployment.launch.py"
)


def _source() -> str:
    return LAUNCH_FILE.read_text()


def test_launches_real_stuck_detection_node_not_odom_based_one():
    source = _source()
    assert "real_stuck_detection_node.py" in source
    assert 'executable="stuck_detection_node.py"' not in source


def test_real_stuck_detection_max_wiggle_attempts_defaults_to_four():
    source = _source()
    idx = source.index("real_stuck_detection_node.py")
    # The node's own parameters dict follows shortly after its executable
    # line; max_wiggle_attempts must not be overridden back down to the
    # sim FSM's default of 3.
    window = source[idx:idx + 2000]
    assert '"max_wiggle_attempts": 3' not in window


def test_gyro_odom_publisher_starts_without_use_slam():
    # Regression, real hardware 2026-07-28: gyro_odom_publisher_node was
    # gated behind use_slam (off by default), so /exomy/odom never published
    # on a default launch. The 2026-07-27 reactive_explorer_node redesign
    # made odom structurally required -- TURN_TO_HEADING is a closed-loop
    # yaw maneuver that can never report completion without it, so the rover
    # sits in STARTUP_CHECK forever. gyro_odom only integrates the IMU's
    # gyro; it has no slam_toolbox dependency and must start on its own.
    source = _source()
    idx = source.index('executable="gyro_odom_publisher_node.py"')
    window = source[idx:idx + 600]
    assert 'LaunchConfiguration("use_slam")' not in window


def test_camera_driver_is_camera_ros_not_v4l2_camera():
    # Regression, real hardware 2026-07-28: the ExoMy camera is an IMX219
    # (Pi Camera v2) on CSI, driven by libcamera. /dev/video0 is unicam and
    # only offers Bayer (SRGGB10_CSI2P / SRGGB8), so v4l2_camera_node cannot
    # publish usable RGB from it -- it failed outright with "Failed opening
    # device /dev/video0". camera_ros drives libcamera properly and was
    # verified live on the rover at 15 fps.
    source = _source()
    assert 'package="camera_ros"' in source
    assert 'executable="camera_node"' in source
    # Only the launched package is pinned, not the string: the comment above
    # the node names v4l2_camera to explain why it was dropped, and that
    # explanation is worth more than a blanket ban on the word.
    assert 'package="v4l2_camera"' not in source


def test_camera_publishes_rgb888_not_the_default_planar_format():
    # camera_ros auto-selects NV21 when no format is given. NV21 is planar
    # YUV 4:2:0, so its buffer is 1.5 bytes per pixel, and
    # dinov2_terrain_node._image_callback reshapes to (h, w, -1) expecting an
    # interleaved 3-channel image. That reshape raises on the very first
    # frame. RGB888 is in the sensor's supported list and yields rgb8.
    source = _source()
    idx = source.index('package="camera_ros"')
    window = source[idx:idx + 1500]
    assert '"format": "RGB888"' in window


def _camera_node_window() -> str:
    """Just the camera_ros Node declaration.

    Bounded at exposure_lock rather than by a character count, because that
    action deliberately mentions AeEnable and would otherwise be read as part
    of the node's startup parameters -- which is the exact distinction these
    tests exist to pin.
    """
    source = _source()
    start = source.index('package="camera_ros"')
    end = source.find("exposure_lock = ", start)
    return source[start:end] if end != -1 else source[start:start + 2500]


def test_camera_does_not_disable_auto_exposure_at_startup():
    # Measured live on the rover 2026-07-28, and it overturns the obvious
    # reading of the 214-frame study. Disabling auto-exposure BEFORE the camera
    # streams does not freeze a good exposure, it freezes the driver's default
    # one, and on this sensor that default is far too short: every frame came
    # back at mean 0.2 out of 255, effectively black, in the lab's normal
    # lighting. Raising AnalogueGain does not rescue it either -- the driver
    # caps the parameter at 8.0 and even there the mean only reached 3.4.
    #
    # What the study actually did was set the controls with ros2 param set on a
    # camera that was already streaming, so auto-exposure had already converged
    # and turning it off froze the exposure time it had chosen. Reproduced live
    # in that order: detail 5.3 -> 55.2 with auto-exposure on -> stays at 55.2
    # after disabling it, mean 85, and DINOv2 classified soil and sand again.
    #
    # So the lock has to happen after the camera settles, not at startup. See
    # test_exposure_is_locked_after_the_camera_settles below.
    assert '"AeEnable"' not in _camera_node_window()


def test_camera_does_not_pin_analogue_gain_at_startup():
    # AnalogueGain is left to auto-exposure for the same reason. Whatever gain
    # it settles on is frozen by the same lock, and pinning 1.0 alongside it
    # only risks re-darkening the image after the exposure time is fixed. The
    # 214-frame study's value is not carried over, because that value was
    # chosen under a converged auto-exposure and is not independent of it.
    assert '"AnalogueGain"' not in _camera_node_window()


def test_camera_pins_the_two_bias_controls_at_startup():
    # These two are safe at startup because they only bias auto-exposure rather
    # than replace it, and pinning them stops a stale value surviving from a
    # previous session. Brightness in particular must be 0.0: a -1.0 left over
    # from an exposure bracket silently produced 217 fully black frames that the
    # probe still labelled soil at 0.913.
    #
    # ExposureTime is deliberately absent: this driver reports a nonsensical
    # min 39 / max 0 range for it and setting it does nothing useful.
    window = _camera_node_window()
    assert '"Brightness": 0.0' in window
    assert '"ExposureValue": 0.0' in window


def test_exposure_is_locked_after_the_camera_settles():
    # The other half of the fix above. Auto-exposure has to run first so it can
    # find a usable exposure time, and only then is it switched off, which
    # freezes that exposure for the rest of the mission. Without the freeze the
    # blank-frame gate is defeated within about a second: measured live, a
    # covered lens reads detail 1.6 and is correctly rejected, then climbs past
    # the 2.0 threshold as the camera ramps gain into its own sensor noise, so
    # the rejection releases while the lens is still covered. With the exposure
    # frozen the same covered lens sits at 0.58 and the rejection holds, which
    # was confirmed over 1061 consecutive frames.
    source = _source()
    assert "exposure_lock_delay_s" in source
    assert "TimerAction" in source
    # Checked as the command that actually runs rather than as separate
    # argv entries: since 2026-07-29 it is wrapped in a retry loop, because
    # a one-shot call raced DDS discovery and silently left the camera on
    # auto-exposure. See test_the_exposure_lock_retries_until_it_succeeds.
    assert "AeEnable false" in source


def test_camera_topics_are_remapped_from_the_private_names():
    # Regression, real hardware 2026-07-28, and the reason the full launch had
    # never once fed DINOv2. camera_ros publishes on PRIVATE topics, ~/image_raw
    # and ~/camera_info, so a remap keyed on the bare name "image_raw" matches
    # nothing at all. The node then published to /camera_node/image_raw while
    # dinov2_terrain_node sat on /camera/image_raw logging "No image" forever,
    # and every safety layer below it correctly held the rover at STOP.
    #
    # It survived because every successful test until now started camera_node by
    # hand, where the node keeps its default name "camera" and ~/image_raw
    # expands to /camera/image_raw on its own. The launch file renames the node,
    # which breaks that and exposes the dead remap.
    #
    # Verified live on the rover: with these keys, ros2 topic list shows
    # /camera/image_raw and /camera/camera_info. With the bare keys it shows
    # /camera_node/image_raw instead.
    window = _camera_node_window()
    assert '"~/image_raw", "/camera/image_raw"' in window
    assert '"~/camera_info", "/camera/camera_info"' in window
    assert '("image_raw", ' not in window
    assert '("camera_info", ' not in window


def test_camera_brightness_is_never_left_at_the_bracket_value():
    # A Brightness of -1.0 left over from an exposure bracket silently
    # produced 217 fully black frames that the probe still labelled soil at
    # 0.913. Anything that pins Brightness must pin it to 0.0.
    assert '"Brightness": -1.0' not in _source()


# ── Cross-node geometry invariants (2026-07-29) ──────────────────────────
# reactive_explorer_node and lidar_proximity_guard_node each declare their own
# copy of the rover's geometry and do not share ROS parameters. Nothing at
# runtime checks that the copies agree, and every one of these mismatches has
# a specific failure mode that only shows up on hardware.

def _node_block(node_marker: str) -> str:
    """The text of one Node(...) declaration, from its executable line to the
    start of the next node. Bounded by structure rather than a character
    count, which silently truncated once the parameter comments grew."""
    source = _source()
    idx = source.index(node_marker)
    rest = source[idx:]
    end = rest.find(" = Node(")
    return rest if end == -1 else rest[:end]


def _param(node_marker: str, key: str) -> float:
    window = _node_block(node_marker)
    marker = f'"{key}":'
    start = window.index(marker) + len(marker)
    end = window.index(",", start)
    return float(window[start:end].strip())


def test_guard_stop_distance_is_inside_the_planner_lookahead():
    # If the hard stop fires at or before the planner's corridor check, the
    # rover halts the instant anything enters the corridor and never turns:
    # avoidance becomes unreachable code. Observed live in Trial A, where
    # both sat at 0.40 m.
    window = _node_block("reactive_explorer_node.py")
    start = window.index('"lookahead_tiers_m": [') + len('"lookahead_tiers_m": [')
    lookahead = min(float(v) for v in window[start:window.index("]", start)].split(","))
    stop = _param("lidar_proximity_guard_node.py", "stop_distance_m")
    assert stop < lookahead, (
        f"guard stops at {stop} m but the planner only checks {lookahead} m ahead"
    )


def test_guard_resume_distance_is_above_its_stop_distance():
    stop = _param("lidar_proximity_guard_node.py", "stop_distance_m")
    resume = _param("lidar_proximity_guard_node.py", "resume_distance_m")
    assert resume > stop


def test_guard_half_width_matches_the_planner_swept_width():
    # The guard measures down the same rectangle the planner plans through. A
    # narrower guard box would refuse to drive headings the planner committed
    # to; a wider one would stop for obstacles the planner correctly ignored.
    width = _param("reactive_explorer_node.py", "rover_width_m")
    margin = _param("reactive_explorer_node.py", "lateral_margin_m")
    guard_half = _param("lidar_proximity_guard_node.py", "half_width_m")
    assert abs(guard_half - (width / 2.0 + margin)) < 1e-9


def test_lidar_yaw_offset_matches_across_both_nodes():
    # Since the guard became direction-aware it needs to know where the
    # rover's front is. Two different offsets means the two layers watch two
    # different directions.
    planner = _param("reactive_explorer_node.py", "lidar_yaw_offset_deg")
    guard = _param("lidar_proximity_guard_node.py", "lidar_yaw_offset_deg")
    assert planner == guard


def test_min_ignore_matches_across_both_nodes():
    planner = _param("reactive_explorer_node.py", "min_ignore_m")
    guard = _param("lidar_proximity_guard_node.py", "min_ignore_m")
    assert planner == guard


def test_heading_step_is_finer_than_the_arrival_tolerance():
    # With a step coarser than the tolerance the nearest heading the picker
    # can offer is always outside tolerance, so the FSM turns, "arrives" on
    # gyro drift, re-scans, and proposes the same heading again. Nine such
    # cycles in 20 s were logged on 2026-07-29 with a 5 deg step and a 3 deg
    # tolerance.
    step = _param("reactive_explorer_node.py", "angle_step_deg")
    tolerance = _param("reactive_explorer_node.py", "angle_tolerance_deg")
    assert step < tolerance


def test_min_correction_turn_is_wider_than_the_arrival_tolerance():
    # 2026-08-04, hardware round 4/5 of the H5 follow-up confirmation
    # testing: the heading picker's "smallest deviation that clears" search
    # let the rover nudge open-loop-small corrections repeatedly while
    # approaching a wall, drifting into it at a shallow angle instead of
    # turning away. min_correction_turn_deg biases pick_heading_tiered
    # towards a heading at least this far from straight ahead. Same class of
    # bug as test_heading_step_is_finer_than_the_arrival_tolerance above: if
    # the enforced minimum were not wider than angle_tolerance_deg, the FSM
    # could "arrive" on a turn too small to have actually corrected anything.
    minimum = _param("reactive_explorer_node.py", "min_correction_turn_deg")
    tolerance = _param("reactive_explorer_node.py", "angle_tolerance_deg")
    assert minimum > tolerance


def test_terrain_stale_timeout_allows_for_real_dinov2_latency():
    # DINOv2 measured 2200 ms per frame with the full stack running on the
    # Pi. A 3 s watchdog expired between almost every pair of inferences.
    timeout = _param("terrain_controller_node.py", "stale_timeout_s")
    assert timeout >= 3 * 2.2


def test_no_reverse_motion_parameters_remain_for_the_explorer():
    # SLOPE_RETREAT was removed on 2026-07-29; the explorer has no reverse
    # path at all. real_stuck_detection_node keeps its own retreat_speed,
    # which is a different node and a deliberate, bounded recovery.
    window = _node_block("reactive_explorer_node.py")
    for gone in ("retreat_speed", "retreat_distance_m", "max_retreat_cycles"):
        assert f'"{gone}":' not in window


def test_sweep_is_disabled_when_the_camera_is_off():
    # The opening sweep samples DINOv2 at each of eight stops. With
    # use_camera:=false nothing publishes /traversability_score, so the SAMPLE
    # phase waits for a score that never comes and the stack fails safe and
    # shuts itself down after max_turn_duration_s. use_camera:=false is the
    # documented elevated bench-test recipe and has to keep working.
    window = _node_block("reactive_explorer_node.py")
    start = window.index('"sweep_headings":')
    decl = window[start:start + 400]
    assert "use_camera" in decl, "sweep_headings must depend on use_camera"


def test_turn_radius_covers_the_rover_corner_sweep():
    # A point turn sweeps hypot(front offset, half width). The forward
    # corridor is a rectangle and does not constrain rotation at all, so this
    # is the only thing standing between a turn and a corner strike.
    import math
    turn_radius = _param("reactive_explorer_node.py", "turn_radius_m")
    half_width = _param("reactive_explorer_node.py", "rover_width_m") / 2.0
    corner = math.hypot(0.20, half_width)      # 0.20 m LiDAR to front wheel
    assert turn_radius >= corner


def test_terrain_confirm_wait_is_bounded():
    timeout = _param("reactive_explorer_node.py", "terrain_confirm_timeout_s")
    assert 0.0 < timeout <= 60.0


def test_sllidar_include_is_told_which_serial_port_to_open():
    # Regression, real hardware 2026-07-29 (Trial A, second attempt): the
    # sllidar include never forwarded serial_port, so sllidar fell back to its
    # own default of /dev/ttyUSB0. Inside the container that path does not
    # exist -- run_container.sh passes the LiDAR through as --device
    # /dev/rplidar, the stable udev name, and Docker creates the node at that
    # path and no other. sllidar died at startup with SDK error 80008004, no
    # /scan was ever published, and reactive_explorer sat in STARTUP_CHECK
    # waiting for a scan that could not arrive. The rover did nothing for a
    # minute and the whole run was lost.
    #
    # docker/fix_ttyusb0_symlink.sh papers over this by symlinking
    # /dev/ttyUSB0 -> /dev/rplidar inside the container, but the symlink dies
    # with every container restart and has to be remembered every single time.
    # Passing the port through the include fixes it at the source.
    source = _source()
    idx = source.index("_sllidar_launch_path")
    window = source[idx:idx + 1500]
    assert "serial_port" in window, (
        "the sllidar include must forward serial_port, or sllidar opens its "
        "own default /dev/ttyUSB0, which does not exist in the container"
    )
    assert "/dev/rplidar" in window


def test_the_sweep_is_bounded_and_logged_not_silent():
    # Trial A, 2026-07-29 evening: the sweep itself worked perfectly and the
    # rover then sat still for 573 s. Three launch values decide whether that
    # can recur, so they are pinned rather than left to a comment.
    window = _node_block("reactive_explorer_node.py")
    rejections = _param("reactive_explorer_node.py", "max_terrain_rejections")
    # 2026-08-04: raised 3 -> 20 (backstop only) once terrain_search_timeout_s
    # became the real liveness bound instead of a rejection count -- see
    # test_terrain_confirm_wait_is_bounded's sibling for the timeout side.
    # Still pinned with an upper bound so a further, unexamined bump does not
    # silently reopen the 573 s hang this test was written to catch.
    assert 0 < rejections <= 25, "a refused heading must cost something"
    lead = _param("reactive_explorer_node.py", "turn_stop_lead_s")
    assert lead > 0.0, (
        "the rover overran every commanded turn by ~20 deg and cannot be told "
        "to turn slowly, so the turn must stop early by the measured rate"
    )
    assert '"sweep_samples_per_heading": 1' in window


def test_sweep_stop_count_is_four():
    # Reduced from 8 deliberately: 4 stops of 90 deg against a ~62 deg field
    # of view leave gaps in terrain coverage, accepted because the LiDAR still
    # vetoes geometry over the full circle and the thesis question is whether
    # the foundation model can pick a heading, not how finely.
    window = _node_block("reactive_explorer_node.py")
    start = window.index('"sweep_headings":')
    decl = window[start:start + 400]
    assert "4 if" in decl


def test_the_bag_records_classified_frames_not_the_raw_camera_stream():
    # 2026-07-29: recording /camera/image_raw uncompressed gave an 11 GB bag
    # for one 573 s run in which the rover never drove, and three earlier runs
    # had between them filled 28 GB of the card. The frame behind each DINOv2
    # verdict is the evidence worth keeping, and there is one per inference.
    source = _source()
    idx = source.index('"ros2", "bag", "record"')
    window = source[idx:idx + 2500]
    assert '"/terrain_classified_image"' in window
    assert '"/camera/image_raw"' not in window


def test_the_bag_records_whether_each_frame_carried_information():
    # Needed to read any run back honestly: /traversability_score is 1.0 both
    # for an impassable rock and for a frame the blank-frame gate rejected, so
    # without this the recorded scores cannot be interpreted.
    source = _source()
    idx = source.index('"ros2", "bag", "record"')
    window = source[idx:idx + 2500]
    assert '"/terrain_frame_informative"' in window


def test_the_exposure_lock_retries_until_it_succeeds():
    # 2026-07-29: the same one-shot `ros2 param set` succeeded on one run and
    # died with "Node not found" on the next two, because discovering
    # camera_node's parameter service takes about 4 s. Nothing failed loudly --
    # the camera simply stayed on auto-exposure, which makes the sweep's four
    # DINOv2 scores incomparable with each other.
    source = _source()
    idx = source.index("exposure_lock = TimerAction")
    window = source[idx:idx + 1800]
    assert "AeEnable" in window
    assert "seq 1 30" in window, "the lock must retry, not fire once"
    assert "FAILED" in window, "a lock that never takes must say so"


def test_the_imu_slope_override_is_off_on_real_hardware():
    # Measured, not preferred. Trial A run 2's bag shows a live 50 Hz IMU whose
    # accelerometer was swamped by chassis vibration: |a| 0.521-2.306 g against
    # a true 1.000 g, apparent tilt 10-74 deg on a flat floor. accel_gate_g
    # gates magnitude and not direction, so tilt=53.04 deg at |a|=0.979 g passes
    # it. A false over-tilt forces a heading re-decision and run 2 ended in
    # FAILSAFE(boxed_in) because of one.
    window = _node_block("imu_slope_fusion_node.py")
    assert '"slope_override_enabled": False' in window


def test_the_imu_still_publishes_tilt_with_the_override_off():
    # The limitation has to stay measurable: the node is not disabled, only its
    # veto is. Chapter 5 quotes these numbers.
    source = _source()
    assert "imu_slope_fusion_node.py" in source
    assert 'condition=IfCondition(LaunchConfiguration("use_imu"))' in source


def test_a_heading_must_have_room_to_travel_on_real_hardware():
    # Observed repeatedly in the lab and the sandpit and visible in the logged
    # sweep tables: with clearance used only as a tie-break, the rover chose
    # directions it could barely move in. The sand run of 2026-07-30 took +0 at
    # 0.60 m over +180 at 2.18 m and drove into the pit wall.
    run = _param("reactive_explorer_node.py", "min_run_m")
    assert run > 0.0, "clearance must eliminate headings, not merely break ties"
    # Must stay under the shortest run actually on offer in the recorded
    # sweeps, or every heading is vetoed and the fallback carries every run.
    assert run <= 1.2


def test_openness_is_weighted_against_the_terrain_score_on_real_hardware():
    # Derived from the five sweep tables recorded on hardware: at weight 0 the
    # terrain score decided all five and chose a heading the user judged wrong
    # in three, including the 0.60 m direction that ended in the sandpit wall.
    # 0.4 to 1.0 fixes all three while the score still changes two of five.
    weight = _param("reactive_explorer_node.py", "openness_weight")
    assert 0.4 <= weight <= 1.0, (
        "outside the plateau measured on hardware: below it the rover drives "
        "into things, above it geometry decides everything"
    )


def test_the_clearance_horizon_bounds_the_openness_term():
    # An 'inf' return means "nothing within the horizon", not "infinitely
    # good", so the openness term has to be clamped or one unbounded reading
    # wins every sweep it appears in.
    horizon = _param("reactive_explorer_node.py", "clearance_horizon_m")
    assert horizon > 0.0
