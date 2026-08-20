#!/usr/bin/env python3
"""
Purpose: Reactive exploration ("bug algorithm") recovery behaviour for ExoMy.
         REDESIGNED 2026-07-27 after a real wall collision during ground
         testing: the rover no longer ever drives backward blind. Every
         "which way do I go" decision -- at boot AND whenever blocked mid-
         drive -- uses the SAME mechanism:
           1. STARTUP_CHECK: on boot (and again whenever re-triggered), hold
              still for settle_time_s so LiDAR/IMU readings aren't corrupted
              by leftover motion, then scan the FULL 360 degrees via
              pick_heading_tiered() (avoidance_planner.py). It tries a
              generous lookahead tier first (1.0m) and only relaxes to a
              tighter tier (0.3m) if nothing at all clears the full circle at
              the looser one -- never picks a looser-tier heading when a
              stricter one exists.
           2. If a heading is found, rotate to it in place (TURN_TO_HEADING)
              -- point-turns don't translate the chassis, so no rear-facing
              sensor data is ever needed to do this safely, unlike the old
              retreat-then-turn design (removed 2026-07-29, see below).
           3. Once facing that heading, drive forward (handled by
              terrain_controller_node once this node cedes cmd_vel back to
              MONITORING).
           4. If NO heading clears even the loosest tier anywhere in the full
              circle, enter FAILSAFE immediately with reason "boxed_in".
         Camera-based terrain confirmation (SAFE_LABELS/is_drivable_label)
         is currently NOT consulted before committing to a heading --
         dinov2_terrain_node cannot run on the Pi until torch is installed
         (2026-07-27 finding), so terrain data defaults to "uncertain" and
         would permanently boxed_in every real-hardware run. TODO once torch
         is installed: re-add a terrain confirm step after TURN_TO_HEADING,
         mirroring the old SCAN_AVOID_CONFIRM design, instead of handing
         cmd_vel back to MONITORING unconditionally.
         A second trigger, IMU over-tilt (/imu_slope_stop, from
         imu_slope_fusion_node.py), preempts everything else -- including a
         turn-to-heading already in progress -- and is handled exactly like a
         blocked corridor: stop, re-scan, turn towards a clear heading.
         REVERSE MOTION WAS REMOVED ENTIRELY on 2026-07-29. Over-tilt used to
         enter a SLOPE_RETREAT state that backed straight away; that was the
         only path in the whole stack that drove the rover backwards, the
         rover has no rear-facing sensing beyond this same LiDAR, and a blind
         reverse is what put it into a wall on 2026-07-27. It was also firing
         on tilt readings that were never real -- Trial A logged apparent
         tilts up to 62.9 deg on a flat lab floor, because an accelerometer-
         only estimate cannot separate gravity from acceleration (fixed
         separately in imu_slope_fusion_node.gate_tilt). The IMU keeps its
         veto over WHERE the rover goes and loses its ability to command a
         manoeuvre of its own.
         The core state machine (ReactiveExplorerFSM) is pure Python with no
         rclpy dependency, fully unit-testable with synthetic input -- see
         test/test_reactive_explorer.py.
         Also cedes cmd_vel to stuck_detection_node.py whenever
         /stuck_detection/active is true -- takes priority over this node's
         own active flag too, since a turn-to-heading manoeuvre can itself
         get stuck in sand just as much as normal forward driving.
         /scan staleness (no message for stale_timeout_s) is now fatal in
         EVERY state, not just a fallback to fixed candidate headings --
         2026-07-27 finding: a corridor check built entirely from an old
         frame could report "clear" when the real world has moved on, which
         is worse than refusing to decide at all. /exomy/odom staleness
         still only aborts states that already committed to a closed-loop
         maneuver (TURN_TO_HEADING); STARTUP_CHECK/MONITORING
         just wait for odom to return since they haven't committed yet.
         Once FAILSAFE is entered, this node calls its configured shutdown
         function (default: SIGINT to its own process group, the same
         signal a manual Ctrl+C on `ros2 launch` sends) so the ENTIRE
         real-hardware stack stops -- motors, other nodes, and the rosbag
         recording all shut down together -- rather than leaving the rover
         idling with other nodes still live. The failsafe reason is logged
         at ERROR severity before the signal is sent, so it is the last
         line in the terminal/log file.
Inputs:  /terrain_controller/stopped (std_msgs/Bool)  terrain-hazard stop flag
         /lidar_proximity_stop (std_msgs/Bool)        LiDAR proximity stop flag
         /scan (sensor_msgs/LaserScan)                 for corridor-aware avoidance
         /imu_slope_stop (std_msgs/Bool)              over-tilt stop flag
         /stuck_detection/active (std_msgs/Bool)      true while stuck_detection_node owns cmd_vel
         /exomy/odom (nav_msgs/Odometry)               yaw + position
Outputs: /exomy/cmd_vel (geometry_msgs/Twist)          only while active
         /reactive_explorer/active (std_msgs/Bool)     true while this node owns cmd_vel
         /reactive_explorer/failsafe (std_msgs/Bool)   true once boxed in or otherwise unrecoverable
         /reactive_explorer/failsafe_reason (std_msgs/String)
             human-readable reason(s), joined with "; " if more than one
             condition applies at once (e.g. "over_tilted; boxed_in")
         /reactive_explorer/commanding_linear_motion (std_msgs/Bool)
             true only while this node commands nonzero linear velocity, for
             lidar_proximity_guard_node's suppression logic. Always false
             since reverse motion was removed on 2026-07-29 -- this node now
             only ever rotates in place or holds still -- but kept so the
             guard's contract does not depend on that staying true
How to run:
    source /opt/ros/humble/setup.bash
    source ros2_ws/install/setup.bash
    ros2 run fm_perception reactive_explorer_node.py
Project: Onboard Visual Foundation Models for Mars Terrain Perception and Rover Navigation
"""
import math
import os
import signal

from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rcl_interfaces.msg import ParameterDescriptor, ParameterType
from sensor_msgs.msg import Imu, LaserScan
from std_msgs.msg import Bool, Float64, String
import rclpy
from rclpy.node import Node

from fm_perception.lidar_proximity_guard_node import (
    apply_angle_mask, min_valid_range,
)
from fm_perception.avoidance_planner import (
    corridor_box_radius_m,
    corridor_clearance_m,
    is_corridor_clear,
    near_returns,
    pick_heading_tiered,
)
from fm_perception.lidar_proximity_guard_node import parse_blocked_sectors


# ── Pure geometry helpers (no rclpy dependency, unit-testable) ─────────────

def normalize_angle(angle: float) -> float:
    """Wrap an angle in radians to (-pi, pi]."""
    return math.atan2(math.sin(angle), math.cos(angle))


def yaw_from_quaternion(x: float, y: float, z: float, w: float) -> float:
    """Extract yaw (rotation about Z) from a quaternion, REP103 convention."""
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def angle_delta(target: float, current: float) -> float:
    """Shortest signed angular distance from current to target, in (-pi, pi]."""
    return normalize_angle(target - current)


def distance_2d(x0: float, y0: float, x1: float, y1: float) -> float:
    return math.hypot(x1 - x0, y1 - y0)


def should_shutdown_now(failsafe: bool, failsafe_since_s, now_s: float,
                        grace_s: float) -> bool:
    """True once the rover has been commanded to a stop for grace_s.

    FAILSAFE tears down the whole launch so the terminal stops too, but the
    ORDER matters: motor_node's stopMotors() only runs from its shutdown
    handler, and if that signal is late or lost the servo HAT just holds the
    last PWM command -- the exact 2026-07-27 wall-collision failure mode.
    So the node first spends grace_s publishing a zero Twist (a stop that
    does not depend on any shutdown path working), and only then signals.
    """
    if not failsafe or failsafe_since_s is None:
        return False
    return (now_s - failsafe_since_s) >= grace_s


def default_shutdown(os_module=None) -> None:
    """Send SIGINT to this process's entire group -- what a manual Ctrl+C on
    `ros2 launch` does, so every node (motor_node included, whose
    destroy_node() calls stopMotors()) and the rosbag recorder all shut down
    gracefully.

    Must be killpg, not kill: os.kill(os.getpgid(0), ...) targets only the
    single process whose PID equals the group id. Verified experimentally
    2026-07-27 -- sibling processes survived it, i.e. every other node in the
    launch kept running with the last motor command latched.

    os_module is injectable so the test can assert which call is made without
    signalling the test runner's own process group.
    """
    _os = os_module or os
    _os.killpg(_os.getpgid(0), signal.SIGINT)


# Mirrors POLICY in terrain_controller_node.py: only these labels drive.
SAFE_LABELS = frozenset({"soil", "sand", "bedrock"})


def is_drivable_label(label: str) -> bool:
    return label in SAFE_LABELS


def correction_min_offset_rad(forward_corridor_blocked: bool,
                              min_correction_turn_rad: float) -> float:
    """The min_offset_rad to hand pick_heading_tiered() this tick.

    2026-08-04, hardware round 6 regression: min_correction_turn_deg (round
    4/5's fix for shallow-angle wall approaches) was first wired to bias
    EVERY heading pick unconditionally, including STARTUP_CHECK's routine
    "is straight ahead still fine" recheck at the end of every turn. With a
    fully open circle, nearly everywhere clears >= the minimum, so that
    recheck could no longer see 0.0 (straight ahead, already fine) as an
    acceptable answer -- the rover looped STARTUP_CHECK <-> TURN_TO_HEADING
    for 76+ seconds on real hardware and never drove forward once.

    The minimum must only bias the search when there is an actual reason to
    turn -- the forward corridor is blocked -- never when the current
    heading is already fine. Terrain-rejection re-picks and the routine
    post-turn recheck both go through the same single picker call in
    _tick(), so this stays 0.0 for them by design; they are not the
    shallow-drift failure mode this was written for.
    """
    return min_correction_turn_rad if forward_corridor_blocked else 0.0


# ── FSM states ───────────────────────────────────────────────────────────

STATE_STARTUP_CHECK       = "STARTUP_CHECK"
STATE_STARTUP_SWEEP       = "STARTUP_SWEEP"
STATE_MONITORING          = "MONITORING"
STATE_TURN_TO_HEADING     = "TURN_TO_HEADING"
STATE_FAILSAFE            = "FAILSAFE"


class ReactiveExplorerFSM:
    """Pure, ROS-independent bug-algorithm state machine.

    Call step() once per control tick with the latest known sensor values;
    it returns the commanded velocity and node status as a dict. No I/O, no
    rclpy dependency -- fully deterministic and unit-testable.
    """

    def __init__(self, stuck_dwell_s=3.0,
                 angle_tolerance_rad=math.radians(3.0),
                 angular_speed=0.3, max_turn_duration_s=20.0,
                 settle_time_s=1.0, sweep_headings=8,
                 sweep_samples_per_heading=2, terrain_reject_margin=0.15,
                 terrain_reject_floor=0.6,
                 terrain_confirm_timeout_s=15.0, turn_radius_m=0.30,
                 turn_stop_lead_s=0.0, max_terrain_rejections=20,
                 terrain_search_timeout_s=180.0,
                 min_run_m=0.0, openness_weight=0.0,
                 clearance_horizon_m=3.0):
        self.stuck_dwell_s = stuck_dwell_s
        self.angle_tolerance_rad = angle_tolerance_rad
        self.angular_speed = angular_speed
        # Wall-clock deadlines, independent of odom availability -- defense
        # in depth against a turn/leg that never converges for any other
        # reason (e.g. a jammed wheel), even with live odom.
        self.max_turn_duration_s = max_turn_duration_s
        # How long to hold still and ignore sensor input at boot (and at
        # every subsequent STARTUP_CHECK re-entry) before trusting LiDAR/IMU
        # readings -- protects against vibration/motion artefacts left over
        # from whatever just happened (e.g. a turn that just completed).
        self.settle_time_s = settle_time_s
        # Opening sweep: rotate through this many equally spaced stops,
        # sampling DINOv2 at each, before committing to a first heading.
        # Sized by the CAMERA, not the LiDAR -- the LiDAR already returns the
        # full circle in one frame and gains nothing from rotating, but the
        # imx219 sees about 62 deg, so a stationary rover has terrain data for
        # roughly a sixth of its surroundings. Eight stops of 45 deg cover the
        # circle with overlap to spare. 0 disables the sweep entirely.
        self.sweep_headings = sweep_headings
        self.sweep_samples_per_heading = sweep_samples_per_heading
        # How much worse than the sweep's baseline a heading's terrain may be
        # before it is refused. Relative because absolute is unusable here:
        # the same camera and model called a wall soil:0.937 across a
        # 214-frame real capture, so a fixed cut-off either passes everything
        # or fails everything, whereas an ordering only needs the model to
        # rate one surface worse than another. Ground is never uniform, so
        # some slack above the baseline is required or the rover blacklists
        # the whole world one heading at a time.
        self.terrain_reject_margin = terrain_reject_margin
        # Floor under that threshold, added 2026-08-04. The baseline tracks the
        # ground the rover is standing on and has a ceiling but nothing holding
        # it up, so a stretch of soil (CLASS_RISK 0.0) drags it to almost zero
        # and baseline+margin becomes an absolute ~0.15. Every terrain class
        # except soil scores above that -- sand is 0.5 -- so the rover ends up
        # refusing the surface it has been driving on all along. That is what
        # happened across six sandpit runs on 2026-08-04: baselines of
        # 0.003/0.034/0.134, thresholds of 0.153/0.184/0.284, and a 0.525 sand
        # reading refused at the last of them, ending the run.
        #
        # 0.6 sits between sand (0.5) and bedrock (0.7), so soil and sand
        # always pass and bedrock and big_rock/uncertain (1.0) are still
        # refused -- the hazard ordering H5 was run to demonstrate is
        # untouched. Only ever raises the threshold: on risky ground the
        # adaptive baseline is stricter and keeps winning.
        self.terrain_reject_floor = terrain_reject_floor
        # Every other state carries a wall-clock deadline; without one here a
        # dead DINOv2 leaves the rover holding cmd_vel at zero forever,
        # neither recovering nor failing safe, with the rosbag still running.
        self.terrain_confirm_timeout_s = terrain_confirm_timeout_s
        # Radius the rover's corners sweep during a point turn: hypot(front
        # offset 0.20, half width 0.16) = 0.256 m, plus margin. The forward
        # corridor is a rectangle and correctly lets the rover drive PAST a
        # wall 0.25 m to its side, but rotating beside that same wall grinds
        # the corner into it. The pre-2026-07-29 direction-blind guard covered
        # this by accident; the box does not cover it at all, and this design
        # turns eight times before it even starts driving.
        self.turn_radius_m = turn_radius_m
        # How far ahead of the target to stop commanding rotation, expressed
        # as time rather than angle so it scales with how fast the chassis
        # happens to be turning. Measured 2026-07-29: every commanded 45 deg
        # sweep stop overran by about 20 deg, consistently, because the rover
        # cannot be told to turn slowly. RoverCommand.vel is not rad/s, and
        # vel 20/25/50 were all measured at the same wheel speed, so the
        # servos saturate -- a commanded 0.15 rad/s produced 40-90 deg/s. The
        # only lever left is stopping sooner. 0.0 keeps the old behaviour.
        self.turn_stop_lead_s = turn_stop_lead_s
        # A heading whose terrain is refused costs one of these. Without a
        # bound the rover rejects, re-picks, rejects again forever: the
        # confirm's timeout only covers "no score is arriving", and a score
        # that arrives and is refused resets that clock. Trial A sat still for
        # 573 s that way, silently, with the bag growing to 11 GB.
        #
        # Raised from 3 to 20 on 2026-08-04. Three was chosen when this was the
        # ONLY bound on the search, and it turned out to bound the wrong thing.
        # Exhausting the picker on a clear circle takes 15 refusals, measured
        # (test_pick_heading_tiered_starves_once_its_own_offers_are_all_excluded):
        # the picker returns the smallest-deviation heading that is not already
        # excluded, so each refusal pushes the frontier out by only
        # exclude_tolerance + angle_step = 25 deg, not by the 40 deg an
        # exclusion covers. Anything at or below 15 makes this count the real
        # bound again -- which is what stopped six sandpit runs on 2026-08-04
        # after one or two refusals, with whole arcs of the compass never
        # looked at. The search now normally ends by exhausting the picker
        # (boxed_in), the honest "there is nowhere to go"; this count survives
        # only as a backstop for a picker that somehow keeps offering headings.
        self.max_terrain_rejections = max_terrain_rejections
        # What actually bounds the search now. Wall clock rather than count,
        # because the property worth keeping from the 573 s incident is "this
        # hold has a deadline like every other hold in the FSM", not "this hold
        # is allowed exactly N attempts". Starts on the first refusal of a run
        # and clears the moment a heading is accepted.
        self.terrain_search_timeout_s = terrain_search_timeout_s
        self._rejection_search_start_s = None
        self._terrain_rejections = 0
        # True while the heading the rover is turning to is one DINOv2 chose
        # itself. The sweep IS the model deciding: it ranks every direction the
        # camera can see and commits to the best. Putting that same heading
        # through a numeric threshold on arrival can only overturn the ranking,
        # and a threshold is the one thing this stack established it cannot use
        # -- the 214-frame study found the model calling a wall soil:0.937.
        # DINOv2 also keeps a continuous veto over motion regardless, since
        # terrain_controller_node owns cmd_vel whenever this node is idle and
        # applies its per-class speed policy every frame. The confirm earns its
        # place only on a heading the LiDAR picked, which sees geometry and no
        # terrain at all.
        self._heading_chosen_by_model = False
        # Set when a heading has just been refused. The rejection blacklists the
        # yaw and the picker does then offer a different one, but acting on it
        # needs a tick to pass so the exclusion is in the offer -- and it needs
        # the FSM to reach the turn at all, which the confirm's hold used to
        # prevent. Trial A run 3 spent all three rejections re-measuring one
        # unchanged view at yaw -130.
        self._turn_after_rejection = False
        # How far a heading must be able to travel before the sweep will
        # consider it at all. The LiDAR's other question -- can the rover fit
        # through the first lookahead_tiers_m -- says nothing about whether
        # there is anywhere to go, so every heading in an open room passes it
        # and clearance survived only as a tie-break. Across several runs in
        # both the lab and the sandpit the rover kept choosing directions with
        # almost no room: the sand run of 2026-07-30 took +0 at 0.60 m over
        # +180 at 2.18 m.
        #
        # Expressed as a stricter veto rather than as a weight on the score,
        # deliberately. Weighting would need an exchange rate between metres
        # and a traversability score, and would let geometry overrule the
        # model. This way geometry still only eliminates and DINOv2 still
        # chooses. 0.0 disables it, which is the pre-2026-07-30 behaviour every
        # reported Gazebo result was produced under.
        self.min_run_m = min_run_m
        # Exchange rate between how open a heading is and how good its terrain
        # looks. Cost is score + openness_weight * (1 - min(clearance,
        # horizon)/horizon), so both terms are dimensionless.
        #
        # Chosen from the five sweep tables recorded on hardware rather than
        # guessed. At 0 the model decided all five and picked a heading the user
        # judged wrong in three, including the 0.60 m direction in the sandpit
        # that ended in the pit wall. Anything from 0.4 to 1.0 fixes all three
        # while the score still changes the outcome in two of the five; above
        # 1.5 the model's influence falls to one and then none. 0.5 is the
        # middle of that plateau, not its edge.
        #
        # Honest limits: n=5, and the "right" heading is a human judgement by
        # eye rather than a measured outcome. Enough to set a parameter, not
        # enough to call it optimal. It is also tuned to a cluttered indoor lab,
        # which is the opposite of the target domain -- see
        # sweep_model_changed_the_choice for the measurement that keeps this
        # honest.
        self.openness_weight = openness_weight
        # Distance past which extra room stops counting. An 'inf' return means
        # "nothing within the horizon", not "infinitely good", so clamping here
        # stops one unbounded reading winning every sweep it appears in.
        self.clearance_horizon_m = clearance_horizon_m
        # How far the baseline may drift upward as the rover accepts ground.
        # Tracking the current ground is what makes the check a comparison
        # rather than a threshold, but without a ceiling each acceptance would
        # raise the bar and the rover would walk its own standard up until it
        # accepted anything. The sweep measured what this environment actually
        # offers, so its worst passable direction is the limit.
        self.terrain_baseline_ceiling = None
        # Rejections, for the node to log. Each is (yaw_deg, mean_score,
        # threshold) -- the numbers needed to tell "the ground really is bad"
        # apart from "the baseline is wrong", which is not obvious live.
        self.rejection_log = []
        self._yaw_rate = 0.0
        self._last_step_s = None

        # Filled in by the sweep and then read by the node for logging and by
        # the mid-drive terrain check. sweep_report is one entry per stop;
        # terrain_baseline is the median score across them, which is what
        # makes the mid-drive check relative to THIS environment rather than
        # to a threshold calibrated on a different domain.
        self.sweep_report = []
        self.sweep_choice = None
        # What geometry alone would have picked, and whether the terrain score
        # changed the answer. Recorded so the log can state how often the
        # foundation model actually decided the heading instead of the thesis
        # asserting that it did.
        self.sweep_geometry_choice = None
        self.sweep_model_changed_the_choice = None
        # The yaw the sweep started from, kept separately because _origin_yaw
        # is reassigned on every later heading decision. Needed to report each
        # stop as an offset from where the sweep began.
        self.sweep_origin_yaw = 0.0
        self.terrain_baseline = None
        self._sweep_index = 0
        self._sweep_phase = "TURN"
        self._sweep_scores = []
        self._sweep_last_seq = None
        self._sweep_sample_yaw = 0.0
        self._sweep_done = False
        self._confirm_scores = []
        self._confirm_last_seq = None
        self._confirm_discard_pending = True
        self._confirm_wait_start_s = None
        self._last_seen_score_seq = 0
        self._turn_clearance_m = float("inf")

        self.state = STATE_STARTUP_CHECK
        self.active = True
        self.failsafe = False
        self.failsafe_reason = ""
        self.failsafe_since_s = None

        self._stuck_since_s = None
        self._origin_yaw = 0.0
        self._target_yaw = 0.0
        self._now_s = 0.0
        self._state_deadline_s = float("inf")
        self._settle_start_s = None
        # Absolute odom yaws whose TERRAIN was refused. Stored absolute (not
        # as offsets) so they stay pinned to the same patch of world as the
        # rover turns; excluded_offsets() converts to the current frame for
        # the heading picker, which is LiDAR-only and would otherwise keep
        # re-proposing ground DINOv2 has already rejected.
        self._rejected_yaws = []
        self._last_yaw = 0.0

    # ── public API ──────────────────────────────────────────────────────

    def step(self, now_s, yaw, pos_x, pos_y, terrain_stopped, lidar_stopped,
              imu_slope_stop=False, forward_corridor_blocked=False,
              heading_offset=None,
              odom_fresh=True, imu_fresh=True, scan_fresh=True,
              scan_ever_received=True,
              traversability_score=None, terrain_score_seq=None,
              forward_clearance_m=float("inf"),
              turn_clearance_m=float("inf"),
              terrain_frame_informative=True):
        # Yaw rate, measured rather than assumed. The commanded angular
        # velocity is not what the chassis does (see turn_stop_lead_s), so
        # the only trustworthy rate is the one odom reports.
        if self._last_step_s is not None and now_s > self._last_step_s:
            dt = now_s - self._last_step_s
            self._yaw_rate = angle_delta(yaw, self._last_yaw) / dt
        self._last_step_s = now_s
        self._now_s = now_s
        self._last_yaw = yaw
        self._turn_clearance_m = turn_clearance_m
        if terrain_score_seq is not None:
            self._last_seen_score_seq = terrain_score_seq

        if self.state == STATE_FAILSAFE:
            return self._output(0.0, 0.0)

        # /scan staleness is fatal in every other state (2026-07-27): a
        # corridor check or heading pick built from an old frame could
        # report "clear" when the real world has moved on -- worse than
        # refusing to decide at all.
        #
        # "No scan has EVER arrived" is a different situation from "a scan
        # arrived and went stale", and only in STARTUP_CHECK (2026-07-28):
        # the first tick fires milliseconds after the node starts, well
        # before DDS discovery delivers the first /scan, so failing safe on
        # it sent the rover to FAILSAFE on every launch with a perfectly
        # healthy LiDAR. Nothing has been committed to yet in STARTUP_CHECK,
        # so wait for the first frame instead -- the same call already made
        # for a not-yet-published /exomy/odom just below. This cannot mask a
        # real dead LiDAR: the rover holds active=True and commands zero the
        # whole time it waits, and the node logs the wait once.
        if not scan_fresh:
            waiting_for_first_scan = (
                not scan_ever_received and self.state == STATE_STARTUP_CHECK
            )
            if not waiting_for_first_scan:
                self._enter_failsafe(["lidar_stale"])
            return self._output(0.0, 0.0)

        # Odom-liveness gate: only states that have already committed to a
        # closed-loop maneuver (need live yaw/position to know when it's
        # done) abort here. STARTUP_CHECK/MONITORING haven't committed to
        # anything yet, so they just wait for odom themselves.
        if not odom_fresh and self.state == STATE_TURN_TO_HEADING:
            self._enter_failsafe(["odom_stale"])
            return self._output(0.0, 0.0)

        # Over-tilt preempts everything else, including a turn already in
        # progress, and re-decides from where the rover now is rather than
        # finishing a turn planned before the slope was seen.
        #
        # Until 2026-07-29 this entered SLOPE_RETREAT, which reversed in a
        # straight line. That state is gone. It was the only thing in the
        # stack that drove the rover backwards, the rover has no rear-facing
        # sensing beyond the same LiDAR, and a blind reverse is what put it
        # into a wall on 2026-07-27. It was also firing on tilt readings that
        # were never real -- Trial A logged up to 62.9 deg on a flat lab floor
        # (see imu_slope_fusion_node.gate_tilt for why, and for the fix).
        # An over-tilt is now handled exactly like a blocked corridor: stop,
        # re-scan, turn towards something clear. The IMU keeps its veto over
        # where the rover goes and loses its ability to command a manoeuvre.
        #
        # STARTUP_CHECK is still excluded: the rover has not moved yet, so an
        # over-tilt there means "refuse to start", handled in its own step.
        if imu_slope_stop and self.state != STATE_STARTUP_CHECK:
            if not odom_fresh:
                # A closed-loop turn cannot be executed without live yaw.
                return self._output(0.0, 0.0)
            self._origin_yaw = yaw
            self._commit_to_heading(heading_offset)
            return self._output(0.0, 0.0)

        if self.state == STATE_STARTUP_SWEEP:
            return self._step_startup_sweep(
                now_s, yaw, traversability_score, terrain_score_seq,
                forward_corridor_blocked, forward_clearance_m,
                terrain_frame_informative)

        if self.state == STATE_STARTUP_CHECK:
            return self._step_startup_check(now_s, yaw, imu_slope_stop,
                                            heading_offset, odom_fresh, imu_fresh,
                                            traversability_score, terrain_score_seq,
                                            terrain_frame_informative)

        if self.state == STATE_MONITORING:
            return self._step_monitoring(now_s, yaw, terrain_stopped, lidar_stopped,
                                         forward_corridor_blocked, heading_offset,
                                         odom_fresh)

        if self.state == STATE_TURN_TO_HEADING:
            return self._step_turning(yaw)

        raise AssertionError(f"unhandled state: {self.state}")

    # ── state handlers ──────────────────────────────────────────────────

    def _step_startup_check(self, now_s, yaw, imu_slope_stop, heading_offset,
                            odom_fresh, imu_fresh,
                            traversability_score=None, terrain_score_seq=None,
                            terrain_frame_informative=True):
        if self._settle_start_s is None:
            self._settle_start_s = now_s
        if (now_s - self._settle_start_s) < self.settle_time_s:
            return self._output(0.0, 0.0)

        if not odom_fresh:
            # Nothing committed yet -- just wait, same as MONITORING does
            # with stale odom, rather than fail immediately.
            return self._output(0.0, 0.0)

        reasons = []
        if not imu_fresh:
            reasons.append("imu_not_ready")
        if imu_slope_stop:
            reasons.append("over_tilted")
        if heading_offset is None:
            reasons.append("boxed_in")
        if reasons:
            self._enter_failsafe(reasons)
            return self._output(0.0, 0.0)

        # First time through, look around properly before committing to
        # anything. Later re-entries (after every completed turn) skip
        # straight to the LiDAR pick -- re-running a ~85 s sweep every time
        # the rover changed its mind would make it useless.
        if self.sweep_headings and not self._sweep_done:
            self._enter_startup_sweep(yaw)
            return self._output(0.0, 0.0)

        # A heading was refused on an earlier tick. heading_offset now excludes
        # it, so turn to whatever the picker offers instead of measuring the
        # same view again. Deliberately before the confirm: the confirm holds,
        # and a hold is what stopped the rover acting on its own refusal.
        if self._turn_after_rejection:
            self._turn_after_rejection = False
            self._origin_yaw = yaw
            self._commit_to_heading(heading_offset)
            if self.state != STATE_STARTUP_CHECK:
                return self._output(0.0, 0.0)
            # Still facing the same way -- the picker had nothing better to
            # offer. Fall through and let the rejection budget run out.

        # Terrain confirm. The sweep only chose the OPENING heading; every
        # later one comes from the LiDAR picker, which sees geometry and
        # nothing else. Without this the foundation model would have no say in
        # WHERE the rover goes for the rest of the run -- it could only refuse
        # to move, leaving the rover stopped rather than turning towards better
        # ground. Skipped when no sweep ran, since there is then no baseline to
        # compare against and the rover must not stall.
        if self._heading_chosen_by_model:
            # Not this one. DINOv2 ranked every direction it could see and
            # picked this heading; re-scoring it here learns nothing and can
            # only overrule the model's own decision. Consumed once, so the
            # next heading -- which will come from the LiDAR picker -- is
            # checked normally.
            self._heading_chosen_by_model = False
        elif self.terrain_baseline is not None:
            if (traversability_score is not None
                    and terrain_score_seq is not None
                    and terrain_score_seq != self._confirm_last_seq
                    and terrain_frame_informative):
                # terrain_frame_informative gates this, and it matters: a frame
                # the blank-frame gate rejected is published with traversability
                # 1.0, the same value an impassable rock gets, so a score alone
                # cannot tell "the ground is bad" from "I cannot see". On
                # 2026-07-29 a featureless wall 0.28 m away read detail 1.708
                # against a 2.0 threshold and the rover refused a heading it
                # had never actually measured -- three times in 0.42 s, since
                # rejected frames publish at camera rate rather than at the
                # 1.5 s inference rate, so one unchanged situation spent the
                # entire rejection budget. Blindness now falls through to the
                # wait below and ends in terrain_confirm_timeout, which is what
                # actually happened.
                self._confirm_last_seq = terrain_score_seq
                if self._confirm_discard_pending:
                    # Captured while the rover was still turning -- see the
                    # matching note in the sweep's SAMPLE phase.
                    self._confirm_discard_pending = False
                else:
                    self._confirm_scores.append(traversability_score)
            if not self._confirm_scores:
                # Waiting for a reading taken since the rover stopped turning.
                # DINOv2 runs at roughly 0.5 Hz against this 10 Hz tick, so
                # acting on whatever is already in hand would judge the new
                # heading using a frame captured while facing the old one.
                if self._confirm_wait_start_s is None:
                    self._confirm_wait_start_s = now_s
                elif (now_s - self._confirm_wait_start_s) > self.terrain_confirm_timeout_s:
                    self._enter_failsafe(["terrain_confirm_timeout"])
                return self._output(0.0, 0.0)
            self._confirm_wait_start_s = None
            mean_score = sum(self._confirm_scores) / len(self._confirm_scores)
            threshold = self.terrain_reject_threshold()
            if mean_score > threshold:
                self._terrain_rejections += 1
                self.rejection_log.append(
                    (math.degrees(yaw), mean_score, threshold)
                )
                if self._rejection_search_start_s is None:
                    self._rejection_search_start_s = now_s
                if self._terrain_rejections >= self.max_terrain_rejections:
                    # Backstop only, since 2026-08-04. Reaching this means the
                    # picker kept offering headings long past the point where
                    # the blacklist should have starved it -- worth failing safe
                    # on, but not the expected way for a search to end.
                    self._enter_failsafe(["terrain_rejected_everywhere"])
                    return self._output(0.0, 0.0)
                if (now_s - self._rejection_search_start_s) > self.terrain_search_timeout_s:
                    # Every other hold in this FSM has a deadline. This one
                    # did not, and a refused-then-re-picked heading resets the
                    # confirm timeout, so the rover rejected in silence for as
                    # long as it was left powered. Distinct from
                    # terrain_rejected_everywhere on purpose: that reason means
                    # "N directions refused", this one means "still refusing
                    # after N seconds", and a run that ends this way is
                    # reporting a search that never converged rather than one
                    # that finished.
                    self._enter_failsafe(["terrain_search_timeout"])
                    return self._output(0.0, 0.0)
                self._reject_current_heading(yaw)
                self._turn_after_rejection = True
                self._confirm_scores = []
                self._confirm_last_seq = self._last_seen_score_seq
                self._confirm_discard_pending = True
                self._confirm_wait_start_s = None
                # Hold one tick. The next one turns to whatever the picker
                # offers with this heading excluded, or fails safe as boxed_in
                # if nothing is left.
                return self._output(0.0, 0.0)

            # Accepted, so this reading now describes the ground the rover is
            # on, and that is what the next heading should be judged against.
            # Frozen at the opening winner's score, the baseline compared every
            # later heading to one reading taken somewhere else entirely: the
            # lab floor measured 0.004 to 0.405 across directions from a single
            # spot, so winner+margin refused most of the compass all run.
            self.terrain_baseline = (
                mean_score if self.terrain_baseline_ceiling is None
                else min(mean_score, self.terrain_baseline_ceiling)
            )

        # The heading was accepted, so the run of rejections is over. Counted
        # consecutively on purpose: a rover that refuses one patch, drives on
        # and later refuses another is working, not stuck. The search clock
        # clears with the count, for the same reason -- the next run must not
        # inherit this one's elapsed time.
        self._terrain_rejections = 0
        self._rejection_search_start_s = None

        self._origin_yaw = yaw
        if abs(heading_offset) <= self.angle_tolerance_rad:
            # Already facing a viable heading -- no turn needed.
            self.state = STATE_MONITORING
            self.active = False
            return self._output(0.0, 0.0)

        self._enter_turn_to_heading(heading_offset)
        return self._output(0.0, 0.0)

    def terrain_reject_threshold(self):
        """Score above which a heading's terrain is refused, or None if no
        baseline has been established yet.

        One method rather than the expression inlined at both sites: the sweep
        report prints this number, and a report that disagrees with the rule
        actually applied is worse than no report at all -- these logs are the
        only record of why a run ended.
        """
        if self.terrain_baseline is None:
            return None
        return max(self.terrain_baseline + self.terrain_reject_margin,
                   self.terrain_reject_floor)

    def _openness_cost(self, entry):
        """0.0 for a heading with all the room we care about, 1.0 for none."""
        room = min(entry["clearance_m"], self.clearance_horizon_m)
        return 1.0 - room / self.clearance_horizon_m

    def _heading_cost(self, entry):
        """What the sweep minimises: terrain, plus a weighted penalty for
        having nowhere to go. Clearance breaks remaining ties."""
        return (entry["score"]
                + self.openness_weight * self._openness_cost(entry),
                -entry["clearance_m"])

    def _turn_arrived(self, delta):
        """Is a turn close enough to stop commanding rotation?

        Not simply |delta| <= tolerance. The chassis keeps turning after the
        command stops, by an amount that depends on how fast it is going, so
        the test allows for one actuation lag's worth of travel at the
        measured rate. With turn_stop_lead_s at 0 this is the old behaviour.
        """
        margin = max(self.angle_tolerance_rad,
                     abs(self._yaw_rate) * self.turn_stop_lead_s)
        return abs(delta) <= margin

    def _sweep_offsets(self):
        """Equally spaced offsets from the sweep's origin heading, in visit
        order: 0 first so the rover samples where it already points before
        moving at all."""
        n = self.sweep_headings
        return [normalize_angle(k * 2.0 * math.pi / n) for k in range(n)]

    def _step_startup_sweep(self, now_s, yaw, traversability_score,
                            terrain_score_seq, forward_corridor_blocked,
                            forward_clearance_m,
                            terrain_frame_informative=True):
        """Rotate through sweep_headings stops, sampling DINOv2 at each.

        Three phases per stop: TURN to it, SETTLE so neither the camera nor
        the IMU is reading motion, then SAMPLE until enough distinct DINOv2
        results have arrived. Sampling counts terrain_score_seq changes rather
        than ticks, because DINOv2 runs at roughly 0.5 Hz on the Pi while this
        FSM ticks at 10 Hz -- counting ticks would record the same inference
        twenty times and call it twenty samples.

        terrain_frame_informative gates SAMPLE the same way it already gates
        _step_startup_check's confirm (2026-08-02, pre-Trial-A audit): an
        uninformative frame (camera still converging exposure, or a genuinely
        blank/dark view) is published with traversability_score=1.0, the same
        value an impassable rock gets, so an unfiltered reading here could
        silently win or lose the sweep's ranking on a value that has nothing
        to do with terrain -- corrupting the exact decision the sweep exists
        to protect. A stop stuck on uninformative frames still can't hang
        forever: the deadline check above (sweep_turn_timeout) bounds every
        phase, not just TURN.
        """
        if self._now_s > self._state_deadline_s:
            self._enter_failsafe(["sweep_turn_timeout"])
            return self._output(0.0, 0.0)

        if self._sweep_phase == "TURN":
            delta = angle_delta(self._target_yaw, yaw)
            if not self._turn_arrived(delta):
                if not self._has_room_to_turn():
                    self._enter_failsafe(["no_room_to_turn"])
                    return self._output(0.0, 0.0)
                angular = self.angular_speed if delta > 0 else -self.angular_speed
                return self._output(0.0, angular)
            self._sweep_phase = "SETTLE"
            self._settle_start_s = now_s
            return self._output(0.0, 0.0)

        if self._sweep_phase == "SETTLE":
            if (now_s - self._settle_start_s) < self.settle_time_s:
                return self._output(0.0, 0.0)
            self._sweep_phase = "SAMPLE"
            self._sweep_scores = []
            # Anything already published belongs to the previous heading.
            self._sweep_last_seq = self._last_seen_score_seq
            self._sweep_discard_pending = True
            return self._output(0.0, 0.0)

        # SAMPLE
        fresh = (
            traversability_score is not None
            and terrain_score_seq is not None
            and terrain_score_seq != self._sweep_last_seq
        )
        if fresh:
            self._sweep_last_seq = terrain_score_seq
            if self._sweep_discard_pending:
                # Second rule, separate from the sequence check above. A new
                # sequence number only means "this result is new", not "the
                # image behind it was taken after the rover stopped". DINOv2
                # takes 2.2 s per frame against a 1.0 s settle, so the first
                # result to arrive was captured mid-turn however new it is.
                self._sweep_discard_pending = False
                return self._output(0.0, 0.0)
            if not terrain_frame_informative:
                # Marked seen (via _sweep_last_seq above) so the next fresh
                # result is waited for instead of this one being re-read, but
                # not counted as a sample -- see the docstring above for why.
                return self._output(0.0, 0.0)
            self._sweep_scores.append(traversability_score)
            # The LiDAR verdict is recorded from the same instant as the
            # terrain sample so the two describe the same moment.
            self._sweep_blocked = forward_corridor_blocked
            self._sweep_clearance_m = forward_clearance_m
            # And so is the yaw. This is the whole fix for Trial A: the table
            # used to be labelled with the offset the rover was ASKED for,
            # which it overran by about 20 deg every time, so the score for
            # "-45" described a view from somewhere else entirely.
            self._sweep_sample_yaw = yaw

        if len(self._sweep_scores) < self.sweep_samples_per_heading:
            return self._output(0.0, 0.0)

        self.sweep_report.append({
            "offset_rad": self._sweep_offsets()[self._sweep_index],
            "yaw": self._sweep_sample_yaw,
            "score": sum(self._sweep_scores) / len(self._sweep_scores),
            "blocked": self._sweep_blocked,
            "clearance_m": self._sweep_clearance_m,
        })

        self._sweep_index += 1
        if self._sweep_index < self.sweep_headings:
            self._sweep_phase = "TURN"
            self._target_yaw = normalize_angle(
                self._origin_yaw + self._sweep_offsets()[self._sweep_index]
            )
            self._state_deadline_s = self._now_s + self.max_turn_duration_s
            return self._output(0.0, 0.0)

        return self._finish_sweep(yaw)

    def _finish_sweep(self, yaw):
        """Rank the sweep and commit to the winner.

        LiDAR first and absolutely: anything the rover cannot physically fit
        through is removed, however good the ground looks. DINOv2 then RANKS
        what survives, lowest traversability score winning, with LiDAR
        clearance breaking ties. Ranking rather than thresholding is
        deliberate -- the 214-frame real-camera study found the model
        confidently calling a wall soil:0.937, so an absolute cut-off
        rubber-stamps everything, while an ordering only needs the model to
        rate one surface worse than another.
        """
        self._sweep_done = True
        passable = [s for s in self.sweep_report if not s["blocked"]]
        if not passable:
            self._enter_failsafe(["boxed_in"])
            return self._output(0.0, 0.0)

        # Prefer headings with somewhere to go. If none has that much room --
        # a corner, a narrow pit -- the requirement relaxes rather than
        # eliminating the rover's last option and failing safe on a rover that
        # could still move.
        roomy = [s for s in passable if s["clearance_m"] >= self.min_run_m]
        candidates = roomy if roomy else passable
        self.sweep_choice = min(candidates, key=self._heading_cost)
        # The same choice with the terrain score removed, so the log can say
        # whether DINOv2 changed anything.
        self.sweep_geometry_choice = min(
            candidates, key=lambda s: (self._openness_cost(s), -s["score"])
        )
        self.sweep_model_changed_the_choice = (
            self.sweep_choice is not self.sweep_geometry_choice
        )
        # The winner's OWN score, not the median of the sweep. The median
        # rejects about half of every compass by construction: Trial A
        # measured 0.035/0.006/0.140/0.383/0.158/0.358/0.405/0.004, whose
        # median 0.149 plus a 0.15 margin refuses three of the eight headings
        # outright, and refusing is silent and unbounded. It also contradicts
        # the ranking design -- the sweep exists to say which direction is
        # best, so the thing to hold a later reading against is how good that
        # direction measured, not how good a typical direction was.
        self.terrain_baseline = self.sweep_choice["score"]
        self.terrain_baseline_ceiling = max(s["score"] for s in passable)
        # This heading is the model's own choice. Do not re-test it on arrival.
        self._heading_chosen_by_model = True

        self._commit_to_yaw(self.sweep_choice["yaw"], yaw)
        if self.state == STATE_STARTUP_SWEEP:
            # Already facing the winner, so no turn was started.
            self.state = STATE_MONITORING
            self.active = False
        return self._output(0.0, 0.0)

    def _commit_to_yaw(self, target_yaw, yaw):
        """Turn to an absolute odom yaw, as opposed to an offset from the
        sweep's origin. The sweep's stops are wherever the chassis actually
        came to rest, which is not where it was aimed, so a stop can only be
        returned to by the yaw it was measured at."""
        delta = angle_delta(target_yaw, yaw)
        if self._turn_arrived(delta):
            return
        if not self._has_room_to_turn():
            self._enter_failsafe(["no_room_to_turn"])
            return
        self.active = True
        self.state = STATE_TURN_TO_HEADING
        self._target_yaw = normalize_angle(target_yaw)
        self._state_deadline_s = self._now_s + self.max_turn_duration_s

    def _step_monitoring(self, now_s, yaw, terrain_stopped, lidar_stopped,
                        forward_corridor_blocked, heading_offset, odom_fresh=True):
        if forward_corridor_blocked:
            if odom_fresh:
                self._origin_yaw = yaw
                self._commit_to_heading(heading_offset)
            # else: odom is stale -- a closed-loop turn cannot be executed
            # safely, so stay in MONITORING commanding zero and rely on the
            # independent lidar_proximity_guard_node hard stop.
            return self._output(0.0, 0.0)

        stopped = terrain_stopped or lidar_stopped
        if not stopped:
            self._stuck_since_s = None
            self.active = False
            # Terrain and LiDAR both agree the rover is fine where it is, so
            # anything previously refused belongs to ground it has left
            # behind -- start the next decision with a clean slate.
            self._rejected_yaws = []
            return self._output(0.0, 0.0)

        if self._stuck_since_s is None:
            self._stuck_since_s = now_s

        if (now_s - self._stuck_since_s) < self.stuck_dwell_s:
            return self._output(0.0, 0.0)

        if not odom_fresh:
            return self._output(0.0, 0.0)

        # DINOv2 (via terrain_controller_node's stop flag) has refused the
        # ground the rover is pointing at, and it has held that opinion for a
        # full dwell. Record it so the LiDAR-only picker stops re-proposing
        # this heading -- without this the rover re-picks the same
        # geometrically-clear-but-untraversable direction forever. A LiDAR
        # proximity stop deliberately does NOT count: that is an object in
        # the way, not a verdict on the terrain, and blacklisting a heading
        # for it would permanently rule out directions a person or a rock
        # merely happened to be standing in.
        if terrain_stopped:
            self._reject_current_heading(yaw)

        self._origin_yaw = yaw
        self._commit_to_heading(heading_offset)
        return self._output(0.0, 0.0)

    def _has_room_to_turn(self) -> bool:
        return self._turn_clearance_m >= self.turn_radius_m

    def _commit_to_heading(self, heading_offset):
        """Turn towards heading_offset, or fail safe if there is nowhere to go.

        Shared by every "the way ahead is refused, pick another" path --
        a blocked corridor, a terrain refusal, and (since 2026-07-29) an
        over-tilt. Staying put when already within tolerance is deliberate:
        terrain_controller_node resumes driving on its own once this node
        cedes cmd_vel back.
        """
        if heading_offset is None:
            self._enter_failsafe(["boxed_in"])
        elif abs(heading_offset) > self.angle_tolerance_rad:
            if not self._has_room_to_turn():
                self._enter_failsafe(["no_room_to_turn"])
                return
            self._enter_turn_to_heading(heading_offset)

    def _step_turning(self, yaw):
        if self._now_s > self._state_deadline_s:
            self._enter_failsafe(["turn_timeout"])
            return self._output(0.0, 0.0)

        delta = angle_delta(self._target_yaw, yaw)
        if self._turn_arrived(delta):
            # Back to STARTUP_CHECK rather than straight to MONITORING: the
            # rover has just been rotating, so this re-runs the settle window
            # and re-scans from the new heading before committing to drive.
            # Accelerometer-derived tilt in particular is corrupted by the
            # rotation that just ended (the same physics behind
            # imu_slope_fusion_node's gate_tilt), and the corridor ahead is a
            # different piece of world than the one scanned before the turn.
            self._enter_startup_check()
            return self._output(0.0, 0.0)

        angular = self.angular_speed if delta > 0 else -self.angular_speed
        return self._output(0.0, angular)

    # ── transition helpers ──────────────────────────────────────────────

    def _enter_failsafe(self, reasons):
        self.state = STATE_FAILSAFE
        self.failsafe = True
        # Deliberately stays True (changed 2026-07-27): FAILSAFE must keep
        # OWNING cmd_vel so the node actively publishes a zero Twist rather
        # than merely going quiet and letting whoever else publishes take the
        # rover over. This does keep lidar_proximity_guard_node's suppression
        # engaged, but only until the failsafe shutdown tears the stack down
        # a couple of seconds later (see should_shutdown_now), and the guard's
        # own zero-publish would be redundant with ours anyway.
        self.active = True
        self.failsafe_reason = "; ".join(reasons)
        if self.failsafe_since_s is None:
            self.failsafe_since_s = self._now_s

    def _enter_startup_sweep(self, yaw):
        self.active = True
        self.state = STATE_STARTUP_SWEEP
        self._origin_yaw = yaw
        self._sweep_index = 0
        self._sweep_phase = "TURN"
        self._sweep_scores = []
        self._sweep_last_seq = self._last_seen_score_seq
        self._sweep_discard_pending = True
        self._sweep_blocked = True
        self._sweep_clearance_m = 0.0
        self._sweep_sample_yaw = yaw
        self.sweep_origin_yaw = yaw
        self.sweep_report = []
        self._target_yaw = normalize_angle(yaw + self._sweep_offsets()[0])
        self._state_deadline_s = self._now_s + self.max_turn_duration_s

    def _enter_startup_check(self):
        """(Re-)enter the hold-still-and-look-around state. Used at boot and
        after every completed turn -- see _step_turning for why."""
        self.state = STATE_STARTUP_CHECK
        self.active = True
        self._settle_start_s = None
        self._stuck_since_s = None
        self._confirm_scores = []
        self._confirm_last_seq = self._last_seen_score_seq
        self._confirm_discard_pending = True
        self._confirm_wait_start_s = None

    def _enter_turn_to_heading(self, offset):
        self.active = True
        self.state = STATE_TURN_TO_HEADING
        self._target_yaw = normalize_angle(self._origin_yaw + offset)
        self._state_deadline_s = self._now_s + self.max_turn_duration_s

    def excluded_offsets(self, yaw: float) -> list:
        """Terrain-rejected headings as offsets from `yaw`, for the picker."""
        return [normalize_angle(y - yaw) for y in self._rejected_yaws]

    def _reject_current_heading(self, yaw: float) -> None:
        """Remember that the ground the rover is facing was refused."""
        for existing in self._rejected_yaws:
            if abs(angle_delta(existing, yaw)) <= self.angle_tolerance_rad:
                return
        self._rejected_yaws.append(normalize_angle(yaw))

    def _output(self, linear_x, angular_z):
        return {
            "state": self.state,
            "active": self.active,
            "failsafe": self.failsafe,
            "failsafe_reason": self.failsafe_reason,
            "excluded_offsets": self.excluded_offsets(self._last_yaw),
            "linear_x": linear_x,
            "angular_z": angular_z,
        }


# ── ROS2 node wrapper ────────────────────────────────────────────────────

class ReactiveExplorerNode(Node):

    def __init__(self, shutdown_fn=None):
        super().__init__("reactive_explorer_node")

        self.declare_parameter("stuck_dwell_s", 3.0)
        self.declare_parameter("angle_tolerance_deg", 3.0)
        self.declare_parameter("angular_speed", 0.3)
        self.declare_parameter("publish_rate_hz", 10.0)
        self.declare_parameter("cmd_vel_topic", "/exomy/cmd_vel")
        self.declare_parameter("odom_topic", "/exomy/odom")
        self.declare_parameter("scan_topic", "/scan")
        self.declare_parameter("rover_width_m", 0.32)
        self.declare_parameter("lookahead_buffer_m", 0.3)
        self.declare_parameter("angle_step_deg", 5.0)
        # Consecutive fresh scans that must agree the forward corridor is
        # blocked before avoidance is triggered -- debounces LiDAR dropouts.
        self.declare_parameter("trigger_confirm_count", 2)
        # Angular width of a terrain rejection -- see _exclude_tolerance_rad.
        self.declare_parameter("exclude_tolerance_deg", 20.0)
        # Tried in order for both STARTUP_CHECK and mid-drive re-scans: a
        # generous clearance requirement first, relaxed only if NOTHING in
        # the full 360deg scan clears it. 0.3m (not the original 0.5m)
        # reflects the LiDAR's mast/neck self-view now sitting ahead of it
        # rather than behind it -- re-measure with
        # experiments/lidar_self_view_characterisation.py after any mounting
        # change before trusting this on real hardware.
        self.declare_parameter("lookahead_tiers_m", [1.0, 0.3])
        # Matches lidar_proximity_guard_node's own stale_timeout_s default --
        # if /scan stops arriving, EVERY state (not just proactive avoidance)
        # must fail safe rather than keep re-evaluating an arbitrarily old
        # frame (2026-07-27: a stale-but-clear-looking frame is worse than
        # refusing to decide).
        self.declare_parameter("stale_timeout_s", 3.0)
        # No /exomy/odom for this long -> treat yaw/position as unusable and
        # abort any in-progress timed state to FAILSAFE. On the real-hardware
        # launch profile with use_slam left at its default False, there is NO
        # publisher for /exomy/odom at all, so this must fire fast.
        self.declare_parameter("odom_stale_timeout_s", 3.0)
        # No RAW IMU message for this long -> STARTUP_CHECK cannot verify it
        # is safe to move at all and must fail safe rather than assume "not
        # tilted" from a stale/never-received default. Mid-drive IMU
        # disconnects are NOT gated this way (accepted hardware limitation,
        # documented rather than engineered around).
        #
        # This deliberately watches the RAW driver topic, not /imu_slope_stop.
        # imu_slope_fusion_node publishes /imu_slope_stop from a fixed-rate
        # timer that runs whether or not the IMU is alive, holding its last
        # (initially 0.0 deg) tilt -- so with a dead I2C bus it happily
        # publishes "not tilted" forever. Observed exactly that on 2026-07-27:
        # icm20948_driver_node died with OSError(errno 5) at startup and the
        # fusion node logged tilt=0.0 deg every 0.5s for the rest of the run.
        # Watching /imu_slope_stop would therefore have reported a healthy IMU
        # during the one failure this check exists to catch.
        self.declare_parameter("imu_raw_topic", "/exomy/imu_raw")
        self.declare_parameter("imu_stale_timeout_s", 3.0)
        # How long to hold still at STARTUP_CHECK (and any later re-entry)
        # before trusting LiDAR/IMU readings.
        self.declare_parameter("settle_time_s", 1.0)
        # Must match lidar_proximity_guard_node's min_ignore_m / blocked_sectors_deg
        # exactly -- these are NOT shared automatically between the two nodes,
        # keep both launch-file param blocks in sync by hand.
        self.declare_parameter("min_ignore_m", 0.2)
        # Clearance demanded beyond the rover's physical half-width, to absorb
        # LiDAR yaw-offset uncertainty, point-turn overshoot and wheel slip.
        # Without it a heading whose obstacle sits a millimetre outside the
        # chassis counts as clear.
        self.declare_parameter("lateral_margin_m", 0.04)
        # 2026-08-04, hardware round 4/5 of the H5 follow-up confirmation
        # testing: pick_heading_tiered's "smallest deviation that clears"
        # preference let the rover nudge open a few degrees at a time while
        # approaching a wall nearly head on, drifting into it at a shallow
        # angle instead of turning away. This biases the picker towards a
        # heading at least this far from straight ahead, falling back to the
        # unrestricted search if nothing clears the minimum anywhere (see
        # pick_heading_tiered's min_offset_rad) -- can only make the rover
        # turn further, never refuse a correction that was available before.
        # 30 deg, not e.g. 45: must stay comfortably above angle_tolerance_deg
        # (see test_min_correction_turn_is_wider_than_the_arrival_tolerance)
        # or the FSM could "arrive" on a turn too small to have corrected
        # anything, the same bug angle_step_deg < angle_tolerance_deg guards
        # against below.
        self.declare_parameter("min_correction_turn_deg", 30.0)
        # Opening sweep. 8 stops of 45 deg is set by the camera's ~62 deg
        # field of view, not by the LiDAR, which already returns the full
        # circle in one frame. 0 disables the sweep.
        self.declare_parameter("sweep_headings", 8)
        # DINOv2 runs at ~2.2 s/frame as deployed, so each extra sample per
        # stop costs ~18 s across the whole sweep.
        self.declare_parameter("sweep_samples_per_heading", 2)
        self.declare_parameter("terrain_reject_margin", 0.15)
        # Floor under baseline+margin, so a stretch of soil cannot collapse the
        # baseline to ~0 and leave the rover refusing sand. Between sand's
        # CLASS_RISK (0.5) and bedrock's (0.7): soil and sand always pass,
        # bedrock and big_rock/uncertain still refuse.
        self.declare_parameter("terrain_reject_floor", 0.6)
        # How far ahead the sweep bothers measuring open space. Only a
        # tie-break, so a generous value costs nothing but a slightly longer
        # near-return list once per scan.
        self.declare_parameter("clearance_horizon_m", 3.0)
        # Radius the rover's corners sweep during a point turn, plus margin:
        # hypot(0.20 LiDAR-to-front-wheel, 0.16 half width) = 0.256 m.
        self.declare_parameter("turn_radius_m", 0.30)
        self.declare_parameter("terrain_confirm_timeout_s", 15.0)
        self.declare_parameter("turn_stop_lead_s", 0.0)
        # Backstop, not the working bound -- see the FSM's own comment. Must
        # stay above the 15 refusals it takes to exhaust the picker, or the
        # count becomes the bound again and the rover gives up with directions
        # it never looked at.
        self.declare_parameter("max_terrain_rejections", 20)
        # What actually bounds a run of refusals. One refusal costs a turn, a
        # settle, and 2-3 DINOv2 frames at the deployed 2.2 s, so ~8 s; the 15
        # needed to exhaust the picker is therefore ~2 minutes. 180 s clears
        # that with margin, so this fires only for a search that is not
        # converging rather than cutting an honest one short.
        self.declare_parameter("terrain_search_timeout_s", 180.0)
        self.declare_parameter("min_run_m", 0.0)
        self.declare_parameter("openness_weight", 0.0)
        self.declare_parameter(
            "blocked_sectors_deg", [],
            ParameterDescriptor(dynamic_typing=True),
        )
        # LiDAR mounting yaw relative to the rover's physical front. Measured
        # live 2026-07-26 (see experiments/measure_lidar_yaw_offset_direct.py
        # and project_lidar_avoidance_hardware_trial_20260726). Default 0.0
        # (no correction) so simulation/synthetic-scan callers and existing
        # tests are unaffected; the real-hardware launch file sets this
        # explicitly.
        self.declare_parameter("lidar_yaw_offset_deg", 0.0)
        # Worst-case turn duration for the wall-clock timeout (Layer B,
        # independent of odom_fresh).
        self.declare_parameter("max_turn_duration_s", 20.0)
        # How long FAILSAFE actively publishes a zero Twist before signalling
        # the launch to shut down. Must be long enough for the command to
        # reach motor_node through the cmd_vel -> RoverCommand -> motor_node
        # chain at publish_rate_hz; 2.0s is ~20 zero commands at the default
        # 10Hz. See should_shutdown_now() for why the order matters.
        self.declare_parameter("failsafe_stop_grace_s", 2.0)

        self.fsm = ReactiveExplorerFSM(
            stuck_dwell_s=self.get_parameter("stuck_dwell_s").value,
            angle_tolerance_rad=math.radians(self.get_parameter("angle_tolerance_deg").value),
            angular_speed=self.get_parameter("angular_speed").value,
            max_turn_duration_s=self.get_parameter("max_turn_duration_s").value,
            sweep_headings=self.get_parameter("sweep_headings").value,
            sweep_samples_per_heading=self.get_parameter(
                "sweep_samples_per_heading").value,
            terrain_reject_margin=self.get_parameter("terrain_reject_margin").value,
            terrain_reject_floor=self.get_parameter("terrain_reject_floor").value,
            terrain_confirm_timeout_s=self.get_parameter(
                "terrain_confirm_timeout_s").value,
            turn_radius_m=self.get_parameter("turn_radius_m").value,
            turn_stop_lead_s=self.get_parameter("turn_stop_lead_s").value,
            max_terrain_rejections=self.get_parameter(
                "max_terrain_rejections").value,
            terrain_search_timeout_s=self.get_parameter(
                "terrain_search_timeout_s").value,
            min_run_m=self.get_parameter("min_run_m").value,
            openness_weight=self.get_parameter("openness_weight").value,
            clearance_horizon_m=self.get_parameter(
                "clearance_horizon_m").value,
            settle_time_s=self.get_parameter("settle_time_s").value,
        )

        self._rover_width_m = self.get_parameter("rover_width_m").value
        self._angle_step_rad = math.radians(self.get_parameter("angle_step_deg").value)
        self._min_correction_turn_rad = math.radians(
            self.get_parameter("min_correction_turn_deg").value)
        self._lookahead_tiers_m = list(self.get_parameter("lookahead_tiers_m").value)
        # How close a candidate must be to a terrain-rejected heading to count
        # as the same direction. Wider than angle_step_deg so one rejection
        # removes a real wedge rather than a single 5deg sample the picker
        # would just step around.
        self._exclude_tolerance_rad = math.radians(
            self.get_parameter("exclude_tolerance_deg").value
        )
        self._stale_timeout_s = self.get_parameter("stale_timeout_s").value
        self._odom_stale_timeout_s = self.get_parameter("odom_stale_timeout_s").value
        self._imu_stale_timeout_s = self.get_parameter("imu_stale_timeout_s").value
        self._min_ignore_m = self.get_parameter("min_ignore_m").value
        blocked_sectors_param = self.get_parameter("blocked_sectors_deg").get_parameter_value()
        if blocked_sectors_param.type == ParameterType.PARAMETER_INTEGER_ARRAY:
            sectors_deg = [float(v) for v in blocked_sectors_param.integer_array_value]
        else:
            sectors_deg = [float(v) for v in blocked_sectors_param.double_array_value]
        self._blocked_sectors = parse_blocked_sectors(sectors_deg)
        self._lidar_yaw_offset_rad = math.radians(
            self.get_parameter("lidar_yaw_offset_deg").value
        )
        # Forward fast-trigger corridor (bypasses stuck_dwell_s): sized to
        # the strictest (first) lookahead tier, same rover_width_m geometry
        # as the heading picker uses.
        self._forward_lookahead_m = self._lookahead_tiers_m[0]
        # Box geometry, replacing the wedge on 2026-07-29. The wedge was one
        # rover wide only AT the lookahead distance and narrower everywhere
        # closer, so an obstacle beside a front wheel read as clear -- see
        # avoidance_planner.corridor_half_angle_rad's docstring and the lab
        # cupboard collision it caused.
        self._lateral_margin_m = self.get_parameter("lateral_margin_m").value
        self._clearance_horizon_m = self.get_parameter("clearance_horizon_m").value
        self._corridor_half_width_m = (
            self._rover_width_m / 2.0 + self._lateral_margin_m
        )
        self._latest_scan = None
        self._trigger_confirm_count = self.get_parameter("trigger_confirm_count").value
        self._forward_blocked_count = 0
        self._last_counted_scan_stamp = None

        cmd_vel_topic = self.get_parameter("cmd_vel_topic").value
        odom_topic    = self.get_parameter("odom_topic").value
        publish_hz    = self.get_parameter("publish_rate_hz").value

        # ── Live sensor state (updated by subscription callbacks) ─────────
        self._yaw              = 0.0
        self._pos_x            = 0.0
        self._pos_y            = 0.0
        self._terrain_stopped  = False
        self._lidar_stopped    = False
        self._imu_slope_stop   = False
        # DINOv2's continuous risk score, 0.0 safe .. 1.0 stop. The sequence
        # number is what lets the FSM tell a fresh inference from the same one
        # read again: DINOv2 delivers roughly 0.5 results per second on the Pi
        # while this node ticks at 10 Hz, so counting ticks would record one
        # inference twenty times over.
        self._traversability_score = None
        self._terrain_score_seq = 0
        # True until dinov2_terrain_node says otherwise, so a stack running
        # without it is unaffected.
        self._terrain_frame_informative = True
        self._sweep_logged = False
        self._rejections_logged = 0
        self._stuck_detection_active = False
        self._prev_state       = self.fsm.state
        self._last_odom_time   = None
        self._odom_never_seen_warned = False
        self._scan_never_seen_warned = False
        self._last_imu_time    = None
        # Injectable so tests can verify shutdown was requested without
        # actually killing the test process.
        self._shutdown_fn = shutdown_fn or default_shutdown
        self._shutdown_triggered = False
        self._failsafe_logged = False
        self._failsafe_stop_grace_s = self.get_parameter(
            "failsafe_stop_grace_s"
        ).value

        # ── ROS2 pub / sub ─────────────────────────────────────────────
        self.create_subscription(Bool, "/terrain_controller/stopped", self._terrain_stopped_cb, 10)
        self.create_subscription(Bool, "/lidar_proximity_stop", self._lidar_stopped_cb, 10)
        self.create_subscription(Bool, "/imu_slope_stop", self._imu_slope_stop_cb, 10)
        self.create_subscription(
            Float64, "/traversability_score", self._traversability_score_cb, 10
        )
        # Paired with the score above. A frame the blank-frame gate rejected is
        # published with traversability 1.0, indistinguishable from an
        # impassable rock, so the score alone cannot separate "the ground is
        # bad" from "I cannot see". Defaults True so a run without
        # dinov2_terrain_node behaves as it did before this topic existed.
        self.create_subscription(
            Bool, "/terrain_frame_informative",
            self._terrain_frame_informative_cb, 10
        )
        self.create_subscription(
            Imu, self.get_parameter("imu_raw_topic").value, self._imu_raw_cb, 10
        )
        self.create_subscription(Bool, "/stuck_detection/active", self._stuck_detection_active_cb, 10)
        self.create_subscription(Odometry, odom_topic, self._odom_cb, 10)
        scan_topic = self.get_parameter("scan_topic").value
        self.create_subscription(LaserScan, scan_topic, self._scan_cb, 10)

        self.pub_cmd_vel  = self.create_publisher(Twist, cmd_vel_topic, 10)
        self.pub_active   = self.create_publisher(Bool, "/reactive_explorer/active", 10)
        self.pub_failsafe = self.create_publisher(Bool, "/reactive_explorer/failsafe", 10)
        self.pub_failsafe_reason = self.create_publisher(
            String, "/reactive_explorer/failsafe_reason", 10
        )
        self.pub_translating = self.create_publisher(
            Bool, "/reactive_explorer/commanding_linear_motion", 10
        )

        self.create_timer(1.0 / publish_hz, self._tick)

        self.get_logger().info(
            "reactive_explorer_node ready\n"
            f"  stuck_dwell_s={self.fsm.stuck_dwell_s:.1f} "
            f"settle_time_s={self.fsm.settle_time_s:.1f} "
            f"lookahead_tiers_m={self._lookahead_tiers_m}"
        )
        self.get_logger().info(
            f"  rover_width_m={self._rover_width_m:.2f} "
            f"forward_corridor_half_width_m={self._corridor_half_width_m:.2f}"
        )

    # ── subscription callbacks ─────────────────────────────────────────

    def _terrain_stopped_cb(self, msg: Bool) -> None:
        self._terrain_stopped = msg.data

    def _lidar_stopped_cb(self, msg: Bool) -> None:
        self._lidar_stopped = msg.data

    def _imu_slope_stop_cb(self, msg: Bool) -> None:
        self._imu_slope_stop = msg.data

    def _log_sweep_report_once(self) -> None:
        """Print the opening sweep's full table the moment it completes.

        This is the record of the foundation model actually making a
        decision -- eight directions, what DINOv2 thought of each, what the
        LiDAR allowed, and which one won -- so it belongs in the log verbatim
        rather than only as the single heading that came out.
        """
        if self._sweep_logged or not self.fsm.sweep_report:
            return
        if self.fsm.sweep_choice is None and not self.fsm.failsafe:
            return
        self._sweep_logged = True
        lines = ["reactive_explorer: opening sweep complete",
                 "  asked | reached | error | DINOv2 score | LiDAR | clearance"]
        for entry in self.fsm.sweep_report:
            chosen = entry is self.fsm.sweep_choice
            clearance = entry["clearance_m"]
            # Asked and reached are both printed, and so is the gap between
            # them. Trial A overran every stop by about 20 deg and nothing in
            # the log said so, because only the asked-for heading was
            # recorded. The error column is a real measurement of this
            # rover's open-loop turn accuracy and belongs in Chapter 4.
            reached_rad = normalize_angle(
                entry["yaw"] - self.fsm.sweep_origin_yaw
            )
            error_rad = normalize_angle(reached_rad - entry["offset_rad"])
            asked = math.degrees(entry["offset_rad"])
            reached = math.degrees(reached_rad)
            lines.append(
                f"  {asked:+5.0f} | {reached:+7.0f} | "
                f"{math.degrees(error_rad):+5.0f} | "
                f"{entry['score']:12.3f} | "
                f"{'BLOCKED' if entry['blocked'] else '  open ':>7} | "
                f"{clearance:6.2f} m{'   <-- chosen' if chosen else ''}"
            )
        if self.fsm.sweep_geometry_choice is not None:
            # The ablation, printed every run: what the LiDAR alone would have
            # chosen, and whether DINOv2's ranking changed it. This is the
            # number that says how much the foundation model contributed, and
            # it belongs in the record rather than in a claim.
            geo = self.fsm.sweep_geometry_choice
            geo_reached = normalize_angle(
                geo["yaw"] - self.fsm.sweep_origin_yaw
            )
            lines.append(
                f"  geometry alone would have chosen "
                f"{math.degrees(geo_reached):+.0f} deg "
                f"({geo['clearance_m']:.2f} m, score {geo['score']:.3f}) -- "
                f"DINOv2 "
                + ("CHANGED the heading"
                   if self.fsm.sweep_model_changed_the_choice
                   else "agreed")
            )
        if self.fsm.terrain_baseline is not None:
            lines.append(
                f"  baseline (chosen heading's own score) = "
                f"{self.fsm.terrain_baseline:.3f}, "
                f"reject above {self.fsm.terrain_reject_threshold():.3f}"
            )
        self.get_logger().info("\n".join(lines))

    def _log_new_terrain_rejections(self) -> None:
        """Say out loud when DINOv2 refuses a heading.

        This was silent. On Trial A the rover refused heading after heading
        for 573 s and the only visible symptom was a rover that would not
        move: no state change, no warning, nothing. The three numbers below
        are what separates "the ground really is worse here" from "the
        baseline is wrong", which is the distinction that mattered.
        """
        while self._rejections_logged < len(self.fsm.rejection_log):
            yaw_deg, score, threshold = self.fsm.rejection_log[self._rejections_logged]
            self._rejections_logged += 1
            # "at yaw", not "heading": this is where the rover is pointing and
            # what it therefore just measured, not a candidate it was offered.
            # The first version said "heading" and read as though three
            # different directions had been tried when the rover had not moved.
            self.get_logger().warn(
                f"reactive_explorer: terrain REFUSED the ground ahead at yaw "
                f"{yaw_deg:+.0f} deg -- DINOv2 {score:.3f} > {threshold:.3f} "
                f"(rejection {self._rejections_logged} of "
                f"{self.fsm.max_terrain_rejections})"
            )

    def _traversability_score_cb(self, msg: Float64) -> None:
        self._traversability_score = msg.data
        self._terrain_score_seq += 1

    def _terrain_frame_informative_cb(self, msg: Bool) -> None:
        self._terrain_frame_informative = msg.data

    def _imu_raw_cb(self, msg: Imu) -> None:
        # Liveness only -- the tilt decision itself stays with
        # imu_slope_fusion_node. See imu_stale_timeout_s for why this must
        # watch the raw driver topic rather than the fused stop flag.
        self._last_imu_time = self.get_clock().now()

    def _stuck_detection_active_cb(self, msg: Bool) -> None:
        self._stuck_detection_active = msg.data

    def _odom_cb(self, msg: Odometry) -> None:
        q = msg.pose.pose.orientation
        self._yaw   = yaw_from_quaternion(q.x, q.y, q.z, q.w)
        self._pos_x = msg.pose.pose.position.x
        self._pos_y = msg.pose.pose.position.y
        self._last_odom_time = self.get_clock().now()

    def _scan_cb(self, msg: LaserScan) -> None:
        self._latest_scan = msg

    # ── control tick ────────────────────────────────────────────────────

    def _tick(self) -> None:
        now_s = self.get_clock().now().nanoseconds / 1e9

        forward_corridor_blocked = False
        heading_offset = None
        forward_clearance_m = float("inf")
        turn_clearance_m = float("inf")
        scan = self._latest_scan
        scan_fresh = False
        if scan is not None:
            stamp = scan.header.stamp
            scan_age_s = now_s - (stamp.sec + stamp.nanosec / 1e9)
            scan_fresh = scan_age_s <= self._stale_timeout_s
        if not scan_fresh:
            self._forward_blocked_count = 0
            self._last_counted_scan_stamp = None
            if scan is not None:
                self.get_logger().warn(
                    f"reactive_explorer: /scan is {scan_age_s:.1f}s old "
                    f"(stale_timeout_s={self._stale_timeout_s:.1f}) -- "
                    "failing safe, a stale frame must never be trusted as clear"
                )
            elif not self._scan_never_seen_warned:
                self._scan_never_seen_warned = True
                self.get_logger().warn(
                    "reactive_explorer: no /scan received yet -- STARTUP_CHECK "
                    "will wait rather than fail, but nothing can be decided "
                    "until the LiDAR driver starts publishing -- logged once."
                )
        else:
            # max(range_min, min_ignore_m) rejects the rover's own chassis/
            # mast returns as candidates for "closest obstacle" here, same
            # floor lidar_proximity_guard_node's min_valid_range() applies --
            # without it a self-view reading between range_min and
            # min_ignore_m inside the forward corridor could falsely latch
            # forward_corridor_blocked or reject an otherwise-safe heading.
            effective_range_min = max(scan.range_min, self._min_ignore_m)
            # How much room there is straight ahead, for the sweep's tie-break
            # between headings DINOv2 scores equally. Uses its own wider
            # radius than the corridor check, which only ever needs to know
            # whether the next forward_lookahead_m is clear.
            clearance_near = near_returns(
                scan.ranges, scan.angle_min, scan.angle_increment,
                effective_range_min, scan.range_max,
                corridor_box_radius_m(self._corridor_half_width_m,
                                      self._clearance_horizon_m),
                self._blocked_sectors,
            )
            forward_clearance_m = corridor_clearance_m(
                clearance_near, self._lidar_yaw_offset_rad,
                self._corridor_half_width_m,
            )
            # Rotation needs the whole circle, not the forward box: a point
            # turn sweeps the rover's corners through every bearing, so the
            # nearest return in ANY direction is what limits it. This is the
            # one place a direction-blind measure is the right question.
            turn_clearance_m = min_valid_range(
                apply_angle_mask(scan.ranges, scan.angle_min,
                                 scan.angle_increment, self._blocked_sectors),
                scan.range_min, scan.range_max, self._min_ignore_m,
            )
            stamp_key = (stamp.sec, stamp.nanosec)
            if stamp_key != self._last_counted_scan_stamp:
                self._last_counted_scan_stamp = stamp_key
                forward_clear = is_corridor_clear(
                    scan.ranges, scan.angle_min, scan.angle_increment,
                    effective_range_min, scan.range_max,
                    heading_offset_rad=self._lidar_yaw_offset_rad,
                    half_width_m=self._corridor_half_width_m,
                    far_m=self._forward_lookahead_m,
                    blocked_sectors=self._blocked_sectors,
                )
                # Require trigger_confirm_count CONSECUTIVE fresh scans before
                # believing the corridor is blocked. Restored 2026-07-27 after
                # the redesign dropped it: a single dropout in one frame would
                # otherwise commit the rover to a turn, and real C1 frames do
                # contain scattered +Inf/short returns (seen directly in the
                # 65cm-wall capture that same day).
                self._forward_blocked_count = (
                    0 if forward_clear else self._forward_blocked_count + 1
                )
            # else: same message already counted on a previous tick -- hold
            # the counter and just re-derive the flag from it below.
            forward_corridor_blocked = (
                self._forward_blocked_count >= self._trigger_confirm_count
            )
            # pick_heading_tiered() walks the full scan up to len(tiers)
            # times -- too slow to run every tick on a Pi. Only compute it
            # when the FSM might actually consult it: STARTUP_CHECK always
            # needs it, MONITORING only when about to trigger avoidance.
            if (self.fsm.state == STATE_STARTUP_CHECK or forward_corridor_blocked
                    or self._terrain_stopped or self._lidar_stopped):
                heading_offset = pick_heading_tiered(
                    scan.ranges, scan.angle_min, scan.angle_increment,
                    effective_range_min, scan.range_max,
                    rover_width_m=self._rover_width_m,
                    lookahead_tiers=self._lookahead_tiers_m,
                    angle_step_rad=self._angle_step_rad,
                    blocked_sectors=self._blocked_sectors,
                    lidar_yaw_offset_rad=self._lidar_yaw_offset_rad,
                    lateral_margin_m=self._lateral_margin_m,
                    # Headings DINOv2 already refused. Geometry cannot see
                    # terrain type, so without this the picker keeps
                    # re-proposing the same untraversable-but-clear direction.
                    exclude_offsets=self.fsm.excluded_offsets(self._yaw),
                    exclude_tolerance_rad=self._exclude_tolerance_rad,
                    min_offset_rad=correction_min_offset_rad(
                        forward_corridor_blocked, self._min_correction_turn_rad
                    ),
                )

        # Mirrors the /scan staleness check above: no /exomy/odom message
        # yet, or none for odom_stale_timeout_s, means yaw/position cannot
        # be trusted. gyro_odom_publisher_node is the publisher, and since
        # 2026-07-28 it starts unconditionally on the real-hardware launch
        # profile (it used to be gated behind use_slam, which left this
        # permanently False and the rover stuck in STARTUP_CHECK forever).
        if self._last_odom_time is None:
            odom_fresh = False
            if not self._odom_never_seen_warned:
                self._odom_never_seen_warned = True
                self.get_logger().warn(
                    "reactive_explorer: no /exomy/odom received yet -- "
                    "STARTUP_CHECK/MONITORING will wait rather than fail, but "
                    "no turn-to-heading can ever complete until odom starts "
                    "publishing -- logged once."
                )
        else:
            odom_age_s = (self.get_clock().now() - self._last_odom_time).nanoseconds / 1e9
            odom_fresh = odom_age_s <= self._odom_stale_timeout_s
            if not odom_fresh:
                self.get_logger().warn(
                    f"reactive_explorer: /exomy/odom is {odom_age_s:.1f}s old "
                    f"(odom_stale_timeout_s={self._odom_stale_timeout_s:.1f}) -- "
                    "aborting any in-progress timed state to FAILSAFE"
                )

        if self._last_imu_time is None:
            imu_fresh = False
        else:
            imu_age_s = (self.get_clock().now() - self._last_imu_time).nanoseconds / 1e9
            imu_fresh = imu_age_s <= self._imu_stale_timeout_s

        out = self.fsm.step(
            now_s=now_s, yaw=self._yaw, pos_x=self._pos_x, pos_y=self._pos_y,
            terrain_stopped=self._terrain_stopped, lidar_stopped=self._lidar_stopped,
            imu_slope_stop=self._imu_slope_stop,
            forward_corridor_blocked=forward_corridor_blocked,
            heading_offset=heading_offset,
            odom_fresh=odom_fresh, imu_fresh=imu_fresh, scan_fresh=scan_fresh,
            scan_ever_received=scan is not None,
            traversability_score=self._traversability_score,
            terrain_score_seq=self._terrain_score_seq,
            terrain_frame_informative=self._terrain_frame_informative,
            forward_clearance_m=forward_clearance_m,
            turn_clearance_m=turn_clearance_m,
        )

        active_msg = Bool()
        active_msg.data = out["active"]
        self.pub_active.publish(active_msg)

        failsafe_msg = Bool()
        failsafe_msg.data = out["failsafe"]
        self.pub_failsafe.publish(failsafe_msg)

        reason_msg = String()
        reason_msg.data = out["failsafe_reason"]
        self.pub_failsafe_reason.publish(reason_msg)

        self._log_sweep_report_once()
        self._log_new_terrain_rejections()

        translating_msg = Bool()
        translating_msg.data = out["linear_x"] != 0.0
        self.pub_translating.publish(translating_msg)

        if out["active"] and not self._stuck_detection_active:
            # stuck_detection_node takes priority -- a turn-to-heading can
            # get stuck in sand just as much as normal forward driving can,
            # so its handshake overrides this
            # node's own active flag too, not just terrain_controller_node's.
            twist = Twist()
            twist.linear.x  = out["linear_x"]
            twist.angular.z = out["angular_z"]
            self.pub_cmd_vel.publish(twist)

        if out["state"] != self._prev_state:
            self.get_logger().info(f"reactive_explorer: {self._prev_state} -> {out['state']}")
            self._prev_state = out["state"]

        if out["failsafe"] and not self._failsafe_logged:
            self._failsafe_logged = True
            self.get_logger().error(
                f"reactive_explorer: FAILSAFE ({out['failsafe_reason']}) -- "
                f"holding the rover stopped for "
                f"{self._failsafe_stop_grace_s:.1f}s, then shutting the whole "
                "launch stack down."
            )
        # Stop first, tear down second -- see should_shutdown_now().
        if not self._shutdown_triggered and should_shutdown_now(
            out["failsafe"], self.fsm.failsafe_since_s, now_s,
            self._failsafe_stop_grace_s,
        ):
            self._shutdown_triggered = True
            self.get_logger().error(
                f"reactive_explorer: FAILSAFE ({out['failsafe_reason']}) -- "
                "rover held at zero velocity, shutting down now."
            )
            self._shutdown_fn()


def main(args=None):
    rclpy.init(args=args)
    node = ReactiveExplorerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
