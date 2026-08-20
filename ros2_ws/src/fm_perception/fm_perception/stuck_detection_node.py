#!/usr/bin/env python3
"""
Purpose: Detects "commanded to move but not actually moving" (e.g. wheels
         spinning uselessly stuck in loose sand) and attempts recovery before
         giving up. Watches /exomy/cmd_vel (published by whichever node is
         currently driving -- terrain_controller_node or
         reactive_explorer_node) against /exomy/odom: over a sliding window,
         if the commanded speed implies meaningful motion but actual
         displacement is far below what that speed should have produced,
         the rover is declared stuck.
         Confirmed empirically (2026-07-16) that Gazebo's /exomy/odom is
         genuine physics-linked ground truth, not open-loop integration of
         commanded velocity: driving into a known solid obstacle at 0.10 m/s
         for 25s produced only 3.7cm of actual displacement, not the ~2.5m a
         naive encoder-style odometry would report. This makes the
         commanded-vs-actual comparison below a physically meaningful signal
         in Gazebo. On real hardware there is currently no equivalent ground
         truth (no wheel encoders, and the IMU has no gyro) -- detection
         there needs either LiDAR scan-matching (once RPLIDAR C1 is
         available) or camera-based motion detection, neither of which
         exists yet. This node is Gazebo-verified only; real-hardware
         detection is future work.
         Recovery sequence, escalating and eventually giving up rather than
         retrying forever (protects the hardware from continuously driving
         motors against something that will not move):
           1. BOOST_SPEED: try the platform's existing conservative speed
              ceiling (0.10 m/s, reused from cmd_vel_relay_node's own limit
              -- never exceeded, staying within the safety margin already
              established for this thesis) instead of whatever slower
              terrain-appropriate speed was originally commanded.
           2. WIGGLE_TURN + WIGGLE_RETRY: if boosting alone doesn't free it,
              turn a small angle (much smaller than reactive_explorer_node's
              90 deg hazard-check turn -- this is a nudge to find traction
              from a different wheel angle, not a search for a new route)
              and try driving again. Alternates left/right across repeated
              attempts.
           3. STUCK_FAILSAFE: after max_wiggle_attempts, give up permanently
              (zero Twist held) rather than keep retrying indefinitely.
         A second, independent escalation path (BOOST_ANGULAR / ROTATION_RETRY)
         covers rotational stuck-ness: commanded to turn in place (angular_z
         != 0, linear_x == 0 -- e.g. this node's own WIGGLE_TURN above, or
         reactive_explorer_node's check-left/check-right turns) but yaw isn't
         actually changing (a wheel dug in preventing rotation). The linear
         checks above never see this case since they require commanded_linear_x
         to be nonzero. Same philosophy: boost to the angular ceiling, then
         alternate turn direction up to max_wiggle_attempts before falling
         into the same shared STUCK_FAILSAFE.
         Takes priority over BOTH terrain_controller_node and
         reactive_explorer_node -- retreat and slope-retreat manoeuvres can
         get stuck in sand just as much as normal forward driving can, so
         this node's /stuck_detection/active handshake is checked by both of
         those nodes in addition to their own existing handshakes.
         The core state machine (StuckDetectionFSM) is pure Python with no
         rclpy dependency, fully unit-testable with synthetic input -- see
         test/test_stuck_detection.py. Reuses the geometry helpers already
         built and tested in reactive_explorer_node.py rather than
         duplicating them.
Inputs:  /exomy/cmd_vel (geometry_msgs/Twist)   currently commanded velocity
         /exomy/odom (nav_msgs/Odometry)         yaw + position
Outputs: /exomy/cmd_vel (geometry_msgs/Twist)    only while active
         /stuck_detection/active (std_msgs/Bool)  true while this node owns cmd_vel
         /stuck_detection/failsafe (std_msgs/Bool) true once max_wiggle_attempts is hit
How to run:
    source /opt/ros/humble/setup.bash
    source ros2_ws/install/setup.bash
    ros2 run fm_perception stuck_detection_node.py
Project: Onboard Visual Foundation Models for Mars Terrain Perception and Rover Navigation
"""
import math

from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool
import rclpy
from rclpy.node import Node

from fm_perception.reactive_explorer_node import (
    normalize_angle, yaw_from_quaternion, angle_delta, distance_2d,
)


def is_rotation_stuck(commanded_angular_z, actual_yaw_change_rad, window_s,
                       min_commanded_angular_speed=0.05, yaw_change_fraction=0.2):
    """True if commanded_angular_z implies a meaningful turn over window_s but
    actual_yaw_change_rad falls far short of the angle that speed should have
    produced. Sign-agnostic (checks magnitude only, covers either turn
    direction). Mirrors is_stuck() but for the rotational degree of freedom --
    catches a pure in-place turn (linear_x == 0, e.g. this node's own
    WIGGLE_TURN or reactive_explorer_node's check-left/check-right) getting
    physically stuck, which is_stuck() alone can never see since it requires
    commanded_linear_x to be nonzero."""
    if abs(commanded_angular_z) < min_commanded_angular_speed:
        return False
    expected = abs(commanded_angular_z) * window_s
    return abs(actual_yaw_change_rad) < yaw_change_fraction * expected


def lidar_motion_signal(ranges_a, ranges_b):
    """Mean absolute range difference between two scans, over beams that are
    finite (a real return, not inf/nan) in both. Real-hardware substitute for
    an odom-based displacement: no wheel encoders or gyro exist on the real
    rover, so "did it move" is judged from how much the raw environment
    looked different, not from a fitted pose. Returns None if there are no
    beams valid in both scans to compare (caller must not treat that as
    "didn't move" -- there was nothing to measure)."""
    diffs = [
        abs(a - b) for a, b in zip(ranges_a, ranges_b)
        if math.isfinite(a) and math.isfinite(b)
    ]
    if not diffs:
        return None
    return sum(diffs) / len(diffs)


def is_stuck_lidar(commanded_linear_x, motion_signal, min_commanded_speed=1.0,
                    stuck_motion_threshold_m=0.05):
    """True if commanded_linear_x implies meaningful motion but the LiDAR-based
    motion_signal shows the surroundings barely changed. Sign-agnostic, and
    never triggers on a None signal (insufficient data to judge).

    commanded_linear_x/min_commanded_speed are on RoverCommand.vel's own
    0-100 scale (see motors.py setDriving: driving_command/100.0*drive_range),
    NOT metres/second -- unlike StuckDetectionFSM's is_stuck() above, which
    genuinely is m/s because Gazebo drives off a geometry_msgs/Twist."""
    if abs(commanded_linear_x) < min_commanded_speed:
        return False
    if motion_signal is None:
        return False
    return motion_signal < stuck_motion_threshold_m


def front_sector_min_range(ranges, angle_min, angle_increment,
                            sector_half_width_deg=20.0, center_bearing_rad=0.0,
                            min_ignore_m=0.0):
    """Closest finite return within +/- sector_half_width_deg of
    center_bearing_rad (0 = straight ahead in the sensor frame; pi = directly
    behind). Returns None if no beam in that sector clears min_ignore_m. Used
    instead of lidar_motion_signal's whole-scan diff for judging forward
    progress: a rover yawing in place against an obstacle shifts every beam's
    bearing just as much as real forward motion would, so a whole-scan diff
    can't tell the two apart, but the closest thing directly ahead only gets
    nearer if the rover actually moved toward it.

    center_bearing_rad also lets the same function answer "what's behind the
    rover" (pass math.pi) for gating a blind reverse move before it starts --
    see RealStuckDetectionFSM.retreat_min_clearance_m.

    min_ignore_m rejects the rover's own self-view (2026-08-02): the
    centre-mounted C1 sees its own mast at ~0.072 m in a narrow ~13.5 deg
    sector (measured live in the sandpit, see
    lidar_proximity_guard_node.min_valid_range's min_ignore parameter, the
    same fix for the same sensor on this same rover). That sector's bearing
    relative to the rover is NOT assumed to avoid front or rear -- filtering
    by range instead of by a guessed bearing means this is correct regardless
    of which sector the mast happens to occupy. Defaults to 0.0 (disabled) so
    existing callers/tests are unchanged; callers reading real hardware scans
    should pass the same footprint floor lidar_proximity_guard_node uses
    (0.2 m)."""
    half = math.radians(sector_half_width_deg)
    in_sector = []
    for i, r in enumerate(ranges):
        if not math.isfinite(r) or r < min_ignore_m:
            continue
        angle = angle_min + i * angle_increment
        wrapped = math.atan2(math.sin(angle - center_bearing_rad),
                              math.cos(angle - center_bearing_rad))
        if abs(wrapped) <= half:
            in_sector.append(r)
    if not in_sector:
        return None
    return min(in_sector)


def is_stuck_lidar_front(commanded_linear_x, front_range_start, front_range_end,
                          min_commanded_speed=1.0, stuck_progress_threshold_m=0.05):
    """True if commanded_linear_x implies meaningful forward motion but the
    closest return directly ahead didn't get meaningfully closer.

    commanded_linear_x/min_commanded_speed are on RoverCommand.vel's own
    0-100 scale, NOT metres/second -- a 2026-07-25 live-hardware test found
    this matters in practice: a value that looks like a sensible small
    speed if read as m/s (0.10) is instead deep in the drive servos' own
    dead-band on the real 0-100 scale, causing exactly the stochastic
    per-wheel-friction spin this node exists to detect, rather than any
    real forward motion. Every verified-working real drive command this
    session used values in the 15-30 range.

    Only covers forward (positive) commands -- reverse-stuck detection would
    need the equivalent rear-sector reading, out of scope for now (real
    hardware has no reverse driving requirement documented yet, unlike the
    Gazebo FSM's sign-agnostic is_stuck)."""
    if commanded_linear_x < min_commanded_speed:
        # Below-threshold commands are simply not moving; negative
        # (reverse) commands are out of scope for this forward-only metric
        # -- reversing legitimately increases the front-sector range, which
        # would otherwise read as "stuck" every time.
        return False
    if front_range_start is None or front_range_end is None:
        return False
    progress = front_range_start - front_range_end
    return progress < stuck_progress_threshold_m


def is_stuck(commanded_linear_x, actual_displacement_m, window_s,
             min_commanded_speed=0.01, displacement_fraction=0.2):
    """True if commanded_linear_x implies meaningful motion over window_s but
    actual_displacement_m falls far short of the distance that speed should
    have produced. Sign-agnostic (checks magnitude only, covers reverse
    commands too)."""
    if abs(commanded_linear_x) < min_commanded_speed:
        return False
    expected = abs(commanded_linear_x) * window_s
    return actual_displacement_m < displacement_fraction * expected


# ── FSM states ───────────────────────────────────────────────────────────

STATE_MONITORING     = "MONITORING"
STATE_BOOST_SPEED    = "BOOST_SPEED"
STATE_WIGGLE_TURN    = "WIGGLE_TURN"
STATE_WIGGLE_RETRY   = "WIGGLE_RETRY"
STATE_BOOST_ANGULAR  = "BOOST_ANGULAR"
STATE_ROTATION_RETRY = "ROTATION_RETRY"
STATE_STUCK_FAILSAFE = "STUCK_FAILSAFE"
STATE_RETREAT        = "RETREAT"


class StuckDetectionFSM:
    """Pure, ROS-independent stuck-detection-and-recovery state machine.

    Call step() once per control tick with the latest known sensor values;
    it returns the commanded velocity and node status as a dict. No I/O, no
    rclpy dependency -- fully deterministic and unit-testable.
    """

    def __init__(self, stuck_window_s=4.0, stuck_displacement_fraction=0.2,
                 min_commanded_speed=0.01, boost_max_speed=0.10,
                 boost_duration_s=4.0, wiggle_angle_deg=20.0,
                 angular_speed=0.3, angle_tolerance_deg=3.0,
                 max_wiggle_attempts=3, boost_max_angular_speed=0.3,
                 min_commanded_angular_speed=0.05):
        self.stuck_window_s = stuck_window_s
        self.stuck_displacement_fraction = stuck_displacement_fraction
        self.min_commanded_speed = min_commanded_speed
        self.boost_max_speed = boost_max_speed
        self.boost_duration_s = boost_duration_s
        self.wiggle_angle_rad = math.radians(wiggle_angle_deg)
        self.angular_speed = angular_speed
        self.angle_tolerance_rad = math.radians(angle_tolerance_deg)
        self.max_wiggle_attempts = max_wiggle_attempts
        self.boost_max_angular_speed = boost_max_angular_speed
        self.min_commanded_angular_speed = min_commanded_angular_speed

        self.state = STATE_MONITORING
        self.active = False
        self.failsafe = False

        self._window_start_s = None
        self._window_start_pos = (0.0, 0.0)
        self._window_start_yaw = 0.0
        self._last_commanded_linear_x = 0.0
        self._last_commanded_angular_z = 0.0
        self._boost_target_speed = 0.0
        self._boost_start_s = None
        self._boost_start_pos = (0.0, 0.0)
        self._wiggle_attempt_count = 0
        self._wiggle_direction = 1
        self._target_yaw = 0.0
        self._retry_start_s = None
        self._retry_start_pos = (0.0, 0.0)
        self._boost_angular_sign = 1.0
        self._boost_target_angular = 0.0
        self._boost_start_yaw = 0.0
        self._retry_start_yaw = 0.0

    # ── public API ──────────────────────────────────────────────────────

    def step(self, now_s, yaw, pos_x, pos_y, commanded_linear_x, commanded_angular_z=0.0):
        if self.state == STATE_STUCK_FAILSAFE:
            return self._output(0.0, 0.0)

        if self.state == STATE_MONITORING:
            return self._step_monitoring(now_s, yaw, pos_x, pos_y,
                                          commanded_linear_x, commanded_angular_z)

        if self.state == STATE_BOOST_SPEED:
            return self._step_boost_speed(now_s, yaw, pos_x, pos_y)

        if self.state == STATE_WIGGLE_TURN:
            return self._step_wiggle_turn(now_s, yaw, pos_x, pos_y)

        if self.state == STATE_WIGGLE_RETRY:
            return self._step_wiggle_retry(now_s, yaw, pos_x, pos_y)

        if self.state == STATE_BOOST_ANGULAR:
            return self._step_boost_angular(now_s, yaw, pos_x, pos_y)

        if self.state == STATE_ROTATION_RETRY:
            return self._step_rotation_retry(now_s, yaw, pos_x, pos_y)

        raise AssertionError(f"unhandled state: {self.state}")

    # ── state handlers ──────────────────────────────────────────────────

    def _step_monitoring(self, now_s, yaw, pos_x, pos_y, commanded_linear_x, commanded_angular_z):
        linear_active = abs(commanded_linear_x) >= self.min_commanded_speed
        angular_active = abs(commanded_angular_z) >= self.min_commanded_angular_speed
        if not linear_active and not angular_active:
            self._window_start_s = None
            return self._output(0.0, 0.0)

        if self._window_start_s is None:
            self._window_start_s = now_s
            self._window_start_pos = (pos_x, pos_y)
            self._window_start_yaw = yaw
            self._last_commanded_linear_x = commanded_linear_x
            self._last_commanded_angular_z = commanded_angular_z
            return self._output(0.0, 0.0)

        self._last_commanded_linear_x = commanded_linear_x
        self._last_commanded_angular_z = commanded_angular_z
        elapsed = now_s - self._window_start_s
        if elapsed < self.stuck_window_s:
            return self._output(0.0, 0.0)

        travelled = distance_2d(
            self._window_start_pos[0], self._window_start_pos[1], pos_x, pos_y
        )
        yaw_change = angle_delta(yaw, self._window_start_yaw)
        self._window_start_s = None
        if is_stuck(self._last_commanded_linear_x, travelled, elapsed,
                    self.min_commanded_speed, self.stuck_displacement_fraction):
            self._enter_boost_speed(now_s, pos_x, pos_y, self._last_commanded_linear_x)
        elif is_rotation_stuck(self._last_commanded_angular_z, yaw_change, elapsed,
                                self.min_commanded_angular_speed, self.stuck_displacement_fraction):
            self._enter_boost_angular(now_s, yaw, self._last_commanded_angular_z)
        return self._output(0.0, 0.0)

    def _step_boost_speed(self, now_s, yaw, pos_x, pos_y):
        elapsed = now_s - self._boost_start_s
        if elapsed < self.boost_duration_s:
            return self._output(self._boost_target_speed, 0.0)

        travelled = distance_2d(
            self._boost_start_pos[0], self._boost_start_pos[1], pos_x, pos_y
        )
        if not is_stuck(self._boost_target_speed, travelled, elapsed,
                        self.min_commanded_speed, self.stuck_displacement_fraction):
            self._release()
            return self._output(0.0, 0.0)

        self._enter_wiggle_turn(yaw)
        return self._output(0.0, 0.0)

    def _step_wiggle_turn(self, now_s, yaw, pos_x, pos_y):
        delta = angle_delta(self._target_yaw, yaw)
        if abs(delta) <= self.angle_tolerance_rad:
            self.state = STATE_WIGGLE_RETRY
            self._retry_start_s = now_s
            self._retry_start_pos = (pos_x, pos_y)
            return self._output(0.0, 0.0)
        angular = self.angular_speed if delta > 0 else -self.angular_speed
        return self._output(0.0, angular)

    def _step_wiggle_retry(self, now_s, yaw, pos_x, pos_y):
        elapsed = now_s - self._retry_start_s
        if elapsed < self.boost_duration_s:
            return self._output(self._boost_target_speed, 0.0)

        travelled = distance_2d(
            self._retry_start_pos[0], self._retry_start_pos[1], pos_x, pos_y
        )
        if not is_stuck(self._boost_target_speed, travelled, elapsed,
                        self.min_commanded_speed, self.stuck_displacement_fraction):
            self._release()
            return self._output(0.0, 0.0)

        self._wiggle_attempt_count += 1
        if self._wiggle_attempt_count >= self.max_wiggle_attempts:
            self.state = STATE_STUCK_FAILSAFE
            self.failsafe = True
            return self._output(0.0, 0.0)

        self._wiggle_direction *= -1
        self._enter_wiggle_turn(yaw)
        return self._output(0.0, 0.0)

    def _step_boost_angular(self, now_s, yaw, pos_x, pos_y):
        elapsed = now_s - self._boost_start_s
        if elapsed < self.boost_duration_s:
            return self._output(0.0, self._boost_target_angular)

        yaw_change = angle_delta(yaw, self._boost_start_yaw)
        if not is_rotation_stuck(self._boost_target_angular, yaw_change, elapsed,
                                  self.min_commanded_angular_speed, self.stuck_displacement_fraction):
            self._release()
            return self._output(0.0, 0.0)

        self._enter_rotation_retry(now_s, yaw)
        return self._output(0.0, 0.0)

    def _step_rotation_retry(self, now_s, yaw, pos_x, pos_y):
        elapsed = now_s - self._retry_start_s
        if elapsed < self.boost_duration_s:
            return self._output(0.0, self._boost_target_angular)

        yaw_change = angle_delta(yaw, self._retry_start_yaw)
        if not is_rotation_stuck(self._boost_target_angular, yaw_change, elapsed,
                                  self.min_commanded_angular_speed, self.stuck_displacement_fraction):
            self._release()
            return self._output(0.0, 0.0)

        self._wiggle_attempt_count += 1
        if self._wiggle_attempt_count >= self.max_wiggle_attempts:
            self.state = STATE_STUCK_FAILSAFE
            self.failsafe = True
            return self._output(0.0, 0.0)

        self._enter_rotation_retry(now_s, yaw)
        return self._output(0.0, 0.0)

    # ── transition helpers ──────────────────────────────────────────────

    def _enter_boost_speed(self, now_s, pos_x, pos_y, last_commanded):
        self.active = True
        self.state = STATE_BOOST_SPEED
        sign = 1.0 if last_commanded >= 0.0 else -1.0
        self._boost_target_speed = sign * self.boost_max_speed
        self._boost_start_s = now_s
        self._boost_start_pos = (pos_x, pos_y)

    def _enter_wiggle_turn(self, yaw):
        self.state = STATE_WIGGLE_TURN
        self._target_yaw = normalize_angle(yaw + self._wiggle_direction * self.wiggle_angle_rad)

    def _enter_boost_angular(self, now_s, yaw, last_commanded_angular):
        self.active = True
        self.state = STATE_BOOST_ANGULAR
        self._boost_angular_sign = 1.0 if last_commanded_angular >= 0.0 else -1.0
        self._boost_target_angular = self._boost_angular_sign * self.boost_max_angular_speed
        self._boost_start_s = now_s
        self._boost_start_yaw = yaw

    def _enter_rotation_retry(self, now_s, yaw):
        # Alternates direction each attempt, same escalation philosophy as
        # WIGGLE_TURN/WIGGLE_RETRY: if turning one way doesn't free the
        # rover, try the opposite way before giving up.
        self.state = STATE_ROTATION_RETRY
        self._wiggle_direction *= -1
        self._boost_target_angular = (
            self._wiggle_direction * self._boost_angular_sign * self.boost_max_angular_speed
        )
        self._retry_start_s = now_s
        self._retry_start_yaw = yaw

    def _release(self):
        self.active = False
        self.state = STATE_MONITORING
        self._window_start_s = None
        self._wiggle_attempt_count = 0
        self._wiggle_direction = 1

    def _output(self, linear_x, angular_z):
        return {
            "state": self.state,
            "active": self.active,
            "failsafe": self.failsafe,
            "linear_x": linear_x,
            "angular_z": angular_z,
        }


class RealStuckDetectionFSM:
    """Linear-only stuck-detection-and-recovery FSM for real hardware.

    StuckDetectionFSM above is Gazebo-verified and driven by odom
    (pos_x/pos_y/yaw), which does not exist on the real rover (no wheel
    encoders, no gyro). This variant is driven by front_sector_min_range /
    is_stuck_lidar_front, not the whole-scan lidar_motion_signal/is_stuck_lidar
    pair: live testing on the real rover (2026-07-25) found that a whole-scan
    diff cannot tell "drove forward" from "yawed in place against an
    obstacle" -- both shift every beam's bearing by similar amounts, so a
    rover skidding against a wall (never actually progressing) still read as
    "moved" and was released back to MONITORING. Tracking only the closest
    return directly ahead avoids that: it only shrinks if the rover actually
    got closer to whatever is blocking it.
    The caller (the ROS2 node) computes front_sector_min_range from the raw
    LaserScan and passes the scalar in -- this class has no scan geometry of
    its own. Only covers the linear-stuck case: there is no sensor to
    confirm a wiggle turn actually rotated the rover, so WIGGLE_TURN here
    just runs for a fixed duration (wiggle_angle_deg / angular_speed) instead
    of watching a target yaw. Rotation-stuck detection (StuckDetectionFSM's
    BOOST_ANGULAR/ROTATION_RETRY) has no real-hardware equivalent and is
    deliberately not reproduced here.

    commanded_linear_x/min_commanded_speed/boost_max_speed are all on
    RoverCommand.vel's own 0-100 scale (see motors.py setDriving), NOT
    metres/second like StuckDetectionFSM's Gazebo/Twist-based fields above.
    Getting this confused is a real, live-hardware-verified failure mode
    (2026-07-25): 0.10 read as this scale sits inside the drive servos'
    dead-band, producing stochastic per-wheel spin instead of any real
    forward motion. Every verified-working real drive command this session
    used values in the 15-30 range -- boost_max_speed's default reflects
    that, not an arbitrary guess.
    """

    def __init__(self, stuck_window_s=4.0, stuck_progress_threshold_m=0.05,
                 min_commanded_speed=1.0, boost_max_speed=25.0,
                 boost_duration_s=4.0, wiggle_angle_deg=20.0,
                 angular_speed=0.3, max_wiggle_attempts=4,
                 retreat_speed=20.0, retreat_duration_s=2.0,
                 retreat_min_clearance_m=0.30):
        self.stuck_window_s = stuck_window_s
        self.stuck_progress_threshold_m = stuck_progress_threshold_m
        self.min_commanded_speed = min_commanded_speed
        self.boost_max_speed = boost_max_speed
        self.boost_duration_s = boost_duration_s
        self.wiggle_turn_duration_s = math.radians(wiggle_angle_deg) / angular_speed
        self.angular_speed = angular_speed
        self.max_wiggle_attempts = max_wiggle_attempts
        self.retreat_speed = retreat_speed
        self.retreat_duration_s = retreat_duration_s
        # 2026-08-02: the retreat move used to run its full fixed duration
        # with zero sensing at all -- a pre-hardware-run audit found that
        # WIGGLE_TURN/WIGGLE_RETRY (run up to max_wiggle_attempts times right
        # before this) drive the rover forward on a hard steering arc, not a
        # true in-place rotation (real_stuck_detection_node.py's FAKE_ACKERMANN
        # steering bands), so the rover's heading and position before a
        # retreat are not reliably "back the way it came" and a blind reverse
        # could back into something never seen on the way in. Gated on the
        # same rear-sector LiDAR read used everywhere else in this stack
        # (front_sector_min_range with center_bearing_rad=pi), same
        # not-a-veto-on-the-model philosophy as every other geometry gate
        # here: this only ever prevents a motion, it never chooses one.
        self.retreat_min_clearance_m = retreat_min_clearance_m

        self.state = STATE_MONITORING
        self.active = False
        self.failsafe = False

        self._window_start_s = None
        self._window_start_front = None
        self._last_commanded_linear_x = 0.0
        self._boost_target_speed = 0.0
        self._boost_start_s = None
        self._boost_start_front = None
        self._wiggle_start_s = None
        self._wiggle_direction = 1
        self._retry_start_s = None
        self._retry_start_front = None
        self._wiggle_attempt_count = 0
        self._retreat_start_s = None

    def step(self, now_s, front_range, commanded_linear_x, rear_range=None):
        if self.state == STATE_STUCK_FAILSAFE:
            return self._output(0.0, 0.0)
        if self.state == STATE_MONITORING:
            return self._step_monitoring(now_s, front_range, commanded_linear_x)
        if self.state == STATE_BOOST_SPEED:
            return self._step_boost_speed(now_s, front_range)
        if self.state == STATE_WIGGLE_TURN:
            return self._step_wiggle_turn(now_s, front_range)
        if self.state == STATE_WIGGLE_RETRY:
            return self._step_wiggle_retry(now_s, front_range, rear_range)
        if self.state == STATE_RETREAT:
            return self._step_retreat(now_s, rear_range)
        raise AssertionError(f"unhandled state: {self.state}")

    def _step_monitoring(self, now_s, front_range, commanded_linear_x):
        if abs(commanded_linear_x) < self.min_commanded_speed:
            self._window_start_s = None
            return self._output(0.0, 0.0)

        if self._window_start_s is None:
            self._window_start_s = now_s
            self._window_start_front = front_range
            self._last_commanded_linear_x = commanded_linear_x
            return self._output(0.0, 0.0)

        self._last_commanded_linear_x = commanded_linear_x
        elapsed = now_s - self._window_start_s
        if elapsed < self.stuck_window_s:
            return self._output(0.0, 0.0)

        self._window_start_s = None
        if is_stuck_lidar_front(self._last_commanded_linear_x,
                                 self._window_start_front, front_range,
                                 self.min_commanded_speed,
                                 self.stuck_progress_threshold_m):
            self._enter_boost_speed(now_s, front_range, self._last_commanded_linear_x)
        return self._output(0.0, 0.0)

    def _step_boost_speed(self, now_s, front_range):
        elapsed = now_s - self._boost_start_s
        if elapsed < self.boost_duration_s:
            return self._output(self._boost_target_speed, 0.0)

        if not is_stuck_lidar_front(self._boost_target_speed,
                                     self._boost_start_front, front_range,
                                     self.min_commanded_speed,
                                     self.stuck_progress_threshold_m):
            self._release()
            return self._output(0.0, 0.0)

        self._enter_wiggle_turn(now_s)
        return self._output(0.0, 0.0)

    def _step_wiggle_turn(self, now_s, front_range):
        elapsed = now_s - self._wiggle_start_s
        if elapsed < self.wiggle_turn_duration_s:
            return self._output(0.0, self._wiggle_direction * self.angular_speed)

        self.state = STATE_WIGGLE_RETRY
        self._retry_start_s = now_s
        self._retry_start_front = front_range
        return self._output(0.0, 0.0)

    def _step_wiggle_retry(self, now_s, front_range, rear_range=None):
        elapsed = now_s - self._retry_start_s
        if elapsed < self.boost_duration_s:
            return self._output(self._boost_target_speed, 0.0)

        if not is_stuck_lidar_front(self._boost_target_speed,
                                     self._retry_start_front, front_range,
                                     self.min_commanded_speed,
                                     self.stuck_progress_threshold_m):
            self._release()
            return self._output(0.0, 0.0)

        self._wiggle_attempt_count += 1
        if self._wiggle_attempt_count >= self.max_wiggle_attempts:
            if not self._rear_clear(rear_range):
                # Every wiggle attempt failed AND there is nowhere safe to
                # back into either -- give up in place rather than reverse
                # blind. See retreat_min_clearance_m above for why this
                # check exists.
                self.state = STATE_STUCK_FAILSAFE
                self.failsafe = True
                return self._output(0.0, 0.0)
            self._enter_retreat(now_s)
            return self._output(-self.retreat_speed, 0.0)

        self._wiggle_direction *= -1
        self._enter_wiggle_turn(now_s)
        return self._output(0.0, 0.0)

    def _step_retreat(self, now_s, rear_range=None):
        if not self._rear_clear(rear_range):
            # Something is now within retreat_min_clearance_m behind the
            # rover -- stop immediately rather than keep backing into it.
            # Checked every tick, not just at entry, since a 2-second reverse
            # covers real distance and the rear reading can change mid-move.
            self.state = STATE_STUCK_FAILSAFE
            self.failsafe = True
            return self._output(0.0, 0.0)

        elapsed = now_s - self._retreat_start_s
        if elapsed < self.retreat_duration_s:
            return self._output(-self.retreat_speed, 0.0)

        self.state = STATE_STUCK_FAILSAFE
        self.failsafe = True
        return self._output(0.0, 0.0)

    def _rear_clear(self, rear_range):
        # None means the rear sector returned no finite beam at all (nothing
        # within the LiDAR's own range in that direction) -- treated as
        # clear, the same convention used everywhere else this LiDAR is read
        # in this stack. A rear_range value that IS present and below
        # retreat_min_clearance_m is the only thing that blocks the move.
        if rear_range is None:
            return True
        return rear_range >= self.retreat_min_clearance_m

    def _enter_retreat(self, now_s):
        # Recovery has exhausted every wiggle attempt: back straight away for
        # a short, fixed distance before finally giving up, so the rover
        # doesn't sit unattended in whatever pose it was fighting an obstacle
        # in (e.g. front wheels resting mid-climb on a wall or sand mound).
        # Gated on rear clearance by the caller (_step_wiggle_retry) before
        # this is even entered, and rechecked every tick in _step_retreat --
        # this is a safety repositioning move, not a recovery attempt, but
        # "no progress check" no longer means "no sensing at all" (2026-08-02).
        self.state = STATE_RETREAT
        self._retreat_start_s = now_s

    def _enter_boost_speed(self, now_s, front_range, last_commanded):
        self.active = True
        self.state = STATE_BOOST_SPEED
        sign = 1.0 if last_commanded >= 0.0 else -1.0
        self._boost_target_speed = sign * self.boost_max_speed
        self._boost_start_s = now_s
        self._boost_start_front = front_range

    def _enter_wiggle_turn(self, now_s):
        self.state = STATE_WIGGLE_TURN
        self._wiggle_start_s = now_s

    def _release(self):
        self.active = False
        self.state = STATE_MONITORING
        self._window_start_s = None
        self._wiggle_attempt_count = 0
        self._wiggle_direction = 1

    def _output(self, linear_x, angular_z):
        return {
            "state": self.state,
            "active": self.active,
            "failsafe": self.failsafe,
            "linear_x": linear_x,
            "angular_z": angular_z,
        }


# ── ROS2 node wrapper ────────────────────────────────────────────────────

class StuckDetectionNode(Node):

    def __init__(self):
        super().__init__("stuck_detection_node")

        self.declare_parameter("stuck_window_s", 4.0)
        self.declare_parameter("stuck_displacement_fraction", 0.2)
        self.declare_parameter("min_commanded_speed", 0.01)
        self.declare_parameter("boost_max_speed", 0.10)
        self.declare_parameter("boost_duration_s", 4.0)
        self.declare_parameter("wiggle_angle_deg", 20.0)
        self.declare_parameter("angular_speed", 0.3)
        self.declare_parameter("angle_tolerance_deg", 3.0)
        self.declare_parameter("max_wiggle_attempts", 3)
        self.declare_parameter("boost_max_angular_speed", 0.3)
        self.declare_parameter("min_commanded_angular_speed", 0.05)
        self.declare_parameter("publish_rate_hz", 10.0)
        self.declare_parameter("cmd_vel_topic", "/exomy/cmd_vel")
        self.declare_parameter("odom_topic", "/exomy/odom")

        self.fsm = StuckDetectionFSM(
            stuck_window_s=self.get_parameter("stuck_window_s").value,
            stuck_displacement_fraction=self.get_parameter("stuck_displacement_fraction").value,
            min_commanded_speed=self.get_parameter("min_commanded_speed").value,
            boost_max_speed=self.get_parameter("boost_max_speed").value,
            boost_duration_s=self.get_parameter("boost_duration_s").value,
            wiggle_angle_deg=self.get_parameter("wiggle_angle_deg").value,
            angular_speed=self.get_parameter("angular_speed").value,
            angle_tolerance_deg=self.get_parameter("angle_tolerance_deg").value,
            max_wiggle_attempts=self.get_parameter("max_wiggle_attempts").value,
            boost_max_angular_speed=self.get_parameter("boost_max_angular_speed").value,
            min_commanded_angular_speed=self.get_parameter("min_commanded_angular_speed").value,
        )

        cmd_vel_topic = self.get_parameter("cmd_vel_topic").value
        odom_topic    = self.get_parameter("odom_topic").value
        publish_hz    = self.get_parameter("publish_rate_hz").value

        self._yaw           = 0.0
        self._pos_x         = 0.0
        self._pos_y         = 0.0
        self._commanded_linear_x = 0.0
        self._commanded_angular_z = 0.0
        self._prev_state    = self.fsm.state

        # Subscribes to the SAME topic it publishes to: while inactive, this
        # observes whichever upstream node (terrain_controller_node or
        # reactive_explorer_node) is currently driving. Once active, this
        # node is the sole publisher (both upstream nodes cede via
        # /stuck_detection/active), so its own commands are what's read back.
        self.create_subscription(Twist, cmd_vel_topic, self._cmd_vel_cb, 10)
        self.create_subscription(Odometry, odom_topic, self._odom_cb, 10)

        self.pub_cmd_vel  = self.create_publisher(Twist, cmd_vel_topic, 10)
        self.pub_active   = self.create_publisher(Bool, "/stuck_detection/active", 10)
        self.pub_failsafe = self.create_publisher(Bool, "/stuck_detection/failsafe", 10)

        self.create_timer(1.0 / publish_hz, self._tick)

        self.get_logger().info(
            "stuck_detection_node ready\n"
            f"  stuck_window_s={self.fsm.stuck_window_s:.1f} "
            f"boost_max_speed={self.fsm.boost_max_speed:.2f} "
            f"wiggle_angle={math.degrees(self.fsm.wiggle_angle_rad):.0f}deg "
            f"max_wiggle_attempts={self.fsm.max_wiggle_attempts}"
        )

    def _cmd_vel_cb(self, msg: Twist) -> None:
        self._commanded_linear_x = msg.linear.x
        self._commanded_angular_z = msg.angular.z

    def _odom_cb(self, msg: Odometry) -> None:
        q = msg.pose.pose.orientation
        self._yaw   = yaw_from_quaternion(q.x, q.y, q.z, q.w)
        self._pos_x = msg.pose.pose.position.x
        self._pos_y = msg.pose.pose.position.y

    def _tick(self) -> None:
        now_s = self.get_clock().now().nanoseconds / 1e9
        out = self.fsm.step(
            now_s=now_s, yaw=self._yaw, pos_x=self._pos_x, pos_y=self._pos_y,
            commanded_linear_x=self._commanded_linear_x,
            commanded_angular_z=self._commanded_angular_z,
        )

        active_msg = Bool()
        active_msg.data = out["active"]
        self.pub_active.publish(active_msg)

        failsafe_msg = Bool()
        failsafe_msg.data = out["failsafe"]
        self.pub_failsafe.publish(failsafe_msg)

        if out["active"]:
            twist = Twist()
            twist.linear.x  = out["linear_x"]
            twist.angular.z = out["angular_z"]
            self.pub_cmd_vel.publish(twist)

        if out["state"] != self._prev_state:
            self.get_logger().info(f"stuck_detection: {self._prev_state} -> {out['state']}")
            self._prev_state = out["state"]


def main(args=None):
    rclpy.init(args=args)
    node = StuckDetectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
