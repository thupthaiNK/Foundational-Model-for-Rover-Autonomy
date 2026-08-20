#!/usr/bin/env python3
"""
Purpose: RPLIDAR C1 proximity safety guard. Subscribes to /scan and forces a
         STOP (zero Twist) whenever any obstacle is closer than stop_distance,
         regardless of what it is (rock, equipment, a person standing or
         walking through the scan) -- this node does not classify or
         distinguish obstacle types, only distance. It is a standalone node,
         independent of safety_watchdog_node.py (terrain fail-rate E-stop) and
         terrain_controller_node.py (terrain-based speed policy); all three
         may independently command a stop, and any one of them stopping the
         rover is sufficient for safety.
         This implements Use Case 1 (Proximity Hazard Detection) from the
         RPLIDAR hardware decision. Written ahead of the RPLIDAR C1's delivery
         (ETA 2026-07-23 to 07-29) so it is ready to test the moment the
         sensor is mounted; the pure filtering/hysteresis functions are fully
         unit-testable today with synthetic LaserScan data, no hardware
         required. Deliberately out of scope: steering around obstacles,
         human-vs-object detection, or motion prediction -- those are Layer
         4/5 (path planning) per the thesis's own L1-L6 scope framing, and
         pretending otherwise without real validation would be unsafe.
         Detection and reporting (/lidar_proximity_stop, /lidar_proximity_reason)
         remain fully independent of reactive_explorer_node at all times, but
         this node's own zero-Twist /exomy/cmd_vel publish is skipped only
         while /reactive_explorer/active is true AND
         /reactive_explorer/commanding_linear_motion is false -- i.e. only
         during reactive_explorer_node's rotation-only/zero-output states
         (SCAN_AVOID_TURN, SCAN_AVOID_CONFIRM, RETREAT_TURN, MONITORING,
         FAILSAFE), which never drive the rover further into a hazard. During
         RETREAT_DRIVE/SLOPE_RETREAT, which command real translation, this
         guard's zero-Twist publish is NOT suppressed: those states' one-time
         entry-point corridor checks only validate a snapshot at the moment
         retreat begins and cannot protect against a new or moving hazard
         (e.g. a person) entering the escape path mid-leg, so the independent
         0.4m hard stop must stay live for the whole leg. This narrows the
         window where both independently-timed publishers race for the same
         topic (see _publish_status) without ever ceding cmd_vel during
         actual movement. The two nodes' publish timers are still
         independent, so a one-or-two-tick gap around each state transition
         is reduced, not fully eliminated.
Inputs:  /scan (sensor_msgs/LaserScan)
Outputs: /lidar_proximity_stop   (std_msgs/Bool)    True = obstacle too close
         /lidar_proximity_reason (std_msgs/String)  human-readable reason
         /exomy/cmd_vel (geometry_msgs/Twist)        zero velocity while stopped
Parameters:
    scan_topic     (string, default /scan)
    cmd_vel_topic  (string, default /exomy/cmd_vel)
    stop_distance_m   (float, default 0.4): trigger STOP below this range
    resume_distance_m (float, default 0.5): must clear back past this range
        before resuming -- hysteresis band prevents flicker right at the edge
    min_ignore_m (float, default 0.2): reject returns closer than this as the
        rover seeing its own chassis. The centre-mounted C1 sees ExoMy's legs/
        wheels/mast at ~0.08-0.17m; without this floor the guard STOPs forever
        on the rover itself (verified live 2026-07-23). Must be < stop_distance.
    blocked_sectors_deg (double array, default []): angular mask for the
        bearings occupied by the rover's own structure. Flat
        [start1, end1, start2, end2, ...] in degrees, each pair swept
        counter-clockwise, so [170, -170] is the 20-degree rear sector.
        Readings in these sectors are discarded. Needed because ExoMy's
        centre-mounted C1 sees its own mast and servo arms at ~0.20-0.23m,
        which sits above min_ignore_m but below stop_distance_m, latching
        STOP forever (verified live 2026-07-23/24). Derive the values from
        recorded scans with experiments/lidar_self_view_characterisation.py;
        every masked bearing is a direction the guard cannot see.
    stale_timeout_s (float, default 3.0): no /scan for this long -> treat as
        STOP (missing LiDAR data is unsafe, not "clear")
    publish_rate_hz (float, default 5.0): /lidar_proximity_stop publish rate
How to run:
    source /opt/ros/humble/setup.bash
    source ros2_ws/install/setup.bash
    ros2 run fm_perception lidar_proximity_guard_node.py
Project: Onboard Visual Foundation Models for Mars Terrain Perception and Rover Navigation
"""
import math

from fm_perception.corridor_geometry import corridor_clearance_m, near_returns

from geometry_msgs.msg import Twist
from rcl_interfaces.msg import ParameterDescriptor, ParameterType
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool, String
import rclpy
from rclpy.node import Node


def min_valid_range(ranges, range_min: float, range_max: float,
                    min_ignore: float = 0.0) -> float:
    """Closest in-spec reading in a LaserScan's ranges array.

    NaN entries are sensor noise and are ignored. A reading of +Inf (or any
    value outside [range_min, range_max]) means "nothing detected there" per
    the LaserScan convention -- that is a normal, safe reading, not an error,
    so it is excluded from the *candidates* for "closest obstacle" rather
    than treated as a fault. If no reading is in-spec, the path is clear:
    returns +Inf.

    min_ignore rejects returns closer than the rover's own footprint. On
    ExoMy the RPLIDAR C1 sits in the middle of the chassis and sees the
    rover's own legs, wheels and mast at ~0.08-0.17m in many directions, and
    the C1's range_min is well below that, so without this floor the guard
    reads a permanent sub-0.4m "obstacle" and STOPs forever. Anything closer
    than min_ignore is treated as the rover seeing itself, not a hazard.
    Defaults to 0.0 (disabled) so existing callers/tests are unchanged; the
    node sets it to the rover footprint. Effective floor is
    max(range_min, min_ignore).

    SAFETY NOTE: a real obstacle sweeps through the [min_ignore, stop_distance]
    band on its way in and trips STOP there, before it can reach the ignored
    near zone; keep min_ignore strictly below stop_distance and no larger than
    the true footprint so it never masks a genuine hazard.
    """
    floor = max(range_min, min_ignore)
    valid = [r for r in ranges if math.isfinite(r) and floor <= r <= range_max]
    return min(valid) if valid else float("inf")


def normalise_angle(angle: float) -> float:
    """Wrap an angle into the half-open interval [-pi, pi).

    LaserScan bearings run from angle_min upwards and can exceed +pi on a
    360-degree scanner depending on where angle_min sits, so every bearing
    and every mask bound is normalised before comparison. Half-open, not
    closed, so the rear bearing has exactly one representation (-pi) and a
    sector bound written as 180 or -180 behaves identically.
    """
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def parse_blocked_sectors(flat_degrees):
    """Turn a flat [start1, end1, start2, end2, ...] degree list into radian pairs.

    ROS2 parameters cannot hold a list of lists, so the mask is declared as a
    flat double array. Each pair is a sector swept counter-clockwise from
    start to end, so a sector may wrap through +/-180 (e.g. [170, -170] is
    the 20-degree rear sector). Bounds are inclusive.

    Raises ValueError on an odd-length list or an out-of-range bound: a
    mis-typed mask must fail loudly at startup rather than silently leave
    part of the rover's own structure unmasked, which is the exact failure
    this mask exists to prevent.
    """
    if len(flat_degrees) % 2 != 0:
        raise ValueError(
            f"blocked_sectors_deg must contain start/end pairs, got "
            f"{len(flat_degrees)} values (odd). A dangling bound means one "
            "sector would be silently dropped."
        )
    sectors = []
    for start, end in zip(flat_degrees[0::2], flat_degrees[1::2]):
        for bound in (start, end):
            if not -360.0 <= bound <= 360.0:
                raise ValueError(
                    f"blocked_sectors_deg bound {bound} is not a plausible "
                    "angle in degrees (expected within +/-360)."
                )
        sectors.append((normalise_angle(math.radians(start)),
                        normalise_angle(math.radians(end))))
    return sectors


def sector_width(start: float, end: float) -> float:
    """Counter-clockwise angular width of a sector, in radians (0 to 2*pi)."""
    return (end - start) % (2.0 * math.pi)


def angle_is_blocked(angle: float, sectors) -> bool:
    """True if this bearing falls inside any blocked sector (bounds inclusive)."""
    a = normalise_angle(angle)
    for start, end in sectors:
        # Measure both the query angle and the sector end as offsets
        # counter-clockwise from the sector start, which makes a wrapping
        # sector no different from a non-wrapping one.
        offset = (a - start) % (2.0 * math.pi)
        if offset <= sector_width(start, end) + 1e-9:
            return True
    return False


def apply_angle_mask(ranges, angle_min: float, angle_increment: float, sectors):
    """Replace readings whose bearing is inside a blocked sector with +Inf.

    +Inf is the LaserScan convention for "nothing detected", so masked
    bearings flow through both min_valid_range() and near_returns() as clear
    rather than needing any special case in either.

    WHY THIS EXISTS (verified live 2026-07-23/24): the RPLIDAR C1 is mounted
    at the centre of ExoMy's top plate and sees the rover's own mast and
    servo arms at ~0.20-0.23m. That is beyond min_ignore_m (0.2, sized for
    the chassis at ~0.075m) but inside stop_distance_m (0.4), so a naive
    360-degree distance guard latches STOP forever on the rover itself.
    Raising min_ignore instead would leave a uselessly thin 0.3-0.4m
    detection band. Masking the bearings the structure occupies is the
    correct fix, because the obstruction is angular, not radial.

    SAFETY NOTE: every masked bearing is a direction in which this guard is
    blind. Mask only the sectors real structure occupies, measured from
    recorded scans (experiments/lidar_self_view_characterisation.py), never
    a sector guessed to make the STOP go away.
    """
    if not sectors:
        return list(ranges)
    out = list(ranges)
    for i in range(len(out)):
        if angle_is_blocked(angle_min + i * angle_increment, sectors):
            out[i] = float("inf")
    return out


def masked_fraction(sectors) -> float:
    """Fraction of the full 360 degrees the mask covers, for startup reporting.

    Overlapping sectors are counted twice, so this is an upper bound. It is
    used only to warn the operator how blind the guard has been made, never
    to change behaviour.
    """
    total = sum(sector_width(start, end) for start, end in sectors)
    return total / (2.0 * math.pi)


def nearest_forward_obstacle_m(ranges, angle_min: float, angle_increment: float,
                               range_min: float, range_max: float,
                               half_width_m: float,
                               heading_offset_rad: float = 0.0) -> float:
    """Distance to the nearest obstacle inside the rover's swept width along
    heading_offset_rad, or +Inf if the way ahead is clear.

    Replaces min_valid_range() as the guard's stop measure on 2026-07-29.
    min_valid_range answers "is anything close in ANY direction", which is not
    the question a forward-driving rover needs, and it was wrong in both
    directions at once: a wall 0.25 m to the side of a rover that fits past it
    latched a permanent STOP, while an obstacle off the front-left corner read
    as a safe 0.39 m straight-line range right up until a wheel hit it. What
    matters is how far AHEAD an obstacle is and whether it is inside the width
    the rover covers.

    Uses the same rectangle as the heading planner (corridor_geometry), so the
    two layers agree on what "in the way" means and differ only in how far
    ahead each looks: this guard watches stop_distance, the planner watches
    the full lookahead.

    `ranges` should already have self-view sectors masked (apply_angle_mask).
    """
    near = near_returns(ranges, angle_min, angle_increment, range_min,
                        range_max, float("inf"))
    return corridor_clearance_m(near, heading_offset_rad, half_width_m)


def should_stop(min_range: float, currently_stopped: bool,
                stop_distance: float, resume_distance: float) -> bool:
    """Hysteresis so a reading right at the boundary doesn't flicker STOP/GO.

    Mirrors the stop_latch_s / resume_confirm_count pattern already used in
    terrain_controller_node.py, but keyed on distance instead of consecutive
    message counts, since a proximity reading is inherently continuous.
    """
    if currently_stopped:
        return min_range < resume_distance
    return min_range < stop_distance


class LidarProximityGuardNode(Node):

    def __init__(self):
        super().__init__("lidar_proximity_guard_node")

        # ── Parameters ─────────────────────────────────────────────────
        self.declare_parameter("scan_topic", "/scan")
        self.declare_parameter("cmd_vel_topic", "/exomy/cmd_vel")
        self.declare_parameter("stop_distance_m", 0.4)
        self.declare_parameter("resume_distance_m", 0.5)
        self.declare_parameter("stale_timeout_s", 3.0)
        self.declare_parameter("publish_rate_hz", 5.0)
        # Reject the rover's own chassis/legs/wheels/mast, which the
        # centre-mounted C1 sees at ~0.08-0.17m. Verified live 2026-07-23:
        # without this the guard STOPs permanently on the rover itself.
        self.declare_parameter("min_ignore_m", 0.2)
        # Half the width of ground the rover sweeps, chassis half-width plus
        # margin. MUST match reactive_explorer_node's rover_width_m/2 +
        # lateral_margin_m: the two nodes each declare their own copy and do
        # not share parameters, and if the guard's box were narrower than the
        # planner's the planner could commit to a heading the guard then
        # refuses to drive.
        self.declare_parameter("half_width_m", 0.20)
        # Where the rover's physical front sits in this scan's bearings.
        # MUST match reactive_explorer_node's lidar_yaw_offset_deg. Before
        # 2026-07-29 the guard was direction-blind and did not need it;
        # measuring down a rectangle does. Re-measure with
        # experiments/measure_lidar_yaw_offset_direct.py if the LiDAR moves.
        self.declare_parameter("lidar_yaw_offset_deg", 3.5)
        # Angular mask for the bearings the rover's own structure occupies.
        # Flat [start1, end1, start2, end2, ...] in degrees, each pair swept
        # counter-clockwise (so [170, -170] is the 20-degree rear sector).
        # Empty by default = full 360-degree coverage, unchanged behaviour.
        # Derive the real values with
        # experiments/lidar_self_view_characterisation.py, do not guess them.
        # dynamic_typing is required, not cosmetic: rclpy infers BYTE_ARRAY
        # from a bare [] default and then rejects a DOUBLE_ARRAY override
        # with InvalidParameterTypeException, so a statically-typed empty
        # default makes the mask impossible to set.
        self.declare_parameter(
            "blocked_sectors_deg", [],
            ParameterDescriptor(dynamic_typing=True),
        )

        scan_topic       = self.get_parameter("scan_topic").get_parameter_value().string_value
        cmd_vel_topic    = self.get_parameter("cmd_vel_topic").get_parameter_value().string_value
        self.stop_distance    = self.get_parameter("stop_distance_m").get_parameter_value().double_value
        self.resume_distance  = self.get_parameter("resume_distance_m").get_parameter_value().double_value
        self.stale_s      = self.get_parameter("stale_timeout_s").get_parameter_value().double_value
        publish_hz        = self.get_parameter("publish_rate_hz").get_parameter_value().double_value
        self.min_ignore   = self.get_parameter("min_ignore_m").get_parameter_value().double_value
        self.half_width_m = self.get_parameter("half_width_m").get_parameter_value().double_value
        self.lidar_yaw_offset_rad = math.radians(
            self.get_parameter("lidar_yaw_offset_deg").get_parameter_value().double_value
        )
        blocked_sectors_param = self.get_parameter("blocked_sectors_deg").get_parameter_value()
        if blocked_sectors_param.type == ParameterType.PARAMETER_INTEGER_ARRAY:
            sectors_deg = [float(v) for v in blocked_sectors_param.integer_array_value]
        else:
            sectors_deg = [float(v) for v in blocked_sectors_param.double_array_value]
        # Raises on a malformed mask -- see parse_blocked_sectors().
        self.blocked_sectors = parse_blocked_sectors(sectors_deg)
        self._masked_fraction = masked_fraction(self.blocked_sectors)

        if self._masked_fraction >= 1.0:
            raise ValueError(
                "blocked_sectors_deg masks the entire 360-degree scan, which "
                "would make this guard permanently blind and silently report "
                "a clear path. Mask only the sectors the rover's own "
                "structure occupies."
            )

        if self.min_ignore >= self.stop_distance:
            raise ValueError(
                f"min_ignore_m ({self.min_ignore}) must be < stop_distance_m "
                f"({self.stop_distance}); otherwise the near zone the guard "
                "ignores overlaps the zone it must STOP for, masking real "
                "hazards."
            )

        if self.resume_distance <= self.stop_distance:
            raise ValueError(
                f"resume_distance_m ({self.resume_distance}) must be > "
                f"stop_distance_m ({self.stop_distance}) or the hysteresis band "
                "collapses and the guard will flicker."
            )

        # ── State ──────────────────────────────────────────────────────
        self._stopped = False
        self._reason = ""
        self._last_scan_time = None
        self._explorer_active = False
        self._explorer_translating = False

        # ── ROS2 pub / sub ─────────────────────────────────────────────
        self.create_subscription(LaserScan, scan_topic, self._scan_callback, 10)
        self.create_subscription(Bool, "/reactive_explorer/active", self._explorer_active_cb, 10)
        self.create_subscription(
            Bool, "/reactive_explorer/commanding_linear_motion",
            self._explorer_translating_cb, 10,
        )
        self._pub_stop   = self.create_publisher(Bool, "/lidar_proximity_stop", 10)
        self._pub_reason = self.create_publisher(String, "/lidar_proximity_reason", 10)
        self._pub_cmd    = self.create_publisher(Twist, cmd_vel_topic, 10)

        self.create_timer(1.0 / publish_hz, self._publish_status)
        self.create_timer(1.0, self._stale_watchdog)

        self.get_logger().info(
            f"lidar_proximity_guard_node ready\n"
            f"  Listening:  {scan_topic}\n"
            f"  Stop out:   /lidar_proximity_stop, /lidar_proximity_reason, {cmd_vel_topic}\n"
            f"  Stop distance:   {self.stop_distance:.2f}m\n"
            f"  Resume distance: {self.resume_distance:.2f}m\n"
            f"  Stale timeout:   {self.stale_s:.1f}s\n"
            f"  Min ignore:      {self.min_ignore:.2f}m\n"
            f"  Corridor half-width: {self.half_width_m:.2f}m\n"
            f"  LiDAR yaw offset:    {math.degrees(self.lidar_yaw_offset_rad):.1f} deg\n"
            f"  Blocked sectors: {sectors_deg if sectors_deg else 'none (full 360 coverage)'}\n"
            "  No obstacle classification -- anything inside the forward corridor stops the rover."
        )
        if self.blocked_sectors:
            self.get_logger().warn(
                f"Angle mask active: {self._masked_fraction * 100:.1f}% of the "
                "scan is ignored. The guard is BLIND in those directions by "
                "design (they are the rover's own structure). Re-measure the "
                "mask with experiments/lidar_self_view_characterisation.py "
                "after any change to the top plate or mast."
            )

    # ── /scan callback ───────────────────────────────────────────────────

    def _scan_callback(self, msg: LaserScan) -> None:
        self._last_scan_time = self.get_clock().now()

        ranges = apply_angle_mask(msg.ranges, msg.angle_min,
                                  msg.angle_increment, self.blocked_sectors)
        # Measured down the rover's swept rectangle rather than as a
        # direction-blind nearest return -- see nearest_forward_obstacle_m.
        r = nearest_forward_obstacle_m(
            ranges, msg.angle_min, msg.angle_increment,
            max(msg.range_min, self.min_ignore), msg.range_max,
            half_width_m=self.half_width_m,
            heading_offset_rad=self.lidar_yaw_offset_rad,
        )
        stop = should_stop(r, self._stopped, self.stop_distance, self.resume_distance)

        if stop != self._stopped:
            transition = (
                f"LiDAR proximity STOP {'engaged' if stop else 'cleared'} "
                f"(forward_clearance={r:.2f}m, stop_distance={self.stop_distance:.2f}m, "
                f"resume_distance={self.resume_distance:.2f}m)"
            )
            # rclpy caches severity per call site and refuses to change it
            # between calls ("Logger severity cannot be changed between
            # calls"), so warn and info must be two distinct call sites, not
            # a single `log(...)` bound to either method.
            if stop:
                self.get_logger().warn(transition)
            else:
                self.get_logger().info(transition)

        self._stopped = stop
        self._reason = (
            f"Obstacle {r:.2f}m ahead < stop_distance {self.stop_distance:.2f}m"
            if stop else ""
        )

    def _explorer_active_cb(self, msg: Bool) -> None:
        self._explorer_active = msg.data

    def _explorer_translating_cb(self, msg: Bool) -> None:
        self._explorer_translating = msg.data

    # ── Stale watchdog (no /scan for stale_timeout_s) ────────────────────

    def _stale_watchdog(self) -> None:
        if self._last_scan_time is None:
            return
        elapsed = (self.get_clock().now() - self._last_scan_time).nanoseconds / 1e9
        if elapsed > self.stale_s and not self._stopped:
            self.get_logger().warn(
                f"No /scan for {elapsed:.1f}s -- forcing STOP (missing LiDAR data)"
            )
            self._stopped = True
            self._reason = f"No /scan for {elapsed:.1f}s (timeout={self.stale_s:.1f}s)"

    # ── Periodic publish ──────────────────────────────────────────────────

    def _publish_status(self) -> None:
        stop_msg = Bool()
        stop_msg.data = self._stopped
        self._pub_stop.publish(stop_msg)

        reason_msg = String()
        reason_msg.data = self._reason
        self._pub_reason.publish(reason_msg)

        if self._stopped and not (self._explorer_active and not self._explorer_translating):
            # Suppress this guard's own zero-Twist publish ONLY while
            # reactive_explorer_node is active AND currently commanding zero
            # linear velocity -- i.e. only during its rotation-only/
            # zero-output states (SCAN_AVOID_TURN, SCAN_AVOID_CONFIRM,
            # RETREAT_TURN, MONITORING, FAILSAFE). During RETREAT_DRIVE/
            # SLOPE_RETREAT (real translation) this guard stays fully live:
            # those states' one-time entry corridor checks only validate a
            # snapshot at the start of the leg and cannot see a new or moving
            # hazard entering the escape path mid-leg, so the independent
            # 0.4m hard stop must not be ceded for the whole leg duration.
            # This narrows (does not fully close -- the two nodes' publish
            # timers are independent) the window where both race to publish
            # conflicting Twist messages on this topic.
            self._pub_cmd.publish(Twist())  # zero velocity


def main(args=None):
    rclpy.init(args=args)
    node = LidarProximityGuardNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutdown -- final STOP command sent")
        node._pub_cmd.publish(Twist())
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
