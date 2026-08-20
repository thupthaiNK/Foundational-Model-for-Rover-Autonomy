#!/usr/bin/env python3
"""
Purpose: "L5-lite" -- a lightweight waypoint path planner and pure-pursuit
         follower, scoped via grill-thesis 2026-07-17 as an alternative to
         the full Nav2 stack that D1 (§4.8.13) found alone drove load
         average to 9.33 on this 4-core development machine (confirmed
         independent of DINOv2, via the Condition A re-run removing DINOv2
         and still getting a null result). Combines astar_planner.py (pure
         A* over the existing OccupancyGridBuilder costmap) and
         path_follower.py (pure-pursuit) into a single ROS2 node. Uses
         odom-assisted slam_toolbox's /pose (§4.8.22, ~0.18 m mean error)
         as its localisation source -- the only source that has ever
         actually worked in this thesis; odom-free SLAM was investigated
         across 9 configurations/5 rounds and never produced a usable
         result (§4.8.23), so it is explicitly not an option here. This
         means this node's result is Gazebo-only and not real-hardware-
         representative, matching the same caveat pattern already used
         honestly elsewhere (Phase A/A2, D1, §5.5.4d-g).
         Priority: lowest of the four driving nodes -- cedes cmd_vel to
         reactive_explorer_node and stuck_detection_node exactly like
         terrain_controller_node.py does, via the same /reactive_explorer/
         active and /stuck_detection/active flags. Publishes its own
         /l5_lite_planner/active so terrain_controller_node.py can cede to
         it in turn (added as a new opt-in check there, same pattern as the
         existing two).
         Replans periodically (default every 2.5 s), not every control
         tick like Nav2 -- astar_path() on the real 300x240 cell grid over
         a route the same length as D1's goal took ~4 ms measured directly
         (see thesis write-up), so a multi-second cadence has large margin.
         If no path is found (goal unreachable through non-hazard cells
         given the current costmap), the rover stops and waits rather than
         attempting anything -- the costmap keeps updating from live DINOv2
         classifications, so this can self-resolve on a later replan.
         Bootstrap crawl (found via systematic debugging, 2026-07-17): a
         stationary rover's odom->base_link TF never changes, so odom-
         assisted slam_toolbox's minimum_travel_distance/heading gate is
         never crossed and it never publishes even a first /pose -- but
         planning needs /pose, and moving needs a plan. Chicken-and-egg
         deadlock, confirmed by direct log evidence (grid arrived in <100ms,
         pose stayed missing 40+s). Fixed by crawling straight ahead at
         linear_speed until the first /pose arrives (path_follower.py's
         bootstrap_crawl_command()), purely to seed scan-matching -- not a
         change to slam_toolbox itself or its already-proven config.
         Hybrid rotate/creep (found via systematic debugging, 2026-07-17,
         reopened session): after the above fix, D1's mission (requiring an
         initial ~175deg turn) still made zero net progress. Three
         hypotheses -- scan_queue_size, a slower rotation rate, and a
         feature-rich (near Q4) spawn point -- were each tested live in
         Gazebo and none fixed it; ground truth confirmed the rover really
         was rotating throughout, but slam_toolbox never produced a second
         /pose during CONTINUOUS pure rotation (zero translation) in any of
         the three configurations. hybrid_rotate_command() (path_follower.py)
         alternates short rotate-only bursts with short creep-only bursts
         instead of committing to continuous rotation, so the scan-matcher
         periodically gets translation to anchor a new pose estimate on.
Inputs:  /traversability_costmap (nav_msgs/OccupancyGrid) -- from
             live_traversability_costmap_node.py, pose_source="slam"
         /pose (geometry_msgs/PoseWithCovarianceStamped) -- from
             odom-assisted slam_toolbox
         /reactive_explorer/active (std_msgs/Bool)
         /stuck_detection/active (std_msgs/Bool)
Outputs: /exomy/cmd_vel (geometry_msgs/Twist)
         /l5_lite_plan (nav_msgs/Path) -- the current planned path, for
             visualisation/logging
         /l5_lite_planner/active (std_msgs/Bool)
         "L6-lite" waypoint sequencing (scoped via grill-thesis 2026-07-18,
         built after L5-lite first reached a goal end-to-end): the smallest
         possible sliver of L6 (Mission Autonomy) -- autonomously advancing
         to the next waypoint in a short list when the current one is
         reached, with no operator command in between, instead of stopping.
         Not intelligent goal selection, just sequencing; framed honestly in
         the thesis as exactly that, not as general mission autonomy (no
         resource detection, no reasoning about what to explore -- both
         blocked by hardware ExoMy does not have, a monocular RGB camera
         only). waypoint_xs/waypoint_ys (optional, default empty) name
         waypoints *after* goal_x/goal_y; when empty, behaviour is
         byte-for-byte identical to every pre-L6-lite L5-lite result (a
         single-waypoint mission). advance_waypoint_index() (path_follower.py)
         is the pure function deciding when to switch.
         "Explore-then-return-home" (scoped via grill-thesis 2026-07-19,
         frontier_mode only): once no frontiers remain, instead of ending
         the mission there, the rover autonomously drives back to its own
         first recorded /pose and only then declares the mission complete
         -- opt-in via return_home (default false, zero behaviour change to
         every pre-existing frontier-exploration-lite result), gated to
         require frontier_mode=true (inert otherwise, since there is no
         "exploration complete" signal outside frontier mode to trigger
         from). should_transition_to_return_home() (path_follower.py) is
         the pure function deciding when to switch to the return leg;
         the leg itself reuses frontier mode's existing planning path
         (unknown-as-free, grid_with_start_freed) and goal_reached(), no
         new planning logic. Mirrors L6-lite's own success bar: the
         primary claim is the autonomous transition itself firing with no
         operator command, not that the return leg necessarily completes
         within any given time budget.
         "A2 re-observation mode" (scoped via grill-thesis 2026-07-19,
         frontier_mode only, opt-in via reobserve_mode): on frontier
         exhaustion, instead of ending (or going home), autonomously
         revisit the known non-hazard cell with the LOWEST recorded
         confidence (from /traversability_confidence, published by the
         costmap node under track_confidence:=true), one cell at a time --
         visited-set so no cell is selected twice, nearest tiebreak among
         equal minima, same min-distance guard as frontier selection.
         Driving at the cell re-observes it automatically (the paint disc
         sits 0.6 m ahead of the rover). should_transition_to_reobservation()
         and select_lowest_confidence_cell() are the pure functions;
         per-selection invariant logging (selected_conf=/pool_min=)
         provides the pre-registered P2 evidence. Precedence: runs before
         return_home; if both are enabled, home fires only once
         re-observation itself exhausts.
ROS2 parameters:
         goal_x, goal_y (float, default -7.5, -9.0) -- same goal D1 used
             (§4.8.13, sand_zone opposite side), for direct comparability.
             This is waypoint 0; see waypoint_xs/waypoint_ys below for
             further waypoints.
         waypoint_xs, waypoint_ys (float array, default []) -- waypoints
             *after* goal_x/goal_y, visited in order once each prior one is
             reached. Empty by default -- zero behaviour change from every
             pre-L6-lite result.
         goal_tolerance_m (float, default 0.2)
         linear_speed (float, default 0.10) -- matches this thesis's
             established real-world speed ceiling
         angular_gain (float, default 2.0)
         max_angular_z (float, default 0.3) -- see path_follower.py's pure_pursuit_command()
         lookahead_m (float, default 0.6) -- matches the costmap's own
             lookahead_m default (traversability_grid.py)
         replan_period_s (float, default 2.5)
         control_rate_hz (float, default 10.0)
         hybrid_rotate_phase_s (float, default 3.0) -- see path_follower.py's
             hybrid_rotate_command()
         hybrid_creep_phase_s (float, default 2.0)
         hybrid_creep_speed (float, default 0.05)
How to run:
    source /opt/ros/humble/setup.bash
    source ros2_ws/install/setup.bash
    ros2 run fm_perception l5_lite_planner_node.py
Project: Onboard Visual Foundation Models for Mars Terrain Perception and Rover Navigation
"""
import math
import time

import rclpy
from geometry_msgs.msg import Pose, PointStamped, PoseStamped, PoseWithCovarianceStamped, Twist
from nav_msgs.msg import OccupancyGrid, Path
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool

from fm_perception.astar_planner import astar_path, path_to_world
from fm_perception.frontier_explorer import (
    count_known_cells_in_box, find_frontier_cells, grid_for_planning,
    grid_with_start_freed, is_bedrock_adjacent, select_lowest_confidence_cell,
    select_nearest_frontier,
)
from fm_perception.path_follower import (
    advance_waypoint_index, bootstrap_crawl_command, find_lookahead_point, goal_reached,
    hybrid_rotate_command, pure_pursuit_command, should_abort_to_home,
    should_transition_to_reobservation,
    should_transition_to_return_home, should_use_hybrid_rotation,
)
from fm_perception.traversability_grid import (
    COST_HAZARD, HEIGHT_CELLS, RESOLUTION_M, WIDTH_CELLS, world_to_cell,
)


def _yaw_from_quaternion(q) -> float:
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


class L5LitePlannerNode(Node):

    def __init__(self):
        super().__init__("l5_lite_planner_node")

        self.declare_parameter("goal_x", -7.5)
        self.declare_parameter("goal_y", -9.0)
        # Empty-list defaults need to declare the TYPE, not a [] default value
        # -- rclpy infers the array element type from the Python default
        # value regardless of any ParameterDescriptor.type set explicitly
        # (confirmed by reading rclpy/node.py's declare_parameters()), and
        # an empty list infers as BYTE_ARRAY, which then rejects a real
        # float-array launch override with InvalidParameterTypeException
        # (found live, 2026-07-18). Declaring the type directly (no default
        # value) avoids the inference step entirely; unset value reads back
        # as None, handled explicitly below.
        self.declare_parameter("waypoint_xs", rclpy.Parameter.Type.DOUBLE_ARRAY)
        self.declare_parameter("waypoint_ys", rclpy.Parameter.Type.DOUBLE_ARRAY)
        self.declare_parameter("goal_tolerance_m", 0.2)
        self.declare_parameter("linear_speed", 0.10)
        self.declare_parameter("angular_gain", 2.0)
        self.declare_parameter("max_angular_z", 0.3)
        self.declare_parameter("lookahead_m", 0.6)
        self.declare_parameter("replan_period_s", 2.5)
        self.declare_parameter("control_rate_hz", 10.0)
        self.declare_parameter("hybrid_rotate_phase_s", 3.0)
        self.declare_parameter("hybrid_creep_phase_s", 2.0)
        self.declare_parameter("hybrid_creep_speed", 0.05)
        # -- Frontier exploration-lite (scoped via grill-thesis 2026-07-18).
        # Default false: zero behaviour change from every pre-existing
        # L5-lite/L6-lite result. When true, goals are selected autonomously
        # (nearest frontier of the perception-explored region inside the box
        # below) instead of taken from goal_x/goal_y/waypoint lists, and the
        # costmap is expected to run with init_unknown:=true.
        self.declare_parameter("frontier_mode", False)
        self.declare_parameter("frontier_box_x_min", -9.0)
        self.declare_parameter("frontier_box_x_max", -3.0)
        self.declare_parameter("frontier_box_y_min", 3.0)
        self.declare_parameter("frontier_box_y_max", 9.0)
        # Above goal_tolerance (0.2) so a fresh selection can't already be
        # "reached"; below the first painted patch's outer frontier ring
        # (~0.9) so a first selection is always possible. See
        # select_nearest_frontier()'s docstring for the smoke-test bug
        # this prevents.
        self.declare_parameter("min_frontier_distance_m", 0.5)
        # -- Semantic-biased frontier selection (scoped via grill-thesis
        # 2026-07-18 night). Default false: zero behaviour change from the
        # be39399 official frontier result. When true, frontier selection
        # gets lexicographic bedrock-adjacency priority (science-interest
        # bias, Candela & Wettergreen 2022 framing) via
        # select_nearest_frontier's grid/width parameters.
        self.declare_parameter("semantic_frontier", False)
        # -- Explore-then-return-home (scoped via grill-thesis 2026-07-19).
        # Default false: zero behaviour change from every pre-existing
        # frontier-exploration-lite result. When true (frontier_mode only --
        # inert otherwise), once no frontiers remain the rover autonomously
        # drives back to its own first recorded /pose instead of stopping
        # there, and only then declares the mission complete. See
        # should_transition_to_return_home() (path_follower.py).
        self.declare_parameter("return_home", False)
        # -- A2 re-observation mode (scoped via grill-thesis 2026-07-19,
        # frontier_mode only). Default false: zero behaviour change from
        # every pre-existing frontier result. When true, frontier
        # exhaustion transitions (once, autonomously) into revisiting the
        # known non-hazard cell with the LOWEST recorded confidence, one
        # cell at a time (visited-set, nearest tiebreak, same min-distance
        # guard as frontier selection), until nothing is selectable.
        # Requires the costmap node to run with track_confidence:=true --
        # without /traversability_confidence data the transition refuses
        # (fails loud-and-safe) and the mission ends normally. Precedence
        # over return_home: re-observation runs first; return_home (if also
        # enabled) fires only once re-observation itself exhausts.
        self.declare_parameter("reobserve_mode", False)
        # -- Mission-level failsafe reaction (item 12, grill-scoped
        # 2026-07-20, frontier_mode only). Default false: zero behaviour
        # change from every pre-existing result. When true, a
        # reactive_explorer terminal FAILSAFE (it has given up on a
        # specific hazard but the rover is still mobile) autonomously
        # aborts the current exploration and drives back to the rover's
        # own recorded start pose -- reusing return_home's own goal-switch
        # machinery, but triggered by failsafe instead of frontier
        # exhaustion, and recording its own home pose independently of
        # return_home so either can be enabled alone. Deliberately does NOT
        # respond to stuck_detection's failsafe (wheels commanded but the
        # rover not physically displacing -- "drive home" would be
        # physically meaningless there); see should_abort_to_home()
        # (path_follower.py). One-way latch: once fired, takes precedence
        # over every other mode (reobserve_mode, return_home, waypoint
        # sequencing) for the rest of the mission, and the planner stops
        # ceding cmd_vel to reactive_explorer (which has already given up)
        # while continuing to cede to stuck_detection (a still-valid,
        # independent safety mechanism).
        self.declare_parameter("abort_to_home", False)

        goal_x = self.get_parameter("goal_x").value
        goal_y = self.get_parameter("goal_y").value
        # get_parameter_or(), not get_parameter(): an uninitialized
        # statically-typed array parameter (no default value, no override)
        # raises ParameterUninitializedException from get_parameter().
        waypoint_xs = self.get_parameter_or(
            "waypoint_xs", rclpy.parameter.Parameter("waypoint_xs", value=[])
        ).value
        waypoint_ys = self.get_parameter_or(
            "waypoint_ys", rclpy.parameter.Parameter("waypoint_ys", value=[])
        ).value
        self._waypoints = [(goal_x, goal_y)] + list(zip(waypoint_xs, waypoint_ys))
        self._waypoint_index = 0
        self._goal_x, self._goal_y = self._waypoints[self._waypoint_index]
        self._goal_tolerance_m = self.get_parameter("goal_tolerance_m").value
        self._linear_speed = self.get_parameter("linear_speed").value
        self._angular_gain = self.get_parameter("angular_gain").value
        self._max_angular_z = self.get_parameter("max_angular_z").value
        self._lookahead_m = self.get_parameter("lookahead_m").value
        replan_period_s = self.get_parameter("replan_period_s").value
        control_rate_hz = self.get_parameter("control_rate_hz").value
        self._hybrid_rotate_phase_s = self.get_parameter("hybrid_rotate_phase_s").value
        self._hybrid_creep_phase_s = self.get_parameter("hybrid_creep_phase_s").value
        self._hybrid_creep_speed = self.get_parameter("hybrid_creep_speed").value

        self._frontier_mode = self.get_parameter("frontier_mode").value
        if self._frontier_mode:
            c1 = world_to_cell(self.get_parameter("frontier_box_x_min").value,
                                self.get_parameter("frontier_box_y_min").value)
            c2 = world_to_cell(self.get_parameter("frontier_box_x_max").value,
                                self.get_parameter("frontier_box_y_max").value)
            self._frontier_box = (min(c1[0], c2[0]), min(c1[1], c2[1]),
                                   max(c1[0], c2[0]), max(c1[1], c2[1]))
            self._frontier_blacklist = set()
            self._frontier_count = 0
            self._min_frontier_cells = (
                self.get_parameter("min_frontier_distance_m").value / RESOLUTION_M)
            self._semantic_frontier = self.get_parameter("semantic_frontier").value
            self._return_home = self.get_parameter("return_home").value
            self._returning_home = False
            self._home_xy = None     # (x, y), set from the first /pose received
            self._reobserve_mode = self.get_parameter("reobserve_mode").value
            self._reobserving = False
            self._reobserve_visited = set()   # cells selected this mission, reached or not
            self._reobserve_count = 0
            self._confidence_grid = None      # latest /traversability_confidence data
            self._abort_to_home = self.get_parameter("abort_to_home").value
            self._aborting_to_home = False
            self._reactive_failsafe = False

        self._grid = None            # latest costmap data, list[int]
        self._pose = None            # (x, y, yaw)
        self._path_world = []        # list[(x, y)], current plan
        self._reached = False
        self._explorer_active = False
        self._stuck_detection_active = False
        self._start_time = time.time()  # reference for hybrid_rotate_command's cycle_time_s

        costmap_qos = QoSProfile(depth=1)
        costmap_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        costmap_qos.reliability = ReliabilityPolicy.RELIABLE
        self.create_subscription(OccupancyGrid, "/traversability_costmap",
                                  self._costmap_cb, costmap_qos)
        if self._frontier_mode and self._reobserve_mode:
            self.create_subscription(OccupancyGrid, "/traversability_confidence",
                                      self._confidence_cb, costmap_qos)
        self.create_subscription(PoseWithCovarianceStamped, "/pose", self._pose_cb, 10)
        self.create_subscription(Bool, "/reactive_explorer/active", self._explorer_cb, 10)
        self.create_subscription(Bool, "/stuck_detection/active", self._stuck_cb, 10)
        if self._frontier_mode and self._abort_to_home:
            self.create_subscription(Bool, "/reactive_explorer/failsafe",
                                      self._reactive_failsafe_cb, 10)

        self.pub_cmd = self.create_publisher(Twist, "/exomy/cmd_vel", 10)
        self.pub_path = self.create_publisher(Path, "/l5_lite_plan", 10)
        self.pub_active = self.create_publisher(Bool, "/l5_lite_planner/active", 10)
        self.pub_frontier_goal = self.create_publisher(
            PointStamped, "/l5_lite_frontier_goal", 10)
        # Separate topic from frontier goals so recorders counting frontier
        # selections on /l5_lite_frontier_goal are not contaminated.
        self.pub_reobserve_goal = self.create_publisher(
            PointStamped, "/l5_lite_reobserve_goal", 10)

        self.create_timer(replan_period_s, self._replan)
        self.create_timer(1.0 / control_rate_hz, self._control_tick)

        self.get_logger().info(
            f"l5_lite_planner_node ready | {len(self._waypoints)} waypoint(s), "
            f"starting at ({self._goal_x:.2f}, {self._goal_y:.2f}) "
            f"replan_period={replan_period_s}s linear_speed={self._linear_speed}m/s"
        )

    # ── Subscriptions ──────────────────────────────────────────────────

    def _costmap_cb(self, msg: OccupancyGrid) -> None:
        if self._grid is None:
            self.get_logger().info(
                f"First /traversability_costmap received: {msg.info.width}x{msg.info.height} cells"
            )
        self._grid = list(msg.data)

    def _confidence_cb(self, msg: OccupancyGrid) -> None:
        if self._confidence_grid is None:
            self.get_logger().info("First /traversability_confidence received")
        self._confidence_grid = list(msg.data)

    def _pose_cb(self, msg: PoseWithCovarianceStamped) -> None:
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        if self._pose is None:
            self.get_logger().info(f"First /pose received: ({p.x:.2f}, {p.y:.2f})")
            if self._frontier_mode and (self._return_home or self._abort_to_home):
                self._home_xy = (p.x, p.y)
        self._pose = (p.x, p.y, _yaw_from_quaternion(q))

    def _explorer_cb(self, msg: Bool) -> None:
        self._explorer_active = msg.data

    def _stuck_cb(self, msg: Bool) -> None:
        self._stuck_detection_active = msg.data

    def _reactive_failsafe_cb(self, msg: Bool) -> None:
        self._reactive_failsafe = msg.data

    # ── Replanning ─────────────────────────────────────────────────────

    def _replan(self) -> None:
        if self._reached:
            return
        if self._grid is None or self._pose is None:
            self.get_logger().warn(
                f"Replan skipped -- waiting for data (grid={'ok' if self._grid else 'MISSING'}, "
                f"pose={'ok' if self._pose else 'MISSING'})"
            )
            return

        if self._frontier_mode and self._frontier_count == 0:
            # First frontier goal: needs at least one painted (known) cell
            # in the box before any frontier can exist -- until then, wait.
            if not self._select_next_frontier():
                return

        start_cell = world_to_cell(self._pose[0], self._pose[1])
        goal_cell = world_to_cell(self._goal_x, self._goal_y)
        # Frontier mode plans over a sanitized copy: unknown (-1) cells
        # become free (0), since A* must be able to route into unexplored
        # space and step_cost() would misbehave on a negative cost.
        plan_grid = grid_for_planning(self._grid) if self._frontier_mode else self._grid
        if self._frontier_mode:
            # 2026-07-19 start-cell-hazard deadlock fix: pure-pursuit
            # deviation can carry the rover onto a cell its own paint disc
            # mis-labelled hazard, which otherwise makes astar_path() refuse
            # every goal (start cell blocked). Frontier mode only -- hold-
            # position semantics for non-frontier missions are unchanged.
            freed_grid = grid_with_start_freed(plan_grid, WIDTH_CELLS, start_cell)
            if freed_grid is not plan_grid:
                self.get_logger().info(
                    f"Start cell {start_cell} was hazard-painted -- escaped "
                    "(self-paint hazard cells freed within escape radius)"
                )
            plan_grid = freed_grid
        cell_path = astar_path(plan_grid, WIDTH_CELLS, HEIGHT_CELLS, start_cell, goal_cell)

        if cell_path is None:
            if self._frontier_mode:
                start_cost = plan_grid[start_cell[1] * WIDTH_CELLS + start_cell[0]]
                goal_cost = plan_grid[goal_cell[1] * WIDTH_CELLS + goal_cell[0]]
                if self._reobserving and not self._returning_home:
                    # A2: an unreachable re-observation target is simply
                    # skipped -- it already joined the visited set at
                    # selection time, so selecting the next one cannot
                    # re-pick it. No blacklist involvement (that is
                    # frontier bookkeeping; mixing the two would corrupt
                    # both counts).
                    self.get_logger().warn(
                        f"Reobservation target {goal_cell} unreachable -- "
                        f"skipping, selecting next "
                        f"[start_cell={start_cell} cost={start_cost}, "
                        f"goal cost={goal_cost}]"
                    )
                    self._select_next_reobservation()
                    return
                # This frontier is unreachable on the current costmap --
                # blacklist it and immediately try the next one. Log WHY
                # (start blocked / goal blocked / no route) -- astar_path
                # returns a bare None for all three, which made the
                # 2026-07-19 semantic smoke test's every-goal-unreachable
                # churn undiagnosable from the log alone.
                self._frontier_blacklist.add(goal_cell)
                self.get_logger().warn(
                    f"Frontier {goal_cell} unreachable -- blacklisted "
                    f"({len(self._frontier_blacklist)} total), selecting next "
                    f"[start_cell={start_cell} cost={start_cost}, "
                    f"goal cost={goal_cost}]"
                )
                self._select_next_frontier()
                return
            self.get_logger().warn(
                f"No path found to goal on this replan (start_cell={start_cell}, "
                f"goal_cell={goal_cell}) -- holding position. "
                "Costmap may still update from live classification."
            )
            self._path_world = []
            return

        self.get_logger().info(
            f"Replanned: {len(cell_path)} cells, start_cell={start_cell}, goal_cell={goal_cell}"
        )
        self._path_world = path_to_world(cell_path)
        self._publish_path()

    def _select_next_frontier(self) -> bool:
        """Frontier mode: set the nearest frontier as the new goal. Returns
        True once a goal is set. When nothing is selectable: before the
        first selection this means the costmap has no painted cells yet
        (keep waiting -- the initial init_unknown grid is entirely -1, so
        frontiers only appear once DINOv2 paints a first known patch);
        after at least one selection it means exploration of the box is
        complete (or every remaining frontier is blacklisted), so the
        mission ends."""
        frontiers = find_frontier_cells(
            self._grid, WIDTH_CELLS, HEIGHT_CELLS, self._frontier_box)
        robot_cell = world_to_cell(self._pose[0], self._pose[1])
        semantic = self._semantic_frontier
        cell = select_nearest_frontier(frontiers, robot_cell, self._frontier_blacklist,
                                        min_distance_cells=self._min_frontier_cells,
                                        grid=self._grid if semantic else None,
                                        width=WIDTH_CELLS if semantic else None)
        known = count_known_cells_in_box(self._grid, WIDTH_CELLS, self._frontier_box)
        if cell is None:
            if self._frontier_count == 0:
                self.get_logger().info(
                    "Frontier mode: waiting for first painted cells in the box",
                    throttle_duration_sec=5.0,
                )
                return False
            if should_transition_to_reobservation(
                    frontier_selection_is_none=True,
                    reobserve_enabled=self._reobserve_mode,
                    already_reobserving=self._reobserving,
                    confidence_grid_available=self._confidence_grid is not None):
                # A2: exploration exhausted -- switch to revisiting the
                # lowest-confidence known cells instead of ending here.
                # Precedence over return_home is deliberate: if both are
                # enabled, home comes after re-observation finishes.
                self._reobserving = True
                self.get_logger().info(
                    f"Exploration complete: no frontiers remain after "
                    f"{self._frontier_count} autonomous selections "
                    f"({known} known cells in box, "
                    f"{len(self._frontier_blacklist)} blacklisted) -- "
                    f"entering re-observation mode with no operator command"
                )
                return self._select_next_reobservation()
            if self._reobserve_mode and not self._reobserving and self._confidence_grid is None:
                self.get_logger().error(
                    "reobserve_mode enabled but no /traversability_confidence data "
                    "received -- was the costmap launched with track_confidence:=true? "
                    "Ending mission normally."
                )
            if should_transition_to_return_home(
                    frontier_selection_is_none=True,
                    return_home_enabled=self._return_home,
                    already_returning_home=self._returning_home):
                self._returning_home = True
                self._goal_x, self._goal_y = self._home_xy
                self._path_world = []  # force a fresh plan toward home
                self.get_logger().info(
                    f"Exploration complete: no frontiers remain after "
                    f"{self._frontier_count} autonomous selections "
                    f"({known} known cells in box, "
                    f"{len(self._frontier_blacklist)} blacklisted) -- "
                    f"returning home to ({self._goal_x:.2f}, {self._goal_y:.2f}) "
                    f"with no operator command"
                )
                return True
            self._reached = True
            self.pub_cmd.publish(Twist())
            self.get_logger().info(
                f"Exploration complete: no frontiers remain after "
                f"{self._frontier_count} autonomous selections "
                f"({known} known cells in box, "
                f"{len(self._frontier_blacklist)} blacklisted)"
            )
            return False
        self._goal_x, self._goal_y = path_to_world([cell])[0]
        self._frontier_count += 1
        self._path_world = []  # force a fresh plan toward the new goal
        semantic_note = ""
        if semantic:
            # Per-selection invariant evidence for the pre-registered
            # criterion: bedrock_candidates counts the pool AFTER the same
            # blacklist/min-distance filters the selector applied, so
            # "bedrock_candidates>0 and selected_bedrock=False" in any log
            # line is a genuine invariant violation, not a filtering
            # artefact.
            rc, rr = robot_cell
            n_bedrock = sum(
                1 for c in frontiers
                if c not in self._frontier_blacklist
                and math.hypot(c[0] - rc, c[1] - rr) >= self._min_frontier_cells
                and is_bedrock_adjacent(self._grid, WIDTH_CELLS, c))
            sel_bedrock = is_bedrock_adjacent(self._grid, WIDTH_CELLS, cell)
            semantic_note = (f", bedrock_candidates={n_bedrock}, "
                             f"selected_bedrock={sel_bedrock}")
        self.get_logger().info(
            f"Frontier {self._frontier_count} selected (no operator command): "
            f"cell={cell} world=({self._goal_x:.2f}, {self._goal_y:.2f}), "
            f"known cells in box={known}{semantic_note}"
        )
        msg = PointStamped()
        msg.header.frame_id = "map"
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.point.x = self._goal_x
        msg.point.y = self._goal_y
        self.pub_frontier_goal.publish(msg)
        return True

    def _select_next_reobservation(self) -> bool:
        """A2 re-observation mode: set the lowest-confidence known cell as
        the new goal (visited-set so each cell is selected at most once per
        mission, reached or not). Driving at the cell re-observes it
        automatically -- the paint disc sits 0.6 m ahead of the rover, so
        the approach itself repaints the target before arrival; no separate
        stand-and-look behaviour exists or is needed. When nothing is
        selectable, re-observation is complete: hand over to return_home if
        enabled, else end the mission."""
        robot_cell = world_to_cell(self._pose[0], self._pose[1])
        cell = select_lowest_confidence_cell(
            self._grid, self._confidence_grid, WIDTH_CELLS, self._frontier_box,
            robot_cell, self._reobserve_visited,
            min_distance_cells=self._min_frontier_cells)
        if cell is None:
            if should_transition_to_return_home(
                    frontier_selection_is_none=True,
                    return_home_enabled=self._return_home,
                    already_returning_home=self._returning_home):
                self._returning_home = True
                self._goal_x, self._goal_y = self._home_xy
                self._path_world = []
                self.get_logger().info(
                    f"Re-observation complete after {self._reobserve_count} revisits -- "
                    f"returning home to ({self._goal_x:.2f}, {self._goal_y:.2f}) "
                    f"with no operator command"
                )
                return True
            self._reached = True
            self.pub_cmd.publish(Twist())
            self.get_logger().info(
                f"Re-observation complete: nothing selectable after "
                f"{self._reobserve_count} revisits "
                f"({len(self._reobserve_visited)} cells visited)"
            )
            return False
        # Invariant evidence (pre-registered P2): recompute the eligible
        # pool's minimum confidence independently of the selector, same
        # pattern as the semantic-frontier bedrock_candidates logging --
        # "selected_conf > pool_min" in any log line is a genuine
        # violation, not a filtering artefact. Computed BEFORE the selected
        # cell joins the visited set, so the selected cell is part of the
        # pool it is checked against.
        rc, rr = robot_cell
        col_min, row_min, col_max, row_max = self._frontier_box
        pool_confs = []
        for row in range(row_min, row_max + 1):
            for col in range(col_min, col_max + 1):
                idx = row * WIDTH_CELLS + col
                if not (0 <= self._grid[idx] < COST_HAZARD):
                    continue
                conf = self._confidence_grid[idx]
                if conf < 0:
                    continue
                if (col, row) in self._reobserve_visited:
                    continue
                if math.hypot(col - rc, row - rr) < self._min_frontier_cells:
                    continue
                pool_confs.append(conf)
        pool_min = min(pool_confs)
        selected_conf = self._confidence_grid[cell[1] * WIDTH_CELLS + cell[0]]
        self._reobserve_visited.add(cell)
        self._reobserve_count += 1
        self._goal_x, self._goal_y = path_to_world([cell])[0]
        self._path_world = []  # force a fresh plan toward the new goal
        self.get_logger().info(
            f"Reobservation {self._reobserve_count} selected (no operator command): "
            f"cell={cell} world=({self._goal_x:.2f}, {self._goal_y:.2f}), "
            f"selected_conf={selected_conf}, pool_min={pool_min}, "
            f"pool_size={len(pool_confs)}, visited={len(self._reobserve_visited)}"
        )
        msg = PointStamped()
        msg.header.frame_id = "map"
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.point.x = self._goal_x
        msg.point.y = self._goal_y
        self.pub_reobserve_goal.publish(msg)
        return True

    def _publish_path(self) -> None:
        msg = Path()
        msg.header.frame_id = "map"
        msg.header.stamp = self.get_clock().now().to_msg()
        for x, y in self._path_world:
            pose = PoseStamped()
            pose.header = msg.header
            pose.pose = Pose()
            pose.pose.position.x = x
            pose.pose.position.y = y
            msg.poses.append(pose)
        self.pub_path.publish(msg)

    # ── Control loop ───────────────────────────────────────────────────

    def _control_tick(self) -> None:
        # l5_lite owns cmd_vel from start until goal reached -- including
        # during the bootstrap crawl below, before /pose exists, so
        # terrain_controller_node cedes throughout, not just once planning
        # has begun (otherwise both would fight over cmd_vel).
        active = not self._reached
        self.pub_active.publish(Bool(data=active))

        if self._reached:
            self.pub_cmd.publish(Twist())
            return

        if self._frontier_mode and should_abort_to_home(
                reactive_failsafe=self._reactive_failsafe,
                abort_to_home_enabled=self._abort_to_home,
                already_aborting=self._aborting_to_home,
                frontier_mode_enabled=self._frontier_mode):
            # Mission-level failsafe reaction (item 12): reactive_explorer's
            # terminal FAILSAFE means it has given up on this hazard, so the
            # cede contract to it is no longer meaningful -- reusing
            # _returning_home gives this the same top-of-precedence
            # treatment as explore-then-return-home in every downstream
            # check (goal-reached branching, _replan()) with no new code
            # needed there. _aborting_to_home is kept only as a distinct
            # marker so the eventual "reached home" log line can say why.
            self._aborting_to_home = True
            self._returning_home = True
            self._goal_x, self._goal_y = self._home_xy
            self._path_world = []  # force a fresh plan toward home
            self.get_logger().info(
                f"Reactive failsafe -- aborting exploration, returning home "
                f"to ({self._goal_x:.2f}, {self._goal_y:.2f}) with no "
                f"operator command"
            )
            self.pub_cmd.publish(Twist())
            return

        # Once aborting, reactive_explorer has already given up and stays in
        # its terminal FAILSAFE state -- ceding to it forever would prevent
        # ever driving home, so the planner stops honouring its active flag.
        # stuck_detection is a separate, still-valid safety mechanism and is
        # honoured throughout, including during the home leg.
        ignore_explorer_cede = self._aborting_to_home
        if (self._explorer_active and not ignore_explorer_cede) or self._stuck_detection_active:
            return  # ceded — don't fight over cmd_vel

        if self._pose is None:
            linear_x, angular_z = bootstrap_crawl_command(
                pose_received=False, linear_speed=self._linear_speed
            )
            twist = Twist()
            twist.linear.x = linear_x
            twist.angular.z = angular_z
            self.pub_cmd.publish(twist)
            return

        robot_x, robot_y, robot_yaw = self._pose

        if self._frontier_mode and self._frontier_count == 0:
            # No frontier goal exists yet (all-unknown grid). Hold still;
            # the replan timer keeps trying to select the first frontier as
            # painted cells appear around the rover.
            self.pub_cmd.publish(Twist())
            return

        if goal_reached(robot_x, robot_y, self._goal_x, self._goal_y, self._goal_tolerance_m):
            if self._frontier_mode:
                if self._returning_home:
                    # Explore-then-return-home (grill-thesis 2026-07-19):
                    # the return leg's own goal (home) was just reached --
                    # this, not the earlier "no frontiers remain" moment, is
                    # what ends the mission.
                    self._reached = True
                    self.pub_cmd.publish(Twist())
                    if self._aborting_to_home:
                        # Item 12: distinct log line from the normal
                        # explore-then-return-home success below -- this
                        # mission was aborted after a reactive failsafe, not
                        # completed after exhausting exploration.
                        self.get_logger().info(
                            f"Returned home ({self._goal_x:.2f}, {self._goal_y:.2f}) -- "
                            f"mission aborted after reactive failsafe, returned home, "
                            f"no operator command"
                        )
                    else:
                        self.get_logger().info(
                            f"Returned home ({self._goal_x:.2f}, {self._goal_y:.2f}) -- "
                            f"explore-then-return-home mission complete, no operator command"
                        )
                    return
                if self._reobserving:
                    # A2: re-observation target reached (and repainted
                    # during the approach) -- pick the next lowest-
                    # confidence cell; _select_next_reobservation() ends
                    # the mission itself once nothing is selectable.
                    self.pub_cmd.publish(Twist())
                    self._select_next_reobservation()
                    return
                # Autonomous continuation: pick the next frontier with no
                # operator command; _select_next_frontier() ends the mission
                # itself once none remain (or starts the return leg /
                # re-observation phase above).
                self.pub_cmd.publish(Twist())
                self._select_next_frontier()
                return
            next_index = advance_waypoint_index(self._waypoint_index, len(self._waypoints))
            if next_index is None:
                self._reached = True
                self.get_logger().info(
                    f"Mission complete: waypoint {self._waypoint_index} "
                    f"({self._goal_x:.2f}, {self._goal_y:.2f}) was the last of "
                    f"{len(self._waypoints)}"
                )
                self.pub_cmd.publish(Twist())
                return
            self.get_logger().info(
                f"Waypoint {self._waypoint_index} reached "
                f"({self._goal_x:.2f}, {self._goal_y:.2f}) -- autonomously advancing "
                f"to waypoint {next_index}/{len(self._waypoints) - 1}"
            )
            self._waypoint_index = next_index
            self._goal_x, self._goal_y = self._waypoints[self._waypoint_index]
            self._path_world = []  # force a fresh plan toward the new goal
            return

        if not self._path_world:
            self.pub_cmd.publish(Twist())  # no valid plan yet — hold position
            return

        target = find_lookahead_point(self._path_world, robot_x, robot_y, self._lookahead_m)
        linear_x, angular_z = pure_pursuit_command(
            robot_x, robot_y, robot_yaw, target[0], target[1],
            self._linear_speed, self._angular_gain, self._max_angular_z,
        )
        if should_use_hybrid_rotation(linear_x, self._hybrid_creep_speed):
            # Heading error too large for the cos()-based forward component to
            # out-translate a creep burst -- continuous near-pure rotation
            # (negligible translation) stops slam_toolbox from producing
            # further /pose updates (systematic-debugging, 2026-07-17; dead
            # band below the old `== 0.0` trigger found live 2026-07-18, see
            # should_use_hybrid_rotation()). Alternate rotate-only and
            # creep-only bursts instead, so the scan-matcher periodically
            # gets translation to anchor a new estimate on.
            cycle_time_s = time.time() - self._start_time
            linear_x, angular_z = hybrid_rotate_command(
                cycle_time_s, self._hybrid_rotate_phase_s, self._hybrid_creep_phase_s,
                angular_z, self._hybrid_creep_speed,
            )
        self.get_logger().info(
            f"follow: pose=({robot_x:.2f},{robot_y:.2f},yaw={robot_yaw:.2f}) "
            f"target={target} -> cmd=(lin={linear_x:.3f},ang={angular_z:.3f})",
            throttle_duration_sec=2.0,
        )
        twist = Twist()
        twist.linear.x = linear_x
        twist.angular.z = angular_z
        self.pub_cmd.publish(twist)


def main(args=None):
    rclpy.init(args=args)
    node = L5LitePlannerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
