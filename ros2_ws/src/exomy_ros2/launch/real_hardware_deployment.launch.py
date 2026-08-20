"""
Purpose: The single entry point for running the full perception -> safety ->
         control -> hardware pipeline on real ExoMy, chaining every piece
         built for this thesis that was previously tested only in isolation
         or only in Gazebo:
           camera -> dinov2_terrain_node -> terrain_controller_node
                                          -> reactive_explorer_node (takes
                                             over on hazard, per the
                                             /reactive_explorer/active
                                             handshake)
           LiDAR  -> lidar_proximity_guard_node (independent proximity stop)
           IMU    -> icm20948_driver_node -> imu_slope_fusion_node
                     (feeds reactive_explorer_node's /imu_slope_stop trigger;
                     also publishes /traversability_fused for monitoring)
           IMU    -> gyro_odom_publisher_node -> /exomy/odom (always on since
                     2026-07-28; reactive_explorer_node's TURN_TO_HEADING
                     cannot complete without it)
           LiDAR/rover_command -> real_stuck_detection_node (independent
                     stuck-in-sand detection; the real rover has no wheel
                     encoders, and gyro-integrated odom drifts, so this LiDAR
                     front-sector-based node is used instead of the
                     odom-based stuck_detection_node.py -- see its own
                     module docstring)
           safety_watchdog_node (independent aggregate fail-rate E-stop)
           -> cmd_vel_to_rovercommand_bridge_node -> robot_node -> motor_node
              -> real PWM (or DRY-RUN log-only if Adafruit_PCA9685 is
              unavailable, e.g. when smoke-testing this file on a non-RPi
              dev machine)
         No such single launch file existed before this thesis -- every
         node above was previously only reachable through its own isolated
         launch file or a manual `ros2 run`, so nothing could be run
         end-to-end on real hardware with one command.
         The camera (camera_ros, libcamera-backed) and IMU drivers are real, small, verified
         ROS2 packages/nodes and are launched directly. The LiDAR driver
         (`sllidar_ros2`) is NOT installed on this development machine (no
         RPLIDAR C1 hardware yet) and its exact per-model launch file name
         has not been verified here -- rather than guess, `use_lidar` is
         off by default with instructions below for once the sensor and
         package are both available.
Inputs:  None (launches everything automatically).
         Optional args: use_camera, use_lidar, use_imu, dry_run,
         use_attention_viz, use_state_viz, use_int8_model, use_foxglove,
         use_slam
Outputs: Full pipeline running; see individual node docstrings for topics.
         A ros2 bag under bags/real_hardware_<timestamp>/
How to run:
    # On the RPi, with everything physically connected:
    source /opt/ros/humble/setup.bash
    source ~/Full-Thesis/ros2_ws/install/setup.bash
    ros2 launch exomy_ros2 real_hardware_deployment.launch.py

    # Dry-run smoke test on a dev machine (no camera/LiDAR/IMU/motors):
    ros2 launch exomy_ros2 real_hardware_deployment.launch.py \
        use_camera:=false use_lidar:=false use_imu:=false
    # motor_node still starts and falls back to DRY-RUN mode on its own
    # (Adafruit_PCA9685 import fails on a non-RPi machine).

    # Once RPLIDAR C1 + `sllidar_ros2` are both available on the RPi:
    #   1. sudo apt install ros-humble-sllidar-ros2  (package name may
    #      differ -- verify with `apt search sllidar` once on the RPi)
    #   2. Confirm the correct per-model launch file (e.g. a C1-specific
    #      launch file under that package's share/launch/ directory) and
    #      update the `sllidar_launch_file` argument below to match --
    #      do not assume the name without checking, since it varies by
    #      sllidar_ros2 release.
    #   3. ros2 launch exomy_ros2 real_hardware_deployment.launch.py use_lidar:=true

    # Live demo / debugging viz (off by default -- extra CPU load,
    # conservative default per this repo's "no unrequested load" convention):
    #   ros2 launch exomy_ros2 real_hardware_deployment.launch.py \
    #       use_attention_viz:=true use_state_viz:=true
    #   ros2 run rqt_image_view rqt_image_view /terrain_attention
    #   rviz2  (add MarkerArray topic /traversability_marker)

    # Foxglove Studio: connect to ws://<rpi-ip>:8765 (use_foxglove:=true, default).
    # Prefer the hybrid approach over WiFi -- subscribe to lightweight decision
    # topics (/terrain_classification, /stuck_detection/active, /exomy/cmd_vel,
    # /rover_command, ...) live, and pull the imagery from the recorded bag
    # afterwards rather than streaming full video live (bandwidth). Since
    # 2026-07-29 the bag holds /terrain_classified_image, one JPEG per DINOv2
    # verdict, rather than the raw camera stream.
    #   ros2 launch exomy_ros2 real_hardware_deployment.launch.py use_foxglove:=false

    # INT8-quantized DINOv2 encoder is the default for real-hardware deployment
    # (B1 result, 2026-06-30: 570ms->386ms latency, 220MB->163MB RAM, -0.5pp
    # accuracy). Requires dinov2_reg_small_encoder_int8.onnx on the RPi --
    # copy it from experiments/results/ (see experiments/int8_quantization_rpi.py).
    # Fall back to the full FP32 torch model if the ONNX file isn't present yet:
    #   ros2 launch exomy_ros2 real_hardware_deployment.launch.py use_int8_model:=false

    # L4 localisation (SLAM), off by default -- PREPARED, UNTESTED against
    # real hardware (see gyro_odom_publisher_node.py's docstring: this exact
    # architecture failed to produce a pose estimate in all 3 Gazebo
    # attempts, root cause not isolated). Requires use_imu:=true (default)
    # and use_lidar:=true (NOT default, RPLIDAR C1 not yet available):
    #   ros2 launch exomy_ros2 real_hardware_deployment.launch.py \
    #       use_lidar:=true use_slam:=true
Project: Onboard Visual Foundation Models for Mars Terrain Perception and Rover Navigation
"""
import datetime
import math
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription, OpaqueFunction,
    TimerAction,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def _sllidar_launch_path(context):
    """Resolve the sllidar_ros2 launch file path at launch time (only needed
    when use_lidar:=true), so an uninstalled package doesn't break parsing
    this file when the LiDAR is disabled (the default).

    serial_port is forwarded explicitly. sllidar's own default is
    /dev/ttyUSB0, which does not exist inside the container: docker/
    run_container.sh passes the LiDAR through as --device /dev/rplidar, the
    stable udev name from 99-rplidar.rules, and Docker creates the device node
    at that path and nowhere else. Until 2026-07-29 this include forwarded
    nothing, so sllidar opened /dev/ttyUSB0, died at startup with SDK error
    80008004, and no /scan was ever published -- reactive_explorer then sat in
    STARTUP_CHECK waiting for a scan that could not arrive, and the rover did
    nothing at all. docker/fix_ttyusb0_symlink.sh papers over it with a
    symlink that has to be re-made after every container restart; this fixes
    it at the source instead."""
    launch_file = LaunchConfiguration("sllidar_launch_file").perform(context)
    return [
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(get_package_share_directory("sllidar_ros2"), "launch", launch_file)
            ),
            launch_arguments={
                "serial_port": LaunchConfiguration("lidar_serial_port"),
            }.items(),
        )
    ]


def generate_launch_description():

    use_camera = DeclareLaunchArgument(
        "use_camera", default_value="true",
        description="Start camera_node (camera_ros, real RPi IMX219 via libcamera). "
                    "Disable for a dry-run smoke test."
    )
    use_lidar = DeclareLaunchArgument(
        "use_lidar", default_value="false",
        description="Start the RPLIDAR C1 driver (sllidar_ros2). Off by default -- "
                     "package not installed/verified on this dev machine, see docstring."
    )
    use_imu = DeclareLaunchArgument(
        "use_imu", default_value="true",
        description="Start icm20948_driver_node + imu_slope_fusion_node."
    )
    lidar_serial_port = DeclareLaunchArgument(
        "lidar_serial_port", default_value="/dev/rplidar",
        description="Serial device the sllidar driver opens. Defaults to the stable "
                    "udev name, which is the ONLY LiDAR path that exists inside the "
                    "container -- run_container.sh passes --device /dev/rplidar. "
                    "sllidar's own default is /dev/ttyUSB0 and opening it there fails "
                    "with SDK error 80008004."
    )
    sllidar_launch_file = DeclareLaunchArgument(
        "sllidar_launch_file", default_value="sllidar_c1_launch.py",
        description="Launch file name inside the sllidar_ros2 package share dir -- "
                     "VERIFY this against the actually-installed package before use_lidar:=true."
    )
    use_attention_viz = DeclareLaunchArgument(
        "use_attention_viz", default_value="false",
        description="Start dinov2_attention_node.py (self-attention heatmap overlay on "
                     "/terrain_attention). Off by default -- opt-in for live demos/debugging."
    )
    use_state_viz = DeclareLaunchArgument(
        "use_state_viz", default_value="false",
        description="Start traversability_viz_node.py (RViz2 colour-coded state marker on "
                     "/traversability_marker). Off by default -- opt-in for live demos/debugging."
    )
    use_int8_model = DeclareLaunchArgument(
        "use_int8_model", default_value="true",
        description="Load the INT8-quantized DINOv2 ONNX encoder instead of the full FP32 "
                     "torch model (B1 result: 1.45x speedup, -1.35x RAM, -0.5pp accuracy). "
                     "Default true for real-hardware deployment specifically -- Gazebo launches "
                     "are unaffected and stay on FP32."
    )
    use_foxglove = DeclareLaunchArgument(
        "use_foxglove", default_value="true",
        description="Start foxglove_bridge on port 8765 for live viewing in Foxglove Studio "
                     "(requires ros-humble-foxglove-bridge). RViz2 remains the offline/no-network "
                     "fallback for viewing the recorded ros2 bag."
    )
    use_slam = DeclareLaunchArgument(
        "use_slam", default_value="false",
        description="Start gyro_odom_publisher_node + slam_toolbox for L4 localisation. Off by "
                     "default -- PREPARED, UNTESTED staging work (see gyro_odom_publisher_node.py's "
                     "docstring). Requires use_imu:=true (default) and use_lidar:=true (not "
                     "default) to actually receive data; this launch file does not enforce that "
                     "combination automatically."
    )
    # ── Field-test visualisation (added 2026-08-11) ────────────────────────
    # All three default off, so the stack behaves exactly as it did when the
    # report's latency figures were measured unless a field run asks otherwise.
    # The Pi runs thirteen nodes on four cores; every one of these takes CPU
    # from inference, which is why they are opt-in and why /inference_latency_ms
    # is now recorded (run once each way and compare before trusting field
    # numbers against the report's).
    use_preview = DeclareLaunchArgument(
        "use_preview", default_value="false",
        description="Start camera_preview_node.py, a frame-dropped JPEG preview on "
                    "/camera/preview/compressed for a smooth live feed in RViz2 over "
                    "the Pi's WiFi hotspot. The raw 15 fps RGB888 stream is ~110 Mbps "
                    "and cannot be sent at all; this is ~0.57 Mbps at 5 fps."
    )
    preview_fps = DeclareLaunchArgument(
        "preview_fps", default_value="5.0",
        description="Target rate for the preview stream. Drop to 2.0 if the preview "
                    "is measurably stealing CPU from DINOv2."
    )
    use_sensor_overlay = DeclareLaunchArgument(
        "use_sensor_overlay", default_value="false",
        description="Start sensor_overlay_node.py, a single RViz2 text marker on "
                    "/sensor_overlay/markers carrying the terrain verdict, LiDAR front "
                    "range, IMU angles, commands and whatever is asserting a stop, each "
                    "shown as STALE rather than its last value once it goes quiet."
    )
    use_terrain_viz = DeclareLaunchArgument(
        "use_terrain_viz", default_value="false",
        description="Make dinov2_terrain_node publish the annotated /terrain_viz image. "
                    "This is the frame each verdict was actually made on, with the "
                    "verdict drawn onto it, and it is a raw sensor_msgs/Image so RViz2 "
                    "can read it without any transport plugin. Off by default because "
                    "drawing the overlay costs CPU on the rover; at the ~0.5 Hz the "
                    "full stack achieves it is about 3.7 Mbps on the wire."
    )
    use_sensor_tf = DeclareLaunchArgument(
        "use_sensor_tf", default_value="true",
        description="Publish the static base_link -> laser transform. Without it RViz2 "
                    "shows no LaserScan at all: nothing in this launch file published "
                    "a sensor TF before 2026-08-11, because RViz2 had only ever been "
                    "run against Gazebo, which supplies its own."
    )

    exposure_lock_delay_s = DeclareLaunchArgument(
        "exposure_lock_delay_s", default_value="8.0",
        description="Seconds to let auto-exposure converge before freezing it. 8s was "
                    "comfortably enough on the rover, where the image was already "
                    "settled a couple of seconds after the camera started. Raise it "
                    "for a scene that takes longer to settle, or set it very high to "
                    "leave auto-exposure running (see the exposure_lock comment)."
    )

    # ── 1. Camera driver (real RPi camera) ─────────────────────────────────
    # camera_ros, not v4l2_camera (changed 2026-07-28 after testing on the
    # rover). The ExoMy camera is an IMX219 (Pi Camera v2) on CSI, driven by
    # libcamera: /dev/video0 is unicam and offers only Bayer
    # (SRGGB10_CSI2P / SRGGB8), so v4l2_camera_node could not publish usable
    # RGB from it and in fact never opened the device at all. camera_ros
    # drives libcamera and was verified live end to end, feeding
    # dinov2_terrain_node at 15 fps.
    #
    # format matters as much as the package. Left to itself camera_ros
    # auto-selects NV21, which is planar YUV at 1.5 bytes per pixel, and
    # dinov2_terrain_node._image_callback reshapes to (h, w, -1) expecting an
    # interleaved 3-channel buffer -- that raises on the first frame. RGB888
    # gives rgb8, which that callback already handles.
    #
    # 640x480 rather than the sensor's full 3280x2464: DINOv2 resizes the
    # shortest edge to 256 and centre-crops 224 regardless, so larger frames
    # only cost the Pi CPU that the encoder needs.
    #
    # Exposure matters more than anything else here, and it is handled in two
    # stages: auto-exposure runs at startup, and exposure_lock below freezes it
    # once it has settled. Both halves are needed and neither works alone.
    #
    # Why it must be frozen: across 214 screened frames of 10 real surfaces,
    # live auto-exposure collapsed the probe to a single class, every surface
    # reading soil, because the camera flattens the image before DINOv2 sees
    # it. Running auto-exposure also defeats the blank-frame gate within about
    # a second -- a covered lens reads detail 1.6 and is correctly rejected,
    # then climbs past the 2.0 threshold as the camera ramps gain into its own
    # sensor noise, so the rejection releases while the lens is still covered.
    #
    # Why it cannot simply be disabled here: measured on the rover 2026-07-28,
    # setting AeEnable false before the camera streams freezes the driver's
    # DEFAULT exposure, not a good one, and that default is far too short. Every
    # frame came back at mean 0.2 out of 255 in the lab's normal lighting.
    # AnalogueGain cannot rescue it either, the driver caps it at 8.0 and even
    # there the mean only reached 3.4. AnalogueGain is therefore left to
    # auto-exposure as well, since pinning it after the exposure time is frozen
    # would only re-darken the image.
    #
    # Brightness and ExposureValue are safe to pin at startup because they bias
    # auto-exposure rather than replace it. Brightness must be 0.0: a -1.0 left
    # over from an exposure bracket silently produced 217 fully black frames
    # that the probe still labelled soil at 0.913. ExposureTime is deliberately
    # not set, this driver reports a nonsensical min 39 / max 0 range for it.
    camera_node = Node(
        package="camera_ros",
        executable="camera_node",
        name="camera_node",
        output="screen",
        condition=IfCondition(LaunchConfiguration("use_camera")),
        parameters=[{
            "format": "RGB888",
            "width": 640,
            "height": 480,
            "Brightness": 0.0,
            "ExposureValue": 0.0,
        }],
        # The tilde is load-bearing. camera_ros publishes on PRIVATE topics,
        # ~/image_raw and ~/camera_info, so a remap keyed on the bare name
        # matches nothing and is silently ignored. The node then published to
        # /camera_node/image_raw, dinov2_terrain_node waited on
        # /camera/image_raw and logged "No image" forever, and the full launch
        # had never once fed the classifier. It hid for so long because every
        # test until 2026-07-28 started camera_node by hand, where it keeps its
        # default name "camera" and ~/image_raw already resolves to
        # /camera/image_raw without any remap at all.
        remappings=[
            ("~/image_raw", "/camera/image_raw"),
            ("~/camera_info", "/camera/camera_info"),
        ],
    )

    # Freeze the exposure once auto-exposure has settled. This is the second
    # half of the camera configuration above and the two only work together.
    # Verified live on the rover in exactly this order: detail 5.3 with
    # auto-exposure disabled from the start, 55.2 once it was allowed to run,
    # and still 55.2 at mean 85 after switching it off again, with DINOv2
    # classifying soil and sand at real confidences rather than reporting one
    # class everywhere.
    #
    # A one-shot `ros2 param set` rather than a node, because that is exactly
    # what was verified by hand and it adds nothing to maintain. It is fired by
    # a timer rather than chained to the camera starting, since camera_ros has
    # no readiness signal to chain to; the delay is generous rather than tight
    # for that reason. If it fires before the camera is up the set simply fails
    # and auto-exposure keeps running, which is the safe direction to fail in:
    # a usable image with a weaker gate, not a black one.
    # Retried rather than fired once. `ros2 param set` has to discover
    # camera_node's parameter service over DDS first, and that took about 4 s
    # on 2026-07-29 -- close enough to the CLI's own patience that the same
    # command succeeded on one run and died with "Node not found" on the next
    # two, leaving the camera on auto-exposure for the whole run without
    # anything failing. Auto-exposure is not cosmetic here: the sweep ranks
    # four directions by DINOv2 score, so if the camera re-exposes between
    # stops then part of the difference between those scores is the camera
    # rather than the ground, and the ranking cannot be attributed to terrain.
    # Measured 2026-07-28 on one fixed scene: auto-exposure flip-flopped
    # soil/sand at 0.41-0.69, frozen exposure held soil 0.65-0.82.
    #
    # It cannot simply be disabled at startup instead -- auto-exposure has to
    # converge first and then be frozen, or every frame comes out black.
    exposure_lock = TimerAction(
        period=LaunchConfiguration("exposure_lock_delay_s"),
        condition=IfCondition(LaunchConfiguration("use_camera")),
        actions=[
            ExecuteProcess(
                cmd=[
                    "bash", "-lc",
                    "for i in $(seq 1 30); do "
                    "  if ros2 param set /camera_node AeEnable false; then "
                    "    echo 'exposure locked'; exit 0; "
                    "  fi; sleep 1; "
                    "done; "
                    "echo 'EXPOSURE LOCK FAILED -- camera stayed on "
                    "auto-exposure, terrain scores are not comparable "
                    "between headings' >&2; exit 1",
                ],
                output="screen",
            ),
        ],
    )

    # ── 2. LiDAR driver (RPLIDAR C1, off by default -- see docstring) ──────
    # Resolved lazily via OpaqueFunction so this file still parses cleanly
    # with use_lidar left at its default (false) even though sllidar_ros2
    # is not installed on this development machine.
    lidar_driver = OpaqueFunction(
        function=_sllidar_launch_path,
        condition=IfCondition(LaunchConfiguration("use_lidar")),
    )

    # ── 3. Perception: DINOv2 terrain classification ────────────────────────
    # use_int8_model defaults true here (real-hardware deployment only, B1
    # result) -- skips loading the 86MB FP32 torch encoder entirely in favour
    # of the 25MB INT8 ONNX encoder. Falls back to FP32 torch if the ONNX
    # file isn't on the RPi yet (use_int8_model:=false).
    dinov2_node = Node(
        package="fm_perception",
        executable="dinov2_terrain_node.py",
        name="dinov2_terrain_node",
        output="screen",
        parameters=[{
            "device": "cpu",
            "confidence_threshold": 0.40,
            "publish_viz": LaunchConfiguration("use_terrain_viz"),
            "use_int8_model": LaunchConfiguration("use_int8_model"),
            # Real hardware only. AI4Mars is Curiosity NAVCAM, and its EDR
            # images really are single-channel grayscale (verified
            # 2026-07-27: PIL mode="L", R==G==B), while the RPi camera
            # delivers RGB -- the luminance mismatch Exp 5b identified as the
            # root cause of its 20.0% real-camera accuracy. Converting the
            # camera frame to grayscale before classification was measured on
            # the same 20 real ExoMy images (2026-07-16,
            # experiments/results/sim_to_real_gap_results_grayscale.csv):
            # 20.0% -> 25.0% overall, sand 40% -> 50%, big_rock unchanged at
            # 0%. That is +5.0 pp from one extra correct image out of twenty,
            # so it is a small directional confirmation of the luminance
            # hypothesis, NOT a fix for the domain gap -- do not quote it as
            # one. Gazebo launches are untouched and stay on RGB.
            "grayscale_preprocessing": True,
        }],
    )

    # ── 3b. Interpretability/debug viz -- off by default, opt-in ────────────
    attention_viz_node = Node(
        package="fm_perception",
        executable="dinov2_attention_node.py",
        name="dinov2_attention_node",
        output="screen",
        condition=IfCondition(LaunchConfiguration("use_attention_viz")),
        parameters=[{"device": "cpu", "skip_frames": 3, "alpha": 0.5}],
    )
    state_viz_node = Node(
        package="fm_perception",
        executable="traversability_viz_node.py",
        name="traversability_viz_node",
        output="screen",
        condition=IfCondition(LaunchConfiguration("use_state_viz")),
    )

    # ── 3c. Field-test visualisation (2026-08-11) ──────────────────────────
    # Both read-only on topics that already exist. Neither can change what any
    # other node decides, so a run with them on produces the same behaviour as
    # a run without -- only slower, which is the whole reason they are opt-in.
    preview_node = Node(
        package="fm_perception",
        executable="camera_preview_node.py",
        name="camera_preview_node",
        output="screen",
        parameters=[{
            "target_fps": LaunchConfiguration("preview_fps"),
            "jpeg_quality": 80,
        }],
        condition=IfCondition(LaunchConfiguration("use_preview")),
    )

    sensor_overlay_node = Node(
        package="fm_perception",
        executable="sensor_overlay_node.py",
        name="sensor_overlay_node",
        output="screen",
        parameters=[{
            "publish_rate_hz": 2.0,
            # base_link, not gyro_odom: gyro_odom_publisher_node's transform has
            # translation pinned to zero (the rover has no wheel encoders), so
            # the odom frame carries heading and nothing else. Anchoring the
            # overlay to base_link keeps it on the rover whether or not the IMU
            # is running.
            "frame_id": "base_link",
        }],
        condition=IfCondition(LaunchConfiguration("use_sensor_overlay")),
    )

    # Static base_link -> laser. Nothing in this launch file published a sensor
    # TF before 2026-08-11, because RViz2 had only ever been pointed at Gazebo,
    # which supplies its own -- so against real hardware the LaserScan display
    # showed nothing at all and failed with "No transform".
    #
    # The numbers assert nothing new about the rover. The C1 is centre-mounted,
    # and lidar_proximity_guard_node and reactive_explorer_node already model it
    # as coincident with the rover centre carrying a measured 3.5 degree yaw
    # offset (lidar_yaw_offset_deg, below). This transform encodes exactly that
    # existing model: zero translation, 3.5 degrees of yaw. It is a display aid,
    # not a calibration, and the mounting height was never measured.
    lidar_static_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="base_link_to_laser_tf",
        arguments=[
            "--x", "0.0", "--y", "0.0", "--z", "0.0",
            "--yaw", str(math.radians(3.5)), "--pitch", "0.0", "--roll", "0.0",
            "--frame-id", "base_link", "--child-frame-id", "laser",
        ],
        condition=IfCondition(LaunchConfiguration("use_sensor_tf")),
    )

    # ── 4. Layer-3 reactive speed controller ────────────────────────────────
    controller_node = Node(
        package="fm_perception",
        executable="terrain_controller_node.py",
        name="terrain_controller_node",
        output="screen",
        parameters=[{
            "cmd_vel_topic": "/exomy/cmd_vel",
            "terrain_topic": "/terrain_classification",
            "confidence_threshold": 0.40,
            # Real hardware only: all three set to 0.025 m/s (RoverCommand
            # vel=25) instead of the graded 0.10/0.05/0.03 Gazebo uses.
            # 2026-07-27 finding: PWM dead-zone measurements
            # (project_real_wheel_speed_pwm_deadzone_20260726) show vel=20,
            # 25, 50 all produce the SAME real speed (~0.235-0.245 m/s), and
            # vel=100 (what speed_soil=0.10 maps to) stops BOTH wheel sides
            # entirely -- so the terrain-graded speed tiers are not just
            # unimplementable on this hardware, the "fastest/safest" tier
            # was actually non-functional. Gazebo's launch files are
            # untouched and keep the graded speeds; this is a real-hardware-
            # only limitation, documented as such in the thesis rather than
            # engineered around.
            "speed_soil": 0.025,
            "speed_sand": 0.025,
            "speed_bedrock": 0.025,
            "publish_rate_hz": 10.0,
            # 7.0, raised from 3.0 on 2026-07-29. DINOv2 measured 2200 ms per
            # frame with the full stack running on the Pi (the 454 ms figure
            # was camera + DINOv2 alone), so a 3 s watchdog expired between
            # almost every pair of inferences: Trial A's log alternates
            # "Terrain: soil SAFE" with "No terrain classification for 3.x s
            # -- forcing STOP" for the entire run. Sized at ~3 inference
            # periods so a genuinely dead classifier is still caught quickly.
            "stale_timeout_s": 7.0,
        }],
    )

    # ── 5. LiDAR proximity guard (independent depth-based stop) ────────────
    lidar_guard_node = Node(
        package="fm_perception",
        executable="lidar_proximity_guard_node.py",
        name="lidar_proximity_guard_node",
        output="screen",
        parameters=[{
            "scan_topic": "/scan",
            "cmd_vel_topic": "/exomy/cmd_vel",
            # 0.30 m, lowered from 0.40 on 2026-07-29 so it sits BELOW the
            # planner's 0.40 m lookahead. When the two were equal the guard
            # always fired first: the rover hard-stopped the instant anything
            # entered the corridor and the planner never got the chance to
            # turn, so avoidance was dead code. Derived, not chosen: the front
            # wheels sit 0.20 m ahead of the LiDAR and the chain from a /scan
            # arriving to the wheels stopping is ~0.3 s, which at the bridge's
            # 0.10 m/s ceiling is 0.03 m, leaving ~0.07 m of margin.
            "stop_distance_m": 0.30,
            "resume_distance_m": 0.40,
            "stale_timeout_s": 3.0,
            # MUST match reactive_explorer_node's min_ignore_m and
            # blocked_sectors_deg -- see the comment there.
            "min_ignore_m": 0.2,
            # MUST match reactive_explorer_node's rover_width_m/2 +
            # lateral_margin_m (0.32/2 + 0.04). Since 2026-07-29 the guard
            # measures down the same swept rectangle the planner uses instead
            # of taking a direction-blind nearest return, which is what used
            # to make a wall 0.25 m to the SIDE latch a permanent stop.
            "half_width_m": 0.20,
            # MUST match reactive_explorer_node's lidar_yaw_offset_deg.
            "lidar_yaw_offset_deg": 3.5,
            # blocked_sectors_deg is deliberately NOT set here yet. The
            # centre-mounted C1 sees ExoMy's own mast and servo arms at
            # ~0.20-0.23m, inside stop_distance, so the guard latches STOP
            # forever without an angular mask (verified live 2026-07-23/24).
            # Measure the real sectors on the as-built top plate v2 with
            # experiments/lidar_self_view_characterisation.py and paste the
            # printed list here. Do not guess them: every masked bearing is
            # a direction this guard cannot see.
        }],
    )

    # ── 6. Reactive hazard-recovery FSM ─────────────────────────────────────
    reactive_explorer_node = Node(
        package="fm_perception",
        executable="reactive_explorer_node.py",
        name="reactive_explorer_node",
        output="screen",
        parameters=[{
            "stuck_dwell_s": 3.0,
            # 5.0, widened from 3.0 on 2026-07-29 so it is larger than
            # angle_step_deg. With a 5 deg step and a 3 deg tolerance the
            # nearest heading the picker could ever propose was already
            # outside tolerance, so the FSM turned, "arrived" on gyro drift
            # alone, re-scanned, and proposed the same 5 deg again --
            # observed live as nine TURN/STARTUP_CHECK cycles in 20 s.
            "angle_tolerance_deg": 5.0,
            # 0.3 (the FSM's own default) maps through
            # cmd_vel_to_rovercommand_bridge_node's max_angular_velocity=0.3
            # scaling to RoverCommand.vel=100 -- the exact value live
            # hardware trial 2026-07-26 found stops BOTH wheel sides
            # entirely (matches project_real_wheel_speed_pwm_deadzone_20260726's
            # "vel=100 stops both sides" finding). 0.15 maps to vel=50, one
            # of that memory's already-confirmed-working values.
            # max_turn_duration_s below is doubled to match (a 180deg turn
            # now takes ~20.9s instead of ~10.5s).
            "angular_speed": 0.15,
            "max_turn_duration_s": 40.0,
            # retreat_speed/retreat_distance_m/max_retreat_cycles are gone:
            # SLOPE_RETREAT was removed on 2026-07-29 and the rover has no
            # reverse motion at all any more.
            "publish_rate_hz": 10.0,
            "cmd_vel_topic": "/exomy/cmd_vel",
            "odom_topic": "/exomy/odom",
            "scan_topic": "/scan",
            # Real-hardware measured chassis width (2026-07-26), margin
            # already included. Do not change without re-measuring.
            "rover_width_m": 0.32,
            # Tried in order (2026-07-27 redesign): 1.0m first, relaxed to
            # 0.3m only if nothing in the full 360deg scan clears 1.0m
            # anywhere. 0.3m (not the original design's 0.5m) reflects the
            # LiDAR's mast/neck self-view now sitting ahead of it rather
            # than behind -- RE-MEASURE with
            # experiments/lidar_self_view_characterisation.py after the new
            # top-plate mounting is installed, before trusting this value on
            # real hardware. See project docs for the 2026-07-27 finding
            # that the mast previously hid a real 65cm-away wall completely
            # from the forward corridor check.
            # ONE tier, 0.40 m, replacing [1.0, 0.3] on 2026-07-29. Two tiers
            # meant the rover applied one clearance rule until nothing in the
            # whole circle satisfied it and then silently switched to another,
            # which is impossible to reason about from a log. Measured from
            # the LiDAR, so it already covers the 0.20 m to the front wheels
            # and leaves 0.20 m of real warning; it must stay above the
            # guard's 0.30 m stop distance so the planner gets to turn first.
            "lookahead_tiers_m": [0.40],
            "lookahead_buffer_m": 0.3,
            # 2.0, finer than angle_tolerance_deg -- see the note there.
            "angle_step_deg": 2.0,
            # Clearance demanded beyond the chassis half-width, absorbing
            # LiDAR yaw uncertainty, point-turn overshoot and wheel slip.
            # 0.32/2 + 0.04 = 0.20, which is the guard's half_width_m.
            "lateral_margin_m": 0.04,
            # 2026-08-04, hardware round 4/5 of the H5 follow-up confirmation
            # testing: biases the LiDAR heading picker away from the
            # smallest-possible correction, which was letting the rover
            # drift into a wall at a shallow angle a few degrees at a time.
            # Falls back to an unrestricted search if nothing clears this
            # minimum anywhere, so it can only make the rover turn further,
            # never refuse a correction that was available before. 30, not
            # wider: must stay above angle_tolerance_deg (5.0 below) with
            # margin -- see test_min_correction_turn_is_wider_than_the_arrival_tolerance.
            "min_correction_turn_deg": 30.0,
            # Opening sweep, 4 stops of 90 deg. Set by the camera, not the
            # LiDAR, which already returns the full circle per frame: DINOv2
            # can only judge terrain the camera is pointed at, so the rover
            # turns to give it several directions before choosing.
            #
            # Reduced from 8 to 4 on 2026-07-29 after the first sweep that ran
            # end to end took about 50 s of a run in which the rover never
            # drove. The imx219 sees ~62 deg, so 4 stops of 90 leave ~28 deg
            # unsampled between each pair -- deliberately accepted. The LiDAR
            # still vetoes geometry over the full 360 deg, so the gaps cost
            # terrain ranking on 4 candidate directions rather than any safety
            # coverage, and the thesis question is whether the foundation model
            # can choose a heading at all, not how finely it can resolve one.
            #
            # Tied to use_camera, not a fixed number. The sweep exists solely
            # to give DINOv2 a view of each direction, so without a camera
            # there is nothing to sample: the SAMPLE phase would wait for a
            # score that never arrives and the whole stack would fail safe and
            # shut down after max_turn_duration_s. Launching with
            # use_camera:=false (the elevated bench-test recipe) has to keep
            # working, and the honest behaviour there is no sweep rather than a
            # broken one.
            "sweep_headings": ParameterValue(
                PythonExpression(
                    ["4 if '", LaunchConfiguration("use_camera"), "' == 'true' else 0"]
                ),
                value_type=int,
            ),
            # 1, down from 2 on 2026-07-29. The first result after each stop is
            # discarded regardless (it was captured mid-turn), so this still
            # costs two inferences per stop, and dropping the second averaged
            # sample buys back ~2 s per heading against noisier scores. The
            # noise is visible in the data -- consecutive readings of one
            # stationary view span about 0.07 -- and terrain_reject_margin is
            # 0.15, twice that, so a single sample stays inside the margin.
            "sweep_samples_per_heading": 1,
            # How much worse than the CHOSEN heading's own sweep score a later
            # reading may be before that heading is refused. Relative, because
            # the same camera and model called a wall soil:0.937 across a
            # 214-frame capture, so an absolute threshold passes or fails
            # everything. Compared against the chosen heading rather than the
            # sweep median since 2026-07-29: the median refuses about half of
            # any compass by construction, which is what stalled Trial A.
            "terrain_reject_margin": 0.15,
            # Floor under baseline+margin, added 2026-08-04. The baseline
            # follows the ground the rover is standing on and has a ceiling but
            # nothing holding it up, so a stretch of soil (CLASS_RISK 0.0)
            # drags it to almost zero and the threshold becomes an absolute
            # ~0.15 -- below every terrain class except soil, sand included at
            # 0.5. Six sandpit runs that day ran with baselines of
            # 0.003/0.034/0.134 and one of them ended by refusing a 0.525 sand
            # reading, the surface it had been driving on all afternoon.
            # 0.6 sits between sand (0.5) and bedrock (0.7): soil and sand
            # always pass, bedrock and big_rock/uncertain (1.0) still refuse,
            # so the hazard ordering H5 demonstrated is unchanged.
            "terrain_reject_floor": 0.6,
            # A refused heading used to cost nothing and be logged nowhere, so
            # the rover could reject and re-pick in silence indefinitely -- it
            # did so for 573 s on 2026-07-29 while the bag grew to 11 GB. The
            # confirm timeout does not cover this: a score that arrives and is
            # refused resets that clock.
            #
            # 3 -> 20 on 2026-08-04, with the real bound moved to
            # terrain_search_timeout_s below. Three bounded the wrong thing.
            # Exhausting the picker takes 15 refusals, measured in
            # test_avoidance_planner: it offers the smallest-deviation heading
            # not already excluded, so each refusal advances the frontier by
            # exclude_tolerance + angle_step = 25 deg, not by the 40 deg an
            # exclusion covers. Across six sandpit runs that day the rover
            # failed safe after one or two refusals with whole arcs of the
            # compass never looked at. A search should now normally end by
            # exhausting the picker (boxed_in) -- the honest "nowhere left to
            # go" -- and this count is only a backstop. Keep it above 15.
            "max_terrain_rejections": 20,
            # The deadline that replaces the count. 15 refusals, each costing a
            # turn plus a settle plus 2-3 DINOv2 frames at the deployed 2.2 s,
            # is about 2 minutes, so 180 s lets a genuine exhaustive search
            # finish while still bounding one that never converges. A run that
            # ends this way rotated in place for 3 minutes: that is the
            # designed behaviour when genuinely surrounded, not a hang.
            "terrain_search_timeout_s": 180.0,
            # Stop commanding rotation this many seconds' worth of travel
            # before the target, at the yaw rate odom is actually reporting.
            # Originally measured 2026-07-29: every commanded 45 deg sweep
            # stop overran by about 20 deg at 40-90 deg/s (RoverCommand.vel
            # is not rad/s and the servos saturate, so the only lever is
            # stopping sooner) -- gave 0.35 s.
            # RECALIBRATED 2026-08-19: field report of consistent undershoot
            # on every turn (commanded 90 deg landing short, return-to-origin
            # turns landing short too) since the wheel/head reprint (17 Aug)
            # and IMU board swap (19 Aug) -- the 0.35s lead, tuned against the
            # OLD chassis's momentum, was over-compensating against the new,
            # lower-friction one. Measured native overshoot with the
            # anticipation temporarily zeroed, two sand-pit sweeps (opening
            # sweep's 90/180/-90 deg stops, turn_stop_lead_s=0.0):
            #   run 1: +7, +6, +3 deg   run 2: +13, +9, +7 deg
            # Average 7.5 deg at the same 40-90 (~60) deg/s used for the
            # original calibration -> 7.5/60 = 0.125 s, rounded up slightly
            # for margin against undershoot recurring.
            # RE-TESTED at 0.13 the same day: sand-pit sweep telemetry STILL
            # showed the same +6/+8/+9 deg overshoot as at 0.0, essentially
            # unchanged -- three different values of this constant (0.35,
            # 0.0, 0.13) produced no material change in the rover's own
            # gyro-measured error, and that measured error is the opposite
            # sign from what the user visually observes (gyro says overshoot,
            # user sees undershoot in sand only, not on flat ground). This
            # constant is very likely NOT the real fix: the leading
            # hypothesis is IMU vibration coupling into gyro yaw in sand
            # because it is still not mounted rigidly to the chassis. Do not
            # retune this value again without first mounting the IMU rigidly
            # and re-measuring against the rover's *actual* physical heading
            # (protractor/marked reference), not the gyro's own report of
            # itself. See project_turn_undershoot_sandpit_investigation_20260819
            # and project_imu_mpu6050_saga_resolved_20260819 in memory.
            "turn_stop_lead_s": 0.13,
            # A heading must have at least this much room to travel before the
            # sweep will consider it. The LiDAR's other question -- does the
            # rover fit through the first lookahead_tiers_m -- says nothing
            # about whether there is anywhere to go, so in an open room every
            # heading passes and clearance was left as a mere tie-break. Across
            # several runs in the lab and the sandpit the rover kept choosing
            # directions with almost no room; the sand run of 2026-07-30 took
            # +0 at 0.60 m over +180 at 2.18 m and drove into the pit wall.
            #
            # 1.0 m is taken from the measured tables: it excludes the 0.60 m
            # and 0.92 m choices actually observed while leaving three or four
            # headings on offer in every run recorded so far. Expressed as a
            # stricter veto, not as a weight on the DINOv2 score, so geometry
            # still only eliminates and the model still chooses. Relaxes on its
            # own if no heading has that much room, rather than failing safe on
            # a rover that could still move.
            "min_run_m": 1.0,
            # Exchange rate between how open a heading is and how good its
            # terrain looks: cost = score + openness_weight * (1 - min(
            # clearance, clearance_horizon_m) / clearance_horizon_m).
            #
            # Derived from the five sweep tables recorded on hardware, not
            # guessed. At 0 the terrain score decided all five sweeps and chose
            # a heading the user judged wrong in three of them, including the
            # 0.60 m direction in the sandpit that ended in the pit wall. Every
            # weight from 0.4 to 1.0 fixes all three while the score still
            # changes the outcome in two of the five; from 1.5 upward the
            # model's influence collapses to one and then none. 0.5 is the
            # middle of that plateau rather than its edge.
            #
            # This is tuned to a cluttered indoor lab, which is the opposite of
            # the target domain, so it is a parameter and not a constant. The
            # sweep now logs what geometry alone would have chosen and whether
            # DINOv2 changed it, so the model's contribution is measured on
            # every run instead of asserted.
            #
            # n=5, and the reference heading is a human judgement by eye rather
            # than a measured outcome. Enough to set a parameter, not enough to
            # call it optimal.
            "openness_weight": 0.5,
            "clearance_horizon_m": 3.0,
            # Radius the corners sweep in a point turn: hypot(0.20 front
            # offset, 0.16 half width) = 0.256 m, plus margin. The forward
            # corridor is a rectangle and rightly lets the rover drive PAST a
            # wall 0.25 m to its side, but rotating beside that wall grinds a
            # corner into it, and this design turns eight times before it even
            # starts driving.
            "turn_radius_m": 0.30,
            # Bound on the wait for a terrain verdict after a turn. Without it
            # a dead DINOv2 leaves the rover holding cmd_vel at zero forever,
            # never failing safe, with the rosbag still recording.
            "terrain_confirm_timeout_s": 15.0,
            "settle_time_s": 1.0,
            "imu_stale_timeout_s": 3.0,
            # MUST match lidar_proximity_guard_node's min_ignore_m and
            # blocked_sectors_deg above exactly -- these two nodes each declare
            # their own copy and do not share parameters automatically. If you
            # change one, change the other.
            "min_ignore_m": 0.2,
            # Re-measured live 2026-07-28 via experiments/measure_lidar_yaw_offset_direct.py
            # after the LiDAR was remounted to the front of the mast on the new
            # 3D-printed top plate (was 22.0 deg on the old mount -- see
            # project_lidar_avoidance_hardware_trial_20260726 for that
            # measurement). Wall at 0.80m, single 118-beam contiguous run,
            # bearing 3.5 deg -- confirmed with a repeat measurement at 0.55m
            # (which split into two runs due to a narrow obstruction, combined
            # bearing ~1.4 deg, consistent within noise). Bearing 0 in this
            # rover's /scan messages is NOT its physical front; without this
            # correction the proactive corridor trigger silently checks the
            # wrong direction. Re-measure and update this value if the LiDAR is
            # ever unmounted/remounted again.
            "lidar_yaw_offset_deg": 3.5,
        }],
    )

    # ── 6b. Stuck-in-sand detection + recovery -- highest /rover_command
    #         priority, downstream of the cmd_vel->RoverCommand bridge (no
    #         odom/gyro on real hardware by default; see docstring) ─────────
    stuck_detection_node = Node(
        package="fm_perception",
        executable="real_stuck_detection_node.py",
        name="real_stuck_detection_node",
        output="screen",
        parameters=[{
            "stuck_window_s": 4.0,
            "stuck_progress_threshold_m": 0.05,
            "min_commanded_speed": 1.0,
            "boost_max_speed": 25.0,
            "boost_duration_s": 4.0,
            "wiggle_angle_deg": 20.0,
            "angular_speed": 0.3,
            "max_wiggle_attempts": 4,
            "wiggle_vel": 20.0,
            "retreat_speed": 20.0,
            "retreat_duration_s": 2.0,
            # 2026-08-02: gate the retreat on the rear-sector LiDAR reading
            # (0.30 m matches lidar_proximity_guard_node's stop_distance_m)
            # instead of reversing with zero sensing at all -- see
            # RealStuckDetectionFSM.retreat_min_clearance_m.
            "retreat_min_clearance_m": 0.30,
            # Matches lidar_proximity_guard_node's own min_ignore_m -- see
            # front_sector_min_range's docstring for why this is a range
            # floor, not a bearing guess.
            "min_ignore_m": 0.2,
            "sector_half_width_deg": 20.0,
            "publish_rate_hz": 10.0,
            "scan_topic": "/scan",
            "rover_command_topic": "/rover_command",
        }],
    )

    # ── 7. Aggregate fail-rate safety net ───────────────────────────────────
    watchdog_node = Node(
        package="fm_perception",
        executable="safety_watchdog_node.py",
        name="safety_watchdog_node",
        output="screen",
        parameters=[{
            "fail_rate_threshold": 0.5,
            "window_s": 10.0,
            "stale_timeout_s": 3.0,
        }],
    )

    # ── 8. IMU driver + slope fusion (feeds reactive_explorer_node's
    #        /imu_slope_stop trigger, §4.8, and terrain_controller_node's
    #        continuous-score path if use_continuous_score is enabled) ──────
    imu_driver_node = Node(
        package="fm_imu_fusion",
        executable="icm20948_driver_node.py",
        name="icm20948_driver_node",
        output="screen",
        condition=IfCondition(LaunchConfiguration("use_imu")),
        parameters=[{
            # Replacement Servo HAT (2026-07-23): hardware I2C1 works, so the
            # old software i2c-gpio bus 3 workaround is gone.
            # IMU swapped 2026-08-19: the replacement board was bought as a
            # like-for-like ICM-20948 swap but turned out, on inspection of
            # the physical part (silkscreen "HW-123" / "ITG/MPU"), to be an
            # MPU-6050 breakout instead -- see icm20948_driver_node.py's
            # module docstring for the full identification and the driver
            # rewrite that followed. AD0 tied low puts it at 0x68, not the
            # old board's 0x69. Verified with i2cdetect AND a live end-to-end
            # run on the rover (reactive_explorer_node reached POINT_TURN).
            "i2c_bus": 1,
            "i2c_address": 0x68,
            "publish_hz": 50.0,
        }],
    )
    imu_fusion_node = Node(
        package="fm_imu_fusion",
        executable="imu_slope_fusion_node.py",
        name="imu_slope_fusion_node",
        output="screen",
        condition=IfCondition(LaunchConfiguration("use_imu")),
        parameters=[{
            "slope_caution_deg": 10.0,
            "slope_stop_deg": 20.0,
            "conf_threshold": 0.40,
            "publish_hz": 2.0,
            "rotation_gate_rad_s": 0.05,
            # OFF on real hardware since 2026-07-29, and this is a measurement
            # not a preference. Read back from Trial A run 2's bag: while the
            # rover drove on a flat lab floor, /exomy/imu_raw was live at 50 Hz
            # with every message different -- nothing frozen -- but chassis
            # vibration swung |a| between 0.521 g and 2.306 g against a true
            # 1.000 g and produced apparent tilts of 10-74 deg. accel_gate_g
            # cannot filter that: it gates magnitude, not direction, and the
            # recorded sample tilt=53.04 deg at |a|=0.979 g sits inside a
            # 1.000 +/- 0.05 g gate while pointing nowhere near down.
            #
            # The false STOP was not harmless. Over-tilt is handled like a
            # blocked corridor, so it forces reactive_explorer to re-decide its
            # heading, and run 2 ended in FAILSAFE(boxed_in) when that fired in
            # front of a cupboard the LiDAR could offer no way around.
            #
            # The node still computes and publishes tilt, so the limitation
            # stays measurable and Chapter 5 can quote it. Separating gravity
            # from vibration needs gyro fusion (complementary/Madgwick), which
            # is future work. Gazebo keeps the override -- every reported
            # simulation result ran with it live.
            "slope_override_enabled": False,
        }],
    )

    # ── 8b. Gyro odometry -- ALWAYS ON, not part of the SLAM opt-in ─────────
    # Un-gated from use_slam on 2026-07-28: the 2026-07-27 reactive_explorer_
    # node redesign made /exomy/odom structurally required, because
    # TURN_TO_HEADING is a closed-loop yaw maneuver that can never report
    # completion without live yaw. With this gated off by default the rover
    # waited in STARTUP_CHECK forever on every real-hardware launch. This node
    # only integrates the IMU's gyro into an Odometry message plus a TF; it
    # has no slam_toolbox dependency, so starting it always costs nothing.
    # The "never produced a pose estimate in 3 Gazebo attempts" caveat in
    # gyro_odom_publisher_node.py's docstring is about the SLAM stack built on
    # top of this, not about the odometry this node itself publishes.
    gyro_odom_node = Node(
        package="fm_imu_fusion",
        executable="gyro_odom_publisher_node.py",
        name="gyro_odom_publisher",
        output="screen",
    )

    # ── 8c. L4 localisation (SLAM) -- off by default, PREPARED/UNTESTED ─────
    # See use_slam's DeclareLaunchArgument description and
    # gyro_odom_publisher_node.py's docstring for the important caveat: this
    # exact architecture never produced a pose estimate in 3 Gazebo attempts.
    slam_toolbox_node = Node(
        package="slam_toolbox",
        executable="async_slam_toolbox_node",
        name="slam_toolbox",
        output="screen",
        condition=IfCondition(LaunchConfiguration("use_slam")),
        parameters=[{
            "odom_frame": "gyro_odom",
            "map_frame": "map",
            "base_frame": "base_link",
            "scan_topic": "/scan",
            "mode": "mapping",
            "transform_publish_period": 0.02,
            "map_update_interval": 2.0,
            "resolution": 0.05,
            "max_laser_range": 12.0,
            "minimum_travel_distance": 0.2,
            "minimum_travel_heading": 0.2,
            "transform_timeout": 0.5,
        }],
    )

    # ── 9. cmd_vel -> RoverCommand bridge ───────────────────────────────────
    cmd_vel_bridge_node = Node(
        package="exomy_ros2",
        executable="cmd_vel_to_rovercommand_bridge_node.py",
        name="cmd_vel_to_rovercommand_bridge_node",
        output="screen",
        parameters=[{
            "cmd_vel_topic": "/exomy/cmd_vel",
            "rover_command_topic": "/rover_command",
            "max_linear_velocity": 0.10,
            "max_angular_velocity": 0.3,
            "stale_timeout_s": 3.0,
            "publish_rate_hz": 10.0,
        }],
    )

    # ── 10. Real-hardware driver chain (robot_node + motor_node) ────────────
    hardware_driver = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory("exomy_ros2"), "launch", "exomy_ros2.launch.py")
        ),
    )

    # ── 11. Foxglove bridge -- live viewing in Foxglove Studio (ws://8765) ──
    # Requires ros-humble-foxglove-bridge; disable with use_foxglove:=false if
    # not installed yet (the recorded bag below + RViz2 work regardless).
    foxglove_bridge = Node(
        package="foxglove_bridge",
        executable="foxglove_bridge",
        name="foxglove_bridge",
        output="screen",
        condition=IfCondition(LaunchConfiguration("use_foxglove")),
        parameters=[{"port": 8765}],
    )

    # ── 12. ROS2 bag recording -- full session log for offline review, and
    #         the "pull full video after each run" half of the hybrid
    #         streaming approach (see docstring). Relative to the directory
    #         `ros2 launch` is run from (repo root, per this repo's own run
    #         instructions) -- not resolved through the installed package
    #         share dir, which points at very different paths on the dev
    #         machine vs an RPi install and isn't a reliable repo-root anchor.
    bag_name = os.path.join(
        "bags", f"real_hardware_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    bag_record = ExecuteProcess(
        cmd=[
            "ros2", "bag", "record",
            # NOT /camera/image_raw. Recording the raw stream produced an 11 GB
            # bag for a 573 s run on 2026-07-29, nearly all of it uncompressed
            # duplicates of a view that was not changing because the rover was
            # not moving, and it filled a third of the card. This topic carries
            # the frame behind each DINOv2 verdict as JPEG, one per inference,
            # which is the evidence the thesis wants and is roughly 500x
            # smaller.
            "/terrain_classified_image",
            "/terrain_classification",
            # Separates "I cannot see" from "the ground is bad": a frame the
            # blank-frame gate rejected scores 1.0 exactly like an impassable
            # rock, and on 2026-07-29 a featureless wall 0.28 m away made the
            # rover refuse a heading it had never measured.
            "/terrain_frame_informative",
            "/terrain_class_probs",
            "/traversability_score",
            # Added 2026-08-11 for the field test. Without it there is no way to
            # answer whether the camera preview stream slowed inference down, and
            # so no way to say whether field numbers are comparable with the ones
            # already in the report. It is also the only record of the gap between
            # the 386 ms isolated benchmark and the 1,865-2,200 ms the full
            # thirteen-node stack actually achieves on four cores.
            "/inference_latency_ms",
            "/terrain_controller/stopped",
            "/scan",
            "/lidar_proximity_stop",
            # The bag recorded THAT the guard stopped but never why. The reason
            # string is what turns a stop event in a graph into an explanation.
            "/lidar_proximity_reason",
            "/reactive_explorer/active",
            "/reactive_explorer/failsafe",
            "/exomy/imu_raw",
            "/imu_slope_stop",
            "/traversability_fused",
            "/stuck_detection/active",
            "/stuck_detection/failsafe",
            "/exomy/cmd_vel",
            # Needed to place events along the traverse in the offline graphs:
            # "the rover stopped 2.4 m in" reads better than "at t=51 s", and a
            # field run has no zone grid to fall back on the way Gazebo did.
            # Gyro-integrated and drifting, so use it for relative position within
            # a run and never as ground truth.
            "/exomy/odom",
            "/rover_command",
            "/e_stop",
            "/e_stop_reason",
            "/map",
            "/pose",
            "-o", bag_name,
        ],
        output="screen",
    )

    return LaunchDescription([
        use_camera, use_lidar, use_imu, lidar_serial_port, sllidar_launch_file,
        use_attention_viz, use_state_viz, use_int8_model, use_foxglove, use_slam,
        exposure_lock_delay_s,
        use_preview, preview_fps, use_sensor_overlay, use_terrain_viz, use_sensor_tf,
        camera_node,
        exposure_lock,
        lidar_driver,
        dinov2_node,
        attention_viz_node,
        state_viz_node,
        preview_node,
        sensor_overlay_node,
        lidar_static_tf,
        controller_node,
        lidar_guard_node,
        reactive_explorer_node,
        stuck_detection_node,
        watchdog_node,
        imu_driver_node,
        imu_fusion_node,
        gyro_odom_node,
        slam_toolbox_node,
        cmd_vel_bridge_node,
        hardware_driver,
        foxglove_bridge,
        bag_record,
    ])
