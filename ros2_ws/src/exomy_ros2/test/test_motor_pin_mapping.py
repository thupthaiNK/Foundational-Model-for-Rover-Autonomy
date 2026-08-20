"""
Purpose: Lock ExoMy's as-built PCA9685 channel assignment so it cannot drift
         back to the upstream ExoMy defaults. Until 2026-07-23 both
         MotorNode.DEFAULT_PINS and exomy_ros2.launch.py carried the upstream
         sequential layout (drive 0,2,4,6,8,10 / steer 1,3,5,7,9,11), which
         does not match how this rover is wired. That mismatch is dangerous,
         not cosmetic: with it in place, channels the rover uses for
         positional steering servos receive continuous-rotation throttle
         commands, which drives a steering servo into its end stop and holds
         it there drawing stall current.

         The correct mapping was established by driving one PCA9685 channel
         at a time on the real rover (2026-07-23, replacement Servo HAT) and
         recording which wheel moved. All 12 channels were confirmed;
         channels 3, 11, 12, 13 are unused.

         Also asserts the launch file agrees with the node defaults, since
         the two are separate literals that must not diverge.
Inputs:  None (parses exomy_ros2.launch.py as text; no ROS2 runtime needed).
Outputs: pytest results.
How to run:
    cd ros2_ws && colcon test --packages-select exomy_ros2
    # or, without a ROS2 environment:
    python3 -m pytest src/exomy_ros2/test/test_motor_pin_mapping.py -q
Project: Onboard Visual Foundation Models for Mars Terrain Perception and Rover Navigation
"""

import ast
import pathlib

# The as-built mapping, verified wheel-by-wheel on the rover 2026-07-23.
AS_BUILT_PINS = {
    "pin_drive_fl": 14, "pin_steer_fl": 8,
    "pin_drive_fr": 2,  "pin_steer_fr": 0,
    "pin_drive_cl": 6,  "pin_steer_cl": 7,
    "pin_drive_cr": 4,  "pin_steer_cr": 5,
    "pin_drive_rl": 10, "pin_steer_rl": 9,
    "pin_drive_rr": 15, "pin_steer_rr": 1,
}

_SRC = pathlib.Path(__file__).resolve().parents[1]
_NODE_FILE = _SRC / "exomy_ros2" / "motor_node.py"
_LAUNCH_FILE = _SRC / "launch" / "exomy_ros2.launch.py"


def _pin_literals(path):
    """Collect every `"pin_*": <int>` pair in a source file.

    Parsed from the AST rather than imported, so the test runs without
    rclpy or Adafruit_PCA9685 installed (neither exists on the dev laptop).
    """
    tree = ast.parse(path.read_text())
    pins = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        for key, value in zip(node.keys, node.values):
            if (
                isinstance(key, ast.Constant)
                and isinstance(key.value, str)
                and key.value.startswith("pin_")
                and isinstance(value, ast.Constant)
                and isinstance(value.value, int)
            ):
                pins[key.value] = value.value
    return pins


def test_motor_node_defaults_match_as_built_wiring():
    assert _pin_literals(_NODE_FILE) == AS_BUILT_PINS


def test_launch_file_matches_node_defaults():
    assert _pin_literals(_LAUNCH_FILE) == _pin_literals(_NODE_FILE)


def test_every_channel_is_used_exactly_once():
    channels = list(AS_BUILT_PINS.values())
    assert len(set(channels)) == 12, "two wheels share a PCA9685 channel"


def test_steer_and_drive_channels_are_disjoint():
    steer = {v for k, v in AS_BUILT_PINS.items() if k.startswith("pin_steer_")}
    drive = {v for k, v in AS_BUILT_PINS.items() if k.startswith("pin_drive_")}
    # A channel serving both would mean a positional servo receives
    # continuous-rotation throttle commands, the exact stall risk above.
    assert steer.isdisjoint(drive)


def test_channels_are_within_pca9685_range():
    assert all(0 <= c <= 15 for c in AS_BUILT_PINS.values())
