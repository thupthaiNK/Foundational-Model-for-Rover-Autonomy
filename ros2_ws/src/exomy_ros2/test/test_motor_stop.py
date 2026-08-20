"""
Purpose: Lock the behaviour of Motors.stopMotors(), which is the single code
         path every safety mechanism in this project ends in: the LiDAR
         proximity guard, the terrain safety watchdog, stuck detection, the
         IMU slope retreat and motor_node's own command watchdog all stop the
         rover by calling it.

         Until 2026-07-24 it wrote `drive_pwm_neutral` (307 ticks, a 1.5 ms
         pulse) to each drive channel. 1.5 ms is the textbook neutral, but a
         continuous-rotation servo's real stop point is set by its own trim
         and is never exactly the textbook value. Observed live: with nobody
         commanding the rover at all, the watchdog fired every 5 s, wrote 307
         to all six drive channels, and the rover span in place continuously.
         It span rather than crept because the left and right motors are
         mounted facing opposite ways (wheel_directions), so one raw value
         drives the left wheels forward and the right wheels backward.

         The fix is to cut the drive signal instead of sending a pulse the
         servo has to interpret. With no pulse there is nothing to trim away
         from, so the stop cannot depend on per-servo calibration. This was
         confirmed on the rover: writing 0 to the drive channels stopped it
         when writing 307 had not.

         Steering channels are deliberately left alone. They are positional
         servos, and cutting their signal would drop their holding torque and
         let the wheels flop to wherever the terrain pushes them.
Inputs:  None (stubs Adafruit_PCA9685 and the ROS2 node; no hardware needed).
Outputs: pytest results.
How to run:
    python3 -m pytest src/exomy_ros2/test/test_motor_stop.py -q
Project: Onboard Visual Foundation Models for Mars Terrain Perception and Rover Navigation
"""
import sys
import types

import pytest

# Values copied from MotorNode.DEFAULT_PINS / motors.py so the stub node can
# answer get_parameter() the way the real node does.
PARAMS = {
    "pin_drive_fl": 14, "pin_steer_fl": 8,
    "pin_drive_fr": 2,  "pin_steer_fr": 0,
    "pin_drive_cl": 6,  "pin_steer_cl": 7,
    "pin_drive_cr": 4,  "pin_steer_cr": 5,
    "pin_drive_rl": 10, "pin_steer_rl": 9,
    "pin_drive_rr": 15, "pin_steer_rr": 1,
    "steer_pwm_neutral_fl": 307, "steer_pwm_neutral_fr": 307,
    "steer_pwm_neutral_cl": 307, "steer_pwm_neutral_cr": 307,
    "steer_pwm_neutral_rl": 307, "steer_pwm_neutral_rr": 307,
    "steer_pwm_range": 100,
    "drive_pwm_neutral": 307,
    "drive_pwm_range": 200,
    "i2c_bus": 1,
}

DRIVE_PINS = {PARAMS[f"pin_drive_{s}"] for s in
              ("fl", "fr", "cl", "cr", "rl", "rr")}
STEER_PINS = {PARAMS[f"pin_steer_{s}"] for s in
              ("fl", "fr", "cl", "cr", "rl", "rr")}


class FakePWM:
    def __init__(self, busnum=None):
        self.writes = []

    def set_pwm_freq(self, hz):
        pass

    def set_pwm(self, channel, on, off):
        self.writes.append((channel, off))


class FakeLogger:
    def info(self, *a, **k):
        pass

    def warn(self, *a, **k):
        pass


class FakeNode:
    def get_logger(self):
        return FakeLogger()

    def get_parameter(self, name):
        value = PARAMS[name]
        return types.SimpleNamespace(
            get_parameter_value=lambda: types.SimpleNamespace(
                integer_value=value)
        )


@pytest.fixture
def motors(monkeypatch):
    stub = types.ModuleType("Adafruit_PCA9685")
    stub.PCA9685 = FakePWM
    monkeypatch.setitem(sys.modules, "Adafruit_PCA9685", stub)

    # Load motors.py from the source tree by path rather than importing the
    # installed package, so the test always exercises the file being edited
    # and not a stale copy under install/.
    import importlib.util
    import pathlib
    path = (pathlib.Path(__file__).resolve().parents[1]
            / "exomy_ros2" / "motors.py")
    spec = importlib.util.spec_from_file_location("_motors_under_test", path)
    motors_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(motors_mod)
    monkeypatch.setattr(motors_mod.time, "sleep", lambda *_: None)

    m = motors_mod.Motors(node=FakeNode())
    m.pwm.writes.clear()          # discard init + startup wiggle
    return m


def test_stop_cuts_the_drive_signal_rather_than_sending_neutral(motors):
    motors.stopMotors()
    for channel, off in motors.pwm.writes:
        assert off == 0, (
            f"drive channel {channel} was sent pulse width {off}; a stop must "
            "cut the signal, because a continuous-rotation servo's real stop "
            "point depends on its own trim and 307 made the rover spin"
        )


def test_stop_covers_every_drive_wheel(motors):
    motors.stopMotors()
    assert {c for c, _ in motors.pwm.writes} == DRIVE_PINS


def test_stop_does_not_touch_the_steering_channels(motors):
    motors.stopMotors()
    touched = {c for c, _ in motors.pwm.writes}
    assert not (touched & STEER_PINS), (
        "steering servos are positional; cutting their signal drops their "
        "holding torque and lets the wheels flop"
    )


def test_zero_speed_command_also_stops_rather_than_creeping(motors):
    # A commanded speed of 0 must be as final as an emergency stop. Going
    # through the drive_neutral arithmetic instead would creep for the same
    # reason the watchdog did.
    motors.setDriving([0.0] * 6)
    for channel, off in motors.pwm.writes:
        assert off == 0, (
            f"drive channel {channel} got {off} for a commanded speed of 0"
        )


def test_nonzero_speed_still_produces_a_pulse(motors):
    # Regression guard: the stop fix must not disable normal driving.
    motors.setDriving([20.0] * 6)
    assert motors.pwm.writes
    for channel, off in motors.pwm.writes:
        assert off != 0, f"drive channel {channel} got no pulse for speed 20"
