"""
Purpose: Wheel-by-wheel check of all 12 ExoMy servos using the as-built
         PCA9685 channel map verified on the rover 2026-07-23. Replaces the
         stale ~/test_servo.py on the Pi, which drives channels 3, 12 and 13
         (nothing wired) and never touches 8, 9 or 10, making three wheels
         look dead when they are fine.

         Each wheel is exercised on its own and named before it moves, so a
         non-responding wheel is unambiguous. Drive motors always return to
         zero throttle, including on Ctrl+C or an exception, because
         continuous-rotation servos hold their last command indefinitely.

Inputs:  none. Channel map is the as-built table, kept in sync with
         ros2_ws/src/exomy_ros2/exomy_ros2/motor_node.py DEFAULT_PINS
         (locked by test/test_motor_pin_mapping.py).
Outputs: console log naming each wheel as it moves; physical servo motion.
How to run:
    # From the dev laptop, copy to the Pi:
    scp experiments/test_servo_as_built.py pi@172.20.10.13:~/
    # Then on the Pi, with the rover LIFTED and all six wheels clear:
    python3 ~/test_servo_as_built.py
    # One wheel only:
    python3 ~/test_servo_as_built.py fl
Project: Onboard Visual Foundation Models for Mars Terrain Perception and Rover Navigation
"""

import sys
import time

from adafruit_servokit import ServoKit

# As-built channels, verified one at a time on the rover 2026-07-23.
# Channels 3, 11, 12, 13 are unused.
WHEELS = [
    ("fl", "front-left", 8, 14),
    ("fr", "front-right", 0, 2),
    ("cl", "centre-left", 7, 6),
    ("cr", "centre-right", 5, 4),
    ("rl", "rear-left", 9, 10),
    ("rr", "rear-right", 1, 15),
]

STEER_CENTRE_DEG = 90
STEER_SWEEP_DEG = 30      # kept small: enough to see, far from the end stops
DRIVE_THROTTLE = 0.3
PAUSE_S = 0.6


def test_wheel(kit, name, label, steer_ch, drive_ch):
    print(f"\n[{name.upper()}] {label}")

    print(f"  steer ch{steer_ch}: centre -> left -> centre")
    kit.servo[steer_ch].angle = STEER_CENTRE_DEG
    time.sleep(PAUSE_S)
    kit.servo[steer_ch].angle = STEER_CENTRE_DEG - STEER_SWEEP_DEG
    time.sleep(PAUSE_S)
    kit.servo[steer_ch].angle = STEER_CENTRE_DEG
    time.sleep(PAUSE_S)

    print(f"  drive ch{drive_ch}: spin {PAUSE_S}s then stop")
    kit.continuous_servo[drive_ch].throttle = DRIVE_THROTTLE
    time.sleep(PAUSE_S)
    kit.continuous_servo[drive_ch].throttle = 0.0
    time.sleep(0.3)


def stop_all_drives(kit):
    for _, _, _, drive_ch in WHEELS:
        kit.continuous_servo[drive_ch].throttle = 0.0


def main():
    wanted = [a.lower() for a in sys.argv[1:]]
    selected = [w for w in WHEELS if not wanted or w[0] in wanted]
    if not selected:
        print(f"No wheel matched {wanted}. Valid: {[w[0] for w in WHEELS]}")
        return 1

    print("Rover must be LIFTED with all wheels clear. Drive wheels will spin.")
    kit = ServoKit(channels=16)

    try:
        for name, label, steer_ch, drive_ch in selected:
            test_wheel(kit, name, label, steer_ch, drive_ch)
    finally:
        stop_all_drives(kit)

    print(f"\nDONE. Tested {len(selected)} wheel(s), all drives at zero throttle.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
