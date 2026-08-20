#!/usr/bin/env python3
"""
Purpose:    ExoMy hardware motor driver — controls 12 PWM channels (6 steering
            + 6 drive) via Adafruit PCA9685 over I2C on Raspberry Pi.
Inputs:     ROS2 Node (for parameter access), Adafruit PCA9685 hardware
Outputs:    PWM signals to servo/ESC motors
How to run: Imported by motor_node.py — not run directly. RPi 4 only.
Project:    Onboard Visual Foundation Models for Mars Terrain Perception and Rover Navigation
Original:   github.com/esa-prl/ExoMy_Software motors.py
            (rospy.get_param → ROS2 parameters; logger via node)
"""

import time
import Adafruit_PCA9685


class Motors:
    """Hardware driver for ExoMy's 6-wheel drive+steer system (RPi only)."""

    FL, FR, CL, CR, RL, RR = range(6)
    wheel_directions = [-1, 1, -1, 1, -1, 1]

    def __init__(self, node):
        self._log = node.get_logger()

        def p(name):
            return node.get_parameter(name).get_parameter_value().integer_value

        self.pins = {"drive": {}, "steer": {}}
        for w, side in [(self.FL, "fl"), (self.FR, "fr"), (self.CL, "cl"),
                        (self.CR, "cr"), (self.RL, "rl"), (self.RR, "rr")]:
            self.pins["drive"][w] = p(f"pin_drive_{side}")
            self.pins["steer"][w] = p(f"pin_steer_{side}")

        # Adafruit_GPIO's platform auto-detection can't recognise a Pi 4 on
        # Debian trixie / kernel 6.18 ("Could not determine default I2C bus
        # for platform"), so pass the bus number explicitly. The Servo HAT's
        # PCA9685 is on hardware I2C1 (bus 1) -- same bus i2cdetect -y 1 shows
        # 0x40 on, and the same bus the IMU uses.
        busnum = p("i2c_bus")
        self.pwm = Adafruit_PCA9685.PCA9685(busnum=busnum)
        self.pwm.set_pwm_freq(50)

        self.steer_neutral = [
            p(f"steer_pwm_neutral_{s}")
            for s in ["fl", "fr", "cl", "cr", "rl", "rr"]
        ]
        self.steer_range    = p("steer_pwm_range")
        self.drive_neutral  = p("drive_pwm_neutral")
        self.drive_low      = 100
        self.drive_high     = 500
        self.drive_range    = p("drive_pwm_range")

        # Silence the drive channels before anything else. The PCA9685 keeps
        # its registers across process restarts as long as it stays powered,
        # so a channel left driving by a previous run would keep driving from
        # the moment this one starts. Nothing should move until a command
        # actually asks for it.
        for pin in self.pins["drive"].values():
            self.pwm.set_pwm(pin, 0, 0)

        for w, pin in self.pins["steer"].items():
            self.pwm.set_pwm(pin, 0, self.steer_neutral[w])
            time.sleep(0.05)

        self._log.info("Motors initialised — running startup wiggle")
        self._wiggle()

    def _wiggle(self):
        for offset in [0.3, -0.3, 0.0]:
            for w, pin in self.pins["steer"].items():
                self.pwm.set_pwm(
                    pin, 0, int(self.steer_neutral[w] + self.steer_range * offset)
                )
            time.sleep(0.3)

    def setSteering(self, steering_command):
        for w, pin in self.pins["steer"].items():
            dc = int(
                self.steer_neutral[w]
                + steering_command[w] / 90.0 * self.steer_range
            )
            self.pwm.set_pwm(pin, 0, dc)

    def setDriving(self, driving_command):
        for w, pin in self.pins["drive"].items():
            if driving_command[w] == 0:
                # A commanded speed of zero has to be as final as an
                # emergency stop, so it takes the same signal-off path rather
                # than going through the drive_neutral arithmetic below. See
                # stopMotors() for why a neutral pulse does not stop a
                # continuous-rotation servo.
                self.pwm.set_pwm(pin, 0, 0)
                continue
            dc = int(
                self.drive_neutral
                + driving_command[w] / 100.0
                * self.drive_range
                * self.wheel_directions[w]
            )
            self.pwm.set_pwm(pin, 0, dc)

    def stopMotors(self):
        """Cut the drive signal. Every safety path in this project ends here.

        This used to write drive_neutral (307 ticks, a 1.5 ms pulse). 1.5 ms
        is the textbook neutral, but a continuous-rotation servo's real stop
        point is set by its own trim and is never exactly the textbook value.
        Observed on the rover 2026-07-24: with nothing commanding it at all,
        motor_node's watchdog wrote 307 to the six drive channels every 5 s
        and the rover span in place continuously, digging sand into its own
        wheels. It span rather than crept because the left and right motors
        face opposite ways (wheel_directions), so one raw value drives the
        left wheels forward and the right wheels backward.

        Writing 0 disables the channel, so no pulse is produced and there is
        nothing for the servo to trim away from. Confirmed live: this stopped
        the rover when 307 had not.

        The steering channels are left alone on purpose. They are positional
        servos, and cutting their signal would drop their holding torque and
        let the wheels flop to wherever the terrain pushes them.
        """
        for pin in self.pins["drive"].values():
            self.pwm.set_pwm(pin, 0, 0)
