#!/usr/bin/env python3
"""
Purpose: I2C driver node for the IMU breakout on ExoMy. Reads raw
         accelerometer registers over I2C, converts to a tilt quaternion,
         and publishes sensor_msgs/Imu so imu_slope_fusion_node.py has real
         hardware data to fuse instead of only working in Gazebo.
         Gyroscope registers are also read (2026-07-17 addition) and published
         as angular_velocity: the L4 SLAM feasibility investigation
         (simulation/launch/slam_test_odom_free.launch.py) found that
         ExoMy's lack of wheel encoders leaves slam_toolbox with no motion
         prior at all, and gyro-based yaw-rate is the most promising
         low-cost substitute -- much less prone to drift from wheel slip on
         sand than translation-integrating alternatives. Magnetometer is
         still not read: neither the original slope-detection use case nor
         this new gyro-assist use case needs full 9-DOF orientation, so this
         has never needed more than the 6-DOF this chip provides.
Inputs:  I2C bus 1 (hardware I2C1, via the replacement Servo HAT's SDA/SCL
         breakout pads), MPU-6050 at address 0x68 (AD0 tied low, the HW-123
         breakout's default).
         History: this node originally targeted a SparkFun ICM-20948 (9-DOF,
         bank-switched register map, address 0x69). That board was replaced
         2026-08-19 after failing intermittently; the replacement bought as
         a like-for-like swap turned out on visual inspection (silkscreen
         "HW-123" / "ITG/MPU", pinout VCC/GND/SCL/SDA/XDA/XCL/AD0/INT) to be
         an MPU-6050 breakout, not an ICM-20948 -- a completely different
         register map (flat, no bank select) under a similar-looking
         breakout form factor. This explained every symptom seen while it
         ran as an ICM-20948: i2cdetect and raw smbus2 reads/writes to 0x68
         always succeeded (right chip, right address, right wiring), but the
         ICM-20948 driver's bank-select write (register 0x7F, meaningless on
         this chip) and WHO_AM_I read (register 0x00, which on the MPU-6050
         is SMPLRT_DIV, not an identity register) never lined up, and the
         nonsense writes intermittently made the full launch stack's very
         first transaction NACK with OSError: [Errno 5]. This class was
         rewritten for the MPU-6050's actual register map on 2026-08-19; see
         [[project_imu_new_board_i2c_read_failure_20260819]] in memory.
         Before this: until 2026-07-23 this node defaulted to bus 3 (software
         i2c-gpio on GPIO20=SDA/GPIO21=SCL), because the original HAT's
         hardware I2C1 breakout pads were damaged during assembly (see
         H3c/H3d-IMU session notes). That HAT later failed completely and
         was replaced; on the replacement HAT, hardware I2C1 works.
Outputs: /exomy/imu_raw (sensor_msgs/Imu) — orientation quaternion (accel-only
         static tilt) and angular_velocity (rad/s, from the gyroscope) at
         ~50 Hz. angular_velocity_covariance[0] is set to -1.0 (unknown, no
         calibrated noise model yet) per REP-103, same convention already
         used for orientation_covariance below.
Failure behaviour:
         The node NEVER exits because of an I2C fault. If the chip cannot be
         reached at startup, or stops responding mid-run, the node stays up,
         publishes nothing, and retries the connection every
         reconnect_period_s until it succeeds -- then resumes publishing.
         This matters because every consumer downstream is chained to this
         topic: no /exomy/imu_raw means gyro_odom_publisher_node emits no
         /exomy/odom, which means reactive_explorer_node can never leave
         STARTUP_CHECK and the rover never moves at all. Before 2026-08-19
         the driver raised out of its constructor instead, so a single failed
         I2C transaction at boot permanently stranded the rover for that run.
         Silence on /exomy/imu_raw still stops the rover -- that is the
         deliberate fail-safe, unchanged -- but it is now recoverable without
         a relaunch.
ROS2 parameters:
         i2c_bus            (int,   default 1)    — /dev/i2c-<N>
         i2c_address        (int,   default 0x68) — MPU-6050 address (AD0 low)
         publish_hz         (float, default 50.0) — read/publish rate
         reconnect_period_s (float, default 2.0)  — retry spacing while down
         fault_log_period_s (float, default 10.0) — repeat-warning rate limit
How to run:
    pip3 install smbus2   # once, on the RPi
    source /opt/ros/humble/setup.bash
    cd ros2_ws && colcon build --packages-select fm_imu_fusion
    source install/setup.bash
    ros2 run fm_imu_fusion icm20948_driver_node.py
    # Verify:
    ros2 topic echo /exomy/imu_raw
Project: Onboard Visual Foundation Models for Mars Terrain Perception and Rover Navigation
"""

import math
import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu

# MPU-6050 register map — flat (no bank-select, unlike the ICM-20948 this
# node originally targeted), Register Map document RM-MPU-6000A-00.
SMPLRT_DIV = 0x19
CONFIG = 0x1A
GYRO_CONFIG = 0x1B
ACCEL_CONFIG = 0x1C
PWR_MGMT_1 = 0x6B
# One 14-byte burst starting here covers accel(6) + temp(2) + gyro(6); see
# Mpu6050.read_accel_and_gyro().
ACCEL_XOUT_H = 0x3B
GYRO_XOUT_H = 0x43   # 6 contiguous bytes: X,Y,Z high/low
WHO_AM_I = 0x75
# Fixed identity value(s) baked into the chip's silicon -- NOT the I2C
# address. 0x68 is the RM-MPU-6000A-00 datasheet value for a genuine
# MPU-6050. Confirmed live on the rover 2026-08-19: this specific HW-123
# breakout reads back 0x72 instead, on a chip that otherwise matches the
# MPU-6050 register map exactly (WHO_AM_I read succeeds cleanly, reset/wake/
# configure/read-burst all behave exactly as the datasheet describes). The
# HW-123 is a generic breakout that different sellers populate with
# different silicon (MPU-6050 clones, MPU-6500, relabelled parts) under an
# identical silkscreen and pinout, so a WHO_AM_I that doesn't match the
# canonical MPU-6050 constant is expected here, not a sign of a wrong
# chip or a bad read. Both values are accepted rather than widening this to
# "accept anything": an unrecognised value should still fail loudly, because
# that is what actually caught the real ICM-20948-vs-MPU-6050 mixup this
# module was rewritten for.
WHO_AM_I_ACCEPTED = frozenset({0x68, 0x72})

# PWR_MGMT_1 bit fields.
PWR_DEVICE_RESET = 0x80  # self-clearing; chip needs ~100 ms afterwards
PWR_SLEEP = 0x40         # set on power-up and after reset -- must be cleared
PWR_CLKSEL_PLL_X = 0x01  # PLL locked to the X gyro, more stable than the
                         # internal 8 MHz RC oscillator

ACCEL_SENSITIVITY_2G = 16384.0    # LSB/g at the default +-2g full-scale range
# sensor_msgs/Imu.linear_acceleration is specified in m/s^2 (REP-103), while
# this driver works internally in g.
G_TO_M_S2 = 9.80665
GYRO_SENSITIVITY_250DPS = 131.0   # LSB/(deg/s) at the default +-250 dps full-scale range


def raw_to_g(raw: int, sensitivity: float = ACCEL_SENSITIVITY_2G) -> float:
    """Convert an unsigned 16-bit register reading (two's complement) to g's."""
    if raw > 32767:
        raw -= 65536
    return raw / sensitivity


def raw_to_dps(raw: int, sensitivity: float = GYRO_SENSITIVITY_250DPS) -> float:
    """Convert an unsigned 16-bit register reading (two's complement) to
    degrees per second. Same two's-complement conversion as raw_to_g(), kept
    as a separate function (rather than reusing raw_to_g under a different
    sensitivity) since the two represent physically different quantities and
    a shared name would be misleading."""
    if raw > 32767:
        raw -= 65536
    return raw / sensitivity


def accel_magnitude_g(ax_g: float, ay_g: float, az_g: float) -> float:
    """Length of the measured acceleration vector, in g.

    A stationary rover measures gravity alone, so this is 1.000 g whatever
    its orientation -- parked level or parked on a 30 deg slope both read
    1 g. Any departure from 1 g therefore means a non-gravitational force is
    present (acceleration, braking, a wheel hitting a joint, vibration), and
    while that is true the accel-only tilt angle from accel_to_pitch_roll()
    is measuring the sum of gravity and that force rather than gravity.

    imu_slope_fusion_node.gate_tilt() uses exactly this to reject the
    artifacts that produced apparent tilts up to 62.9 deg on a flat lab floor
    during Trial A (2026-07-29).
    """
    return math.sqrt(ax_g * ax_g + ay_g * ay_g + az_g * az_g)


def accel_to_pitch_roll(ax_g: float, ay_g: float, az_g: float) -> tuple:
    """Static-accelerometer tilt estimate. Returns (pitch_deg, roll_deg).

    Assumes the rover is roughly stationary (no significant linear
    acceleration) so gravity dominates the reading — true for the slow
    slope-detection use case here, not a general-purpose attitude filter.
    """
    pitch = math.degrees(math.atan2(-ax_g, math.sqrt(ay_g * ay_g + az_g * az_g)))
    roll = math.degrees(math.atan2(ay_g, az_g))
    return pitch, roll


def euler_to_quaternion(roll_deg: float, pitch_deg: float, yaw_deg: float = 0.0) -> tuple:
    """Roll/pitch/yaw (degrees, ZYX intrinsic) -> quaternion (x, y, z, w).

    Matches the convention decoded by imu_slope_fusion_node.py's
    _quaternion_to_pitch/_quaternion_to_roll, so whatever this publishes
    round-trips correctly through the existing fusion node.
    """
    roll = math.radians(roll_deg)
    pitch = math.radians(pitch_deg)
    yaw = math.radians(yaw_deg)

    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)

    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    return qx, qy, qz, qw


def decode_accel_gyro_burst(data) -> tuple:
    """Decode the 14-byte register burst starting at ACCEL_XOUT_H.

    Layout (RM-MPU-6000A-00 sec. 4.17-4.19): accel X/Y/Z, then temperature,
    then gyro X/Y/Z, each big-endian 16-bit two's complement. Temperature is
    skipped -- nothing downstream consumes it.

    Returns (ax_g, ay_g, az_g, gx_dps, gy_dps, gz_dps). Pure function, so it
    is unit-testable without I2C hardware, unlike the class below.
    """
    if len(data) != 14:
        raise ValueError(f"expected a 14-byte burst, got {len(data)} bytes")
    ax_raw = (data[0] << 8) | data[1]
    ay_raw = (data[2] << 8) | data[3]
    az_raw = (data[4] << 8) | data[5]
    # data[6:8] is the temperature reading -- deliberately unused.
    gx_raw = (data[8] << 8) | data[9]
    gy_raw = (data[10] << 8) | data[11]
    gz_raw = (data[12] << 8) | data[13]
    return (
        raw_to_g(ax_raw),
        raw_to_g(ay_raw),
        raw_to_g(az_raw),
        raw_to_dps(gx_raw),
        raw_to_dps(gy_raw),
        raw_to_dps(gz_raw),
    )


class Mpu6050:
    """smbus2 wrapper for the MPU-6050 — I/O paths need real hardware, but
    the byte decoding lives in decode_accel_gyro_burst() above so it can be
    unit tested.

    Renamed from Icm20948 on 2026-08-19 when the actual breakout on the
    rover was identified as an MPU-6050, not an ICM-20948 (see module
    docstring). Kept as its own class rather than a generic "Imu" name so a
    future re-swap back to a real ICM-20948 is a clean revert, not another
    register-map archaeology exercise.

    Connection lifecycle is explicit (connect()/close()/connected) rather
    than done in __init__. Constructing this object touches no hardware, so
    the owning node can come up, publish nothing, and keep retrying instead
    of dying when the bus is unavailable -- which is what actually stranded
    the rover on 2026-08-19: the driver raised out of __init__, its process
    died, /exomy/imu_raw never appeared, gyro_odom_publisher never produced
    /exomy/odom, and reactive_explorer_node sat in STARTUP_CHECK forever.
    """

    def __init__(self, bus_num: int, address: int):
        self._bus_num = bus_num
        self._addr = address
        self._bus = None

    @property
    def connected(self) -> bool:
        return self._bus is not None

    def connect(self) -> None:
        """Open the I2C bus and bring the chip into a known, configured state.

        Raises OSError (bus/device I/O) or RuntimeError (wrong or unresponsive
        chip). On any failure the bus handle is closed again, so repeated
        reconnect attempts cannot leak file descriptors.
        """
        import smbus2  # imported here so the pure functions above stay
        # importable/testable on machines without smbus2 installed (e.g. this
        # WSL dev PC, which has no /dev/i2c device at all).

        self.close()
        bus = smbus2.SMBus(self._bus_num)
        try:
            who_am_i = bus.read_byte_data(self._addr, WHO_AM_I)
            if who_am_i not in WHO_AM_I_ACCEPTED:
                accepted = ", ".join(f"0x{v:02X}" for v in sorted(WHO_AM_I_ACCEPTED))
                raise RuntimeError(
                    f"MPU-6050 WHO_AM_I mismatch: got 0x{who_am_i:02X}, "
                    f"accepted values are {{{accepted}}} — wrong chip, wrong "
                    f"address, or a bus read that returned garbage"
                )

            # Full device reset first, so a half-configured chip left behind
            # by a previous crashed run cannot silently persist. The bit is
            # self-clearing and the datasheet requires a settling delay.
            bus.write_byte_data(self._addr, PWR_MGMT_1, PWR_DEVICE_RESET)
            time.sleep(0.1)

            # Wake (SLEEP is set out of reset) and select the X-gyro PLL.
            bus.write_byte_data(self._addr, PWR_MGMT_1, PWR_CLKSEL_PLL_X)
            time.sleep(0.05)

            # Confirm the wake actually took. Without this a NACKed-but-not-
            # errored write leaves the chip asleep and every subsequent read
            # returns zeros, which downstream reads as "perfectly level,
            # perfectly still" -- a silent, dangerous failure mode rather
            # than a loud one.
            pwr = bus.read_byte_data(self._addr, PWR_MGMT_1)
            if pwr & PWR_SLEEP:
                raise RuntimeError(
                    f"MPU-6050 stayed asleep after wake command "
                    f"(PWR_MGMT_1=0x{pwr:02X})"
                )

            # Force the full-scale ranges the sensitivity constants above
            # assume. These are the power-on defaults, but writing them
            # explicitly means the constants stay correct even if something
            # else on the bus (or a previous run) changed them.
            bus.write_byte_data(self._addr, GYRO_CONFIG, 0x00)   # +-250 dps
            bus.write_byte_data(self._addr, ACCEL_CONFIG, 0x00)  # +-2 g
        except BaseException:
            try:
                bus.close()
            except OSError:
                pass
            raise

        self._bus = bus

    def close(self) -> None:
        """Release the bus handle. Safe to call when already closed."""
        if self._bus is not None:
            try:
                self._bus.close()
            except OSError:
                pass
            self._bus = None

    def read_accel_and_gyro(self) -> tuple:
        """Return (ax_g, ay_g, az_g, gx_dps, gy_dps, gz_dps).

        One 14-byte burst rather than the two separate 6-byte reads this used
        to do. Two reasons: it halves the number of I2C transactions per
        publish cycle (at 50 Hz that is 50 fewer transactions per second on a
        bus shared with the PCA9685 servo HAT, which matters because that
        contention is a live suspect for the dropouts), and accelerometer and
        gyroscope samples now come from one instant instead of two reads a
        few hundred microseconds apart.
        """
        if self._bus is None:
            raise OSError("MPU-6050 read attempted while disconnected")
        data = self._bus.read_i2c_block_data(self._addr, ACCEL_XOUT_H, 14)
        return decode_accel_gyro_burst(data)


class Icm20948DriverNode(Node):
    """Node name, executable name and topic are all still "icm20948"-flavoured
    even though the chip is an MPU-6050. Renaming them would mean touching
    setup.py, every launch file and the deployed Pi workspace at the same time
    as changing the driver logic; keeping them stable makes this change
    independently revertable. The chip identity lives in Mpu6050 above.
    """

    def __init__(self):
        super().__init__("icm20948_driver_node")

        self.declare_parameter("i2c_bus", 1)
        self.declare_parameter("i2c_address", 0x68)
        self.declare_parameter("publish_hz", 50.0)
        # How long to wait between reconnect attempts while the IMU is down.
        # At 50 Hz the publish timer fires every 20 ms; retrying that fast
        # would hammer a sick bus (and, if bus contention is the cause, make
        # it worse) while filling the log.
        self.declare_parameter("reconnect_period_s", 2.0)
        # Rate limit for repeated failure messages, so a persistent fault
        # leaves a readable trail rather than 50 lines a second.
        self.declare_parameter("fault_log_period_s", 10.0)

        bus_num = self.get_parameter("i2c_bus").value
        address = self.get_parameter("i2c_address").value
        publish_hz = self.get_parameter("publish_hz").value
        self._reconnect_period_s = self.get_parameter("reconnect_period_s").value
        self._fault_log_period_s = self.get_parameter("fault_log_period_s").value

        self._imu = Mpu6050(bus_num, address)
        self._pub = self.create_publisher(Imu, "/exomy/imu_raw", 10)

        self._failure_count = 0
        self._last_attempt_s = 0.0
        self._last_fault_log_s = 0.0

        self.get_logger().info(
            f"icm20948_driver_node started | chip=MPU-6050 bus={bus_num} "
            f"addr=0x{address:02X} rate={publish_hz}Hz "
            f"reconnect_every={self._reconnect_period_s}s"
        )
        # Attempt the first connection here for a fast, clearly-logged start,
        # but never let a failure escape: an exception out of __init__ kills
        # the process, and a dead IMU process is what strands the whole rover
        # (no /exomy/imu_raw -> no /exomy/odom -> reactive_explorer_node stuck
        # in STARTUP_CHECK forever). Failing here just means the timer will
        # keep retrying.
        self._try_connect()
        self.create_timer(1.0 / publish_hz, self._read_and_publish)

    def _try_connect(self) -> bool:
        """Attempt one connection. Returns True on success. Never raises."""
        self._last_attempt_s = time.monotonic()
        try:
            self._imu.connect()
        except (OSError, RuntimeError) as exc:
            self._failure_count += 1
            self._log_fault(
                f"MPU-6050 not available (attempt {self._failure_count}): {exc}. "
                f"Retrying every {self._reconnect_period_s}s. Until it "
                f"connects, /exomy/imu_raw stays silent and the rover will "
                f"hold in STARTUP_CHECK."
            )
            return False

        if self._failure_count:
            self.get_logger().info(
                f"MPU-6050 online after {self._failure_count} failed "
                f"attempt(s) — publishing /exomy/imu_raw"
            )
        else:
            self.get_logger().info("MPU-6050 online — publishing /exomy/imu_raw")
        self._failure_count = 0
        return True

    def _log_fault(self, message: str) -> None:
        """Rate-limited warning. The first fault after a good period is always
        logged; repeats are suppressed to one per fault_log_period_s."""
        now = time.monotonic()
        if self._failure_count <= 1 or now - self._last_fault_log_s >= self._fault_log_period_s:
            self._last_fault_log_s = now
            self.get_logger().warn(message)

    def _read_and_publish(self) -> None:
        if not self._imu.connected:
            if time.monotonic() - self._last_attempt_s >= self._reconnect_period_s:
                self._try_connect()
            return

        try:
            ax_g, ay_g, az_g, gx_dps, gy_dps, gz_dps = self._imu.read_accel_and_gyro()
        except (OSError, ValueError) as exc:
            # Drop the handle so the branch above owns reconnection. Leaving a
            # half-dead handle open and retrying reads on it is how the old
            # code produced an endless "read failed" stream with no recovery.
            self._imu.close()
            self._failure_count += 1
            self._log_fault(f"MPU-6050 read failed, will reconnect: {exc}")
            return

        pitch_deg, roll_deg = accel_to_pitch_roll(ax_g, ay_g, az_g)
        qx, qy, qz, qw = euler_to_quaternion(roll_deg, pitch_deg, yaw_deg=0.0)

        msg = Imu()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "imu_link"
        msg.orientation.x = qx
        msg.orientation.y = qy
        msg.orientation.z = qz
        msg.orientation.w = qw
        # Orientation covariance unknown (accel-only tilt, no filter) —
        # first element -1 signals "no covariance estimate" per REP-103.
        msg.orientation_covariance[0] = -1.0
        # angular_velocity is in rad/s per REP-103; the sensor/raw_to_dps()
        # give degrees/s.
        msg.angular_velocity.x = math.radians(gx_dps)
        msg.angular_velocity.y = math.radians(gy_dps)
        msg.angular_velocity.z = math.radians(gz_dps)
        msg.angular_velocity_covariance[0] = -1.0
        # Added 2026-07-29. The raw acceleration used to be consumed by
        # accel_to_pitch_roll() above and then discarded, which left
        # imu_slope_fusion_node unable to tell a real slope (magnitude 1 g)
        # from an acceleration or vibration artifact (magnitude != 1 g). That
        # gap is what let Trial A report 62.9 deg of tilt on a flat floor.
        # Covariance is left at zero rather than -1: the data IS available
        # here, only its uncertainty is unquantified.
        msg.linear_acceleration.x = ax_g * G_TO_M_S2
        msg.linear_acceleration.y = ay_g * G_TO_M_S2
        msg.linear_acceleration.z = az_g * G_TO_M_S2
        self._pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = Icm20948DriverNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Release the /dev/i2c-N file descriptor before tearing down rclpy, so
        # a relaunch straight after Ctrl+C never finds the handle still held.
        node._imu.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
