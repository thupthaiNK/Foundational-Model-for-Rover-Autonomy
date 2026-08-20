"""
Purpose: Unit tests for the pure-Python conversion functions in
         icm20948_driver_node.py — raw register -> g's/deg-per-second,
         accel -> tilt angle, and Euler -> quaternion. No I2C hardware
         required.
Inputs:  None.
Outputs: pytest results.
How to run:
    cd ros2_ws && colcon build --packages-select fm_imu_fusion
    source /opt/ros/humble/setup.bash && source install/setup.bash
    python3 -m pytest src/fm_imu_fusion/test/test_icm20948_driver.py -v
Project: Onboard Visual Foundation Models for Mars Terrain Perception and Rover Navigation
"""
import math

import pytest

from fm_imu_fusion.icm20948_driver_node import (
    raw_to_g, raw_to_dps, accel_to_pitch_roll, euler_to_quaternion,
    accel_magnitude_g, decode_accel_gyro_burst, G_TO_M_S2,
)
from fm_imu_fusion.imu_slope_fusion_node import _quaternion_to_pitch, _quaternion_to_roll


def test_raw_to_g_zero_is_zero_g():
    assert raw_to_g(0) == 0.0


def test_raw_to_g_positive_full_scale_is_one_g():
    # +-2g range: 16384 LSB/g (ICM-20948 default ACCEL_FS_SEL=0)
    assert raw_to_g(16384) == 1.0


def test_raw_to_g_negative_twos_complement_is_negative_g():
    # 49152 stored unsigned == -16384 as signed int16 == -1.0g
    assert raw_to_g(49152) == -1.0


def test_raw_to_dps_zero_is_zero_dps():
    assert raw_to_dps(0) == 0.0


def test_raw_to_dps_positive_full_scale_is_correct():
    # +-250 dps range: 131 LSB/(deg/s) (ICM-20948 default GYRO_FS_SEL=0)
    assert raw_to_dps(131) == 1.0


def test_raw_to_dps_negative_twos_complement_is_negative_dps():
    # 65536-131=65405 stored unsigned == -131 as signed int16 == -1.0 dps
    assert raw_to_dps(65405) == -1.0


def test_accel_to_pitch_roll_level_is_zero():
    pitch, roll = accel_to_pitch_roll(0.0, 0.0, 1.0)
    assert abs(pitch) < 1e-6
    assert abs(roll) < 1e-6


def test_accel_to_pitch_roll_pitched_forward_is_ninety():
    # ax=+1g, ay=0, az=0 -> nose-down/up 90 degree pitch
    pitch, roll = accel_to_pitch_roll(1.0, 0.0, 0.0)
    assert abs(abs(pitch) - 90.0) < 1e-3


def test_accel_to_pitch_roll_rolled_sideways_is_ninety():
    # ax=0, ay=+1g, az=0 -> 90 degree roll
    pitch, roll = accel_to_pitch_roll(0.0, 1.0, 0.0)
    assert abs(abs(roll) - 90.0) < 1e-3


def test_euler_to_quaternion_identity_is_zero_tilt():
    qx, qy, qz, qw = euler_to_quaternion(0.0, 0.0, 0.0)
    assert abs(qx) < 1e-9 and abs(qy) < 1e-9 and abs(qz) < 1e-9
    assert abs(qw - 1.0) < 1e-9


def test_euler_to_quaternion_roundtrips_through_existing_fusion_node_extraction():
    # This is the compatibility contract with imu_slope_fusion_node.py: whatever
    # quaternion this driver publishes must decode back to the same pitch/roll
    # via the fusion node's own extraction functions.
    for roll_deg, pitch_deg in [(0.0, 0.0), (15.0, 0.0), (0.0, -20.0), (10.0, -12.0)]:
        qx, qy, qz, qw = euler_to_quaternion(roll_deg, pitch_deg, yaw_deg=0.0)
        assert abs(_quaternion_to_pitch(qx, qy, qz, qw) - pitch_deg) < 1e-3
        assert abs(_quaternion_to_roll(qx, qy, qz, qw) - roll_deg) < 1e-3


def test_euler_to_quaternion_is_unit_length():
    qx, qy, qz, qw = euler_to_quaternion(23.0, -37.0, 5.0)
    norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    assert abs(norm - 1.0) < 1e-9


# ── accel_magnitude_g -- the signal imu_slope_fusion_node needs to tell a real
# slope from an acceleration artifact. Until 2026-07-29 this driver read the
# accelerometer, converted it straight to an angle and threw the magnitude
# away, so nothing downstream could make that distinction.

def test_accel_magnitude_of_a_level_stationary_rover_is_one_g():
    assert abs(accel_magnitude_g(0.0, 0.0, 1.0) - 1.0) < 1e-9


def test_accel_magnitude_is_orientation_independent_for_a_parked_rover():
    # Tilted onto a 30 deg slope but not accelerating: the vector rotates,
    # the magnitude does not.
    ax = math.sin(math.radians(30.0))
    az = math.cos(math.radians(30.0))
    assert abs(accel_magnitude_g(ax, 0.0, az) - 1.0) < 1e-9


def test_accel_magnitude_exceeds_one_g_under_horizontal_acceleration():
    # 1 g of forward acceleration on flat ground reads 45 deg of apparent
    # tilt and sqrt(2) g of magnitude -- the artifact the gate must catch.
    assert abs(accel_magnitude_g(1.0, 0.0, 1.0) - math.sqrt(2.0)) < 1e-9


def test_accel_magnitude_is_below_one_g_in_partial_free_fall():
    assert accel_magnitude_g(0.0, 0.0, 0.8) < 1.0


def test_g_to_m_s2_is_standard_gravity():
    # sensor_msgs/Imu.linear_acceleration is specified in m/s^2, not g.
    assert abs(G_TO_M_S2 - 9.80665) < 1e-9


# --- decode_accel_gyro_burst: the 14-byte MPU-6050 register burst -----------
# Added 2026-08-19 with the ICM-20948 -> MPU-6050 driver rewrite. The burst
# layout is the one part of that rewrite that is easy to get subtly wrong
# (temperature sits between accel and gyro, so a naive 12-byte read would
# silently shift every gyro axis), hence direct coverage here.

def _burst(ax=0, ay=0, az=0, temp=0, gx=0, gy=0, gz=0):
    """Build a 14-byte ACCEL_XOUT_H burst from unsigned 16-bit register values."""
    out = []
    for value in (ax, ay, az, temp, gx, gy, gz):
        out += [(value >> 8) & 0xFF, value & 0xFF]
    return out


def test_decode_burst_all_zero_reads_as_stationary_and_weightless():
    assert decode_accel_gyro_burst(_burst()) == (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


def test_decode_burst_maps_each_axis_to_its_own_slot():
    # Distinct value per axis so a transposed or shifted decode cannot pass.
    result = decode_accel_gyro_burst(
        _burst(ax=16384, ay=8192, az=4096, temp=0x1234, gx=131, gy=262, gz=393)
    )
    ax_g, ay_g, az_g, gx_dps, gy_dps, gz_dps = result
    assert ax_g == 1.0
    assert ay_g == 0.5
    assert az_g == 0.25
    assert gx_dps == 1.0
    assert gy_dps == 2.0
    assert gz_dps == 3.0


def test_decode_burst_ignores_the_temperature_bytes_between_accel_and_gyro():
    # Temperature occupies data[6:8]. Changing it must not perturb any output;
    # if it did, the gyro axes would be reading shifted bytes.
    without = decode_accel_gyro_burst(_burst(az=16384, gz=131, temp=0x0000))
    with_temp = decode_accel_gyro_burst(_burst(az=16384, gz=131, temp=0xFFFF))
    assert without == with_temp


def test_decode_burst_handles_negative_twos_complement_values():
    # 49152 unsigned == -16384 signed == -1 g; 65405 unsigned == -131 == -1 dps
    ax_g, _, _, gx_dps, _, _ = decode_accel_gyro_burst(_burst(ax=49152, gx=65405))
    assert ax_g == -1.0
    assert gx_dps == -1.0


def test_decode_burst_level_stationary_rover_reads_one_g_on_z():
    ax_g, ay_g, az_g, _, _, _ = decode_accel_gyro_burst(_burst(az=16384))
    assert abs(accel_magnitude_g(ax_g, ay_g, az_g) - 1.0) < 1e-9
    pitch, roll = accel_to_pitch_roll(ax_g, ay_g, az_g)
    assert abs(pitch) < 1e-9
    assert abs(roll) < 1e-9


def test_decode_burst_rejects_a_short_read_rather_than_misdecoding_it():
    # A truncated I2C read must raise, not quietly return garbage that
    # downstream would treat as a real attitude measurement.
    with pytest.raises(ValueError):
        decode_accel_gyro_burst([0] * 12)
