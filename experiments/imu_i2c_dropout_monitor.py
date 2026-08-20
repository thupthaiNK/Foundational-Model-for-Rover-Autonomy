#!/usr/bin/env python3
"""
Purpose: Catch the ExoMy IMU (MPU-6050 breakout, bus 1, 0x68) in the act of
         dropping off I2C, instead of only ever seeing the aftermath. Every
         cycle it records which of the four known addresses answer, whether
         the IMU still returns an accepted WHO_AM_I, the CPU temperature, and
         the Pi's under-voltage/throttling flags. That is enough to separate
         the three hypothesis groups without touching a soldering iron:

           - 0x69 appears while 0x68 vanishes  -> the AD0 pin is floating
             and the chip simply moved address. Fixable in software.
           - all four addresses vanish together -> a bus-level fault after
             all, contradicting the 2026-07-28 observation. Re-investigate.
           - only 0x68 vanishes, 0x69 does not appear -> the chip is not
             powered, not clocked, or dead. Check the throttled/volts columns
             and the temperature at the moment of the drop.
           - 0x68 answers but WHO_AM_I is not 0x68/0x72 -> the chip is present
             and confused rather than absent, a different problem entirely.

         Note: this replaced an ICM-20948 (bus 1, 0x69) 2026-08-19; see
         project_imu_mpu6050_saga_resolved_20260819 memory. The physical
         board is now an MPU-6050-family chip whose WHO_AM_I register lives
         at 0x75, not 0x00 -- addresses and WHO_AM_I logic below were updated
         to match the driver in fm_imu_fusion/icm20948_driver_node.py.

         Runs on the Raspberry Pi HOST, not inside the exomy_ros container,
         because vcgencmd only exists on the host. Standard library only, so
         it needs no pip install. i2c-tools must be present (it already is).

Inputs:  --bus (default 1), --interval seconds (default 5.0),
         --duration minutes (default 20.0, 0 means run until Ctrl+C),
         --out CSV path (default ~/imu_dropout_log.csv)
Outputs: A CSV of one row per cycle, plus a line on stdout for every change
         of state so the console alone tells the story.
How to run:
    # laptop
    scp experiments/imu_i2c_dropout_monitor.py pi@172.20.10.13:~/
    # Pi HOST (not the container). Stop icm20948_driver_node first, or it
    # and this script will take turns owning the bus and confuse each other.
    python3 ~/imu_i2c_dropout_monitor.py --duration 20
Project: Onboard Visual Foundation Models for Mars Terrain Perception and Rover Navigation
"""

import argparse
import csv
import os
import re
import subprocess
import sys
import time

# The four addresses that matter on ExoMy's bus 1.
ADDR_IMU = 0x68          # MPU-6050 breakout, current board (AD0 low)
ADDR_IMU_ALT = 0x69      # where it would land if AD0 read high instead
ADDR_SERVO = 0x40        # PCA9685 servo HAT
ADDR_SERVO_ALLCALL = 0x70

WATCHED = (ADDR_SERVO, ADDR_IMU_ALT, ADDR_IMU, ADDR_SERVO_ALLCALL)
WHO_AM_I_REG = 0x75
WHO_AM_I_ACCEPTED = frozenset({0x68, 0x72})

CSV_FIELDS = [
    "iso_time",
    "elapsed_s",
    "uptime_s",
    "present_0x40",
    "present_0x68",
    "present_0x69",
    "present_0x70",
    "who_am_i",
    "cpu_temp_c",
    "core_volt_v",
    "throttled",
    "note",
]


def run(cmd, timeout=10.0):
    """Run a command, returning stdout or None if it fails for any reason."""
    try:
        completed = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.decode("utf-8", errors="replace")


def parse_i2cdetect(text):
    """Extract the set of answering addresses from i2cdetect's table.

    The table has a leading '<row>:' column that must not be read as data,
    and prints '--' for silence and 'UU' for an address claimed by a kernel
    driver. UU counts as present: something is there, it is just busy.

    Cells are read by character position, not by splitting on whitespace.
    When a scan range starts partway through a row, i2cdetect pads the
    addresses below it with blanks, and splitting would shift every
    remaining cell left and report the wrong addresses.
    """
    found = set()
    if not text:
        return found
    for raw_line in text.splitlines():
        match = re.match(r"^([0-9a-f]{2}):", raw_line, re.IGNORECASE)
        if not match:
            continue
        row_base = int(match.group(1), 16)
        body = raw_line[3:]
        for column in range(16):
            cell = body[1 + 3 * column : 3 + 3 * column]
            if len(cell) < 2 or cell in ("--", "  "):
                continue
            if cell.upper() == "UU" or re.fullmatch(r"[0-9a-f]{2}", cell, re.IGNORECASE):
                found.add(row_base + column)
    return found


def scan_bus(bus):
    """Scan only 0x40-0x70 rather than the whole bus, to poke as few
    unrelated chips as possible. -r uses a read byte, which is the gentler
    probe of the two i2cdetect offers."""
    text = run(["i2cdetect", "-y", "-r", str(bus), hex(ADDR_SERVO), hex(ADDR_SERVO_ALLCALL)])
    if text is None:
        return None
    return parse_i2cdetect(text)


def read_who_am_i(bus, addr):
    """Read WHO_AM_I. Returns an int, or None if the read failed."""
    text = run(["i2cget", "-y", str(bus), hex(addr), hex(WHO_AM_I_REG)])
    if text is None:
        return None
    try:
        return int(text.strip(), 16)
    except ValueError:
        return None


def cpu_temp_c():
    try:
        with open("/sys/class/thermal/thermal_zone0/temp", "r") as handle:
            return round(int(handle.read().strip()) / 1000.0, 1)
    except (OSError, ValueError):
        return ""


def core_volt_v():
    text = run(["vcgencmd", "measure_volts", "core"])
    if not text:
        return ""
    match = re.search(r"([0-9.]+)V", text)
    return match.group(1) if match else ""


def throttled_flags():
    """vcgencmd get_throttled. Bit 0 set means under-voltage right now, bit
    16 means it happened at some point since boot. Anything non-zero is worth
    seeing next to a dropout."""
    text = run(["vcgencmd", "get_throttled"])
    if not text:
        return ""
    match = re.search(r"throttled=(0x[0-9a-fA-F]+)", text)
    return match.group(1) if match else text.strip()


def uptime_s():
    try:
        with open("/proc/uptime", "r") as handle:
            return round(float(handle.read().split()[0]), 1)
    except (OSError, ValueError, IndexError):
        return ""


def describe(present, who):
    """One-line human summary of a single sample."""
    if present is None:
        return "i2cdetect FAILED (is i2c-tools installed, and is the bus enabled?)"
    marks = []
    for addr in WATCHED:
        marks.append(f"{addr:#04x}{'+' if addr in present else '-'}")
    line = " ".join(marks)
    if ADDR_IMU in present:
        if who is None:
            line += "  WHO_AM_I unreadable"
        elif who not in WHO_AM_I_ACCEPTED:
            accepted = "/".join(f"{v:#04x}" for v in sorted(WHO_AM_I_ACCEPTED))
            line += f"  WHO_AM_I {who:#04x} NOT IN {{{accepted}}}"
    return line


def verdict(present, who):
    """The interpretation the whole script exists to produce."""
    if present is None:
        return "scan_failed"
    imu = ADDR_IMU in present
    alt = ADDR_IMU_ALT in present
    servo = ADDR_SERVO in present
    if imu and who in WHO_AM_I_ACCEPTED:
        return "healthy"
    if imu and who is None:
        return "imu_acks_address_but_read_failed"
    if imu:
        return "imu_wrong_who_am_i"
    if alt:
        return "IMU MOVED TO 0x69 -- address pin, not a broken wire"
    if not servo:
        return "WHOLE BUS DOWN -- servo HAT gone too, this is not IMU-only"
    return "IMU GONE, bus and servo HAT fine -- power, clock, or dead chip"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("Inputs:")[0])
    parser.add_argument("--bus", type=int, default=1)
    parser.add_argument("--interval", type=float, default=5.0)
    parser.add_argument(
        "--duration",
        type=float,
        default=20.0,
        help="minutes; 0 runs until Ctrl+C",
    )
    parser.add_argument(
        "--out",
        default=os.path.expanduser("~/imu_dropout_log.csv"),
    )
    args = parser.parse_args(argv)

    if run(["which", "i2cdetect"]) is None:
        print("i2cdetect not found. Install i2c-tools, or run this on the Pi host.")
        return 2

    deadline = None if args.duration <= 0 else time.time() + args.duration * 60.0
    started = time.time()
    last_verdict = None
    samples = 0
    drops = 0

    print(f"Logging every {args.interval:g}s to {args.out}")
    print("Stop icm20948_driver_node first, or it will fight this script for the bus.")
    print("Columns: 0x40 servo HAT, 0x68 IMU, 0x69 IMU alt address, 0x70 all-call.")
    print("-" * 72)

    is_new_file = not os.path.exists(args.out)
    try:
        with open(args.out, "a", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
            if is_new_file:
                writer.writeheader()

            while deadline is None or time.time() < deadline:
                present = scan_bus(args.bus)
                who = (
                    read_who_am_i(args.bus, ADDR_IMU)
                    if present and ADDR_IMU in present
                    else None
                )
                state = verdict(present, who)
                now = time.time()
                stamp = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(now))

                writer.writerow(
                    {
                        "iso_time": stamp,
                        "elapsed_s": round(now - started, 1),
                        "uptime_s": uptime_s(),
                        "present_0x40": int(bool(present) and ADDR_SERVO in present),
                        "present_0x68": int(bool(present) and ADDR_IMU in present),
                        "present_0x69": int(bool(present) and ADDR_IMU_ALT in present),
                        "present_0x70": int(
                            bool(present) and ADDR_SERVO_ALLCALL in present
                        ),
                        "who_am_i": "" if who is None else f"{who:#04x}",
                        "cpu_temp_c": cpu_temp_c(),
                        "core_volt_v": core_volt_v(),
                        "throttled": throttled_flags(),
                        "note": state,
                    }
                )
                handle.flush()
                samples += 1

                if state != last_verdict:
                    if last_verdict is not None and state != "healthy":
                        drops += 1
                    print(f"{stamp}  {describe(present, who)}")
                    print(f"    -> {state}   (t+{round(now - started)}s, "
                          f"{cpu_temp_c()}C, throttled={throttled_flags()})")
                    last_verdict = state

                time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nstopped by user")

    print("-" * 72)
    print(f"{samples} samples, {drops} state changes away from healthy.")
    print(f"CSV: {args.out}")
    print("If 0x68 never dropped, run it longer or drive the rover while it logs.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
