"""
Purpose: Extract IMU and LiDAR time series from a mixed-obstacles sensor-
         validation rosbag, alongside /exomy/odom as Gazebo's ground truth,
         so the author can check IMU-reported motion and LiDAR-reported
         obstacle range against what the rover actually did/faced.
         Answers "can this be added" from the 2026-08-19 review: yes, the
         data was already being recorded (bag_record topic list in
         mixed_obstacles_sensor_validation.launch.py includes /exomy/
         imu_raw and /scan) -- it just hadn't been extracted from the bag
         yet, same as terrain_classification was in
         bag_to_video_mixed_obstacles.py's summary.txt.
Inputs:  A reindexed ros2 bag directory with /exomy/imu_raw (sensor_msgs/
         Imu), /scan (sensor_msgs/LaserScan), /exomy/odom (nav_msgs/
         Odometry).
Outputs: <out_dir>/imu_lidar_summary.csv — one row per IMU sample, with
         the nearest-in-time odom (ground truth yaw/position) and LiDAR
         min-range joined in.
         <out_dir>/imu_lidar_summary.txt — human-readable stats: IMU yaw-
         rate vs odom-derived yaw-rate correlation, LiDAR min-range
         distribution, timestamps where LiDAR range < stop_distance_m
         (0.4m, the lidar_proximity_guard_node threshold) with the odom
         position at that instant.
How to run:
    source /opt/ros/humble/setup.bash && source ros2_ws/install/setup.bash
    python3 experiments/bag_imu_lidar_summary.py \
        bags/mixed_obstacles_sensor_validation_20260819_203031 \
        docs/figures/mixed_obstacles_sensor_validation
Project: Onboard Visual Foundation Models for Mars Terrain Perception and Rover Navigation
"""
import sys
import os
import csv
import math
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import Imu, LaserScan
from nav_msgs.msg import Odometry


def yaw_from_quaternion(q):
    # standard yaw-from-quaternion, z-up
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def load_bag(bag_dir):
    reader = SequentialReader()
    reader.open(
        StorageOptions(uri=bag_dir, storage_id="sqlite3"),
        ConverterOptions(input_serialization_format="cdr",
                          output_serialization_format="cdr"),
    )
    topic_types = {t.name: t.type for t in reader.get_all_topics_and_types()}

    imu_samples, scan_samples, odom_samples = [], [], []
    t0 = None
    while reader.has_next():
        topic, data, t_ns = reader.read_next()
        if t0 is None:
            t0 = t_ns
        t_s = (t_ns - t0) / 1e9

        if topic == "/exomy/imu_raw" and topic_types.get(topic) == "sensor_msgs/msg/Imu":
            msg = deserialize_message(data, Imu)
            imu_samples.append((t_s, msg.angular_velocity.z,
                                 msg.linear_acceleration.x,
                                 msg.linear_acceleration.y,
                                 msg.linear_acceleration.z))
        elif topic == "/scan" and topic_types.get(topic) == "sensor_msgs/msg/LaserScan":
            msg = deserialize_message(data, LaserScan)
            valid = [r for r in msg.ranges if msg.range_min < r < msg.range_max]
            min_r = min(valid) if valid else float("nan")
            scan_samples.append((t_s, min_r))
        elif topic == "/exomy/odom" and topic_types.get(topic) == "nav_msgs/msg/Odometry":
            msg = deserialize_message(data, Odometry)
            p = msg.pose.pose.position
            yaw = yaw_from_quaternion(msg.pose.pose.orientation)
            odom_samples.append((t_s, p.x, p.y, yaw, msg.twist.twist.angular.z))

    return imu_samples, scan_samples, odom_samples


def nearest(sorted_samples, t, key=lambda s: s[0]):
    if not sorted_samples:
        return None
    lo, hi = 0, len(sorted_samples) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if key(sorted_samples[mid]) < t:
            lo = mid + 1
        else:
            hi = mid
    return sorted_samples[lo]


def analyze(bag_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    imu, scan, odom = load_bag(bag_dir)
    print(f"IMU samples: {len(imu)}  scan samples: {len(scan)}  odom samples: {len(odom)}")

    csv_path = os.path.join(out_dir, "imu_lidar_summary.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_s", "imu_yaw_rate", "imu_accel_x", "imu_accel_y", "imu_accel_z",
                    "odom_x", "odom_y", "odom_yaw", "odom_yaw_rate", "lidar_min_range_m"])
        for t, yr, ax, ay, az in imu:
            o = nearest(odom, t)
            s = nearest(scan, t)
            w.writerow([
                round(t, 3), round(yr, 4), round(ax, 3), round(ay, 3), round(az, 3),
                round(o[1], 3) if o else "", round(o[2], 3) if o else "",
                round(o[3], 4) if o else "", round(o[4], 4) if o else "",
                round(s[1], 3) if s and s[1] == s[1] else "",  # NaN check
            ])
    print(f"CSV -> {csv_path}")

    # IMU vs odom yaw-rate agreement (ground truth is odom, since this is sim)
    diffs = []
    for t, yr, *_ in imu:
        o = nearest(odom, t)
        if o:
            diffs.append(abs(yr - o[4]))
    mean_diff = sum(diffs) / len(diffs) if diffs else float("nan")

    stop_distance_m = 0.4
    close_calls = []
    for t, min_r in scan:
        if min_r == min_r and min_r < stop_distance_m:  # not NaN
            o = nearest(odom, t)
            close_calls.append((t, min_r, o[1] if o else None, o[2] if o else None))

    txt_path = os.path.join(out_dir, "imu_lidar_summary.txt")
    with open(txt_path, "w") as f:
        f.write(f"Bag: {bag_dir}\n")
        f.write(f"IMU samples: {len(imu)} | Scan samples: {len(scan)} | Odom samples: {len(odom)}\n\n")
        f.write("== IMU yaw-rate vs odom (ground truth) yaw-rate ==\n")
        f.write(f"Mean |imu_yaw_rate - odom_yaw_rate|: {mean_diff:.4f} rad/s "
                f"(over {len(diffs)} paired samples)\n\n")
        f.write(f"== LiDAR: instants where min range < {stop_distance_m}m stop threshold ==\n")
        f.write(f"Count: {len(close_calls)}\n")
        for t, min_r, ox, oy in close_calls[:40]:
            f.write(f"  [{t:7.2f}s] min_range={min_r:.3f}m  rover_pos=({ox:.2f},{oy:.2f})\n")
        if len(close_calls) > 40:
            f.write(f"  ... and {len(close_calls) - 40} more (see CSV)\n")
    print(f"Summary -> {txt_path}")
    print(f"Mean IMU/odom yaw-rate disagreement: {mean_diff:.4f} rad/s")
    print(f"LiDAR close-range instants: {len(close_calls)}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: bag_imu_lidar_summary.py <bag_dir> <out_dir>")
        sys.exit(1)
    analyze(sys.argv[1], sys.argv[2])
