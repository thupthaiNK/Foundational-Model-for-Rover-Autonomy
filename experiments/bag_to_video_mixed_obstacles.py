"""
Purpose: Extract onboard-camera and chase-cam (third-person) video, plus a
         terrain_classification/IMU/LiDAR text summary, from the mixed-
         obstacles sensor-validation rosbag (2026-08-19 sanity check) so
         the author can visually review the run.
Inputs:  A reindexed ros2 bag directory (sensor_msgs/Image topics
         /exomy/camera/image_raw and /exomy/chase_cam/image_raw, plus
         /terrain_classification, /exomy/imu_raw, /scan).
Outputs: <out_dir>/onboard.mp4, <out_dir>/thirdperson.mp4,
         <out_dir>/summary.txt
How to run:
    source /opt/ros/humble/setup.bash && source ros2_ws/install/setup.bash
    python3 experiments/bag_to_video_mixed_obstacles.py \
        bags/mixed_obstacles_sensor_validation_20260819_203031 \
        docs/figures/mixed_obstacles_sensor_validation
Project: Onboard Visual Foundation Models for Mars Terrain Perception and Rover Navigation
"""
import sys
import os
import numpy as np
import cv2
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import Image
from std_msgs.msg import String


ENCODING_CHANNELS = {"rgb8": 3, "bgr8": 3, "mono8": 1, "rgba8": 4}


def image_to_bgr(msg: Image):
    ch = ENCODING_CHANNELS.get(msg.encoding)
    if ch is None:
        return None
    arr = np.frombuffer(bytes(msg.data), dtype=np.uint8)
    try:
        arr = arr.reshape(msg.height, msg.width, ch)
    except ValueError:
        return None
    if ch == 1:
        return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    if msg.encoding == "rgb8":
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    if msg.encoding == "rgba8":
        return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
    return arr  # already bgr8


def extract(bag_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    reader = SequentialReader()
    reader.open(
        StorageOptions(uri=bag_dir, storage_id="sqlite3"),
        ConverterOptions(input_serialization_format="cdr",
                          output_serialization_format="cdr"),
    )
    topic_types = {t.name: t.type for t in reader.get_all_topics_and_types()}

    frames = {"/exomy/camera/image_raw": [], "/exomy/chase_cam/image_raw": []}
    terrain_log = []
    t0 = None

    while reader.has_next():
        topic, data, t_ns = reader.read_next()
        if t0 is None:
            t0 = t_ns
        t_s = (t_ns - t0) / 1e9

        if topic in frames and topic_types.get(topic) == "sensor_msgs/msg/Image":
            msg = deserialize_message(data, Image)
            bgr = image_to_bgr(msg)
            if bgr is not None:
                frames[topic].append((t_s, bgr))
        elif topic == "/terrain_classification":
            msg = deserialize_message(data, String)
            terrain_log.append(f"[{t_s:7.2f}s] {msg.data}")

    def write_video(name, entries, target_fps=10.0):
        """Resample onto a uniform real-time grid at target_fps, instead of
        writing captured frames back-to-back at one averaged constant fps
        (2026-08-19 fix -- author caught this: frames arrive at wildly
        uneven real intervals under CPU contention, sometimes 5 frames
        within 0.05s, sometimes a 0.5s gap; writing them at one averaged
        fps compresses/stretches real elapsed time non-uniformly, which
        looks like the rover stutters or walks back and forth even though
        odometry ground truth showed smooth, monotonic motion the whole
        time -- confirmed by hand before this fix, not assumed). Each
        output frame at time k/target_fps holds whichever captured frame
        was most recently available at that instant, so 1 second of video
        always corresponds to 1 second of real sim time."""
        if not entries:
            print(f"  {name}: no frames found, skipping")
            return
        h, w = entries[0][1].shape[:2]
        t_start, t_end = entries[0][0], entries[-1][0]
        duration = max(t_end - t_start, 1.0 / target_fps)
        n_out = max(1, int(duration * target_fps))
        path = os.path.join(out_dir, f"{name}.mp4")
        vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), target_fps, (w, h))
        idx = 0
        held_frame = entries[0][1]
        for k in range(n_out):
            t_grid = t_start + k / target_fps
            while idx + 1 < len(entries) and entries[idx + 1][0] <= t_grid:
                idx += 1
                held_frame = entries[idx][1]
            vw.write(held_frame)
        vw.release()
        print(f"  {name}: {len(entries)} captured frames -> {n_out} output "
              f"frames at fixed {target_fps:.0f} fps -> {path}")

    print(f"Onboard frames captured: {len(frames['/exomy/camera/image_raw'])}")
    write_video("onboard", frames["/exomy/camera/image_raw"])
    print(f"Chase-cam frames captured: {len(frames['/exomy/chase_cam/image_raw'])}")
    write_video("thirdperson", frames["/exomy/chase_cam/image_raw"])

    summary_path = os.path.join(out_dir, "summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"Bag: {bag_dir}\n")
        f.write(f"Terrain classification messages: {len(terrain_log)}\n\n")
        f.write("\n".join(terrain_log))
    print(f"Terrain classification log -> {summary_path}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: bag_to_video_mixed_obstacles.py <bag_dir> <out_dir>")
        sys.exit(1)
    extract(sys.argv[1], sys.argv[2])
