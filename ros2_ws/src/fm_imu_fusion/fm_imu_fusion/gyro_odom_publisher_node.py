#!/usr/bin/env python3
"""
Purpose: Integrates the IMU gyroscope (/exomy/imu_raw angular_velocity.z)
         over time and publishes it as a TF chain (gyro_odom -> base_link)
         with ZERO translation and the integrated yaw as the only rotation
         component -- a gyro-only motion prior for slam_toolbox on real
         ExoMy hardware, which has no wheel encoders and therefore no other
         source of odometry at all.
         ALSO publishes the identical pose as a nav_msgs/Odometry message on
         odom_topic (default /exomy/odom) -- added 2026-07-26 after a live
         hardware trial found this was the only source of that topic, and
         nothing was ever publishing it: this node previously broadcast the
         TF only, so every node expecting /exomy/odom directly (e.g.
         reactive_explorer_node.py's closed-loop turning) silently never
         received a message, on every prior real-hardware run. Same
         zero-translation / integrated-yaw-only content as the TF, computed
         once per callback and reused for both.
         This is a package-promoted copy of experiments/gyro_odom_publisher.py
         (built 2026-07-17 for the L4 Gazebo feasibility investigation),
         moved into fm_imu_fusion so it can be launched as a normal ROS2
         node in real_hardware_deployment.launch.py. The logic is
         unchanged; /exomy/imu_raw is published by icm20948_driver_node.py
         in this package on real hardware (same topic name and message
         type the Gazebo-simulated IMU sensor used during that
         investigation), so this node works identically regardless of
         which one is providing the data.
         IMPORTANT CAVEAT, not resolved as of 2026-07-17: this exact
         gyro-prior architecture was tested three times in Gazebo
         (simulation/launch/slam_test_gyro_prior.launch.py) against
         slam_toolbox and never once produced a pose estimate --
         slam_toolbox's tf2 message filter dropped scans due to a
         persistent "queue is full" condition, the same symptom seen with
         two other odom-frame content strategies tried first. The root
         cause was not isolated (candidates: a slam_toolbox message-filter
         configuration issue, or a Gazebo use_sim_time/discrete-stepping
         timing artefact that may not reproduce on real hardware's
         continuous real-time clock -- genuinely unknown either way). This
         node is included in the real-hardware launch as PREPARED, UNTESTED
         staging work -- wiring it in now costs nothing (opt-in, off by
         default) and means it is ready to test the moment real hardware
         is available, but there is no evidence yet that odom-free SLAM
         will actually work with it. Do not present this as a working
         capability until it has actually been verified against real
         scan/IMU data.
Inputs:  /exomy/imu_raw (sensor_msgs/Imu) -- icm20948_driver_node.py on
         real hardware, or Gazebo's simulated IMU sensor in simulation.
Outputs: TF: gyro_odom -> base_link (translation always 0,0,0; rotation =
         integrated yaw only)
         odom_topic (nav_msgs/Odometry, default /exomy/odom): identical
         pose content to the TF above
How to run:
    source /opt/ros/humble/setup.bash
    cd ros2_ws && colcon build --packages-select fm_imu_fusion
    source install/setup.bash
    ros2 run fm_imu_fusion gyro_odom_publisher_node.py
Project: Onboard Visual Foundation Models for Mars Terrain Perception and Rover Navigation
"""

import math

import rclpy
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import Imu
from tf2_ros import TransformBroadcaster


def yaw_to_quaternion(yaw: float):
    """Quaternion (x, y, z, w) for a pure yaw (Z-axis) rotation.

    Shared by the TF broadcast and the Odometry publish below so the two
    can never disagree -- both are built from this single computation.
    """
    return (0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))


class GyroOdomPublisherNode(Node):

    def __init__(self):
        super().__init__("gyro_odom_publisher")
        # use_sim_time is passed via --ros-args -p on the command line when
        # run in Gazebo (see slam_test_gyro_prior.launch.py) and rclpy
        # pre-declares it from that automatically -- declaring it again
        # here would raise ParameterAlreadyDeclaredException (hit this
        # exact crash once during the Gazebo investigation). On real
        # hardware no --ros-args -p use_sim_time is passed, so this
        # parameter is simply never set, which is correct (real hardware
        # runs on wall-clock time, not simulated time).

        self.declare_parameter("odom_topic", "/exomy/odom")
        odom_topic = self.get_parameter("odom_topic").value

        self._yaw = 0.0
        self._last_time = None
        self._msg_count = 0

        self._br = TransformBroadcaster(self)
        self._odom_pub = self.create_publisher(Odometry, odom_topic, 10)
        self.create_subscription(Imu, "/exomy/imu_raw", self._imu_cb, 10)

        self.get_logger().info(
            "gyro_odom_publisher ready -- integrating /exomy/imu_raw "
            "angular_velocity.z into a gyro_odom->base_link TF and a "
            f"matching {odom_topic} Odometry message "
            "(translation always zero, rotation = integrated yaw only). "
            "UNTESTED against real SLAM data -- see module docstring."
        )

    def _imu_cb(self, msg: Imu) -> None:
        now = self.get_clock().now()
        if now.nanoseconds == 0:
            # Guards against a t=0 TF poisoning tf2's buffer for this frame
            # (found in the Gazebo use_sim_time investigation, §4.8.23
            # follow-up -- see experiments/gyro_odom_publisher.py's matching
            # comment). Real hardware runs on wall-clock time, which is
            # never exactly zero once rclpy is running, so this branch is
            # inert there -- kept for consistency between the two copies of
            # this logic rather than because real hardware needs it.
            return
        if self._last_time is not None:
            dt = (now - self._last_time).nanoseconds / 1e9
            if dt > 0.0:
                self._yaw += msg.angular_velocity.z * dt
        self._last_time = now
        self._msg_count += 1

        qx, qy, qz, qw = yaw_to_quaternion(self._yaw)

        t = TransformStamped()
        t.header.stamp = now.to_msg()
        t.header.frame_id = "gyro_odom"
        t.child_frame_id = "base_link"
        t.transform.translation.x = 0.0
        t.transform.translation.y = 0.0
        t.transform.translation.z = 0.0
        t.transform.rotation.x = qx
        t.transform.rotation.y = qy
        t.transform.rotation.z = qz
        t.transform.rotation.w = qw
        self._br.sendTransform(t)

        odom = Odometry()
        odom.header.stamp = now.to_msg()
        odom.header.frame_id = "gyro_odom"
        odom.child_frame_id = "base_link"
        odom.pose.pose.position.x = 0.0
        odom.pose.pose.position.y = 0.0
        odom.pose.pose.position.z = 0.0
        odom.pose.pose.orientation.x = qx
        odom.pose.pose.orientation.y = qy
        odom.pose.pose.orientation.z = qz
        odom.pose.pose.orientation.w = qw
        odom.twist.twist.angular.z = msg.angular_velocity.z
        self._odom_pub.publish(odom)

        if self._msg_count % 200 == 0:
            self.get_logger().info(
                f"[{self._msg_count}] integrated yaw = {math.degrees(self._yaw):.1f} deg"
            )


def main(args=None):
    rclpy.init(args=args)
    node = GyroOdomPublisherNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
