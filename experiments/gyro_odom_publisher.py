"""
Purpose: L4 Phase A follow-up -- integrates the simulated Gazebo IMU's
         gyroscope (/exomy/imu_raw angular_velocity.z) over time and
         publishes it as a TF chain (gyro_odom -> base_link) with ZERO
         translation and the integrated yaw as the only rotation component.
         This is a gyro-only motion prior for slam_toolbox: both odom-free
         attempts in Phase A (odom_frame==base_frame, and a fully-static
         identity transform) gave slam_toolbox literally zero motion
         information and neither produced a single pose estimate. This
         tests whether giving it correct ROTATION alone -- even with no
         translation information at all, since a gyro physically cannot
         measure translation -- is enough to let scan-matching take over
         successfully, rather than needing a translation prior too.
         /exomy/imu_raw here is the SAME topic name and message type
         (sensor_msgs/Imu) that icm20948_driver_node.py (ros2_ws/src/
         fm_imu_fusion/) publishes on real hardware -- unlike that real
         driver, which converts raw I2C register counts (degrees/second)
         to rad/s, Gazebo's libgazebo_ros_imu_sensor.so plugin already
         publishes angular_velocity in rad/s directly (standard ROS units),
         so no unit conversion is needed here.
Inputs:  /exomy/imu_raw (sensor_msgs/Imu) -- Gazebo's simulated IMU sensor
         (simulation/urdf/exomy.urdf.xacro, libgazebo_ros_imu_sensor.so)
Outputs: TF: gyro_odom -> base_link (translation always 0,0,0; rotation =
         integrated yaw only)
How to run:
    Started automatically by simulation/launch/slam_test_gyro_prior.launch.py
    -- not meant to be run standalone.
Project: Onboard Visual Foundation Models for Mars Terrain Perception and Rover Navigation
"""

import math

import rclpy
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from sensor_msgs.msg import Imu
from tf2_ros import TransformBroadcaster


class GyroOdomPublisher(Node):

    def __init__(self):
        super().__init__("gyro_odom_publisher")
        # use_sim_time is passed via --ros-args -p on the command line (see
        # the launch file) and rclpy pre-declares it from that automatically
        # -- declaring it again here would raise
        # ParameterAlreadyDeclaredException (hit this exact crash once).

        self._yaw = 0.0
        self._last_time = None
        self._msg_count = 0

        self._br = TransformBroadcaster(self)
        self.create_subscription(Imu, "/exomy/imu_raw", self._imu_cb, 10)

        self.get_logger().info(
            "gyro_odom_publisher ready -- integrating /exomy/imu_raw "
            "angular_velocity.z into a gyro_odom->base_link TF "
            "(translation always zero, rotation = integrated yaw only)"
        )

    def _imu_cb(self, msg: Imu) -> None:
        now = self.get_clock().now()
        if now.nanoseconds == 0:
            # use_sim_time is set but no /clock tick has been received yet --
            # self.get_clock().now() returns exactly zero in that window.
            # Publishing a TF stamped at t=0 here causes tf2 to reject every
            # later, correctly-timestamped transform for this frame as
            # "TF_OLD_DATA ignoring data from the past" -- found via direct
            # log inspection (buffer_core.cpp), not assumed. Skip until the
            # clock is real.
            return
        if self._last_time is not None:
            dt = (now - self._last_time).nanoseconds / 1e9
            if dt > 0.0:
                self._yaw += msg.angular_velocity.z * dt
        self._last_time = now
        self._msg_count += 1

        t = TransformStamped()
        t.header.stamp = now.to_msg()
        t.header.frame_id = "gyro_odom"
        t.child_frame_id = "base_link"
        t.transform.translation.x = 0.0
        t.transform.translation.y = 0.0
        t.transform.translation.z = 0.0
        t.transform.rotation.x = 0.0
        t.transform.rotation.y = 0.0
        t.transform.rotation.z = math.sin(self._yaw / 2.0)
        t.transform.rotation.w = math.cos(self._yaw / 2.0)
        self._br.sendTransform(t)

        if self._msg_count % 200 == 0:
            self.get_logger().info(
                f"[{self._msg_count}] integrated yaw = {math.degrees(self._yaw):.1f} deg"
            )


def main(args=None):
    rclpy.init(args=args)
    node = GyroOdomPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
