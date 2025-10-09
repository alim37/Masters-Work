#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np

from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan

class SafetyNode(Node):
    def __init__(self):
        super().__init__("aeb_safety")
        self.odometry_subscriber = self.create_subscription(Odometry, "/ego_racecar/odom", self.odometry_callback, 10)
        self.scan_subscriber = self.create_subscription(LaserScan, "/scan", self.laserscan_callback, 10)
        self.drive_publisher = self.create_publisher(AckermannDriveStamped, "/drive", 10)
        self.current_speed = 0.0
        self.ttc_thresh = 1.25

    def odometry_callback(self, msg):
        self.current_speed = msg.twist.twist.linear.x

    def laserscan_callback(self, msg):
        if self.current_speed == 0.0:
            return

        ranges = np.asarray(msg.ranges)
        angles = np.linspace(msg.angle_min, msg.angle_max, ranges.size)
        rel_rate = self.current_speed * np.cos(angles)

        denom = -rel_rate
        denom = np.where(rel_rate < 0.0, denom, np.inf)

        ttc = ranges / denom 
        min_ttc = np.min(ttc)

        if min_ttc < self.ttc_thresh:
            self.get_logger().info("Using emergency brake!")
            self.emergency_brake()

    def emergency_brake(self):
        brake_msg = AckermannDriveStamped()
        brake_msg.drive.speed = 0.0
        brake_msg.drive.acceleration = -1.0
        self.drive_publisher.publish(brake_msg)

def main(args=None):
    rclpy.init(args=args)
    node = SafetyNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == "__main__":
    main()
