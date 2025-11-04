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

    def odometry_callback(self, msg):
        self.current_speed = msg.twist.twist.linear.x

    def laserscan_callback(self, msg):
        if self.current_speed == 0:
            return

        ranges = np.array(msg.ranges)
        angles = np.linspace(msg.angle_min, msg.angle_max, len(ranges))
    
        min_ttc = np.inf
        for i in range(len(ranges)):
            angle = angles[i]
            
            #ran into some issues on turns as the sides of the car had low ttc which were false pos.
            forward_angle_limit = np.deg2rad(45)
            if abs(angle) > forward_angle_limit:
                continue
  

            if not np.isinf(ranges[i]) and not np.isnan(ranges[i]):
                distance = ranges[i]
                range_rate = self.current_speed * np.cos(angle - msg.angle_increment * i)

                if range_rate < 0:
                    ttc = distance / max(-range_rate, 1e-12)
                    if ttc < min_ttc:
                        min_ttc = ttc

        if min_ttc < 1.0:
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
