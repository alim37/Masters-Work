#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from ackermann_msgs.msg import AckermannDriveStamped
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
import math
import os
import csv

class WallFollowerNode(Node):      

    def __init__(self):
        super().__init__("wall_follower") 
        
        self.scan_sub = self.create_subscription(LaserScan, "/scan", self.scan_callback, 10)
        self.drive_pub = self.create_publisher(AckermannDriveStamped, "/drive", 10)
        self.odom_sub = self.create_subscription(Odometry, '/ego_racecar/odom', self.odom_callback, 10)

        self.kp = 3.0
        self.ki = 0.001
        self.kd = 0.10

        self.prev_error = 0.0
        self.integral = 0.0
        
        self.velocities = []
        self.times = []
        self.start_time = None
        self.lap_end_time = None
        self.lap_completed = False
        
    def scan_callback(self, data):
        error = self.calculate_error(data)
        control = self.pid_control(error)
        self.publish_drive_msg(control)
        
    def get_range(self, data, angle):
        angle = math.radians(angle)

        index = int((angle - data.angle_min) / data.angle_increment)

        if 0 <= index < len(data.ranges):
            return data.ranges[index]
        else:
            return float('inf')
        
    def calculate_error(self, data):

        a = self.get_range(data,45)
        b = self.get_range(data,90)
        alpha = math.atan2(a*math.cos(math.radians(45))-b, a*math.sin(math.radians(45)))
        D_t = b*math.cos(alpha)
        desired_distance = 1.0
        D_t1 = D_t + 1.0*math.sin(alpha)

        error = desired_distance - D_t1
        return error

    def pid_control(self, error):
        self.integral += error
        derivative = error - self.prev_error

        control = -(self.kp * error + self.ki * self.integral + self.kd * derivative)
        
        self.prev_error = error

        return control

    def publish_drive_msg(self, control):
        drive_msg = AckermannDriveStamped()
        drive_msg.drive.steering_angle = control
        
        if abs(control) <= 10.0 * (math.pi / 180.0):
            drive_msg.drive.speed = 1.5
        elif 10.0 * (math.pi / 180.0) < abs(control) <= 20.0 * (math.pi / 180.0):
            drive_msg.drive.speed = 1.0
        else:
            drive_msg.drive.speed = 1.5

        self.drive_pub.publish(drive_msg)
    
    def odom_callback(self, msg):
        if self.start_time is None:
            self.start_time = self.get_clock().now()

        current_time = (self.get_clock().now() - self.start_time).nanoseconds / 1e9
        v_x = msg.twist.twist.linear.x
        v_y = msg.twist.twist.linear.y
        total_velocity = math.sqrt(v_x ** 2 + v_y ** 2)

        self.velocities.append(total_velocity)
        self.times.append(current_time)
        
        current_x = msg.pose.pose.position.x
        current_y = msg.pose.pose.position.y
        
        if current_x <= 1 and current_y <= 1 and not self.lap_completed:
            lap_time = (self.get_clock().now() - self.start_time).nanoseconds / 1e9
            self.lap_completed = True
            self.get_logger().info(f'Lap completed in {lap_time:.2f} seconds')
        
        if current_x and current_y <= 1:
            self.lap_completed = False
            
    def save_data(self):
        save_dir = os.path.expanduser('~/data_pid')
        os.makedirs(save_dir, exist_ok=True)
        
        # Save velocities
        velocities_path = os.path.join(save_dir, 'velocities_pid.csv')
        with open(velocities_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Velocity'])  # Header
            for velocity in self.velocities:
                writer.writerow([velocity])
        
        # Save times
        times_path = os.path.join(save_dir, 'times_pid.csv')
        with open(times_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Time'])  # Header
            for time in self.times:
                writer.writerow([time])

        self.get_logger().info(f'Velocities saved to: {velocities_path}')
        self.get_logger().info(f'Times saved to: {times_path}')


def main(args=None):
    rclpy.init(args=args)   
    node = WallFollowerNode()     
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("Node stopped cleanly.")
    finally:
        node.save_data()
        rclpy.shutdown()       


if __name__ == "__main__":
    main()
