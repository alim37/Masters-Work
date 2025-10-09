#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from ackermann_msgs.msg import AckermannDriveStamped
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
import numpy as np
import math
import os
import csv

class GapFollowNode(Node):      

    def __init__(self):
        super().__init__("gap_follow")
        
        self.scan_sub = self.create_subscription(LaserScan, "/scan", self.scan_callback, 10)
        self.drive_pub = self.create_publisher(AckermannDriveStamped, "/drive", 10)
        self.odom_sub = self.create_subscription(Odometry, '/ego_racecar/odom', self.odom_callback, 10)

        self.velocities = []
        self.times = []
        self.trajectory = []  # List to store the trajectory as a sequence of (x, y) coordinates
        self.start_time = None
        self.lap_end_time = None
        self.lap_completed = False
          
    def scan_callback(self, msg):
        
        # obtain laser scans and preprocess them
        ranges = np.array(msg.ranges)
        self.range_max = msg.range_max
        ranges = np.clip(ranges, 0, self.range_max)
        
        min_angle = math.radians(-55)
        min_idx = int(np.floor((min_angle - msg.angle_min) / msg.angle_increment))
        max_angle = math.radians(55)
        max_idx = int(np.ceil((max_angle - msg.angle_min) / msg.angle_increment))
        
        # Find the closest point in the LiDAR ranges array
        closest_idx = min_idx
        closest_distance = msg.range_max * 5
        for i in range(min_idx, max_idx+1):
            distance = ranges[i-2] + ranges[i-1] + ranges[i] + ranges[i+1] + ranges[i+2]
            if distance < closest_distance:
                closest_distance = distance
                closest_idx = i

        # eliminate all points inside bubble
        radius = 115
        for i in range(closest_idx - radius, closest_idx + radius):
            ranges[i] = 0.0
        
        # return start index and end index of max gap in free space ranges
        start_gap = min_idx
        end_gap = min_idx
        current_start = -1
        duration = 0
        longest_duration = 0
           
        for i in range(min_idx, max_idx +1):
            if ranges[i] > 0.0:
                if current_start == -1:
                    current_start = i
            else:
                if current_start != -1:
                    duration = i - current_start
                    if duration > longest_duration:
                        longest_duration = duration
                        start_gap = current_start
                        end_gap = i - 1
        if current_start != -1:
            duration = max_idx + 1 - current_start
            if duration > longest_duration:
                longest_duration = duration
                start_gap = current_start
                end_gap = max_idx
                
        # return index of best 'goal' in the gap
        current_max = 0
        for i in range(start_gap, end_gap + 1):
            if ranges[i] > current_max:
                current_max = ranges[i]
                self.angle = msg.angle_min + i * msg.angle_increment
            elif ranges[i] == current_max:
                if abs(msg.angle_min + i * msg.angle_increment) < abs(self.angle):
                    self.angle = msg.angle_min + i * msg.angle_increment
        
        self.reactive_control()
        
    def reactive_control(self):
        drive_msg = AckermannDriveStamped()
        drive_msg.drive.steering_angle = self.angle
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

        # Append the current (x, y) position to the trajectory list
        self.trajectory.append((current_x, current_y))
        
        if current_x <= 1 and current_y <= 1 and not self.lap_completed:
            lap_time = (self.get_clock().now() - self.start_time).nanoseconds / 1e9
            self.lap_completed = True
            self.get_logger().info(f'Lap completed in {lap_time:.2f} seconds')
        
        if current_x and current_y <= 1:
            self.lap_completed = False
            
    def save_data(self):
        save_dir = os.path.expanduser('~/data_gap')
        os.makedirs(save_dir, exist_ok=True)
        # Save trajectory
        trajectory_path = os.path.join(save_dir, 'trajectory_gap.csv')
        with open(trajectory_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['X Position', 'Y Position'])  # Header
            for x, y in self.trajectory:
                writer.writerow([x, y])
        
        # Save velocities
        velocities_path = os.path.join(save_dir, 'velocities_gap.csv')
        with open(velocities_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Velocity'])  # Header
            for velocity in self.velocities:
                writer.writerow([velocity])
        
        # Save times
        times_path = os.path.join(save_dir, 'times_gap.csv')
        with open(times_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Time'])  # Header
            for time in self.times:
                writer.writerow([time])

        self.get_logger().info(f'Velocities saved to: {velocities_path}')
        self.get_logger().info(f'Times saved to: {times_path}')
        self.get_logger().info(f'Trajectory saved to: {trajectory_path}')

         

def main(args=None):
    rclpy.init(args=args)   
    node = GapFollowNode()     
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("Node stopped cleanly.")
    finally:
        node.save_data()
        rclpy.shutdown()       
       


if __name__ == "__main__":
    main()
