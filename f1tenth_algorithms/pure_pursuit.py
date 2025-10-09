#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry
import numpy as np
import math
import matplotlib.pyplot as plt
import os
import csv

class PurePursuitNode(Node):      

    def __init__(self):
        super().__init__("pure_pursuit")  
        
        self.targets = np.array([
            [1, 0],   
            [2, 0],   
            [3, 0],   
            [4, 0],   
            [5, 0],   
            [6, 0],
            [7, 0],
            [8,0],
            #[8.9473, 0],
            [9.5, 0],
            #[9.2211, 0.0591],
            #[9.6555, 0.3607],
            #[9.9239, 1.0871],
           # [9.7781, 2.0921],
            [9.8000, 3.0000],
            [9.8000, 4.0000],
            [9.8000, 5.0000],
            [9.8000, 6.0000],
            [9.8000, 7.0000],
            [9.8000, 7.3779],
            [9.8132, 7.7535],
            [9.5962, 8.1122],
            [8.9921, 8.3980],
            [7.7689, 8.4908],
            [7.0000, 8.5759],
            
           # [6.0000, 8.5759],
            #[5.0000, 8.5759],
           # [4.0000, 8.5759],
           # [3.0000, 8.5759],
           # [2.0000, 8.5759],
          #  [1.0000, 8.5759],
            #[0.0000, 8.5759],
           # [-1.0000, 8.5759],
           # [-2.0000, 8.5759],
           # [-3.0000, 8.5759],
          #  [-4.0000, 8.5759],
          #  [-5.0000, 8.5759],
         #   [-6.0000, 8.5759],
          #  [-7.0000, 8.5759],
         #   [-8.0000, 8.5759],
           # [-9.0000, 8.5759],
           # [-10.0000, 8.5759],
           # [-11.0000, 8.5759],
          #  [-12.0000, 8.5759],
            #[-12.4353, 8.5759],
            [6.0000, 9.1],
            [5.0000, 9.1],
            [4.0000, 9.1],
            [3.0000, 9.1],
            [2.0000, 9.1],
            [1.0000,9.1],
            [0.0000, 9.1],
            [-1.0000, 9.1],
            [-2.0000, 9.1],
            [-3.0000, 9.1],
            [-4.0000, 9.1],
            [-5.0000, 9.1],
            [-6.0000, 9.1],
            [-7.0000, 9.1],
            [-8.0000, 9.1],
            [-9.0000, 9.1],
            [-10.0000, 9.1],
            [-11.0000, 9.1],
            [-12.0000, 9.1],
            [-13.0, 9.1],
            #[-12.9307, 8.3914],
            #[-13.3309, 7.8879],
            #[-13.6016, 6.5028],
            [-14, 6.0000],
            [-14, 5.0000],
            [-14, 4.0000],
            [-14, 3.0000],
            [-14, 2.0000],
            [-14, 1.0000],
            [-14, 0.3887],
            [-13.0705, -0.0388],
            [-12.0000, 0],  
            [-11.0000, 0],
            [-10.0000, 0],
            [-9, 0],
            [-8, 0],  
            [-7, 0], 
            [-6, 0], 
            [-5, 0], 
            [-4, 0], 
            [-3, 0], 
            [-2, 0], 
            [-1, 0],                    
        ]) 
                
        self.current_target_idx = 0

        self.lookahead_distance = 2.0
        self.max_steering_angle = 0.5
        
        self.wheelbase = 0.3

        self.drive_pub =self.create_publisher(AckermannDriveStamped, '/drive', 10)
        self.odom_sub = self.create_subscription(Odometry, '/ego_racecar/odom', self.odom_callback, 10)
        self.velocities =[]
        self.times = []
        self.start_time = None
        self.lap_end_time = None

        self.create_timer(0.1, self.pure_pursuit)
    
    def odom_callback(self, msg):
        self.current_pose = msg.pose.pose
        #self.get_logger().info(f'Received odom: x={self.current_pose.position.x:.2f}, y={self.current_pose.position.y:.2f}')
        
        if self.start_time is None:
            self.start_time = self.get_clock().now()
        
        current_time = (self.get_clock().now() - self.start_time).nanoseconds / 1e9
        v_x = msg.twist.twist.linear.x
        v_y = msg.twist.twist.linear.y
        total_velocity = math.sqrt(v_x**2 + v_y**2)
        
        self.velocities.append(total_velocity)
        self.times.append(current_time)
        
        #self.get_logger().info(f'Velocity is: {v_x:.2f} at time: {current_time:.2f}')

    def pure_pursuit(self):
        if self.current_target_idx >= len(self.targets):
            if self.lap_end_time is None:
                self.lap_end_time = (self.get_clock().now() - self.start_time).nanoseconds / 1e9     
            return
        
        current_target = self.targets[self.current_target_idx]

        dx = current_target[0] - self.current_pose.position.x
        dy = current_target[1] - self.current_pose.position.y
        distance = math.sqrt(dx**2 + dy**2)

        if distance < self.lookahead_distance:
            self.current_target_idx += 1
            return
        
        _, _, yaw = self.euler_from_quaternion(self.current_pose.orientation)
        
        goal_x_veh = dx * math.cos(yaw) + dy*math.sin(yaw)
        goal_y_veh = -dx * math.sin(yaw) + dy * math.cos(yaw)
        
        y = dy 
        #steering_angle = 2 * abs(y) / (self.lookahead_distance**2)
        #steering_angle = min(steering_angle, self.max_steering_angle)
        
        steering_angle = math.atan2(2*self.wheelbase * goal_y_veh, distance**2)
        #steering_angle = max(-self.max_steering_angle, min(self.max_steering_angle, steering_angle))

        drive_msg = AckermannDriveStamped()
        drive_msg.drive.speed = 1.5
        drive_msg.drive.steering_angle = steering_angle

        self.drive_pub.publish(drive_msg)
    
    # https://gist.github.com/salmagro/2e698ad4fbf9dae40244769c5ab74434
    def euler_from_quaternion(self, quaternion):
        x = quaternion.x
        y = quaternion.y
        z = quaternion.z
        w = quaternion.w

        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = math.atan2(sinr_cosp, cosr_cosp)

        sinp = 2 * (w * y - z * x)
        pitch = math.asin(sinp)

        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        return roll, pitch, yaw
    
    #def plot_velocity(self):
    #    plt.figure(figsize=(10,6))
    #    plt.plot(self.times, self.velocities)
    #    plt.xlabel('Time (s)')
    #    plt.ylabel('Total velocity (m/s)')
    #    plt.grid(True)
    #    
    #    save_dir = os.path.expanduser('~/graphs/velocity_vs_time.png')
    #    os.makedirs(save_dir, exist_ok=True)
    #    
    #    save_path = os.path.join(save_dir, 'velocity_vs_time_pp.png')
    #    plt.savefig(save_path)
    #    plt.show()
    #    plt.close()
    #    
    #    self.get_logger().info(f'Graph saved to: {save_path}')

    def save_data(self):
        save_dir = os.path.expanduser('~/data_pp')
        os.makedirs(save_dir, exist_ok=True)
        
        # Save velocities
        velocities_path = os.path.join(save_dir, 'velocities_pp.csv')
        with open(velocities_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Velocity'])  # Header
            for velocity in self.velocities:
                writer.writerow([velocity])
        
        # Save times
        times_path = os.path.join(save_dir, 'times_pp.csv')
        with open(times_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Time'])  # Header
            for time in self.times:
                writer.writerow([time])
                
        # Save lap time
        lap_time_path = os.path.join(save_dir, 'lap_time_pp.csv')
        with open(lap_time_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Lap Time'])
            if self.lap_end_time is not None:
                writer.writerow([self.lap_end_time])
            
        
        self.get_logger().info(f'Velocities saved to: {velocities_path}')
        self.get_logger().info(f'Times saved to: {times_path}')
        self.get_logger().info(f'Lap time saved to: {lap_time_path}')


def main(args=None):
    rclpy.init(args=args)   

    node = PurePursuitNode()     
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("Node stopped cleanly.")
    finally:
        node.save_data()
        rclpy.shutdown()         

if __name__ == "__main__":
    main()
