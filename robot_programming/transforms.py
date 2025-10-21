import math
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PointStamped
import tf2_ros
from tf2_ros import TransformListener
import tf2_geometry_msgs
import time

class Laser_to_base(Node):

    def __init__(self):
        super().__init__('Laser_to_Base')
        self.laser_sub = self.create_subscription(LaserScan,"/scan",self.scan_callback,10)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer,self)

    def scan_callback(self, msg: LaserScan):
        self.angle_min = msg.angle_min
        self.angle_inc = msg.angle_increment
        self.ranges = msg.ranges

        for i, ri in enumerate(self.ranges):
            theta_i = self.angle_min + i*self.angle_inc

            xL = ri*math.cos(theta_i)
            yL = ri*math.sin(theta_i)

            pt = PointStamped()
            pt.header = msg.header
            pt.header.frame_id = 'ego_racecar/laser_model'
            pt.point.x = xL
            pt.point.y = yL
            pt.point.z = 0.0
            if not self.tf_buffer.can_transform("ego_racecar/base_link", msg.header.frame_id, msg.header.stamp,rclpy.duration.Duration(seconds=0.5)):
                self.get_logger().info("waiting")
                
                return  # wait until TF is available

            transformed = self.tf_buffer.transform(pt,"ego_racecar/base_link")
            xB = transformed.point.x
            yB = transformed.point.y
            self.get_logger().info(f"X Point: {xB} Y Point: {yB}")



def main(args=None):
    rclpy.init(args=args)
    node = Laser_to_base()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__=='__main__':
    main()
