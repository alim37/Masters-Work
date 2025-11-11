#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np

from sensor_msgs.msg import Joy

class YourControlNode(Node):
    def __init__(self):
        # your existing init...
        self.joy_sub = self.create_subscription(
            Joy,
            'joy',
            self.joy_callback,
            10
        )

        self.deadman_enable = True # you could use a launch parameter to disable this.

        self.last_deadman_time = 0
      
    def joy_callback(self, msg):
        if self.deadman_enable:
            r1_value = msg.buttons[5]
            if r1.value > 0:
                self.last_deadman_time = self.get_time_s()

    def get_time_s(self): # Get the program time in seconds
        return self.get_clock().now().nanoseconds / 1e9

    def get_deadman_state(self):
        return self.get_time_s() < self.last_deadman_time + 0.25 # this value can be changed.

def main(args=None):
    rclpy.init(args=args)   
    node = SafetyNode()      
    rclpy.spin(node)
    rclpy.shutdown()        


if __name__ == "__main__":
    main()
