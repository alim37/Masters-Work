#!/usr/bin/env python3
import math
import numpy as np

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker, MarkerArray

class TargetTrackerNode(Node):
    def __init__(self):
        super().__init__('target_tracker')

        self.declare_parameter('lidar_topic', '/autodrive/f1tenth_1/lidar')
        self.declare_parameter('target_angle_window_deg', 40.0)  # lookahead cone: +/- deg
        self.declare_parameter('range_min', 0.2)
        self.declare_parameter('range_max', 15.0)
        self.declare_parameter('cluster_max_gap', 0.25)          # meters between adjacent points
        self.declare_parameter('cluster_min_size', 4)            # min points per cluster
        self.declare_parameter('marker_scale', 0.26)             # sphere diameter (m)

        self.lidar_topic = self.get_parameter('lidar_topic').value
        self.angle_win_rad = math.radians(self.get_parameter('target_angle_window_deg').value)
        self.range_min = float(self.get_parameter('range_min').value)
        self.range_max = float(self.get_parameter('range_max').value)
        self.cluster_max_gap = float(self.get_parameter('cluster_max_gap').value)
        self.cluster_min_size = int(self.get_parameter('cluster_min_size').value)
        self.marker_scale = float(self.get_parameter('marker_scale').value)

        self.scan_sub = self.create_subscription(LaserScan, self.lidar_topic, self.scan_callback, 10)
        self.marker_pub = self.create_publisher(MarkerArray, 'target_marker', 10)

    def scan_callback(self, scan: LaserScan):
        ranges = np.asarray(scan.ranges, dtype=float)
        n = ranges.size
        if n == 0:
            return

        angles = scan.angle_min + np.arange(n) * scan.angle_increment

        forward = (angles >= -self.angle_win_rad) & (angles <= self.angle_win_rad)
        valid = forward & (ranges > self.range_min) & (ranges < self.range_max)
        if not np.any(valid):
            return

        a = angles[valid]
        r = ranges[valid]
        xs = r * np.cos(a)  
        ys = r * np.sin(a)

        order = np.argsort(a)
        xs, ys, r = xs[order], ys[order], r[order]

        clusters = []
        current = [0]
        for i in range(1, xs.size):
            if math.hypot(xs[i] - xs[i-1], ys[i] - ys[i-1]) <= self.cluster_max_gap:
                current.append(i)
            else:
                if len(current) >= self.cluster_min_size:
                    clusters.append(current)
                current = [i]
        if len(current) >= self.cluster_min_size:
            clusters.append(current)

        if not clusters:
            return

        best_idx = min(clusters, key=lambda idxs: float(np.mean(r[idxs])))
        cx = float(np.mean(xs[best_idx]))
        cy = float(np.mean(ys[best_idx]))

        ma = MarkerArray()
        m = Marker()
        m.header.frame_id = scan.header.frame_id 
        m.header.stamp = scan.header.stamp
        m.ns = 'lidar_target'
        m.id = 1
        m.type = Marker.SPHERE
        m.action = Marker.ADD
        m.pose.position.x = cx
        m.pose.position.y = cy
        m.pose.position.z = 0.0
        m.pose.orientation.w = 1.0
        m.scale.x = m.scale.y = m.scale.z = self.marker_scale
        m.color.a = 1.0  
        ma.markers.append(m)
        self.marker_pub.publish(ma)

def main():
    rclpy.init()
    node = TargetTrackerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
