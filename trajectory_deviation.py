#!/usr/bin/env python3
import math
import numpy as np
from collections import deque

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import LaserScan, Imu
from geometry_msgs.msg import Quaternion, Point
from visualization_msgs.msg import Marker, MarkerArray

LIDAR_TOPIC = '/autodrive/f1tenth_1/lidar'
IPS_TOPIC = '/autodrive/f1tenth_1/ips'
IMU_TOPIC = '/autodrive/f1tenth_1/imu'
FRAME_ID = 'map'

ANGLE_WINDOW_DEG = 60.0
RANGE_MIN = 0.2
RANGE_MAX = 3.0

CLUSTER_MAX_GAP = 0.30
CLUSTER_MIN_SIZE = 4

EXPECTED_TARGET_DISTANCE = 1.5
DISTANCE_TOLERANCE = 0.8
MAX_CLUSTER_SPAN = 0.8
MIN_CLUSTER_SPAN = 0.15
COMPACTNESS_THRESHOLD = 2.5

MOTION_VELOCITY_THRESHOLD = 0.15
ENABLE_MOTION_FILTER = True

MAX_POSITION_JUMP = 1.0
TRACK_HISTORY_SIZE = 10

DEVIATION_THRESHOLD = 0.08
DEVIATION_HISTORY_SIZE = 5
MIN_POINTS_FOR_DEVIATION = 5

NUM_LINE_POINTS = 15
DOT_SIZE = 0.12

def quat_to_yaw(q: Quaternion) -> float:
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)

def rot2d(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=float)

class DynamicTargetTracker(Node):
    def __init__(self):
        super().__init__('dynamic_target_tracker')
        
        qos_lidar = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT, 
                               history=HistoryPolicy.KEEP_LAST)
        self.scan_sub = self.create_subscription(LaserScan, LIDAR_TOPIC, self.on_scan, qos_lidar)
        self.ips_sub = self.create_subscription(Point, IPS_TOPIC, self.on_point, 10)
        self.imu_sub = self.create_subscription(Imu, IMU_TOPIC, self.on_imu, 10)
        
        self.marker_pub = self.create_publisher(MarkerArray, 'target_markers', 10)
        
        self.have_ego_pose = False
        self.have_yaw = False
        self.x_e = self.y_e = self.yaw_e = 0.0
        
        self.target_position = None
        self.target_points_ego = None
        self.track_history = deque(maxlen=TRACK_HISTORY_SIZE)
        
        self.prev_scan_world_points = None
        self.prev_scan_timestamp = None
        
        self.deviation_history = deque(maxlen=DEVIATION_HISTORY_SIZE)
        self.current_deviation = 0.0
        self.deviation_direction = 0.0
        
        self.tracking_confidence = 0.0
        
        self.get_logger().info('Target tracker initialized')

    def on_point(self, msg: Point):
        self.x_e, self.y_e = msg.x, msg.y
        self.have_ego_pose = True

    def on_imu(self, msg: Imu):
        self.yaw_e = quat_to_yaw(msg.orientation)
        self.have_yaw = True

    def filter_moving_points(self, current_points_world, dt):
        if self.prev_scan_world_points is None or dt < 0.01:
            return np.ones(len(current_points_world), dtype=bool)
        
        moving_mask = []
        for curr_pt in current_points_world:
            dists = np.linalg.norm(self.prev_scan_world_points - curr_pt, axis=1)
            min_dist = np.min(dists)
            velocity = min_dist / dt
            moving_mask.append(velocity > MOTION_VELOCITY_THRESHOLD)
        
        moving_mask = np.array(moving_mask)
        
        if np.sum(moving_mask) < 3:
            return np.ones(len(current_points_world), dtype=bool)
        
        return moving_mask

    def evaluate_cluster_as_target(self, cluster_points_ego):
        if len(cluster_points_ego) < CLUSTER_MIN_SIZE:
            return 0.0, "too_few_points"
        
        xs, ys = cluster_points_ego[:, 0], cluster_points_ego[:, 1]
        
        centroid_range = math.hypot(np.mean(xs), np.mean(ys))
        distance_error = abs(centroid_range - EXPECTED_TARGET_DISTANCE)
        
        if distance_error > DISTANCE_TOLERANCE:
            return 0.0, f"wrong_distance_{centroid_range:.2f}m"
        
        distance_score = 1.0 - (distance_error / DISTANCE_TOLERANCE)
        
        x_span = np.max(xs) - np.min(xs)
        y_span = np.max(ys) - np.min(ys)
        max_span = max(x_span, y_span)
        
        if max_span > MAX_CLUSTER_SPAN:
            return 0.0, f"too_large_{max_span:.2f}m"
        if max_span < MIN_CLUSTER_SPAN:
            return 0.0, f"too_small_{max_span:.2f}m"
        
        compactness = max_span / len(cluster_points_ego)
        if compactness > COMPACTNESS_THRESHOLD:
            return 0.0, f"elongated_{compactness:.2f}"
        
        compactness_score = 1.0 - (compactness / COMPACTNESS_THRESHOLD)
        
        temporal_score = 1.0
        if self.target_position is not None:
            Rw = rot2d(self.yaw_e)
            centroid_ego = np.array([np.mean(xs), np.mean(ys)])
            centroid_world = Rw @ centroid_ego + np.array([self.x_e, self.y_e])
            
            distance_from_prediction = np.linalg.norm(centroid_world - self.target_position)
            
            if distance_from_prediction > MAX_POSITION_JUMP:
                return 0.0, f"jumped_{distance_from_prediction:.2f}m"
            
            temporal_score = 1.0 - (distance_from_prediction / MAX_POSITION_JUMP)
        
        total_score = distance_score * 0.4 + compactness_score * 0.3 + temporal_score * 0.3
        
        reason = f"d={centroid_range:.2f}m,c={compactness:.2f},t={temporal_score:.2f}"
        return total_score, reason

    def calculate_point_deviation(self, points_ego, centerline_end_x):
        if len(points_ego) < MIN_POINTS_FOR_DEVIATION:
            return 0.0, 0.0, 0.0
        
        xs = points_ego[:, 0]
        ys = points_ego[:, 1]
        
        forward_mask = (xs > 0) & (xs < centerline_end_x + 0.5)
        
        if np.sum(forward_mask) < MIN_POINTS_FOR_DEVIATION:
            return 0.0, 0.0, 0.0
        
        ys_forward = ys[forward_mask]
        
        lateral_deviations = np.abs(ys_forward)
        mean_deviation = np.mean(lateral_deviations)
        std_deviation = np.std(ys_forward)
        
        mean_y = np.mean(ys_forward)
        direction = 1.0 if mean_y > 0.01 else (-1.0 if mean_y < -0.01 else 0.0)
        
        return mean_deviation, direction, std_deviation

    def on_scan(self, scan: LaserScan):
        if not self.have_ego_pose or not self.have_yaw:
            return
        
        n = len(scan.ranges)
        angle_win_rad = math.radians(ANGLE_WINDOW_DEG)
        angles = scan.angle_min + np.arange(n) * scan.angle_increment
        ranges = np.array(scan.ranges, dtype=float)
        
        mask = ((angles >= -angle_win_rad) & (angles <= angle_win_rad) & 
                (ranges > RANGE_MIN) & (ranges < RANGE_MAX))
        
        if not np.any(mask):
            self.handle_no_detection()
            return
        
        a, r = angles[mask], ranges[mask]
        xs, ys = r * np.cos(a), r * np.sin(a)
        
        Rw = rot2d(self.yaw_e)
        all_points_ego = np.column_stack([xs, ys])
        all_points_world = (Rw @ all_points_ego.T).T + np.array([self.x_e, self.y_e])
        
        if ENABLE_MOTION_FILTER and self.prev_scan_world_points is not None:
            if self.prev_scan_timestamp is not None:
                dt = (scan.header.stamp.sec + scan.header.stamp.nanosec * 1e-9) - \
                     (self.prev_scan_timestamp.sec + self.prev_scan_timestamp.nanosec * 1e-9)
                dt = max(0.01, dt)
            else:
                dt = 0.05
            
            moving_mask = self.filter_moving_points(all_points_world, dt)
            
            if np.sum(moving_mask) >= CLUSTER_MIN_SIZE:
                xs = xs[moving_mask]
                ys = ys[moving_mask]
                all_points_world = all_points_world[moving_mask]
        
        self.prev_scan_world_points = all_points_world
        self.prev_scan_timestamp = scan.header.stamp
        
        order = np.argsort(a[:len(xs)])
        xs, ys = xs[order], ys[order]
        
        clusters = []
        current = [0]
        for i in range(1, len(xs)):
            gap = math.hypot(xs[i] - xs[i-1], ys[i] - ys[i-1])
            if gap <= CLUSTER_MAX_GAP:
                current.append(i)
            else:
                if len(current) >= CLUSTER_MIN_SIZE:
                    clusters.append(current)
                current = [i]
        if len(current) >= CLUSTER_MIN_SIZE:
            clusters.append(current)
        
        if not clusters:
            self.handle_no_detection()
            return
        
        best_cluster = None
        best_score = 0.0
        best_reason = ""
        
        for cluster_indices in clusters:
            cluster_points = np.column_stack([xs[cluster_indices], ys[cluster_indices]])
            score, reason = self.evaluate_cluster_as_target(cluster_points)
            
            if score > best_score:
                best_score = score
                best_cluster = cluster_indices
                best_reason = reason
        
        if best_cluster is None or best_score < 0.3:
            self.handle_no_detection()
            return
        
        cluster_xs = xs[best_cluster]
        cluster_ys = ys[best_cluster]
        
        self.target_points_ego = np.column_stack([cluster_xs, cluster_ys])
        
        centroid_ego = np.array([np.mean(cluster_xs), np.mean(cluster_ys)])
        centroid_world = Rw @ centroid_ego + np.array([self.x_e, self.y_e])
        
        self.target_position = centroid_world
        self.track_history.append(centroid_world)
        self.tracking_confidence = min(1.0, self.tracking_confidence + 0.2)
        
        target_distance = np.linalg.norm(centroid_ego)
        mean_dev, dev_dir, dev_std = self.calculate_point_deviation(
            self.target_points_ego, target_distance
        )
        
        self.deviation_history.append(mean_dev)
        self.current_deviation = np.mean(self.deviation_history)
        self.deviation_direction = dev_dir
        
        self.publish_visualization(scan.header.stamp)

    def handle_no_detection(self):
        self.tracking_confidence *= 0.7
        if self.tracking_confidence < 0.2:
            self.target_position = None
            self.target_points_ego = None
            self.current_deviation = 0.0
        self.publish_visualization(self.get_clock().now().to_msg())

    def publish_visualization(self, stamp):
        ma = MarkerArray()
        
        for i in range(NUM_LINE_POINTS + 5):
            m = Marker()
            m.header.frame_id = FRAME_ID
            m.header.stamp = stamp
            m.ns = 'target_line'
            m.id = i
            m.action = Marker.DELETE
            ma.markers.append(m)
        
        if self.target_position is None:
            self.marker_pub.publish(ma)
            return
        
        ego_pos = np.array([self.x_e, self.y_e])
        target_pos = self.target_position
        
        for i in range(NUM_LINE_POINTS):
            t = i / (NUM_LINE_POINTS - 1)
            point = ego_pos + t * (target_pos - ego_pos)
            
            m = Marker()
            m.header.frame_id = FRAME_ID
            m.header.stamp = stamp
            m.ns = 'target_line'
            m.id = i
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position.x = float(point[0])
            m.pose.position.y = float(point[1])
            m.pose.position.z = 0.1
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = m.scale.z = DOT_SIZE
            
            if self.current_deviation > DEVIATION_THRESHOLD:
                m.color.r, m.color.g, m.color.b = 1.0, 1.0, 0.0
                m.color.a = 1.0
            else:
                m.color.r, m.color.g, m.color.b = 0.0, 1.0, 0.0
                m.color.a = 1.0
            
            ma.markers.append(m)
        
        text = Marker()
        text.header.frame_id = FRAME_ID
        text.header.stamp = stamp
        text.ns = 'target_line'
        text.id = 100
        text.type = Marker.TEXT_VIEW_FACING
        text.action = Marker.ADD
        text.pose.position.x = float(target_pos[0])
        text.pose.position.y = float(target_pos[1])
        text.pose.position.z = 0.5
        text.scale.z = 0.2
        text.color.r = text.color.g = text.color.b = text.color.a = 1.0
        
        if self.target_points_ego is not None:
            text.text = f'Dev: {self.current_deviation:.3f}m'
        else:
            text.text = f'Dev: {self.current_deviation:.3f}m'
        
        ma.markers.append(text)
        self.marker_pub.publish(ma)

def main():
    rclpy.init()
    node = DynamicTargetTracker()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
