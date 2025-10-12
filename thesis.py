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

# ============ CONFIGURATION ============
LIDAR_TOPIC = '/autodrive/f1tenth_1/lidar'
IPS_TOPIC = '/autodrive/f1tenth_1/ips'
IMU_TOPIC = '/autodrive/f1tenth_1/imu'
FRAME_ID = 'map'

# LiDAR detection
ANGLE_WINDOW_DEG = 90.0  # Wide cone to see target
RANGE_MIN = 0.2
RANGE_MAX = 3.0  # Only look close (target should be ~1.5m away)

# Clustering
CLUSTER_MAX_GAP = 0.30  # Max distance between points in same cluster
CLUSTER_MIN_SIZE = 3    # Minimum points to be valid cluster

# Target identification (KEY: distinguish car from walls)
EXPECTED_TARGET_DISTANCE = 1.5  # meters - where we expect target to be
DISTANCE_TOLERANCE = 0.8        # ± tolerance for target distance
MAX_CLUSTER_SPAN = 0.8          # meters - max width/length of target cluster
MIN_CLUSTER_SPAN = 0.15         # meters - min size to be a car
COMPACTNESS_THRESHOLD = 2.5     # ratio of span to points (walls are elongated)

# Tracking
MAX_POSITION_JUMP = 1.0  # Stricter - target can't jump far between frames
TRACK_HISTORY_SIZE = 10  # Store last N target positions
VELOCITY_SMOOTHING = 0.7 # Exponential smoothing for velocity

# Visualization
NUM_LINE_POINTS = 15
DOT_SIZE = 0.12
# =======================================

def quat_to_yaw(q: Quaternion) -> float:
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)

def rot2d(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=float)

class TargetTrackerNode(Node):
    def __init__(self):
        super().__init__('target_tracker')
        
        # Subscriptions
        qos_lidar = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT, 
                               history=HistoryPolicy.KEEP_LAST)
        self.scan_sub = self.create_subscription(LaserScan, LIDAR_TOPIC, self.on_scan, qos_lidar)
        self.ips_sub = self.create_subscription(Point, IPS_TOPIC, self.on_point, 10)
        self.imu_sub = self.create_subscription(Imu, IMU_TOPIC, self.on_imu, 10)
        
        # Publisher
        self.marker_pub = self.create_publisher(MarkerArray, 'target_markers', 10)
        
        # Ego state
        self.have_ego_pose = False
        self.have_yaw = False
        self.x_e = self.y_e = self.yaw_e = 0.0
        
        # Target tracking state
        self.target_position = None      # Current target position (world frame)
        self.target_velocity = np.array([0.0, 0.0])  # Velocity estimate
        self.track_history = deque(maxlen=TRACK_HISTORY_SIZE)  # Recent positions
        self.last_timestamp = None
        
        # Current detection for visualization
        self.current_cluster_world = None
        self.tracking_confidence = 0.0  # 0-1, how confident we are
        
        self.get_logger().info('Robust LiDAR target tracker initialized')

    def on_point(self, msg: Point):
        self.x_e, self.y_e = msg.x, msg.y
        self.have_ego_pose = True

    def on_imu(self, msg: Imu):
        self.yaw_e = quat_to_yaw(msg.orientation)
        self.have_yaw = True

    def evaluate_cluster_as_target(self, cluster_points_ego):
        """
        Score a cluster based on how "car-like" it is.
        Returns: (score, reason) where higher score = more likely to be target
        """
        if len(cluster_points_ego) < CLUSTER_MIN_SIZE:
            return 0.0, "too_few_points"
        
        xs, ys = cluster_points_ego[:, 0], cluster_points_ego[:, 1]
        
        # Metric 1: Distance from ego (should be ~1.5m ahead)
        centroid_range = math.hypot(np.mean(xs), np.mean(ys))
        distance_error = abs(centroid_range - EXPECTED_TARGET_DISTANCE)
        
        if distance_error > DISTANCE_TOLERANCE:
            return 0.0, f"wrong_distance_{centroid_range:.2f}m"
        
        distance_score = 1.0 - (distance_error / DISTANCE_TOLERANCE)
        
        # Metric 2: Cluster compactness (car is compact, walls are elongated)
        x_span = np.max(xs) - np.min(xs)
        y_span = np.max(ys) - np.min(ys)
        max_span = max(x_span, y_span)
        
        if max_span > MAX_CLUSTER_SPAN:
            return 0.0, f"too_large_{max_span:.2f}m"
        
        if max_span < MIN_CLUSTER_SPAN:
            return 0.0, f"too_small_{max_span:.2f}m"
        
        # Calculate compactness: spread per point (lower = more compact)
        compactness = max_span / len(cluster_points_ego)
        
        if compactness > COMPACTNESS_THRESHOLD:
            return 0.0, f"elongated_{compactness:.2f}"
        
        compactness_score = 1.0 - (compactness / COMPACTNESS_THRESHOLD)
        
        # Metric 3: Point density (car has moderate density, walls are sparse)
        area = x_span * y_span if x_span > 0 and y_span > 0 else 0.01
        density = len(cluster_points_ego) / area
        
        # Good density is 10-100 points/m²
        if density < 5:
            density_score = density / 5.0
        elif density > 100:
            density_score = 0.5
        else:
            density_score = 1.0
        
        # Metric 4: Temporal consistency (if we have tracking history)
        temporal_score = 1.0
        if self.target_position is not None:
            # Transform centroid to world
            Rw = rot2d(self.yaw_e)
            centroid_ego = np.array([np.mean(xs), np.mean(ys)])
            centroid_world = Rw @ centroid_ego + np.array([self.x_e, self.y_e])
            
            # Check distance from predicted position
            predicted = self.predict_position(0.0)
            distance_from_prediction = np.linalg.norm(centroid_world - predicted)
            
            if distance_from_prediction > MAX_POSITION_JUMP:
                return 0.0, f"jumped_{distance_from_prediction:.2f}m"
            
            temporal_score = 1.0 - (distance_from_prediction / MAX_POSITION_JUMP)
        
        # Metric 5: Forward direction (target should be generally ahead, not behind)
        centroid_angle = math.atan2(np.mean(ys), np.mean(xs))
        angle_window_rad = math.radians(ANGLE_WINDOW_DEG)
        
        if abs(centroid_angle) > angle_window_rad:
            return 0.0, f"wrong_angle_{math.degrees(centroid_angle):.1f}deg"
        
        angle_score = 1.0 - (abs(centroid_angle) / angle_window_rad)
        
        # Combined score (weighted average)
        total_score = (
            distance_score * 0.30 +      # Distance is very important
            compactness_score * 0.25 +   # Shape matters
            temporal_score * 0.25 +      # Consistency matters
            density_score * 0.10 +       # Density helps
            angle_score * 0.10           # Direction helps
        )
        
        reason = f"dist={centroid_range:.2f}m,compact={compactness:.2f},temp={temporal_score:.2f}"
        return total_score, reason

    def predict_position(self, dt):
        """Predict target position after dt seconds using velocity."""
        if self.target_position is None:
            return np.array([self.x_e + EXPECTED_TARGET_DISTANCE, self.y_e])
        
        predicted = self.target_position + self.target_velocity * dt
        return predicted

    def update_target_state(self, centroid_world, timestamp):
        """Update target position and velocity estimates."""
        if self.target_position is None:
            # First detection
            self.target_position = centroid_world
            self.track_history.append(centroid_world)
            self.last_timestamp = timestamp
            return
        
        # Calculate dt
        if self.last_timestamp is not None:
            dt = (timestamp.sec + timestamp.nanosec * 1e-9) - \
                 (self.last_timestamp.sec + self.last_timestamp.nanosec * 1e-9)
            dt = max(0.01, min(dt, 0.5))  # Clamp to reasonable range
        else:
            dt = 0.05
        
        # Update velocity with exponential smoothing
        measured_velocity = (centroid_world - self.target_position) / dt
        alpha = VELOCITY_SMOOTHING
        self.target_velocity = alpha * self.target_velocity + (1 - alpha) * measured_velocity
        
        # Update position
        self.target_position = centroid_world
        self.track_history.append(centroid_world)
        self.last_timestamp = timestamp
        
        # Increase confidence
        self.tracking_confidence = min(1.0, self.tracking_confidence + 0.2)

    def on_scan(self, scan: LaserScan):
        if not self.have_ego_pose or not self.have_yaw:
            return
        
        # Extract points in wide cone
        n = len(scan.ranges)
        angle_win_rad = math.radians(ANGLE_WINDOW_DEG)
        angles = scan.angle_min + np.arange(n) * scan.angle_increment
        ranges = np.array(scan.ranges, dtype=float)
        
        # Filter to forward cone and valid range
        mask = ((angles >= -angle_win_rad) & (angles <= angle_win_rad) & 
                (ranges > RANGE_MIN) & (ranges < RANGE_MAX))
        
        if not np.any(mask):
            self.handle_lost_target()
            return
        
        # Convert to cartesian (ego frame)
        a, r = angles[mask], ranges[mask]
        xs, ys = r * np.cos(a), r * np.sin(a)
        
        # Sort by angle for clustering
        order = np.argsort(a)
        xs, ys = xs[order], ys[order]
        
        # Cluster points
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
            self.handle_lost_target()
            return
        
        # Evaluate each cluster and pick best
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
        
        if best_cluster is None or best_score < 0.3:  # Minimum confidence threshold
            self.get_logger().warn(f'No good target found. Best score: {best_score:.2f}', 
                                  throttle_duration_sec=0.5)
            self.handle_lost_target()
            return
        
        # Found target!
        cluster_xs = xs[best_cluster]
        cluster_ys = ys[best_cluster]
        
        # Transform to world frame
        Rw = rot2d(self.yaw_e)
        cluster_points_ego = np.column_stack([cluster_xs, cluster_ys])
        cluster_points_world = (Rw @ cluster_points_ego.T).T + np.array([self.x_e, self.y_e])
        
        centroid_ego = np.array([np.mean(cluster_xs), np.mean(cluster_ys)])
        centroid_world = Rw @ centroid_ego + np.array([self.x_e, self.y_e])
        
        # Update tracking state
        self.update_target_state(centroid_world, scan.header.stamp)
        self.current_cluster_world = cluster_points_world
        
        self.get_logger().info(
            f'Target found: score={best_score:.2f}, {best_reason}, '
            f'{len(best_cluster)} pts, conf={self.tracking_confidence:.2f}',
            throttle_duration_sec=0.5
        )
        
        self.publish_visualization(scan.header.stamp)

    def handle_lost_target(self):
        """Called when no valid target detected."""
        # Decay confidence
        self.tracking_confidence *= 0.8
        
        if self.tracking_confidence < 0.1:
            # Lost track completely
            self.get_logger().warn('Target lost - resetting tracker')
            self.target_position = None
            self.target_velocity = np.array([0.0, 0.0])
            self.current_cluster_world = None
            self.track_history.clear()
        else:
            # Predict forward
            if self.last_timestamp is not None:
                self.target_position = self.predict_position(0.05)
        
        self.publish_visualization(self.get_clock().now().to_msg())

    def publish_visualization(self, stamp):
        """Visualize target as line of dots."""
        ma = MarkerArray()
        
        if self.target_position is None:
            # Clear markers
            for i in range(NUM_LINE_POINTS):
                m = Marker()
                m.header.frame_id = FRAME_ID
                m.header.stamp = stamp
                m.ns = 'target'
                m.id = i
                m.action = Marker.DELETE
                ma.markers.append(m)
            self.marker_pub.publish(ma)
            return
        
        # Create line from ego to target
        ego_pos = np.array([self.x_e, self.y_e])
        target_pos = self.target_position
        
        for i in range(NUM_LINE_POINTS):
            t = i / (NUM_LINE_POINTS - 1)
            point = ego_pos + t * (target_pos - ego_pos)
            
            m = Marker()
            m.header.frame_id = FRAME_ID
            m.header.stamp = stamp
            m.ns = 'target'
            m.id = i
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position.x = float(point[0])
            m.pose.position.y = float(point[1])
            m.pose.position.z = 0.1
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = m.scale.z = DOT_SIZE
            
            # Color based on confidence
            if self.tracking_confidence > 0.7:
                m.color.r, m.color.g, m.color.b = 0.0, 1.0, 0.0  # Green - confident
            elif self.tracking_confidence > 0.3:
                m.color.r, m.color.g, m.color.b = 1.0, 1.0, 0.0  # Yellow - uncertain
            else:
                m.color.r, m.color.g, m.color.b = 1.0, 0.0, 0.0  # Red - lost
            
            m.color.a = self.tracking_confidence
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
