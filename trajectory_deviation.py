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
from std_msgs.msg import Float32

# ============ CONFIGURATION ============
LIDAR_TOPIC = '/autodrive/f1tenth_1/lidar'
IPS_TOPIC = '/autodrive/f1tenth_1/ips'
IMU_TOPIC = '/autodrive/f1tenth_1/imu'
FRAME_ID = 'map'

# LiDAR detection
ANGLE_WINDOW_DEG = 60.0  # Wider cone to see target
RANGE_MIN = 0.2
RANGE_MAX = 3.0

# Clustering (to find distinct objects)
CLUSTER_MAX_GAP = 0.30
CLUSTER_MIN_SIZE = 4

# Target identification (distinguish car from walls)
EXPECTED_TARGET_DISTANCE = 1.5
DISTANCE_TOLERANCE = 0.8
MAX_CLUSTER_SPAN = 0.8  # meters - max width of car
MIN_CLUSTER_SPAN = 0.15  # meters - min size to be real object
COMPACTNESS_THRESHOLD = 2.5  # reject elongated wall segments

# Motion filtering
MOTION_VELOCITY_THRESHOLD = 0.15  # m/s - moving vs static
ENABLE_MOTION_FILTER = True

# Tracking
MAX_POSITION_JUMP = 1.0
TRACK_HISTORY_SIZE = 10

# IPS trajectory fallback
TRAJECTORY_MEMORY_SIZE = 500
TRAJECTORY_LOOKAHEAD = 1.5

# Deviation-based turn detection
DEVIATION_THRESHOLD = 0.08  # meters - lateral deviation triggers turn indication
DEVIATION_HISTORY_SIZE = 5
MIN_POINTS_FOR_DEVIATION = 5

# Visualization
NUM_LINE_POINTS = 8
DOT_SIZE = 0.12
# =======================================

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
        
        # Subscriptions
        qos_lidar = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT, 
                               history=HistoryPolicy.KEEP_LAST)
        self.scan_sub = self.create_subscription(LaserScan, LIDAR_TOPIC, self.on_scan, qos_lidar)
        self.ips_sub = self.create_subscription(Point, IPS_TOPIC, self.on_point, 10)
        self.imu_sub = self.create_subscription(Imu, IMU_TOPIC, self.on_imu, 10)
        
        # Publishers
        self.marker_pub = self.create_publisher(MarkerArray, 'target_markers', 10)
        
        # Ego state
        self.have_ego_pose = False
        self.have_yaw = False
        self.x_e = self.y_e = self.yaw_e = 0.0
        
        # Ego trajectory for IPS fallback
        self.ego_trajectory = deque(maxlen=TRAJECTORY_MEMORY_SIZE)
        
        # Target state
        self.target_position = None  # World frame
        self.target_points_ego = None  # Current LiDAR points on target (ego frame)
        self.track_history = deque(maxlen=TRACK_HISTORY_SIZE)
        
        # Motion filtering state
        self.prev_scan_world_points = None
        self.prev_scan_timestamp = None
        
        # Deviation tracking (KEY FOR PAPER)
        self.deviation_history = deque(maxlen=DEVIATION_HISTORY_SIZE)
        self.current_deviation = 0.0  # Lateral deviation from centerline
        self.deviation_direction = 0.0  # +1 = left, -1 = right
        
        # Detection confidence
        self.tracking_confidence = 0.0
        self.using_ips_fallback = False
        
        self.get_logger().info('Dynamic deviation-based tracker initialized')

    def on_point(self, msg: Point):
        self.x_e, self.y_e = msg.x, msg.y
        self.have_ego_pose = True
        self.ego_trajectory.append((self.x_e, self.y_e))

    def on_imu(self, msg: Imu):
        self.yaw_e = quat_to_yaw(msg.orientation)
        self.have_yaw = True

    def find_trajectory_point_ahead(self):
        """IPS trajectory fallback when LiDAR loses target."""
        if len(self.ego_trajectory) < 10:
            return None
        
        traj = np.array(self.ego_trajectory)
        ego_pos = np.array([self.x_e, self.y_e])
        
        dists = np.sqrt(np.sum((traj - ego_pos)**2, axis=1))
        closest_idx = np.argmin(dists)
        
        cumulative_dist = 0.0
        for i in range(closest_idx, len(traj)):
            if i > 0:
                segment_dist = np.linalg.norm(traj[i] - traj[i-1])
                cumulative_dist += segment_dist
                if cumulative_dist >= TRAJECTORY_LOOKAHEAD:
                    return traj[i]
        
        if len(traj) > closest_idx + 1:
            return traj[-1]
        return None

    def filter_moving_points(self, current_points_world, dt):
        """Filter to only moving points (target car vs static walls)."""
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
        """
        Score cluster on how "car-like" it is.
        Returns: (score, reason)
        """
        if len(cluster_points_ego) < CLUSTER_MIN_SIZE:
            return 0.0, "too_few_points"
        
        xs, ys = cluster_points_ego[:, 0], cluster_points_ego[:, 1]
        
        # Metric 1: Distance
        centroid_range = math.hypot(np.mean(xs), np.mean(ys))
        distance_error = abs(centroid_range - EXPECTED_TARGET_DISTANCE)
        
        if distance_error > DISTANCE_TOLERANCE:
            return 0.0, f"wrong_distance_{centroid_range:.2f}m"
        
        distance_score = 1.0 - (distance_error / DISTANCE_TOLERANCE)
        
        # Metric 2: Compactness (key: rejects elongated walls)
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
        
        # Metric 3: Temporal consistency
        temporal_score = 1.0
        if self.target_position is not None:
            Rw = rot2d(self.yaw_e)
            centroid_ego = np.array([np.mean(xs), np.mean(ys)])
            centroid_world = Rw @ centroid_ego + np.array([self.x_e, self.y_e])
            
            distance_from_prediction = np.linalg.norm(centroid_world - self.target_position)
            
            if distance_from_prediction > MAX_POSITION_JUMP:
                return 0.0, f"jumped_{distance_from_prediction:.2f}m"
            
            temporal_score = 1.0 - (distance_from_prediction / MAX_POSITION_JUMP)
        
        # Combined score
        total_score = distance_score * 0.4 + compactness_score * 0.3 + temporal_score * 0.3
        
        reason = f"d={centroid_range:.2f}m,c={compactness:.2f},t={temporal_score:.2f}"
        return total_score, reason

    def project_centerline(self, target_distance):
        """
        Project a straight line from ego forward to target.
        This is the "ideal" path if target goes straight.
        """
        # In ego frame, centerline is just the x-axis
        # From ego (0,0) to target at (target_distance, 0)
        return np.array([[0.0, 0.0], [target_distance, 0.0]])

    def calculate_point_deviation(self, points_ego, centerline_end_x):
        """
        Calculate lateral deviation of points from centerline.
        
        Theory: If target going straight, all points align on x-axis (y ≈ 0).
        When target turns, points spread laterally (y values increase).
        
        Returns: (mean_deviation, deviation_direction, deviation_std)
        """
        if len(points_ego) < MIN_POINTS_FOR_DEVIATION:
            return 0.0, 0.0, 0.0
        
        xs = points_ego[:, 0]
        ys = points_ego[:, 1]
        
        # Filter to points in front of us (x > 0) and within target distance
        forward_mask = (xs > 0) & (xs < centerline_end_x + 0.5)
        
        if np.sum(forward_mask) < MIN_POINTS_FOR_DEVIATION:
            return 0.0, 0.0, 0.0
        
        ys_forward = ys[forward_mask]
        
        # Lateral deviation = absolute y-coordinates
        # (centerline is y=0 in ego frame)
        lateral_deviations = np.abs(ys_forward)
        mean_deviation = np.mean(lateral_deviations)
        std_deviation = np.std(ys_forward)
        
        # Direction: which side is target turning towards?
        mean_y = np.mean(ys_forward)
        direction = 1.0 if mean_y > 0.01 else (-1.0 if mean_y < -0.01 else 0.0)
        
        return mean_deviation, direction, std_deviation

    def detect_walls(self, scan: LaserScan):
        """Check for walls/obstacles."""
        n = len(scan.ranges)
        angles = scan.angle_min + np.arange(n) * scan.angle_increment
        ranges = np.array(scan.ranges, dtype=float)
        
        # Check for very close obstacles
        close_mask = (ranges > RANGE_MIN) & (ranges < 0.5)
        
        if np.any(close_mask):
            min_dist = np.min(ranges[close_mask])
            return True, min_dist
        
        return False, float('inf')

    def on_scan(self, scan: LaserScan):
        if not self.have_ego_pose or not self.have_yaw:
            return
        
        # Extract points in cone
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
        
        # Transform to world for motion filtering
        Rw = rot2d(self.yaw_e)
        all_points_ego = np.column_stack([xs, ys])
        all_points_world = (Rw @ all_points_ego.T).T + np.array([self.x_e, self.y_e])
        
        # Motion filtering
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
                self.get_logger().info(
                    f'Motion filter: kept {np.sum(moving_mask)}/{len(moving_mask)} moving points',
                    throttle_duration_sec=1.0
                )
        
        self.prev_scan_world_points = all_points_world
        self.prev_scan_timestamp = scan.header.stamp
        
        # Cluster
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
        
        # Evaluate each cluster
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
            self.get_logger().warn(f'No valid target. Best score: {best_score:.2f}', 
                                  throttle_duration_sec=0.5)
            self.handle_no_detection()
            return
        
        # Found target!
        cluster_xs = xs[best_cluster]
        cluster_ys = ys[best_cluster]
        
        self.target_points_ego = np.column_stack([cluster_xs, cluster_ys])
        
        # Calculate centroid
        centroid_ego = np.array([np.mean(cluster_xs), np.mean(cluster_ys)])
        centroid_world = Rw @ centroid_ego + np.array([self.x_e, self.y_e])
        
        self.target_position = centroid_world
        self.track_history.append(centroid_world)
        self.tracking_confidence = min(1.0, self.tracking_confidence + 0.2)
        self.using_ips_fallback = False
        
        # Calculate deviation
        target_distance = np.linalg.norm(centroid_ego)
        mean_dev, dev_dir, dev_std = self.calculate_point_deviation(
            self.target_points_ego, target_distance
        )
        
        self.deviation_history.append(mean_dev)
        self.current_deviation = np.mean(self.deviation_history)
        self.deviation_direction = dev_dir
        
        self.get_logger().info(
            f'Target: score={best_score:.2f}, {best_reason}, '
            f'{len(best_cluster)} pts, dev={self.current_deviation:.3f}m, '
            f'dir={dev_dir:.1f}, conf={self.tracking_confidence:.2f}',
            throttle_duration_sec=0.5
        )
        
        self.publish_visualization(scan.header.stamp)

    def handle_no_detection(self):
        """Fallback when LiDAR loses target."""
        # Try IPS trajectory
        ips_target = self.find_trajectory_point_ahead()
        
        if ips_target is not None:
            self.target_position = np.array(ips_target)
            self.target_points_ego = None
            self.using_ips_fallback = True
            self.tracking_confidence *= 0.9
            
            self.get_logger().info('Using IPS fallback', throttle_duration_sec=1.0)
            self.publish_visualization(self.get_clock().now().to_msg())
        else:
            # Lost completely
            self.tracking_confidence *= 0.7
            if self.tracking_confidence < 0.2:
                self.get_logger().warn('Target lost')
                self.target_position = None
                self.target_points_ego = None
                self.current_deviation = 0.0
            
            self.publish_visualization(self.get_clock().now().to_msg())

    def publish_visualization(self, stamp):
        """
        Visualize deviation: 
        - Green dots = aligned (straight)
        - Yellow dots = deviating (turning)
        - Blue dots = IPS fallback
        """
        ma = MarkerArray()
        
        if self.target_position is None:
            # Clear
            for i in range(NUM_LINE_POINTS + 1):
                m = Marker()
                m.header.frame_id = FRAME_ID
                m.header.stamp = stamp
                m.ns = 'deviation'
                m.id = i
                m.action = Marker.DELETE
                ma.markers.append(m)
            self.marker_pub.publish(ma)
            return
        
        # Draw line from ego to target
        ego_pos = np.array([self.x_e, self.y_e])
        target_pos = self.target_position
        
        for i in range(NUM_LINE_POINTS):
            t = i / (NUM_LINE_POINTS - 1)
            point = ego_pos + t * (target_pos - ego_pos)
            
            m = Marker()
            m.header.frame_id = FRAME_ID
            m.header.stamp = stamp
            m.ns = 'deviation'
            m.id = i
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position.x = float(point[0])
            m.pose.position.y = float(point[1])
            m.pose.position.z = 0.1
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = m.scale.z = DOT_SIZE
            
            # Color based on state
            if self.using_ips_fallback:
                # Blue = using IPS trajectory
                m.color.r, m.color.g, m.color.b = 0.0, 0.5, 1.0
            elif self.current_deviation > DEVIATION_THRESHOLD:
                # Yellow = turning detected
                m.color.r, m.color.g, m.color.b = 1.0, 1.0, 0.0
            else:
                # Green = straight
                m.color.r, m.color.g, m.color.b = 0.0, 1.0, 0.0
            
            m.color.a = self.tracking_confidence
            ma.markers.append(m)
        
        # Add text marker showing deviation
        text = Marker()
        text.header.frame_id = FRAME_ID
        text.header.stamp = stamp
        text.ns = 'deviation'
        text.id = 100
        text.type = Marker.TEXT_VIEW_FACING
        text.action = Marker.ADD
        text.pose.position.x = float(target_pos[0])
        text.pose.position.y = float(target_pos[1])
        text.pose.position.z = 0.5
        text.scale.z = 0.2
        text.color.r = text.color.g = text.color.b = text.color.a = 1.0
        
        if self.using_ips_fallback:
            text.text = 'IPS Fallback'
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
