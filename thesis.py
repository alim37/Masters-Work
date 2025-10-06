#!/usr/bin/env python3
import math
import numpy as np
from collections import deque

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import LaserScan, Imu
from geometry_msgs.msg import Quaternion, Point
from visualization_msgs.msg import Marker

# ---------------- Utils ----------------
def quat_to_yaw(q: Quaternion) -> float:
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)

def rot2d(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=float)

# --------------- Node -------------------
class TargetTrackerNode(Node):
    def __init__(self):
        super().__init__('target_tracker')

        # Parameters
        self.declare_parameter('lidar_topic', '/autodrive/f1tenth_1/lidar')
        self.declare_parameter('ips_topic', '/autodrive/f1tenth_1/ips')
        self.declare_parameter('imu_topic', '/autodrive/f1tenth_1/imu')
        self.declare_parameter('frame_id', 'map')
        
        self.declare_parameter('target_angle_window_deg', 60.0)
        self.declare_parameter('range_min', 0.2)
        self.declare_parameter('range_max', 15.0)
        self.declare_parameter('cluster_max_gap', 0.25)
        self.declare_parameter('cluster_min_size', 4)
        
        # IPS-based trajectory memory
        self.declare_parameter('trajectory_memory_size', 500)  # store last N positions
        self.declare_parameter('trajectory_lookahead', 1.5)    # meters ahead on stored path
        
        # Turn detection
        self.declare_parameter('lateral_deviation_threshold', 0.15)
        self.declare_parameter('turn_rate_smoothing', 0.7)
        
        # Tracking
        self.declare_parameter('max_position_jump', 1.5)
        self.declare_parameter('detection_loss_timeout', 1.0)
        self.declare_parameter('R_pos_std', 0.20)
        self.declare_parameter('Q_accel_std', 1.5)

        # Fetch
        self.lidar_topic = self.get_parameter('lidar_topic').value
        self.ips_topic = self.get_parameter('ips_topic').value
        self.imu_topic = self.get_parameter('imu_topic').value
        self.frame_id = self.get_parameter('frame_id').value
        self.angle_win_rad = math.radians(self.get_parameter('target_angle_window_deg').value)
        self.range_min = self.get_parameter('range_min').value
        self.range_max = self.get_parameter('range_max').value
        self.cluster_max_gap = self.get_parameter('cluster_max_gap').value
        self.cluster_min_size = int(self.get_parameter('cluster_min_size').value)
        
        self.traj_memory_size = int(self.get_parameter('trajectory_memory_size').value)
        self.traj_lookahead = self.get_parameter('trajectory_lookahead').value
        
        self.lateral_dev_threshold = self.get_parameter('lateral_deviation_threshold').value
        self.turn_rate_alpha = self.get_parameter('turn_rate_smoothing').value
        self.max_position_jump = self.get_parameter('max_position_jump').value
        self.loss_timeout = self.get_parameter('detection_loss_timeout').value
        self.R_pos_std = self.get_parameter('R_pos_std').value
        self.Q_accel_std = self.get_parameter('Q_accel_std').value

        # Subscriptions
        qos_lidar = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT, 
                               history=HistoryPolicy.KEEP_LAST)
        self.scan_sub = self.create_subscription(LaserScan, self.lidar_topic, self.on_scan, qos_lidar)
        self.ips_sub = self.create_subscription(Point, self.ips_topic, self.on_point, 10)
        self.imu_sub = self.create_subscription(Imu, self.imu_topic, self.on_imu, 10)

        # Publisher - only marker
        self.marker_pub = self.create_publisher(Marker, 'target_marker', 10)

        # State
        self.have_ego_pose = False
        self.have_yaw = False
        self.x_e = self.y_e = self.yaw_e = 0.0
        self.last_scan_stamp = None

        # Store ego vehicle's trajectory as a reference for target
        self.ego_trajectory = deque(maxlen=self.traj_memory_size)  # [(x, y), ...]
        
        # EKF: X=[x,y,vx,vy]
        self.X = None
        self.P = None
        self.R = np.diag([self.R_pos_std**2, self.R_pos_std**2])
        
        # Turn tracking
        self.estimated_turn_rate = 0.0
        self.estimated_heading = None
        self.is_turning = False
        
        # Detection state
        self.using_ips_fallback = False
        self.last_valid_lidar_time = None
        self.last_lidar_detection = None

        self.get_logger().info('Tracker with IPS trajectory fallback ready.')

    def on_point(self, msg: Point):
        self.x_e, self.y_e = msg.x, msg.y
        self.have_ego_pose = True
        
        # Store ego trajectory
        self.ego_trajectory.append((self.x_e, self.y_e))

    def on_imu(self, msg: Imu):
        self.yaw_e = quat_to_yaw(msg.orientation)
        self.have_yaw = True

    def find_trajectory_point_ahead(self):
        """
        Find a point on ego's stored trajectory that is lookahead distance ahead.
        Assumption: target car is following same path, just ahead of ego.
        """
        if len(self.ego_trajectory) < 10:
            return None
        
        # Convert trajectory to array
        traj = np.array(self.ego_trajectory)
        
        # Current ego position
        ego_pos = np.array([self.x_e, self.y_e])
        
        # Find distances along trajectory from current position
        dists = np.sqrt(np.sum((traj - ego_pos)**2, axis=1))
        
        # Find the closest point on trajectory to current ego position
        closest_idx = np.argmin(dists)
        
        # Walk forward along trajectory from closest point
        cumulative_dist = 0.0
        for i in range(closest_idx, len(traj)):
            if i > 0:
                segment_dist = np.linalg.norm(traj[i] - traj[i-1])
                cumulative_dist += segment_dist
                
                if cumulative_dist >= self.traj_lookahead:
                    # Found point at lookahead distance
                    return traj[i]
        
        # If we don't have enough trajectory ahead, return furthest point
        if len(traj) > closest_idx + 1:
            return traj[-1]
        
        return None

    def detect_turn_from_cluster(self, xs, ys):
        """Analyze lateral spread to detect turning"""
        if len(xs) < 6:
            return False, 0.0, 0.0
        
        cx, cy = np.mean(xs), np.mean(ys)
        heading = math.atan2(cy, cx)
        cos_h, sin_h = math.cos(-heading), math.sin(-heading)
        
        ys_rot = [(xs[i]-cx)*sin_h + (ys[i]-cy)*cos_h for i in range(len(xs))]
        lateral_std = np.std(ys_rot)
        lateral_mean = np.mean(ys_rot)
        
        is_turning = lateral_std > self.lateral_dev_threshold
        turn_dir = 1.0 if lateral_mean > 0.02 else (-1.0 if lateral_mean < -0.02 else 0.0)
        
        return is_turning, lateral_std, turn_dir

    def predict_target_position(self, dt):
        """Predict using curved motion model"""
        if self.X is None:
            return (0.0, 0.0)
        
        x, y, vx, vy = float(self.X[0,0]), float(self.X[1,0]), float(self.X[2,0]), float(self.X[3,0])
        speed = math.hypot(vx, vy)
        heading = self.estimated_heading if self.estimated_heading else math.atan2(vy, vx)
        
        if self.is_turning and abs(self.estimated_turn_rate) > 0.05:
            omega = self.estimated_turn_rate
            if abs(omega) > 1e-3:
                R = speed / omega
                dtheta = omega * dt
                px = x + R * (math.sin(heading + dtheta) - math.sin(heading))
                py = y + R * (-math.cos(heading + dtheta) + math.cos(heading))
                return (px, py)
        
        return (x + vx*dt, y + vy*dt)

    def validate_lidar_detection(self, detection_world):
        """Check if LiDAR detection seems reasonable"""
        if self.X is None:
            return True  # Accept first detection
        
        # Check against EKF prediction
        pred_pos = self.predict_target_position(0.0)
        dist = math.hypot(detection_world[0] - pred_pos[0], detection_world[1] - pred_pos[1])
        
        return dist <= self.max_position_jump

    def on_scan(self, scan: LaserScan):
        if not self.have_ego_pose or not self.have_yaw:
            return

        # Extract points in cone
        n = len(scan.ranges)
        angles = scan.angle_min + np.arange(n) * scan.angle_increment
        ranges = np.array(scan.ranges, dtype=float)
        
        mask = ((angles >= -self.angle_win_rad) & (angles <= self.angle_win_rad) & 
                (ranges > self.range_min) & (ranges < self.range_max))
        
        lidar_has_detection = np.any(mask)
        
        if lidar_has_detection:
            a, r = angles[mask], ranges[mask]
            xs, ys = r * np.cos(a), r * np.sin(a)
            
            # Cluster
            order = np.argsort(a)
            xs, ys = xs[order], ys[order]
            
            clusters = []
            current = [0]
            for i in range(1, len(xs)):
                gap = math.hypot(xs[i]-xs[i-1], ys[i]-ys[i-1])
                if gap <= self.cluster_max_gap:
                    current.append(i)
                else:
                    if len(current) >= self.cluster_min_size:
                        clusters.append(current)
                    current = [i]
            if len(current) >= self.cluster_min_size:
                clusters.append(current)
            
            if clusters:
                # Pick nearest cluster
                best_idx = min(clusters, key=lambda idxs: np.mean(np.hypot(xs[idxs], ys[idxs])))
                cx_ego, cy_ego = float(np.mean(xs[best_idx])), float(np.mean(ys[best_idx]))
                
                # Turn detection
                is_turning, lat_dev, turn_dir = self.detect_turn_from_cluster(xs[best_idx], ys[best_idx])
                self.is_turning = is_turning
                if is_turning:
                    speed = math.hypot(self.X[2,0], self.X[3,0]) if self.X is not None else 1.0
                    range_to_target = math.hypot(cx_ego, cy_ego)
                    raw_turn_rate = turn_dir * lat_dev * 3.0 * speed / max(range_to_target, 0.5)
                    self.estimated_turn_rate = (self.turn_rate_alpha * self.estimated_turn_rate + 
                                               (1-self.turn_rate_alpha) * np.clip(raw_turn_rate, -2.0, 2.0))
                else:
                    self.estimated_turn_rate *= 0.8
                
                # Transform to world
                Rw = rot2d(self.yaw_e)
                c_world = Rw @ np.array([cx_ego, cy_ego]) + np.array([self.x_e, self.y_e])
                
                # Validate LiDAR detection
                if self.validate_lidar_detection(c_world):
                    # GOOD LiDAR detection - use it!
                    self.ekf_update(scan.header.stamp, c_world.reshape(2,1))
                    self.using_ips_fallback = False
                    self.last_valid_lidar_time = self.get_clock().now()
                    self.last_lidar_detection = c_world
                    self.last_scan_stamp = scan.header.stamp
                    self.publish_marker(scan.header.stamp, detection_valid=True, using_ips=False)
                    self.get_logger().info('Using LiDAR detection', throttle_duration_sec=1.0)
                    return
                else:
                    self.get_logger().warn('LiDAR detection invalid - switching to IPS', 
                                          throttle_duration_sec=0.5)
        
        # LiDAR lost or invalid - use IPS trajectory fallback
        ips_target = self.find_trajectory_point_ahead()
        
        if ips_target is not None:
            # Use IPS-based trajectory estimate
            self.ekf_update(scan.header.stamp, np.array([[ips_target[0]], [ips_target[1]]]))
            self.using_ips_fallback = True
            self.last_scan_stamp = scan.header.stamp
            self.publish_marker(scan.header.stamp, detection_valid=True, using_ips=True)
            self.get_logger().info('Using IPS trajectory fallback', throttle_duration_sec=1.0)
        else:
            # No LiDAR and insufficient trajectory - just predict
            self.predict_only(scan.header.stamp)
            self.publish_marker(scan.header.stamp, detection_valid=False, using_ips=False)

    def ekf_update(self, stamp, z):
        now = self.get_clock().now()
        
        if self.X is None:
            self.X = np.array([[z[0,0]], [z[1,0]], [0.0], [0.0]])
            self.P = np.diag([1.0, 1.0, 4.0, 4.0])
            self.last_valid_lidar_time = now
            return
        
        # Predict
        dt = 0.05
        if self.last_scan_stamp:
            dt = max(1e-3, (stamp.sec + stamp.nanosec*1e-9) - 
                          (self.last_scan_stamp.sec + self.last_scan_stamp.nanosec*1e-9))
        
        self.ekf_predict(dt)
        
        # Update with higher noise if using IPS fallback
        R = self.R * (5.0 if self.using_ips_fallback else 1.0)
        
        H = np.array([[1,0,0,0], [0,1,0,0]], dtype=float)
        y = z - H @ self.X
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.X = self.X + K @ y
        self.P = (np.eye(4) - K @ H) @ self.P
        
        # Update heading
        vx, vy = float(self.X[2,0]), float(self.X[3,0])
        if math.hypot(vx, vy) > 0.1:
            new_heading = math.atan2(vy, vx)
            if self.estimated_heading is None:
                self.estimated_heading = new_heading
            else:
                diff = new_heading - self.estimated_heading
                while diff > math.pi: diff -= 2*math.pi
                while diff < -math.pi: diff += 2*math.pi
                self.estimated_heading += 0.3 * diff

    def ekf_predict(self, dt):
        if self.X is None:
            return
        
        F = np.array([[1,0,dt,0], [0,1,0,dt], [0,0,1,0], [0,0,0,1]], dtype=float)
        q = self.Q_accel_std
        Q = np.array([[0.25*dt**4*q**2, 0, 0.5*dt**3*q**2, 0],
                      [0, 0.25*dt**4*q**2, 0, 0.5*dt**3*q**2],
                      [0.5*dt**3*q**2, 0, dt**2*q**2, 0],
                      [0, 0.5*dt**3*q**2, 0, dt**2*q**2]], dtype=float)
        
        self.X = F @ self.X
        self.P = F @ self.P @ F.T + Q

    def predict_only(self, stamp):
        if self.X is None:
            return
        
        dt = 0.05
        if self.last_scan_stamp:
            dt = max(1e-3, (stamp.sec + stamp.nanosec*1e-9) - 
                          (self.last_scan_stamp.sec + self.last_scan_stamp.nanosec*1e-9))
        
        self.ekf_predict(dt)
        
        # Check timeout
        now = self.get_clock().now()
        if self.last_valid_lidar_time:
            dt_loss = (now.nanoseconds - self.last_valid_lidar_time.nanoseconds) * 1e-9
            if dt_loss > self.loss_timeout:
                self.get_logger().warn('Target lost - resetting')
                self.X = None
                self.estimated_heading = None
                self.estimated_turn_rate = 0.0
                return

    def publish_marker(self, stamp, detection_valid, using_ips):
        if self.X is None:
            return
        
        m = Marker()
        m.header.frame_id = self.frame_id
        m.header.stamp = stamp
        m.ns = 'target'
        m.id = 0
        m.type = Marker.SPHERE
        m.action = Marker.ADD
        m.pose.position.x = float(self.X[0,0])
        m.pose.position.y = float(self.X[1,0])
        m.pose.position.z = 0.12
        m.pose.orientation.w = 1.0
        m.scale.x = m.scale.y = m.scale.z = 0.26
        
        # Color coding:
        # Green = LiDAR detection, straight
        # Yellow = LiDAR detection, turning
        # Blue = IPS trajectory fallback
        # Red = Lost/predicting only
        
        if not detection_valid:
            m.color.r, m.color.g, m.color.b = 1.0, 0.0, 0.0  # Red - lost
            m.color.a = 0.5
        elif using_ips:
            m.color.r, m.color.g, m.color.b = 0.0, 0.5, 1.0  # Blue - IPS fallback
            m.color.a = 0.8
        elif self.is_turning:
            m.color.r, m.color.g, m.color.b = 1.0, 1.0, 0.0  # Yellow - turning
            m.color.a = 1.0
        else:
            m.color.r, m.color.g, m.color.b = 0.0, 1.0, 0.0  # Green - good LiDAR
            m.color.a = 1.0
        
        self.marker_pub.publish(m)

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
