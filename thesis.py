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

ANGLE_WINDOW_DEG = 60.0
RANGE_MIN = 0.2
RANGE_MAX = 15.0
CLUSTER_MAX_GAP = 0.25
CLUSTER_MIN_SIZE = 4

TRAJECTORY_MEMORY_SIZE = 500
TRAJECTORY_LOOKAHEAD = 1.5  

LATERAL_DEV_THRESHOLD = 0.15
TURN_RATE_SMOOTHING = 0.7

MAX_POSITION_JUMP = 1.5
DETECTION_LOSS_TIMEOUT = 1.0
R_POS_STD = 0.20
Q_ACCEL_STD = 1.5

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

        qos_lidar = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST)
        self.scan_sub = self.create_subscription(LaserScan, '/autodrive/f1tenth_1/lidar', self.on_scan, qos_lidar)
        self.ips_sub = self.create_subscription(Point, '/autodrive/f1tenth_1/imu', self.on_point, 10)
        self.imu_sub = self.create_subscription(Imu, '/autodrive/f1tenth_1/ips', self.on_imu, 10)

        self.marker_pub = self.create_publisher(Marker, 'target_marker', 10)

        self.have_ego_pose = False
        self.have_yaw = False
        self.x_e = self.y_e = self.yaw_e = 0.0
        self.last_scan_stamp = None

        self.ego_trajectory = deque(maxlen=TRAJECTORY_MEMORY_SIZE)
        
        self.X = None
        self.P = None
        self.R = np.diag([R_POS_STD**2, R_POS_STD**2])
        
        self.estimated_turn_rate = 0.0
        self.estimated_heading = None
        self.is_turning = False
        
        self.using_ips_fallback = False
        self.last_valid_lidar_time = None

        self.get_logger().info('Target tracker ready (LiDAR + IPS fallback)')

    def on_point(self, msg: Point):
        self.x_e, self.y_e = msg.x, msg.y
        self.have_ego_pose = True
        self.ego_trajectory.append((self.x_e, self.y_e))

    def on_imu(self, msg: Imu):
        self.yaw_e = quat_to_yaw(msg.orientation)
        self.have_yaw = True

    def find_trajectory_point_ahead(self):
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

    def detect_turn_from_cluster(self, xs, ys):
        if len(xs) < 6:
            return False, 0.0, 0.0
        
        cx, cy = np.mean(xs), np.mean(ys)
        heading = math.atan2(cy, cx)
        cos_h, sin_h = math.cos(-heading), math.sin(-heading)
        
        ys_rot = [(xs[i]-cx)*sin_h + (ys[i]-cy)*cos_h for i in range(len(xs))]
        lateral_std = np.std(ys_rot)
        lateral_mean = np.mean(ys_rot)
        
        is_turning = lateral_std > LATERAL_DEV_THRESHOLD
        turn_dir = 1.0 if lateral_mean > 0.02 else (-1.0 if lateral_mean < -0.02 else 0.0)
        
        return is_turning, lateral_std, turn_dir

    def predict_target_position(self, dt):
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
        if self.X is None:
            return True
        
        pred_pos = self.predict_target_position(0.0)
        dist = math.hypot(detection_world[0] - pred_pos[0], detection_world[1] - pred_pos[1])
        
        return dist <= MAX_POSITION_JUMP

    def on_scan(self, scan: LaserScan):
        if not self.have_ego_pose or not self.have_yaw:
            return

        n = len(scan.ranges)
        angle_win_rad = math.radians(ANGLE_WINDOW_DEG)
        angles = scan.angle_min + np.arange(n) * scan.angle_increment
        ranges = np.array(scan.ranges, dtype=float)
        
        mask = ((angles >= -angle_win_rad) & (angles <= angle_win_rad) & 
                (ranges > RANGE_MIN) & (ranges < RANGE_MAX))
        
        lidar_has_detection = np.any(mask)
        
        if lidar_has_detection:
            a, r = angles[mask], ranges[mask]
            xs, ys = r * np.cos(a), r * np.sin(a)
            
            order = np.argsort(a)
            xs, ys = xs[order], ys[order]
            
            clusters = []
            current = [0]
            for i in range(1, len(xs)):
                gap = math.hypot(xs[i]-xs[i-1], ys[i]-ys[i-1])
                if gap <= CLUSTER_MAX_GAP:
                    current.append(i)
                else:
                    if len(current) >= CLUSTER_MIN_SIZE:
                        clusters.append(current)
                    current = [i]
            if len(current) >= CLUSTER_MIN_SIZE:
                clusters.append(current)
            
            if clusters:
                best_idx = min(clusters, key=lambda idxs: np.mean(np.hypot(xs[idxs], ys[idxs])))
                cx_ego, cy_ego = float(np.mean(xs[best_idx])), float(np.mean(ys[best_idx]))
                
                is_turning, lat_dev, turn_dir = self.detect_turn_from_cluster(xs[best_idx], ys[best_idx])
                self.is_turning = is_turning
                if is_turning:
                    speed = math.hypot(self.X[2,0], self.X[3,0]) if self.X is not None else 1.0
                    range_to_target = math.hypot(cx_ego, cy_ego)
                    raw_turn_rate = turn_dir * lat_dev * 3.0 * speed / max(range_to_target, 0.5)
                    self.estimated_turn_rate = (TURN_RATE_SMOOTHING * self.estimated_turn_rate + (1-TURN_RATE_SMOOTHING) * np.clip(raw_turn_rate, -2.0, 2.0))
                else:
                    self.estimated_turn_rate *= 0.8
                
                Rw = rot2d(self.yaw_e)
                c_world = Rw @ np.array([cx_ego, cy_ego]) + np.array([self.x_e, self.y_e])
                
                if self.validate_lidar_detection(c_world):
                    self.ekf_update(scan.header.stamp, c_world.reshape(2,1))
                    self.using_ips_fallback = False
                    self.last_valid_lidar_time = self.get_clock().now()
                    self.last_scan_stamp = scan.header.stamp
                    self.publish_marker(scan.header.stamp, detection_valid=True, using_ips=False)
                    self.get_logger().info('LiDAR tracking', throttle_duration_sec=1.0)
                    return
                else:
                    self.get_logger().warn('LiDAR invalid -> IPS fallback', throttle_duration_sec=0.5)
        
        ips_target = self.find_trajectory_point_ahead()
        
        if ips_target is not None:
            self.ekf_update(scan.header.stamp, np.array([[ips_target[0]], [ips_target[1]]]))
            self.using_ips_fallback = True
            self.last_scan_stamp = scan.header.stamp
            self.publish_marker(scan.header.stamp, detection_valid=True, using_ips=True)
            self.get_logger().info('IPS fallback tracking', throttle_duration_sec=1.0)
        else:
            self.predict_only(scan.header.stamp)
            self.publish_marker(scan.header.stamp, detection_valid=False, using_ips=False)

    def ekf_update(self, stamp, z):
        now = self.get_clock().now()
        
        if self.X is None:
            self.X = np.array([[z[0,0]], [z[1,0]], [0.0], [0.0]])
            self.P = np.diag([1.0, 1.0, 4.0, 4.0])
            self.last_valid_lidar_time = now
            return
        
        dt = 0.05
        if self.last_scan_stamp:
            dt = max(1e-3, (stamp.sec + stamp.nanosec*1e-9) - (self.last_scan_stamp.sec + self.last_scan_stamp.nanosec*1e-9))
        
        self.ekf_predict(dt)
        
        R = self.R * (5.0 if self.using_ips_fallback else 1.0)
        
        H = np.array([[1,0,0,0], [0,1,0,0]], dtype=float)
        y = z - H @ self.X
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.X = self.X + K @ y
        self.P = (np.eye(4) - K @ H) @ self.P
        
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
        q = Q_ACCEL_STD
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
        
        now = self.get_clock().now()
        if self.last_valid_lidar_time:
            dt_loss = (now.nanoseconds - self.last_valid_lidar_time.nanoseconds) * 1e-9
            if dt_loss > DETECTION_LOSS_TIMEOUT:
                self.get_logger().warn('Target lost - resetting')
                self.X = None
                self.estimated_heading = None
                self.estimated_turn_rate = 0.0
                return

    def publish_marker(self, stamp, detection_valid, using_ips):
        if self.X is None:
            return
        
        m = Marker()
        m.header.frame_id = 'map'
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
        
        if not detection_valid:
            m.color.r, m.color.g, m.color.b = 1.0, 0.0, 0.0
            m.color.a = 0.5
        elif using_ips:
            m.color.r, m.color.g, m.color.b = 0.0, 0.5, 1.0
            m.color.a = 0.8
        elif self.is_turning:
            m.color.r, m.color.g, m.color.b = 1.0, 1.0, 0.0
            m.color.a = 1.0
        else:
            m.color.r, m.color.g, m.color.b = 0.0, 1.0, 0.0
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
