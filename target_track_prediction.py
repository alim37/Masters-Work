#!/usr/bin/env python3
import math
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import LaserScan, Imu
from geometry_msgs.msg import Point, Quaternion
from visualization_msgs.msg import Marker, MarkerArray

def quat_to_yaw(q: Quaternion) -> float:
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)

def rot2d(theta: float):
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s],
                     [s,  c]], dtype=float)

class TargetTrackerNode(Node):
    def __init__(self):
        super().__init__('target_tracker')

        self.declare_parameter('lidar_topic', '/autodrive/f1tenth_1/lidar')
        self.declare_parameter('ips_topic',   '/autodrive/f1tenth_1/ips')   # geometry_msgs/Point
        self.declare_parameter('imu_topic',   '/autodrive/f1tenth_1/imu')   # sensor_msgs/Imu
        self.declare_parameter('frame_id',    'map')

        self.declare_parameter('target_angle_window_deg', 40.0)  # +/- deg
        self.declare_parameter('range_min', 0.2)
        self.declare_parameter('range_max', 15.0)
        self.declare_parameter('cluster_max_gap', 0.25)          # meters
        self.declare_parameter('cluster_min_size', 4)            # points

        self.declare_parameter('R_pos_std', 0.20)                # meas noise [m]
        self.declare_parameter('Q_accel_std', 1.5)               # process accel std [m/s^2]
        self.declare_parameter('detection_loss_timeout', 1.0)    # seconds

        # Fetch params
        self.lidar_topic = self.get_parameter('lidar_topic').value
        self.ips_topic   = self.get_parameter('ips_topic').value
        self.imu_topic   = self.get_parameter('imu_topic').value
        self.frame_id    = self.get_parameter('frame_id').value

        self.angle_win_rad = math.radians(self.get_parameter('target_angle_window_deg').value)
        self.range_min = self.get_parameter('range_min').value
        self.range_max = self.get_parameter('range_max').value
        self.cluster_max_gap = self.get_parameter('cluster_max_gap').value
        self.cluster_min_size = int(self.get_parameter('cluster_min_size').value)
        self.R_pos_std = self.get_parameter('R_pos_std').value
        self.Q_accel_std = self.get_parameter('Q_accel_std').value
        self.loss_timeout = self.get_parameter('detection_loss_timeout').value

        qos_lidar = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST)
        self.scan_sub = self.create_subscription(LaserScan, self.lidar_topic, self.on_scan, qos_lidar)
        self.ips_sub  = self.create_subscription(Point,      self.ips_topic,  self.on_point, 10)
        self.imu_sub  = self.create_subscription(Imu,        self.imu_topic,  self.on_imu,   10)

        self.marker_pub = self.create_publisher(MarkerArray, 'target_marker', 10)

        self.have_ego_pose = False
        self.have_yaw = False
        self.x_e = 0.0
        self.y_e = 0.0
        self.yaw_e = 0.0

        self.last_scan_stamp = None

        self.X = None
        self.P = None
        self.R = np.diag([self.R_pos_std**2, self.R_pos_std**2])
        self.q_sigma = self.Q_accel_std
        self.last_detection_time = None

        self.get_logger().info('target_tracker: LiDAR detection with single ball marker ready.')

    def on_point(self, msg: Point):
        self.x_e = msg.x
        self.y_e = msg.y
        self.have_ego_pose = True

    def on_imu(self, msg: Imu):
        self.yaw_e = quat_to_yaw(msg.orientation)
        self.have_yaw = True

    def on_scan(self, scan: LaserScan):
        if not (self.have_ego_pose and self.have_yaw):
            return

        n = len(scan.ranges)
        angles = scan.angle_min + np.arange(n) * scan.angle_increment
        ranges = np.array(scan.ranges, dtype=float)

        forward_mask = (angles >= -self.angle_win_rad) & (angles <= self.angle_win_rad)
        valid = (ranges > self.range_min) & (ranges < self.range_max) & forward_mask

        if not np.any(valid):
            self.predict_only(scan.header.stamp)
            self.last_scan_stamp = scan.header.stamp
            return

        a = angles[valid]
        r = ranges[valid]
        xs = r * np.cos(a)
        ys = r * np.sin(a)

        order = np.argsort(a)
        xs, ys, r, a = xs[order], ys[order], r[order], a[order]

        clusters = []
        current = [0]
        for i in range(1, len(xs)):
            gap = math.hypot(xs[i] - xs[i-1], ys[i] - ys[i-1])
            if gap <= self.cluster_max_gap:
                current.append(i)
            else:
                if len(current) >= self.cluster_min_size:
                    clusters.append(current)
                current = [i]
        if len(current) >= self.cluster_min_size:
            clusters.append(current)

        if not clusters:
            self.predict_only(scan.header.stamp)
            self.last_scan_stamp = scan.header.stamp
            return

        best_idx = None
        best_mean = float('inf')
        for idxs in clusters:
            mean_r = float(np.mean(r[idxs]))
            if mean_r < best_mean:
                best_mean = mean_r
                best_idx = idxs

        cx_ego = float(np.mean(xs[best_idx]))
        cy_ego = float(np.mean(ys[best_idx]))

        Rw = rot2d(self.yaw_e)
        c_world = Rw @ np.array([cx_ego, cy_ego]) + np.array([self.x_e, self.y_e])
        z = c_world.reshape(2, 1)

        self.ekf_update(scan.header.stamp, z)
        self.last_scan_stamp = scan.header.stamp
        self.publish_marker(scan.header.stamp)

    def ekf_predict(self, dt: float):
        if self.X is None:
            return
        F = np.array([[1, 0, dt, 0],
                      [0, 1, 0, dt],
                      [0, 0, 1,  0],
                      [0, 0, 0,  1]], dtype=float)
        q = self.q_sigma
        q11 = 0.25 * dt**4 * q**2
        q13 = 0.5  * dt**3 * q**2
        q33 =       dt**2 * q**2
        Q = np.array([[q11, 0.0, q13, 0.0],
                      [0.0, q11, 0.0, q13],
                      [q13, 0.0, q33, 0.0],
                      [0.0, q13, 0.0, q33]], dtype=float)
        self.X = F @ self.X
        self.P = F @ self.P @ F.T + Q

    def ekf_update(self, stamp, z: np.ndarray):
        now = self.get_clock().now()

        if self.X is None:
            self.X = np.zeros((4, 1))
            self.X[0, 0], self.X[1, 0] = z[0, 0], z[1, 0]
            self.P = np.diag([1.0, 1.0, 4.0, 4.0])
            self.last_detection_time = now
            return

        if self.last_scan_stamp is not None:
            t_prev = self.last_scan_stamp.sec + self.last_scan_stamp.nanosec * 1e-9
            t_now  = stamp.sec + stamp.nanosec * 1e-9
            dt = max(1e-3, t_now - t_prev)
        else:
            dt = 0.05

        self.ekf_predict(dt)

        H = np.array([[1,0,0,0],
                      [0,1,0,0]], dtype=float)
        y = z - H @ self.X
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.X = self.X + K @ y
        I = np.eye(4)
        self.P = (I - K @ H) @ self.P

        self.last_detection_time = now

    def predict_only(self, stamp):
        if self.X is None:
            return
        if self.last_scan_stamp is not None:
            t_prev = self.last_scan_stamp.sec + self.last_scan_stamp.nanosec * 1e-9
            t_now  = stamp.sec + stamp.nanosec * 1e-9
            dt = max(1e-3, t_now - t_prev)
        else:
            dt = 0.05

        self.ekf_predict(dt)

        now = self.get_clock().now()
        if self.last_detection_time is not None:
            dt_loss = (now.nanoseconds - self.last_detection_time.nanoseconds) * 1e-9
            if dt_loss > self.loss_timeout:
                self.get_logger().warn(f'Lost target for {dt_loss:.2f}s — resetting.')
                self.X, self.P = None, None

        self.publish_marker(stamp)

    def publish_marker(self, stamp):
        if self.X is None:
            return
        ma = MarkerArray()
        m = Marker()
        m.header.frame_id = self.frame_id
        m.header.stamp = stamp
        m.ns = 'target_tracker'
        m.id = 1
        m.type = Marker.SPHERE
        m.action = Marker.ADD
        m.pose.position.x = float(self.X[0,0])
        m.pose.position.y = float(self.X[1,0])
        m.pose.position.z = 0.12
        m.pose.orientation.w = 1.0
        m.scale.x = m.scale.y = m.scale.z = 0.26
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
