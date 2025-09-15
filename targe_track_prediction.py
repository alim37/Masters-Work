#!/usr/bin/env python3
import math
import numpy as np
from collections import deque

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import LaserScan, Imu
from geometry_msgs.msg import PoseStamped, TwistStamped, Pose, Quaternion, Point
from nav_msgs.msg import Path
from visualization_msgs.msg import Marker, MarkerArray

# ---------------- Utils ----------------
def quat_to_yaw(q: Quaternion) -> float:
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)

def rot2d(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s],
                     [s,  c]], dtype=float)

# --------------- Node -------------------
class TargetTrackerNode(Node):
    def __init__(self):
        super().__init__('target_tracker')

        # ------------ Parameters ------------
        self.declare_parameter('lidar_topic', '/autodrive/f1tenth_1/lidar')
        self.declare_parameter('ips_topic',   '/autodrive/f1tenth_1/ips')   # geometry_msgs/Point
        self.declare_parameter('imu_topic',   '/autodrive/f1tenth_1/imu')   # sensor_msgs/Imu
        self.declare_parameter('ips_msg_type','Point')                      # fixed for your setup
        self.declare_parameter('yaw_source',  'imu')                        # 'imu' | 'diff'
        self.declare_parameter('frame_id',    'map')

        # Detection / clustering
        self.declare_parameter('target_angle_window_deg', 40.0)  # +/- deg
        self.declare_parameter('range_min', 0.2)
        self.declare_parameter('range_max', 15.0)
        self.declare_parameter('cluster_max_gap', 0.25)          # meters
        self.declare_parameter('cluster_min_size', 4)            # points

        # EKF noise
        self.declare_parameter('R_pos_std', 0.20)                # meas noise [m]
        self.declare_parameter('Q_accel_std', 1.5)               # process accel std [m/s^2]
        self.declare_parameter('detection_loss_timeout', 1.0)    # seconds

        # Outputs
        self.declare_parameter('path_length', 300)               # history buffer size
        self.declare_parameter('enable_markers', True)
        self.declare_parameter('show_measurement_marker', False) # set True to see raw centroid sphere
        self.declare_parameter('publish_history_path', True)
        self.declare_parameter('publish_prediction_path', True)

        # Prediction horizon
        self.declare_parameter('pred_dt', 0.1)   # seconds per step
        self.declare_parameter('pred_N', 30)     # steps (e.g., 30 @ 0.1s = 3s ahead)

        # Fetch params
        self.lidar_topic = self.get_parameter('lidar_topic').value
        self.ips_topic   = self.get_parameter('ips_topic').value
        self.imu_topic   = self.get_parameter('imu_topic').value
        self.ips_msg_type = self.get_parameter('ips_msg_type').value
        self.yaw_source   = self.get_parameter('yaw_source').value
        self.frame_id     = self.get_parameter('frame_id').value

        self.angle_win_rad = math.radians(self.get_parameter('target_angle_window_deg').value)
        self.range_min = self.get_parameter('range_min').value
        self.range_max = self.get_parameter('range_max').value
        self.cluster_max_gap = self.get_parameter('cluster_max_gap').value
        self.cluster_min_size = int(self.get_parameter('cluster_min_size').value)
        self.R_pos_std = self.get_parameter('R_pos_std').value
        self.Q_accel_std = self.get_parameter('Q_accel_std').value
        self.loss_timeout = self.get_parameter('detection_loss_timeout').value

        self.path_length = int(self.get_parameter('path_length').value)
        self.enable_markers = self.get_parameter('enable_markers').value
        self.show_meas_marker = self.get_parameter('show_measurement_marker').value
        self.publish_history_path = self.get_parameter('publish_history_path').value
        self.publish_prediction_path = self.get_parameter('publish_prediction_path').value

        self.pred_dt = self.get_parameter('pred_dt').value
        self.pred_N  = int(self.get_parameter('pred_N').value)

        # ------------ Subscriptions ----------
        qos_lidar = QoSProfile(depth=10,
                               reliability=ReliabilityPolicy.BEST_EFFORT,
                               history=HistoryPolicy.KEEP_LAST)
        self.scan_sub = self.create_subscription(LaserScan, self.lidar_topic, self.on_scan, qos_lidar)

        if self.ips_msg_type != 'Point':
            raise ValueError("Set -p ips_msg_type:=Point for your setup.")
        self.ips_sub = self.create_subscription(Point, self.ips_topic, self.on_point, 10)

        if self.yaw_source == 'imu':
            self.imu_sub = self.create_subscription(Imu, self.imu_topic, self.on_imu, 10)

        # ------------ Publishers -------------
        self.est_pose_pub  = self.create_publisher(PoseStamped, 'target_estimate', 10)
        self.est_twist_pub = self.create_publisher(TwistStamped, 'target_twist', 10)
        self.path_pub_hist = self.create_publisher(Path, 'target_path', 10)          # history/trail
        self.path_pub_pred = self.create_publisher(Path, 'target_prediction', 10)    # forward horizon
        self.marker_pub    = self.create_publisher(MarkerArray, 'target_markers', 10) if self.enable_markers else None

        # ------------ State ------------------
        self.have_ego_pose = False
        self.have_yaw = (self.yaw_source == 'diff')
        self.x_e = 0.0
        self.y_e = 0.0
        self.yaw_e = 0.0
        self.prev_ips_xy = None

        # LiDAR timing (Point has no header)
        self.last_scan_stamp = None

        # EKF: X=[x,y,vx,vy]^T
        self.X = None
        self.P = None
        self.R = np.diag([self.R_pos_std**2, self.R_pos_std**2])
        self.q_sigma = self.Q_accel_std

        # Paths
        self.hist_buf = deque(maxlen=self.path_length)
        self.path_hist = Path(); self.path_hist.header.frame_id = self.frame_id
        self.path_pred = Path(); self.path_pred.header.frame_id = self.frame_id

        self.last_detection_time = None

        # Republish paths at ~10 Hz
        self.create_timer(0.1, self.republish_paths)

        self.get_logger().info('target_tracker (Point IPS + IMU yaw) with prediction ready.')

    # --------- Callbacks (Ego) -------------
    def on_point(self, msg: Point):
        self.x_e = msg.x
        self.y_e = msg.y
        self.have_ego_pose = True

        if self.yaw_source == 'diff':
            if self.prev_ips_xy is not None:
                dx = self.x_e - self.prev_ips_xy[0]
                dy = self.y_e - self.prev_ips_xy[1]
                if abs(dx) + abs(dy) > 1e-4:
                    self.yaw_e = math.atan2(dy, dx)
                    self.have_yaw = True
            self.prev_ips_xy = (self.x_e, self.y_e)

    def on_imu(self, msg: Imu):
        self.yaw_e = quat_to_yaw(msg.orientation)
        self.have_yaw = True

    # --------- LiDAR handler ---------------
    def on_scan(self, scan: LaserScan):
        if not self.have_ego_pose or (self.yaw_source == 'imu' and not self.have_yaw):
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

        # Simple adjacent-point clustering
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

        # Pick nearest cluster by mean range
        best_idx = None
        best_mean = float('inf')
        for idxs in clusters:
            mean_r = float(np.mean(r[idxs]))
            if mean_r < best_mean:
                best_mean = mean_r
                best_idx = idxs

        cx_ego = float(np.mean(xs[best_idx]))
        cy_ego = float(np.mean(ys[best_idx]))

        # Ego->World
        Rw = rot2d(self.yaw_e)
        c_world = Rw @ np.array([cx_ego, cy_ego]) + np.array([self.x_e, self.y_e])
        z = c_world.reshape(2, 1)

        self.ekf_update(scan.header.stamp, z)
        self.last_scan_stamp = scan.header.stamp

        if self.marker_pub and self.enable_markers:
            self.publish_markers(scan.header.stamp, c_world)

    # ------------- EKF ---------------------
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
            self.publish_outputs(stamp)
            return

        # dt from LiDAR stamps
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
        self.publish_outputs(stamp)

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
                self.get_logger().warn(f'Lost target for {dt_loss:.2f}s — resetting EKF.')
                self.X, self.P = None, None
                return

        self.publish_outputs(stamp)

    # ------------- Outputs -----------------
    def publish_outputs(self, stamp):
        if self.X is None:
            return
        x, y, vx, vy = (float(self.X[0,0]), float(self.X[1,0]),
                        float(self.X[2,0]), float(self.X[3,0]))

        # Pose & twist (estimate)
        ps = PoseStamped()
        ps.header.stamp = stamp
        ps.header.frame_id = self.frame_id
        ps.pose.position.x = x
        ps.pose.position.y = y
        ps.pose.position.z = 0.0
        ps.pose.orientation.w = 1.0
        self.est_pose_pub.publish(ps)

        tw = TwistStamped()
        tw.header = ps.header
        tw.twist.linear.x = vx
        tw.twist.linear.y = vy
        self.est_twist_pub.publish(tw)

        # History path
        if self.publish_history_path:
            self.hist_buf.append(ps)
            self.path_hist.header.stamp = self.get_clock().now().to_msg()
            self.path_hist.header.frame_id = self.frame_id
            self.path_hist.poses = list(self.hist_buf)
            self.path_pub_hist.publish(self.path_hist)

        # Predicted (forward) path
        if self.publish_prediction_path:
            self.publish_prediction(stamp)

    def publish_prediction(self, stamp):
        """Roll out a constant-velocity horizon from EKF state."""
        if self.X is None:
            return
        x, y, vx, vy = float(self.X[0,0]), float(self.X[1,0]), float(self.X[2,0]), float(self.X[3,0])
        dt = max(1e-3, float(self.pred_dt))
        N = max(1, int(self.pred_N))

        poses = []
        px, py = x, y
        for k in range(1, N+1):
            t = k * dt
            # CV model (no curvature). If you later estimate heading rate, you can arc here.
            px = x + vx * t
            py = y + vy * t

            p = PoseStamped()
            p.header.stamp = stamp  # same origin stamp; visualization is fine
            p.header.frame_id = self.frame_id
            p.pose.position.x = px
            p.pose.position.y = py
            p.pose.position.z = 0.0
            p.pose.orientation.w = 1.0
            poses.append(p)

        self.path_pred.header.stamp = self.get_clock().now().to_msg()
        self.path_pred.header.frame_id = self.frame_id
        self.path_pred.poses = poses
        self.path_pub_pred.publish(self.path_pred)

    def republish_paths(self):
        """Timer to keep paths fresh in RViz."""
        if self.publish_history_path and self.hist_buf:
            self.path_hist.header.stamp = self.get_clock().now().to_msg()
            self.path_pub_hist.publish(self.path_hist)
        if self.publish_prediction_path and self.X is not None:
            # recompute using current state to keep smooth
            self.publish_prediction(self.get_clock().now().to_msg())

    def publish_markers(self, stamp, c_world):
        ma = MarkerArray()

        # (optional) Detected centroid (measurement)
        if self.show_meas_marker:
            m = Marker()
            m.header.frame_id = self.frame_id
            m.header.stamp = stamp
            m.ns = 'target_tracker'
            m.id = 0
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position.x = float(c_world[0])
            m.pose.position.y = float(c_world[1])
            m.pose.position.z = 0.1
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = m.scale.z = 0.18
            m.color.a = 1.0
            ma.markers.append(m)

        # Estimated position (EKF) — single ball
        if self.X is not None:
            me = Marker()
            me.header.frame_id = self.frame_id
            me.header.stamp = stamp
            me.ns = 'target_tracker'
            me.id = 1
            me.type = Marker.SPHERE
            me.action = Marker.ADD
            me.pose.position.x = float(self.X[0,0])
            me.pose.position.y = float(self.X[1,0])
            me.pose.position.z = 0.12
            me.pose.orientation.w = 1.0
            me.scale.x = me.scale.y = me.scale.z = 0.26
            me.color.a = 1.0
            ma.markers.append(me)

        self.marker_pub.publish(ma)

# --------------- Main --------------------
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
