#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
odom_recorder_node.py
=====================
ROS 2 節點：以 10 Hz 固定頻率同步擷取三種定位來源，並將對齊好的資料
即時寫入 CSV 檔案，供後續離線分析使用。

三種來源：
  1. /raw_odom           (nav_msgs/Odometry)  — 純輪速里程計
  2. /odometry/filtered  (nav_msgs/Odometry)  — EKF 融合里程計
  3. TF: map → base_link                      — SLAM Ground Truth

執行方式：
  ros2 run car_control odom_recorder_node.py
  # 或
  python3 odom_recorder_node.py
"""

import math
import csv
import os

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from nav_msgs.msg import Odometry
import tf2_ros


# ────────────────────────────────────────────────────────────
#  工具函式
# ────────────────────────────────────────────────────────────
def quaternion_to_yaw(q) -> float:
    """從四元數 (x, y, z, w) 提取 Yaw（繞 Z 軸旋轉角），單位 rad。"""
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


# ────────────────────────────────────────────────────────────
#  節點
# ────────────────────────────────────────────────────────────
class OdomRecorderNode(Node):

    def __init__(self):
        super().__init__('odom_recorder_node')

        # ---- ROS 參數 ----
        self.declare_parameter('output_csv', '~/trajectory_data.csv')
        csv_path: str = self.get_parameter('output_csv').get_parameter_value().string_value
        csv_path = os.path.expanduser(csv_path)
        self._csv_path = csv_path

        self.declare_parameter('timer_hz', 10.0)
        timer_hz: float = self.get_parameter('timer_hz').get_parameter_value().double_value

        # ---- 快取最新訊息 ----
        self._latest_raw_odom: Odometry | None = None
        self._latest_ekf_odom: Odometry | None = None

        # ---- QoS (兼容各種發布端) ----
        odom_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )

        # ---- 訂閱 ----
        self.create_subscription(
            Odometry, '/raw_odom', self._raw_odom_cb, odom_qos)
        self.create_subscription(
            Odometry, '/odometry/filtered', self._ekf_odom_cb, odom_qos)

        # ---- TF Listener ----
        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

        # ---- CSV 檔案 ----
        self._csv_file = open(csv_path, mode='w', newline='')
        self._csv_writer = csv.writer(self._csv_file)
        self._csv_writer.writerow([
            'timestamp',
            'raw_x', 'raw_y', 'raw_yaw',
            'ekf_x', 'ekf_y', 'ekf_yaw',
            'gt_x',  'gt_y',  'gt_yaw',
        ])
        self._csv_file.flush()

        # ---- 定時器 ----
        timer_period = 1.0 / timer_hz
        self.create_timer(timer_period, self._timer_callback)

        self.get_logger().info(
            f'OdomRecorder 啟動: {timer_hz:.0f} Hz, 輸出 → {csv_path}')

    # ============================================================
    #  回呼函數
    # ============================================================
    def _raw_odom_cb(self, msg: Odometry):
        self._latest_raw_odom = msg

    def _ekf_odom_cb(self, msg: Odometry):
        self._latest_ekf_odom = msg

    def _timer_callback(self):
        """10 Hz 定時擷取 + 寫入 CSV。"""
        now = self.get_clock().now()

        # --- Raw Odom ---
        if self._latest_raw_odom is not None:
            p = self._latest_raw_odom.pose.pose
            raw_x = p.position.x
            raw_y = p.position.y
            raw_yaw = quaternion_to_yaw(p.orientation)
        else:
            raw_x = raw_y = raw_yaw = float('nan')

        # --- EKF Odom ---
        if self._latest_ekf_odom is not None:
            p = self._latest_ekf_odom.pose.pose
            ekf_x = p.position.x
            ekf_y = p.position.y
            ekf_yaw = quaternion_to_yaw(p.orientation)
        else:
            ekf_x = ekf_y = ekf_yaw = float('nan')

        # --- Ground Truth (TF: map → base_link) ---
        try:
            tf_stamped = self._tf_buffer.lookup_transform(
                'map', 'base_link', rclpy.time.Time())
            t = tf_stamped.transform.translation
            gt_x = t.x
            gt_y = t.y
            gt_yaw = quaternion_to_yaw(tf_stamped.transform.rotation)
        except (tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException):
            gt_x = gt_y = gt_yaw = float('nan')

        # --- 寫入 CSV ---
        stamp_sec = now.nanoseconds * 1e-9
        self._csv_writer.writerow([
            f'{stamp_sec:.6f}',
            f'{raw_x:.6f}', f'{raw_y:.6f}', f'{raw_yaw:.6f}',
            f'{ekf_x:.6f}', f'{ekf_y:.6f}', f'{ekf_yaw:.6f}',
            f'{gt_x:.6f}',  f'{gt_y:.6f}',  f'{gt_yaw:.6f}',
        ])
        self._csv_file.flush()

    # ============================================================
    #  清理
    # ============================================================
    def destroy_node(self):
        self._csv_file.close()
        self.get_logger().info(f'CSV 已儲存 → {self._csv_path}')
        super().destroy_node()


# ────────────────────────────────────────────────────────────
#  Entry point
# ────────────────────────────────────────────────────────────
def main(args=None):
    rclpy.init(args=args)
    node = OdomRecorderNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
