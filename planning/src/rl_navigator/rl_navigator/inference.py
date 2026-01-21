#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rclpy.duration import Duration
import tf2_ros

# ROS 2 訊息格式
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2, Imu
from visualization_msgs.msg import Marker, MarkerArray
import sensor_msgs_py.point_cloud2 as pc2

# Python 標準庫
import numpy as np
import math
import threading
from collections import deque
import sys
import os
import time

# 數學工具
from squaternion import Quaternion
from stable_baselines3 import PPO
from filterpy.kalman import KalmanFilter 

# ====================================================================
# [路徑修正] 自動搜尋 ppo_nn.py
# 解決 ROS 2 colcon build 後找不到同目錄模組的問題
# ====================================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

try:
    # 嘗試導入自定義網路結構
    # 如果你的 ppo_nn.py 定義了 PPOTorchModel，請確保檔案在同目錄
    from ppo_nn import PPOTorchModel
except ImportError:
    try:
        from rl_navigator.ppo_nn import PPOTorchModel
    except ImportError:
        print(f"⚠️ Warning: Could not import PPOTorchModel. If you are using a custom policy, this will fail.")

# =====================
# Constants
# =====================
V_MAX             = 1.0     
W_MAX             = 1.0
TIME_DELTA        = 0.1             
LIDAR_MAX_OBSDIS  = 10.0
FRONT_FOV_RAD     = math.pi       
ROBOT_RADIUS      = 0.6
ELEV_FOV_MIN      = -15 
ELEV_FOV_MAX      = 15  
ORIGINAL_SEGEMNTS = 16      
VERTICAL_LINES    = 180
Z_INGNORE         = 6       
HORIZONTAL        = ORIGINAL_SEGEMNTS - Z_INGNORE  
STATE_DIM         = 11 

# ====================================================================
# [核心邏輯] FlowAidedDynamicTracker
# 負責追蹤動態障礙物並計算 Sim2Real 獎勵/危險係數
# ====================================================================
class FlowAidedDynamicTracker:
    def __init__(self, H, lidar_max=10.0, eps=0.1, min_len=3, z_start=3, z_end=7):
        self.H = H                  
        self.lidar_max = lidar_max  
        self.eps = eps
        self.min_len = min_len
        self.z_start = z_start
        self.z_end = z_end
        self.prev_objs = None

    def segment_1d(self, f_line):
        segs = []
        j = 0 
        H = len(f_line)
        while j < H:
            if f_line[j] < self.lidar_max - self.eps:
                start = j
                base = f_line[j]
                j += 1
                while j < H and abs(f_line[j] - base) < self.eps and f_line[j] < self.lidar_max - self.eps:
                    j += 1
                if j - start >= self.min_len:
                    segs.append((start, j, (start + j - 1) / 2.0, float(np.mean(f_line[start:j]))))
            else: j += 1
        return segs

    def segment_with_xy(self, fmap, fmap_xy):
        if fmap.size == 0: return []
        
        center_rows = fmap[:, self.z_start:self.z_end, 0]
        f_line = np.mean(center_rows, axis=1)
        raw_segs = self.segment_1d(f_line)
        segs = []
        for (start, end, center, dist) in raw_segs:
            sub_fmap = fmap[start:end, self.z_start:self.z_end, 0]
            sub_xy   = fmap_xy[start:end, self.z_start:self.z_end, :]
            mask_valid = (sub_fmap < self.lidar_max - self.eps)
            if not np.any(mask_valid): continue
            segs.append({
                "center_idx": center, 
                "dist": dist, 
                "x": float(sub_xy[..., 0][mask_valid].mean()), 
                "y": float(sub_xy[..., 1][mask_valid].mean())
            })
        return segs

    def track(self, segs, dt, robot_state):
        if not segs: 
            self.prev_objs = []
            return []
        rx, ry, yaw, v_lin = robot_state["odom_x"], robot_state["odom_y"], robot_state["yaw_now"], robot_state["v"]
        v_robot_x, v_robot_y = v_lin * math.cos(yaw), v_lin * math.sin(yaw)
        objs = [{"x": s["x"], "y": s["y"], "dist": s["dist"], "vx": 0.0, "vy": 0.0, "speed": 0.0, "v_r": 0.0, "theta": math.pi/2} for s in segs]
        
        if not self.prev_objs or dt <= 1e-6: 
            self.prev_objs = objs
            return objs
        
        for ob in objs:
            if not self.prev_objs: break
            best = min(self.prev_objs, key=lambda p: (p["x"]-ob["x"])**2 + (p["y"]-ob["y"])**2)
            
            vx_dobs, vy_dobs = (ob["x"] - best["x"])/dt, (ob["y"] - best["y"])/dt
            px_rel, py_rel = rx - ob["x"], ry - ob["y"]
            vx_rel, vy_rel = vx_dobs - v_robot_x, vy_dobs - v_robot_y
            dist_rel, vel_rel = math.hypot(px_rel, py_rel), math.hypot(vx_rel, vy_rel)

            if dist_rel > 1e-6 and vel_rel > 1e-6:
                dot = px_rel * vx_rel + py_rel * vy_rel
                cos_theta = dot / (dist_rel * vel_rel)
                cos_theta = max(-1.0, min(1.0, cos_theta))
                theta = math.acos(cos_theta)
            else:
                theta = math.pi / 2.0

            ob.update({"vx": vx_dobs, "vy": vy_dobs, "speed": math.hypot(vx_dobs, vy_dobs), "v_r": vel_rel, "theta": theta})
        
        self.prev_objs = objs
        return objs

    def compute_flow_reward(self, objs):
        danger_terms = []
        for ob in objs:
            if ob["v_r"] <= 0: continue
            k = 1 + ob["v_r"] * math.exp(1.0/(ob["dist"]+1)) * max(0.0, 1.0 - ob["theta"]/(math.pi/2))
            if ob["dist"] < 6.0: 
                danger_terms.append(np.clip(math.log(max((ob["dist"]-0.8)/k, 0.0001)), -5.0, 0.05))
        return 2.0 * (sum(danger_terms)/len(danger_terms)) if danger_terms else 0.05

    def update(self, fmap, fmap_xy, dt, robot_state):
        segs = self.segment_with_xy(fmap, fmap_xy)
        objs = self.track(segs, dt, robot_state)
        return objs, self.compute_flow_reward(objs)

# ====================================================================
# [主節點] RL Inference Node (ROS 2)
# ====================================================================
class RLInferenceNode(Node):
    def __init__(self):
        super().__init__('rl_inference_node')
        
        # 1. 載入模型 (動態絕對路徑)
        # 指向我們剛剛轉換好的 NumPy 1.x 相容模型
        home_dir = os.path.expanduser("~")
        model_path = os.path.join(home_dir, "ros2_ws/src/models/best_model_1x.zip")
        self.get_logger().info(f"Loading model from {model_path}...")
        
        try:
            # 載入模型 (若有 custom_objects 需傳入)
            if 'PPOTorchModel' in globals():
                 self.model = PPO.load(model_path, custom_objects={"policy_class": PPOTorchModel})
            else:
                 self.model = PPO.load(model_path)
            self.get_logger().info("✅ Model loaded successfully!")
        except Exception as e:
            self.get_logger().error(f"Failed to load model: {e}")
            self.model = None

        # 2. 初始化模組
        self.lock = threading.Lock()
        self.raw = {"odom": None, "imu": None, "lidar": None, "mobile": None}
        
        self.flow_tracker = FlowAidedDynamicTracker(H=VERTICAL_LINES, min_len=4)
        
        self.distance_map = np.full((VERTICAL_LINES, HORIZONTAL, 1), LIDAR_MAX_OBSDIS, np.float32)
        self.fmap_xy = np.zeros((VERTICAL_LINES, HORIZONTAL, 2), np.float32)
        self.lidar_history = deque(maxlen=3)
        for _ in range(3):
            self.lidar_history.append(np.zeros((VERTICAL_LINES, HORIZONTAL, 1), dtype=np.float32))

        # =========================================================
        # [FilterPy] 狀態估計
        # =========================================================
        self.kf = KalmanFilter(dim_x=6, dim_z=6)
        self.kf.x = np.zeros(6)
        self.kf.F = np.eye(6)
        self.kf.H = np.eye(6)
        self.kf.P *= 1.0
        self.kf.Q = np.eye(6) * 0.05
        self.kf.R = np.eye(6) * 0.8
        
        self.odom_x = self.odom_y = self.odom_yaw = 0.0
        self.odom_roll = self.odom_pitch = 0.0
        self.sensor_height = 0.0
        self.imu_xy = 0.0
        self.mobile = np.zeros(2, np.float32)
        self.latest_dyna_reward = 0.0
        
        # 目標點 (測試用)
        self.stage_goal = [(3.0, -0.5, 0.04)] 

        # 3. ROS 2 通訊介面
        # Best Effort 用於感測器以降低延遲
        qos_sensor = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=1)
        qos_reliable = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, depth=10)

        self.create_subscription(Odometry, '/odom', self._cb_odom, qos_reliable)
        self.create_subscription(Imu, '/imu/data', self._cb_imu, qos_sensor)
        self.create_subscription(PointCloud2, '/velodyne_points', self._cb_lidar, qos_sensor)
        self.create_subscription(Odometry, '/mobile_odom', self._cb_mobile, qos_sensor)
        
        self.vel_pub = self.create_publisher(Twist, '/cmd_vel', qos_reliable)
        self.goal_pub = self.create_publisher(MarkerArray, 'goal_marker', qos_reliable)

        # Main Loop Timer
        self.create_timer(TIME_DELTA, self.timer_process_cb)

        self.get_logger().info("RL Inference Node (Full Feature + FilterPy) Started!")

    # ==========================
    # Callbacks
    # ==========================
    def _cb_odom(self, msg):   
        with self.lock: self.raw["odom"] = msg
    def _cb_imu(self, msg):    
        with self.lock: self.raw["imu"] = msg
    def _cb_lidar(self, msg): 
        with self.lock: self.raw["lidar"] = msg
    def _cb_mobile(self, msg):
        with self.lock: self.raw["mobile"] = msg

    # ==========================
    # Main Loop
    # ==========================
    def timer_process_cb(self):
        with self.lock:
            od, imu, lid_msg = self.raw["odom"], self.raw["imu"], self.raw["lidar"]
            mob_msg = self.raw["mobile"]
        
        if not all([od, imu, lid_msg]):
            return

        # 1. IMU KF Update
        q = Quaternion(imu.orientation.w, imu.orientation.x, imu.orientation.y, imu.orientation.z)
        roll, pitch, yaw = q.to_euler()
        
        z_meas = np.array([
            imu.linear_acceleration.z, -imu.linear_acceleration.x, -imu.linear_acceleration.y,
            imu.angular_velocity.x, imu.angular_velocity.y, imu.angular_velocity.z
        ])
        
        self.kf.predict()
        self.kf.update(z_meas)
        
        # 使用 filterpy 標準屬性 (.x)
        self.imu_xy = np.sqrt(self.kf.x[0]**2 + self.kf.x[1]**2)
        self.odom_roll, self.odom_pitch, self.odom_yaw = roll, -pitch, yaw
        self.odom_x, self.odom_y, self.sensor_height = od.pose.pose.position.x, od.pose.pose.position.y, od.pose.pose.position.z

        if mob_msg:
            self.mobile[0] = mob_msg.twist.twist.linear.x
            self.mobile[1] = mob_msg.twist.twist.angular.z

        # 2. LiDAR 處理 (Vectorized)
        self.process_lidar_and_track(lid_msg)

        # 3. 準備 Observation
        obs = self.get_observation()

        # 4. 模型推論
        if self.model:
            action, _ = self.model.predict(obs, deterministic=True)
            
            cmd = Twist()
            cmd.linear.x = float(max(0, action[0] * 0.9))
            cmd.angular.z = float(np.clip(action[1], -W_MAX, W_MAX))
            self.vel_pub.publish(cmd)

    def process_lidar_and_track(self, lid_msg):
        # ROS 2 讀取點雲
        gen = pc2.read_points(lid_msg, skip_nans=True, field_names=("x", "y", "z"))
        pts = np.array(list(gen), dtype=np.float32)
        if pts.size == 0: return

        x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]

        # 向量化計算
        xy_sq = x**2 + y**2
        dists = np.sqrt(xy_sq + z**2)
        betas = np.arctan2(y, x)
        thetas = np.arctan2(z, np.sqrt(xy_sq))

        # FOV Filtering
        mask = (betas >= -FRONT_FOV_RAD/2) & (betas < FRONT_FOV_RAD/2) & (thetas > -3*np.pi/180)
        x, y, dists, betas, thetas = x[mask], y[mask], dists[mask], betas[mask], thetas[mask]

        if dists.size == 0: return

        # Grid Mapping
        j = ((betas + FRONT_FOV_RAD/2) * VERTICAL_LINES / FRONT_FOV_RAD).astype(np.int32)
        k = ((thetas - (ELEV_FOV_MIN*np.pi/180)) * ORIGINAL_SEGEMNTS / ((ELEV_FOV_MAX-ELEV_FOV_MIN)*np.pi/180)).astype(np.int32)
        j, k = np.clip(j, 0, VERTICAL_LINES-1), np.clip(k, 0, ORIGINAL_SEGEMNTS-1)

        d_clear = np.clip(dists - ROBOT_RADIUS, 0.01, LIDAR_MAX_OBSDIS)

        # Lexsort 優化
        idx_sort = np.lexsort((d_clear, k, j))
        j_s, k_s, d_s = j[idx_sort], k[idx_sort], d_clear[idx_sort]
        x_s, y_s = x[idx_sort], y[idx_sort]
        _, first_idx = np.unique(j_s * 100 + k_s, return_index=True)

        fmap = np.full((VERTICAL_LINES, ORIGINAL_SEGEMNTS, 1), LIDAR_MAX_OBSDIS, np.float32)
        f_xy = np.zeros((VERTICAL_LINES, ORIGINAL_SEGEMNTS, 2), np.float32)
        
        fmap[j_s[first_idx], k_s[first_idx], 0] = d_s[first_idx]
        f_xy[j_s[first_idx], k_s[first_idx], 0] = x_s[first_idx]
        f_xy[j_s[first_idx], k_s[first_idx], 1] = y_s[first_idx]

        self.distance_map = fmap[:, Z_INGNORE:, :]
        self.fmap_xy = f_xy[:, Z_INGNORE:, :]
        self.lidar_history.append(self.distance_map.copy())

        # Flow Tracker Update
        robot_state = {"odom_x": self.odom_x, "odom_y": self.odom_y, "yaw_now": self.odom_yaw, "v": self.mobile[0]}
        _, self.latest_dyna_reward = self.flow_tracker.update(self.distance_map, self.fmap_xy, TIME_DELTA, robot_state)

    def get_observation(self):
        skew_x = self.stage_goal[0][0] - self.odom_x
        skew_y = self.stage_goal[0][1] - self.odom_y
        x_local = skew_x * math.cos(-self.odom_yaw) - skew_y * math.sin(-self.odom_yaw)
        y_local = skew_x * math.sin(-self.odom_yaw) + skew_y * math.cos(-self.odom_yaw)
        
        dist_to_goal = float(math.hypot(x_local, y_local))
        beta = math.atan2(y_local, x_local)
        
        state = np.array([
            self.imu_xy, 
            self.kf.x[2], 
            self.kf.x[3], 
            self.kf.x[4], 
            self.odom_roll, 
            self.odom_pitch, 
            dist_to_goal,
            beta, 
            self.stage_goal[0][2] - self.sensor_height, 
            self.mobile[0], 
            self.mobile[1]
        ], dtype=np.float32)

        lidar_stack = np.concatenate(list(self.lidar_history), axis=2)
        
        return {
            "lidar": lidar_stack, 
            "state_current": state
        }

def main(args=None):
    rclpy.init(args=args)
    node = RLInferenceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()