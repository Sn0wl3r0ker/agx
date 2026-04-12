### 2d lidar
主機須先設定ethernet ip:
```bash
if [ "$ARCH" = "aarch64" ]; then
    # [AGX 端]
    echo "Configuring Network for AGX (eno1)..."
    # 檢查 eno1 是否存在，避免報錯
    if ip link show eno1 > /dev/null 2>&1; then
        # 嘗試設定 IP，如果已經存在則忽略錯誤 (|| true)
        ip addr add 192.168.1.100/24 dev eno1 || echo "IP 192.168.1.100 may already be assigned to eno1."
        ip link set eno1 up
    else
        echo "WARNING: Interface eno1 not found!"
    fi

elif [ "$ARCH" = "x86_64" ]; then
    # [PC 端]
    echo "Configuring Network for PC (eno1)..."
    # 檢查 eno1 是否存在
    if ip link show eno1 > /dev/null 2>&1; then
        sudo ip addr add 192.168.1.100/24 dev eno1 || echo "IP 192.168.1.100 may already be assigned to eno1."
        sudo ip link set eno1 up
    else
        echo "WARNING: Interface eno1 not found! Trying to find generic ethernet..."
        # (選用) 如果 PC 不叫 eno1 (例如叫 eno1 或 enp3s0)，可以在這裡加入備用邏輯
    fi
fi
```
### Sensor Bringup (底盤控制 + 雷達)
```bash
ros2 launch car_control car_sensor.launch.py
```
### keyboard control
```bash
ros2 run teleop_twist_keyboard teleop_twist_keyboard
```

### save map
```bash
ros2 run nav2_map_server map_saver_cli -f /root/ros2_ws/src/car_control/config/my_map
```

---

### RealSense D455 (IMU + RGB)
啟動相機並開啟 IMU 融合模式（加速度計 + 陀螺儀合併為單一 `/camera/camera/imu` Topic）：
```bash
ros2 launch realsense2_camera rs_launch.py \
  enable_gyro:=true \
  enable_accel:=true \
  unite_imu_method:=2 \
  enable_sync:=true
```

### EKF 感測器融合 (robot_localization)
融合 `/raw_odom`（輪式里程計 X 速度）與 `/camera/camera/imu`（IMU Yaw 角速度），輸出 `/odometry/filtered` 並發布 `odom → base_footprint` TF：
```bash
ros2 launch car_control ekf.launch.py
```

### SLAM 建圖
```bash
ros2 launch car_control slam_bringup.launch.py
```

### Nav2 導航
```bash
ros2 launch nav2_bringup navigation_launch.py \
  map:=/root/ros2_ws/src/car_control/config/my_map.yaml
```
