#!/bin/bash
set -eo pipefail

ROS_DISTRO=${ROS_DISTRO:-humble}
WORKSPACE=/root/ros2_ws
BASHRC_FILE="/root/.bashrc"
# 定義 LiDAR 設定檔路徑
LIDAR_CONFIG="${WORKSPACE}/install/urg_node2/share/urg_node2/config/params_ether.yaml"

echo "========================================="
echo "   ROS 2 ${ROS_DISTRO} Planning Container"
echo "========================================="

# -------------------------------------------------
# 1. Source System ROS
# -------------------------------------------------
source /opt/ros/${ROS_DISTRO}/setup.bash
if ! grep -Fxq "source /opt/ros/${ROS_DISTRO}/setup.bash" $BASHRC_FILE; then
    echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> $BASHRC_FILE
fi

# -------------------------------------------------
# 2. 智慧編譯檢查 (Dev vs Prod)
# -------------------------------------------------
if [ -d "${WORKSPACE}/src" ]; then
    if [ ! -f "${WORKSPACE}/install/setup.bash" ]; then
        echo ">>> Workspace not built. Building now..."
        cd ${WORKSPACE}
        colcon build --symlink-install
    else
        echo ">>> Workspace already built. Skipping compilation."
    fi

    source ${WORKSPACE}/install/setup.bash
    if ! grep -Fxq "source ${WORKSPACE}/install/setup.bash" $BASHRC_FILE; then
        echo "source ${WORKSPACE}/install/setup.bash" >> $BASHRC_FILE
    fi
fi

# -------------------------------------------------
# 3. 強制修正 LiDAR IP (不論原值為何)
# -------------------------------------------------
# 放在 Source 之後，確保 install 資料夾已存在
if [ -f "$LIDAR_CONFIG" ]; then
    echo ">>> Setting LiDAR IP to 192.168.1.201..."
    sed -i "s/ip_address:.*/ip_address: '192.168.1.201'/g" "$LIDAR_CONFIG"
else
    echo ">>> [Warning] LiDAR config not found at: $LIDAR_CONFIG"
    echo "    Check if urg_node2 is built correctly."
fi

echo "=== Environment ready ==="
exec "$@"