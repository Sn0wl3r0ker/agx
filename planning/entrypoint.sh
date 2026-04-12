#!/bin/bash
set -eo pipefail

ROS_DISTRO=${ROS_DISTRO:-humble}
WORKSPACE=/root/ros2_ws
BASHRC_FILE="/root/.bashrc"
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
    if [ ! -f "${WORKSPACE}/install/setup.bash" ] || \
       { [ -f "${WORKSPACE}/src/vla_demo/package.xml" ] && [ ! -d "${WORKSPACE}/install/vla_demo" ]; }; then
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
# NOTE: LiDAR 參數 (如 ip_address) 已透過 ROS 2 Launch File 覆寫注入
#       參見 car_control/config/urg_node2_override.yaml
#       不再需要在此處用 sed 修改第三方套件的設定檔
# -------------------------------------------------

echo "=== Environment ready ==="
exec "$@"
