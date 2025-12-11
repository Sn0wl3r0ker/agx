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
# ROS 2 使用 install 資料夾
if [ -d "${WORKSPACE}/src" ]; then
    if [ ! -f "${WORKSPACE}/install/setup.bash" ]; then
        echo ">>> Workspace not built (Dev Mode detected). Building now..."
        cd ${WORKSPACE}
        # 使用 colcon 編譯
        colcon build --symlink-install
    else
        echo ">>> Workspace already built (Production Mode). Skipping compilation."
    fi

    # Source Workspace
    source ${WORKSPACE}/install/setup.bash
    if ! grep -Fxq "source ${WORKSPACE}/install/setup.bash" $BASHRC_FILE; then
        echo "source ${WORKSPACE}/install/setup.bash" >> $BASHRC_FILE
    fi
fi

echo "=== Environment ready ==="
exec "$@"