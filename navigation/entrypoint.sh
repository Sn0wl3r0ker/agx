#!/bin/bash

# 如果任何命令出錯，不立即退出，但保留錯誤提示
set -eo pipefail

ROS_DISTRO=noetic
WORKSPACE=/root

echo "========================================="
echo "   ROS ${ROS_DISTRO} Development Container"
echo "========================================="

# -------------------------------------------------
# Source ROS
# -------------------------------------------------
if [ -f "/opt/ros/${ROS_DISTRO}/setup.bash" ]; then
    source /opt/ros/${ROS_DISTRO}/setup.bash
fi

# -------------------------------------------------
# Build HDL Workspace (optional)
# -------------------------------------------------
HDL_WS=${WORKSPACE}/hdl_ws

if [ -d "${HDL_WS}/src" ]; then
    echo "=== Checking hdl_ws ==="
    if [ ! -f "${HDL_WS}/devel/setup.bash" ]; then
        echo ">>> hdl_ws not built. Building now..."
        cd ${HDL_WS}
        catkin_make -j$(nproc) || echo "!!! HDL build failed, continuing..."
    else
        echo ">>> hdl_ws already built. Skipping build."
    fi

    # 允許即使 devel/setup.bash 不存在也不中斷
    if [ -f "${HDL_WS}/devel/setup.bash" ]; then
        source ${HDL_WS}/devel/setup.bash
    fi
else
    echo "!!! WARNING: hdl_ws/src not found. Skipping."
fi

# -------------------------------------------------
# Build LiDAR Workspace (isolated, optional)
# -------------------------------------------------
LIDAR_WS=${WORKSPACE}/lidar_ws

if [ -d "${LIDAR_WS}/src" ]; then
    echo "=== Checking lidar_ws ==="
    if [ ! -f "${LIDAR_WS}/devel_isolated/setup.bash" ]; then
        echo ">>> lidar_ws not built. Building now..."
        cd ${LIDAR_WS}
        catkin_make_isolated -j$(nproc) || echo "!!! LiDAR build failed, continuing..."
    else
        echo ">>> lidar_ws already built. Skipping build."
    fi

    if [ -f "${LIDAR_WS}/devel_isolated/setup.bash" ]; then
        source ${LIDAR_WS}/devel_isolated/setup.bash
    fi
else
    echo "!!! WARNING: lidar_ws/src not found. Skipping."
fi

# -------------------------------------------------
# Execute passed command or fallback to bash
# -------------------------------------------------
echo "=== Environment ready ==="

# Drop to shell or execute passed command
if [ $# -gt 0 ]; then
    exec "$@"
else
    # 沒有傳入命令時保持容器活著
    exec bash -c "while true; do sleep 1000; done"
fi
