#!/bin/bash

# 如果任何命令出錯，立即退出
set -eo pipefail

ROS_DISTRO=noetic
WORKSPACE=/root

echo "========================================="
echo "   ROS ${ROS_DISTRO} Development Container"
echo "========================================="

# -------------------------------------------------
# 1. Source System ROS
# -------------------------------------------------
if [ -f "/opt/ros/${ROS_DISTRO}/setup.bash" ]; then
    source /opt/ros/${ROS_DISTRO}/setup.bash
fi

BASHRC_FILE="${WORKSPACE}/.bashrc"

# ROS
if ! grep -Fxq "source /opt/ros/${ROS_DISTRO}/setup.bash" $BASHRC_FILE; then
    echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> $BASHRC_FILE
fi

# -------------------------------------------------
# 2. Build HDL Workspace (Core SLAM)
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

    if [ -f "${HDL_WS}/devel/setup.bash" ]; then
        source ${HDL_WS}/devel/setup.bash
    fi
else
    echo "!!! WARNING: hdl_ws/src not found. Skipping."
fi

# -------------------------------------------------
# 3. Build LiDAR Workspace
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
# Append sourcing to .bashrc for future shells
# -------------------------------------------------
BASHRC_FILE="${WORKSPACE}/.bashrc"

# ROS
if ! grep -Fxq "source /opt/ros/${ROS_DISTRO}/setup.bash" $BASHRC_FILE; then
    echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> $BASHRC_FILE
fi

# HDL workspace
if [ -f "${HDL_WS}/devel/setup.bash" ]; then
    if ! grep -Fxq "source ${HDL_WS}/devel/setup.bash" $BASHRC_FILE; then
        echo "source ${HDL_WS}/devel/setup.bash" >> $BASHRC_FILE
    fi
fi

# LiDAR workspace
if [ -f "${LIDAR_WS}/devel_isolated/setup.bash" ]; then
    if ! grep -Fxq "source ${LIDAR_WS}/devel_isolated/setup.bash" $BASHRC_FILE; then
        echo "source ${LIDAR_WS}/devel_isolated/setup.bash" >> $BASHRC_FILE
    fi
fi

# -------------------------------------------------
# Execute passed command or fallback to persistent bash
# -------------------------------------------------
echo "=== Environment ready ==="

if [ $# -gt 0 ]; then
    exec "$@"
else
    exec bash
fi