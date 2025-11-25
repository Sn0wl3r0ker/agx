#!/bin/bash
set -e

ROS_DISTRO=noetic
WORKSPACE=/root

# -------------------------------------------------
# Source ROS
# -------------------------------------------------
if [ -f "/opt/ros/${ROS_DISTRO}/setup.bash" ]; then
    source /opt/ros/${ROS_DISTRO}/setup.bash
fi

echo "========================================="
echo "   ROS ${ROS_DISTRO} Development Container"
echo "========================================="

# -------------------------------------------------
# Build HDL Workspace
# -------------------------------------------------
HDL_WS=${WORKSPACE}/hdl_ws

if [ -d "${HDL_WS}/src" ]; then
    echo "=== Checking hdl_ws ==="

    if [ ! -f "${HDL_WS}/devel/setup.bash" ]; then
        echo ">>> hdl_ws not built. Building now..."
        cd ${HDL_WS}
        catkin_make -DCMAKE_BUILD_TYPE=Release -DBUILD_VGICP_CUDA=ON
    else
        echo ">>> hdl_ws already built. Skipping build."
    fi

    source ${HDL_WS}/devel/setup.bash
else
    echo "!!! WARNING: hdl_ws/src not found. Skipping."
fi

# -------------------------------------------------
# Build LiDAR Workspace (isolated)
# -------------------------------------------------
LIDAR_WS=${WORKSPACE}/lidar_ws

if [ -d "${LIDAR_WS}/src" ]; then
    echo "=== Checking lidar_ws ==="

    if [ ! -f "${LIDAR_WS}/devel_isolated/setup.bash" ]; then
        echo ">>> lidar_ws not built. Building now..."
        cd ${LIDAR_WS}
        catkin_make_isolated -j$(nproc)
    else
        echo ">>> lidar_ws already built. Skipping build."
    fi

    source ${LIDAR_WS}/devel_isolated/setup.bash
else
    echo "!!! WARNING: lidar_ws/src not found. Skipping."
fi

# -------------------------------------------------
# Drop to shell
# -------------------------------------------------
echo "=== Environment ready. Launching bash ==="
exec bash
