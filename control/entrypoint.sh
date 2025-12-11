#!/bin/bash
# 如果任何命令出錯，立即退出 (保護機制)
set -eo pipefail

ROS_DISTRO=${ROS_DISTRO:-noetic}
WORKSPACE=/root
BASHRC_FILE="${WORKSPACE}/.bashrc"

echo "========================================="
echo "   ROS ${ROS_DISTRO} Hybrid Container"
echo "========================================="

# -------------------------------------------------
# Helper 函數：檢查並決定是否編譯
# -------------------------------------------------
build_workspace() {
    local ws_path=$1
    local build_type=$2 # "catkin_make" or "catkin_make_isolated"

    if [ -d "${ws_path}/src" ]; then
        echo "=== Checking $(basename $ws_path) ==="
        
        # 定義 setup 檔案路徑
        local setup_file="${ws_path}/devel/setup.bash"
        if [ "$build_type" == "catkin_make_isolated" ]; then
            setup_file="${ws_path}/devel_isolated/setup.bash"
        fi

        # [智慧判斷] 只有在找不到編譯檔時，才執行編譯
        if [ ! -f "$setup_file" ]; then
            echo ">>> $(basename $ws_path) not built (Dev Mode detected). Building now..."
            
            # [WSL 修復] 刪除壞掉的 CMakeLists.txt 連結
            if [ "$build_type" == "catkin_make" ] && [ -f "${ws_path}/src/CMakeLists.txt" ]; then
                echo "   -> Removing potential broken CMakeLists.txt symlink..."
                rm "${ws_path}/src/CMakeLists.txt"
            fi

            cd ${ws_path}
            
            # 執行編譯
            $build_type -j$(nproc)
        else
            echo ">>> $(basename $ws_path) already built (Production Mode). Skipping compilation."
        fi

        # Source 環境
        if [ -f "$setup_file" ]; then
            source $setup_file
            # 將 source 加入 .bashrc 以便進入 bash 時也能生效
            if ! grep -Fxq "source $setup_file" $BASHRC_FILE; then
                echo "source $setup_file" >> $BASHRC_FILE
            fi
        fi
    else
        echo "!!! WARNING: $(basename $ws_path)/src not found. Skipping."
    fi
}

# -------------------------------------------------
# 1. Source System ROS & Setup .bashrc
# -------------------------------------------------
source /opt/ros/${ROS_DISTRO}/setup.bash

if ! grep -Fxq "source /opt/ros/${ROS_DISTRO}/setup.bash" $BASHRC_FILE; then
    echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> $BASHRC_FILE
fi

# -------------------------------------------------
# 2. 依序處理所有工作空間
# -------------------------------------------------

# (A) Keyboard Control Workspace
build_workspace "${WORKSPACE}/keyboard_control_ws" "catkin_make"

# (B) LiDAR Workspace
build_workspace "${WORKSPACE}/lidar_ws" "catkin_make_isolated"

# (C) Realsense Workspace
build_workspace "${WORKSPACE}/realsense_ws" "catkin_make"

# (D) HDL Workspace
build_workspace "${WORKSPACE}/hdl_ws" "catkin_make"

# -------------------------------------------------
# 3. 執行指令
# -------------------------------------------------
source $BASHRC_FILE

echo "=== Environment ready ==="

if [ $# -gt 0 ]; then
    exec "$@"
else
    exec bash
fi