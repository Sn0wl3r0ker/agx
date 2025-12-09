#!/bin/bash
set -e

# ==========================================
# 設定變數
# ==========================================
ROS_DISTRO="humble"  # 根據你的 Dockerfile 設定，或是用 ${ROS_DISTRO}
WORKSPACE="/root/ros2_ws"
BASHRC="/root/.bashrc"

# ==========================================
# 1. 配置 .bashrc (針對 docker exec 進入時的環境)
# ==========================================
# 定義一個函數來安全地寫入 .bashrc (避免重複)
add_to_bashrc() {
    if ! grep -qF "$1" "$BASHRC"; then
        echo "$1" >> "$BASHRC"
    fi
}

# 確保 .bashrc 存在
touch $BASHRC

# 加入 ROS 2 底層環境
add_to_bashrc "source /opt/ros/${ROS_DISTRO}/setup.bash"

# 加入 Workspace 環境 (如果有編譯過的話)
add_to_bashrc "source ${WORKSPACE}/install/setup.bash"

# [加分項目] 加入常用縮寫，開發更順手
add_to_bashrc "alias cb='colcon build --symlink-install'"
add_to_bashrc "alias s='source install/setup.bash'"
add_to_bashrc "export ROS_DOMAIN_ID=0"

# ==========================================
# 2. 當前 Shell 的環境載入 (針對 entrypoint 本身)
# ==========================================
source /opt/ros/${ROS_DISTRO}/setup.bash

# ==========================================
# 3. 自動編譯檢查
# ==========================================
# 邏輯：如果 src 存在，但 install 不存在，代表是第一次啟動，執行編譯
if [ -d "${WORKSPACE}/src" ] && [ ! -d "${WORKSPACE}/install" ]; then
    echo "----------------------------------------"
    echo "偵測到尚未編譯，正在執行初次 colcon build..."
    echo "----------------------------------------"
    cd ${WORKSPACE}
    colcon build --symlink-install
else
    echo "----------------------------------------"
    echo "工作空間已存在或無原始碼，跳過自動編譯。"
    echo "若需重新編譯，進入容器後輸入 'cb' 即可。"
    echo "----------------------------------------"
fi

# 嘗試載入 workspace (如果剛剛編譯成功，或者原本就有)
if [ -f "${WORKSPACE}/install/setup.bash" ]; then
    source ${WORKSPACE}/install/setup.bash
fi

if [ $# -gt 0 ]; then
    # 執行傳入的 command，例如 docker-compose run navigation bash
    exec "$@"
else
    # 沒有傳入 command → 保持容器活著並使用交互 shell
    exec bash
fi