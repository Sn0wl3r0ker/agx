# ./vlm/Dockerfile

# 1. 基底映像檔
FROM nvcr.io/nvidia/isaac/ros:aarch64-ros2_humble_4c0c55dddd2bbcc3e8d5f9753bee634c

ENV DEBIAN_FRONTEND=noninteractive

# 2. 安裝 RealSense 驅動 (這是 buildx 最重要的任務)
RUN apt-get update && apt-get install -y \
    ros-humble-realsense2-camera \
    ros-humble-diagnostic-updater \
    ros-humble-xacro \
    git-lfs \
    vim \
    && rm -rf /var/lib/apt/lists/*

# 3. (進階技巧) 預先安裝 Isaac ROS 常見的 apt 依賴
# 這裡我們手動列出 Isaac ROS 常用的依賴，省去 rosdep 的時間
RUN apt-get update && apt-get install -y \
    ros-humble-isaac-ros-common \
    ros-humble-isaac-ros-apriltag \
    ros-humble-isaac-ros-image-pipeline \
    ros-humble-isaac-ros-nitros \
    # 這裡可以加入更多你已知需要的包
    && rm -rf /var/lib/apt/lists/*

# 4. 設定工作目錄
WORKDIR /workspaces/isaac_ros-dev

# 5. 把 source setup.bash 寫入，確保一進去就有環境
RUN echo "source /workspaces/isaac_ros-dev/install/setup.bash" >> /root/.bashrc