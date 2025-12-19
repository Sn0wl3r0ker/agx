# ./vlm/Dockerfile

# 1. 基底映像檔
FROM nvcr.io/nvidia/isaac/ros:aarch64-ros2_humble_4c0c55dddd2bbcc3e8d5f9753bee634c

ENV DEBIAN_FRONTEND=noninteractive

# 2. 安裝基礎工具與 RealSense 驅動
RUN apt-get update && apt-get install -y \
    python3-pip \
    usbutils \
    ros-humble-realsense2-camera \
    ros-humble-diagnostic-updater \
    ros-humble-xacro \
    git-lfs \
    vim \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install -U jetson-stats

# 4. 安裝 Isaac ROS 相關套件
RUN apt-get update && apt-get install -y \
    ros-humble-isaac-ros-common \
    ros-humble-isaac-ros-apriltag \
    ros-humble-isaac-ros-image-pipeline \
    ros-humble-isaac-ros-nitros \
    ros-humble-isaac-ros-jetson-stats \
    ros-humble-rmw-cyclonedds-cpp \
    ros-humble-isaac-ros-peoplesemseg-models-install \
    && rm -rf /var/lib/apt/lists/*

# 5. 設定工作目錄
WORKDIR /workspaces/isaac_ros-dev
ENV RMW_IMPLEMENTATION=rmw_cyclonedds_cpp

# 6. 把 source setup.bash 寫入
RUN echo "source /workspaces/isaac_ros-dev/install/setup.bash" >> /root/.bashrc